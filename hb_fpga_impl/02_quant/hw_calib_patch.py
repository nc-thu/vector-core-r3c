# hw_calib_patch.py -- patch calibration for the modules that never made it
# into hw_calib_table.json (26 placeholder requant warnings in the full
# compile, build_full/host_plan.json).
#
# Why they are missing (verified against 02_quant/hw_calib.py):
#   * decoder.head.convs.0 / .1  -- nn.Conv1d. calibrate() only hooks
#     nn.Linear and nn.Conv2d, so no sa/so stats were ever collected.
#   * feature_enhancer.text_attn_blocks.{0..5}.self_attn.attn.in_proj_weight
#     -- fused qkv weight inside nn.MultiheadAttention. There is no child
#     nn.Linear for in_proj (out_proj is one, which is why out_proj IS in
#     the table), so hooks never saw it either.
# The 26 compiler warnings = 10 rollout steps x 2 convs + 6 one-shot in_proj,
# i.e. 8 unique calibration keys.
#
# What this script does (exactly the same math as build_hw_params):
#   fp32 model on the same 8 perturbed bringup samples (perturbed_batch,
#   seeds 4000+i, identical to last round) -> sa/so per module -> sw from the
#   fused/conv weight -> r=(sa*sw)/so -> (m,s)=v1_encode -> K+1 bias aug
#   (c in 1..64) or fp fallback. Output entries are field-compatible with
#   hw_calib_table.json gemms entries, so the compiler can consume a merged
#   table (--calib hw_calib_table_v2.json).
#
# Run (server) -- outputs to /tmp because /home is currently full:
#   mkdir -p /tmp/hwcalib_patch
#   cp hw_calib_patch.py /tmp/hwcalib_patch/
#   cd /tmp/hwcalib_patch
#   CUDA_VISIBLE_DEVICES=1 TMPDIR=/tmp \
#       ~/.conda/envs/holobrain/bin/python hw_calib_patch.py

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
HW_CAL = "~/workspace/holobrain/hw_calib"   # read-only source
sys.path.insert(0, HW_CAL)

import hw_calib as HC              # reuses bringup/gate path setup + encoders
from gate import force_mha_slow_path, forward_actions, load_everything

N_CAL = 8                          # identical to the full-mode run

CONV_KEYS = ["decoder.head.convs.0", "decoder.head.convs.1"]
# compiler wkey = <mha module>.in_proj_weight (no '.weight' suffix to strip)
INPROJ_KEYS = [f"feature_enhancer.text_attn_blocks.{i}.self_attn.attn"
               ".in_proj_weight" for i in range(6)]
MHA_MODULES = [k[:-len(".in_proj_weight")] for k in INPROJ_KEYS]


def _mk_entry(name, w, bias, in_max, out_max, k_eff, n_eff, typ, calls):
    sa = max(float(in_max), 1e-12) / 127.0
    so = max(float(out_max), 1e-12) / 127.0
    sw = max(float(w.detach().float().abs().amax()), 1e-12) / 127.0

    c_sel, wb, bias_fp, b_acc_absmax = None, None, False, None
    if bias is None:
        pass
    else:
        b_acc = bias.detach().double() / (sa * sw)
        b_acc_absmax = float(b_acc.abs().amax())
        for c in HC.C_CANDIDATES:
            w_try = torch.round(b_acc / c)
            if float(w_try.abs().amax()) <= 127.0:
                c_sel, wb = c, w_try
                break
        if c_sel is None:
            bias_fp = True

    r_star = (sa * sw) / so
    m8, _ = HC.s8_encode(r_star)
    mv1, sv1 = HC.v1_encode(r_star)
    acc_absmax_est = 127.0 / r_star

    ent = {
        "type": typ,
        "gemm_k": k_eff,
        "gemm_n": n_eff,
        "gemm_m": "dynamic(tokens)",
        "sa": sa, "sw": sw, "so": so,
        "r_star": r_star,
        "acc_absmax_est": acc_absmax_est,
        "m_s8_q8_8": m8,
        "m_s8_dead": bool(m8 == 0),
        "m_requant": mv1,
        "s_shift": sv1,
        "bias_aug_c": c_sel,
        "w_bias_int8": ([int(round(v)) for v in wb.cpu().tolist()]
                        if wb is not None else None),
        "bias_fp_fallback": bias_fp,
        "b_acc_absmax": b_acc_absmax,
        "patch_calls": calls,
    }
    flags = []
    if acc_absmax_est > HC.ACC_XW27_LIMIT:
        flags.append("acc_over_xw27")
    if mv1 == 1:
        flags.append("m_v1_underflow")
    if flags:
        ent["patch_flags"] = flags
    return ent


def main():
    t0 = time.perf_counter()
    processor, model, batch = load_everything()
    n_mha = force_mha_slow_path(model)
    print(f"[load] model up; forced eager path on {n_mha} MHA modules",
          flush=True)
    mods = dict(model.named_modules())

    stats = {}
    handles = []

    # ---- Conv1d: plain pre/post hooks, same as calibrate() ----
    for name in CONV_KEYS:
        mod = mods[name]
        assert isinstance(mod, nn.Conv1d), (name, type(mod))
        stats[name] = {"in_max": 0.0, "out_max": 0.0, "calls": 0,
                       "nonfloat": 0}

        def pre(m, args, key=name):
            x = args[0]
            e = stats[key]
            if torch.is_tensor(x) and torch.is_floating_point(x):
                e["in_max"] = max(e["in_max"],
                                  float(x.detach().abs().amax()))
                e["calls"] += 1
            else:
                e["nonfloat"] += 1

        def post(m, args, out, key=name):
            if torch.is_tensor(out) and torch.is_floating_point(out):
                e = stats[key]
                e["out_max"] = max(e["out_max"],
                                   float(out.detach().abs().amax()))

        handles.append(mod.register_forward_pre_hook(pre))
        handles.append(mod.register_forward_hook(post))

    # ---- MHA in_proj: hook the MHA module input, evaluate the fused qkv
    #      GEMM manually (x @ in_proj_weight.T + in_proj_bias) for so ----
    for mha_path, key in zip(MHA_MODULES, INPROJ_KEYS):
        mod = mods[mha_path]
        assert isinstance(mod, nn.MultiheadAttention), (mha_path, type(mod))
        assert mod.in_proj_weight is not None, mha_path
        W = mod.in_proj_weight.detach().float()          # [3C, C]
        b = (mod.in_proj_bias.detach().float()
             if mod.in_proj_bias is not None else None)  # [3C]
        C = W.shape[1]
        stats[key] = {"in_max": 0.0, "out_max": 0.0, "calls": 0,
                      "nonfloat": 0}

        def pre(m, args, kwargs, key=key, W=W, b=b, C=C):
            x = kwargs.get("query", args[0] if args else None)
            e = stats[key]
            if x is None or not torch.is_tensor(x) \
                    or not torch.is_floating_point(x):
                e["nonfloat"] += 1
                return
            with torch.no_grad():
                e["in_max"] = max(e["in_max"], float(x.detach().abs().amax()))
                x2 = x.detach().reshape(-1, C).float()
                qkv = x2 @ W.t()
                if b is not None:
                    qkv = qkv + b
                e["out_max"] = max(e["out_max"],
                                   float(qkv.detach().abs().amax()))
            e["calls"] += 1

        handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))

    # ---- the same 8 perturbed samples, same seeds as the full run ----
    for i in range(N_CAL):
        batch_i = HC.perturbed_batch(processor, i)
        forward_actions(model, batch_i, 4000 + i)
        print(f"[cal ] sample {i} done {time.perf_counter()-t0:.1f}s",
              flush=True)
    for h in handles:
        h.remove()

    # ---- build table entries (identical math to build_hw_params) ----
    table = {}
    for name in CONV_KEYS:
        mod = mods[name]
        s = stats[name]
        assert s["calls"] > 0, f"{name} never called"
        k_eff = (mod.in_channels // mod.groups) * int(
            np.prod(mod.kernel_size))
        table[name] = _mk_entry(name, mod.weight,
                                mod.bias if mod.bias is not None else None,
                                s["in_max"], s["out_max"],
                                k_eff, mod.out_channels, "nn.Conv1d",
                                s["calls"])
        print(f"[ent ] {name}: calls={s['calls']} k={k_eff} "
              f"n={mod.out_channels}", flush=True)

    for mha_path, key in zip(MHA_MODULES, INPROJ_KEYS):
        mod = mods[mha_path]
        s = stats[key]
        assert s["calls"] > 0, f"{key} never called"
        C = mod.in_proj_weight.shape[1]
        table[key] = _mk_entry(key, mod.in_proj_weight, mod.in_proj_bias,
                               s["in_max"], s["out_max"],
                               C, 3 * C, "nn.MultiheadAttention.in_proj",
                               s["calls"])
        print(f"[ent ] {key}: calls={s['calls']} k={C} n={3*C}", flush=True)

    out = {
        "_meta": {
            "model": "HoloBrain_v0.0_GD@post_training_robotwin",
            "mode": "patch",
            "calib_samples": N_CAL,
            "calib_seeds": [4000 + i for i in range(N_CAL)],
            "patch_reason": {
                "nn.Conv1d": "calibrate() only hooked nn.Linear/nn.Conv2d",
                "mha_in_proj": "fused weight, no child Linear to hook",
            },
            "compiler_warnings_covered": 26,
            "unique_keys": len(table),
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "gemms": table,
    }
    with open(os.path.join(HERE, "hw_calib_patch.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[out ] hw_calib_patch.json ({len(table)} entries) "
          f"{time.perf_counter()-t0:.0f}s", flush=True)

    # ---- merge + validate ----
    orig = json.load(open(os.path.join(HW_CAL, "hw_calib_table.json")))
    og = orig["gemms"]
    dup = sorted(set(og) & set(table))
    assert not dup, f"duplicate keys: {dup}"
    merged = dict(og)
    merged.update(table)
    with open(os.path.join(HERE, "hw_calib_table_v2.json"), "w") as f:
        json.dump({"_meta": dict(orig["_meta"],
                                 patched_with="hw_calib_patch.json",
                                 gemm_entries_v2=len(merged)),
                   "gemms": merged}, f, indent=1)

    req = ["sa", "sw", "so", "m_requant", "s_shift", "bias_aug_c",
           "w_bias_int8", "bias_fp_fallback"]
    n_bad = 0
    print(f"[v2  ] {len(og)} + {len(table)} = {len(merged)} entries, "
          f"no duplicate keys")
    for k, e in table.items():
        miss = [f for f in req if f not in e]
        m, s = e["m_requant"], e["s_shift"]
        r_true = e["sa"] * e["sw"] / e["so"]
        r_err = abs(m / (1 << s) - r_true)
        ok = (not miss and 1 <= m <= 32767 and 8 <= s <= 47
              and r_err <= 0.5 / (1 << s))
        n_bad += (not ok)
        branch = ("aug c=%d" % e["bias_aug_c"]
                  if not e["bias_fp_fallback"] and e["bias_aug_c"]
                  else "fp_fallback" if e["bias_fp_fallback"] else "no_bias")
        print(f"[v2  ] {'OK ' if ok else 'BAD'} {k}\n"
              f"        m={m} s={s} r={r_true:.4e} (|m/2^s-r|="
              f"{r_err:.2e}) acc_est={e['acc_absmax_est']:.0f} "
              f"(lim {HC.ACC_XW27_LIMIT:.0f}) bias={branch}")
    assert n_bad == 0, f"{n_bad} entries failed validation"
    print(f"[v2  ] hw_calib_table_v2.json written, all {len(table)} "
          f"patch entries validated", flush=True)


if __name__ == "__main__":
    main()
