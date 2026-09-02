# gate.py -- W8A8/W8A16 fake-quant gate experiment for HoloBrain_v0.0_GD on V100.
# Protocol adapted from the SwiftVLA line (bench_latency.py):
#   1. determinism residual: two fp32 forwards, same seed -> must be ~0
#   2. noise floor: fp32 vs fp32, different denoise-init noise seeds (3 pairs),
#      actions compared in DENORMALIZED action space (pred_actions[..., 0] is the
#      executed joint action; we report the full 8-dim state vector + groups)
#   3. quant error: fp32 vs quant mode, SAME seed (fixed noise), same metric
#      modes: w8a8 / w8a16 / w4a16 / w8a8@<scope> (see hb_quant.MODULE_SCOPES)
#   4. deformable sensitivity: sampling_offsets Linear outputs (fusion
#      deformable attention) captured by hooks under fp32 vs w8a8, same seed;
#      drift reported in feature-map cells, normalized grid units, input px
#   5. SmoothQuant recon: per-Linear per-channel input absmax on an fp32
#      forward; outlier ratio = max_channel / median_channel; top-20
#
# Run: cd /home/nc23/workspace/holobrain && CUDA_VISIBLE_DEVICES=2 \
#        /home/nc23/.conda/envs/holobrain/bin/python quant/gate.py

import json
import os
import sys
import time

HB = "/home/nc23/workspace/holobrain"
sys.path.insert(0, HB)  # for bringup.py
sys.path.insert(0, os.path.join(HB, "quant"))

import numpy as np
import torch

import bringup  # reuses the bringup bring-up path verbatim (scene, processor)

from hb_quant import MODULE_SCOPES, apply_fake_quant, _in_scope
from robo_orchard_lab.models.holobrain.processor import (
    HoloBrainProcessor,
    MultiArmManipulationInput,
)
from robo_orchard_lab.models.mixin import ModelMixin

OUT_JSON = os.path.join(HB, "gate_results.json")
SEEDS_QUANT = [1000, 1100, 1200]
SEED_PAIRS = [(1000, 2000), (1100, 2100), (1200, 2200)]
MODES = [
    "w8a16",
    "w8a8",
    "w4a16",
    "w8a8@vision_2d",
    "w8a8@vision_3d",
    "w8a8@text_bert",
    "w8a8@fusion",
    "w8a8@action_head",
    "w8a8@neck_convs",
]


def _mha_eager(self, query, key, value, key_padding_mask=None,
               need_weights=True, attn_mask=None, **kw):
    """Eager re-implementation of torch nn.MultiheadAttention (self-attention
    path, eval, no dropout). torch's eval fast path routes out_proj through a
    fused kernel and never calls the module, so monkey-patched Linear forwards
    and hooks silently do not fire; this version calls self.out_proj as a
    module. in_proj is a raw Parameter (not nn.Linear) and stays fp -- same
    convention as the rest of the protocol (embeddings/norms unquantized)."""
    E, H = self.embed_dim, self.num_heads
    w, b = self.in_proj_weight, self.in_proj_bias
    q = torch.nn.functional.linear(query, w[:E], b[:E] if b is not None else None)
    k = torch.nn.functional.linear(key, w[E:2 * E], b[E:2 * E] if b is not None else None)
    v = torch.nn.functional.linear(value, w[2 * E:], b[2 * E:] if b is not None else None)
    N, B, _ = q.shape
    hd = E // H
    q = (q.reshape(N, B * H, hd).transpose(0, 1)) * (hd ** -0.5)
    k = k.reshape(N, B * H, hd).transpose(0, 1)
    v = v.reshape(N, B * H, hd).transpose(0, 1)
    attn = torch.bmm(q, k.transpose(1, 2)).view(B, H, N, N)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn = attn.masked_fill(attn_mask.unsqueeze(0), float("-inf"))
        else:
            attn = attn + attn_mask.unsqueeze(0)
    if key_padding_mask is not None:
        attn = attn.masked_fill(
            key_padding_mask[:, None, None, :], float("-inf")
        )
    attn = attn.softmax(dim=-1).view(B * H, N, N)
    out = torch.bmm(attn, v).transpose(0, 1).reshape(N, B, E)
    out = self.out_proj(out)  # module call -> fake-quant / hooks fire
    return out, None


def force_mha_slow_path(model):
    import types

    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.MultiheadAttention):
            mod.forward = types.MethodType(_mha_eager, mod)
            n += 1
    return n


def load_everything():
    processor = HoloBrainProcessor.load(bringup.CKPT, bringup.PROCESSOR_JSON)
    model = ModelMixin.load_model(bringup.MODEL_DIR, load_impl="native")
    model = model.cuda().float().eval()
    images, depths, _ = bringup.build_raw_inputs()
    inp = MultiArmManipulationInput(
        image=images,
        depth=depths,
        intrinsic={c: bringup.K44.copy() for c in bringup.CAM_NAMES},
        t_world2cam={c: bringup.CAM_POSES[c].copy() for c in bringup.CAM_NAMES},
        t_robot2world=bringup.T_BASE2WORLD.copy(),
        t_robot2ego=None,
        history_joint_state=[bringup.JOINT_STATE.copy()],
        history_ee_pose=None,
        instruction=bringup.INSTRUCTION,
        urdf=None,
        remaining_actions=None,
        delay_horizon=None,
    )
    batch = processor.pre_process(inp)
    return processor, model, batch


class NoiseRecorder:
    """Capture the denoise-init noise sampled inside the decoder (provenance
    for the fixed-noise A/B protocol)."""

    def __init__(self, model):
        self.decoder = model.decoder
        self.orig = self.decoder.sample_noise
        self.noises = []

    def __enter__(self):
        dec = self.decoder

        def wrapped(*args, **kwargs):
            n = self.orig(*args, **kwargs)
            self.noises.append(n.detach().cpu().clone())
            return n

        dec.sample_noise = wrapped
        return self

    def __exit__(self, *a):
        self.decoder.__dict__.pop("sample_noise", None)


def forward_actions(model, batch, seed):
    """One full model forward (10 DPM-Solver denoise steps) with fixed noise
    seed. Returns pred_actions [1, 64, 14, 8] (denormalized) on CPU."""
    torch.manual_seed(seed)
    with torch.no_grad():
        outs = model(batch)
    pa = outs[0]["pred_actions"]
    return pa.detach().cpu()


def act_metrics(a, b):
    d = (a - b).abs()
    return {
        "mae_all": float(d.mean()),
        "max_all": float(d.max()),
        "mae_jointpos": float(d[..., 0].mean()),
        "max_jointpos": float(d[..., 0].max()),
        "mae_rot6d": float(d[..., 1:7].mean()),
        "mae_gripper": float(d[..., 7].mean()),
        "max_gripper": float(d[..., 7].max()),
    }


METRIC_KEYS = (
    "mae_all", "max_all", "mae_jointpos", "max_jointpos",
    "mae_rot6d", "mae_gripper", "max_gripper",
)


def mean_of_dicts(ds):
    return {k: float(np.mean([d[k] for d in ds])) for k in METRIC_KEYS}


# ---------------------------------------------------------------- deform hooks
def register_deform_hooks(model, store):
    """Hooks on every MultiScaleDeformableAttention in the fusion enhancer:
    parent forward captures spatial_shapes/level_start_index kwargs; the
    sampling_offsets Linear hook captures raw offset output."""
    handles = []
    parents = []
    for name, mod in model.named_modules():
        if (
            isinstance(getattr(mod, "sampling_offsets", None), torch.nn.Linear)
            and "img_attn_blocks" in name
        ):
            parents.append((name, mod))
    for name, mod in parents:
        def parent_hook(m, args, kwargs, output, key=name):
            ss = kwargs.get("spatial_shapes")
            lsi = kwargs.get("level_start_index")
            if ss is not None:
                store.setdefault(key, {})["spatial_shapes"] = ss.detach().cpu()
                if lsi is not None:
                    store[key]["level_start_index"] = lsi.detach().cpu()

        def offset_hook(m, args, output, key=name):
            store.setdefault(key, {}).setdefault("offsets", []).append(
                output.detach().cpu()
            )

        h1 = mod.register_forward_hook(parent_hook, with_kwargs=True)
        h2 = mod.sampling_offsets.register_forward_hook(offset_hook)
        handles.extend([h1, h2])
    return handles, [n for n, _ in parents]


def analyze_offsets(fp_store, q_store, H_in, W_in):
    """Drift of sampling_offsets between two captured runs, per layer/level.
    Raw offset units are cells of THAT level's feature map (the model computes
    sampling_loc = ref_point + offset / (w, h)_level)."""
    out = {}
    for name in fp_store:
        a, b = fp_store[name], q_store.get(name)
        if b is None or "offsets" not in a or "offsets" not in b:
            continue
        ss = a["spatial_shapes"]  # [L, 2] = (h, w)
        hw = (ss[:, 0] * ss[:, 1]).long()
        starts = torch.cat([torch.zeros(1, dtype=torch.long), hw.cumsum(0)[:-1]])
        levels = []
        for lvl in range(ss.shape[0]):
            oa = torch.cat(a["offsets"], 0)[:, starts[lvl] : starts[lvl] + hw[lvl]]
            ob = torch.cat(b["offsets"], 0)[:, starts[lvl] : starts[lvl] + hw[lvl]]
            if oa.shape[0] != ob.shape[0]:  # should not happen (same seed)
                n = min(oa.shape[0], ob.shape[0])
                oa, ob = oa[:n], ob[:n]
            d = (oa - ob).abs()  # [..., 2] last dim is (x, y)
            h_l, w_l = int(ss[lvl][0]), int(ss[lvl][1])
            # cells at this level == raw units; px at input scale per axis
            px = torch.stack(
                [d[..., 0] * (W_in / w_l), d[..., 1] * (H_in / h_l)], -1
            )
            nrm = torch.stack([d[..., 0] / w_l, d[..., 1] / h_l], -1)
            levels.append(
                {
                    "level": lvl,
                    "feature_hw": [h_l, w_l],
                    "stride": [H_in / h_l, W_in / w_l],
                    "mean_cells": float(d.mean()),
                    "max_cells": float(d.max()),
                    "mean_norm_units": float(nrm.mean()),
                    "max_norm_units": float(nrm.max()),
                    "mean_input_px": float(px.mean()),
                    "max_input_px": float(px.max()),
                }
            )
        out[name] = {"levels": levels}
    return out


# ---------------------------------------------------------------- smoothquant
def smoothquant_scan(model, batch, seed):
    """fp32 forward with pre-hooks on every nn.Linear: running per-channel
    absmax of the input activations (aggregated over all calls incl. the 10
    denoise steps). outlier ratio = max_c / median_c of that vector."""
    acc = {}

    def make_pre(name):
        def pre(mod, args):
            x = args[0]
            if not torch.is_floating_point(x) or x.dim() < 2:
                return
            a = x.detach().abs().reshape(-1, x.shape[-1])
            ch = a.max(0).values
            if name in acc:
                acc[name]["ch"] = torch.maximum(acc[name]["ch"], ch)
                acc[name]["calls"] += 1
                acc[name]["tokens"] += a.shape[0]
            else:
                acc[name] = {
                    "ch": ch,
                    "calls": 1,
                    "tokens": a.shape[0],
                    "in_f": mod.in_features,
                    "out_f": mod.out_features,
                }
        return pre

    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            handles.append(mod.register_forward_pre_hook(make_pre(name)))
    forward_actions(model, batch, seed)
    for h in handles:
        h.remove()

    rows = []
    for name, d in acc.items():
        ch = d["ch"].float()
        med = ch.median().clamp(min=1e-8)
        scope = next(
            (s for s, ps in MODULE_SCOPES.items() if _in_scope(name, ps)), "other"
        )
        rows.append(
            {
                "name": name,
                "scope": scope,
                "in_f": d["in_f"],
                "out_f": d["out_f"],
                "ratio": float(ch.max() / med),
                "ch_max": float(ch.max()),
                "ch_median": float(ch.median()),
                "calls": d["calls"],
                "tokens": d["tokens"],
            }
        )
    rows.sort(key=lambda r: -r["ratio"])
    return rows


# ---------------------------------------------------------------- main
def main():
    torch.backends.cudnn.deterministic = True
    t_start = time.perf_counter()
    processor, model, batch = load_everything()
    n_mha = force_mha_slow_path(model)
    print(f"[mha ] forced eager path on {n_mha} nn.MultiheadAttention "
          f"(text self-attn): out_proj now really quantized", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[load] model {n_params/1e6:.1f}M params on "
          f"{next(model.parameters()).device}", flush=True)

    H_in, W_in = int(batch["imgs"].shape[2]), int(batch["imgs"].shape[3])
    meta = {
        "ckpt": "HoloBrain_v0.0_GD@post_training_robotwin",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dtype": "float32",
        "input_hw": [H_in, W_in],
        "n_params": n_params,
        "seeds_quant": SEEDS_QUANT,
        "seed_pairs_noise_floor": SEED_PAIRS,
        "protocol": (
            "fixed-noise A/B: torch.manual_seed(seed) before each forward "
            "fixes the denoise-init noise; quant error = fp32 vs quant at the "
            "same seed; noise floor = fp32(seedA) vs fp32(seedB)"
        ),
        "mha_slow_path_forced": n_mha,
    }

    # 1. determinism residual (same seed twice)
    with NoiseRecorder(model) as nr:
        a1 = forward_actions(model, batch, 1000)
    with NoiseRecorder(model) as nr2:
        a2 = forward_actions(model, batch, 1000)
    det = float((a1 - a2).abs().max())
    print(f"[det ] same-seed fp32 max_abs diff = {det:.3e}", flush=True)

    # 2. fp32 references at all seeds + noise floor
    fp = {}
    for s in sorted(set(SEEDS_QUANT + [x for p in SEED_PAIRS for x in p])):
        fp[s] = forward_actions(model, batch, s)
    pairs = []
    for sa, sb in SEED_PAIRS:
        m = act_metrics(fp[sa], fp[sb])
        pairs.append({"seed_a": sa, "seed_b": sb, **m})
        print(f"[flor] {sa} vs {sb}: mae_all={m['mae_all']:.5f} "
              f"max={m['max_all']:.5f} jointpos_mae={m['mae_jointpos']:.5f}",
              flush=True)
    floor_mean = mean_of_dicts(pairs)
    print(f"[flor] mean over {len(pairs)} pairs: "
          f"mae_all={floor_mean['mae_all']:.5f} "
          f"max_all={floor_mean['max_all']:.5f}", flush=True)

    action_scale = {
        "jointpos_std": float(fp[1000][..., 0].std()),
        "jointpos_range": [
            float(fp[1000][..., 0].min()), float(fp[1000][..., 0].max())
        ],
        "all_std": float(fp[1000].std()),
    }

    # 3. quant modes
    modes_out = {}
    for mode in MODES:
        qstats, restore = apply_fake_quant(model, mode)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        per_seed = []
        for s in SEEDS_QUANT:
            q = forward_actions(model, batch, s)
            per_seed.append({"seed": s, **act_metrics(fp[s], q)})
        torch.cuda.synchronize()
        ms_fwd = (time.perf_counter() - t0) * 1000 / len(SEEDS_QUANT)
        restore()
        mean = mean_of_dicts(
            [{k: v for k, v in d.items() if k != "seed"} for d in per_seed]
        )
        modes_out[mode] = {
            "quant_stats": qstats,
            "ms_per_forward_incl_quant_overhead": ms_fwd,
            "per_seed": per_seed,
            "mean": mean,
            "mae_all_over_floor": mean["mae_all"] / floor_mean["mae_all"],
        }
        print(
            f"[q   ] {mode:16s} lin {qstats['linear_quantized']:3d}/"
            f"{qstats['linear_total']} conv {qstats['conv_quantized']:2d}/"
            f"{qstats['conv_total']} | mae_all={mean['mae_all']:.5f} "
            f"(x{mean['mae_all']/floor_mean['mae_all']:.1f} floor) "
            f"max={mean['max_all']:.4f} jpos_mae={mean['mae_jointpos']:.5f} "
            f"| {ms_fwd:.0f} ms/fwd",
            flush=True,
        )

    # 4. deformable offsets drift (fp32 vs w8a8, seed 1000)
    fp_store, q_store = {}, {}
    handles, layer_names = register_deform_hooks(model, fp_store)
    forward_actions(model, batch, 1000)
    for h in handles:
        h.remove()
    qstats_w8a8, restore = apply_fake_quant(model, "w8a8")
    handles, _ = register_deform_hooks(model, q_store)
    forward_actions(model, batch, 1000)
    for h in handles:
        h.remove()
    restore()
    deform = analyze_offsets(fp_store, q_store, H_in, W_in)
    for name in sorted(deform):
        lv = deform[name]["levels"]
        print(f"[def ] {name}: " + " | ".join(
            f"L{l['level']}({l['feature_hw'][0]}x{l['feature_hw'][1]}) "
            f"mean {l['mean_cells']:.4f}c/{l['mean_input_px']:.3f}px "
            f"max {l['max_cells']:.3f}c/{l['max_input_px']:.2f}px"
            for l in lv
        ), flush=True)

    # 5. SmoothQuant recon
    sq_rows = smoothquant_scan(model, batch, 1000)
    n_lin_total = sum(
        1 for m in model.modules() if isinstance(m, torch.nn.Linear)
    )
    if len(sq_rows) < n_lin_total:
        seen = {r["name"] for r in sq_rows}
        uncalled = [
            n for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and n not in seen
        ]
        print(f"[sq  ] WARNING {len(uncalled)} Linears never called: "
              f"{uncalled}", flush=True)
    print(f"[sq  ] scanned {len(sq_rows)}/{n_lin_total} Linears; "
          f"top-20 outlier ratios:",
          flush=True)
    for r in sq_rows[:20]:
        print(f"      {r['ratio']:8.1f}  {r['name'][:70]:70s} "
              f"{r['scope']:11s} max={r['ch_max']:8.3f} "
              f"med={r['ch_median']:7.4f} x{r['calls']}", flush=True)

    results = {
        "meta": meta,
        "determinism_same_seed_max_abs": det,
        "action_scale": action_scale,
        "noise_floor": {"pairs": pairs, "mean": floor_mean},
        "modes": modes_out,
        "deformable_offsets": deform,
        "smoothquant": {"top20": sq_rows[:20], "all_modules": sq_rows},
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=1)
    print(f"[save] {OUT_JSON} ({os.path.getsize(OUT_JSON)/1e3:.0f} kB)", flush=True)
    print(f"[done] total {time.perf_counter()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
