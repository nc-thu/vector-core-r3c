# hw_calib.py -- hardware-semantics PTQ calibration for the hw_zcu104-style
# integer pipeline (hb_fpga_impl/02_quant).
#
# Last round's w8a8 gate used per-token DYNAMIC activation scales, per-channel
# weights, fp32 bias and fp requant. The accelerator is stricter; this file
# re-runs the gate with the real hardware numerics:
#
#   1. weights:      per-TENSOR symmetric int8, sw = absmax(W)/127
#   2. activations:  per-TENSOR STATIC symmetric int8, sa = calib absmax(a)/127
#                    (calibrated on >=8 perturbed bringup synthetic samples)
#   3. accumulation: exact integer arithmetic, simulated in float64
#                    (all values < 2^53, IEEE-exact); XW=27 acc bound checked
#   4. no bias port: K+1 augmentation
#                        acc += w_bias * c
#                        w_bias = round(b/(sa*sw*c)) in [-127,127]
#                    c = smallest power of two that fits (<= 64, per spec).
#                    If even c=64 overflows (|b|/(sa*sw) > 127*64) the layer
#                    keeps an fp bias (counted + listed; models a CPU-side
#                    bias add, which the pipeline already allows for norm/
#                    gelu-class ops).
#   5. requant: ONE static (m, s) pair per GEMM
#                        y  = sat8( (acc * m) >>> s )      r = m * 2^-s
#      DIRECTION (derived, and matches the RTL's own FA test vectors where
#      r = 0.25): acc*(sa*sw) = fp output, y_int8 = fp/so, so the multiplier
#      is r = (sa*sw)/so -- it SCALES DOWN (typ. 1.6e-4 .. 1.6e-2 here).
#      (The task brief wrote r* = so/(sa*sw); that is the inverse/dequant
#      factor -- using it slams every output onto the +-127 rail.)
#      RTL ground truth (hw_zcu104/rtl):
#        ae_requant.sv / rq_v1.sv : 32x16 LUT multiply + 48b BARREL shifter,
#                                    s full 8-bit [0,255].
#        rq_v2.sv / rq_ms.sv (instantiated in ae_gemm, currently T_MAX=0):
#                                    byte-split m, t = s-8 >= 0, T_MAX=39
#                                    covers s in [8,47] -- one parameter
#                                    change away from full coverage.
#      Measured on this model: r = (sa*sw)/so is 1.55e-4 .. 1.6e-2, needing
#      s in [8, 27] with m normalized into [16384, 32767] (15-bit precision).
#      So:
#        mode B_s8 (rq_v2 with T_MAX=0, s=8 fixed): m = round(r*256) would be
#          0.04..4.05 -> most layers dead (m=0) or 25%-coarse; infeasible.
#          Run 1 seed to document the failure mode.
#        mode B_v1 (rq_v2 T_MAX>=19, or the v1 barrel shifter): m/2^s covers
#          everything at 2^-15 precision. THIS is the hardware-capable
#          encoding; gate + constants table use it.
#      No rounding constant exists in any rq variant: the shift is a pure
#      arithmetic (floor) shift, which carries a systematic -0.5 LSB bias
#      per GEMM output.
#   Output tensor = (int8 y, so). CPU-side ops consume dequantized fp y*so;
#   the next GEMM re-quantizes with its own calibrated sa.
#
# Exempt (stays fp, last round's conclusion): spatial_enhancer.pts_prob_fc.layers.1
#
# nn.MultiheadAttention eval fast path bypasses out_proj (the hook/patch trap
# from last round): gate.force_mha_slow_path is applied to every run here,
# including the fp references.
#
# Run (server):
#   cd /home/nc23/workspace/holobrain
#   CUDA_VISIBLE_DEVICES=1 /home/nc23/.conda/envs/holobrain/bin/python \
#       hw_calib/hw_calib.py smoke|full

import argparse
import json
import math
import os
import sys
import time
import types

HB = "/home/nc23/workspace/holobrain"
sys.path.insert(0, HB)
sys.path.insert(0, os.path.join(HB, "quant"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import bringup  # sets shims/repo paths + HF_ENDPOINT

from hb_quant import MODULE_SCOPES, _in_scope, apply_fake_quant
from gate import (
    act_metrics,
    force_mha_slow_path,
    forward_actions,
    load_everything,
    mean_of_dicts,
)
from robo_orchard_lab.models.holobrain.processor import (
    HoloBrainProcessor,
    MultiArmManipulationInput,
)

OUT_DIR = os.path.join(HB, "hw_calib")
EXEMPT = {"spatial_enhancer.pts_prob_fc.layers.1"}
EVAL_SEEDS = [1000, 1100, 1200]
GATE_GREEN = 0.030
GATE_YELLOW = 0.060
M_LIM = 32767  # |m| <= 32767 (signed 16b, Q8.8 when s=8)
ACC_XW27_LIMIT = 2.0 ** 26  # |acc| <= K*128*128 <= 2^26 for K <= 4096 (RTL)

INSTRUCTIONS = [
    "put the bowl on the plate",
    "pick up the bottle and put it into the box",
    "place the cup on the left side of the table",
    "push the cube to the corner",
    "open the drawer",
    "put the fork next to the knife",
    "grab the marker and hand it over",
    "move the plate to the right",
]

# camera eye positions used by bringup.CAM_POSES (re-derived here so they can
# be perturbed per calibration sample)
BASE_EYES = {
    "front_camera": (0.00, 0.00, 0.55),
    "left_camera": (0.25, -0.45, 0.50),
    "right_camera": (0.25, 0.45, 0.50),
    "head_camera": (0.40, 0.10, 0.80),
}


# ---------------------------------------------------------------- requant enc
def v1_encode(r_star):
    """r = m / 2^s with m in [1, 32767]; pick the largest s (finest m) whose
    rounded m still fits. Returns (m, s)."""
    r = max(float(r_star), 1e-30)
    s = max(0, int(math.floor(math.log2(M_LIM / r))))
    m = int(round(r * (1 << s)))
    while m > M_LIM and s > 0:
        s -= 1
        m = int(round(r * (1 << s)))
    if m < 1:
        m, s = 1, min(s, 63)  # r* below 2^-s grid: dead layer, flag upstream
    return m, s


def s8_encode(r_star):
    """deployed rq_v2 (T_MAX=0): s=8 fixed, m Q8.8, r = m/256 <= 127.996."""
    m_raw = int(round(float(r_star) * 256))
    return max(0, min(M_LIM, m_raw)), m_raw


# ---------------------------------------------------------------- selftest
def _selftest():
    """The fp64 fake-quant core must match a pure-python integer pipeline
    (independent implementation, exact int arithmetic incl. >> and sat)."""
    torch.manual_seed(7)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- requant encodings ----
    for r in (1.6e-2, 1.6e-4, 4.3e-3, 0.25, 100.4, 6443.0, 1e-4):
        m, s = v1_encode(r)
        assert 1 <= m <= M_LIM, (r, m, s)
        assert abs(m / (1 << s) - r) <= 0.5 / (1 << s) + 1e-12, (r, m, s)

    for shift in (8, 13, 0, 3, None):  # None = true v1 encoding (m/s paired)
        # ---- Linear ----
        W = torch.randn(6, 9, dtype=torch.float64, device=dev)
        b = torch.randn(6, dtype=torch.float64, device=dev)
        x = torch.randn(11, 9, dtype=torch.float64, device=dev)
        sa = float(x.abs().max()) / 127.0
        sw = float(W.abs().max()) / 127.0
        out_fp = F.linear(x, W) + b
        so = float(out_fp.abs().max()) / 127.0  # consistent output scale
        xq = torch.clamp(torch.round(x / sa), -127.0, 127.0)
        wq = torch.clamp(torch.round(W / sw), -127.0, 127.0)
        b_acc = b / (sa * sw)
        c_sel, wb = None, None
        for p_ in (1, 2, 4, 8, 16, 32, 64):
            t = torch.round(b_acc / p_)
            if float(t.abs().amax()) <= 127.0:
                c_sel, wb = p_, t
                break
        assert c_sel is not None
        w_acc = wb * c_sel
        if shift is None:
            m_rq, s_sh = v1_encode((sa * sw) / so)
        elif shift == 8:
            m_rq, s_sh = s8_encode((sa * sw) / so)[0], 8
        else:
            m_rq, _ = v1_encode((sa * sw) / so)
            s_sh = shift
        acc = F.linear(xq, wq) + w_acc
        yq = torch.clamp(torch.floor(acc * m_rq / float(1 << s_sh)),
                         -128.0, 127.0)
        if shift is None:
            # SEMANTIC check: with the m-normalized v1 encoding, yq*so must
            # reconstruct the fp output (error = 1 LSB floor bias + int8
            # input/weight quantization only)
            err = (yq * so - out_fp).abs() / so
            assert float(err.median()) < 4.0, (shift, float(err.median()))

        xq_i = [[int(round(v)) for v in r] for r in xq.cpu().tolist()]
        wq_i = [[int(round(v)) for v in r] for r in wq.cpu().tolist()]
        wacc_i = [int(round(v)) for v in w_acc.cpu().tolist()]
        for i in range(11):
            for j in range(6):
                a = sum(xq_i[i][k] * wq_i[j][k] for k in range(9)) + wacc_i[j]
                ref = max(-128, min(127, (a * m_rq) >> s_sh))
                got = int(round(yq[i, j].item()))
                assert got == ref, f"lin selftest shift={s_sh} [{i},{j}]"

    # ---- Conv2d (3x3, padding=1, explicit zero-padded integer reference) ----
    conv = nn.Conv2d(2, 3, 3, padding=1, bias=True).double().to(dev)
    xc = torch.randn(1, 2, 6, 6, dtype=torch.float64, device=dev)
    sa2 = float(xc.abs().max()) / 127.0
    sw2 = float(conv.weight.abs().max()) / 127.0
    so2 = 0.5
    xq2 = torch.clamp(torch.round(xc / sa2), -127.0, 127.0)
    wq2 = torch.clamp(torch.round(conv.weight / sw2), -127.0, 127.0)
    b2_acc = conv.bias / (sa2 * sw2)
    c2, wb2 = None, None
    for p_ in (1, 2, 4, 8, 16, 32, 64):
        t = torch.round(b2_acc / p_)
        if float(t.abs().amax()) <= 127.0:
            c2, wb2 = p_, t
            break
    m2, s2 = v1_encode((sa2 * sw2) / so2)
    acc2 = F.conv2d(xq2, wq2, None, conv.stride, conv.padding,
                    conv.dilation, conv.groups) + (wb2 * c2).view(-1, 1, 1)
    yq2 = torch.clamp(torch.floor(acc2 * m2 / float(1 << s2)),
                      -128.0, 127.0)

    x2i = xq2[0].cpu().round().to(torch.int64).numpy()
    w2i = wq2.cpu().round().to(torch.int64).numpy()
    b2i = [int(round(v * c2)) for v in wb2.cpu().tolist()]
    xp = np.zeros((2, 8, 8), dtype=np.int64)
    xp[:, 1:7, 1:7] = x2i
    for oc in range(3):
        for oy in range(6):
            for ox in range(6):
                a = int(b2i[oc])
                for ic in range(2):
                    for ky in range(3):
                        for kx in range(3):
                            a += int(xp[ic, oy + ky, ox + kx]) * int(
                                w2i[oc, ic, ky, kx])
                ref = max(-128, min(127, (a * m2) >> s2))
                got = int(round(yq2[0, oc, oy, ox].item()))
                assert got == ref, f"conv selftest [{oc},{oy},{ox}]"
    print("[self] fp64 core == pure-integer reference "
          "(linear x4 shifts + conv + encodings) OK", flush=True)


# ------------------------------------------------ perturbed calibration set
def make_rgb_p(cam_idx, bowl_c, bowl_r, plate_c, plate_r):
    W0, H0 = bringup.W0, bringup.H0
    yy, xx = np.mgrid[0:H0, 0:W0].astype(np.float32)
    img = np.zeros((H0, W0, 3), np.float32)
    img[..., 0] = 40 + 0.10 * xx
    img[..., 1] = 60 + 0.08 * yy
    img[..., 2] = 90 + 0.05 * 0.5 * (xx + yy)
    img[yy > H0 / 2] = np.array([122, 96, 64], np.float32)
    img[np.hypot(xx - bowl_c[0], yy - bowl_c[1]) < bowl_r] = np.array(
        [40, 60, 180], np.float32)
    img[np.hypot(xx - plate_c[0], yy - plate_c[1]) < plate_r] = np.array(
        [212, 205, 198], np.float32)
    img += 4 * cam_idx
    return np.clip(img, 0, 255).astype(np.uint8)


def make_depth_p(T_w2c, bowl_c, bowl_r, plate_c, plate_r, table_z):
    """bringup.make_depth with parametrized geometry (same math)."""
    Kinv = np.linalg.inv(bringup.K44[:3, :3])
    yy, xx = np.mgrid[0:bringup.H0, 0:bringup.W0]
    pix = np.stack([xx, yy, np.ones_like(xx)], -1).astype(np.float64)
    dirs_cam = pix @ Kinv.T
    dirs_w = dirs_cam @ T_w2c[:3, :3].T
    origin_w = T_w2c[:3, 3]
    s = (table_z - origin_w[2]) / dirs_w[..., 2]
    s = np.where(np.isfinite(s) & (s > 0.2) & (s < 1.15), s, 1.0)
    s = s.astype(np.float32)
    s[np.hypot(xx - bowl_c[0], yy - bowl_c[1]) < bowl_r] -= 0.08
    s[np.hypot(xx - plate_c[0], yy - plate_c[1]) < plate_r] -= 0.03
    s = np.clip(s, 0.2, 1.15)
    return (np.round(s * 1000).astype(np.uint16)).astype(np.float32) / 1000.0


def perturbed_batch(processor, i):
    rng = np.random.default_rng(1000 + i)
    bowl_c = np.array([260, 300]) + rng.integers(-40, 41, 2)
    bowl_r = int(60 + rng.integers(-10, 11))
    plate_c = np.array([380, 310]) + rng.integers(-40, 41, 2)
    plate_r = int(75 + rng.integers(-12, 13))
    table_z = 0.18 + rng.uniform(-0.02, 0.02)
    target = np.array(bringup.TARGET3D) + rng.uniform(-0.04, 0.04, 3)
    images, depths, poses = {}, {}, {}
    for ci, cam in enumerate(bringup.CAM_NAMES):
        eye = np.array(BASE_EYES[cam]) + rng.uniform(-0.04, 0.04, 3)
        T = bringup.look_at_cv(eye, target)
        poses[cam] = T
        images[cam] = [make_rgb_p(ci, bowl_c, bowl_r, plate_c, plate_r)]
        depths[cam] = [make_depth_p(T, bowl_c, bowl_r, plate_c, plate_r, table_z)]
    js = bringup.JOINT_STATE.copy()
    js[:6] += rng.normal(0, 0.06, 6)
    js[7:13] += rng.normal(0, 0.06, 6)
    js[6] = js[13] = rng.uniform(0.25, 0.85)
    inp = MultiArmManipulationInput(
        image=images,
        depth=depths,
        intrinsic={c: bringup.K44.copy() for c in bringup.CAM_NAMES},
        t_world2cam={c: poses[c].copy() for c in bringup.CAM_NAMES},
        t_robot2world=bringup.T_BASE2WORLD.copy(),
        t_robot2ego=None,
        history_joint_state=[js.copy()],
        history_ee_pose=None,
        instruction=INSTRUCTIONS[i % len(INSTRUCTIONS)],
        urdf=None,
        remaining_actions=None,
        delay_horizon=None,
    )
    return processor.pre_process(inp)


# ---------------------------------------------------------------- calibration
def calibrate(model, processor, n_cal):
    """fp32 forwards on perturbed synthetic samples; per-module RUNNING max of
    input / output absmax (per-tensor)."""
    calib = {}
    handles = []
    for name, mod in model.named_modules():
        if not isinstance(mod, (nn.Linear, nn.Conv2d)):
            continue
        calib[name] = {
            "in_max": torch.zeros((), device="cuda"),
            "out_max": torch.zeros((), device="cuda"),
            "calls": 0,
            "nonfloat_calls": 0,
        }

        def pre(m, args, key=name):
            x = args[0]
            e = calib[key]
            if torch.is_tensor(x) and torch.is_floating_point(x):
                e["in_max"] = torch.maximum(e["in_max"], x.detach().abs().amax())
                e["calls"] += 1
            else:
                e["nonfloat_calls"] += 1

        def post(m, args, out, key=name):
            if torch.is_tensor(out) and torch.is_floating_point(out):
                e = calib[key]
                e["out_max"] = torch.maximum(
                    e["out_max"], out.detach().abs().amax())

        handles.append(mod.register_forward_pre_hook(pre))
        handles.append(mod.register_forward_hook(post))

    for i in range(n_cal):
        t0 = time.perf_counter()
        batch_i = perturbed_batch(processor, i)
        forward_actions(model, batch_i, 4000 + i)
        print(f"[cal ] sample {i} "
              f"(instr {i % len(INSTRUCTIONS)}) "
              f"{time.perf_counter()-t0:.1f}s", flush=True)
    for h in handles:
        h.remove()
    for e in calib.values():
        e["in_max"] = float(e["in_max"].item())
        e["out_max"] = float(e["out_max"].item())
    return calib


# ------------------------------------------------------- build hw constants
C_CANDIDATES = (1, 2, 4, 8, 16, 32, 64)


def build_hw_params(model, calib):
    inv = json.load(open(os.path.join(HB, "module_inventory.json")))
    inv_shapes = {m["name"]: m for m in inv["nn.Linear"] + inv["nn.Conv2d"]}

    params, table = {}, {}
    st = {
        "n_linear": 0, "n_conv": 0, "n_linear_quant": 0, "n_conv_quant": 0,
        "bias_free": 0, "bias_aug": 0,
        "c_hist": {}, "bias_fp_layers": [],
        "m_s8_zero_layers": [], "m_s8_tiny_layers": [],
        "m_v1_underflow_layers": [],
        "acc_over_xw27_layers": [], "never_called": [],
        "r_star_quartiles": {},
    }
    all_r = []
    for name, mod in model.named_modules():
        if not isinstance(mod, (nn.Linear, nn.Conv2d)):
            continue
        is_lin = isinstance(mod, nn.Linear)
        st["n_linear" if is_lin else "n_conv"] += 1
        if is_lin:
            dims = {"in_features": mod.in_features,
                    "out_features": mod.out_features}
            k_eff, n_eff = mod.in_features, mod.out_features
        else:
            dims = {"in_channels": mod.in_channels,
                    "out_channels": mod.out_channels,
                    "kernel_size": list(mod.kernel_size),
                    "stride": list(mod.stride), "padding": list(mod.padding),
                    "groups": mod.groups}
            k_eff = (mod.in_channels // mod.groups) * int(
                np.prod(mod.kernel_size))
            n_eff = mod.out_channels
        ent = {"type": "nn.Linear" if is_lin else "nn.Conv2d",
               "inventory": {k: inv_shapes.get(name, {}).get(k) for k in
                             ("params", "in_features", "out_features",
                              "kernel_size", "in_channels", "out_channels")},
               "gemm_k": k_eff, "gemm_n": n_eff,
               "gemm_m": "dynamic(tokens)" if is_lin else "dynamic(B*H_out*W_out)"}

        if name in EXEMPT:
            ent["exempt_fp"] = True
            table[name] = ent
            continue
        cal = calib.get(name)
        if cal is None or cal["calls"] == 0:
            st["never_called"].append(name)
            ent["never_called_in_calib"] = True
            table[name] = ent
            continue

        sa = max(cal["in_max"], 1e-12) / 127.0
        so = max(cal["out_max"], 1e-12) / 127.0
        W = mod.weight.detach().float()
        sw = max(float(W.abs().amax()), 1e-12) / 127.0
        Wq = torch.clamp(torch.round(W / sw), -127.0, 127.0)

        w_acc, c_sel, wb, bias_fp = None, None, None, False
        b_acc_absmax = None
        if mod.bias is None:
            st["bias_free"] += 1
        else:
            b_acc = mod.bias.detach().double() / (sa * sw)
            b_acc_absmax = float(b_acc.abs().amax())
            for c in C_CANDIDATES:
                w_try_raw = torch.round(b_acc / c)
                if float(w_try_raw.abs().amax()) <= 127.0:
                    c_sel, wb = c, w_try_raw
                    w_acc = (wb * c).cuda()
                    st["bias_aug"] += 1
                    st["c_hist"][str(c)] = st["c_hist"].get(str(c), 0) + 1
                    break
            if c_sel is None:
                bias_fp = True
                w_acc = b_acc.cuda()
                st["bias_fp_layers"].append(
                    {"name": name, "b_acc_absmax": b_acc_absmax,
                     "would_fit_c128": bool(b_acc_absmax <= 127 * 128)})

        r_star = (sa * sw) / so  # true multiplier r = m*2^-s (scales DOWN)
        all_r.append(r_star)
        m8, m8_raw = s8_encode(r_star)
        mv1, sv1 = v1_encode(r_star)
        if m8 == 0:
            st["m_s8_zero_layers"].append(
                {"name": name, "r_star": r_star})
        elif m8 < 8:  # multiplier precision worse than 1/8: unusable
            st["m_s8_tiny_layers"].append(
                {"name": name, "m_s8": m8})
        if mv1 / (1 << sv1) <= 0:
            st["m_v1_underflow_layers"].append(
                {"name": name, "r_star": r_star})
        acc_absmax_est = 127.0 / r_star
        if acc_absmax_est > ACC_XW27_LIMIT:
            st["acc_over_xw27_layers"].append(
                {"name": name, "acc_absmax_est": acc_absmax_est})

        params[name] = {
            "sa": sa, "so": so, "sw": sw,
            "m_s8": m8, "m_v1": mv1, "s_v1": sv1,
            "Wq64": Wq.double().cuda(), "w_acc": w_acc,
        }
        ent.update({
            "sa": sa, "sw": sw, "so": so, "r_star": r_star,
            "acc_absmax_est": acc_absmax_est,
            # rq_v2 s=8 encoding (T_MAX=0): m = round(r*256) -> 0..4, mostly dead
            "m_s8_q8_8": m8, "m_s8_dead": bool(m8 == 0),
            # hardware-capable encoding r = m / 2^s  <-- use these two
            "m_requant": mv1, "s_shift": sv1,
            "bias_aug_c": c_sel,
            "w_bias_int8": ([int(round(v)) for v in wb.cpu().tolist()]
                            if wb is not None else None),
            "bias_fp_fallback": bias_fp,
            "b_acc_absmax": b_acc_absmax,
        })
        table[name] = ent
        st["n_linear_quant" if is_lin else "n_conv_quant"] += 1

    all_r.sort()
    if all_r:
        n = len(all_r)
        st["r_star_quartiles"] = {
            "n": n, "min": all_r[0], "p25": all_r[n // 4],
            "p50": all_r[n // 2], "p75": all_r[3 * n // 4], "max": all_r[-1],
            # s=8 feasibility: m = round(r*256) >= 1 needs r >= 1/512;
            # usable precision (m >= 8) needs r >= 8/256
            "n_s8_m_ge1": sum(1 for v in all_r if v * 256 >= 0.5),
            "n_s8_m_ge8": sum(1 for v in all_r if v * 256 >= 8),
        }
    return params, table, st


# ---------------------------------------------------------------- hw patch
def _hw_linear_fwd(p, m_rq, s_sh):
    sa, so = p["sa"], p["so"]
    Wq64, w_acc = p["Wq64"], p["w_acc"]
    denom = float(1 << s_sh)

    def forward(self, x):
        out_dtype = x.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            xq = torch.clamp(torch.round(x.float() / sa),
                             -127.0, 127.0).double()
            acc = F.linear(xq, Wq64)
            if w_acc is not None:
                acc = acc + w_acc
            yq = torch.clamp(torch.floor(acc * m_rq / denom), -128.0, 127.0)
            y = yq.float() * so
        return y.to(out_dtype)

    return forward


def _hw_conv_fwd(p, m_rq, s_sh):
    sa, so = p["sa"], p["so"]
    Wq64, w_acc = p["Wq64"], p["w_acc"]
    denom = float(1 << s_sh)

    def forward(self, x):
        out_dtype = x.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            xq = torch.clamp(torch.round(x.float() / sa),
                             -127.0, 127.0).double()
            acc = F.conv2d(xq, Wq64, None, self.stride, self.padding,
                           self.dilation, self.groups)
            if w_acc is not None:
                acc = acc + w_acc.reshape(-1, 1, 1)
            yq = torch.clamp(torch.floor(acc * m_rq / denom), -128.0, 127.0)
            y = yq.float() * so
        return y.to(out_dtype)

    return forward


def patch_hw(model, params, enc="v1"):
    """enc: 'v1' -> (m_requant, s_shift); 's8' -> (m_s8_q8_8, 8)."""
    patched = []
    for name, mod in model.named_modules():
        p = params.get(name)
        if p is None:
            continue
        if enc == "v1":
            m_rq, s_sh = p["m_v1"], p["s_v1"]
        elif enc == "s8":
            m_rq, s_sh = p["m_s8"], 8
        else:
            raise ValueError(enc)
        fwd = (_hw_linear_fwd(p, m_rq, s_sh) if isinstance(mod, nn.Linear)
               else _hw_conv_fwd(p, m_rq, s_sh))
        mod.forward = types.MethodType(fwd, mod)
        patched.append(mod)

    def restore():
        for m in patched:
            m.__dict__.pop("forward", None)

    return len(patched), restore


# ------------------------------------------------------- diagnostics passes
def cache_fp_outputs(model, batch):
    """fp pass (model MUST be unpatched); cache every Linear/Conv output."""
    cache = {}
    handles = []
    budget = [0]

    def mk(name):
        def h(mod, inp, out):
            if torch.is_tensor(out) and torch.is_floating_point(out):
                t = out.detach().float().cpu()
                budget[0] += t.numel() * 4
                if budget[0] < 16 * 2**30:  # 16 GiB guard
                    cache.setdefault(name, []).append(t)
        return h

    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            handles.append(mod.register_forward_hook(mk(name)))
    forward_actions(model, batch, 1000)
    for h in handles:
        h.remove()
    print(f"[cache] fp outputs cached for {len(cache)} modules "
          f"({budget[0]/2**30:.1f} GiB CPU)", flush=True)
    return cache


def diff_pass(model, batch, params, cache):
    """mode-B pass; per-module streaming diff vs the cached fp outputs."""
    counters, rec = {}, {}
    handles = []
    mods = dict(model.named_modules())

    def mk(name):
        def h(mod, inp, out):
            i = counters.get(name, 0)
            counters[name] = i + 1
            refs = cache.get(name)
            if not refs or i >= len(refs) or not torch.is_tensor(out):
                return
            ref = refs[i].to(out.device)
            o = out.detach().float()
            if o.shape != ref.shape:
                return
            d = (o - ref).abs()
            r = rec.setdefault(name, {"sum_d": 0.0, "sum_o": 0.0,
                                      "n": 0, "calls": 0})
            r["sum_d"] += float(d.sum())
            r["sum_o"] += float(ref.abs().sum())
            r["n"] += d.numel()
            r["calls"] += 1
        return h

    for name in params:
        handles.append(mods[name].register_forward_hook(mk(name)))
    forward_actions(model, batch, 1000)
    for h in handles:
        h.remove()

    rows = []
    for name, r in rec.items():
        if r["n"] == 0:
            continue
        rel = r["sum_d"] / max(r["sum_o"], 1e-12)
        rows.append({
            "name": name,
            "scope": next((s for s, ps in MODULE_SCOPES.items()
                           if _in_scope(name, ps)), "other"),
            "rel_mae": rel,
            "abs_mae": r["sum_d"] / r["n"],
            "fp_mean_abs": r["sum_o"] / r["n"],
            "calls": r["calls"],
        })
    rows.sort(key=lambda x: -x["rel_mae"])
    return rows


def coverage_pass(model, batch, params, table):
    """mode-B pass; where do eval-time activations exceed the calibrated sa,
    and how often does the requant output sit on the saturation rail."""
    info = {}
    handles = []
    mods = dict(model.named_modules())

    def mkpre(name):
        def h(mod, args):
            x = args[0]
            if torch.is_tensor(x) and torch.is_floating_point(x):
                e = info.setdefault(name, {"eval_in_max": 0.0, "sat_frac": 0.0})
                e["eval_in_max"] = max(e["eval_in_max"],
                                       float(x.detach().abs().amax()))
        return h

    def mkpost(name):
        def h(mod, inp, out):
            if not (torch.is_tensor(out) and torch.is_floating_point(out)):
                return
            so = table[name]["so"]
            yq = out.detach() / so
            f = float((yq.abs() >= 127.0).float().mean())
            e = info.setdefault(name, {"eval_in_max": 0.0, "sat_frac": 0.0})
            e["sat_frac"] = max(e["sat_frac"], f)
        return h

    for name in params:
        handles.append(mods[name].register_forward_pre_hook(mkpre(name)))
        handles.append(mods[name].register_forward_hook(mkpost(name)))
    forward_actions(model, batch, 1000)
    for h in handles:
        h.remove()

    exceed, sat = [], []
    for name, e in info.items():
        sa_limit = table[name]["sa"] * 127.0
        ratio = e["eval_in_max"] / max(sa_limit, 1e-12)
        if ratio > 1.05:
            exceed.append({"name": name, "ratio": ratio})
        sat.append({"name": name, "sat_frac": e["sat_frac"]})
    exceed.sort(key=lambda x: -x["ratio"])
    sat.sort(key=lambda x: -x["sat_frac"])
    return {
        "n_modules_checked": len(params),
        "n_input_exceeds_calib_gt5pct": len(exceed),
        "worst_input_exceed": exceed[:20],
        "top_sat_frac": sat[:20],
        "n_sat_frac_gt1pct": sum(1 for s in sat if s["sat_frac"] > 0.01),
    }


# ---------------------------------------------------------------- main
def main(mode):
    full = mode == "full"
    n_cal = 8 if full else 2
    eval_seeds = EVAL_SEEDS if full else EVAL_SEEDS[:1]
    torch.backends.cudnn.deterministic = True
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.perf_counter()
    _selftest()

    processor, model, batch = load_everything()
    n_mha = force_mha_slow_path(model)
    print(f"[load] model up; forced eager path on {n_mha} MHA modules",
          flush=True)

    last = json.load(open(os.path.join(HB, "gate_results.json")))
    floor = last["noise_floor"]["mean"]
    gate_seed1000 = next(p for p in last["modes"]["w8a8"]["per_seed"]
                         if p["seed"] == 1000)

    # 1. fp references on the canonical bringup batch
    fp = {s: forward_actions(model, batch, s) for s in eval_seeds}
    print(f"[fp  ] references at seeds {eval_seeds} "
          f"({time.perf_counter()-t_start:.0f}s)", flush=True)

    # 2. fp output cache for the per-layer diff ranking (full mode only;
    #    MUST run while the model is still unpatched)
    cache = cache_fp_outputs(model, batch) if full else None

    # 3. calibration on perturbed synthetic samples
    calib = calibrate(model, processor, n_cal)

    # 4. build hardware constants
    params, table, st = build_hw_params(model, calib)
    q = st["r_star_quartiles"]
    print(f"[bl.d] linear {st['n_linear_quant']}/{st['n_linear']} conv "
          f"{st['n_conv_quant']}/{st['n_conv']} | r=min/p50/max = "
          f"{q.get('min', 0):.2e}/{q.get('p50', 0):.2e}/{q.get('max', 0):.2e} "
          f"| s=8 encoding: m>=1 on {q.get('n_s8_m_ge1', 0)}/{q.get('n', 0)}, "
          f"m>=8 on {q.get('n_s8_m_ge8', 0)} | bias_aug "
          f"{st['bias_aug']} (c hist {st['c_hist']}) bias_free {st['bias_free']} "
          f"bias_fp {len(st['bias_fp_layers'])} | m_v1 underflow "
          f"{len(st['m_v1_underflow_layers'])} acc>XW27 "
          f"{len(st['acc_over_xw27_layers'])} | never_called "
          f"{len(st['never_called'])}", flush=True)
    if st["never_called"]:
        print(f"[bl.d] never_called: {st['never_called']}", flush=True)
    if st["acc_over_xw27_layers"]:
        print(f"[bl.d] acc_over_xw27: {st['acc_over_xw27_layers']}", flush=True)

    # 5. mode A: last round's w8a8 semantics, 1 seed (environment re-check)
    _, restore_a = apply_fake_quant(model, "w8a8")
    t0 = time.perf_counter()
    qa = forward_actions(model, batch, 1000)
    t_modeA = time.perf_counter() - t0
    restore_a()
    modeA = act_metrics(fp[1000], qa)
    dA = abs(modeA["mae_all"] - gate_seed1000["mae_all"])
    print(f"[A  ] w8a8 rerun seed1000 mae_all={modeA['mae_all']:.5f} "
          f"(gate: {gate_seed1000['mae_all']:.5f}, |d|={dA:.5f}) "
          f"jpos={modeA['mae_jointpos']:.5f} | {t_modeA:.1f}s", flush=True)

    # 6a. mode B_s8: deployed requant encoding (s=8 fixed, m Q8.8 <= 32767)
    n_s8, restore_s8 = patch_hw(model, params, enc="s8")
    qs8 = forward_actions(model, batch, 1000)
    modeB_s8 = act_metrics(fp[1000], qs8)
    restore_s8()
    print(f"[B_s8] rq_v2 T_MAX=0 encoding: mae_all={modeB_s8['mae_all']:.5f} "
          f"jpos={modeB_s8['mae_jointpos']:.5f} "
          f"({len(st['m_s8_zero_layers'])} dead (m=0), "
          f"{len(st['m_s8_tiny_layers'])} precision-starved (1<=m<8) -> "
          f"documented failure mode)", flush=True)

    # 6b. mode B_v1: barrel-shifter requant encoding r = m/2^s (the gate)
    n_v1, restore_b = patch_hw(model, params, enc="v1")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    per_seed = []
    for s in eval_seeds:
        qv = forward_actions(model, batch, s)
        m = act_metrics(fp[s], qv)
        per_seed.append({"seed": s, **m})
        print(f"[B_v1] seed {s}: mae_all={m['mae_all']:.5f} "
              f"jpos={m['mae_jointpos']:.5f} max_jpos={m['max_jointpos']:.4f} "
              f"grip={m['mae_gripper']:.5f}", flush=True)
    torch.cuda.synchronize()
    t_modeB = (time.perf_counter() - t0) / len(eval_seeds)
    mean_b = mean_of_dicts(per_seed)
    jpos_mean = mean_b["mae_jointpos"]
    gate = ("green" if jpos_mean <= GATE_GREEN
            else "yellow" if jpos_mean <= GATE_YELLOW else "red")

    cov = coverage_pass(model, batch, params, table)
    print(f"[cov ] input>calib*1.05 on {cov['n_input_exceeds_calib_gt5pct']}/"
          f"{cov['n_modules_checked']} modules; sat_frac>1% on "
          f"{cov['n_sat_frac_gt1pct']}", flush=True)

    if full:
        top_rows = diff_pass(model, batch, params, cache)
        del cache
        print("[top10] highest relative output drift (B_v1 vs fp, seed 1000):",
              flush=True)
        for r in top_rows[:10]:
            print(f"      {r['rel_mae']:8.4f}  {r['name'][:72]:72s} "
                  f"{r['scope']:11s} abs={r['abs_mae']:.5f}", flush=True)
    else:
        top_rows = []
    restore_b()
    print(f"[B_v1] mean jpos={jpos_mean:.5f} (gate {GATE_GREEN}/{GATE_YELLOW} "
          f"-> {gate}) | {t_modeB:.1f}s/fwd", flush=True)

    # 7. save
    meta = {
        "model": "HoloBrain_v0.0_GD@post_training_robotwin",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dtype": "float32 base; int8 hw path simulated in float64 (exact)",
        "mha_slow_path_forced": n_mha,
        "mode": mode,
        "calib_samples": n_cal,
        "calib_instructions": INSTRUCTIONS,
        "eval_seeds": eval_seeds,
        "exempt": sorted(EXEMPT),
        "semantics": (
            "per-tensor static symmetric int8 A and W; exact int32 accum; "
            "bias via K+1 aug (acc += round(b/(sa*sw*c))*c, c pow2<=64, "
            "fp fallback if overflow); requant ONE static (m,s) per GEMM: "
            "y = sat8((acc*m)>>>s), pure floor shift, no rounding constant; "
            "output (int8 y, so); CPU ops consume fp y*so and the next GEMM "
            "re-quantizes with its calibrated sa"
        ),
        "requant_encodings": {
            "direction": (
                "r = m*2^-s = (sa*sw)/so (acc*(sa*sw) = fp output; "
                "y_int8 = fp/so). NOTE: the task brief's r* = so/(sa*sw) is "
                "the inverse; using it saturates every output"
            ),
            "s8_tmax0_deployed": (
                "m=round(r*256) with s=8 fixed (rq_v2 T_MAX=0 in ae_gemm); "
                "measured r*256 = 0.04..4.05 -> "
                f"{len(st['m_s8_zero_layers'])} layers dead (m=0), "
                f"{len(st['m_s8_tiny_layers'])} at 12-50% multiplier steps; "
                "infeasible without widening the shift range"
            ),
            "v1_or_tmax_ge19": (
                "r = m/2^s with m normalized into [16384,32767] (15-bit "
                "precision); needed s in [8,27]; covered by ae_requant/rq_v1 "
                "barrel shifter (any s) or rq_v2 with T_MAX>=19 (t=s-8); use "
                "the m_requant + s_shift fields"
            ),
        },
        "gate_thresholds": {"green": GATE_GREEN, "yellow": GATE_YELLOW,
                            "metric": "mean mae_jointpos (mode B_v1)"},
    }
    results = {
        "meta": meta,
        "noise_floor_reference_last_round": floor,
        "modeA_w8a8_rerun": {
            "seed": 1000, **modeA,
            "gate_seed1000_last_round": gate_seed1000,
            "abs_diff_mae_all": dA,
            "reproduces": bool(dA < 0.002),
            "seconds": t_modeA,
        },
        "modeB_s8_deployed_requant": {
            "seed": 1000, **modeB_s8,
            "n_layers_m_zero": len(st["m_s8_zero_layers"]),
            "n_layers_m_tiny": len(st["m_s8_tiny_layers"]),
            "note": ("rq_v2 with T_MAX=0 (s=8 only): multiplier r*256 rounds "
                     "to 0..4; dead or precision-starved on most layers. "
                     "rq_v2 already supports T_MAX=39 (s in [8,47]); setting "
                     "T_MAX>=19 in ae_gemm makes the B_v1 encoding legal"),
        },
        "modeB_hw_v1": {
            "n_modules_patched": n_v1,
            "per_seed": per_seed,
            "mean": mean_b,
            "jpos_mae_over_floor_mae_all": jpos_mean / floor["mae_all"],
            "jpos_mae_over_floor_jpos": jpos_mean / floor["mae_jointpos"],
            "gate": gate,
            "seconds_per_forward": t_modeB,
        },
        "build_stats": st,
        "coverage": cov,
        "top_error_layers": top_rows[:30] if full else [],
        "total_seconds": time.perf_counter() - t_start,
    }
    p_res = os.path.join(OUT_DIR, "hw_calib_results.json")
    with open(p_res, "w") as f:
        json.dump(results, f, indent=1)
    p_tab = os.path.join(OUT_DIR, "hw_calib_table.json")
    with open(p_tab, "w") as f:
        json.dump({"_meta": meta, "gemms": table}, f, indent=1)
    print(f"[save] {p_res} ({os.path.getsize(p_res)/1e3:.0f} kB)", flush=True)
    print(f"[save] {p_tab} ({os.path.getsize(p_tab)/1e3:.0f} kB)", flush=True)
    print(f"[done] gate={gate} total {time.perf_counter()-t_start:.0f}s",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "full"])
    main(ap.parse_args().mode)
