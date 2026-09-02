# int4_scan.py -- W4 weight fake-quant sensitivity scan for HoloBrain HB-GD 0.2B
# (hb_fpga_impl/07_int4 round, 2026-08-31). Pure software: no RTL, no compiler.
#
# QUESTION: which layers can drop weights from INT8 to INT4 (per-tensor or
# group-wise) with policy deviation staying under control, given the deployed
# W8A8 integer pipeline everywhere else?
#
# Baseline reproduced here = hw_calib.py mode B_v1 (the gate-green 0.0288 rad
# run): per-tensor static sa/so (from hw_calib_table_v2.json, calibrated on the
# 8 perturbed synthetic samples), per-tensor sw, exact int accum in float64,
# K+1 bias augmentation, requant y = sat8((acc*m)>>>s) with r=(sa*sw)/so,
# floor shift (no rounding constant), output y*so. in_proj stays fp in this
# baseline exactly like mode B_v1 did.
#
# W4 variants for a layer (activation INT8 per-tensor and the requant
# structure are kept):
#   w4pt : per-tensor symmetric INT4, sw4=absmax(W)/7. acc stays PURE INTEGER
#          (acc = sum xq*wq4int), r=(sa*sw4)/so -> (m,s)=v1_encode. Bit-level
#          identical to hardware that just stores 4-bit weights and swaps the
#          constant table -- zero requant change, zero RTL change.
#   w4gN : group-wise symmetric INT4 along the reduction dim (group=N, zero
#          pad). acc = sum xq*(wq4int*sg): per-group dequant folded into the
#          accumulator, then ONE requant multiplier r = sa/so (measured sa/so
#          in [0.029,12.9] -> v1 s in [11,20], m at 15-bit precision).
#          Models the standard W4A8-groupwise deployment; the per-group scale
#          multiplies are a NEXT-round RTL item, out of scope here.
#   bias : K+1 aug with the same c-loop; denom = sa*sw (w8) / sa*sw4 (w4pt) /
#          sa (w4gN, acc already in weight units). fp fallback if overflow.
#
# Eval protocol ("same as 02_quant"):
#   real  : 04_dataset samples 000/001/002 rebuilt bit-exactly from the
#           RoboTwin HDF5s (same code path as extract_sample.py), fp ref vs
#           quant at fixed denoise seed, act_metrics in rad.
#   synth : the bringup batch, seeds 1000/1100/1200 -> directly comparable
#           with the 0.0288 gate number.
#
# Run (server, /tmp because /home is full):
#   cd /tmp/int4_hb && CUDA_VISIBLE_DEVICES=2 \
#     /home/nc23/.conda/envs/holobrain/bin/python int4_scan.py <cmd> ...
# cmds: prep | w8base | anchor | scan | evalcfg | bytes

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
sys.path.insert(0, "/tmp/int4_hb")  # hw_calib_table_v2.json lives here

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import bringup  # noqa: E402  (sets shims/repo paths + HF_ENDPOINT)
from gate import (  # noqa: E402
    act_metrics,
    force_mha_slow_path,
    forward_actions,
    mean_of_dicts,
)
import extract_sample as XS  # noqa: E402
from robo_orchard_lab.models.holobrain.processor import (  # noqa: E402
    HoloBrainProcessor,
    MultiArmManipulationInput,
)
from robo_orchard_lab.models.mixin import ModelMixin  # noqa: E402

WORK = "/tmp/int4_hb"
TAB = json.load(open(os.path.join(WORK, "hw_calib_table_v2.json")))["gemms"]
EXEMPT = {"spatial_enhancer.pts_prob_fc.layers.1"}
M_LIM = 32767
C_CAND = (1, 2, 4, 8, 16, 32, 64)
SEEDS_REAL = [20260830, 20260901, 20260902]
SEEDS_SYN = [1000, 1100, 1200]
GATE_W8_SYN = 0.028811758384108543  # hw_calib modeB_v1 mean jpos (regression)

# layer key -> mode ("w8" | "w4pt" | "w4g64" | "w4g128" | "fp").
# Defaults: "w8" for everything in the table; MHA in_proj keys default "fp"
# (mode B_v1 never quantized them); exempt stays fp.
CFG = {}
CACHE_W8 = {}   # key -> (Wq_int, w_acc, m, denom, fp_bias_or_None)
CACHE_W4 = {}   # (key, mode) -> same tuple; cleared per scan step
STATS = {"bias_fp_w4": [], "cache_skipped": [], "w4_r_over": []}

LINEAR_KEYS = {}
CONV_KEYS = {}
CONV1D_KEYS = {}
MHA_KEYS = []


def v1_encode(r_star):
    r = max(float(r_star), 1e-30)
    s = max(0, int(math.floor(math.log2(M_LIM / r))))
    m = int(round(r * (1 << s)))
    while m > M_LIM and s > 0:
        s -= 1
        m = int(round(r * (1 << s)))
    if m < 1:
        m, s = 1, min(s, 63)
    return m, s


# ---------------------------------------------------------------- weight prep
def _bias_aug(b_acc):
    """K+1 augmentation: w_bias = round(b_acc/c) in [-127,127], c pow2<=64."""
    for c in C_CAND:
        t = torch.round(b_acc / c)
        if float(t.abs().amax()) <= 127.0:
            return t * c
    return None


def w8_tensors(key, weight, shape):
    e = TAB[key]
    sw = e["sw"]
    Wd = weight.double()
    if shape is not None:
        flat = Wd.reshape(shape[0], -1)
        Wq = torch.clamp(torch.round(flat / sw), -127.0, 127.0)
        Wq = Wq.reshape(shape).double()
    else:
        Wq = torch.clamp(torch.round(Wd / sw), -127.0, 127.0)
    w_acc, fp_bias = None, None
    if e["w_bias_int8"] is not None:
        w_acc = torch.tensor(e["w_bias_int8"], dtype=torch.float64,
                             device=weight.device) * e["bias_aug_c"]
    # else: no bias, or bias_fp_fallback -> fp add in forward
    return Wq, w_acc, fp_bias, e["m_requant"], float(1 << e["s_shift"]), e


def w4_tensors(key, weight, bias, shape, mode):
    """INT4 weights. Returns (Wq, w_acc, fp_bias, m, denom, e).

    w4pt:  Wq = int4 VALUES (integers in fp64), acc pure integer,
           r = (sa*sw4)/so.
    w4gN:  Wq = wq4int*sg (dequantized), acc in weight units, r = sa/so.
    """
    e = TAB[key]
    sa, so = e["sa"], e["so"]
    Wd = weight.double()
    flat = Wd.reshape(shape[0], -1) if shape is not None else Wd
    out_f, in_f = flat.shape

    if mode == "w4pt":
        sw4 = max(float(flat.abs().amax()), 1e-12) / 7.0
        Wq = torch.clamp(torch.round(flat / sw4), -7.0, 7.0)
        if shape is not None:
            Wq = Wq.reshape(shape).double()
        m, s = v1_encode((sa * sw4) / so)
        w_acc = fp_bias = None
        if bias is not None:
            w_acc = _bias_aug(bias.detach().double() / (sa * sw4))
            if w_acc is None:
                fp_bias = bias.detach().double() / (sa * sw4)
        return Wq, w_acc, fp_bias, m, float(1 << s), e

    if mode.startswith("w4q"):
        # QoQ-style (QServe, MLSys'25) deployment: group-wise INT4 storage,
        # but unpacked back to INT8 on a PER-TENSOR scale at load time
        # (w_int8 = round(wq4*sg/sw_eff)). Zero datapath change: the GEMM
        # stays a plain int8 GEMM, requant r = (sa*sw_eff)/so as today.
        # sw_eff = max_g sg = absmax/7 exactly, so small-scale groups lose
        # resolution vs exact dequant -- this measures that cost.
        group = int(mode[len("w4q"):])
        pad = (-in_f) % group
        Wp = F.pad(flat, (0, pad)) if pad else flat
        G_ = Wp.view(out_f, -1, group)
        sg = G_.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 7.0
        wq4 = torch.round(G_ / sg).clamp_(-7.0, 7.0)
        sw_eff = max(float(flat.abs().amax()), 1e-12) / 7.0  # == max sg
        Wq = torch.clamp(torch.round(wq4 * sg / sw_eff), -127.0, 127.0)
        Wq = Wq.view(out_f, -1)[:, :in_f]
        if shape is not None:
            Wq = Wq.reshape(shape).double()
        m, s = v1_encode((sa * sw_eff) / so)
        w_acc = fp_bias = None
        if bias is not None:
            w_acc = _bias_aug(bias.detach().double() / (sa * sw_eff))
            if w_acc is None:
                fp_bias = bias.detach().double() / (sa * sw_eff)
        return Wq, w_acc, fp_bias, m, float(1 << s), e

    group = int(mode[len("w4g"):])
    pad = (-in_f) % group
    Wp = F.pad(flat, (0, pad)) if pad else flat
    G = Wp.view(out_f, -1, group)
    sg = G.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 7.0
    Wexp = torch.round(G / sg).clamp_(-7.0, 7.0) * sg
    Wexp = Wexp.view(out_f, -1)[:, :in_f]
    if shape is not None:
        Wexp = Wexp.reshape(shape).double()
    r = sa / so
    if r > M_LIM:
        STATS["w4_r_over"].append((key, mode, r))
    m, s = v1_encode(r)
    w_acc = fp_bias = None
    if bias is not None:
        w_acc = _bias_aug(bias.detach().double() / sa)
        if w_acc is None:
            fp_bias = bias.detach().double() / sa
    return Wexp, w_acc, fp_bias, m, float(1 << s), e


def get_tensors(key, mod, shape, mode):
    if mode == "w8":
        if key not in CACHE_W8:
            CACHE_W8[key] = w8_tensors(key, mod.weight, shape)
        return CACHE_W8[key]
    ck = (key, mode)
    if ck not in CACHE_W4:
        bias = getattr(mod, "bias", None)
        CACHE_W4[ck] = w4_tensors(key, mod.weight, bias, shape, mode)
    return CACHE_W4[ck]


# ---------------------------------------------------------------- fwd patches
def make_linear_fwd(key):
    def forward(self, x):
        mode = CFG.get(key, "w8")
        if mode == "fp":
            return nn.Linear.forward(self, x)
        Wq, w_acc, fp_bias, m, denom, e = get_tensors(key, self, None, mode)
        out_dtype = x.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            xq = torch.clamp(torch.round(x.float() / e["sa"]),
                             -127.0, 127.0).double()
            acc = F.linear(xq, Wq)
            if w_acc is not None:
                acc = acc + w_acc
            elif fp_bias is not None:
                acc = acc + fp_bias
            elif self.bias is not None and e["bias_fp_fallback"]:
                acc = acc + self.bias.detach().double() / (e["sa"] * e["sw"])
            yq = torch.clamp(torch.floor(acc * m / denom), -128.0, 127.0)
            y = yq.float() * e["so"]
        return y.to(out_dtype)

    return forward


def make_conv_fwd(key, is_conv1d):
    def forward(self, x):
        mode = CFG.get(key, "w8")
        if mode == "fp":
            return (nn.Conv1d.forward if is_conv1d else nn.Conv2d.forward)(
                self, x)
        Wq, w_acc, fp_bias, m, denom, e = get_tensors(
            key, self, self.weight.shape, mode)
        out_dtype = x.dtype
        bshape = (-1, 1) if is_conv1d else (-1, 1, 1)
        with torch.autocast(device_type="cuda", enabled=False):
            xq = torch.clamp(torch.round(x.float() / e["sa"]),
                             -127.0, 127.0).double()
            acc = (F.conv1d if is_conv1d else F.conv2d)(
                xq, Wq, None, self.stride, self.padding, self.dilation,
                self.groups)
            if w_acc is not None:
                acc = acc + w_acc.view(bshape)
            elif fp_bias is not None:
                acc = acc + fp_bias.view(bshape)
            elif self.bias is not None and e["bias_fp_fallback"]:
                acc = acc + (self.bias.detach().double()
                             / (e["sa"] * e["sw"])).view(bshape)
            yq = torch.clamp(torch.floor(acc * m / denom), -128.0, 127.0)
            y = yq.float() * e["so"]
        return y.to(out_dtype)

    return forward


class _WCarrier:
    """Weight carrier so MHA in_proj (a raw Parameter) shares the quant path."""

    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias


def make_mha_fwd(mha_key):
    """Eager MHA (same math as gate._mha_eager) whose fused qkv GEMM runs the
    quant pipeline when CFG[mha_key+'.in_proj_weight'] != 'fp'. out_proj is a
    child Linear and follows its own patch."""
    ipk = mha_key + ".in_proj_weight"

    def _qkv(self, x):
        mode = CFG.get(ipk, "fp")
        E = self.embed_dim
        w, b = self.in_proj_weight, self.in_proj_bias
        if mode == "fp":
            return (F.linear(x, w[:E], b[:E] if b is not None else None),
                    F.linear(x, w[E:2 * E], b[E:2 * E] if b is not None
                             else None),
                    F.linear(x, w[2 * E:], b[2 * E:] if b is not None
                             else None))
        Wq, w_acc, fp_bias, m, denom, e = get_tensors(
            ipk, _WCarrier(w, b), None, mode)
        out_dtype = x.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            xq = torch.clamp(torch.round(x.float() / e["sa"]),
                             -127.0, 127.0).double()
            acc = F.linear(xq, Wq)
            if w_acc is not None:
                acc = acc + w_acc
            elif fp_bias is not None:
                acc = acc + fp_bias
            elif b is not None and e["bias_fp_fallback"]:
                acc = acc + b.detach().double() / (e["sa"] * e["sw"])
            yq = torch.clamp(torch.floor(acc * m / denom), -128.0, 127.0)
            y3 = yq.float() * e["so"]
        y3 = y3.to(out_dtype)
        return y3[..., :E], y3[..., E:2 * E], y3[..., 2 * E:]

    def forward(self, query, key=None, value=None, key_padding_mask=None,
                need_weights=True, attn_mask=None, **kw):
        E, H = self.embed_dim, self.num_heads
        q, k, v = _qkv(self, query)
        if key is not None and key is not query:
            w, b = self.in_proj_weight, self.in_proj_bias
            k = F.linear(key, w[E:2 * E],
                         b[E:2 * E] if b is not None else None)
            v = F.linear(value, w[2 * E:],
                         b[2 * E:] if b is not None else None)
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
                key_padding_mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1).view(B * H, N, N)
        out = torch.bmm(attn, v).transpose(0, 1).reshape(N, B, E)
        out = self.out_proj(out)  # child Linear -> own dispatcher patch
        return out, None

    return forward


def patch_all(model):
    """One dispatcher patch per quantizable module; CFG decides the mode."""
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.MultiheadAttention):
            if (name + ".in_proj_weight") in TAB:
                mod.forward = types.MethodType(make_mha_fwd(name), mod)
                MHA_KEYS.append(name)
                n += 1
        if name in EXEMPT or name not in TAB:
            continue
        if isinstance(mod, nn.Linear):
            mod.forward = types.MethodType(make_linear_fwd(name), mod)
            LINEAR_KEYS[name] = mod
            n += 1
        elif isinstance(mod, nn.Conv2d):
            mod.forward = types.MethodType(make_conv_fwd(name, False), mod)
            CONV_KEYS[name] = mod
            n += 1
        elif isinstance(mod, nn.Conv1d):
            mod.forward = types.MethodType(make_conv_fwd(name, True), mod)
            CONV1D_KEYS[name] = mod
            n += 1
    return n


# ---------------------------------------------------------------- data
def load_model():
    processor = HoloBrainProcessor.load(bringup.CKPT, bringup.PROCESSOR_JSON)
    model = ModelMixin.load_model(bringup.MODEL_DIR, load_impl="native")
    model = model.cuda().float().eval()
    n_mha = force_mha_slow_path(model)
    return processor, model, n_mha


def real_batch(processor, k):
    """Rebuild sample k exactly like extract_sample.py did (same HDF5, same
    code path); cmd prep verifies the fp forward against fp32_ref_00k.npz."""
    fname, t, imode, vidx = XS.SAMPLES[k]
    h = h5py.File(os.path.join(XS.DATA_DIR, fname), "r")
    if imode == "scalar":
        text = h["instruction"][()]
        text = text.decode() if isinstance(text, bytes) else str(text)
    else:
        v = h["instructions"][vidx]
        text = v.decode() if isinstance(v, bytes) else str(v)
    images, depths, intr, w2c, joint, z_plane = XS.build_sample(h, t, text)
    inp = MultiArmManipulationInput(
        image=images, depth=depths, intrinsic=intr, t_world2cam=w2c,
        t_robot2world=XS.T_BASE2WORLD.copy(), t_robot2ego=None,
        history_joint_state=[joint.copy()], history_ee_pose=None,
        instruction=text, urdf=None, remaining_actions=None, delay_horizon=None)
    batch = processor.pre_process(inp)
    h.close()
    return batch, text


def synth_batch(processor):
    images, depths, _ = bringup.build_raw_inputs()
    inp = MultiArmManipulationInput(
        image=images, depth=depths,
        intrinsic={c: bringup.K44.copy() for c in bringup.CAM_NAMES},
        t_world2cam={c: bringup.CAM_POSES[c].copy()
                     for c in bringup.CAM_NAMES},
        t_robot2world=bringup.T_BASE2WORLD.copy(), t_robot2ego=None,
        history_joint_state=[bringup.JOINT_STATE.copy()],
        history_ee_pose=None, instruction=bringup.INSTRUCTION, urdf=None,
        remaining_actions=None, delay_horizon=None)
    return processor.pre_process(inp)


def batches_and_refs(processor, model, force=False):
    """MUST run on an UNPATCHED model (fp references). Cached afterwards."""
    p_b = os.path.join(WORK, "batches.pt")
    p_r = os.path.join(WORK, "fp_refs.pt")
    if os.path.exists(p_b) and os.path.exists(p_r) and not force:
        B = torch.load(p_b, weights_only=False)
        R = torch.load(p_r, weights_only=False)
        return B, R
    real, texts = [], []
    for k in range(3):
        b, text = real_batch(processor, k)
        real.append(b)
        texts.append(text)
        print(f"[prep] real batch {k} built ({text!r})", flush=True)
    syn = synth_batch(processor)
    refs = {}
    for k in range(3):
        for s in SEEDS_REAL:
            refs[f"real:{k}:{s}"] = forward_actions(model, real[k], s)
    for s in SEEDS_SYN:
        refs[f"syn:{s}"] = forward_actions(model, syn, s)
    pairs = []
    for sa_, sb_ in [(20260830, 20260901), (20260830, 20260902),
                     (20260901, 20260902)]:
        pairs.append({"seeds": [sa_, sb_],
                      **act_metrics(refs[f"real:0:{sa_}"],
                                    refs[f"real:0:{sb_}"])})
    B = {"real": real, "syn": syn, "texts": texts}
    R = {"refs": refs, "noise_pairs_real0": pairs}
    torch.save(B, p_b)
    torch.save(R, p_r)
    return B, R


def _eval_all(model, B, R, tag):
    out = {"real": [], "syn": []}
    for k in range(3):
        per = []
        for s in SEEDS_REAL:
            q = forward_actions(model, B["real"][k], s)
            per.append({"seed": s, **act_metrics(R["refs"][f"real:{k}:{s}"], q)})
        out["real"].append({"sample": k, "per_seed": per,
                            "mean": mean_of_dicts(per)})
    for s in SEEDS_SYN:
        q = forward_actions(model, B["syn"], s)
        out["syn"].append({"seed": s,
                           **act_metrics(R["refs"][f"syn:{s}"], q)})
    syn_mean = mean_of_dicts(out["syn"])
    jpos_syn = syn_mean["mae_jointpos"]
    print(f"[{tag}] real jpos: " + " ".join(
        f"s{r['sample']}={r['mean']['mae_jointpos']:.5f}"
        for r in out["real"])
        + f" | syn mean jpos={jpos_syn:.5f} "
          f"(W8 gate 0.02881 -> x{jpos_syn / GATE_W8_SYN:.2f})", flush=True)
    return {"tag": tag, "real": out["real"],
            "syn": {"per_seed": out["syn"], "mean": syn_mean}}


# ---------------------------------------------------------------- cmds
def cmd_prep():
    torch.backends.cudnn.deterministic = True
    processor, model, n_mha = load_model()
    print(f"[load] model up (mha eager on {n_mha})", flush=True)
    B, R = batches_and_refs(processor, model)
    fm = mean_of_dicts(R["noise_pairs_real0"])
    print(f"[flor] real0 noise floor mae_all={fm['mae_all']:.5f} "
          f"jpos={fm['mae_jointpos']:.5f}", flush=True)
    ref = np.load("/home/nc23/workspace/holobrain/robotwin_subset/samples/"
                  "fp32_ref_000.npz")
    torch.manual_seed(20260830)
    with torch.no_grad():
        outs = model(B["real"][0])
    pa = outs[0]["pred_actions"].detach().cpu().numpy()
    d_ref = float(np.abs(pa - ref["pred_actions_raw"]).max())
    action = processor.post_process(outs, B["real"][0]).action.cpu().numpy()
    d_act = float(np.abs(action - ref["action"]).max())
    truth = np.load("/home/nc23/workspace/holobrain/robotwin_subset/samples/"
                    "truth_000.npz")["future_actions"]
    mae = float(np.abs(action - truth).mean())
    print(f"[vrfy] vs fp32_ref_000: pred max|d|={d_ref:.3e} "
          f"action max|d|={d_act:.3e} | action-vs-truth MAE={mae:.4f} rad "
          f"(stored {float(ref['mae_all14']):.4f})", flush=True)
    ok = d_act < 1e-5 and abs(mae - float(ref["mae_all14"])) < 2e-4
    print(f"[vrfy] reconstruction "
          f"{'BIT-EXACT OK' if ok else 'MISMATCH -- STOP AND DEBUG'}",
          flush=True)
    return ok


def cmd_w8base():
    torch.backends.cudnn.deterministic = True
    processor, model, _ = load_model()
    B, R = batches_and_refs(processor, model)  # refs BEFORE any patch
    n_patch = patch_all(model)
    print(f"[patch] dispatcher on {n_patch} modules (default W8, in_proj fp)",
          flush=True)
    res = _eval_all(model, B, R, "w8base")
    res["meta"] = {"note": "mode B_v1 semantics on real samples + synth; "
                    "regression anchor = 0.02881 synth mean jpos"}
    with open(os.path.join(WORK, "w8base.json"), "w") as f:
        json.dump(res, f, indent=1)
    for k in MHA_KEYS:  # deployed runtime also quantizes in_proj (W8)
        CFG[k + ".in_proj_weight"] = "w8"
    res2 = _eval_all(model, B, R, "w8+inprojW8")
    with open(os.path.join(WORK, "w8base_inproj.json"), "w") as f:
        json.dump(res2, f, indent=1)


SCOPES = {
    "vision_2d": ("backbone",),
    "vision_3d": ("backbone_3d", "neck_3d"),
    "text_bert": ("text_encoder",),
    "fusion": ("feature_enhancer", "spatial_enhancer", "text_feat_map"),
    "action_head": ("decoder",),
    "neck_convs": ("neck",),
}


def _in_scope(name, prefixes):
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def cmd_anchor(modes):
    torch.backends.cudnn.deterministic = True
    processor, model, _ = load_model()
    B, R = batches_and_refs(processor, model)
    patch_all(model)
    results = {}
    for mode in modes:
        for sc, prefixes in SCOPES.items():
            keys = [k for k in TAB if k not in EXEMPT and _in_scope(k, prefixes)]
            for k in keys:
                CFG[k] = mode
            t0 = time.perf_counter()
            r = _eval_all(model, B, R, f"{mode}@{sc}")
            r["n_layers"] = len(keys)
            r["seconds"] = time.perf_counter() - t0
            results[f"{mode}@{sc}"] = r
            for k in keys:
                CFG.pop(k, None)
            CACHE_W4.clear()
            with open(os.path.join(WORK, "anchors.json"), "w") as f:
                json.dump(results, f, indent=1)
    print("[done] anchors.json", flush=True)


def cmd_scan(sample, seed, modes):
    torch.backends.cudnn.deterministic = True
    processor, model, _ = load_model()
    B, R = batches_and_refs(processor, model)
    patch_all(model)
    batch = B["real"][sample]
    fp_ref = R["refs"][f"real:{sample}:{seed}"]

    # W8-everything run: policy reference + per-module output cache
    w8_act = forward_actions(model, batch, seed)
    base_m = act_metrics(fp_ref, w8_act)
    print(f"[base] W8all sample{sample} seed{seed}: "
          f"jpos={base_m['mae_jointpos']:.5f} mae_all={base_m['mae_all']:.5f}",
          flush=True)
    cache = {}
    budget = [0.0]
    mods = dict(model.named_modules())
    handles = []

    def mk(name):
        def h(mod, inp, out):
            if torch.is_tensor(out) and torch.is_floating_point(out):
                t = out.detach().float().cpu()
                budget[0] += t.numel() * 4
                if budget[0] < 12 * 2**30:
                    cache.setdefault(name, []).append(t)
                elif name not in STATS["cache_skipped"]:
                    STATS["cache_skipped"].append(name)
        return h

    for key in list(LINEAR_KEYS) + list(CONV_KEYS) + list(CONV1D_KEYS):
        handles.append(mods[key].register_forward_hook(mk(key)))
    forward_actions(model, batch, seed)
    for h in handles:
        h.remove()
    print(f"[base] w8 output cache {len(cache)} modules "
          f"({budget[0] / 2**30:.1f} GiB CPU)", flush=True)

    out_path = os.path.join(WORK, f"scan_s{sample}.json")
    results = {}
    if os.path.exists(out_path):
        results = json.load(open(out_path))
        print(f"[scan] resume: {len(results)} entries already done",
              flush=True)

    scan_keys = sorted(set(TAB) - EXEMPT)
    t0 = time.perf_counter()
    n_done = 0
    for key in scan_keys:
        is_ip = key.endswith("in_proj_weight")
        for mode in modes:
            rk = f"{key}|{mode}"
            if rk in results:
                continue
            CFG[key] = mode
            outs = []
            handles = []
            if is_ip:
                mh = key[: -len(".in_proj_weight")]
                if mh in mods:
                    def h(m, i, o, _k=key):
                        t = o[0] if isinstance(o, tuple) else o
                        if torch.is_tensor(t):
                            outs.append(t.detach().float().cpu())
                    handles.append(mods[mh].register_forward_hook(h))
            elif key in mods:
                def h(m, i, o, _k=key):
                    t = o[0] if isinstance(o, tuple) else o
                    if torch.is_tensor(t):
                        outs.append(t.detach().float().cpu())
                handles.append(mods[key].register_forward_hook(h))
            q = forward_actions(model, batch, seed)
            for h in handles:
                h.remove()
            m_fp = act_metrics(fp_ref, q)
            m_dir = act_metrics(w8_act, q)  # quant run vs W8 run directly
            loc = None
            refs_c = cache.get(key)
            if refs_c and outs and torch.is_tensor(outs[0]):
                try:
                    ds, os_ = 0.0, 0.0
                    for a, b in zip(outs, refs_c):
                        if a.shape != b.shape or a.dim() == 0:
                            continue
                        d = (a - b).abs()
                        ds += float(d.sum())
                        os_ += float(b.abs().sum())
                    if os_ > 0:
                        loc = ds / os_
                except Exception:
                    loc = None
            results[rk] = {
                "key": key, "mode": mode,
                "jpos": m_fp["mae_jointpos"], "mae_all": m_fp["mae_all"],
                "max_jpos": m_fp["max_jointpos"],
                "djpos_vs_w8": m_fp["mae_jointpos"] - base_m["mae_jointpos"],
                "dmae_vs_w8": m_fp["mae_all"] - base_m["mae_all"],
                "jpos_direct_vs_w8": m_dir["mae_jointpos"],
                "local_rel_vs_w8": loc,
            }
            CFG.pop(key, None)
            CACHE_W4.pop((key, mode), None)
            n_done += 1
            if n_done % 20 == 0:
                el = time.perf_counter() - t0
                print(f"[scan] {n_done} runs, {el / 60:.1f} min "
                      f"(~{el / n_done:.2f} s/run)", flush=True)
                with open(out_path, "w") as f:
                    json.dump(results, f)
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"[done] {len(results)} scan entries -> {out_path} | "
          f"bias_fp_w4 {len(STATS['bias_fp_w4'])}, cache_skipped "
          f"{len(set(STATS['cache_skipped']))}, w4_r_over "
          f"{len(STATS['w4_r_over'])}", flush=True)


def cmd_evalcfg(cfg_path, tag):
    torch.backends.cudnn.deterministic = True
    cfg = json.load(open(cfg_path))
    processor, model, _ = load_model()
    B, R = batches_and_refs(processor, model)
    patch_all(model)
    base_real = [forward_actions(model, B["real"][k], 20260830)
                 for k in range(3)]
    for k, v in cfg.items():
        CFG[k] = v
    n4 = sum(1 for v in CFG.values() if v.startswith("w4"))
    print(f"[cfg ] {tag}: {n4} W4 layers among {len(cfg)} overrides",
          flush=True)
    res = _eval_all(model, B, R, tag)
    pol = []
    for k in range(3):
        torch.manual_seed(20260830)
        with torch.no_grad():
            outs = model(B["real"][k])
        action = processor.post_process(
            outs, B["real"][k]).action.cpu().numpy()
        truth = np.load(
            "/home/nc23/workspace/holobrain/robotwin_subset/samples/"
            f"truth_{k:03d}.npz")["future_actions"]
        pol.append({
            "sample": k,
            "mae_vs_truth_all14": float(np.abs(action - truth).mean()),
            "mae_vs_w8_all14": float(np.abs(
                action - base_real[k].numpy()[..., 0]).mean()),
        })
        print(f"[pol ] sample{k}: vs truth "
              f"{pol[-1]['mae_vs_truth_all14']:.4f} rad | vs W8 "
              f"{pol[-1]['mae_vs_w8_all14']:.5f} rad", flush=True)
    res["policy"] = pol
    res["config"] = cfg
    with open(os.path.join(WORK, f"cfg_{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"[done] cfg_{tag}.json", flush=True)


def cmd_bytes():
    rows = []
    for key, e in TAB.items():
        k_eff, n_eff = e["gemm_k"], e["gemm_n"]
        numel = k_eff * n_eff
        rows.append({
            "key": key, "type": e.get("type", ""),
            "k": k_eff, "n": n_eff, "numel": numel,
            "bytes_w8": numel,
            "bytes_w4pt": math.ceil(numel / 2) + 4,
            "bytes_w4g128": math.ceil(numel / 2)
            + n_eff * math.ceil(k_eff / 128) * 2,
            "bytes_w4g64": math.ceil(numel / 2)
            + n_eff * math.ceil(k_eff / 64) * 2,
            "exempt": key in EXEMPT,
        })
    tot8 = sum(r["bytes_w8"] for r in rows)
    for g in ("w4pt", "w4g128", "w4g64"):
        tot = sum(r[f"bytes_{g}"] for r in rows)
        print(f"[bytes] all-W4 {g}: {tot / 1e6:.1f} MB vs W8 {tot8 / 1e6:.1f}"
              f" MB -> saving {1 - tot / tot8:.1%}", flush=True)
    with open(os.path.join(WORK, "bytes.json"), "w") as f:
        json.dump({"rows": rows, "total_w8": tot8}, f, indent=1)
    print(f"[done] bytes.json ({len(rows)} layers)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "w8base", "anchor", "scan",
                                    "evalcfg", "bytes"])
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--modes", default="w4g128,w4pt")
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--tag", default="cfg")
    args = ap.parse_args()
    t0 = time.perf_counter()
    if args.cmd == "prep":
        cmd_prep()
    elif args.cmd == "w8base":
        cmd_w8base()
    elif args.cmd == "anchor":
        cmd_anchor(args.modes.split(","))
    elif args.cmd == "scan":
        cmd_scan(args.sample, args.seed, args.modes.split(","))
    elif args.cmd == "evalcfg":
        cmd_evalcfg(args.cfg, args.tag)
    elif args.cmd == "bytes":
        cmd_bytes()
    print(f"[exit] {args.cmd} total {time.perf_counter() - t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
