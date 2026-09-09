"""trace_model.py -- HB-GD (BIP3D) execution-order operator trace + INT8 weight
export, server side of hb_fpga_impl/03_compiler.

What it does (one run, ~5 min, needs one idle GPU):
  1. loads HoloBrain_v0.0_GD exactly like bringup.py (native impl, fp32, eval)
  2. forces nn.MultiheadAttention onto the eager path (gate.py's _mha_eager,
     verbatim) so hooks see real module calls
  3. registers forward hooks on every compute-bearing module:
     nn.Linear / Conv2d / Conv1d / MultiheadAttention, norms (LayerNorm,
     RMSNorm, AdaRMSNorm, GroupNorm), activations (GELU/SiLU/ReLU/Softmax),
     Embedding, and all custom blocks (Swin WindowMSA/SwinBlock, MSDeform
     attention, RotaryAttention / TemporalJointGraphAttention /
     JointGraphAttention, BertLayer, PSE DepthFusionSpatialEnhancer, decoder
     head, top-level subtrees as phase markers)
  4. runs ONE full forward from out/fixture.pt's model_input_batch with the
     bringup seed (10 DPM denoise steps, 6 enhancer layers, 12 BERT layers,
     PSE, decoder), records the hook firing order (post-order = temporal
     completion order) with tensor shapes and id() refs for dataflow edges
  5. sanity: traced forward's pred_actions must match the fixture's stored
     output bit-for-bit-ish (same seed) -- proves the trace run is the real
     inference, not a perturbed one
  6. exports every Linear/Conv2d weight as per-tensor symmetric INT8
     (sw = max|w|/127, w8 = round(w/sw) clamp [-127,127], row-major int8
     bytes) + all matching biases as fp32 bytes -> w8_export/ + manifest.json

Run:
  cd ~/workspace/holobrain
  CUDA_VISIBLE_DEVICES=1 ~/.conda/envs/holobrain/bin/python trace_model.py
"""

import collections
import json
import os
import sys
import time

HB = "~/workspace/holobrain"
OUT_DIR = os.path.join(HB, "trace_out")
W8_DIR = os.path.join(OUT_DIR, "w8_export")
FIXTURE = os.path.join(HB, "out", "fixture.pt")

sys.path.insert(0, os.path.join(HB, "shims"))
sys.path.insert(0, os.path.join(HB, "robo_orchard_lab"))
sys.path.insert(0, HB)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import bringup  # noqa: E402  (path constants + SEED only)
from robo_orchard_lab.models.mixin import ModelMixin  # noqa: E402

SEED = bringup.SEED  # 20260830, same as bringup fixture run


# ---------------------------------------------------------------- MHA eager
# verbatim from quant/gate.py: torch's eval fast path routes MHA through a
# fused kernel, hooks on inner modules never fire. This reimplementation
# calls self.out_proj as a module; in_proj is a raw Parameter.
def _mha_eager(self, query, key, value, key_padding_mask=None,
               need_weights=True, attn_mask=None, **kw):
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
    out = self.out_proj(out)  # module call -> hooks fire
    return out, None


def force_mha_slow_path(model):
    import types
    n = 0
    for _, mod in model.named_modules():
        if isinstance(mod, torch.nn.MultiheadAttention):
            mod.forward = types.MethodType(_mha_eager, mod)
            n += 1
    return n


# ---------------------------------------------------------------- taxonomy
ATTN_CLS = {
    "WindowMSA", "ShiftWindowMSA", "RotaryAttention",
    "TemporalJointGraphAttention", "JointGraphAttention",
    "BiMultiHeadAttention", "BertAttention",
    "MultiScaleDeformableAttention",
}
ELEM_NORM_CLS = {
    "LayerNorm", "RMSNorm", "AdaRMSNorm", "GroupNorm",
    "GELU", "GELUActivation", "SiLU", "ReLU", "Softmax",
}
CUSTOM_CLS = {
    "SwinBlock", "PatchEmbed", "PatchMerging", "Unfold", "AdaptivePadding",
    "ScalarEmbedder", "RotaryEmbedding", "BertLayer",
    "DeformableDetrTransformerEncoderLayer", "DetrTransformerEncoderLayer",
    "SingleScaleBiAttentionBlock", "DepthFusionSpatialEnhancer", "MLP",
    "SinePositionalEncoding", "Upsample", "UpsampleHead",
    "HoloBrainActionDecoder", "HoloBrainRobotStateEncoder",
    "TextImageDeformable2DEnhancer", "ChannelMapper",
    "BatchDepthProbGTGenerator", "Embedding",
}


def classify(mod):
    if isinstance(mod, nn.Linear):        # incl. NonDynamicallyQuantizableLinear
        return "gemm"
    if isinstance(mod, (nn.Conv2d, nn.Conv1d)):
        return "gemm"
    if isinstance(mod, nn.MultiheadAttention):
        return "attn"
    cn = type(mod).__name__
    if cn in ATTN_CLS:
        return "attn"
    if cn in ELEM_NORM_CLS:
        return "elem_norm"
    if cn in CUSTOM_CLS:
        return "custom"
    return None  # not hooked (Sequential/ModuleList/Dropout/FFN/Identity/...)


# ---------------------------------------------------------------- hooks
def describe(x, depth=0):
    """shape list for tensors, type string otherwise (bounded recursion)."""
    try:
        if torch.is_tensor(x):
            return list(x.shape)
        if depth >= 2:
            return type(x).__name__
        if isinstance(x, (tuple, list)):
            return [describe(a, depth + 1) for a in x]
        if isinstance(x, dict):
            return {str(k): describe(v, depth + 1) for k, v in x.items()}
        if isinstance(x, (int, float, bool)):
            return x
        return type(x).__name__
    except Exception:
        return "<?>"

def collect_tensors(x, out, depth=0):
    if torch.is_tensor(x):
        out.append(x)
    elif depth < 3 and isinstance(x, (tuple, list)):
        for a in x:
            collect_tensors(a, out, depth + 1)
    elif depth < 3 and isinstance(x, dict):
        for v in x.values():
            collect_tensors(v, out, depth + 1)


def make_hook(name, cls_name, op, sd_keys):
    def hook(mod, args, kwargs, output):
        ins = []
        for a in args:
            collect_tensors(a, ins)
        for v in kwargs.values():
            collect_tensors(v, ins)
        outs = []
        collect_tensors(output, outs)
        rec = {
            "seq": len(RECORDS),
            "module": name,
            "cls": cls_name,
            "op": op,
            "in_shapes": [describe(a) for a in args],
            "in_ids": [id(t) for t in ins],
            "out_shapes": [list(t.shape) for t in outs],
            "out_ids": [id(t) for t in outs],
        }
        if kwargs:
            rec["kwargs_shapes"] = {k: describe(v) for k, v in kwargs.items()}
        if op == "gemm":
            wk = name + ".weight"
            rec["weight_key"] = wk if wk in sd_keys else None
            rec["w_shape"] = list(mod.weight.shape)
            rec["has_bias"] = mod.bias is not None
        RECORDS.append(rec)
    return hook


RECORDS = []


# ---------------------------------------------------------------- w8 export
def sanitize(key):
    return key.replace(".", "__")


def export_int8(key, tensor, kind, manifest, extra=False):
    t = tensor.detach().float().cpu().contiguous()
    max_abs = t.abs().max().item()
    sw = max_abs / 127.0 if max_abs > 0 else 1.0
    q = torch.round(t / sw).clamp_(-127, 127).to(torch.int8)
    # dequant sanity: error must stay within half a step
    err = (t - q.float() * sw).abs().max().item()
    assert err <= 0.5 * sw + 1e-6, f"{key}: dequant err {err} > sw/2 {0.5*sw}"
    fn = sanitize(key) + ".bin"
    with open(os.path.join(W8_DIR, fn), "wb") as f:
        f.write(q.numpy().tobytes())  # row-major (C order)
    manifest["tensors"].append({
        "key": key, "file": fn, "shape": list(t.shape),
        "dtype": "int8", "sw": sw, "max_abs": max_abs,
        "kind": kind, "extra": extra, "numel": t.numel(),
        "bytes": q.numel(),
    })
    return q.numel()


def export_fp32(key, tensor, kind, manifest, extra=False):
    t = tensor.detach().float().cpu().contiguous()
    fn = sanitize(key) + ".bin"
    with open(os.path.join(W8_DIR, fn), "wb") as f:
        f.write(t.numpy().tobytes())
    manifest["tensors"].append({
        "key": key, "file": fn, "shape": list(t.shape),
        "dtype": "float32", "sw": None, "max_abs": t.abs().max().item(),
        "kind": kind, "extra": extra, "numel": t.numel(),
        "bytes": t.numel() * 4,
    })
    return t.numel() * 4


# ---------------------------------------------------------------- main
def main():
    os.makedirs(W8_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    t0 = time.perf_counter()
    model = ModelMixin.load_model(bringup.MODEL_DIR, load_impl="native")
    model = model.cuda().float().eval()
    sd_keys = set(model.state_dict().keys())
    print(f"[load] model on {next(model.parameters()).device}, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
          f"{time.perf_counter()-t0:.1f}s", flush=True)

    n_mha = force_mha_slow_path(model)
    print(f"[mha ] forced eager on {n_mha} nn.MultiheadAttention", flush=True)

    # rebuild the FULL batch via processor pre_process (bringup scene/seed).
    # fixture.pt's model_input_batch only kept tensor entries; the model also
    # needs non-tensor 'text' (token id lists) and 'kinematics' (FK chain),
    # so we regenerate deterministically and cross-check every tensor key
    # against the fixture to prove we trace the exact fixture input.
    from robo_orchard_lab.models.holobrain.processor import (
        HoloBrainProcessor,
        MultiArmManipulationInput,
    )
    processor = HoloBrainProcessor.load(bringup.CKPT, bringup.PROCESSOR_JSON)
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
    with torch.no_grad():
        batch = processor.pre_process(inp)
    batch = {k: (v.cuda() if torch.is_tensor(v) else v)
             for k, v in batch.items()}
    print(f"[prep] rebuilt batch: "
          + ", ".join(f"{k}({type(v).__name__ if not torch.is_tensor(v) else tuple(v.shape)})"
                      for k, v in batch.items()), flush=True)

    fx = torch.load(FIXTURE, map_location="cpu", weights_only=False)
    max_d = 0.0
    for k, v_ref in fx["model_input_batch"].items():
        v = batch[k]
        assert torch.is_tensor(v) and tuple(v.shape) == tuple(v_ref.shape), k
        if v.dtype == torch.bool:
            assert not bool((v.cpu() ^ v_ref).any()), f"{k} bool mismatch"
        else:
            max_d = max(max_d, (v.cpu().float() - v_ref.float()).abs().max().item())
    print(f"[prep] rebuilt batch vs fixture tensors: max|d|={max_d:.3e} "
          f"({'OK' if max_d == 0 else 'DIFF'})", flush=True)

    # register hooks
    n_hook = 0
    root_hooked = False
    for name, mod in model.named_modules():
        op = classify(mod)
        if op is None:
            continue
        mod.register_forward_hook(make_hook(name, type(mod).__name__, op,
                                            sd_keys), with_kwargs=True)
        n_hook += 1
        if name == "":
            root_hooked = True
    # top-level subtrees as phase markers (backbone/text_encoder/decoder/...)
    for name, child in model.named_children():
        if classify(child) is None:
            child.register_forward_hook(
                make_hook(name, type(child).__name__, "custom", sd_keys),
                with_kwargs=True)
            n_hook += 1
    print(f"[hook] {n_hook} modules hooked (root={root_hooked})", flush=True)

    # ---- one full traced forward ----
    t1 = time.perf_counter()
    with torch.no_grad():
        torch.manual_seed(SEED)  # denoise init noise uses global RNG
        model_outs = model(batch)
        torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t1
    print(f"[fwd] traced forward: {t_fwd*1000:.1f} ms, "
          f"{len(RECORDS)} hook records", flush=True)

    # ---- sanity vs fixture output (same seed -> must match) ----
    pa = model_outs[0]["pred_actions"].detach().cpu()
    ref = fx["model_output"]["pred_actions_denorm"]
    diff = (pa - ref).abs().max().item()
    print(f"[check] pred_actions vs fixture: max|d|={diff:.3e} "
          f"({'OK' if diff < 1e-5 else 'MISMATCH'})", flush=True)

    # ---- summary ----
    op_counts = collections.Counter(r["op"] for r in RECORDS)
    lin_mods = {r["module"] for r in RECORDS
                if r["op"] == "gemm" and r["cls"] in ("Linear",
                                                      "NonDynamicallyQuantizableLinear")}
    conv2d_calls = sum(1 for r in RECORDS if r["cls"] == "Conv2d")
    lin_calls = sum(1 for r in RECORDS if r["op"] == "gemm"
                    and r["cls"] != "Conv2d" and r["cls"] != "Conv1d")
    wkeys = {r["weight_key"] for r in RECORDS
             if r.get("weight_key")}
    n_lin_total = sum(1 for _, m in model.named_modules()
                      if isinstance(m, nn.Linear))
    miss_wk = sum(1 for r in RECORDS
                  if r["op"] == "gemm" and r.get("weight_key") is None)

    summary = {
        "op_counts": dict(op_counts),
        "total_records": len(RECORDS),
        "unique_linear_modules_fired": len(lin_mods),
        "linear_modules_total": n_lin_total,
        "gemm_calls": {"linear": lin_calls, "conv2d": conv2d_calls,
                       "total": op_counts.get("gemm", 0)},
        "unique_linear_weight_keys_fired": len(wkeys),
        "gemm_records_missing_weight_key": miss_wk,
        "mha_forced_eager": n_mha,
        "forward_ms": t_fwd * 1000,
        "fixture_check_max_abs_diff": diff,
    }
    print("[sum] " + json.dumps(summary), flush=True)

    trace = {
        "meta": {
            "model": "HorizonRobotics/HoloBrain_v0.0_GD@post_training_robotwin",
            "input": "out/fixture.pt model_input_batch "
                     "(instruction: 'put the bowl on the plate')",
            "seed": SEED, "dtype": "float32",
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "hook_note": (
                "records are forward-hook firing order = post-order "
                "(children before parents). id() links tensors across "
                "records for dataflow edges; views get new ids and CPython "
                "reuses ids of freed tensors, so treat edges as hints to be "
                "shape-validated. in_shapes covers positional args; modules "
                "called purely by kwargs (e.g. the forced-eager MHA) carry "
                "their inputs under kwargs_shapes. MHA q/k/v are functional "
                "linears inside the MHA record, out_proj fires as its own "
                "Linear. BERT BertSdpaSelfAttention core is SDPA -- captured "
                "by the BertAttention/BertLayer custom records, its q/k/v/out "
                "Linears fire individually."),
        },
        "summary": summary,
        "ops": RECORDS,
    }
    with open(os.path.join(OUT_DIR, "ops_trace.json"), "w") as f:
        json.dump(trace, f)
    print(f"[save] ops_trace.json "
          f"({os.path.getsize(os.path.join(OUT_DIR,'ops_trace.json'))/1e6:.1f} MB)",
          flush=True)

    # ================= w8 export =================
    print("[w8 ] exporting INT8 weights + fp32 biases ...", flush=True)
    sd = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
    manifest = {"tensors": [], "_comment": (
        "per-tensor symmetric INT8: sw=max|w|/127, w8=round(w/sw) clamped "
        "[-127,127], row-major int8 bytes. bias exported as fp32 raw bytes "
        "(for K+1 augmented GEMM). extra=true items are outside the 431 "
        "Linear/Conv2d weights (2x Conv1d, 6x MHA in_proj) exported for "
        "compiler completeness.")}
    n_w = n_b = n_extra = 0
    total_bytes = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            total_bytes += export_int8(name + ".weight", sd[name + ".weight"],
                                       "linear", manifest)
            n_w += 1
            if mod.bias is not None:
                total_bytes += export_fp32(name + ".bias", sd[name + ".bias"],
                                           "linear_bias", manifest)
                n_b += 1
        elif isinstance(mod, nn.Conv2d):
            total_bytes += export_int8(name + ".weight", sd[name + ".weight"],
                                       "conv2d", manifest)
            n_w += 1
            if mod.bias is not None:
                total_bytes += export_fp32(name + ".bias", sd[name + ".bias"],
                                           "conv2d_bias", manifest)
                n_b += 1
        elif isinstance(mod, nn.Conv1d):
            total_bytes += export_int8(name + ".weight", sd[name + ".weight"],
                                       "conv1d", manifest, extra=True)
            n_extra += 1
            if mod.bias is not None:
                total_bytes += export_fp32(name + ".bias", sd[name + ".bias"],
                                           "conv1d_bias", manifest, extra=True)
                n_extra += 1
        elif isinstance(mod, nn.MultiheadAttention):
            total_bytes += export_int8(name + ".in_proj_weight",
                                       sd[name + ".in_proj_weight"],
                                       "mha_in_proj", manifest, extra=True)
            n_extra += 1
            if name + ".in_proj_bias" in sd:
                total_bytes += export_fp32(name + ".in_proj_bias",
                                           sd[name + ".in_proj_bias"],
                                           "mha_in_proj_bias", manifest,
                                           extra=True)
                n_extra += 1

    # integrity checks
    files = [t["file"] for t in manifest["tensors"]]
    assert len(files) == len(set(files)), "filename collision in w8_export"
    core_w = [t for t in manifest["tensors"]
              if t["kind"] in ("linear", "conv2d")]
    assert len(core_w) == 431, f"expected 431 core weights, got {len(core_w)}"
    core_keys = {t["key"] for t in core_w}
    assert core_keys <= sd_keys, "weight key not in state_dict"

    manifest["summary"] = {
        "core_int8_weights": n_w,          # 421 linear + 10 conv2d
        "fp32_biases": n_b,
        "extras": n_extra,                 # conv1d + mha in_proj (+bias)
        "tensor_count": len(manifest["tensors"]),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / 1e6, 1),
    }
    with open(os.path.join(W8_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[w8 ] {n_w} int8 weights + {n_b} fp32 biases + {n_extra} extras, "
          f"{total_bytes/1e6:.1f} MB -> {W8_DIR}", flush=True)
    print("[done]")


if __name__ == "__main__":
    main()
