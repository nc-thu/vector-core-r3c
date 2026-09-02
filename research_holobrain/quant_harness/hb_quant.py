# hb_quant.py -- fake-quant PTQ for HoloBrain_v0.0_GD (BIP3D + HoloBrainActionDecoder).
# Adapted from the SwiftVLA line's quant.py (same numerics discipline):
#   Linear: W INT8 symmetric per-output-channel; A INT8 per-token (per-row),
#           fp32 accumulation (simulates INT8 tensor-core GEMM).
#   Conv2d : W INT8 symmetric per-output-channel; A INT8 per-TENSOR
#           (conv has no per-token semantics -- noted), fp32 accumulation.
#   w4a16  : Linear W INT4 per-group (group=128 along in_features, last partial
#           group allowed), A untouched; Conv2d stays fp in this mode.
#
# Modes:
#   fp32 | w8a8 | w8a16 | w4a16   optionally suffixed "@<scope>" to restrict
#   quantization to one subtree (all other Linear/Conv2d stay fp).
#
# HB scope partition (matches module_inventory.json top-level subtrees, and the
# partition is exhaustive: 421 Linear + 10 Conv2d = 100% of inventory):
#   vision_2d  backbone            (Swin RGB tower, 51 Linear + 1 Conv2d)
#   vision_3d  backbone_3d+neck_3d (Swin depth tower, 51 Linear + 5 Conv2d)
#   text_bert  text_encoder        (BERT, 72 Linear)
#   fusion     feature_enhancer + spatial_enhancer + text_feat_map (98 Linear)
#   action_head decoder            (diffusion head, 149 Linear)
#   neck_convs neck                (4 Conv2d, no Linear)
#
# Embeddings / norms / the Swin window-attention internals that are not
# nn.Linear or nn.Conv2d stay unquantized (same convention as SwiftVLA line).

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP = 128

MODULE_SCOPES = {
    "vision_2d": ("backbone",),
    "vision_3d": ("backbone_3d", "neck_3d"),
    "text_bert": ("text_encoder",),
    "fusion": ("feature_enhancer", "spatial_enhancer", "text_feat_map"),
    "action_head": ("decoder",),
    "neck_convs": ("neck",),
}


# ---------------------------------------------------------------- fake-quant ops
def _fq_int8_per_channel(W: torch.Tensor):
    s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
    return torch.round(W / s).clamp(-127, 127) * s


def _fq_int8_per_token(A: torch.Tensor):
    s = A.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 127.0
    return torch.round(A / s).clamp(-127, 127) * s


def _fq_int8_per_tensor(A: torch.Tensor):
    s = A.abs().amax().clamp(min=1e-12) / 127.0
    return torch.round(A / s).clamp(-127, 127) * s


def _fq_int4_per_group(W: torch.Tensor, group: int = GROUP):
    out_f, in_f = W.shape
    pad = (-in_f) % group
    Wp = F.pad(W, (0, pad)) if pad else W
    G = Wp.view(out_f, -1, group)
    s = G.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 7.0
    Gq = torch.round(G / s).clamp(-7, 7) * s
    return Gq.reshape(out_f, -1)[:, :in_f]


# ---------------------------------------------------------------- module patches
def _make_linear_forward(mode: str):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            W = self.weight.float()
            if mode in ("w8a8", "w8a16"):
                Wq = _fq_int8_per_channel(W)
            elif mode == "w4a16":
                Wq = _fq_int4_per_group(W)
            else:
                raise RuntimeError(mode)
            if mode == "w8a8":
                xq = _fq_int8_per_token(x.float())
            else:
                xq = x.float()
            b = self.bias.float() if self.bias is not None else None
            y = F.linear(xq, Wq, b)
        return y.to(out_dtype)

    return forward


def _make_conv2d_forward(mode: str):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            W = self.weight.float()  # [out, in/groups, kh, kw]
            if mode in ("w8a8", "w8a16"):
                flat = W.reshape(W.shape[0], -1)
                Wq = _fq_int8_per_channel(flat).view_as(W)
            else:  # w4a16: convs stay fp (per-group W4 on conv kernels not defined here)
                Wq = W
            xq = _fq_int8_per_tensor(x.float()) if mode == "w8a8" else x.float()
            b = self.bias.float() if self.bias is not None else None
            y = F.conv2d(
                xq, Wq, b, self.stride, self.padding, self.dilation, self.groups
            )
        return y.to(out_dtype)

    return forward


def parse_mode(mode: str):
    if "@" in mode:
        m, scope = mode.split("@", 1)
        return m, scope
    return mode, None


def _in_scope(name: str, prefixes) -> bool:
    """Boundary-safe prefix match: 'backbone' must NOT match 'backbone_3d'."""
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def apply_fake_quant(model, mode: str):
    """Monkey-patch forwards of the selected nn.Linear/nn.Conv2d modules
    in-place. Call AFTER loading the checkpoint.
    Returns (stats, restore). restore() removes all patches."""
    base_mode, scope = parse_mode(mode)
    assert base_mode in ("fp32", "w8a8", "w8a16", "w4a16"), f"bad mode {mode}"
    if scope is not None:
        assert scope in MODULE_SCOPES, f"bad scope {scope}"

    stats = {
        "mode": mode,
        "linear_total": 0,
        "linear_quantized": 0,
        "conv_total": 0,
        "conv_quantized": 0,
        "linear_params_quantized": 0,
        "conv_params_quantized": 0,
    }
    patched = []
    if base_mode == "fp32":
        return stats, lambda: None
    prefixes = MODULE_SCOPES[scope] if scope else ()

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            stats["linear_total"] += 1
            if prefixes and not _in_scope(name, prefixes):
                continue
            mod.forward = types.MethodType(_make_linear_forward(base_mode), mod)
            patched.append(mod)
            stats["linear_quantized"] += 1
            stats["linear_params_quantized"] += mod.weight.numel()
        elif isinstance(mod, nn.Conv2d):
            stats["conv_total"] += 1
            if prefixes and not _in_scope(name, prefixes):
                continue
            if base_mode == "w4a16":
                continue  # w4a16 leaves convs fp
            mod.forward = types.MethodType(_make_conv2d_forward(base_mode), mod)
            patched.append(mod)
            stats["conv_quantized"] += 1
            stats["conv_params_quantized"] += mod.weight.numel()

    def restore():
        for mod in patched:
            # remove the instance-level override, restoring the class forward
            mod.__dict__.pop("forward", None)

    return stats, restore
