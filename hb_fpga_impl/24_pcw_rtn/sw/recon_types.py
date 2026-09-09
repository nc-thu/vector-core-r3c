"""Recon: census of module types in HoloBrain_v0.0_GD + fixture batch keys.
CPU-only (no GPU needed). Informs hook selection for trace_model.py.
"""
import collections
import json
import os
import sys

HB = "~/workspace/holobrain"
sys.path.insert(0, HB)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch  # noqa: E402

import bringup  # noqa: E402
from robo_orchard_lab.models.mixin import ModelMixin  # noqa: E402

model = ModelMixin.load_model(bringup.MODEL_DIR, load_impl="native")
model = model.float().eval()

hist = collections.Counter()
examples = {}
for name, mod in model.named_modules():
    cn = type(mod).__name__
    hist[cn] += 1
    examples.setdefault(cn, name)

print("=== module type histogram (count, class, example path) ===")
for cn, c in hist.most_common():
    print(f"{c:5d}  {cn:45s} {examples[cn]}")

n_mha = sum(1 for _, m in model.named_modules()
            if isinstance(m, torch.nn.MultiheadAttention))
print(f"\nnn.MultiheadAttention instances: {n_mha}")
for name, m in model.named_modules():
    if isinstance(m, torch.nn.MultiheadAttention):
        print(f"  MHA {name} embed_dim={m.embed_dim} heads={m.num_heads} "
              f"batch_first={m.batch_first}")

sd = model.state_dict()
n_lin_w = sum(1 for k in sd if k.endswith(".weight") and
              sd[k].dim() == 2 and "in_proj" not in k)
print(f"\nstate_dict tensors: {len(sd)}, total elems "
      f"{sum(v.numel() for v in sd.values())/1e6:.1f}M")
lin_bias = [k for k in sd if k.endswith(".bias")]
print(f"bias tensors in state_dict: {len(lin_bias)}")

fx = torch.load(os.path.join(HB, "out", "fixture.pt"),
                map_location="cpu", weights_only=False)
print("\n=== fixture model_input_batch ===")
for k, v in fx["model_input_batch"].items():
    if torch.is_tensor(v):
        print(f"  {k:20s} {str(tuple(v.shape)):28s} {v.dtype}")
    else:
        print(f"  {k:20s} {type(v).__name__}")
print("\nfixture meta:", json.dumps(fx["meta"], default=str)[:800])
