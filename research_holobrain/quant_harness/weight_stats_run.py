import json
import torch
from safetensors import safe_open

CKPT = "~/workspace/holobrain/ckpt/HoloBrain_v0.0_GD/post_training_robotwin/model.safetensors"
INV = "~/workspace/holobrain/module_inventory.json"
OUT = "~/workspace/holobrain/weight_stats.json"

G4 = 128
Q8, Q4 = 127.0, 7.0


def group_of(name):
    top = name.split(".")[0]
    if top == "text_encoder":
        return "bert"
    if top == "backbone":
        return "vision_2d"
    if top == "neck":
        return "neck"
    if top in ("feature_enhancer", "spatial_enhancer", "text_feat_map"):
        return "fusion"
    if top == "decoder":
        return "action_head"
    return "other"  # backbone_3d, neck_3d


def stats_2d(w):
    """w: [O, K] float32, K = flattened input dim."""
    O, K = w.shape
    denom = torch.linalg.vector_norm(w)
    # W8 per-out-channel symmetric
    s8 = w.abs().amax(dim=1, keepdim=True) / Q8
    s8 = torch.where(s8 > 0, s8, torch.ones_like(s8))
    q8 = torch.round(w / s8).clamp(-Q8, Q8) * s8
    e8 = (torch.linalg.vector_norm(q8 - w) / denom).item()
    # W4 per-group(128, along input dim) symmetric
    G = (K + G4 - 1) // G4
    pad = G * G4 - K
    wp = torch.nn.functional.pad(w, (0, pad))
    wg = wp.view(O, G, G4)
    s4 = wg.abs().amax(dim=2, keepdim=True) / Q4
    s4 = torch.where(s4 > 0, s4, torch.ones_like(s4))
    q4 = torch.round(wg / s4).clamp(-Q4, Q4) * s4
    e4 = (torch.linalg.vector_norm((q4 - wg).reshape(O, G * G4)[:, :K]) / denom).item()
    # outlier flag over whole tensor
    a = w.abs().flatten()
    mx = a.max().item()
    p999 = torch.quantile(a, 0.999).item()
    ratio = mx / p999 if p999 > 0 else float("inf")
    return e8, e4, ratio, mx, p999


inv = json.load(open(INV))
mods = [(m["name"], m["type"]) for cls in ("nn.Linear", "nn.Conv2d") for m in inv[cls]]

results, missing = [], []
with safe_open(CKPT, framework="pt") as f:
    keys = set(f.keys())
    for name, mtype in mods:
        key = name + ".weight"
        if key not in keys:
            missing.append(name)
            continue
        w = f.get_tensor(key).float()
        if w.dim() == 4:
            O, Ic, kh, kw = w.shape
            w2 = w.reshape(O, Ic * kh * kw)
        elif w.dim() == 2:
            w2 = w
        else:
            missing.append(name + " [dim=%d]" % w.dim())
            continue
        e8, e4, ratio, mx, p999 = stats_2d(w2)
        results.append({
            "name": name, "type": mtype, "group": group_of(name),
            "shape": list(w.shape), "params": int(w.numel()),
            "err_w8": round(e8, 6), "err_w4": round(e4, 6),
            "outlier_ratio": round(ratio, 3), "max_abs": mx, "p999": p999,
        })

groups = {}
for r in results:
    g = groups.setdefault(r["group"], {"n": 0, "params": 0, "err_w8": [], "err_w4": [], "outlier_ratio": []})
    g["n"] += 1
    g["params"] += r["params"]
    g["err_w8"].append(r["err_w8"])
    g["err_w4"].append(r["err_w4"])
    g["outlier_ratio"].append(r["outlier_ratio"])


def agg(v):
    return {"mean": round(sum(v) / len(v), 6), "max": round(max(v), 6), "min": round(min(v), 6)}


groups_out = {}
for k, g in groups.items():
    groups_out[k] = {"n": g["n"], "params": g["params"],
                     "err_w8": agg(g["err_w8"]), "err_w4": agg(g["err_w4"]),
                     "outlier_ratio": agg(g["outlier_ratio"])}

worst_w8 = sorted(results, key=lambda r: -r["err_w8"])[:10]
worst_w4 = sorted(results, key=lambda r: -r["err_w4"])[:10]

out = {
    "meta": {
        "ckpt": CKPT,
        "n_tensors": len(results), "n_missing": len(missing),
        "defs": {
            "err_w8": "per-out-channel symmetric INT8 (qmax=127, scale=max|w| per out-channel), rel Frobenius err",
            "err_w4": "per-group(128 along input dim) symmetric INT4 (qmax=7, scale=max|w| per group), rel Frobenius err",
            "outlier_ratio": "max|w| / p99.9(|w|) over whole tensor (bigger = harder)",
        },
        "group_map": "text_encoder->bert; backbone->vision_2d; neck->neck; feature_enhancer+spatial_enhancer+text_feat_map->fusion; decoder->action_head; backbone_3d+neck_3d->other",
        "missing": missing,
    },
    "groups": groups_out,
    "worst_w8": [{k: r[k] for k in ("name", "group", "type", "err_w8", "err_w4", "outlier_ratio", "params")} for r in worst_w8],
    "worst_w4": [{k: r[k] for k in ("name", "group", "type", "err_w8", "err_w4", "outlier_ratio", "params")} for r in worst_w4],
    "tensors": results,
}
json.dump(out, open(OUT, "w"), indent=1)

print("tensors:", len(results), "missing:", len(missing), missing[:10])
for k in sorted(groups_out):
    g = groups_out[k]
    print("%-11s n=%3d params=%9d  w8 mean/max=%.4f/%.4f  w4 mean/max=%.4f/%.4f  outl mean/max=%.2f/%.2f"
          % (k, g["n"], g["params"], g["err_w8"]["mean"], g["err_w8"]["max"],
             g["err_w4"]["mean"], g["err_w4"]["max"],
             g["outlier_ratio"]["mean"], g["outlier_ratio"]["max"]))
print("\nworst W8:")
for r in worst_w8:
    print("  w8=%.4f w4=%.4f outl=%.1f  %s [%s]" % (r["err_w8"], r["err_w4"], r["outlier_ratio"], r["name"], r["group"]))
print("\nworst W4:")
for r in worst_w4:
    print("  w8=%.4f w4=%.4f outl=%.1f  %s [%s]" % (r["err_w8"], r["err_w4"], r["outlier_ratio"], r["name"], r["group"]))
