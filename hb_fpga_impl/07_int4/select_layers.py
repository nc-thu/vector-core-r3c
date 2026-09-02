# select_layers.py -- local (Windows) analysis of the int4 scan results.
# Reads scan_s0.json + bytes.json from the server dump dir, ranks layers by
# W4 marginal cost vs byte benefit, and writes evalcfg config JSONs.
#
# Usage: python select_layers.py <dump_dir>
import csv
import json
import os
import sys
from collections import defaultdict

DUMP = sys.argv[1] if len(sys.argv) > 1 else "."
EXEMPT = {"spatial_enhancer.pts_prob_fc.layers.1"}

scan = json.load(open(os.path.join(DUMP, "scan_s0.json")))
byt = json.load(open(os.path.join(DUMP, "bytes.json")))
rows = {r["key"]: r for r in byt["rows"]}
TOTAL_W8 = byt["total_w8"]

BASE = json.load(open(os.path.join(DUMP, "w8base.json")))
base_syn = BASE["syn"]["mean"]["mae_jointpos"]
base_real = [r["mean"]["mae_jointpos"] for r in BASE["real"]]

# ---------------------------------------------------------------- module map
def scope_of(key):
    for sc, prefs in (("vision_2d", ("backbone.",)),
                      ("vision_3d", ("backbone_3d.", "neck_3d.")),
                      ("text_bert", ("text_encoder.",)),
                      ("fusion", ("feature_enhancer.", "spatial_enhancer.",
                                  "text_feat_map.")),
                      ("action_head", ("decoder.",)),
                      ("neck_convs", ("neck.",))):
        if key.startswith(prefs):
            return sc
    return "?"


# ---------------------------------------------------------------- table
per_mode = defaultdict(list)
for rk, r in scan.items():
    b = rows[r["key"]]
    per_mode[r["mode"]].append({
        "key": r["key"],
        "scope": scope_of(r["key"]),
        "numel": b["numel"],
        "bytes_w8": b["bytes_w8"],
        "save_g128": b["bytes_w8"] - b["bytes_w4g128"],
        "save_pt": b["bytes_w8"] - b["bytes_w4pt"],
        "djpos": r["djpos_vs_w8"],
        "dmae": r["dmae_vs_w8"],
        "jpos_direct": r["jpos_direct_vs_w8"],
        "local_rel": r["local_rel_vs_w8"] if r["local_rel_vs_w8"] is not None
        else -1.0,
        "jpos": r["jpos"],
    })

for mode, lst in per_mode.items():
    lst.sort(key=lambda x: -abs(x["djpos"]))

with open(os.path.join(DUMP, "scan_table.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["mode", "key", "scope", "numel", "bytes_w8", "save_g128",
                "save_pt", "djpos_vs_w8", "jpos_direct_vs_w8", "local_rel",
                "jpos_vs_fp"])
    for mode, lst in sorted(per_mode.items()):
        for x in lst:
            w.writerow([mode, x["key"], x["scope"], x["numel"], x["bytes_w8"],
                        x["save_g128"], x["save_pt"], f"{x['djpos']:.5f}",
                        f"{x['jpos_direct']:.5f}", f"{x['local_rel']:.5f}",
                        f"{x['jpos']:.5f}"])
print(f"[out ] scan_table.csv ({sum(len(v) for v in per_mode.values())} rows)")

# ---------------------------------------------------------------- module rollup
print("\n=== per-module rollup (median/mean djpos over layers, g128 vs pt) ===")
mod_tab = defaultdict(lambda: {"g128": [], "pt": []})
for mode, lst in per_mode.items():
    tag = "g128" if mode == "w4g128" else "pt"
    for x in lst:
        mod_tab[(x["scope"], x["key"].split(".")[0])][tag].append(x)
roll = []
for (sc, _), d in mod_tab.items():
    for tag in ("g128", "pt"):
        if d[tag]:
            dj = sorted(x["djpos"] for x in d[tag])
            n = len(dj)
            roll.append({
                "scope": sc, "mode": tag, "n_layers": n,
                "djpos_median": dj[n // 2],
                "djpos_mean": sum(dj) / n,
                "djpos_p90": dj[int(0.9 * (n - 1))],
                "n_djpos_gt_5e-3": sum(1 for v in dj if v > 5e-3),
                "bytes_w8": sum(x["bytes_w8"] for x in d[tag]),
            })
roll.sort(key=lambda r: (r["scope"], r["mode"]))
with open(os.path.join(DUMP, "module_rollup.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(roll[0].keys()))
    w.writeheader()
    w.writerows(roll)
for r in roll:
    print(f"  {r['scope']:11s} {r['mode']:5s} n={r['n_layers']:3d} "
          f"median_djpos={r['djpos_median']:+.5f} p90={r['djpos_p90']:+.5f} "
          f"n>5e-3={r['n_djpos_gt_5e-3']:3d} "
          f"bytes={r['bytes_w8'] / 1e6:6.1f}MB")

# ---------------------------------------------------------------- greedy sets
print("\n=== greedy candidate sets (g128 primary; layer kept if djpos<=thr) ===")
g = per_mode.get("w4g128", [])
pt = per_mode.get("w4pt", [])
pt_by_key = {x["key"]: x for x in pt}


def build_set(thr_g, thr_pt, use_pt_when_ok):
    """Greedy: take g128 layers with djpos<=thr_g; optionally per-tensor for
    layers that are also safe at pt (same bytes saved more... pt saves MORE
    bytes: no group scales). Order by save desc."""
    sel = {}
    est_dj = 0.0
    for x in sorted(g, key=lambda v: -v["save_g128"]):
        if x["key"] in EXEMPT:
            continue
        if x["djpos"] <= thr_g:
            p = pt_by_key.get(x["key"])
            if use_pt_when_ok and p is not None and p["djpos"] <= thr_pt:
                sel[x["key"]] = "w4pt"
                est_dj += p["djpos"]
            else:
                sel[x["key"]] = "w4g128"
                est_dj += x["djpos"]
    return sel, est_dj


configs = {}
for thr in (0.000, 0.002, 0.004):
    for pt_ok in (False, True):
        sel, est = build_set(thr, thr, pt_ok)
        sav_g = sum(rows[k]["bytes_w8"] - rows[k]["bytes_w4g128"]
                    for k in sel)
        sav_pt = sum(rows[k]["bytes_w8"] - rows[k][
            "bytes_w4pt" if m == "w4pt" else "bytes_w4g128"]
            for k, m in sel.items())
        tag = f"thr{int(thr * 1000):03d}{'_pt' if pt_ok else ''}"
        frac = sav_pt / TOTAL_W8
        print(f"  {tag}: {len(sel)} layers, saving {sav_pt / 1e6:.1f}MB "
              f"({frac:.1%}), est sum djpos {est:+.4f}")
        if 0.25 < frac < 0.60:
            configs[tag] = dict(sel)

for tag, cfg in configs.items():
    with open(os.path.join(DUMP, f"config_{tag}.json"), "w") as f:
        json.dump(cfg, f, indent=0)
print(f"[out ] wrote {len(configs)} config jsons: "
      f"{', '.join(sorted(configs))}")
print(f"[base] W8 synth jpos={base_syn:.5f}, real="
      f"{['%.5f' % v for v in base_real]}")
