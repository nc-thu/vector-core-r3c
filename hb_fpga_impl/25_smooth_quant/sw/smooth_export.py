# smooth_export.py -- 平滑模型的导出 + 重标定（25_smooth_quant，服务器）
# 一次模型加载做四件事（都在修改版 state_dict 上）：
#   1) pcW 权重导出（pcw_export.py 同款循环，读修改后 sd）→ pcw_export_{tag}/
#      + pcw_scales_{tag}.json
#   2) fp 偏置导出（dump_bias.py 同款）→ fp_biases_{tag}.json
#      （平滑不动 Linear bias，应与基线 fp_biases.json 逐位一致——脚本会核对）
#   3) sa 重标定：hw_calib.calibrate 同款合成扰动批 n_cal=8（v2 表当年流程），
#      只取平滑组成员的 in_max/127 作为新 sa。
#   4) 拼表：v2 表 deepcopy，仅成员层的 sa 替换 → hw_calib_table_v2mod_{tag}.json
#      （其余层的 sa 与 v2 应一致——脚本核对非成员层 sa 是否被本次前向改变）
#
# 之后由 mk_pcw_calib.py v3 流程合成整表（--v2 v2mod --scales --biases），
# 增广偏置整数由表整体重算，不手改。
#
# Run:
#   cd ~/workspace/holobrain && CUDA_VISIBLE_DEVICES= $PY /tmp/alg_smooth/smooth_export.py \
#       --sd /tmp/alg_smooth/smoothed_sd_s5.pt --plan /tmp/alg_smooth/probe_out.json \
#       --scope s5 --tag s5 --out-dir /tmp/alg_smooth
import argparse
import json
import os
import sys
import time

HB = os.path.expanduser("~/workspace/holobrain")
for p in (HB, os.path.join(HB, "quant"), "/tmp/pcw_rtn"):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import bringup  # noqa: E402
import hw_calib  # noqa: E402  (/tmp/pcw_rtn/hw_calib.py, CPU 口径)
from gate import force_mha_slow_path  # noqa: E402
hw_calib.HB = HB
from robo_orchard_lab.models.holobrain.processor import (  # noqa: E402
    HoloBrainProcessor,
)
from robo_orchard_lab.models.mixin import ModelMixin  # noqa: E402

PCW_KINDS = {"linear", "mha_in_proj"}
SCOPES = {
    "s5": lambda n: n.startswith("decoder.robot_encoder."),
    "s6": lambda n: True,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sd", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--scope", default="s5")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="/tmp/alg_smooth")
    ap.add_argument("--manifest", default="/tmp/pcw_rtn/manifest.json")
    ap.add_argument("--v2", default="/tmp/ae_hostdrv/hw_calib_table_v2.json")
    ap.add_argument("--base-biases",
                    default="/tmp/pcw_rtn/fp_biases.json")
    ap.add_argument("--n-cal", type=int, default=8)
    a = ap.parse_args()

    t0 = time.perf_counter()
    processor = HoloBrainProcessor.load(bringup.CKPT, bringup.PROCESSOR_JSON)
    model = ModelMixin.load_model(bringup.MODEL_DIR, load_impl="native")
    model = model.float().eval()
    n_mha = force_mha_slow_path(model)
    model.load_state_dict(torch.load(a.sd, map_location="cpu"), strict=True)
    sd = model.state_dict()
    print(f"[load] model up (MHA x{n_mha}) {time.perf_counter()-t0:.0f}s",
          flush=True)

    plan = json.load(open(a.plan, encoding="utf-8"))["plan"]
    keep = [e for e in plan if SCOPES[a.scope](e["norm"])]
    members = sorted({m for e in keep for m in e["members"]})
    print(f"[scope] {a.scope}: {len(keep)} groups, {len(members)} member "
          f"layers", flush=True)

    man = json.load(open(a.manifest, encoding="utf-8"))
    outdir = os.path.join(a.out_dir, f"pcw_export_{a.tag}")
    os.makedirs(outdir, exist_ok=True)

    # ---- 1) pcW 导出（pcw_export.py 同款语义） ----
    scales = {}
    n_exp = 0
    max_ratio, max_ratio_key = 0.0, ""
    w_ratio_note = {}
    for t in man["tensors"]:
        if t.get("kind") not in PCW_KINDS or t.get("dtype") != "int8":
            continue
        key = t["key"]
        if key not in sd:
            continue
        W = sd[key].detach().float()
        dims = tuple(range(1, W.dim()))
        swc = W.abs().amax(dim=dims, keepdim=True).clamp(min=1e-12) / 127.0
        wqi = torch.clamp(torch.round(W / swc), -127.0, 127.0)
        arr = wqi.to(torch.int8).contiguous().numpy()
        assert arr.shape == tuple(t["shape"]), (key, arr.shape, t["shape"])
        arr.tofile(os.path.join(outdir, t["file"]))
        sl = swc.flatten().tolist()
        scales[key] = sl
        r = max(sl) / max(min(sl), 1e-30)
        if r > max_ratio:
            max_ratio, max_ratio_key = r, key
        base_key = key[:-len(".weight")] if key.endswith(".weight") \
            else key[:-len("_weight")] if key.endswith("_weight") else key
        if base_key in members or key in members:
            w_ratio_note[base_key] = r
        n_exp += 1
    with open(os.path.join(a.out_dir, f"pcw_scales_{a.tag}.json"), "w") as f:
        json.dump({"_meta": {"semantics": "pcw_export.py identical, "
                          "smoothed sd", "tag": a.tag},
                   "swc": scales}, f)
    print(f"[exp ] {n_exp} tensors, max channel ratio {max_ratio:.1f}x "
          f"({max_ratio_key}) {time.perf_counter()-t0:.0f}s", flush=True)

    # ---- 2) fp 偏置导出（dump_bias.py 同款键归一） ----
    biases = {}
    for t in man["tensors"]:
        if t.get("dtype") != "int8":
            continue
        key = t["key"]
        bk = key[:-len(".weight")] + ".bias" if key.endswith(".weight") \
            else key[:-len("_weight")] + "_bias" \
            if key.endswith("_weight") else key + ".bias"
        b = sd.get(bk)
        if b is None:
            continue
        biases[key[:-len(".weight")] if key.endswith(".weight")
               else key[:-len("_weight")] if key.endswith("_weight")
               else key] = b.detach().float().flatten().tolist()
    with open(os.path.join(a.out_dir, f"fp_biases_{a.tag}.json"), "w") as f:
        json.dump(biases, f)
    base_b = json.load(open(a.base_biases))
    diff = [k for k in base_b if base_b[k] != biases.get(k)]
    print(f"[bias] {len(biases)} biased tensors; 与基线不同: {diff[:10]} "
          f"(应为空)", flush=True)

    # ---- 3) sa 重标定（v2 当年同款合成扰动流程） ----
    calib = hw_calib.calibrate(model, processor, a.n_cal)
    new_sa = {}
    for m in members:
        c = calib.get(m)
        if c is None or c["calls"] == 0:
            print(f"[warn] {m} 标定未调用", flush=True)
            continue
        new_sa[m] = max(c["in_max"], 1e-12) / 127.0

    # ---- 4) 拼表：v2 + 成员 sa 替换 ----
    v2 = json.load(open(a.v2, encoding="utf-8"))
    v2mod = json.loads(json.dumps(v2))
    sa_changed = {}
    for m, sa in new_sa.items():
        # v2 键 = 模块名（无 .weight）；in_proj 例外带后缀
        for cand in (m, m + ".weight", m + "_weight"):
            if cand in v2mod["gemms"] and "sa" in v2mod["gemms"][cand]:
                old = float(v2mod["gemms"][cand]["sa"])
                v2mod["gemms"][cand]["sa"] = sa
                sa_changed[cand] = {"old": old, "new": sa,
                                    "ratio": sa / old if old else None}
                break
        else:
            print(f"[warn] {m} 不在 v2 表", flush=True)
    out_v2mod = os.path.join(a.out_dir, f"hw_calib_table_v2mod_{a.tag}.json")
    with open(out_v2mod, "w", encoding="utf-8") as f:
        json.dump(v2mod, f, ensure_ascii=False)

    rr = sorted(v["ratio"] for v in sa_changed.values() if v["ratio"])
    summ = {
        "tag": a.tag, "n_groups": len(keep), "n_members": len(members),
        "n_exported": n_exp, "n_sa_changed": len(sa_changed),
        "sa_ratio_min": rr[0] if rr else None,
        "sa_ratio_med": rr[len(rr) // 2] if rr else None,
        "sa_ratio_max": rr[-1] if rr else None,
        "w_channel_ratio_members": w_ratio_note,
        "bias_diff_vs_base": diff,
        "seconds": time.perf_counter() - t0,
    }
    with open(os.path.join(a.out_dir, f"export_summary_{a.tag}.json"),
              "w", encoding="utf-8") as f:
        json.dump({"summary": summ, "sa_changed": sa_changed}, f, indent=1)
    print(f"[done] sa changed {len(sa_changed)} layers, ratio "
          f"[{summ['sa_ratio_min']:.3f},{summ['sa_ratio_med']:.3f},"
          f"{summ['sa_ratio_max']:.3f}] -> {out_v2mod} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
