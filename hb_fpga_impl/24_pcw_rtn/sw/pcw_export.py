# pcw_export.py -- 逐通道（per-output-channel）INT8 权重导出（服务器跑）
#
# 语义权威 = research_w8a8_error/analysis_02_server_ab.py 的 get_w()（V2c_pcW/
# V2_rn_pcW 变体）：
#   swc_j  = absmax(W[j, :]) / 127          （j = 输出通道 = GEMM 列方向）
#   wq_ij  = clamp(round(W[i,j] / swc_j), -127, 127)   （torch.round = 四舍六入五成双）
# 本脚本从 ckpt 的 fp32 权重重算一份逐通道 INT8 权重（w8_export 是逐 tensor
# 的，信息已丢，不能由它反推），写到 pcw_export/（文件名与 03_compiler/
# manifest.json 一致，只有 kind ∈ {linear, mha_in_proj} 的权重），并把每层
# swc 数组写进 pcw_scales.json 供 mk_pcw_calib.py 合并校准表。
#
# conv（conv1d/conv2d）权重不在导出范围：语义权威 V2_rn_pcW 里 conv 保持
# 逐 tensor 权重（A/B 实验只对 Linear 做了逐通道），继续用原 w8_export 文件。
#
# Run (server):
#   cd ~/workspace/holobrain
#   CUDA_VISIBLE_DEVICES= ~/.conda/envs/holobrain/bin/python \
#       /tmp/pcw_rtn/sw/pcw_export.py \
#       --manifest /tmp/pcw_rtn/manifest.json --out /tmp/pcw_rtn
import argparse
import json
import os
import sys
import time

HB = os.path.expanduser("~/workspace/holobrain")
sys.path.insert(0, HB)
sys.path.insert(0, os.path.join(HB, "quant"))

import numpy as np   # noqa: E402
import torch         # noqa: E402

import bringup  # noqa: E401  (sets shims/repo paths)
from robo_orchard_lab.models.holobrain.processor import HoloBrainProcessor  # noqa: E402
from robo_orchard_lab.models.mixin import ModelMixin  # noqa: E402

PCW_KINDS = {"linear", "mha_in_proj"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="03_compiler/manifest.json")
    ap.add_argument("--out", required=True, help="输出目录（pcw_export/ 与 pcw_scales.json 的父目录）")
    a = ap.parse_args()

    t0 = time.perf_counter()
    processor = HoloBrainProcessor.load(bringup.CKPT, bringup.PROCESSOR_JSON)
    model = ModelMixin.load_model(bringup.MODEL_DIR, load_impl="native")
    model = model.float().eval()
    sd = model.state_dict()
    print(f"[load] model up {time.perf_counter()-t0:.0f}s", flush=True)

    man = json.load(open(a.manifest, encoding="utf-8"))
    outdir = os.path.join(a.out, "pcw_export")
    os.makedirs(outdir, exist_ok=True)

    scales = {}
    stats = {"n": 0, "max_ratio": 0.0, "max_ratio_key": "", "missing": [],
             "non127": []}
    for t in man["tensors"]:
        if t.get("kind") not in PCW_KINDS or t.get("dtype") != "int8":
            continue
        key = t["key"]
        if key not in sd:
            stats["missing"].append(key)
            continue
        W = sd[key].detach().float()
        n = W.shape[0]
        # 权威 get_w()：dim=1 是 [n,k] 的 k 维；对 [n,k] 之外的形状（理论上
        # linear/mha_in_proj 都是 2D）沿除第 0 维外的全部维取 absmax
        dims = tuple(range(1, W.dim()))
        swc = W.abs().amax(dim=dims, keepdim=True).clamp(min=1e-12) / 127.0
        wqi = torch.clamp(torch.round(W / swc), -127.0, 127.0)
        arr = wqi.to(torch.int8).contiguous().numpy()
        assert arr.shape == tuple(t["shape"]), (key, arr.shape, t["shape"])
        arr.tofile(os.path.join(outdir, t["file"]))
        sl = swc.flatten().tolist()
        scales[key] = sl
        r = max(sl) / max(min(sl), 1e-30)
        if r > stats["max_ratio"]:
            stats["max_ratio"], stats["max_ratio_key"] = r, key
        if int(np.abs(arr.astype(np.int64)).max()) < 127:
            stats["non127"].append(key)
        stats["n"] += 1
        if stats["n"] % 50 == 0:
            print(f"[exp ] {stats['n']} tensors {time.perf_counter()-t0:.0f}s",
                  flush=True)

    meta = {
        "semantics": "swc_j=absmax(W[j,:])/127; wq=clamp(round(W/swc),-127,127)"
                     " (torch.round, = ab_quant.py get_w w_mode='pc')",
        "kinds": sorted(PCW_KINDS),
        "n_tensors": stats["n"],
        "max_channel_ratio": stats["max_ratio"],
        "max_channel_ratio_key": stats["max_ratio_key"],
        "missing_keys": stats["missing"],
        "no_full_scale_tensors": stats["non127"][:20],
    }
    with open(os.path.join(a.out, "pcw_scales.json"), "w") as f:
        json.dump({"_meta": meta, "swc": scales}, f)
    print(f"[done] {stats['n']} tensors, max channel ratio "
          f"{stats['max_ratio']:.1f}x ({stats['max_ratio_key']}), "
          f"missing={len(stats['missing'])}, total "
          f"{time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
