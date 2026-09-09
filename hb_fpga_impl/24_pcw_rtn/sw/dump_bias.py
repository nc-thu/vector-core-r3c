# dump_bias.py -- 导出量化层的 fp 偏置（服务器，24_pcw_rtn 逐通道 aug 用）
#
# 背景：pcW 层的 K+1 增广偏置单位是「累加器域、逐通道」：
#   w_bias_j = round(b_j / (sa * swc_j * c))，c 为每层统一的 2 的幂（<=64）
# 逐 tensor 时代 swc 换成 sw。本脚本把 state_dict 里所有带 .bias 的
# 量化层偏置导成 json，供 mk_pcw_calib.py 重算逐通道增广。
#
# Run (server, CPU):
#   cd ~/workspace/holobrain
#   CUDA_VISIBLE_DEVICES= ~/.conda/envs/holobrain/bin/python \
#       /tmp/pcw_rtn/sw/dump_bias.py --manifest /tmp/pcw_rtn/manifest.json \
#       --out /tmp/pcw_rtn/fp_biases.json
import argparse
import json
import os
import sys
import time

HB = os.path.expanduser("~/workspace/holobrain")
for p in (HB, os.path.join(HB, "quant")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

import bringup  # noqa: E401
from robo_orchard_lab.models.holobrain.processor import HoloBrainProcessor  # noqa: E402
from robo_orchard_lab.models.mixin import ModelMixin  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.perf_counter()
    processor = HoloBrainProcessor.load(bringup.CKPT, bringup.PROCESSOR_JSON)
    model = ModelMixin.load_model(bringup.MODEL_DIR, load_impl="native")
    model = model.float().eval()
    sd = model.state_dict()
    print(f"[load] model up {time.perf_counter()-t0:.0f}s", flush=True)

    man = json.load(open(a.manifest, encoding="utf-8"))
    biases = {}
    n = 0
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
        n += 1
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(biases, f)
    print(f"[done] {n} biased tensors -> {a.out} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
