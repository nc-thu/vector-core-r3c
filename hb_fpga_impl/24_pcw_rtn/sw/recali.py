# recali.py -- 现标部署语义校准表（服务器 CPU，24_pcw_rtn 用）
#
# 背景（2026-09-04 主会话 gate_real 发现）：部署版 v2 表对真实输入的
# sa/so 失配是独立于权重粒度的大误差源（边界深度同语义只换表：
# v2 表 jpos≈0.24 红 / 现标表 ≈0.034 绿）。全深度 pcW+RTN 实验必须用
# 现标表，否则表格误差和粒度收益混在一起无法归因。
#
# 流程与 02_quant/hw_calib.py 完全同款（合成扰动批，n_cal=8；即 v2 表
# 当年的标定流程，只是今天重跑），只是去掉 gate/coverage/diff 附属步骤，
# 只留 calibrate + build_hw_params + 落表：
#   calib = hw_calib.calibrate(model, processor, n_cal)   # 逐层 in/out absmax
#   params, table, st = hw_calib.build_hw_params(model, calib)
#   -> {"_meta": ..., "gemms": table}（与 v2 表同 schema）
#
# 注意：hw_calib.calibrate 的 forward 走 MHA 慢路径（gate_real 同款
# force_mha_slow_path），否则 nn.MultiheadAttention eval 快路径绕过
# out_proj，标定 hooks 抓不到该层。
#
# Run (server, CPU):
#   cd ~/workspace/holobrain
#   CUDA_VISIBLE_DEVICES= ~/.conda/envs/holobrain/bin/python \
#       /tmp/pcw_rtn/sw/recali.py --n-cal 8 --out /tmp/pcw_rtn/hw_calib_table_fresh.json
import argparse
import json
import os
import sys
import time

PCW = "/tmp/pcw_rtn"
HB = os.path.expanduser("~/workspace/holobrain")
for p in (PCW, os.path.join(HB), os.path.join(HB, "quant")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

import bringup  # noqa: E401  (sets shims/repo paths)
import hw_calib  # noqa: E402  (/tmp/pcw_rtn/hw_calib.py, CPU 口径)
from gate import force_mha_slow_path  # noqa: E402  (gate_real 同款)

# hw_calib.HB 是未展开的 "~/workspace/holobrain" 字面量，build_hw_params
# 读 module_inventory.json 会炸（gate_real.py 同款修复）
hw_calib.HB = HB
from robo_orchard_lab.models.holobrain.processor import HoloBrainProcessor  # noqa: E402
from robo_orchard_lab.models.mixin import ModelMixin  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cal", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.perf_counter()
    processor = HoloBrainProcessor.load(bringup.CKPT, bringup.PROCESSOR_JSON)
    model = ModelMixin.load_model(bringup.MODEL_DIR, load_impl="native")
    model = model.float().eval()
    n_mha = force_mha_slow_path(model)
    print(f"[load] model up on CPU (eager MHA x{n_mha}) "
          f"{time.perf_counter()-t0:.0f}s", flush=True)

    calib = hw_calib.calibrate(model, processor, a.n_cal)
    params, table, st = hw_calib.build_hw_params(model, calib)
    print(f"[cal ] n_cal={a.n_cal} linear {st['n_linear_quant']}/"
          f"{st['n_linear']} conv {st['n_conv_quant']}/{st['n_conv']} "
          f"bias_fp {len(st['bias_fp_layers'])} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)

    meta = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "flow": "hw_calib.calibrate + build_hw_params (synthetic perturbed, "
                "CPU, eager MHA) — same flow as v2, re-run 2026-09-04 for "
                "24_pcw_rtn full-depth attribution",
        "n_cal": a.n_cal,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"_meta": meta, "gemms": table}, f, ensure_ascii=False,
                  indent=1)
    print(f"[done] {a.out} ({time.perf_counter()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
