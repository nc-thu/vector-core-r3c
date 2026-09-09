# collect.py -- 汇总 25_alg_calib 全部 result json（T5：数字必须从 json 读）
import glob
import json
import os

ROWS = [
    ("baseline pcwv3 (v3 合成表)", "000", "/tmp/pcw_rtn/result_000_pcwv3_vs_fp32.json", "out-sample 对照"),
    ("baseline pcwv3 (v3 合成表)", "001", "/tmp/pcw_rtn/result_001_pcwv3_vs_fp32.json", "out-sample 对照"),
    ("S1 absmax 全网真实", "000", "/tmp/alg_calib/result_000_S1a_vs_fp32.json", "IN-sample（标定=评估同样本）"),
    ("S1 absmax 全网真实", "001", "/tmp/alg_calib/result_001_S1a_vs_fp32.json", "OUT-sample"),
    ("S1 p99.9 全网真实", "000", "/tmp/alg_calib/result_000_S1p_vs_fp32.json", "IN-sample"),
    ("S1 p99.9 全网真实", "001", "/tmp/alg_calib/result_001_S1p_vs_fp32.json", "OUT-sample"),
    ("S2 仅 patch_embed 真实", "000", "/tmp/alg_calib/result_000_S2_vs_fp32.json", "IN-sample"),
    ("S3 仅 #140-250 段真实", "000", "/tmp/alg_calib/result_000_S3_vs_fp32.json", "IN-sample"),
    ("S4 k=0.8", "000", "/tmp/alg_calib/result_000_S4k08_vs_fp32.json", "IN-sample"),
    ("S4 k=0.9", "000", "/tmp/alg_calib/result_000_S4k09_vs_fp32.json", "IN-sample"),
    ("S4 k=1.1", "000", "/tmp/alg_calib/result_000_S4k11_vs_fp32.json", "IN-sample"),
    ("S4 k=1.25", "000", "/tmp/alg_calib/result_000_S4k125_vs_fp32.json", "IN-sample"),
]

print(f"{'experiment':24s} {'s':3s} {'jpos_mae':>9s} {'max':>7s} {'arm12':>8s} {'grip':>7s}  口径")
for name, s, p, tag in ROWS:
    if not os.path.exists(p):
        print(f"{name:24s} {s:3s} {'MISSING':>9s}")
        continue
    d = json.load(open(p))
    print(f"{name:24s} {s:3s} {d['jpos_mae']:9.4f} {d['jpos_max']:7.4f} "
          f"{d.get('jpos_mae_arm12', float('nan')):8.4f} "
          f"{d.get('jpos_mae_gripper', float('nan')):7.4f}  {tag}")
