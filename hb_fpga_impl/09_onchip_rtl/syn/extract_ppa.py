# -*- coding: utf-8 -*-
"""extract_ppa.py — 从综合报告提取 PPA 摘要到 results/PPA.md
用法: python extract_ppa.py [报告目录]   # 默认 syn/out，可指 E:/ae_syn/out"""
import os, re, sys

SYN = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SYN, "out")
RES = os.path.join(SYN, "..", "results")
os.makedirs(RES, exist_ok=True)

def grab(path, pats):
    txt = open(path, encoding="utf-8", errors="ignore").read()
    rows = {}
    for name, pat in pats.items():
        m = re.search(pat, txt)
        rows[name] = m.group(1) if m else "?"
    return rows

util = grab(os.path.join(OUT, "utilization.rpt"), {
    "LUT":     r"\| CLB LUTs\*?\s+\|\s*(\d+)",
    "LUTRAM":  r"\|   LUT as Memory\s+\|\s*(\d+)",
    "FF":      r"\| CLB Registers\s+\|\s*(\d+)",
    "BRAM":    r"\| Block RAM Tile\s+\|\s*(\d+\.?\d*)",
    "URAM":    r"\| URAM\s+\|\s*(\d+)",
    "DSP":     r"\| DSPs\s+\|\s*(\d+)",
})
def grab_timing(path):
    """WNS 表：表头 → 分隔线 → 数值行 —— 向下找第一行含小数的"""
    lines = open(path, encoding="utf-8", errors="ignore").read().splitlines()
    hdr_i = next((i for i, l in enumerate(lines) if "WNS(ns)" in l), None)
    if hdr_i is None:
        return {}
    for l in lines[hdr_i + 1:hdr_i + 4]:
        if re.search(r"\d+\.\d+", l):
            vals = [float(x) for x in re.findall(r"-?\d+\.?\d*", l)]
            # 列序: WNS TNS failN failTotal WHS ...（Vivado 2021.2 时序摘要）
            return {"WNS": vals[0], "TNS": vals[1], "violated": int(vals[2]),
                    "WHS": vals[4]}
    return {}

timing = grab_timing(os.path.join(OUT, "timing.rpt"))

avail = dict(LUT=230400, FF=460800, BRAM=312, URAM=96, DSP=1728)

lines = ["# PPA — Vivado 2021.2 逻辑综合（ae_top，COLS=108，250 MHz 目标）",
         "",
         "器件 xczu7ev-ffvc1156-2-e（ZCU104）；`synth_design -directive RuntimeOptimized`；",
         "综合后（未布局布线）时序。板容量：LUT 230400 / FF 460800 / BRAM36 312 / URAM 96 / DSP 1728。",
         "",
         "| 资源 | 用量 | 占用率 |",
         "|---|---|---|"]
for k in ("LUT", "LUTRAM", "FF", "BRAM", "URAM", "DSP"):
    v = util.get(k, "?")
    av = avail.get(k)
    pct = f"{100*float(v.replace(',',''))/av:.1f}%" if (av and v != "?") else "—"
    lines.append(f"| {k} | {v} | {pct} |")
wns = timing.get("WNS", 0.0)
fmax = 1000.0 / (4.000 - wns) if isinstance(wns, float) else None  # WNS<0 → 周期=4−WNS
lines += ["",
          f"WNS = {timing.get('WNS','?')} ns @ 4.000 ns 时钟（TNS = {timing.get('TNS','?')} ns，"
          f"违例路径 {timing.get('violated','?')} 条，建立）；"
          f"保持 WHS = {timing.get('WHS','?')} ns。",
          f"综合阶段（未布局布线、慢工艺角）保守 Fmax ≈ {fmax:.1f} MHz；"
          "最差路径 = copy 引擎 16→108 lane 重排交叉矩阵（纯 LUT，19 级），"
          "布局布线 + phys_opt 后有望进一步收敛。",
          "",
          "CTX（2 MB，131072×128b SDP）推断为 URAM 级联（cascade height 8）；"
          "WRAM = 108 lane × 4096×8 BRAM；1728 DSP = 16×108 MAC 阵列，"
          "requant 32×16 乘法与 softmax 乘法走 LUT（use_dsp=no）。",
          ""]
open(os.path.join(RES, "PPA.md"), "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
