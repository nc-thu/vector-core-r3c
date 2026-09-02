#!/usr/bin/env bash
# ============================================================================
# run_synth.sh — hw_zcu104 一键综合（交接入口，2026-08-30 新增）
#
# 做什么：拷 RTL 到无空格工作区 E:\ae_syn\<tag>\ → Vivado 2021.2 OOC 综合
#         → extract_ppa.py 提取 PPA 到 hw_zcu104/results/PPA.md（旧表自动备份为 PPA_prev.md）
# 用法：  cd hw_zcu104/syn && bash run_synth.sh <tag>     # tag 例: r5_xxx，不许带空格
# 耗时：  全芯片约 13 分钟（RuntimeOptimized）
# 为什么要在 E:\ae_syn：仓库路径含空格（"GPU ARCH"），Vivado 批处理在含空格
#         路径下不稳定；synth.tcl 的 $readmemh 也要求 exp2_lut.mem/seq.mem 在工作目录。
# 注意：  ① 非 OOC 必崩（2021.2 + 1728 PE 的 Final Netlist Cleanup 段访问违例，
#         synth.tcl 已带 -mode out_of_context，别去掉）
#         ② 偶发 .Xil 损坏：报 couldn't read .../realtime/... 时删工作区 .Xil 重跑一次
#         ③ 层级账（可选）：综合完成后 cd 工作区 && vivado -mode batch -source hier.tcl
#            （hier.tcl 从上一轮目录拷，例 E:\ae_syn\integ\hier.tcl）
# ============================================================================
set -eu
TAG="${1:?用法: bash run_synth.sh <tag>（如 r5_xxx，不带空格）}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"          # hw_zcu104 绝对路径
WORK="/e/ae_syn/$TAG"
VIVADO="D:/software/Vivado/2021.2/bin/vivado.bat"

[ -e "$WORK" ] && { echo "!! $WORK 已存在——历史轮次不许覆盖，换一个 tag"; exit 1; }
mkdir -p "$WORK"
cp -r "$SRC/rtl" "$WORK/rtl"
cp "$SRC/syn/synth.tcl" "$SRC/syn/exp2_lut.mem" "$SRC/syn/seq.mem" "$WORK/"
echo "== RTL/脚本已就位: $WORK"

cd "$WORK"
"$VIVADO" -mode batch -source synth.tcl 2>&1 | tee "vivado_${TAG}.log"

# PPA 提取（extract_ppa 把摘要写进 hw_zcu104/results/PPA.md——先备份上一轮）
if [ -f "$SRC/results/PPA.md" ]; then cp "$SRC/results/PPA.md" "$SRC/results/PPA_prev.md"; fi
python "$SRC/syn/extract_ppa.py" "$WORK/out"
echo "== 完成。报告: $WORK/out/{utilization,timing}.rpt；摘要: hw_zcu104/results/PPA.md"
