#!/usr/bin/env bash
# vlbuild3.sh — vlbuild.sh + OPT_FAST/OPT_GLOBAL=-O3（NOTES.txt 提的零风险提速项）
# 用法与 vlbuild.sh 相同：bash vlbuild3.sh <verilator 额外参数...>
V=/home/nc23/.conda/envs/vsim/bin
set -e
$V/verilator --binary --timing -j 0 -Wno-fatal "$@" > /tmp/vlbuild3.log 2>&1 || true
if [ ! -f obj_dir/Vtb_ae_v.mk ]; then
  echo "vlbuild3: Verilator 生成阶段失败（SV 错误），日志："; cat /tmp/vlbuild3.log; exit 1
fi
make -C obj_dir -f Vtb_ae_v.mk -j 32 CXX=/usr/bin/g++-10 AR=/usr/bin/ar \
     LINK=/usr/bin/g++-10 OPT_FAST=-O3 OPT_GLOBAL=-O3 >> /tmp/vlbuild3.log 2>&1 || {
  echo "vlbuild3: make 失败，日志尾部："; tail -30 /tmp/vlbuild3.log; exit 1; }
echo "vlbuild3: obj_dir/Vtb_ae_v 构建完成（g++-10 -O3）"
