#!/usr/bin/env bash
# vlbuild_vgate.sh — Verilator 构建 R1+R2 RTL（DDR=512KB，快速 dump）
# 用法：在 /tmp/ae_vgate/ 里跑
V=~/.conda/envs/vsim/bin
cd /tmp/ae_vgate
$V/verilator --binary --timing -j 0 -Wno-fatal --top-module tb_ae_v \
  -DV_COLS=108 -DV_CTX_WORDS=131072 -DV_W_WORDS=4096 -DV_SEQ_N=2048 \
  -DV_DDR_BYTES=524288 -DV_WDG_CYC=20000000 \
  rtl/ae_*.sv rtl/rq_*.sv tb_ae_v.sv > /tmp/ae_vgate/vl.log 2>&1 || true
if [ ! -f obj_dir/Vtb_ae_v.mk ]; then
  echo "GEN_FAIL"; tail -40 /tmp/ae_vgate/vl.log; exit 1
fi
make -C obj_dir -f Vtb_ae_v.mk -j 32 CXX=/usr/bin/g++-10 AR=/usr/bin/ar \
     LINK=/usr/bin/g++-10 >> /tmp/ae_vgate/vl.log 2>&1 || {
  echo "MAKE_FAIL"; tail -30 /tmp/ae_vgate/vl.log; exit 1; }
echo "BUILD_OK"
