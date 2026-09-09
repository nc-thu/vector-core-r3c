#!/bin/bash
# stage2.sh -- mk_pcw_calib v3 流程合成整表 + 编译 + 逐位自检（一个 tag 一跑）
# 用法: bash stage2.sh <tag>   （依赖 /tmp/alg_smooth/{hw_calib_table_v2mod,pcw_scales,fp_biases}_<tag>.*）
set -e
PY=~/.conda/envs/holobrain/bin/python
A=/tmp/alg_smooth
TAG=$1
cd $A
$PY sw/mk_pcw_calib.py \
    --v2 $A/hw_calib_table_v2mod_${TAG}.json \
    --scales $A/pcw_scales_${TAG}.json \
    --biases $A/fp_biases_${TAG}.json \
    --out $A/hw_calib_table_smooth_${TAG}.json 2>&1 | tail -2
# s000 build
cd $A
$PY sw/compiler.py --trace /tmp/ae_hostdrv/trace_s000.json \
    --manifest /tmp/pcw_rtn/manifest.json --w8 /tmp/pcw_rtn/w8_full \
    --calib $A/hw_calib_table_smooth_${TAG}.json \
    --pcw-w8 $A/pcw_export_${TAG} --out $A/build_s000_${TAG} 2>&1 | tail -2
# s001 build
$PY sw/compiler.py --trace /tmp/ae_hostdrv/trace_s001.json \
    --manifest /tmp/pcw_rtn/manifest.json --w8 /tmp/pcw_rtn/w8_full \
    --calib $A/hw_calib_table_smooth_${TAG}.json \
    --pcw-w8 $A/pcw_export_${TAG} --out $A/build_s001_${TAG} 2>&1 | tail -2
$PY sw/fast_selftest.py $A/build_s000_${TAG} 2>&1 | tail -3
echo STAGE2_${TAG}_DONE
