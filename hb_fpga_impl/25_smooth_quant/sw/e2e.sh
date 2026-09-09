#!/bin/bash
# e2e.sh -- host_driver 端到端（s000+s001 顺序跑，~9 分钟/样本）
# 用法: bash e2e.sh <tag>
set -e
PY=~/.conda/envs/holobrain/bin/python
A=/tmp/alg_smooth
TAG=$1
cd $A
$PY sw/host_driver.py --build $A/build_s000_${TAG} \
    --trace /tmp/ae_hostdrv/trace_s000.json --batch /tmp/ae_hostdrv/batch_s000.pt \
    --ref /tmp/ae_hostdrv/fp32_ref_000.npz \
    --calib $A/hw_calib_table_smooth_${TAG}.json --sample-id 000 \
    --out $A/result_000_smooth_${TAG}.npz > $A/run000_smooth_${TAG}.log 2>&1
$PY sw/host_driver.py --build $A/build_s001_${TAG} \
    --trace /tmp/ae_hostdrv/trace_s001.json --batch /tmp/ae_hostdrv/batch_s001.pt \
    --ref /tmp/ae_hostdrv/fp32_ref_001.npz \
    --calib $A/hw_calib_table_smooth_${TAG}.json --sample-id 001 \
    --out $A/result_001_smooth_${TAG}.npz > $A/run001_smooth_${TAG}.log 2>&1
echo E2E_${TAG}_DONE
