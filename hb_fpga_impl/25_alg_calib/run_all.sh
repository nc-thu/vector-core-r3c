#!/bin/bash
# run_all.sh -- 25_alg_calib 标定线编排：表 -> 编译 -> 自检 -> host_driver
# 用法: nohup bash run_all.sh <stage> &   stage = tables|builds|selftest|roundA|roundB
set -e
PY=~/.conda/envs/holobrain/bin/python
CAL=/tmp/alg_calib
PCW=/tmp/pcw_rtn
AED=/tmp/ae_hostdrv
cd $PCW

stage=$1

mk() {  # mk <tag> <scope> <mode> <mult>
  $PY $CAL/mk_real_tables.py --stats $CAL/diag_real_s000.json \
      --v2 $AED/hw_calib_table_v2.json --biases $PCW/fp_biases.json \
      --mode $3 --scope $2 --mult $4 --out $CAL/v2_$1.json
  $PY $PCW/sw/mk_pcw_calib.py --v2 $CAL/v2_$1.json \
      --scales $PCW/pcw_scales.json --biases $PCW/fp_biases.json \
      --out $CAL/t_$1.json
}

compile() {  # compile <tag> <sample>
  local S=$2
  $PY $PCW/sw/compiler.py --trace $AED/trace_s${S}.json \
      --manifest $PCW/manifest.json --w8 $PCW/w8_full \
      --calib $CAL/t_$1.json --pcw-w8 $PCW/pcw_export \
      --out $CAL/b_$1_s${S}
}

run() {  # run <tag> <sample>
  local S=$2
  $PY $PCW/sw/host_driver.py --build $CAL/b_$1_s${S} \
      --trace $AED/trace_s${S}.json --batch $AED/batch_s${S}.pt \
      --ref $AED/fp32_ref_${S}.npz --calib $CAL/t_$1.json \
      --sample-id ${S} --out $CAL/result_${S}_$1.npz \
      > $CAL/run_${S}_$1.log 2>&1
  echo "RUN_DONE $1 s${S} $(date +%H:%M:%S)"
}

case $stage in
tables)
  mk S1a all absmax 1.0
  mk S1p all p999 1.0
  mk S2 patch_embed absmax 1.0
  mk S3 seg140_250 absmax 1.0
  mk S4k08 seg140_250 absmax 0.8
  mk S4k09 seg140_250 absmax 0.9
  mk S4k11 seg140_250 absmax 1.1
  mk S4k125 seg140_250 absmax 1.25
  echo TABLES_DONE $(date +%H:%M:%S)
  ;;
builds)
  compile S1a 000; compile S1a 001
  compile S1p 000; compile S1p 001
  compile S2 000; compile S3 000
  compile S4k08 000; compile S4k09 000; compile S4k11 000; compile S4k125 000
  echo BUILDS_DONE $(date +%H:%M:%S)
  ;;
selftest)
  for b in $CAL/b_*; do
    echo "== selftest $b"
    $PY $PCW/sw/fast_selftest.py $b 2>&1 | tail -3
  done
  echo SELFTEST_DONE $(date +%H:%M:%S)
  ;;
roundA)
  run S1a 000 & run S1a 001 & run S1p 000 & run S1p 001
  wait
  echo ROUNDA_DONE $(date +%H:%M:%S)
  ;;
roundB)
  run S2 000 & run S3 000 & run S4k08 000 & run S4k09 000 & run S4k11 000
  wait
  echo ROUNDB_DONE $(date +%H:%M:%S)
  ;;
roundC)
  run S4k125 000
  echo ROUNDC_DONE $(date +%H:%M:%S)
  ;;
esac
