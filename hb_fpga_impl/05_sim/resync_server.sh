#!/usr/bin/env bash
# ============================================================================
# resync_server.sh — 新 RTL（调度器双发射 pf + rq_ms T_MAX=39）重同步后的跨工具复验
# 由本地 resync_gates.sh 上传并调起，在服务器 ae_sim/ 下运行。
#
# 阶段 1：冒烟档 4 用例（default/tail/pf/rqs）× 三遍（REF / PRIM-pf0 / PRIM-pf1）
#   每用例五道检查：
#     a. gen_vectors --case X 生成 golden
#     b. iverilog tb_ae.sv 三遍连跑（基线）+ compare.py 对 golden
#     c. prim2 vs prim 逐字节（pf 只省拍不改数据）
#     d. iverilog tb_ae_v.sv 三进程（+MODE=0 / +MODE=1 / +MODE=1 +PF=1），
#        dump 与基线逐字节比
#     e. Verilator 同一 tb_ae_v.sv 三进程，dump 与基线逐字节比
#   cycles 判据（同 run_gates.sh 修订版）：tbv-iverilog 与 tbv-Verilator 三遍
#   完全一致 + 基线 REF 行一致；基线 PRIM 行允许差（同进程连跑的 LFSR 相位
#   与 fresh 进程不同，tb_ae.sv 的 mark/load 只对齐 pf1 遍与 pf0 遍）。
# 阶段 2：全参数大负载 Verilator 重测（T_MAX=39 移位器变宽后的新吞吐数字）
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"
V=/home/nc23/.conda/envs/vsim/bin
RTL="ae_pkg.sv ae_dpram.sv ae_ctx_ram.sv ae_pe.sv ae_sysarr.sv ae_requant.sv rq_v2.sv rq_ms.sv ae_exp_lut.sv ae_gemm.sv ae_softmax.sv ae_copy.sv ae_dma.sv ae_sched.sv ae_core.sv"
RTLFILES=""
for f in $RTL; do RTLFILES="$RTLFILES ../rtl/$f"; done
FULLDEF="-DV_COLS=108 -DV_CTX_WORDS=131072 -DV_W_WORDS=4096 -DV_SEQ_N=2048 -DV_DDR_BYTES=8388608 -DV_WDG_CYC=20000000"

PASS=0; FAIL=0
step() {
  local name="$1"; shift
  echo "== $name"
  if "$@" > "gate_${name}.log" 2>&1; then
    PASS=$((PASS+1)); tail -2 "gate_${name}.log" | sed 's/^/   /'
  else
    FAIL=$((FAIL+1)); echo "   *** FAIL — 看 gate_${name}.log"; tail -8 "gate_${name}.log" | sed 's/^/   /'
  fi
}

cyc_case() {  # <case>：tbv 跨工具三遍一致 + 基线 REF 一致
  local a b c a0 b0
  a=$(grep -o 'cycles=[0-9]*' "gate_rs_${1}_iv_base.log" | tr '\n' ' ')
  b=$(grep -o 'cycles=[0-9]*' "gate_rs_${1}_tbv_iv.log" | tr '\n' ' ')
  c=$(grep -o 'cycles=[0-9]*' "gate_rs_${1}_tbv_vl.log" | tr '\n' ' ')
  a0=$(echo "$a" | awk '{print $1}'); b0=$(echo "$b" | awk '{print $1}')
  [ -n "$b" ] && [ "$b" = "$c" ] && [ "$a0" = "$b0" ] && \
    echo "cycles: iverilog==Verilator 三遍一致（$b）+ 基线 REF 一致（$a0）"
}

# ---------------- 阶段 1：冒烟档 4 用例 ----------------
cd sim
step rs_vl_build bash -c "bash ../vlbuild.sh --top-module tb_ae_v tb_ae_v.sv $RTLFILES"
for CASE in default tail pf rqs; do
  step rs_${CASE}_gen $V/python gen_vectors.py --case $CASE
  step rs_${CASE}_iv_base bash -c "$V/iverilog -g2012 -o rs_base.vvp -I ../rtl $RTLFILES tb_ae.sv && $V/vvp rs_base.vvp"
  step rs_${CASE}_golden $V/python compare.py
  step rs_${CASE}_pf_data bash -c "cmp dump_ddr_prim.mem dump_ddr_prim2.mem && echo pf 数据不变: prim==prim2 逐字节"
  cp dump_ddr_ref.mem rs_ref.mem; cp dump_ddr_prim.mem rs_prim.mem; cp dump_ddr_prim2.mem rs_prim2.mem
  step rs_${CASE}_tbv_iv bash -c "$V/iverilog -g2012 -o rs_tbv.vvp -I ../rtl $RTLFILES tb_ae_v.sv && $V/vvp rs_tbv.vvp +MODE=0 && $V/vvp rs_tbv.vvp +MODE=1 && $V/vvp rs_tbv.vvp +MODE=1 +PF=1"
  step rs_${CASE}_cmp_iv bash -c "cmp rs_ref.mem dump_ddr_ref.mem && cmp rs_prim.mem dump_ddr_prim.mem && cmp rs_prim2.mem dump_ddr_prim2.mem && echo DDR 三遍逐字节一致（iverilog tb_ae_v）"
  step rs_${CASE}_tbv_vl bash -c "./obj_dir/Vtb_ae_v +MODE=0 && ./obj_dir/Vtb_ae_v +MODE=1 && ./obj_dir/Vtb_ae_v +MODE=1 +PF=1"
  step rs_${CASE}_cmp_vl bash -c "cmp rs_ref.mem dump_ddr_ref.mem && cmp rs_prim.mem dump_ddr_prim.mem && cmp rs_prim2.mem dump_ddr_prim2.mem && echo DDR 三遍逐字节一致（Verilator）"
  step rs_${CASE}_cycles cyc_case $CASE
done
cd ..

# ---------------- 阶段 2：全参数大负载 Verilator 重测 ----------------
cd sim_big
step rs_big_gen $V/python emit_big.py
step rs_big_vl bash -c "bash ../vlbuild.sh $FULLDEF --top-module tb_ae_v tb_ae_v.sv $RTLFILES && t0=\$(date +%s.%N) && ./obj_dir/Vtb_ae_v +MODE=0 && t1=\$(date +%s.%N) && echo wall_s=\$(echo \"\$t1 - \$t0\" | bc)"
cd ..

echo ""
echo "================ 重同步复验汇总 ================"
echo "  PASS=$PASS FAIL=$FAIL"
[ $FAIL -eq 0 ] && echo "  全部通过" || echo "  存在失败项，看 gate_rs_*.log"
exit $([ $FAIL -eq 0 ] && echo 0 || echo 1)
