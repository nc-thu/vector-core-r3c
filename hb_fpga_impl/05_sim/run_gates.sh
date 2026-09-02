#!/usr/bin/env bash
# ============================================================================
# run_gates.sh — iverilog -> Verilator 迁移的两道验收门（位精确对拍）
#
# 门 1（冒烟档，sim/，COLS=12/CTX=1024/W=64/SEQ=64/DDR=64KB）：
#   a. gen_vectors 生成 golden + iverilog 跑原版 tb_ae.sv（基线）
#   b. compare.py：基线 CTX+DDR 四项 vs golden（既有回归判据）
#   c. tb_ae_v.sv（Verilator 移植版 TB）用 iverilog 跑 +MODE=0/1，
#      dump 与基线逐字节 cmp —— 隔离变量：证明 TB 差别本身不改变结果
#   d. Verilator --binary --timing 编译同一 tb_ae_v.sv 跑 +MODE=0/1，
#      dump 与基线逐字节 cmp；cycles 三方（b/c/d）一致
#
# 门 2（全参数档，sim108/，COLS=108/CTX=131072/W=4096/SEQ=2048/DDR=8MB）：
#   a. gen_vectors_108 生成 golden（DDR 8MB 全覆盖映像）
#   b. iverilog 跑 tb_ae_p.sv（原版 TB 的参数化副本，含 CTX dump）+ compare.py
#   c. iverilog 与 Verilator 各跑 tb_ae_v.sv +MODE=0/1，DDR dump 逐字节互比
#      + 与基线 b 逐字节互比 + cycles 三方一致
#
# 大负载吞吐基准另见 bench.py（依赖本脚本先跑完）。
# 用法：cd ae_sim && bash run_gates.sh   （环境：~/.conda/envs/vsim）
#
# 2026-08-30 cycles 竞态根因（VCD 对拍定位，两处都修在 TB、RTL 零改动）：
#   TB 在 posedge 后用阻塞赋值改 start / rst_n，与 DUT 的 always_ff 同拍竞争，
#   iverilog 先评估 DUT（看到旧值）、Verilator 先执行 TB（看到新值）：
#   ① start=1 → Verilator 整个调度器提前一拍起跑（固定流水线周期数不变，
#      但 DMA stall 由 LFSR 绝对时间决定 → 只有 LOAD 路径 cycles 漂移）；
#   ② rst_n=1 复位释放 → rq_ms 的 slot 自由轮转计数器（复位清零/否则自增）
#      在 Verilator 里当拍就 +1，从此相位领先一拍 → S_DALIGN 等待长度不同。
#   修复：全部 TB 的 start/rst_n/hoist_en 改非阻塞赋值（与 iverilog 原基线
#   逐拍等价，仅消除调度顺序依赖）。
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
V=~/.conda/envs/vsim/bin
RTL="ae_pkg.sv ae_dpram.sv ae_ctx_ram.sv ae_pe.sv ae_sysarr.sv ae_requant.sv rq_v2.sv rq_ms.sv ae_exp_lut.sv ae_gemm.sv ae_softmax.sv ae_copy.sv ae_dma.sv ae_sched.sv ae_core.sv"
RTLFILES=""
for f in $RTL; do RTLFILES="$RTLFILES ../rtl/$f"; done
FULLDEF="-DV_COLS=108 -DV_CTX_WORDS=131072 -DV_W_WORDS=4096 -DV_SEQ_N=2048 -DV_DDR_BYTES=8388608 -DV_WDG_CYC=20000000"

PASS=0; FAIL=0
step() {  # step <名字> <命令...>
  local name="$1"; shift
  echo "== $name"
  if "$@" > "gate_${name}.log" 2>&1; then
    PASS=$((PASS+1)); tail -2 "gate_${name}.log" | sed 's/^/   /'
  else
    FAIL=$((FAIL+1)); echo "   *** FAIL — 看 gate_${name}.log"; tail -8 "gate_${name}.log" | sed 's/^/   /'
  fi
}
cyc3() {  # cyc3 <base日志> <tbv_iv日志> <vl日志>
  # 判据（2026-08-30 修订，两处 TB 竞态修复后）：
  #   1) 同一 TB（tb_ae_v.sv）跨工具：tbv(iverilog) 与 vl(Verilator)
  #      两模式 cycles 序列必须完全一致 —— 这是迁移验收的核心；
  #   2) 基线（tb_ae.sv 同进程连跑 REF+PRIM）的 REF 行必须与 tbv/vl 一致；
  #      PRIM 行允许不同：基线第二遍跑时 LFSR 已推进 ~8 万 ns，stall 相位与
  #      fresh 进程不同 → GEMM 入口对 slot 相位的等待长度不同（S_DALIGN 0-3 拍）。
  #      数据不受影响（dump 门保证逐字节一致），属 TB 方法学差异非工具分歧。
  local b_ref t_all v_all t_ref
  b_ref=$(grep -o 'cycles=[0-9]*' "$1" | head -1)
  t_all=$(grep -o 'cycles=[0-9]*' "$2" | tr '\n' ' ')
  v_all=$(grep -o 'cycles=[0-9]*' "$3" | tr '\n' ' ')
  t_ref=$(echo "$t_all" | awk '{print $1}')
  [ -n "$t_all" ] && [ "$t_all" = "$v_all" ] && [ "$b_ref" = "$t_ref" ] && \
    echo "cycles: iverilog==Verilator 全模式一致（$t_all）+ 基线 REF 一致（$b_ref）"
}

# ---------------------------------------------------------------- 门 1：冒烟档
cd sim
step g1_gen $V/python gen_vectors.py
step g1_iv_base bash -c "$V/iverilog -g2012 -o base.vvp -I ../rtl $RTLFILES tb_ae.sv && $V/vvp base.vvp"
step g1_iv_golden $V/python compare.py
cp dump_ddr_ref.mem iv_ddr_ref.mem; cp dump_ddr_prim.mem iv_ddr_prim.mem
step g1_iv_tbv bash -c "$V/iverilog -g2012 -o tbv_iv.vvp -I ../rtl $RTLFILES tb_ae_v.sv && $V/vvp tbv_iv.vvp +MODE=0 && $V/vvp tbv_iv.vvp +MODE=1"
step g1_cmp_tbv_iv bash -c "cmp iv_ddr_ref.mem dump_ddr_ref.mem && cmp iv_ddr_prim.mem dump_ddr_prim.mem && echo DDR逐字节一致"
step g1_vl_build bash -c "bash ../vlbuild.sh --top-module tb_ae_v tb_ae_v.sv $RTLFILES && ./obj_dir/Vtb_ae_v +MODE=0 && ./obj_dir/Vtb_ae_v +MODE=1"
step g1_cmp_vl_iv bash -c "cmp iv_ddr_ref.mem dump_ddr_ref.mem && cmp iv_ddr_prim.mem dump_ddr_prim.mem && echo DDR逐字节一致"
step g1_cycles cyc3 gate_g1_iv_base.log gate_g1_iv_tbv.log gate_g1_vl_build.log
cd ..

# ---------------------------------------------------------------- 门 2：全参数档
cd sim108
step g2_gen $V/python gen_vectors_108.py
step g2_iv_base bash -c "$V/iverilog -g2012 $FULLDEF -o full.vvp -I ../rtl $RTLFILES tb_ae_p.sv && $V/vvp full.vvp"
step g2_iv_golden $V/python compare.py
cp dump_ddr_ref.mem iv_ddr_ref.mem; cp dump_ddr_prim.mem iv_ddr_prim.mem
step g2_iv_tbv bash -c "$V/iverilog -g2012 $FULLDEF -o tbv_iv.vvp -I ../rtl $RTLFILES tb_ae_v.sv && $V/vvp tbv_iv.vvp +MODE=0 && $V/vvp tbv_iv.vvp +MODE=1"
step g2_cmp_tbv_iv bash -c "cmp iv_ddr_ref.mem dump_ddr_ref.mem && cmp iv_ddr_prim.mem dump_ddr_prim.mem && echo DDR逐字节一致"
step g2_vl_build bash -c "bash ../vlbuild.sh $FULLDEF --top-module tb_ae_v tb_ae_v.sv $RTLFILES && ./obj_dir/Vtb_ae_v +MODE=0 && ./obj_dir/Vtb_ae_v +MODE=1"
step g2_cmp_vl_iv bash -c "cmp iv_ddr_ref.mem dump_ddr_ref.mem && cmp iv_ddr_prim.mem dump_ddr_prim.mem && echo DDR逐字节一致"
step g2_cycles cyc3 gate_g2_iv_base.log gate_g2_iv_tbv.log gate_g2_vl_build.log
cd ..

echo ""
echo "================ 验收门汇总 ================"
echo "  PASS=$PASS FAIL=$FAIL"
[ $FAIL -eq 0 ] && echo "  两道门全部通过" || echo "  存在失败项，看 gate_*.log"
exit $([ $FAIL -eq 0 ] && echo 0 || echo 1)
