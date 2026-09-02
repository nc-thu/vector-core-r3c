#!/usr/bin/env bash
# ============================================================================
# regression.sh — hw_zcu104 一键回归（交接入口，2026-08-30 新增）
#
# 跑什么：全芯片位精确冒烟(tb_ae+compare) + requant 门(tb_rq) + softmax 门(tb_sm16)
#         + packed PE 两个对拍台(tb_pe_pack / tb_pe_pack_dsp) + 周期模型对账(gem_cycles smoke)
# 判定：  全部位精确/对账通过 -> 末尾 ALL PASS，exit 0；任何一步失败 -> exit 1
# 用法：  cd hw_zcu104/sim && bash regression.sh
# 环境：  需要 iverilog(/c/iverilog/bin，脚本自己加 PATH) + python(numpy)；
#         tb_pe_pack_dsp 需要 Vivado 2021.2 的 unisim 模型(路径写死在下方，换机器改一处)
# 日志：  每步输出存 reg_logs/<步骤名>.log，失败先看日志再动手
# ============================================================================
set -u
cd "$(dirname "$0")"
export PATH=/c/iverilog/bin:$PATH
VIVADO_UNISIM="D:/software/Vivado/2021.2/data/verilog/src/unisims/DSP48E2.v"

mkdir -p reg_logs
PASS=(); FAIL=()

step() {  # step <名字> <命令...>
  local name="$1"; shift
  echo "== $name"
  if "$@" > "reg_logs/$name.log" 2>&1; then
    PASS+=("$name"); tail -2 "reg_logs/$name.log" | sed 's/^/   /'
  else
    FAIL+=("$name"); echo "   *** FAIL — 看 reg_logs/$name.log"; tail -5 "reg_logs/$name.log" | sed 's/^/   /'
  fi
}

# 1. 黄金向量生成（位精确神谕 + 冒烟 seq/ddr/expected_*)
step gen_vectors python gen_vectors.py

# 2. 全芯片冒烟仿真（COLS=12，REF/PRIM 两遍，dump CTX/DDR 终态）
#    文件清单与 syn/synth.tcl 的 read_verilog 一致（显式列出，不用 ../rtl/*.sv 通配——
#    通配会把实验性 RTL 卷进来，例如 ae_pe_pack_dsp.sv 例化 DSP48E2 原语，无 unisim 模型必炸）
RTL_MAIN="../rtl/ae_pkg.sv ../rtl/ae_dpram.sv ../rtl/ae_ctx_ram.sv ../rtl/ae_pe.sv ../rtl/ae_sysarr.sv ../rtl/ae_requant.sv ../rtl/rq_v2.sv ../rtl/rq_ms.sv ../rtl/ae_exp_lut.sv ../rtl/ae_gemm.sv ../rtl/ae_softmax.sv ../rtl/ae_copy.sv ../rtl/ae_dma.sv ../rtl/ae_sched.sv ../rtl/ae_core.sv ../rtl/ae_top.sv"
step sim_ae bash -c "iverilog -g2012 -o reg_ae.vvp -I ../rtl $RTL_MAIN tb_ae.sv && vvp reg_ae.vvp"

# 3. 位精确比对（四项 CTX/DDR × REF/PRIM，exit code 即判据）
#    守卫：sim_ae 没跑成时 dump_*.mem 还是上一轮的旧文件，比对会"假 PASS"——直接判 FAIL
if printf '%s\n' "${FAIL[@]:-}" | grep -q '^sim_ae$'; then
  FAIL+=(compare); echo "== compare  *** SKIP->FAIL（sim_ae 未产出新 dump，旧文件比对无意义）"
else
  step compare python compare.py
fi

# 4. requant 门 1 对拍（rq_v1 神谕 vs rq_v2/rq_ms/rq_m6，60k 向量）
step tb_rq bash -c 'iverilog -g2012 -o reg_rq.vvp tb_rq.sv ../rtl/rq_v1.sv ../rtl/rq_v2.sv ../rtl/rq_ms.sv ../rtl/rq_m6.sv && vvp reg_rq.vvp'

# 5. softmax SM16 对拍（读 sm16_ctrl/s/gold.mem，向量已存档，可再生成脚本失传——见交接 PITFALLS）
step tb_sm16 bash -c 'iverilog -g2012 -o reg_sm16.vvp tb_sm16.sv ../rtl/ae_softmax.sv ../rtl/ae_exp_lut.sv && vvp reg_sm16.vvp'

# 6. packed PE 行为级对拍（1×ae_pe_pack vs 2×ae_pe）
step tb_pe_pack bash -c 'iverilog -g2012 -o reg_pp.vvp tb_pe_pack.sv ../rtl/ae_pe.sv ../rtl/ae_pe_pack.sv && vvp reg_pp.vvp'

# 7. packed PE DSP 原语版对拍（需编 Vivado unisim DSP48E2 + glbl）
step tb_pe_pack_dsp bash -c "iverilog -g2012 -o reg_ppd.vvp tb_pe_pack_dsp.sv ../rtl/ae_pe.sv ../rtl/ae_pe_pack_dsp.sv '$VIVADO_UNISIM' glbl.v && vvp reg_ppd.vvp"

# 8. 周期模型对账（gem_cycles.py smoke：RTL 实测拍数 vs 模型，门槛 <5%）
step gem_cycles python gem_cycles.py smoke

echo ""
echo "================ 回归汇总 ================"
for p in "${PASS[@]:-}"; do [ -n "$p" ] && echo "  PASS  $p"; done
for f in "${FAIL[@]:-}"; do [ -n "$f" ] && echo "  FAIL  $f"; done
if [ ${#FAIL[@]} -gt 0 ]; then echo "结果: FAIL (${#FAIL[@]} 项)"; exit 1
else echo "结果: ALL PASS (${#PASS[@]} 项)"; fi
