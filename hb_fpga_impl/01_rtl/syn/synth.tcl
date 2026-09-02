# synth.tcl — ae_top 逻辑综合（ZCU104 / XCZU7EV-2FFVC1156，250 MHz 目标）
#
# ★ 在无空格路径下运行（本目录路径含空格会让 Vivado 批处理不稳定）：
#     mkdir /e/ae_syn && cp -r ../rtl ./rtl && cp ../syn/synth.tcl .
#     vivado -mode batch -source synth.tcl
# ★ 本设计（1728 PE）在 2021.2 上非 OOC 综合必然在 "Final Netlist Cleanup"
#   段崩溃（EXCEPTION_ACCESS_VIOLATION / exit 139），-mode out_of_context 可稳定绕过。
# ★ PE 乘法必须靠 ae_pe.sv 里的 (* use_dsp = "yes" *)，否则 8x8 有符号乘
#   被静默映射进 LUT（1728 个 => LUT 125% 超载、DSP 仅 12 个）。
set OUT ./out
file mkdir $OUT
set RTL ./rtl

read_verilog -sv [list \
  $RTL/ae_pkg.sv \
  $RTL/ae_dpram.sv \
  $RTL/ae_ctx_ram.sv \
  $RTL/ae_pe.sv \
  $RTL/ae_sysarr.sv \
  $RTL/ae_requant.sv \
  $RTL/rq_v2.sv \
  $RTL/rq_ms.sv \
  $RTL/ae_exp_lut.sv \
  $RTL/ae_gemm.sv \
  $RTL/ae_softmax.sv \
  $RTL/ae_copy.sv \
  $RTL/ae_dma.sv \
  $RTL/ae_sched.sv \
  $RTL/ae_core.sv \
  $RTL/ae_top.sv ]

set_param general.maxThreads 2
synth_design -top ae_top -part xczu7ev-ffvc1156-2-e \
  -mode out_of_context -flatten_hierarchy none -directive RuntimeOptimized

create_clock -period 4.000 -name clk [get_ports clk]

report_utilization -file $OUT/utilization.rpt
report_timing_summary -delay_type max -file $OUT/timing.rpt
write_checkpoint -force $OUT/ae_top.dcp

# 控制台摘要（进 vivado.log，便于 grep）
puts "==== AE_PPA ===="
foreach c [list LUT LUTRAM FF BRAM_36Kb URAM DSPs] {
  catch { puts "$c: [lindex [get_utilization $c] 0]" }
}
puts "================"
