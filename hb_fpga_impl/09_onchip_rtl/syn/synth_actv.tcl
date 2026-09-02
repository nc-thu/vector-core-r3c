# OOC 综合 ae_actv（XCZU7EV-2，250MHz 目标）——资源门槛对照（07_onchip_ops 预算表）
set OUT ./out
file mkdir $OUT
read_verilog -sv ./ae_actv.sv
set_param general.maxThreads 2
synth_design -top ae_actv -part xczu7ev-ffvc1156-2-e \
  -mode out_of_context -flatten_hierarchy none -directive RuntimeOptimized
create_clock -period 4.000 -name clk [get_ports clk]
report_utilization -file $OUT/utilization.rpt
report_timing_summary -delay_type max -file $OUT/timing.rpt
puts "==== ACTV_PPA ===="
foreach c [list LUT LUTRAM FF BRAM_36Kb URAM DSPs] {
  catch { puts "$c: [lindex [get_utilization $c] 0]" }
}
puts "================"
