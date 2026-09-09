# ============================================================================
# syn_micro.tcl — ae_hif8_dot16 OOC 综合（HiF8 微架构实验② 判定数据）
# 用法: vivado.bat -mode batch -source syn_micro.tcl -tclargs <PD> <SHARE_B> <目录名>
#   例: vivado.bat -mode batch -source syn_micro.tcl -tclargs 2 0 pd2
# 输出: E:/ae_syn/micro/syn/<目录名>/{util.rpt, timing.rpt, runme.log 由 vivado 记}
# 口径: 与全芯片基线一致 —— xczu7ev-ffvc1156-2-e，synth_design -directive
#       RuntimeOptimized，250 MHz（4 ns）目标，综合后未布局布线时序。
# ============================================================================
set PD      [lindex $argv 0]
set SHARE_B [lindex $argv 1]
set TAG     [lindex $argv 2]
set ROOT    E:/ae_syn/micro/syn/$TAG
file mkdir $ROOT

create_project -in_memory -part xczu7ev-ffvc1156-2-e
read_verilog -sv E:/ae_syn/micro/rtl/ae_hif8_dot16.sv
set_property generic [list PD=$PD SHARE_B=$SHARE_B] [current_fileset]

synth_design -top ae_hif8_dot16 -mode out_of_context -directive RuntimeOptimized
create_clock -period 4.000 -name clk [get_ports clk]

report_utilization      -file $ROOT/util.rpt
report_timing_summary   -file $ROOT/timing.rpt

# 摘要直接打屏（也进 vivado.log，便于 grep）
set wns [get_worst_slack -max]
set luts [llength [get_cells -hier -filter {PRIMITIVE_GROUP == LUT}]]
set ffs  [llength [get_cells -hier -filter {PRIMITIVE_GROUP == FLOP_LATCH}]]
set dsps [llength [get_cells -hier -filter {PRIMITIVE_GROUP == DSP}]]
puts "MICRO_RESULT TAG=$TAG PD=$PD SHARE_B=$SHARE_B LUT=$luts FF=$ffs DSP=$dsps WNS=$wns"
