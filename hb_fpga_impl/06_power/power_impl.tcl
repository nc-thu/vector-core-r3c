# power_impl.tcl — A档：OOC 实现（opt/place/route）后再报一次 vectorless 功耗
# 目的：布线后网线电容真实化，功耗估计比综合态准（报告 Confidence 里
#       "Design implementation state" 一项会从 Low 变 High）。
# 手法：P&R 时把时钟放松到 10ns（100% DSP 占用的网表按 4ns 跑 router 会非常久，
#       而功耗只关心物理布线，不关心时序收敛）；报功耗前把时钟改回 4ns=250MHz，
#       report_power 按约束频率算动态功耗。
# 复现：cd /e/ae_syn/pwr && D:/software/Vivado/2021.2/bin/vivado.bat -mode batch -source power_impl.tcl
set_param general.maxThreads 2

open_checkpoint "E:/ae_syn/pf_tmax/out/ae_top.dcp"

puts "==== 放松时钟到 10ns 跑 P&R"
catch { delete_clock_objects [get_clocks clk] }
create_clock -period 10.000 -name clk [get_ports clk]

opt_design   -directive RuntimeOptimized
place_design -directive RuntimeOptimized
route_design -directive RuntimeOptimized

puts "==== 时钟改回 4ns (250MHz) 报功耗"
catch { delete_clock_objects [get_clocks clk] }
create_clock -period 4.000 -name clk [get_ports clk]

report_utilization -file ./util_impl.rpt
report_timing_summary -delay_type max -max_paths 3 -file ./timing_impl.rpt
report_power -file ./power_impl_vectorless.rpt -name impl_ooc_250M
write_checkpoint -force ./ae_top_impl_ooc.dcp
exit
