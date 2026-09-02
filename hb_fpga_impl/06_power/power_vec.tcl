# power_vec.tcl — A档：开定版综合 DCP，出 vectorless（无活动注记）功耗
# 口径：OOC 综合后、未实现（估计线负载）、无 SAIF → 默认翻转率假设，保守
# 复现：cd /e/ae_syn/pwr && D:/software/Vivado/2021.2/bin/vivado.bat -mode batch -source power_vec.tcl
set_param general.maxThreads 2

set DCP "E:/ae_syn/pf_tmax/out/ae_top.dcp"
puts "==== open_checkpoint $DCP"
open_checkpoint $DCP

# 工艺角/环境：不手动 set_operating_conditions，用 DCP 默认（记录进 log 供口径说明）
puts "==== OPERATING CONDITIONS (default) ===="
catch { report_operating_conditions } roc
catch { puts $roc }

# 确认时钟约束（功耗按约束频率算：clk 4ns = 250MHz）
catch { report_clocks }

# 1) 全设计 vectorless
report_power -file ./power_vectorless.rpt -name vec_synth_ooc_250M

# 2) 探针：current_instance 能否把 report_power 限定到子实例（决定能否出分模块功耗）
if {![catch { current_instance u_core/u_gemm } cerr]} {
  catch { report_power -file ./power_probe_gemm.rpt -name probe_gemm }
  current_instance
} else {
  puts "==== current_instance 不支持/失败: $cerr（分模块功耗改用 utilization 归因）"
}
exit
