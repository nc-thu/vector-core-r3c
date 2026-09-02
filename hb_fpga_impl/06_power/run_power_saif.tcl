# run_power_saif.tcl — SAIF 活动注记功耗（B档主脚本，吃文件路径参数）
#
# 用法（无空格路径下运行，仓库路径含空格会让 Vivado 批处理不稳定）：
#   D:/software/Vivado/2021.2/bin/vivado.bat -mode batch -source run_power_saif.tcl \
#       -tclargs <activity.saif> [DCP] [输出rpt]
#
#   参数 1  activity.saif  必填。由 vcd2saif.py 生成（Windows 版 Vivado 2021.2
#          没有 vcd2saif，用 06_power/vcd2saif.py 替代）。
#   参数 2  DCP            默认 E:/ae_syn/pf_tmax/out/ae_top.dcp（定版综合 OOC）
#   参数 3  输出rpt        默认 ./power_saif.rpt
#
# ★ SAIF 格式与层次（2021.2 read_saif 实测校准，Vivado 没有官方 vcd2saif 文档，
#   以下每条都踩过坑）：
#   1) 文件必须整体包在 (SAIFILE ... ) 括号里，DIRECTION 写 "backward"。
#      缺外壳时解析器不报错，但一条网都不注记（Design nets matched 恒为 1，
#      那 1 个来自时钟约束，不是 SAIF）。
#   2) 网条目是 (名字 (T0 x) (T1 x) (TX x) (TC n))，名字不能加引号——加引号
#      直接 Power 33-52 syntax error。总线位名写 a[3] 即可，可不转义括号。
#   3) 本版本 read_saif 没有 -instance_name/-input 参数，只有 -strip_path/-no_strip，
#      且默认剥掉 SAIF 最外两层 INSTANCE（实测 -strip_path tb 只剥一层、不够；
#      等效正确做法之一是单根 INSTANCE u_core 配 -no_strip）。
#   4) tb_ae 例化的是 ae_core（=netlist 的 ae_top/u_core），所以 vcd2saif.py 用
#      --re-root tb_ae.dut --wrap tb_ae.dut.u_core 生成 tb_ae{dut{u_core{...}}}，
#      剥两层后剩下的 u_core 恰好挂到 netlist 根的 u_core 上。
#   冒烟链路（实测注记 1540 个网）：
#     python vcd2saif.py tb_ae.vcd smoke.saif --re-root tb_ae.dut \
#            --wrap tb_ae.dut.u_core --t-start <窗口起点ps>
#     vivado -mode batch -source run_power_saif.tcl -tclargs smoke.saif
set_param general.maxThreads 2

if {$argc < 1} {
    puts "用法: vivado -mode batch -source run_power_saif.tcl -tclargs <saif> \[dcp\] \[rpt\]"
    exit 1
}
set SAIF [lindex $argv 0]
if {$argc > 1} { set DCP [lindex $argv 1] } else { set DCP "E:/ae_syn/pf_tmax/out/ae_top.dcp" }
if {$argc > 2} { set RPT  [lindex $argv 2] } else { set RPT  "./power_saif.rpt" }

puts "==== SAIF=$SAIF"
puts "==== DCP=$DCP"
open_checkpoint $DCP

# 挂活动：-out_file 收集没匹配上的网（核对覆盖率用）
# 未匹配到的网表网保持 vectorless 默认活动（报告 Confidence 表给出注记覆盖率）
read_saif $SAIF -out_file ${RPT}_unmatched.txt -verbose

report_power -file $RPT -name saif_annotated
exit
