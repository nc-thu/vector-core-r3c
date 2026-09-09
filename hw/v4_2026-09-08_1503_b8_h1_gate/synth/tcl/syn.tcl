# syn.tcl — H1 门 OOC 综合/实现（stage: synth|place|route）
# 用法: vivado -mode batch -source syn.tcl -tclargs <top> <period_ns> <outdir> <srcdir> <threads> <stage>
set top     [lindex $argv 0]
set period  [lindex $argv 1]
set outdir  [lindex $argv 2]
set srcdir  [lindex $argv 3]
set threads [lindex $argv 4]
set stage   [lindex $argv 5]

file mkdir $outdir
create_project -in_memory -part xczu7ev-ffvc1156-2-e
set_property target_language Verilog [current_project]
set_param general.maxThreads $threads
foreach f [glob -nocomplain -directory $srcdir *.sv] { read_verilog -sv $f }
foreach f [glob -nocomplain -directory $srcdir *.v]  { read_verilog $f }

synth_design -top $top -mode out_of_context -part xczu7ev-ffvc1156-2-e \
  -flatten_hierarchy none -directive RuntimeOptimized
create_clock -period $period -name clk [get_ports clk]
opt_design

report_utilization -file $outdir/util.rpt
report_utilization -hierarchical -file $outdir/util_hier.rpt

set wns 0.0
if {$stage eq "place" || $stage eq "route"} {
  place_design
  phys_opt_design -directive RuntimeOptimized
}
if {$stage eq "route"} {
  route_design
  set paths [get_timing_paths -max_paths 1 -nworst 1 -setup]
  set wns [get_property SLACK $paths]
  if {$wns < 0.0 && $wns > -0.2} {
    phys_opt_design -directive RuntimeOptimized
    set paths [get_timing_paths -max_paths 1 -nworst 1 -setup]
    set wns [get_property SLACK $paths]
  }
  report_timing -max_paths 20 -file $outdir/timing.rpt
  report_timing_summary -file $outdir/timing_summary.rpt
  report_drc -file $outdir/drc.rpt
  report_route_status -file $outdir/route_status.rpt
  write_checkpoint -force $outdir/post_route.dcp
} elseif {$stage eq "place"} {
  set paths [get_timing_paths -max_paths 1 -nworst 1 -setup]
  set wns [get_property SLACK $paths]
  report_timing -max_paths 20 -file $outdir/timing.rpt
  report_timing_summary -file $outdir/timing_summary.rpt
  write_checkpoint -force $outdir/post_place.dcp
} else {
  set paths [get_timing_paths -max_paths 1 -nworst 1 -setup]
  set wns [get_property SLACK $paths]
  report_timing -max_paths 20 -file $outdir/timing.rpt
  report_timing_summary -file $outdir/timing_summary.rpt
}
set fp [open $outdir/wns.txt w]; puts $fp $wns; close $fp
puts "H1_OK top=$top period=$period stage=$stage wns=$wns"
