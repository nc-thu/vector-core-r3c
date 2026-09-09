open_checkpoint E:/ae_syn/hb_h1/runs/eng48/route/post_route.dcp
create_clock -period 3.298 -name clk [get_ports clk]
set_param general.maxThreads 8
phys_opt_design -directive AggressiveExplore
set paths [get_timing_paths -max_paths 1 -nworst 1 -setup]
set wns [get_property SLACK $paths]
report_timing -max_paths 20 -file E:/ae_syn/hb_h1/runs/eng48/phsopt2/timing.rpt
report_timing_summary -file E:/ae_syn/hb_h1/runs/eng48/phsopt2/timing_summary.rpt
set fp [open E:/ae_syn/hb_h1/runs/eng48/phsopt2/wns.txt w]; puts $fp $wns; close $fp
puts "PHSOPT2 wns=$wns"
