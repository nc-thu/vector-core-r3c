// top_s48.sv — 全宽阵列综合 wrapper（PCOLS=48，逻辑列 96）
module top_s48 (
  input  logic clk, input  logic rst_n, input  logic clr,
  input  logic feed_vld, input  logic feed_pulse,
  input  logic [16*8-1:0]    a_feed,
  input  logic [48*16-1:0]   b_feed,
  input  logic [3:0]         drain_row,
  output logic [48*2*32-1:0] acc_row
);
  ae_sysarr_p2 #(.ROWS(16), .PCOLS(48)) u_arr (
    .clk(clk), .rst_n(rst_n), .clr(clr),
    .feed_vld(feed_vld), .feed_pulse(feed_pulse),
    .a_feed(a_feed), .b_feed(b_feed),
    .drain_row(drain_row), .acc_row(acc_row)
  );
endmodule
