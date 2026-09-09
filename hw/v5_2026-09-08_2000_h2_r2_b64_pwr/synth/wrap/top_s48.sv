// top_s48.sv — 全宽阵列综合 wrapper（PCOLS=48，逻辑列 96）
// H2：drain_row 改 NREP=12 份副本端口（每份驱动 4 物理列），wrapper 把外部
// 单个标量打到每份上（综合视角等同旧版，验证的是副本分组布线）。
module top_s48 (
  input  logic clk, input  logic rst_n, input  logic clr,
  input  logic feed_vld, input  logic feed_pulse,
  input  logic [16*8-1:0]    a_feed,
  input  logic [48*16-1:0]   b_feed,
  input  logic [3:0]         drain_row,
  output logic [48*2*32-1:0] acc_row
);
  localparam int NREP = 12;   // (48 + 4 - 1) / 4
  logic [NREP*4-1:0] drain_row_rep;
  always_comb begin
    for (int ri = 0; ri < NREP; ri++) drain_row_rep[ri*4 +: 4] = drain_row;
  end
  ae_sysarr_p2 #(.ROWS(16), .PCOLS(48)) u_arr (
    .clk(clk), .rst_n(rst_n), .clr(clr),
    .feed_vld(feed_vld), .feed_pulse(feed_pulse),
    .a_feed(a_feed), .b_feed(b_feed),
    .drain_row_rep(drain_row_rep), .acc_row(acc_row)
  );
endmodule
