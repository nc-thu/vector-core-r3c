// top_eng64.sv — B8-64 引擎综合 wrapper（PCOLS=64 / PULSE_DLY=5，逻辑列 128）
// 用途：B8-48 vs B8-64 定版采点（H1 实测锚点外推 LUT ≈157.6k/79%——用综合实测
// 校准；同时看 303.215MHz 约束下 WNS 是否可收）。
module top_eng64 (
  input  logic clk,
  input  logic rst_n,
  input  logic start,
  output logic busy,
  output logic done,
  input  logic [15:0] m,
  input  logic [15:0] n,
  input  logic [15:0] n_loc,
  input  logic [15:0] j0,
  input  logic [15:0] k,
  input  logic [19:0] a_base, b_base, y_base,
  input  logic        y_tr,
  input  logic signed [15:0] rq_m,
  input  logic [7:0]  rq_s,
  output logic [19:0] ctxa_addr,
  input  logic [16*8-1:0] ctxa_rdata,
  output logic        ctxb_we,
  output logic [15:0] ctxb_welane,
  output logic [19:0] ctxb_addr,
  output logic [16*8-1:0] ctxb_wdata,
  output logic [11:0] w_addr,
  input  logic [64*16-1:0] w_rdata,
  output logic [31:0] mac_cnt,
  output logic        wb_active
);
  ae_gemm_p2 #(.PCOLS(64), .PULSE_DLY(5)) u_eng (
    .clk(clk), .rst_n(rst_n), .start(start), .busy(busy), .done(done),
    .m(m), .n(n), .n_loc(n_loc), .j0(j0), .k(k),
    .a_base(a_base), .b_base(b_base), .y_base(y_base), .y_tr(y_tr),
    .rq_m(rq_m), .rq_s(rq_s),
    .ctxa_addr(ctxa_addr), .ctxa_rdata(ctxa_rdata),
    .ctxb_we(ctxb_we), .ctxb_welane(ctxb_welane),
    .ctxb_addr(ctxb_addr), .ctxb_wdata(ctxb_wdata),
    .w_addr(w_addr), .w_rdata(w_rdata),
    .mac_cnt(mac_cnt), .wb_active(wb_active)
  );
endmodule
