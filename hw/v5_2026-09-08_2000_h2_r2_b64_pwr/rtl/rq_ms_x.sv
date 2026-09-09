// rq_ms_x.sv — rq_ms 输入流水版（H2 时序收口，2026-09-08）
// ----------------------------------------------------------------------------
// 与 rq_ms 的唯一差别：slot 选择 mux 的输出先进一拍寄存器 x_sel_r 再进核心。
// rq_ms 原版里 acc_rq_r → 4 选 1 slot mux → 27×8 窄乘（进 rq_v2 的 T0 寄存）
// 挤在同一个 3.298ns 拍里——这是 eng48 布线 −0.169ns 的第二条腿。本封装拆开：
//   拍 A：x_bus[next_slot] 4 选 1 mux → x_sel_r（只做选择，无算术）
//   拍 B：x_sel_r → rq_v2 核心（只做乘法，rq_ms 孤立综合 487MHz 的口径）
// 相位契约：mux 用 next_slot（slot+1）预选，寄存一拍后送达核心时正好对上
// 核心当拍的 slot 相位——核心在拍 t 消费的列号与 rq_ms 原版逐拍相同，
// 输出标签（slot2）管线不动，y/out_vld 数值逐位一致、只是整体晚 1 拍。
// 集成方（ae_gemm_p2）把 rq_v 提前一拍发射（DALIGN 等 slot==2 发）即可
// 把这 1 拍完全吸收：核心消费窗口与原版逐拍重合，总拍数不变。
`ifndef RQ_MS_X_SV
`define RQ_MS_X_SV
module rq_ms_x #(
  parameter int SHARE = 4,     // 复用列数
  parameter int XW    = 27,
  parameter int T_MAX = 0
)(
  input  logic                       clk,
  input  logic                       rst_n,
  input  logic [SHARE-1:0]           in_vld,     // 列 c 在 slot==c 拍有效
  input  logic [SHARE*XW-1:0]        x_bus,      // 列 c 的 x = x_bus[c*XW +: XW]
  input  logic signed [15:0]         m,
  input  logic [7:0]                 s,
  output logic [SHARE-1:0]           out_vld,
  output logic [SHARE*8-1:0]         y_bus,
  output logic [$clog2(SHARE)-1:0]   slot_o
);
  logic [$clog2(SHARE)-1:0] slot, slot1, slot2, slot_nx;
  logic        core_ov;
  logic signed [7:0] core_y;
  logic        vld_r;
  logic signed [XW-1:0] x_sel_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      slot <= '0; slot1 <= '0; slot2 <= '0;
      out_vld <= '0; y_bus <= '0;
      vld_r <= 1'b0; x_sel_r <= '0;
    end else begin
      if (slot == SHARE-1) slot <= '0; else slot <= slot + 1'b1;
      slot1 <= slot;
      slot2 <= slot1;
      out_vld <= '0;                       // 默认清零，命中后覆盖
      if (core_ov) begin
        y_bus[slot2*8 +: 8] <= core_y;     // 沿取当前 slot2 = 消费列号（同 rq_ms）
        out_vld[slot2]      <= 1'b1;
      end
      // 输入流水：用下一拍相位预选，寄存一拍后核心当拍取到同列数据
      vld_r   <= in_vld[slot];
      x_sel_r <= x_bus[slot_nx*XW +: XW];
    end
  end

  assign slot_nx = (slot == SHARE-1) ? '0 : (slot + 1'b1);
  assign slot_o  = slot;

  rq_v2 #(.XW(XW), .T_MAX(T_MAX)) u_core (
    .clk(clk), .rst_n(rst_n),
    .in_vld(vld_r), .x(x_sel_r), .m(m), .s(s),
    .out_vld(core_ov), .y(core_y)
  );
endmodule
`endif
