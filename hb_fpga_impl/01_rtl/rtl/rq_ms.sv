// ============================================================================
// rq_ms.sv — requant 多列时分复用封装（门 1 变体 C）
// ----------------------------------------------------------------------------
// 依据：ae_gemm 的 requant 只在 drain 段跑（16 拍），K 喂入段（≥64 拍）全空闲，
//       m/s 一次 GEMM 内不变（S_INIT 锁存）→ 一套核心轮流服务 SHARE 列几乎免费。
// 结构：自由轮转 slot 计数（0..SHARE-1）；列 c 只在 slot==c 拍上数据（驱动契约，
//       集成时 drain FSM 以 16×SHARE 拍展开）。x 多路选择 + y 按 slot 解复用寄存。
// 延迟：slot 拍 → +2 拍核心 → +1 拍输出寄存（比 v1 多 1 拍，微架构口径可接受）。
// 数值：与所用核心（rq_v2）逐位一致，封装不引入数值变化。
// ============================================================================
`ifndef RQ_MS_SV
`define RQ_MS_SV
module rq_ms #(
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
  // slot 相位输出：drain FSM 靠它做首拍对齐（集成用；TB 也不再窥层次）
  output logic [$clog2(SHARE)-1:0]  slot_o
);
  logic [$clog2(SHARE)-1:0] slot, slot1, slot2;
  logic        core_ov;
  logic signed [7:0] core_y;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      slot <= '0; slot1 <= '0; slot2 <= '0;
      out_vld <= '0; y_bus <= '0;
    end else begin
      if (slot == SHARE-1) slot <= '0; else slot <= slot + 1'b1;
      slot1 <= slot;
      slot2 <= slot1;
      out_vld <= '0;                       // 默认清零，命中后覆盖
      if (core_ov) begin
        y_bus[slot2*8 +: 8] <= core_y;
        out_vld[slot2]      <= 1'b1;
      end
    end
  end

  logic signed [XW-1:0] x_sel;
  assign x_sel  = x_bus[slot*XW +: XW];
  assign slot_o = slot;

  rq_v2 #(.XW(XW), .T_MAX(T_MAX)) u_core (
    .clk(clk), .rst_n(rst_n),
    .in_vld(in_vld[slot]), .x(x_sel), .m(m), .s(s),
    .out_vld(core_ov), .y(core_y)
  );
endmodule
`endif
