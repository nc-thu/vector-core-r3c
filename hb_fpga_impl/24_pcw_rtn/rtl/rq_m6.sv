// ============================================================================
// rq_m6.sv — requant 乘子量化变体（门 1 变体 D，数值有偏，仅出数据不拍板）
// ----------------------------------------------------------------------------
// 语义: 驱动侧把 m(Q8.8) 分解为 m ≈ m6 · 2^(8-t6)，m6 为 6b 有符号（[-32,31]），
//       y = sat8((x · m6) >>> t6)，t6 为有符号移位量（<0 时左移）。
// 效果: XW×6 窄乘 + 小移位器替代 XW×16；代价 = m 的相对量化误差（≤1/16 量级，
//       TB 对 v1 输出统计 max_abs/mean_abs 差异分布 —— 数据供用户决策）。
// ============================================================================
`ifndef RQ_M6_SV
`define RQ_M6_SV
module rq_m6 #(
  parameter int XW = 27
)(
  input  logic              clk,
  input  logic              rst_n,
  input  logic              in_vld,
  input  logic signed [XW-1:0] x,
  input  logic signed [5:0]  m6,
  input  logic signed [4:0]  t6,     // ∈[-16,15]；<0 → 左移
  output logic              out_vld,
  output logic signed [7:0]  y
);
  localparam int PW = XW + 6 + 16;   // 左移最大 16 保护位

  (* use_dsp = "no" *) logic signed [XW+5:0] p;
  assign p = x * m6;

  logic signed [PW-1:0] p_r, sh;
  logic v1_r, v2_r;
  logic signed [PW-1:0] sh_r;
  logic signed [4:0] t6_r;                 // 移位量与积同拍流水对齐
  assign sh = (t6_r < 0) ? (p_r <<< -t6_r) : (p_r >>> t6_r);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      p_r <= '0; sh_r <= '0; t6_r <= '0; v1_r <= 1'b0; v2_r <= 1'b0;
    end else begin
      p_r   <= p;
      t6_r  <= t6;
      v1_r  <= in_vld;
      sh_r  <= sh;
      v2_r  <= v1_r;
    end
  end

  logic pos_ov, neg_ov;
  assign pos_ov = ~sh_r[PW-1] & (|sh_r[PW-2:7]);
  assign neg_ov =  sh_r[PW-1] & (~&sh_r[PW-2:7]);
  assign y = pos_ov ? 8'sd127 : neg_ov ? -8'sd128 : sh_r[7:0];
  assign out_vld = v2_r;
endmodule
`endif
