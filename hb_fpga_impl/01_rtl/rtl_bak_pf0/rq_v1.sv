// rq_v1.sv — v1 requant 的 OOC 基准/对拍神谕（与 hw_zcu104/rtl/ae_requant.sv 逐位相同，
// 仅改模块名以便 TB 同台实例化）。y = sat8((x * m) >>> s)，32×16 LUT 乘 + 48b 桶形移位。
`ifndef RQ_V1_SV
`define RQ_V1_SV
module rq_v1 (
  input  logic              clk,
  input  logic              rst_n,
  input  logic              in_vld,
  input  logic signed [31:0] x,
  input  logic signed [15:0] m,
  input  logic        [7:0]  s,
  output logic              out_vld,
  output logic signed [7:0]  y
);
  logic signed [47:0] prod_r;
  logic              v1_r, v2_r;
  logic signed [47:0] shr_r;

  (* use_dsp = "no" *) logic signed [47:0] prod;
  assign prod = x * m;   // 32x16 -> 48 bit，LUT 实现

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      prod_r <= '0; shr_r <= '0; v1_r <= 1'b0; v2_r <= 1'b0;
    end else begin
      prod_r <= prod;
      v1_r   <= in_vld;
      shr_r  <= prod_r >>> s;
      v2_r   <= v1_r;
    end
  end

  assign y = (shr_r > 127)        ? 8'sd127 :
             (shr_r < -128)       ? -8'sd128 :
                                    shr_r[7:0];
  assign out_vld = v2_r;
endmodule
`endif
