// ae_requant.sv — INT32 -> INT8 重量化：y = sat8((x * m) >>> s)
// m: Q8.8 有符号乘子。乘法用 LUT（use_dsp=no），把 1728 个 DSP 全留给 MAC 阵列。
// 2 拍流水：T0 乘法寄存，T1 移位+饱和。
`ifndef AE_REQUANT_SV
`define AE_REQUANT_SV
module ae_requant (
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
