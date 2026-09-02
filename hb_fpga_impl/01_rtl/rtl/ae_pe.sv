// ae_pe.sv — 输出驻留脉冲阵列 PE：INT8×INT8 -> INT32 累加
// A 自西向东、B 自北向南，各 1 拍寄存传递；valid 随数据同链传播（en = av_in & bv_in）。
// 每个对应 1 个 DSP48E2（MREG/PCRE 流水）。
`ifndef AE_PE_SV
`define AE_PE_SV
module ae_pe (
  input  logic                   clk,
  input  logic                   rst_n,
  input  logic                   clr,      // 累加器清零（tile 开始）
  input  logic                   av_in,    // a_in 波前有效
  input  logic                   bv_in,    // b_in 波前有效
  input  logic signed [7:0]      a_in,     // 西侧进入
  input  logic signed [7:0]      b_in,     // 北侧进入
  output logic                   av_out,
  output logic                   bv_out,
  output logic signed [7:0]      a_out,    // 东侧出（延迟 1 拍）
  output logic signed [7:0]      b_out,    // 南侧出（延迟 1 拍）
  output logic signed [31:0]     acc        // 累加结果（驻留读出）
);
  logic signed [7:0] a_r, b_r;
  logic              av_r, bv_r;
  // use_dsp=yes：8x8 有符号乘+32b 累加若不加属性，Vivado 2021.2 会静默映射到 LUT
  // （1728 PE 全进 LUT => 125% 超载）；属性强制绑定 DSP48E2（UG901）
  (* use_dsp = "yes" *) logic signed [31:0] acc_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a_r <= '0; b_r <= '0; av_r <= 1'b0; bv_r <= 1'b0; acc_r <= '0;
    end else begin
      a_r <= a_in;  b_r <= b_in;
      av_r <= av_in; bv_r <= bv_in;
      if (clr)       acc_r <= '0;
      else if (av_in & bv_in) acc_r <= acc_r + a_in * b_in;
    end
  end
  assign a_out = a_r;  assign av_out = av_r;
  assign b_out = b_r;  assign bv_out = bv_r;
  assign acc   = acc_r;
endmodule
`endif
