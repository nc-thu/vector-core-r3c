// ae_pe.sv — 输出驻留脉冲阵列 PE：INT8×INT8 -> INT32 累加 + R3C 末脉冲快照
// A 自西向东、B 自北向南，各 1 拍寄存传递；valid 随数据同链传播（en = av_in & bv_in）。
// 每个对应 1 个 DSP48E2（MREG/PCRE 流水）。
// R3C 方案 C：末脉冲 pulse_in 随 A 链一拍传一个 PE 东传（与 a_in 同相偏斜）。
//   控制器在最后一个 k 切片之后隔 1 拍发射脉冲，波前同构 ⇒ 脉冲到拍恰好是
//   本行组最后一个部分和落进 acc_r 的下一拍：该拍把 acc_r 低 27 位抄进 snap_r
//   （|acc| ≤ K·128·128 ≤ 2^26，K ≤ 4096 = W_WORDS，27b 无损——与 requant
//   消费 acc[26:0] 的口径一致），并清零 acc_r。下一行组的喂数随即开始，
//   读出（量化/写回）全部走快照侧，两段并行。
`ifndef AE_PE_SV
`define AE_PE_SV
module ae_pe (
  input  logic                   clk,
  input  logic                   rst_n,
  input  logic                   clr,      // 累加器清零（R3C 后仅复位兜底；行组清零靠末脉冲）
  input  logic                   pulse_in, // R3C 末脉冲（西侧进入，与 a_in 同拍相位）
  input  logic                   av_in,    // a_in 波前有效
  input  logic                   bv_in,    // b_in 波前有效
  input  logic signed [7:0]      a_in,     // 西侧进入
  input  logic signed [7:0]      b_in,     // 北侧进入
  output logic                   av_out,
  output logic                   bv_out,
  output logic                   pulse_out, // R3C 末脉冲东传（延迟 1 拍，与 a_out 同构）
  output logic signed [7:0]      a_out,    // 东侧出（延迟 1 拍）
  output logic signed [7:0]      b_out,    // 南侧出（延迟 1 拍）
  output logic signed [31:0]     acc,      // 累加结果（驻留；R3C 读出改走 snap）
  output logic signed [26:0]     snap      // R3C 快照（requant 消费的低 27b 同口径）
);
  logic signed [7:0] a_r, b_r;
  logic              av_r, bv_r;
  logic              pulse_r;
  // use_dsp=yes：8x8 有符号乘+32b 累加若不加属性，Vivado 2021.2 会静默映射到 LUT
  // （1728 PE 全进 LUT => 125% 超载）；属性强制绑定 DSP48E2（UG901）
  (* use_dsp = "yes" *) logic signed [31:0] acc_r;
  // R3C 快照：27b，寄存器到寄存器直抄（无逻辑），读出侧符号扩展回 32b 进 requant
  logic signed [26:0] snap_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a_r <= '0; b_r <= '0; av_r <= 1'b0; bv_r <= 1'b0; pulse_r <= 1'b0;
      acc_r <= '0; snap_r <= '0;
    end else begin
      a_r <= a_in;  b_r <= b_in;
      av_r <= av_in; bv_r <= bv_in;
      pulse_r <= pulse_in;
      if (pulse_in) begin         // 末脉冲到拍：快照 + 清零（脉冲拍必无有效积——
        snap_r <= acc_r[26:0];    //   它比本组最后切片晚 1 拍、比下组首切片早 ≥1 拍）
        acc_r  <= '0;
      end else if (clr) begin
        acc_r <= '0;
      end else if (av_in & bv_in) begin
        acc_r <= acc_r + a_in * b_in;
      end
    end
  end
  assign a_out = a_r;  assign av_out = av_r;
  assign b_out = b_r;  assign bv_out = bv_r;
  assign pulse_out = pulse_r;
  assign acc  = acc_r;
  assign snap = snap_r;
endmodule
`endif
