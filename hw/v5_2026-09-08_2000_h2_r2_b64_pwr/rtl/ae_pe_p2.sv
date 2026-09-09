// ae_pe_p2.sv — Pack2 脉动 PE（B8×R3C 集成 H1 门）：INT8×{W0,W1} -> 2×INT32 + 末脉冲快照
// ----------------------------------------------------------------------------
// 设计依据（arch v5 选型 B8-48：每接口拍每行喂 1 个激活）：
//   * Pack2：一颗 DSP48E2 的位空间塞 2 对 int8 乘法（offset packing，
//     pe_w8a8_sota/rtl/common/pack2_mult_dsp.sv，B8 线 10,251 dots 位精确已证）。
//     每物理列驻 2 条逻辑列权重（b_in=16b：{w1,w0}），每拍出 2 个积。
//   * 单时钟：B8 的 Pump2 第二相位在"B8-48 映射"下闲置（每行每接口拍只喂
//     1 个 a），所以不需要 606/303 双时钟——本 PE 全部在接口时钟域。
//     v5 模型 26.5% 利用率（= R3C 53% 的一半）正是这个闲置的定价，
//     帧时间 0.909s 的账一分不少（每 DSP 2 有效 MAC/接口拍 × 303.215 MHz）。
//   * R3C 末脉冲快照机制保留：pulse_in 随 A 链东传，到拍把两份 acc 低 27b
//     抄进快照并清零；读出全走快照侧，下一行组喂数与读出并行。
//
// 与 ae_pe（R3C，2 拍落地）的关键差异——积要 6 拍才落进 acc：
//   a_in/b_in@t -> 链寄存 t+1 -> pack2 S0 t+2 -> MREG t+3 -> PREG t+4
//   -> stage1 prod_r t+5 -> stage2 acc 更新结束于 t+5（t+6 可见）。
//   脉冲到拍 t_p 快照的是"start(t_p) 时刻的 acc"= 操作数 ≤ t_p-6 的全部积。
//   背靠背（下一组首切片 t_first = t_last+3）安全窗口 = [t_last+6, t_last+7]，
//   宽 2 拍（实测 tb_sys_p2 PD∈{5,6} 过 / PD=7 丢下组首积 / PD≥8 污染）：
//   上界不是 +8——脉冲拍优先级高于累加，恰好当拍落地的积被"丢弃"而非污染，
//   t_first 的积落在 t_last+8 = 脉冲拍 ⟹ +8 已经丢积。发射滞后取 5 拍
//   （PULSE_DLY=5，见 ae_gemm_p2）。
//   机制上成立的核心：脉冲清的是累加器不是流水线——下一组已进入
//   DSP/stage1/stage2 的在飞积不受清零影响，落到清零后的 acc 上。
//
// 数值：|acc| ≤ K·128·128 ≤ 2^26（K ≤ 4096 = W_WORDS），27b 快照无损
// （与 requant 消费 acc[26:0] 的口径一致，同 ae_pe）。
`ifndef AE_PE_P2_SV
`define AE_PE_P2_SV
module ae_pe_p2 (
  input  logic                   clk,
  input  logic                   rst_n,
  input  logic                   clr,      // 累加器清零（描述符起点幂等发一次；行组清零靠末脉冲）
  input  logic                   pulse_in, // R3C 末脉冲（西侧进入；快照拍 = 本 PE 窗口内）
  input  logic                   av_in,
  input  logic                   bv_in,
  input  logic signed [7:0]      a_in,     // 西侧进入（激活，A 链）
  input  logic [15:0]            b_in,     // 北侧进入（{w1,w0} 两逻辑列权重，B 链）
  output logic                   av_out,
  output logic                   bv_out,
  output logic                   pulse_out,
  output logic signed [7:0]      a_out,
  output logic [15:0]            b_out,
  output logic signed [31:0]     acc0,     // 逻辑列 2j 的累加结果（驻留）
  output logic signed [31:0]     acc1,     // 逻辑列 2j+1
  output logic signed [26:0]     snap0,    // 快照（requant 消费口径）
  output logic signed [26:0]     snap1
);
  // ---- 脉动链寄存（接口时钟域，与 ae_pe 同构；b 链 16b）----
  logic signed [7:0] a_r;
  logic [15:0]       b_r;
  logic              av_r, bv_r, pulse_r;

  // ---- Pack2 乘法（显式 DSP48E2；Verilator 下自动换位精确行为模型）----
  logic [47:0] P;
  logic        v_p;
  logic signed [7:0] a_p;
  pack2_mult_dsp #(.M_REG(1), .P_REG(1)) u_mul (
    .clk(clk), .v_in(av_r & bv_r), .row_in(1'b0),
    .a_in(a_r), .w0_in(b_r[7:0]), .w1_in(b_r[15:8]),
    .P(P), .v_p(v_p), .row_p(), .a_p(a_p)
  );

  // ---- stage1：packed field 抽取/偏置校正 -> 16b 真实有符号积（同 pe_b8）----
  wire signed [15:0] corr = a_p <<< 7;
  wire signed [17:0] lo_e = $signed({{2{P[15]}}, P[15:0]}) - corr;
  wire signed [17:0] hi_e = $signed({{2{P[31]}}, P[31:16]})
                            + (P[15] ? 18'sd1 : 18'sd0) - corr;
  logic signed [15:0] prod0_r, prod1_r;
  logic               v_r;

  // ---- stage2：2× 32b 累加（本映射无行交错，每拍两 lane 各自累加）----
  (* use_dsp = "no" *) logic signed [31:0] acc0_r, acc1_r;
  wire signed [31:0] prod0_ext = {{16{prod0_r[15]}}, prod0_r};
  wire signed [31:0] prod1_ext = {{16{prod1_r[15]}}, prod1_r};

  // ---- 快照：2× 27b，脉冲到拍直抄 + 清零（读出与下一行组喂数并行）----
  logic signed [26:0] snap0_r, snap1_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a_r <= '0; b_r <= '0; av_r <= 1'b0; bv_r <= 1'b0; pulse_r <= 1'b0;
      prod0_r <= '0; prod1_r <= '0; v_r <= 1'b0;
      acc0_r <= '0; acc1_r <= '0; snap0_r <= '0; snap1_r <= '0;
    end else begin
      a_r <= a_in;  b_r <= b_in;
      av_r <= av_in; bv_r <= bv_in;
      pulse_r <= pulse_in;
      prod0_r <= lo_e[15:0];
      prod1_r <= hi_e[15:0];
      v_r     <= v_p;
      if (pulse_in) begin
        // 快照拍：acc 已含本组全部积（操作数 ≤ t_p-6），下一组积还在流水线上
        snap0_r <= acc0_r[26:0];
        snap1_r <= acc1_r[26:0];
        acc0_r  <= '0;
        acc1_r  <= '0;
      end else if (clr) begin
        acc0_r <= '0; acc1_r <= '0;
      end else if (v_r) begin
        acc0_r <= acc0_r + prod0_ext;
        acc1_r <= acc1_r + prod1_ext;
      end
    end
  end
  assign a_out = a_r;    assign av_out = av_r;
  assign b_out = b_r;    assign bv_out = bv_r;
  assign pulse_out = pulse_r;
  assign acc0 = acc0_r;  assign acc1 = acc1_r;
  assign snap0 = snap0_r; assign snap1 = snap1_r;
endmodule
`endif
