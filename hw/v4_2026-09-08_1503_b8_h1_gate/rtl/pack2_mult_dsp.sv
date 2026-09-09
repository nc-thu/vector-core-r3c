// pack2_mult_dsp.sv — 2-lane 打包 INT8×INT8 乘法核（显式 DSP48E2）
// ----------------------------------------------------------------------------
// 数学（offset packing，本仓库 R3C ae_pe_pack_dsp 已验证的同族方案，改为
// 单拍单积、8-bit gap / 16-bit field 版本）：
//   w̃ = w + 128（符号位取反，纯线网）
//   A[26:0] = {3'b000, w̃1[7:0], 8'b0, w̃0[7:0]}   （w̃1 在 [23:16]，2^16 对齐）
//   B[17:0] = 符号扩展的共享操作数 a
//   P = (w̃1·2^16 + w̃0)·a
//   低场 P[15:0]（有符号 16b）= w̃0·a
//   高场 P[31:16]（有符号 16b）= w̃1·a − borrow，borrow = P[15]
//   抽取校正（−128·a 偏置）：
//     prod0 = P[15:0]           − (a <<< 7) = w0·a
//     prod1 = P[31:16] + P[15]  − (a <<< 7) = w1·a
//   两个 lane 的积范围 [−16384, +16384]，16b 有符号可容。
//
// 流水：S0（本模块输入寄存）→ [MREG] → [PREG] → P 输出。
//   M_REG/P_REG 可参数化；P 自输入起延迟 1+M_REG+P_REG 拍可用。
//   a/v/row 做等深延迟管线，与 P 同拍输出（a_p 即校正所需的共享操作数）。
//
// DSP 配置（参数/管脚清单核对自 hb_fpga_impl/22_r3c_rtl/rtl/ae_pe_pack_dsp.sv
// 踩平的坑）：OPMODE=9'b000000101（X=U、Y=V、Z=0、W=0，纯乘法）；乘法输出
// Booth 部分和只能走 X/Y 口；P 无异步复位——本核 P 只承载无状态乘积，
// 复位后首个 start 周期重建语义即可。CE 全 1、RST 全 0。
`ifndef PACK2_MULT_DSP_SV
`define PACK2_MULT_DSP_SV
module pack2_mult_dsp #(
  parameter int M_REG = 1,    // DSP48E2 MREG
  parameter int P_REG = 1     // DSP48E2 PREG
)(
  input  logic              clk,
  input  logic              v_in,     // 本拍积有效
  input  logic              row_in,   // 0: a0 行 / 1: a1 行
  input  logic signed [7:0] a_in,     // 共享行激活（B 口）
  input  logic signed [7:0] w0_in,    // 列 0 权重（低 lane）
  input  logic signed [7:0] w1_in,    // 列 1 权重（高 lane）
  output logic [47:0]       P,        // DSP P 输出（PREG 后）
  output logic              v_p,      // 与 P 对齐的有效标志
  output logic              row_p,
  output logic signed [7:0] a_p       // 与 P 对齐的共享操作数（校正用）
);
  localparam int LAT = 1 + M_REG + P_REG;   // 输入 → P 的总延迟

  // ---- S0 输入寄存 ----
  logic signed [7:0] a_s0, w0_s0, w1_s0;
  logic              v_s0, row_s0;
  always_ff @(posedge clk) begin
    a_s0 <= a_in;  w0_s0 <= w0_in;  w1_s0 <= w1_in;
    v_s0 <= v_in;  row_s0 <= row_in;
  end

  // ---- 打包（纯线网） ----
  wire [7:0] wt0 = {~w0_s0[7], w0_s0[6:0]};   // w̃0 = w0 + 128
  wire [7:0] wt1 = {~w1_s0[7], w1_s0[6:0]};   // w̃1 = w1 + 128
  wire [26:0] ap = {3'b000, wt1, 8'b0, wt0};  // w̃1·2^16 + w̃0
  wire [17:0] bx = {{10{a_s0[7]}}, a_s0};

  // ---- DSP48E2（显式原语） ----
  wire [47:0] p_out;
`ifdef VERILATOR
  // 仿真器不带 AMD UNISIM。这里只在 VERILATOR 宏打开时使用与 MREG/PREG
  // 等延迟的位精确乘法模型；Vivado 和 Icarus+UNISIM 都走下面的真实原语。
  wire signed [29:0] sim_a = {3'b000, ap};
  wire signed [17:0] sim_b = bx;
  wire signed [47:0] sim_m = sim_a * sim_b;
  logic signed [47:0] sim_m_r, sim_p_r;
  generate
    if (M_REG && P_REG) begin : g_sim_mp
      always_ff @(posedge clk) begin
        sim_m_r <= sim_m;
        sim_p_r <= sim_m_r;
      end
      assign p_out = sim_p_r;
    end else if (M_REG || P_REG) begin : g_sim_one
      always_ff @(posedge clk) sim_p_r <= sim_m;
      assign p_out = sim_p_r;
    end else begin : g_sim_zero
      assign p_out = sim_m;
    end
  endgenerate
`else
  DSP48E2 #(
    .ACASCREG            (0),
    .ADREG               (0),
    .ALUMODEREG          (0),
    .AMULTSEL            ("A"),
    .AREG                (0),
    .AUTORESET_PATDET    ("NO_RESET"),
    .AUTORESET_PRIORITY  ("RESET"),
    .A_INPUT             ("DIRECT"),
    .BCASCREG            (0),
    .BMULTSEL            ("B"),
    .BREG                (0),
    .B_INPUT             ("DIRECT"),
    .CARRYINREG          (0),
    .CARRYINSELREG       (0),
    .CREG                (0),
    .DREG                (0),
    .INMODEREG           (0),
    .IS_ALUMODE_INVERTED (4'b0000),
    .IS_CARRYIN_INVERTED (1'b0),
    .IS_CLK_INVERTED     (1'b0),
    .IS_INMODE_INVERTED  (5'b00000),
    .IS_OPMODE_INVERTED  (9'b000000000),
    .IS_RSTALLCARRYIN_INVERTED (1'b0),
    .IS_RSTALUMODE_INVERTED    (1'b0),
    .IS_RSTA_INVERTED          (1'b0),
    .IS_RSTB_INVERTED          (1'b0),
    .IS_RSTCTRL_INVERTED       (1'b0),
    .IS_RSTC_INVERTED          (1'b0),
    .IS_RSTD_INVERTED          (1'b0),
    .IS_RSTINMODE_INVERTED     (1'b0),
    .IS_RSTM_INVERTED          (1'b0),
    .IS_RSTP_INVERTED          (1'b0),
    .MASK                (48'h3FFFFFFFFFFF),
    .MREG                (M_REG),
    .OPMODEREG           (0),
    .PATTERN             (48'h000000000000),
    .PREADDINSEL         ("A"),
    .PREG                (P_REG),
    .RND                 (48'h000000000000),
    .SEL_MASK            ("MASK"),
    .SEL_PATTERN         ("PATTERN"),
    .USE_MULT            ("MULTIPLY"),
    .USE_PATTERN_DETECT  ("NO_PATDET"),
    .USE_SIMD            ("ONE48"),
    .USE_WIDEXOR         ("FALSE"),
    .XORSIMD             ("XOR24_48_96")
  ) u_dsp (
    .CLK          (clk),
    .A            ({3'b0, ap}),
    .B            (bx),
    .C            (48'b0),
    .D            (27'b0),
    .ACIN         (30'b0),
    .BCIN         (18'b0),
    .PCIN         (48'b0),
    .ALUMODE      (4'b0000),
    .INMODE       (5'b00000),
    .OPMODE       (9'b000000101),   // X=U, Y=V, Z=0, W=0（纯乘法）
    .CARRYIN      (1'b0),
    .CARRYINSEL   (3'b000),
    .CARRYCASCIN  (1'b0),
    .MULTSIGNIN   (1'b0),
    .CEA1         (1'b1),
    .CEA2         (1'b1),
    .CEAD         (1'b1),
    .CEALUMODE    (1'b1),
    .CEB1         (1'b1),
    .CEB2         (1'b1),
    .CEC          (1'b1),
    .CECARRYIN    (1'b1),
    .CECTRL       (1'b1),
    .CED          (1'b1),
    .CEINMODE     (1'b1),
    .CEM          (1'b1),
    .CEP          (1'b1),
    .RSTA         (1'b0),
    .RSTALLCARRYIN(1'b0),
    .RSTALUMODE   (1'b0),
    .RSTB         (1'b0),
    .RSTC         (1'b0),
    .RSTCTRL      (1'b0),
    .RSTD         (1'b0),
    .RSTINMODE    (1'b0),
    .RSTM         (1'b0),
    .RSTP         (1'b0),
    .P            (p_out)
  );
`endif
  assign P = p_out;

  // ---- a/v/row 等深延迟（LAT−1 级，S0 已算 1 级） ----
  logic signed [7:0] a_d [LAT-2:0];
  logic              v_d [LAT-2:0];
  logic              r_d [LAT-2:0];
  generate
    if (LAT >= 2) begin : g_pipe
      integer gi;
      always_ff @(posedge clk) begin
        a_d[0] <= a_s0;  v_d[0] <= v_s0;  r_d[0] <= row_s0;
        for (gi = 1; gi < LAT-1; gi = gi + 1) begin
          a_d[gi] <= a_d[gi-1];  v_d[gi] <= v_d[gi-1];  r_d[gi] <= r_d[gi-1];
        end
      end
      assign a_p  = a_d[LAT-2];
      assign v_p  = v_d[LAT-2];
      assign row_p= r_d[LAT-2];
    end else begin : g_nopipe
      assign a_p  = a_s0;
      assign v_p  = v_s0;
      assign row_p= row_s0;
    end
  endgenerate
endmodule
`endif
