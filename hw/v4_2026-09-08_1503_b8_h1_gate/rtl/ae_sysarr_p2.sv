// ae_sysarr_p2.sv — 16×PCOLS Pack2 脉动阵列（B8×R3C 集成 H1 门）
// ----------------------------------------------------------------------------
// 与 ae_sysarr（R3C）同构的三处扩展：
//   1. 物理列 PCOLS，每列 16b B 链（{w1,w0}）—— 一物理列 = 2 逻辑列。
//   2. 快照读出 acc_row 按逻辑列展平：逻辑列 ℓ = 2j+lane（lane0=w0/lane1=w1），
//      总宽 2·PCOLS·32b，drain_row 选行、27b 符号扩展回 32b（requant 口径不变）。
//   3. 脉冲边缘偏斜网与 A 数据网同构（行 i 延迟 i 拍）——不变；
//      脉冲发射滞后由 ae_gemm_p2 的 PULSE_DLY 延迟线承担（阵列本身无感）。
// 边缘偏斜/波前汇合语义与 ae_sysarr 逐条一致：行 i 延迟 i 拍、列 j 延迟 j 拍，
// 波前在 PE(i,j) 汇合于 t=k+i+j，valid 随数据经 PE 链传播，en = av_in & bv_in。
`ifndef AE_SYSARR_P2_SV
`define AE_SYSARR_P2_SV
module ae_sysarr_p2 #(
  parameter int ROWS  = 16,
  parameter int PCOLS = 4            // 物理列数（逻辑列 = 2×PCOLS）
)(
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    clr,
  input  logic                    feed_vld,
  input  logic                    feed_pulse,  // 已含 PULSE_DLY 滞后（ae_gemm_p2 延迟线后）
  input  logic [ROWS*8-1:0]       a_feed,      // [i*8 +: 8] = a[i][k]
  input  logic [PCOLS*16-1:0]     b_feed,      // [j*16 +: 16] = {b[k][2j+1], b[k][2j]}
  input  logic [3:0]              drain_row,
  output logic [PCOLS*2*32-1:0]   acc_row      // [ℓ*32 +: 32] = 逻辑列 ℓ 快照（符号扩展）
);
  localparam int LC = PCOLS * 2;    // 逻辑列数

  logic signed [7:0] a_f [0:ROWS-1];
  logic [15:0]       b_f [0:PCOLS-1];
  always_comb begin
    for (int i = 0; i < ROWS; i++)  a_f[i] = a_feed[i*8 +: 8];
    for (int j = 0; j < PCOLS; j++) b_f[j] = b_feed[j*16 +: 16];
  end

  // ---- 边缘偏斜：行 i 延迟 i 拍、列 j 延迟 j 拍（同 ae_sysarr）----
  logic signed [7:0] a_skew [0:ROWS-1];
  logic              a_v    [0:ROWS-1];
  logic [15:0]       b_skew [0:PCOLS-1];
  logic              b_v    [0:PCOLS-1];

  logic signed [7:0] adly [0:ROWS-1][0:ROWS-1];
  logic              avdly[0:ROWS-1][0:ROWS-1];
  always_ff @(posedge clk) begin
    for (int i = 0; i < ROWS; i++) begin
      adly[0][i] <= a_f[i];
      avdly[0][i] <= feed_vld;
      for (int s = 1; s < ROWS; s++) begin
        if (s <= i) begin
          adly[s][i] <= adly[s-1][i];
          avdly[s][i] <= avdly[s-1][i];
        end
      end
    end
  end
  logic [15:0] bdly [0:PCOLS-1][0:PCOLS-1];
  logic        bvdly[0:PCOLS-1][0:PCOLS-1];
  always_ff @(posedge clk) begin
    for (int j = 0; j < PCOLS; j++) begin
      bdly[0][j] <= b_f[j];
      bvdly[0][j] <= feed_vld;
      for (int s = 1; s < PCOLS; s++) begin
        if (s <= j) begin
          bdly[s][j] <= bdly[s-1][j];
          bvdly[s][j] <= bvdly[s-1][j];
        end
      end
    end
  end
  // R3C 末脉冲边缘偏斜（与 A 数据网同构的 1b 版，行 i 延迟 i 拍）
  logic apdly [0:ROWS-1][0:ROWS-1];
  logic a_pulse [0:ROWS-1];
  always_ff @(posedge clk) begin
    for (int i = 0; i < ROWS; i++) begin
      apdly[0][i] <= feed_pulse;
      for (int s = 1; s < ROWS; s++) begin
        if (s <= i) apdly[s][i] <= apdly[s-1][i];
      end
    end
  end
  always_comb begin
    for (int i = 0; i < ROWS; i++) begin
      a_skew[i]  = adly[i][i];
      a_v[i]     = avdly[i][i];
      a_pulse[i] = apdly[i][i];
    end
    for (int j = 0; j < PCOLS; j++) begin
      b_skew[j] = bdly[j][j];
      b_v[j]    = bvdly[j][j];
    end
  end

  // ---- PE 阵列 ----
  logic signed [7:0]  awire [0:ROWS-1][0:PCOLS-1];
  logic [15:0]        bwire [0:ROWS-1][0:PCOLS-1];
  logic               avwire[0:ROWS-1][0:PCOLS-1];
  logic               bvwire[0:ROWS-1][0:PCOLS-1];
  logic               pwire [0:ROWS-1][0:PCOLS-1];
  logic signed [26:0] snaps0[0:ROWS-1][0:PCOLS-1];
  logic signed [26:0] snaps1[0:ROWS-1][0:PCOLS-1];

  logic signed [7:0]  a_in_pe [0:ROWS-1][0:PCOLS-1];
  logic [15:0]        b_in_pe [0:ROWS-1][0:PCOLS-1];
  logic               av_in_pe[0:ROWS-1][0:PCOLS-1];
  logic               bv_in_pe[0:ROWS-1][0:PCOLS-1];
  logic               pl_in_pe[0:ROWS-1][0:PCOLS-1];

  always_comb begin
    for (int i = 0; i < ROWS; i++) begin
      a_in_pe [i][0] = a_skew[i];
      av_in_pe[i][0] = a_v[i];
      pl_in_pe[i][0] = a_pulse[i];
      for (int j = 1; j < PCOLS; j++) begin
        a_in_pe [i][j] = awire [i][j-1];
        av_in_pe[i][j] = avwire[i][j-1];
        pl_in_pe[i][j] = pwire [i][j-1];
      end
    end
    for (int j = 0; j < PCOLS; j++) begin
      b_in_pe [0][j] = b_skew[j];
      bv_in_pe[0][j] = b_v[j];
      for (int i = 1; i < ROWS; i++) begin
        b_in_pe [i][j] = bwire [i-1][j];
        bv_in_pe[i][j] = bvwire[i-1][j];
      end
    end
  end

  generate
  for (genvar gi = 0; gi < ROWS; gi++) begin : g_row
    for (genvar gj = 0; gj < PCOLS; gj++) begin : g_col
      ae_pe_p2 u_pe (
        .clk(clk), .rst_n(rst_n),
        .clr   (clr),
        .pulse_in (pl_in_pe[gi][gj]),
        .av_in (av_in_pe[gi][gj]),
        .bv_in (bv_in_pe[gi][gj]),
        .a_in  (a_in_pe[gi][gj]),
        .b_in  (b_in_pe[gi][gj]),
        .av_out(avwire[gi][gj]),
        .bv_out(bvwire[gi][gj]),
        .pulse_out(pwire[gi][gj]),
        .a_out (awire[gi][gj]),
        .b_out (bwire[gi][gj]),
        .acc0  (), .acc1  (),
        .snap0 (snaps0[gi][gj]),
        .snap1 (snaps1[gi][gj])
      );
    end
  end
  endgenerate

  // 逻辑列读出：ℓ = 2j + lane；27b 符号扩展回 32b（requant 只吃低 27b）
  always_comb begin
    for (int j = 0; j < PCOLS; j++) begin
      acc_row[(2*j)  *32 +: 32]  = {{5{snaps0[drain_row][j][26]}}, snaps0[drain_row][j]};
      acc_row[(2*j+1)*32 +: 32]  = {{5{snaps1[drain_row][j][26]}}, snaps1[drain_row][j]};
    end
  end
endmodule
`endif
