// ae_sysarr.sv — 16xCOLS 输出驻留 2D 脉冲阵列（16*COLS PE = 16*COLS DSP48E2）
// 控制器送 k 切片：a_feed[16]（CTX lane 直出）+ b_feed[COLS]（WRAM lane 直出）。
// 阵列内部脉冲偏斜：行 i 延迟 i 拍、列 j 延迟 j 拍，波前在 PE(i,j) 汇合于 t=k+i+j，
// valid 随数据经 PE 链逐级传播，en = av_in & bv_in。k 切片不必背靠背（稀疏有效）。
// R3C 方案 C：
//   * feed_pulse 末脉冲走与 a_feed 完全同构的 1b 边缘偏斜网 + PE 内 A 链东传，
//     到 PE(i,j) 的拍 = 本行组最后一个部分和落进该 PE acc_r 的下一拍（见 ae_pe.sv）。
//   * acc_row 读口改从快照侧出：drain_row 选行、读 snaps（27b 符号扩展回 32b，
//     requant 只吃低 27b，与旧 32b acc 口径逐位一致）——读出与下一行组喂数并行。
//   * clr 退化为复位兜底（每行组的清零由末脉冲在 PE 内完成，不再依赖排空）。
//   * 旧 busy/done（vsr 波前排空检测）删除：读出时序由 ae_gemm 的脉冲延迟线对齐。
`ifndef AE_SYSARR_SV
`define AE_SYSARR_SV
module ae_sysarr #(
  parameter int ROWS = 16,
  parameter int COLS = 12
)(
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    clr,        // 复位兜底清零（描述符起点幂等发一次）
  input  logic                    feed_vld,
  input  logic                    feed_pulse, // R3C 末脉冲（最后一个 k 切片之后隔 1 拍）
  input  logic [ROWS*8-1:0]       a_feed,   // [i*8 +: 8] = a[i][k]
  input  logic [COLS*8-1:0]       b_feed,   // [j*8 +: 8] = b[k][j]
  input  logic [3:0]              drain_row,
  output logic [COLS*32-1:0]      acc_row   // [j*32 +: 32] = PE(drain_row, j) 快照（符号扩展）
);

  logic signed [7:0] a_f [0:ROWS-1];
  logic signed [7:0] b_f [0:COLS-1];
  always_comb begin
    for (int i = 0; i < ROWS; i++) a_f[i] = a_feed[i*8 +: 8];
    for (int j = 0; j < COLS; j++) b_f[j] = b_feed[j*8 +: 8];
  end

  // ---- 边缘偏斜：行 i 延迟 i 拍、列 j 延迟 j 拍 ----
  logic signed [7:0] a_skew [0:ROWS-1];
  logic              a_v    [0:ROWS-1];
  logic signed [7:0] b_skew [0:COLS-1];
  logic              b_v    [0:COLS-1];

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
  logic signed [7:0] bdly [0:COLS-1][0:COLS-1];
  logic              bvdly[0:COLS-1][0:COLS-1];
  always_ff @(posedge clk) begin
    for (int j = 0; j < COLS; j++) begin
      bdly[0][j] <= b_f[j];
      bvdly[0][j] <= feed_vld;
      for (int s = 1; s < COLS; s++) begin
        if (s <= j) begin
          bdly[s][j] <= bdly[s-1][j];
          bvdly[s][j] <= bvdly[s-1][j];
        end
      end
    end
  end
  // R3C 末脉冲边缘偏斜：与 a 数据网同构的 1b 版（行 i 延迟 i 拍），
  // 列向偏斜由 PE 内 pulse 链承担（与 A 数据链同构）
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
      a_skew[i] = adly[i][i];
      a_v[i]    = avdly[i][i];
      a_pulse[i] = apdly[i][i];
    end
    for (int j = 0; j < COLS; j++) begin
      b_skew[j] = bdly[j][j];
      b_v[j]    = bvdly[j][j];
    end
  end

  // ---- PE 阵列 ----
  logic signed [7:0]  awire [0:ROWS-1][0:COLS-1];
  logic signed [7:0]  bwire [0:ROWS-1][0:COLS-1];
  logic               avwire[0:ROWS-1][0:COLS-1];
  logic               bvwire[0:ROWS-1][0:COLS-1];
  logic               pwire[0:ROWS-1][0:COLS-1];   // R3C 末脉冲链（东传）
  logic signed [31:0] accs  [0:ROWS-1][0:COLS-1];
  logic signed [26:0] snaps [0:ROWS-1][0:COLS-1];  // R3C 快照
  // 边缘注入网络（边界拆分写法，避免 -1 索引）
  logic signed [7:0]  a_in_pe [0:ROWS-1][0:COLS-1];
  logic signed [7:0]  b_in_pe [0:ROWS-1][0:COLS-1];
  logic               av_in_pe[0:ROWS-1][0:COLS-1];
  logic               bv_in_pe[0:ROWS-1][0:COLS-1];
  logic               pl_in_pe[0:ROWS-1][0:COLS-1];

  always_comb begin
    for (int i = 0; i < ROWS; i++) begin
      a_in_pe [i][0] = a_skew[i];
      av_in_pe[i][0] = a_v[i];
      pl_in_pe[i][0] = a_pulse[i];
      for (int j = 1; j < COLS; j++) begin
        a_in_pe [i][j] = awire [i][j-1];
        av_in_pe[i][j] = avwire[i][j-1];
        pl_in_pe[i][j] = pwire [i][j-1];
      end
    end
    for (int j = 0; j < COLS; j++) begin
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
    for (genvar gj = 0; gj < COLS; gj++) begin : g_col
      ae_pe u_pe (
        .clk(clk), .rst_n(rst_n),
        .clr   (clr),
        .pulse_in (pl_in_pe[gi][gj]),   // R3C 末脉冲（A 链方向）
        .av_in (av_in_pe[gi][gj]),
        .bv_in (bv_in_pe[gi][gj]),
        .a_in  (a_in_pe[gi][gj]),
        .b_in  (b_in_pe[gi][gj]),
        .av_out(avwire[gi][gj]),
        .bv_out(bvwire[gi][gj]),
        .pulse_out(pwire[gi][gj]),
        .a_out (awire[gi][gj]),
        .b_out (bwire[gi][gj]),
        .acc   (accs[gi][gj]),
        .snap  (snaps[gi][gj])          // R3C 快照读出
      );
    end
  end
  endgenerate

  // R3C 读出口：drain_row 选行、从快照侧出（27b 符号扩展回 32b；
  // requant 只消费低 27b —— 与旧 acc_r[26:0] 逐位一致）
  always_comb begin
    for (int j = 0; j < COLS; j++)
      acc_row[j*32 +: 32] = {{5{snaps[drain_row][j][26]}}, snaps[drain_row][j]};
  end
endmodule
`endif
