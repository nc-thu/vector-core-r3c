// ae_sysarr.sv — 16xCOLS 输出驻留 2D 脉冲阵列（16*COLS PE = 16*COLS DSP48E2）
// 控制器送 k 切片：a_feed[16]（CTX lane 直出）+ b_feed[COLS]（WRAM lane 直出）。
// 阵列内部脉冲偏斜：行 i 延迟 i 拍、列 j 延迟 j 拍，波前在 PE(i,j) 汇合于 t=k+i+j，
// valid 随数据经 PE 链逐级传播，en = av_in & bv_in。k 切片不必背靠背（稀疏有效）。
// done 语义 = 「最后一个 feed 脉冲已完全传播且阵列空闲」单拍脉冲
//   （vsr 空且 active 置位 -> 下拍清 active；不能用电平沿检测，长喂入会提前触发）。
`ifndef AE_SYSARR_SV
`define AE_SYSARR_SV
module ae_sysarr #(
  parameter int ROWS = 16,
  parameter int COLS = 12
)(
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    clr,
  input  logic                    feed_vld,
  input  logic [ROWS*8-1:0]       a_feed,   // [i*8 +: 8] = a[i][k]
  input  logic [COLS*8-1:0]       b_feed,   // [j*8 +: 8] = b[k][j]
  output logic                    busy,
  output logic                    done,
  input  logic [3:0]              drain_row,
  output logic [COLS*32-1:0]      acc_row   // [j*32 +: 32] = PE(drain_row, j) 累加
);
  localparam int DEPTH = ROWS + COLS + 4;

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
  always_comb begin
    for (int i = 0; i < ROWS; i++) begin
      a_skew[i] = adly[i][i];
      a_v[i]    = avdly[i][i];
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
  logic signed [31:0] accs  [0:ROWS-1][0:COLS-1];
  // 边缘注入网络（边界拆分写法，避免 -1 索引）
  logic signed [7:0]  a_in_pe [0:ROWS-1][0:COLS-1];
  logic signed [7:0]  b_in_pe [0:ROWS-1][0:COLS-1];
  logic               av_in_pe[0:ROWS-1][0:COLS-1];
  logic               bv_in_pe[0:ROWS-1][0:COLS-1];

  always_comb begin
    for (int i = 0; i < ROWS; i++) begin
      a_in_pe [i][0] = a_skew[i];
      av_in_pe[i][0] = a_v[i];
      for (int j = 1; j < COLS; j++) begin
        a_in_pe [i][j] = awire [i][j-1];
        av_in_pe[i][j] = avwire[i][j-1];
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
        .av_in (av_in_pe[gi][gj]),
        .bv_in (bv_in_pe[gi][gj]),
        .a_in  (a_in_pe[gi][gj]),
        .b_in  (b_in_pe[gi][gj]),
        .av_out(avwire[gi][gj]),
        .bv_out(bvwire[gi][gj]),
        .a_out (awire[gi][gj]),
        .b_out (bwire[gi][gj]),
        .acc   (accs[gi][gj])
      );
    end
  end
  endgenerate

  always_comb begin
    for (int j = 0; j < COLS; j++) acc_row[j*32 +: 32] = accs[drain_row][j];
  end

  // ---- busy/done：active 标志 + vsr 全空检测 ----
  logic [DEPTH-1:0] vsr;
  logic             active;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      vsr <= '0; active <= 1'b0;
    end else if (clr) begin
      vsr <= '0; active <= 1'b0;
    end else begin
      vsr <= {vsr[DEPTH-2:0], feed_vld};
      if (feed_vld)          active <= 1'b1;
      else if (~|vsr)        active <= 1'b0;
    end
  end
  assign busy = active | |vsr;
  assign done = active & ~|vsr & ~feed_vld;
endmodule
`endif
