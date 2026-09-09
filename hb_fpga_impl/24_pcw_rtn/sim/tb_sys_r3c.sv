// tb_sys_r3c.sv — R3C 定向对拍：ae_sysarr 快照/末脉冲/读出链
// 喂 k 个切片（拍 0..k-1 呈现，feed_vld=1），拍 k 呈现 feed_pulse，
// 之后逐行 drain 读 acc_row，与黄金 Σ_κ A[i][κ]·B[κ][j] 对拍。
// 同时验证：脉冲拍后立即喂下一组（背靠背）不污染快照。
`timescale 1ns/1ps
module tb_sys_r3c;
  localparam int ROWS = 16, COLS = 4, K = 5;
  reg clk = 0, rst_n = 0;
  always #5 clk = ~clk;

  reg                     feed_vld = 0, feed_pulse = 0, clr = 0;
  reg signed [ROWS*8-1:0] a_feed = 0;
  reg signed [COLS*8-1:0] b_feed = 0;
  reg [3:0]               drain_row = 0;
  wire [COLS*32-1:0]      acc_row;

  // 黄金
  integer A [0:ROWS-1][0:K-1];
  integer B [0:K-1][0:COLS-1];
  integer gold [0:ROWS-1][0:COLS-1];
  integer gold2 [0:ROWS-1][0:COLS-1];
  integer i, j, kk, err, grp;

  ae_sysarr #(.ROWS(ROWS), .COLS(COLS)) dut (
    .clk(clk), .rst_n(rst_n), .clr(clr),
    .feed_vld(feed_vld), .feed_pulse(feed_pulse),
    .a_feed(a_feed), .b_feed(b_feed),
    .drain_row(drain_row), .acc_row(acc_row)
  );

  task run_group(input integer gsel);
    begin
      // 拍 0..K-1：呈现切片
      for (kk = 0; kk < K; kk = kk + 1) begin
        @(negedge clk);
        feed_vld = 1;
        for (i = 0; i < ROWS; i = i + 1) begin
          A[i][kk] = ((gsel*37 + i*7 + kk*3) % 13) - 6;   // 有符号小值
          a_feed[i*8 +: 8] = A[i][kk][7:0];
        end
        for (j = 0; j < COLS; j = j + 1) begin
          B[kk][j] = ((gsel*29 + kk*5 + j*11) % 15) - 7;
          b_feed[j*8 +: 8] = B[kk][j][7:0];
        end
      end
      @(negedge clk); feed_vld = 0;      // 拍 K：空一拍（同 ae_gemm：脉冲与最后切片隔 1 拍）
      @(negedge clk); feed_pulse = 1;    // 拍 K+1：末脉冲
      @(negedge clk); feed_pulse = 0;
      for (i = 0; i < ROWS; i = i + 1)
        for (j = 0; j < COLS; j = j + 1) begin
          if (gsel == 0) gold[i][j] = 0; else gold2[i][j] = 0;
          for (kk = 0; kk < K; kk = kk + 1) begin
            if (gsel == 0) gold[i][j] = gold[i][j] + A[i][kk]*B[kk][j];
            else           gold2[i][j] = gold2[i][j] + A[i][kk]*B[kk][j];
          end
        end
    end
  endtask

  integer got, exp_v;
  task check_drain(input integer gsel, input [127:0] tag);
    begin
      // 等快照全部落定：最后 PE(15,COLS-1) 的脉冲在拍 K+1+15+COLS-1 完成
      repeat (K + 2 + ROWS + COLS) @(negedge clk);
      for (i = 0; i < ROWS; i = i + 1) begin
        drain_row = i[3:0];
        @(negedge clk);   // 组合读出稳定
        for (j = 0; j < COLS; j = j + 1) begin
          got = $signed(acc_row[j*32 +: 32]);
          exp_v = (gsel == 0) ? gold[i][j] : gold2[i][j];
          if (got !== exp_v) begin
            err = err + 1;
            if (err <= 20)
              $display("[FAIL %0s] grp=%0d i=%0d j=%0d got=%0d exp=%0d t=%0t",
                       tag, gsel, i, j, got, exp_v, $time);
          end
        end
      end
    end
  endtask

  initial begin
    err = 0;
    repeat (3) @(negedge clk);
    rst_n = 1;
    repeat (2) @(negedge clk);

    // ---- 组 0：快照基本功能 ----
    run_group(0);
    check_drain(0, "G0");

    // ---- 组 1：清零后重新喂数，快照应更新为组 1 的值 ----
    @(negedge clk);
    run_group(1);
    check_drain(1, "G1");

    if (err == 0) $display("TB_SYS_R3C PASS");
    else           $display("TB_SYS_R3C FAIL (err=%0d)", err);
    $finish;
  end
endmodule
