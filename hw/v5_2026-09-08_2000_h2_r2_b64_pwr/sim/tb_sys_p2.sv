// tb_sys_p2.sv — Pack2 阵列定向对拍（H1 门）：快照/末脉冲/背靠背/窗口扫描
// ----------------------------------------------------------------------------
// T1（间隔式）：喂 k 切片 -> 拍 K+1+PD 发末脉冲 -> 等扫完 -> 逐行 drain，
//              与黄金 Σ_κ A[i][κ]·B[κ][ℓ]（ℓ=逻辑列）对拍。每组独立检查。
// T2（背靠背 kill 测试）：G 组按精确 k+2 周期连续喂数（下一组首切片在
//              K+3，脉冲 K+1+PD 在下一组喂数窗口内进链），只检查末组——
//              脉冲过晚 -> 末组积被清掉；过早 -> 前组尾巴污染末组。两种错
//              都会体现在末组对拍上。
// PD 扫描：+PD=PD_MIN..PD_MAX（plusarg 控制），期望窗口 {5,6} 全对
//              （实测：PD≤4 末积未落地；PD=7 脉冲拍丢弃下组首积；PD≥8 污染）。
// 判定输出：TB_SYS_P2 PASS_PD=<n> / FAIL_PD=<n> ...（每个 PD 一行）。
`timescale 1ns/1ps
module tb_sys_p2;
  localparam int ROWS = 16, PCOLS = 4, LC = 8;
  localparam int K = 5, GROUPS = 8;
  localparam int PD_MIN = 3, PD_MAX = 9;

  reg clk = 0, rst_n = 0;
  always #5 clk = ~clk;

  integer pd_arg;
  integer gap_arg;                       // T2 组间额外空隙（0 = 真背靠背）
  initial begin
    pd_arg = 5; gap_arg = 0;
    if ($value$plusargs("PD=%d", pd_arg)) ;
    if ($value$plusargs("GAPD=%d", gap_arg)) ;
  end

  reg                  feed_vld = 0, feed_pulse = 0, clr = 0;
  reg signed [ROWS*8-1:0]  a_feed = 0;
  reg [PCOLS*16-1:0]   b_feed = 0;
  reg [3:0]            drain_row_rep;         // H2 副本端口（PCOLS=4 → 1 份 4b）
  wire [LC*32-1:0]     acc_row;

  // 简单 LCG（跨仿真器确定）
  integer seed_r;
  function integer lcg(input integer s);
    begin lcg = (s * 1103515245 + 12345) & 16'h7fff; end
  endfunction

  integer A [0:ROWS-1][0:K-1];
  integer B [0:K-1][0:LC-1];          // 按逻辑列存
  integer gold [0:ROWS-1][0:LC-1];    // 末组黄金
  integer i, j, kk, l, err, pd, grp, cyc, t_err_tot;
  integer pass_cnt, fail_cnt;

  ae_sysarr_p2 #(.ROWS(ROWS), .PCOLS(PCOLS)) dut (
    .clk(clk), .rst_n(rst_n), .clr(clr),
    .feed_vld(feed_vld), .feed_pulse(feed_pulse),
    .a_feed(a_feed), .b_feed(b_feed),
    .drain_row_rep(drain_row_rep), .acc_row(acc_row)
  );

  // 供两组测试复用的数据生成（组号 g 决定随机序列）
  task gen_ab(input integer g);
    begin
      for (i = 0; i < ROWS; i = i + 1)
        for (kk = 0; kk < K; kk = kk + 1) begin
          seed_r = lcg(seed_r + g*131 + i*17 + kk*7);
          A[i][kk] = (seed_r % 13) - 6;
        end
      for (kk = 0; kk < K; kk = kk + 1)
        for (l = 0; l < LC; l = l + 1) begin
          seed_r = lcg(seed_r + g*77 + kk*5 + l*3);
          B[kk][l] = (seed_r % 15) - 7;
        end
    end
  endtask

  task run_pd(input integer pdsel);
    begin
      err = 0;
      // ================= T1：间隔式，逐组对拍（3 组）=================
      for (grp = 0; grp < 3; grp = grp + 1) begin
        gen_ab(grp);
        // 喂 K 切片
        for (kk = 0; kk < K; kk = kk + 1) begin
          @(negedge clk);
          feed_vld = 1;
          for (i = 0; i < ROWS; i = i + 1) a_feed[i*8 +: 8] = A[i][kk][7:0];
          for (j = 0; j < PCOLS; j = j + 1) begin
            b_feed[j*16 +: 8]  = B[kk][2*j][7:0];
            b_feed[j*16+8 +: 8] = B[kk][2*j+1][7:0];
          end
        end
        @(negedge clk); feed_vld = 0;
        // 末脉冲：R3C 在 K+1，这里 +pdsel
        repeat (pdsel) @(negedge clk);
        feed_pulse = 1;
        @(negedge clk); feed_pulse = 0;
        // 等快照落定（末 PE(15,PCOLS-1) 脉冲在 K+1+pd+15+PCOLS-1 完成）
        repeat (K + 2 + pdsel + ROWS + PCOLS) @(negedge clk);
        for (i = 0; i < ROWS; i = i + 1) begin
          drain_row_rep = i[3:0];
          @(negedge clk);
          for (l = 0; l < LC; l = l + 1) begin
            kk = 0; // 复用作累加游标
            gold[i][l] = 0;
            for (kk = 0; kk < K; kk = kk + 1)
              gold[i][l] = gold[i][l] + A[i][kk]*B[kk][l];
            if ($signed(acc_row[l*32 +: 32]) !== gold[i][l]) begin
              err = err + 1;
              if (err <= 8)
                $display("[T1 FAIL] pd=%0d grp=%0d i=%0d l=%0d got=%0d exp=%0d t=%0t",
                         pdsel, grp, i, l, $signed(acc_row[l*32 +: 32]), gold[i][l], $time);
            end
          end
        end
        @(negedge clk);
      end

      // ================= T2：背靠背 kill 测试（G 组，k+2 周期）=========
      // 预生成末组数据
      gen_ab(GROUPS + 100);
      for (i = 0; i < ROWS; i = i + 1)
        for (l = 0; l < LC; l = l + 1) begin
          gold[i][l] = 0;
          for (kk = 0; kk < K; kk = kk + 1)
            gold[i][l] = gold[i][l] + A[i][kk]*B[kk][l];
        end
      // 每组数据预存（组号 g）——直接在驱动时重算会破坏流水节拍
      begin : t2_drive
        integer Aall [0:GROUPS-1][0:ROWS-1][0:K-1];
        integer Ball [0:GROUPS-1][0:K-1][0:LC-1];
        integer g2, gp, s2;
        for (g2 = 0; g2 < GROUPS; g2 = g2 + 1) begin
          for (i = 0; i < ROWS; i = i + 1)
            for (kk = 0; kk < K; kk = kk + 1) begin
              s2 = lcg(seed_r + g2*911 + i*13 + kk*3);
              Aall[g2][i][kk] = (s2 % 13) - 6;
            end
          for (kk = 0; kk < K; kk = kk + 1)
            for (l = 0; l < LC; l = l + 1) begin
              s2 = lcg(seed_r + g2*577 + kk*11 + l*7);
              Ball[g2][kk][l] = (s2 % 15) - 7;
            end
        end
        // 末组黄金（用与驱动同一数组）
        for (i = 0; i < ROWS; i = i + 1)
          for (l = 0; l < LC; l = l + 1) begin
            gold[i][l] = 0;
            for (kk = 0; kk < K; kk = kk + 1)
              gold[i][l] = gold[i][l] + Aall[GROUPS-1][i][kk]*Ball[GROUPS-1][kk][l];
          end
        // 统一周期驱动：组 g 占 [g*(K+2+GAPD), (g+1)*(K+2+GAPD))；切片在组内拍 1..K，
        // 组 g 的脉冲在全局拍 g*(K+2+GAPD)+K+1+pd（pd>1 时落进后面组的喂数窗口——正是要测的）
        // 循环界必须盖过末组脉冲拍（否则末组快照停在上一组 = 全槽位假错）
        for (cyc = 0; cyc < GROUPS*(K+2+gap_arg) + K + 2 + pdsel; cyc = cyc + 1) begin
          g2 = cyc / (K+2+gap_arg);
          kk = cyc - g2*(K+2+gap_arg);  // 组内拍号
          feed_vld = (kk >= 1) && (kk <= K) && (g2 < GROUPS);
          feed_pulse = 0;
          for (gp = 0; gp < GROUPS; gp = gp + 1)
            if (cyc == gp*(K+2+gap_arg) + K+1 + pdsel) feed_pulse = 1;
          if (feed_vld) begin
            for (i = 0; i < ROWS; i = i + 1)
              a_feed[i*8 +: 8] = Aall[g2][i][kk-1][7:0];
            for (j = 0; j < PCOLS; j = j + 1) begin
              b_feed[j*16 +: 8]   = Ball[g2][kk-1][2*j][7:0];
              b_feed[j*16+8 +: 8] = Ball[g2][kk-1][2*j+1][7:0];
            end
          end
          @(negedge clk);
        end
        feed_vld = 0; feed_pulse = 0;
      end
      // 等末组脉冲扫完 + 裕量，再 drain 对拍
      repeat (K + 2 + pdsel + ROWS + PCOLS + 8) @(negedge clk);
      for (i = 0; i < ROWS; i = i + 1) begin
        drain_row_rep = i[3:0];
        @(negedge clk);
        for (l = 0; l < LC; l = l + 1) begin
          if ($signed(acc_row[l*32 +: 32]) !== gold[i][l]) begin
            err = err + 1;
            if (err <= 16)
              $display("[T2 FAIL] pd=%0d i=%0d l=%0d got=%0d exp=%0d t=%0t",
                       pdsel, i, l, $signed(acc_row[l*32 +: 32]), gold[i][l], $time);
          end
        end
      end
      @(negedge clk);
      if (err == 0) begin
        $display("TB_SYS_P2 PASS_PD=%0d", pdsel);
      end else begin
        $display("TB_SYS_P2 FAIL_PD=%0d err=%0d", pdsel, err);
      end
    end
  endtask

  initial begin
    t_err_tot = 0; seed_r = 16'h5eed; drain_row_rep = 4'd0;
    repeat (3) @(negedge clk);
    rst_n = 1;
    repeat (2) @(negedge clk);
    clr = 1; @(negedge clk); clr = 0;
    run_pd(pd_arg);
    $finish;
  end
endmodule
