// tb_gemm_p2.sv — ae_gemm_p2 引擎级位精确对拍（H1 门 E1b）
// ----------------------------------------------------------------------------
// 行为级 CTX（A 口广播读 1 拍回数 / B 口按 lane we 写）+ WRAM（1 拍回数），
// 4 个描述符覆盖：小 k（读出腿主导）、m/n_loc 尾巴、j0 偏移、转置写回、
// 负 rq_m。黄金 = INT8 GEMM -> x(≤27b) -> y = sat8((x·rq_m) >>> rq_s)。
// 每个描述符独立初始化 CTX/WRAM，done 后逐字（逐 lane）对拍，并打印
// start->done 周期数（行组周期 = max(k+2, 2+PULSE_DLY+1+PCOLS+64+wb) 观察）。
// 判定：TB_GEMM_P2 ALL PASS / FAIL（任一描述符任一 lane 错即 FAIL）。
`timescale 1ns/1ps
module tb_gemm_p2 #(
  parameter int PCOLS_TB = 4,     // -P/-G 可覆盖（PCOLS=48 全宽验证用）
  parameter int PULSE_DLY_TB = 5
);
  localparam int PCOLS = PCOLS_TB, LC = PCOLS_TB*2;
  localparam int CTX_WORDS = 4096, W_WORDS = 512;

  reg clk = 0, rst_n = 0;
  always #5 clk = ~clk;

  logic start = 0, busy, done;
  logic [15:0] m, n, n_loc, j0, k;
  logic [19:0] a_base, b_base, y_base;
  logic        y_tr;
  logic signed [15:0] rq_m;
  logic [7:0]  rq_s;
  logic [19:0] ctxa_addr;
  logic [16*8-1:0] ctxa_rd;
  logic        ctxb_we;
  logic [15:0] ctxb_welane;
  logic [19:0] ctxb_addr;
  logic [16*8-1:0] ctxb_wd;
  logic [11:0] w_addr;
  logic [PCOLS*16-1:0] w_rd;
  logic [31:0] mac_cnt;
  logic        wb_active;

  ae_gemm_p2 #(.PCOLS(PCOLS), .PULSE_DLY(PULSE_DLY_TB)) dut (
    .clk(clk), .rst_n(rst_n), .start(start), .busy(busy), .done(done),
    .m(m), .n(n), .n_loc(n_loc), .j0(j0), .k(k),
    .a_base(a_base), .b_base(b_base), .y_base(y_base), .y_tr(y_tr),
    .rq_m(rq_m), .rq_s(rq_s),
    .ctxa_addr(ctxa_addr), .ctxa_rdata(ctxa_rd),
    .ctxb_we(ctxb_we), .ctxb_welane(ctxb_welane),
    .ctxb_addr(ctxb_addr), .ctxb_wdata(ctxb_wd),
    .w_addr(w_addr), .w_rdata(w_rd),
    .mac_cnt(mac_cnt), .wb_active(wb_active)
  );

  // ---- 行为级 CTX：A 口 1 拍回数广播；B 口写按 lane we ----
  logic [16*8-1:0] ctx [0:CTX_WORDS-1];
  always_ff @(posedge clk) ctxa_rd <= ctx[ctxa_addr];
  always_ff @(posedge clk) begin
    if (ctxb_we)
      for (int L = 0; L < 16; L = L + 1)
        if (ctxb_welane[L]) ctx[ctxb_addr][L*8 +: 8] <= ctxb_wd[L*8 +: 8];
  end

  // ---- 行为级 WRAM：1 拍回数，每物理列 16b ----
  logic [PCOLS*16-1:0] wmem [0:W_WORDS-1];
  always_ff @(posedge clk) w_rd <= wmem[w_addr];

  // ---- LCG ----
  integer seed_r;
  function integer lcg(input integer s);
    begin lcg = (s * 1103515245 + 12345) & 16'h7fff; end
  endfunction

  // 黄金模型状态
  integer A [0:63][0:511];       // A[m][k]（m ≤ 64 覆盖）
  integer B [0:511][0:LC-1];     // B[k][l]（逻辑列）
  logic [16*8-1:0] gold_ctx [0:CTX_WORDS-1];
  integer err_total, i, l, kk, mm, t_i0, t_i1;
  longint acc;
  logic [7:0] y8;
  integer t0, cyc_cnt;

  // sat8((x·m) >>> s)：与 rq_v2 全精度口径一致（longint 防 42b 溢出）
  function [7:0] rq_l(input longint x, input integer mmul, input integer sh);
    longint t;
    begin
      t = x * mmul >>> sh;     // 算术右移（floor）
      if (t > 127)       rq_l = 8'd127;
      else if (t < -128) rq_l = 8'd128;   // -128 = 0x80
      else               rq_l = t[7:0];
    end
  endfunction

  // 跑一个描述符并对拍
  task run_desc(input integer dm, input integer dn, input integer dnl,
                input integer dj0, input integer dk, input integer dtr,
                input integer dmq, input integer dsh, input [127:0] tag);
    begin
      // 初始化 CTX/WRAM（每次独立）与黄金映像
      for (i = 0; i < CTX_WORDS; i = i + 1) begin
        seed_r = lcg(seed_r + i*7);
        t_i0 = (seed_r % 11) - 5;                 // 低 2 lane 有确定值即可
        t_i1 = (seed_r % 13) - 6;
        ctx[i] = {112'd0, t_i1[7:0], t_i0[7:0]};
        gold_ctx[i] = ctx[i];
      end
      a_base = 20'd100; b_base = 20'd30; y_base = 20'd2048;
      // A[m][k]：word = a_base + (m div 16)*dk + kk，lane = m mod 16
      for (mm = 0; mm < dm; mm = mm + 1)
        for (kk = 0; kk < dk; kk = kk + 1) begin
          seed_r = lcg(seed_r + mm*131 + kk*17);
          A[mm][kk] = (seed_r % 13) - 6;
        end
      for (kk = 0; kk < dk; kk = kk + 1)
        for (l = 0; l < LC; l = l + 1) begin
          seed_r = lcg(seed_r + kk*57 + l*3);
          B[kk][l] = (seed_r % 15) - 7;
        end
      // 回填行为存储（A 进 CTX lane；B 进 WRAM 物理列打包）
      for (mm = 0; mm < dm; mm = mm + 1)
        for (kk = 0; kk < dk; kk = kk + 1) begin
          ctx[a_base + (mm/16)*dk + kk][(mm%16)*8 +: 8] = A[mm][kk][7:0];
        end
      for (kk = 0; kk < dk; kk = kk + 1)
        for (l = 0; l < LC; l = l + 1)
          wmem[b_base + kk][ (l/2)*16 + (l%2)*8 +: 8 ] = B[kk][l][7:0];
      // gold_ctx 起点 = 回填后的 ctx
      for (i = 0; i < CTX_WORDS; i = i + 1) gold_ctx[i] = ctx[i];

      // ---- 黄金 INT8 GEMM + requant ----
      // normal: addr = y_base + (mt)*dn + dj0 + c，lane = m mod 16，行 = m
      // trans:  addr = y_base + (c/16)*M16 + m，lane = c mod 16
      for (mm = 0; mm < dm; mm = mm + 1)
        for (l = 0; l < dnl; l = l + 1) begin
          acc = 0;
          for (kk = 0; kk < dk; kk = kk + 1)
            acc = acc + A[mm][kk] * B[kk][l];
          y8 = rq_l(acc, dmq, dsh);
          if (dtr == 0) begin
            gold_ctx[y_base + (mm/16)*dn + dj0 + l][(mm%16)*8 +: 8] = y8;
          end else begin
            kk = dj0 + l;   // 全局列 c
            gold_ctx[y_base + (kk/16)*((dm+15)/16*16) + mm][(kk%16)*8 +: 8] = y8;
          end
        end

      // ---- 跑 DUT ----
      @(negedge clk);
      m = dm; n = dn; n_loc = dnl; j0 = dj0; k = dk; y_tr = dtr[0];
      rq_m = dmq; rq_s = dsh;
      start = 1;
      t0 = 0;
      @(negedge clk); start = 0; t0 = 1;
      while (!done && t0 < 200000) begin
        @(negedge clk); t0 = t0 + 1;
      end
      cyc_cnt = t0;
      @(negedge clk); @(negedge clk);

      // ---- 对拍（全 CTX 逐 lane）----
      for (i = 0; i < CTX_WORDS; i = i + 1)
        for (l = 0; l < 16; l = l + 1)
          if (ctx[i][l*8 +: 8] !== gold_ctx[i][l*8 +: 8]) begin
            err_total = err_total + 1;
            if (err_total <= 12)
              $display("[%0s FAIL] word=%0d lane=%0d got=%0d exp=%0d",
                       tag, i, l, $signed(ctx[i][l*8 +: 8]),
                       $signed(gold_ctx[i][l*8 +: 8]));
          end
      if (err_total == 0)
        $display("[%0s] cycles=%0d mac=%0d m=%0d n=%0d nl=%0d j0=%0d k=%0d tr=%0d",
                 tag, cyc_cnt, mac_cnt, dm, dn, dnl, dj0, dk, dtr);
      else
        $display("[%0s] err_total=%0d (cycles=%0d)", tag, err_total, cyc_cnt);
    end
  endtask

`ifdef DBG
  // 调试探针：脉冲事件 + drain 逐行首 lane 快照值（复现 D1 组 1 后 11 行为零）
  integer dbg_prev_drw;
  always @(posedge clk) if (rst_n) begin
    if (dut.feed_pulse_raw) $display("[DBG %0t] pulse_raw", $time);
    if (dut.feed_pulse)     $display("[DBG %0t] pulse_bus(进阵列)", $time);
    if (dut.st_r == 2 /*SR_DRN*/ && dut.drain_row != dbg_prev_drw)
      $display("[DBG %0t] drn row=%0d lane0=%0d", $time, dut.drain_row,
               $signed(dut.acc_row[31:0]));
    dbg_prev_drw <= dut.drain_row;
  end
`endif

  initial begin
    err_total = 0; seed_r = 16'hb00b;
    repeat (3) @(negedge clk);
    rst_n = 1;
    repeat (5) @(negedge clk);

`ifdef DBG
    run_desc(35, 8, 8, 0, 37, 0, 448, 8, "D1a-dbg");
    $finish;
`endif
    // D0：m=16 整组 / n_loc=8 全宽 / k=5（读出腿主导） / rq_m=+1.75(Q8.8=448) s=8
    run_desc(16, 8, 8, 0, 5,  0, 448, 8, "D0");
    // D1：m=35（3 行组，尾 3 行）/ n=22 -> 3 列组(8,8,6) / k=37 / 普通写回
    run_desc(35, 22, 8, 0, 37, 0, 448, 8, "D1a");
    run_desc(35, 22, 8, 8, 37, 0, 448, 8, "D1b");
    run_desc(35, 22, 6, 16, 37, 0, 448, 8, "D1c");
    // D2：m=19 / n=16 / n_loc=6 / j0=10（跨全局 16-lane 组）/ k=64 / 转置
    run_desc(19, 16, 6, 10, 64, 1, 384, 9, "D2");
    // D3：m=48 / n_loc=8 / k=100（喂数腿主导）/ rq_m 负（-0.5 = -128）
    run_desc(48, 8, 8, 0, 100, 0, -128, 9, "D3");
`ifdef WIDE
    // 全宽（PCOLS=48 -> LC=96）：24 套 requant + 47 列偏斜 + 跨 16-lane 组写回
    if (LC >= 96) begin
      run_desc(16, 96, 96, 0, 40, 0, 448, 8, "W0");   // 单组全宽
      run_desc(35, 96, 96, 0, 37, 0, 448, 8, "W1");   // 尾组 + k=37 应力组合
      run_desc(19, 96, 90, 6, 64, 1, 384, 9, "W2");   // j0 偏移 + 转置全宽
    end
`endif

    if (err_total == 0) $display("TB_GEMM_P2 ALL PASS");
    else                $display("TB_GEMM_P2 FAIL err=%0d", err_total);
    $finish;
  end
endmodule
