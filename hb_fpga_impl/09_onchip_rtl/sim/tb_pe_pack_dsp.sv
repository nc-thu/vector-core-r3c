// ============================================================================
// tb_pe_pack_dsp.sv — packed INT8 PE（DSP48E2 原语版）位精确对拍台
//   DUT: 1 × ae_pe_pack_dsp（手工例化 DSP48E2，win=P 寄存器，Z mux 翻窗/清零）
//   REF: 2 × ae_pe（原网单 MAC PE），同 a0/a1/b/en 喂入
//   用例：随机 tile（K=1..4096、稀疏有效 80%）+ 定向极值（全 −128 的 +2^26、
//         混符号窗口 floor 修复、K=1..5 残窗、b=0）
//   断言：每 tile flush 后 acc0/acc1 与两 REF 逐位一致；每拍直通链（a0/a1/b 东
//         南传递）与 REF 一致。对比线网位宽与 tb_pe_pack.sv 精确一致（28b 有
//         符号，符号扩展在对比式里显式做）。
// 用法: iverilog -g2012 -o d.vvp tb_pe_pack_dsp.sv ../rtl/ae_pe.sv \
//                    ../rtl/ae_pe_pack_dsp.sv \
//                    "D:/software/Vivado/2021.2/data/verilog/src/unisims/DSP48E2.v" \
//                    glbl.v && vvp d.vvp
// ============================================================================
`timescale 1ns/1ps
module tb_pe_pack_dsp;
  reg clk = 1'b0, rst_n = 1'b0;
  always #5 clk = ~clk;

  // ---------------- 驱动（DUT 与 REF 共享） ----------------
  reg               av, bv, flush;
  reg signed [7:0]  a0, a1, b;

  // ---------------- DUT ----------------
  wire signed [7:0] d_a0o, d_a1o, d_bo;
  wire              d_avo, d_bvo;
  wire signed [27:0] d_acc0, d_acc1;   // 28b：符号扩展在对比式里显式做
  ae_pe_pack_dsp u_dut (
    .clk(clk), .rst_n(rst_n), .clr(clr), .flush(flush),
    .av_in(av), .bv_in(bv), .a0_in(a0), .a1_in(a1), .b_in(b),
    .av_out(d_avo), .bv_out(d_bvo), .a0_out(d_a0o), .a1_out(d_a1o), .b_out(d_bo),
    .acc0(d_acc0), .acc1(d_acc1)
  );

  // ---------------- REF：两个独立 ae_pe ----------------
  wire signed [7:0] r0_ao, r0_bo, r1_ao, r1_bo;
  wire              r0_avo, r0_bvo, r1_avo, r1_bvo;
  wire signed [31:0] r0_acc, r1_acc;
  ae_pe u_r0 (.clk(clk), .rst_n(rst_n), .clr(clr),
    .av_in(av), .bv_in(bv), .a_in(a0), .b_in(b),
    .av_out(r0_avo), .bv_out(r0_bvo), .a_out(r0_ao), .b_out(r0_bo), .acc(r0_acc));
  ae_pe u_r1 (.clk(clk), .rst_n(rst_n), .clr(clr),
    .av_in(av), .bv_in(bv), .a_in(a1), .b_in(b),
    .av_out(r1_avo), .bv_out(r1_bvo), .a_out(r1_ao), .b_out(r1_bo), .acc(r1_acc));

  reg clr;
  integer err, tiles, kcnt, i, k, mode, K;
  integer lcg;

  function integer rnd;   // 确定性 LCG（跨运行可复现）
    input integer dummy;
    begin
      lcg = lcg * 1103515245 + 12345;
      rnd = lcg & 32'h7FFFFFFF;
    end
  endfunction

  function signed [7:0] rnd8;
    input integer dummy;
    begin
      rnd8 = rnd(0) & 8'hFF;   // 低 8 位原样当有符号
    end
  endfunction

  // 直通链逐拍对拍（数据/有效都是 1 拍延迟）
  task check_links;
    begin
      if (d_a0o !== r0_ao || d_a1o !== r1_ao || d_bo !== r0_bo ||
          d_avo !== r0_avo || d_bvo !== r0_bvo) begin
        err = err + 1;
        if (err <= 20)
          $display("[LINK-FAIL] t=%0t a0 %0d/%0d a1 %0d/%0d b %0d/%0d",
                   $time, d_a0o, r0_ao, d_a1o, r1_ao, d_bo, r0_bo);
      end
    end
  endtask

  // 一个 tile：clr → K 个有效积（带随机间隙）→ flush → 对拍
  task run_tile(input integer KK, input integer mmode);
    integer en_cnt;
    begin
      @(negedge clk); clr = 1'b1; av = 1'b0; bv = 1'b0; flush = 1'b0;
      @(negedge clk); clr = 1'b0;
      en_cnt = 0;
      while (en_cnt < KK) begin
        if (mmode == 0 && (rnd(0) % 10) < 2) begin
          av = 1'b0; bv = 1'b0;            // 间隙：稀疏有效
          a0 = rnd8(0); a1 = rnd8(0); b = rnd8(0);
        end else begin
          av = 1'b1; bv = 1'b1; en_cnt = en_cnt + 1;
          case (mmode)
            0: begin a0 = rnd8(0); a1 = rnd8(0); b = rnd8(0); end
            1: begin a0 = -8'sd128; a1 = -8'sd128; b = -8'sd128; end // +2^26
            2: begin a0 = -8'sd128; a1 = -8'sd128; b =  8'sd127; end // 极负
            3: begin a0 = -8'sd128; a1 =  8'sd127; b = -8'sd128; end // 混符号窗口
            4: begin a0 =  8'sd127; a1 = -8'sd128; b =  8'sd127; end // 混符号窗口
            5: begin a0 = rnd8(0); a1 = rnd8(0); b =  8'sd0; end     // b=0
            6: begin a0 = -8'sd1;  a1 =  8'sd1;  b = -8'sd1;  end    // ±1 长累加
            default: begin a0 = rnd8(0); a1 = rnd8(0); b = rnd8(0); end
          endcase
        end
        @(negedge clk);
        check_links();
      end
      av = 1'b0; bv = 1'b0;
      @(negedge clk);                       // 让最后一个积落进 win
      flush = 1'b1;
      @(negedge clk);
      flush = 1'b0;
      @(negedge clk);
      // DUT 28b 累加器符号扩展到 32b 对拍（合法域 |acc| ≤ 2^26，两口径等价）
      if ({{4{d_acc0[27]}}, d_acc0} !== r0_acc || {{4{d_acc1[27]}}, d_acc1} !== r1_acc) begin
        err = err + 1;
        if (err <= 20)
          $display("[ACC-FAIL] tile=%0d K=%0d mode=%0d acc0 %0d/%0d acc1 %0d/%0d",
                   tiles, KK, mmode, d_acc0, r0_acc, d_acc1, r1_acc);
      end
      tiles = tiles + 1;
    end
  endtask

  initial begin
    lcg = 32'h3C6E_F35B; err = 0; tiles = 0;
    av = 0; bv = 0; flush = 0; clr = 0; a0 = 0; a1 = 0; b = 0;
    repeat (3) @(negedge clk);
    rst_n = 1'b1;
    repeat (2) @(negedge clk);

    // ---- 定向：窗口边界与极值 ----
    for (k = 1; k <= 8; k = k + 1) run_tile(k, 0);
    run_tile(4096, 1);   // 全 (−128)·(−128)：acc = +2^26（32b 回绕口径）
    run_tile(4096, 2);   // 全 (−128)·(+127)：极负
    run_tile(4096, 3);   // 每窗口混符号（floor 修复双向）
    run_tile(4096, 4);
    run_tile(1000, 5);   // b=0
    run_tile(4096, 6);   // ±1：最小步长长累加
    run_tile(4095, 6);   // 非整窗口长 K
    run_tile(4093, 3);

    // ---- 随机：小 K（残窗概率高） ----
    for (i = 0; i < 2000; i = i + 1) run_tile(1 + (rnd(0) % 64), 0);
    // ---- 随机：大 K ----
    for (i = 0; i < 200; i = i + 1) run_tile(1 + (rnd(0) % 4096), 0);

    if (err == 0)
      $display("TB_PE_PACK_DSP PASS (%0d tiles bit-identical to 2x ae_pe)", tiles);
    else
      $display("TB_PE_PACK_DSP FAIL (err=%0d / %0d tiles)", err, tiles);
    $finish;
  end
endmodule
