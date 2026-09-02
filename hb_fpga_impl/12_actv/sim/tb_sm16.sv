// tb_sm16.sv — ae_softmax (SM16) 孤立对拍：随机 S[m x n]
//   golden 向量 = 本目录 sm16_ctrl/s/gold.mem（历史生成脚本已失传，向量是孤本，勿删）
`timescale 1ns/1ps
module tb_sm16;
  localparam int CW = 2048;
  logic clk=0, rst_n=0;
  always #5 clk = ~clk;
  logic start, busy, done, causal;
  logic [19:0] s_base;
  logic [15:0] m_rows, n_cols;
  logic [19:0] raddr, waddr;
  logic [127:0] rdata, wdata;
  logic we;

  ae_softmax dut (.clk(clk), .rst_n(rst_n), .start(start),
    .s_base(s_base), .m_rows(m_rows), .n_cols(n_cols), .causal(causal),
    .ctx_raddr(raddr), .ctx_rdata(rdata), .ctx_we(we), .ctx_waddr(waddr),
    .ctx_wdata(wdata), .busy(busy), .done(done));

  // CTX 模型（与 ae_ctx_ram 同：1 拍读延迟，B 口写）
  logic [127:0] mem [0:CW-1];
  always_ff @(posedge clk) begin
    if (we) for (int b = 0; b < 16; b++) mem[waddr][b*8 +: 8] <= wdata[b*8 +: 8];
    rdata <= mem[raddr];
  end

  // 向量
  logic [7:0] smem [0:262143];      // S 平铺（golden 同序）
  logic [7:0] gmem [0:262143];      // 期望 P
  logic [31:0] ctrl [0:255];
  integer t, fails, total;

  // ctrl 布局：ctrl[0]=用例数 T；用例 k（0 起）参数 ctrl[4k+1..4k+4] = m,n,causal,s_base，
  //   其 S/P 记录在 smem/gmem 的第 k 个 4096 字区（0 起，与生成器一致）。
  task run_one(input int idx);
    integer m, n, i, j, L, addr, guard;
    begin
      m = ctrl[idx*4+1]; n = ctrl[idx*4+2]; causal = ctrl[idx*4+3][0];
      s_base = ctrl[idx*4+4];
      m_rows = m; n_cols = n;
      // 装 S 进 CTX（lane = i mod 16, addr = s_base + (i>>4)*n + j）
      for (i = 0; i < m; i++)
        for (j = 0; j < n; j++) begin
          addr = s_base + (i>>4)*n + j;
          L = i - ((i>>4)<<4);
          mem[addr][L*8 +: 8] = smem[idx*4096 + i*n + j];
        end
      @(negedge clk) rst_n = 1;
      @(negedge clk) start = 1; @(negedge clk) start = 0;
      guard = 0;
      while (!done && guard < 200000) begin @(negedge clk); guard = guard + 1; end
      $display("[tb] t=%0d done=%b guard=%0d st=%0d mem[sb]=%h", idx, done, guard,
               dut.st, mem[s_base]);
      @(negedge clk);
      // 比对（bank-major 读回）
      for (i = 0; i < m; i++)
        for (j = 0; j < n; j++) begin
          addr = s_base + (i>>4)*n + j;
          L = i - ((i>>4)<<4);
          total = total + 1;
          if (mem[addr][L*8 +: 8] !== gmem[idx*4096 + i*n + j]) begin
            fails = fails + 1;
            if (fails <= 1300)
              $display("[FAIL] t=%0d m=%0d n=%0d causal=%0d i=%0d j=%0d got=%02x exp=%02x",
                       idx, m, n, causal, i, j, mem[addr][L*8 +: 8], gmem[idx*4096 + i*n + j]);
          end
        end
    end
  endtask

  integer T;
  initial begin
    $readmemh("sm16_ctrl.mem", ctrl);
    $readmemh("sm16_s.mem", smem);
    $readmemh("sm16_gold.mem", gmem);
    fails = 0; total = 0; start = 0;
    for (int q = 0; q < 16; q++) mem[q] = '0;
    T = ctrl[0];      // 首字 = 用例数
    rst_n = 0;
    for (t = 0; t < T; t = t + 1) run_one(t);      // ctrl[0]=T，用例从 0 起
    if (fails == 0) $display("[tb_sm16] %0d 用例 %0d 元素全部位精确一致 PASS", T, total);
    else            $display("[tb_sm16] FAIL %0d/%0d", fails, total);
    $finish;
  end
endmodule
