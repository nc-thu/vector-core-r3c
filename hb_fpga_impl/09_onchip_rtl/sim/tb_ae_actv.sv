// tb_ae_actv.sv — AE_ACTV 微观对拍台：直接例化引擎 + 行为级 CTX（同 ae_ctx_ram
// 时序：发址次拍回数），逐条描述符驱动，全跑完 dump CTX 终态给 actv_gold.py 比对。
// 用例/CTX 初值/期望值全部由 actv_gold.py gen 生成（numpy 黄金，随机向量）。
`timescale 1ns/1ps
module tb_ae_actv;
  localparam int CTX_WORDS = 1280;
  localparam int N_CASES   = 8;

  logic clk = 0, rst_n = 0;
  always #5 clk = ~clk;

  // ---------------- 行为级 CTX（SDP：A 只读 / B 逐字节写，发址次拍回数）----------------
  logic [19:0] raddr;
  logic [127:0] rdata;
  logic [15:0] we_byte;
  logic [19:0] waddr;
  logic [127:0] wdata;
  (* ram_style = "block" *) logic [127:0] mem [0:CTX_WORDS-1];
  initial $readmemh("actv_ctx0.mem", mem);
  always_ff @(posedge clk) begin
    for (int b = 0; b < 16; b++)
      if (we_byte[b]) mem[waddr[10:0]][b*8 +: 8] <= wdata[b*8 +: 8];
    rdata <= mem[raddr[10:0]];
  end

  // ---------------- 用例描述符（op=6 编码，切片同 ae_core 接线）----------------
  logic [255:0] cases [0:N_CASES-1];
  initial $readmemh("actv_cases.mem", cases);

  // ---------------- 引擎 ----------------
  logic start, busy, done;
  logic [2:0]  submode;
  logic [19:0] y_base, tbl_base;
  logic [15:0] m_rows, n_cols, tbl_len, rq_m;
  logic [7:0]  rq_s;
  logic        eng_we;
  logic [15:0] eng_welane;
  logic [19:0] eng_waddr;
  logic [127:0] eng_wdata;

  ae_actv u_dut (
    .clk(clk), .rst_n(rst_n), .start(start), .busy(busy), .done(done),
    .submode(submode), .y_base(y_base), .m_rows(m_rows), .n_cols(n_cols),
    .tbl_base(tbl_base), .tbl_len(tbl_len), .rq_m(rq_m), .rq_s(rq_s),
    .ctx_raddr(raddr), .ctx_rdata(rdata),
    .ctx_we(eng_we), .ctx_welane(eng_welane),
    .ctx_waddr(eng_waddr), .ctx_wdata(eng_wdata)
  );
  assign we_byte = eng_we ? eng_welane : '0;
  assign waddr   = eng_waddr;
  assign wdata   = eng_wdata;

  // ---------------- 调试探针（ACTV_DBG=1 开启）----------------
  localparam bit ACTV_DBG = 1'b0;
  always @(posedge clk) if (ACTV_DBG && rst_n && eng_we)
    $display("[actv] %0t st=%0d row=%0d jr=%0d jw=%0d rd_r=%h addr=%0d lane=%h data=%h", $time,
             u_dut.st, u_dut.row, u_dut.jr, u_dut.jw, u_dut.rd_r, eng_waddr, eng_welane, eng_wdata);

  // ---------------- 逐用例驱动 ----------------
  int ci;
  int cyc;
  initial begin
    rst_n = 0; start = 0;
    repeat (6) @(posedge clk);
    rst_n = 1; repeat (4) @(posedge clk);
    for (ci = 0; ci < N_CASES; ci++) begin
      // 字段切片与 ae_core 完全同位（b_src/m/n/k/b_base/y_base/rq_m/rq_s）
      submode   = cases[ci][248:246];
      m_rows    = cases[ci][243:228];
      n_cols    = cases[ci][227:212];
      tbl_len   = cases[ci][211:196];
      tbl_base  = cases[ci][175:156];
      y_base    = cases[ci][155:136];
      rq_m      = cases[ci][119:104];
      rq_s      = cases[ci][103:96];
      @(posedge clk); start = 1; @(posedge clk); start = 0;
      cyc = 0;
      while (!done && cyc < 200000) begin @(posedge clk); cyc++; end
      if (!done) begin
        $display("[tb] FATAL: case%0d 超时（st=%0d）", ci, u_dut.st);
        $fatal(1);
      end
      $display("[tb] case%0d done，用时 %0d 拍（sub=%0d m=%0d n=%0d k=%0d y=%0d tbl=%0d）",
               ci, cyc, submode, m_rows, n_cols, tbl_len, y_base, tbl_base);
      // 逐用例快照（文件名与 actv_gold.py dump_ctx 对齐）
      begin
        int fc;
        string fname;
        fc = $fopen($sformatf(fname, "actv_ctx_out_c%0d.mem", ci), "w");
        if (fc == 0) begin $display("[tb] FATAL: 无法打开快照文件 case%0d", ci); $fatal(1); end
        for (int L = 0; L < 16; L++)
          for (int a = 0; a < CTX_WORDS; a++)
            $fwrite(fc, "%02X\n", mem[a][L*8 +: 8]);
        $fclose(fc);
      end
      @(posedge clk);
    end
    // dump 终态：idx = lane*CTX_WORDS + addr（与 actv_gold.py exp 同序）
    begin
      int f;
      f = $fopen("actv_ctx_out.mem", "w");
      for (int L = 0; L < 16; L++)
        for (int a = 0; a < CTX_WORDS; a++)
          $fwrite(f, "%02X\n", mem[a][L*8 +: 8]);
      $fclose(f);
    end
    $display("[tb] dump 完成（actv_ctx_out.mem，%0d 字节）", 16*CTX_WORDS);
    $finish;
  end

  // 看门狗
  initial begin
    #5_000_000;
    $display("[tb] FATAL: 总超时");
    $fatal(1);
  end
endmodule
