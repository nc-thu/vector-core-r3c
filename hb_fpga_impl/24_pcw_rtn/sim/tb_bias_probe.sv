// BIAS 探针 TB — 只跑一个 BIAS 用例
`timescale 1ns/1ps
module tb_bias_probe;
  localparam int CTX_WORDS = 13824;
  logic clk = 0, rst_n = 0;
  always #5 clk = ~clk;

  logic [19:0] raddr;
  logic [127:0] rdata;
  logic [15:0] we_byte;
  logic [19:0] waddr;
  logic [127:0] wdata;
  (* ram_style = "block" *) logic [127:0] mem [0:CTX_WORDS-1];
  initial $readmemh("actv_ctx0.mem", mem);
  always_ff @(posedge clk) begin
    for (int b = 0; b < 16; b++)
      if (we_byte[b]) mem[waddr[13:0]][b*8 +: 8] <= wdata[b*8 +: 8];
    rdata <= mem[raddr[13:0]];
  end

  logic start, busy, done;
  logic [2:0]  submode;
  logic [19:0] y_base, tbl_base;
  logic [15:0] m_rows, n_cols, tbl_len, rq_m, rq_m2;
  logic [7:0]  rq_s;
  logic        eng_we;
  logic [15:0] eng_welane;
  logic [19:0] eng_waddr;
  logic [127:0] eng_wdata;

  ae_actv u_dut (
    .clk(clk), .rst_n(rst_n), .start(start), .busy(busy), .done(done),
    .submode(submode), .y_base(y_base), .m_rows(m_rows), .n_cols(n_cols),
    .tbl_base(tbl_base), .tbl_len(tbl_len), .rq_m(rq_m), .rq_s(rq_s),
    .rq_m2(rq_m2),
    .ctx_raddr(raddr), .ctx_rdata(rdata),
    .ctx_we(eng_we), .ctx_welane(eng_welane),
    .ctx_waddr(eng_waddr), .ctx_wdata(eng_wdata)
  );
  assign we_byte = eng_we ? eng_welane : '0;
  assign waddr   = eng_waddr;
  assign wdata   = eng_wdata;

  int cyc;
  initial begin
    rst_n = 0; start = 0;
    repeat (6) @(posedge clk);
    rst_n = 1; repeat (4) @(posedge clk);
    
    // case5 BIAS: y=100, m=5, n=40, k=40, rqm=-384, rqs=4, tbl=840
    submode = 3'd1;
    y_base = 20'd100;
    m_rows = 16'd5;
    n_cols = 16'd40;
    tbl_base = 20'd840;
    tbl_len = 16'd40;
    rq_m = 16'hFE80;  // -384
    rq_s = 8'd4;
    rq_m2 = 16'd0;
    
    @(posedge clk); start = 1; @(posedge clk); start = 0;
    cyc = 0;
    while (!done && cyc < 200000) begin
      @(posedge clk); cyc++;
      // 探针：在 A_RUN 状态打印 BIAS 计算中间值
      if (u_dut.st == u_dut.A_RUN && cyc <= 60) begin
        $display("[cyc=%0d] st=A_RUN jr=%0d jw=%0d run_v1=%b run_v2=%b bwv=%b",
                 cyc, u_dut.jr, u_dut.jw, u_dut.run_v1, u_dut.run_v2, u_dut.bwv);
        if (u_dut.run_v2 || u_dut.bwv) begin
          $display("  rd_r[7:0]=%0d b_r=%0d rq_m_r=%0d rq_s_r=%0d",
                   $signed(u_dut.rd_r[7:0]), u_dut.b_r, u_dut.rq_m_r, u_dut.rq_s_r);
          $display("  g_bias[0].mb=%0d g_bias[0].prod=%0d g_bias[0].accb=%0d g_bias[0].accb_q=%0d",
                   u_dut.g_bias[0].mb, u_dut.g_bias[0].prod, u_dut.g_bias[0].accb, u_dut.g_bias[0].accb_q);
          $display("  g_bias[0].p_sh=%0d g_bias[0].sat=%0d bwbyte[7:0]=0x%02x",
                   u_dut.g_bias[0].p_sh, $signed(u_dut.g_bias[0].sat), u_dut.bwbyte[7:0]);
        end
      end
    end
    $display("[tb] done, cyc=%0d", cyc);
    
    // dump row 0 (addr 100..139)
    for (int a = 100; a < 108; a++) begin
      logic [127:0] w;
      w = mem[a];
      $display("addr=%0d lane0=0x%02x(%0d)", a, w[7:0], $signed(w[7:0]));
    end
    $finish;
  end
endmodule
