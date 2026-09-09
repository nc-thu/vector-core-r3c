// 最小 BIAS 诊断 TB
`timescale 1ns/1ps
module tb_bias_diag;
  logic clk = 0, rst_n = 0, start = 0;
  logic busy, done;
  // CTX
  logic [19:0] ctx_raddr;
  logic [127:0] ctx_rdata;
  logic ctx_we;
  logic [15:0] ctx_welane;
  logic [19:0] ctx_waddr;
  logic [127:0] ctx_wdata;
  // 描述符
  logic [255:0] desc_r;
  logic [2:0] submode;
  logic [19:0] y_base, tbl_base;
  logic [15:0] m_rows, n_cols, tbl_len, rq_m, rq_m2;
  logic [7:0] rq_s;

  ae_actv u_actv(
    .clk(clk), .rst_n(rst_n), .start(start),
    .submode(submode), .y_base(y_base), .m_rows(m_rows), .n_cols(n_cols),
    .tbl_base(tbl_base), .tbl_len(tbl_len),
    .rq_m(rq_m), .rq_s(rq_s), .rq_m2(rq_m2),
    .ctx_raddr(ctx_raddr), .ctx_rdata(ctx_rdata),
    .ctx_we(ctx_we), .ctx_welane(ctx_welane),
    .ctx_waddr(ctx_waddr), .ctx_wdata(ctx_wdata),
    .busy(busy), .done(done)
  );

  // CTX RAM
  localparam CTX_WORDS = 13824;
  (* ram_style = "block" *) logic [127:0] ctx_mem [0:CTX_WORDS-1];
  initial $readmemh("diag_ctx0.mem", ctx_mem);
  always_ff @(posedge clk) begin
    if (ctx_we) begin
      for (int b = 0; b < 16; b++)
        if (ctx_welane[b]) ctx_mem[ctx_waddr[CTX_AW-1:0]][b*8 +: 8] <= ctx_wdata[b*8 +: 8];
    end
    ctx_rdata <= ctx_mem[ctx_raddr[CTX_AW-1:0]];
  end
  localparam CTX_AW = 14;

  // 读描述符
  initial begin
    logic [255:0] d;
    $readmemh("diag_cases.mem", d);
    desc_r = d;
  end

  always #2 clk = ~clk;

  integer i;
  logic [127:0] dump;
  initial begin
    // 等描述符
    #1 desc_r = 64'h0; // 先占位
    // 手动设置参数
    submode = 3'd1;  // BIAS
    y_base = 20'd100;
    m_rows = 16'd16;
    n_cols = 16'd8;
    tbl_base = 20'd200;
    tbl_len = 16'd8;
    rq_m = 16'hFE80;  // -384 in 16-bit two's complement
    rq_s = 8'd4;
    rq_m2 = 16'd0;

    #10 rst_n = 1;
    #10 start = 1;
    #4 start = 0;

    // 等 done
    wait(done == 1);
    #4;

    // dump addr 100..107 lane 0
    for (i = 100; i < 108; i++) begin
      dump = ctx_mem[i];
      $display("addr=%0d lane0=0x%02x(%0d)", i, dump[7:0], $signed(dump[7:0]));
    end

    // dump 到文件
    begin
      integer f;
      f = $fopen("diag_out.mem", "w");
      for (int a = 0; a < CTX_WORDS; a++) begin
        dump = ctx_mem[a];
        for (int L = 0; L < 16; L++)
          $fdisplay(f, "%02x", dump[L*8 +: 8]);
      end
      $fclose(f);
    end

    $display("DONE");
    $finish;
  end
endmodule
