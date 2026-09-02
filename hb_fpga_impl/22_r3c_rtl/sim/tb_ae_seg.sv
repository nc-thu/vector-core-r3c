// tb_ae_seg.sv — 单段 RTL 仿真（COLS=108，PRIM+pf 模式）
// 用法：iverilog -g2012 -DCOLS=108 -DCTX_WORDS=4096 -DW_WORDS=4096
//       -DSEQ_N=64 -DDDR_BYTES=524288 -o seg.vvp <rtl> tb_ae_seg.sv
//       vvp seg.vvp → 输出 cycles + dump DDR
`timescale 1ns/1ps
module tb_ae_seg;
`ifndef COLS
  `define COLS 108
`endif
`ifndef CTX_WORDS
  `define CTX_WORDS 4096
`endif
`ifndef W_WORDS
  `define W_WORDS 4096
`endif
`ifndef SEQ_N
  `define SEQ_N 64
`endif
`ifndef DDR_BYTES
  `define DDR_BYTES 524288
`endif

  localparam int COLS      = `COLS;
  localparam int CTX_WORDS = `CTX_WORDS;
  localparam int W_WORDS   = `W_WORDS;
  localparam int SEQ_N     = `SEQ_N;
  localparam int DDR_BYTES = `DDR_BYTES;

  logic clk = 0, rst_n = 0, start = 0, hoist_en = 0, pf_en = 0;
  logic busy, done;
  // ★ plusarg 控制 PF（默认 1；+PF=0 关预取做对照）
  integer pf_arg;
  initial begin
    pf_arg = 1;
    if ($value$plusargs("PF=%d", pf_arg)) ;
  end
  logic [31:0] araddr, awaddr;
  logic [7:0]  arlen, awlen;
  logic        arvalid, arready, rvalid, rready, rlast;
  logic        awvalid, awready, wvalid, wready, wlast, bvalid, bready;
  logic [63:0] rdata, wdata;
  logic [7:0]  wstrb;
  logic [31:0] cycles, gemm_cycles, dma_cycles, mac_total, skip_macs;
  logic [15:0] skip_stages;

  always #5 clk = ~clk;

  ae_core #(.COLS(COLS), .CTX_WORDS(CTX_WORDS), .W_WORDS(W_WORDS),
            .SEQ_N(SEQ_N)) dut (
    .clk(clk), .rst_n(rst_n), .start(start), .hoist_en(hoist_en), .pf_en(pf_en),
    .busy(busy), .done(done),
    .araddr(araddr), .arlen(arlen), .arvalid(arvalid), .arready(arready),
    .rdata(rdata), .rvalid(rvalid), .rlast(rlast), .rready(rready),
    .awaddr(awaddr), .awlen(awlen), .awvalid(awvalid), .awready(awready),
    .wdata(wdata), .wstrb(wstrb), .wlast(wlast), .wvalid(wvalid),
    .wready(wready), .bvalid(bvalid), .bready(bready),
    .seq_we(1'b0), .seq_waddr(16'd0), .seq_wdata(256'd0),
    .cycles(cycles), .gemm_cycles(gemm_cycles), .dma_cycles(dma_cycles),
    .mac_total(mac_total), .skip_macs(skip_macs), .skip_stages(skip_stages)
  );

  // ---- 行为级 AXI4 DDR 从机（无随机延迟，追求速度）----
  logic [7:0] ddr [0:DDR_BYTES-1];
  initial $readmemh("ddr_init.mem", ddr);

  // AR/R 通道
  logic        r_run;
  logic [31:0] r_addr;
  logic [8:0]  r_beat, r_total;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      arready <= 1'b1; rvalid <= 1'b0; rlast <= 1'b0; r_run <= 1'b0;
    end else begin
      if (arready && arvalid) begin
        r_addr <= araddr; r_total <= {1'b0, arlen} + 9'd1; r_beat <= 9'd0;
        r_run <= 1'b1; arready <= 1'b0;
      end else if (r_run && !rvalid) begin
        rvalid <= 1'b1;
        rdata  <= {ddr[r_addr + r_beat*8 + 7], ddr[r_addr + r_beat*8 + 6],
                   ddr[r_addr + r_beat*8 + 5], ddr[r_addr + r_beat*8 + 4],
                   ddr[r_addr + r_beat*8 + 3], ddr[r_addr + r_beat*8 + 2],
                   ddr[r_addr + r_beat*8 + 1], ddr[r_addr + r_beat*8]};
        rlast  <= (r_beat == r_total - 9'd1);
      end else if (rvalid && rready) begin
        rvalid <= 1'b0;
        if (rlast) begin r_run <= 1'b0; arready <= 1'b1; end
        else r_beat <= r_beat + 9'd1;
      end
    end
  end

  // AW/W/B 通道
  logic        w_run;
  logic [31:0] w_addr;
  logic [8:0]  w_beat, w_total;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      awready <= 1'b1; wready <= 1'b0; bvalid <= 1'b0; w_run <= 1'b0;
    end else begin
      if (bvalid && bready) bvalid <= 1'b0;
      if (awready && awvalid) begin
        awready <= 1'b0; w_addr <= awaddr; w_total <= {1'b0, awlen} + 9'd1;
        w_beat <= 9'd0; wready <= 1'b1; w_run <= 1'b1;
      end else if (w_run && wready && wvalid) begin
        for (int q = 0; q < 8; q++)
          if (wstrb[q]) ddr[w_addr + w_beat*8 + q] <= wdata[q*8 +: 8];
        if (wlast) begin
          wready <= 1'b0; w_run <= 1'b0; awready <= 1'b1; bvalid <= 1'b1;
        end else w_beat <= w_beat + 9'd1;
      end
    end
  end

  integer f;
  initial begin
    rst_n = 0; repeat (4) @(posedge clk);
    hoist_en = 1; pf_en = pf_arg;
    rst_n = 1; repeat (4) @(posedge clk);
    start = 1; @(posedge clk); start = 0;
    wait (done);
    @(posedge clk);
    $display("[seg] cycles=%0d gemm=%0d dma=%0d mac_total=%0d skip_macs=%0d",
             cycles, gemm_cycles, dma_cycles, mac_total, skip_macs);
    // dump DDR
    f = $fopen("dump_ddr_seg.mem", "w");
    for (int i = 0; i < DDR_BYTES; i++) $fwrite(f, "%02X\n", ddr[i]);
    $fclose(f);
    $finish;
  end

  // 看门狗
  initial begin
    #200_000_000;
    $display("[seg] FATAL: 超时（20M 周期）");
    $fatal(1);
  end

  // ★ R2 调试探针：统计后台 CTX 预取写被 B 口优先级压掉的拍数
  //   bg_wran=1 且 r_tag_r==0（CTX 预取）且 B 口被 eng_g/eng_sm/eng_a 占
  //   这些拍 dma_ctx_we=1 但 ae_core B 口 mux 选了别的引擎 → 写静默丢失
  logic [31:0] pf_ctx_drop_cnt = 0;
  always @(posedge clk) if (rst_n && dut.bg_wran &&
      (dut.eng_g || dut.eng_sm || dut.eng_a ||
       (dut.eng_dma && !dut.dma_iswr)) &&
      dut.u_dma.r_tag_r == 3'd0 && dut.u_dma.rd_st != dut.u_dma.R_IDLE)
    pf_ctx_drop_cnt <= pf_ctx_drop_cnt + 1;

  initial begin
    wait (done);
    @(posedge clk);
    $display("[probe] pf_ctx_drop_cnt=%0d  (CTX 预取写被 B 口优先级压掉的拍数)",
             pf_ctx_drop_cnt);
  end
endmodule
