// tb_ae_p.sv — 01_rtl/sim/tb_ae.sv 的参数化副本（仅把 5 个 localparam 换成可 -D 覆盖的宏，
// 其余逐行一致，含层级清零/CTX dump）。用途：iverilog 在全参数档跑 golden 对拍
// （Verilator 版 tb_ae_v.sv 不做 CTX dump，CTX 的 golden 校验只在此台做）。
//   冒烟档（默认，与原 tb_ae.sv 完全同参）：直接编译
//   全参数档：-DV_COLS=108 -DV_CTX_WORDS=131072 -DV_W_WORDS=4096 -DV_SEQ_N=2048 -DV_DDR_BYTES=8388608
`timescale 1ns/1ps
module tb_ae_p;
`ifndef V_COLS
 `define V_COLS 12
`endif
`ifndef V_CTX_WORDS
 `define V_CTX_WORDS 1024
`endif
`ifndef V_W_WORDS
 `define V_W_WORDS 64
`endif
`ifndef V_SEQ_N
 `define V_SEQ_N 64
`endif
`ifndef V_DDR_BYTES
 `define V_DDR_BYTES 65536
`endif
`ifndef V_WDG_CYC
 `define V_WDG_CYC 2000000
`endif
  localparam int COLS      = `V_COLS;
  localparam int CTX_WORDS = `V_CTX_WORDS;
  localparam int W_WORDS   = `V_W_WORDS;
  localparam int SEQ_N     = `V_SEQ_N;
  localparam int DDR_BYTES = `V_DDR_BYTES;

  logic clk = 0, rst_n = 0, start = 0, hoist_en = 0;
  logic busy, done;
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
    .clk(clk), .rst_n(rst_n), .start(start), .hoist_en(hoist_en),
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

  // ---------------- 行为级 AXI4 DDR 从机 ----------------
  logic [7:0] ddr  [0:DDR_BYTES-1];   // 运行映像
  logic [7:0] ddr0 [0:DDR_BYTES-1];   // 初值快照
  initial $readmemh("ddr_init.mem", ddr0);

  // 伪随机延迟源（LFSR）
  logic [15:0] lfsr = 16'hACE1;
  always_ff @(posedge clk) lfsr <= {lfsr[14:0], lfsr[15]^lfsr[13]^lfsr[12]^lfsr[10]};
  wire stall_r = (lfsr[2:0] == 3'b000);   // ~1/8 概率晚一拍给读数据

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
      end else if (r_run && !rvalid && !stall_r) begin
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

  // ---------------- 层级访问（iverilog 专用；Verilator 用 tb_ae_v.sv 替代本台） ----------------
  logic        zc_pulse, zw_pulse, dc_pulse;
  logic [7:0]  ctx_flat [0:16*CTX_WORDS-1];
  generate
    always @(posedge zc_pulse)
      for (int i = 0; i < CTX_WORDS; i++) dut.u_ctx.mem[i] = '0;
    for (genvar gw = 0; gw < COLS; gw++) begin : g_zw
      always @(posedge zw_pulse)
        for (int i = 0; i < W_WORDS; i++) dut.g_w[gw].u_bank.mem[i] = '0;
    end
    // dump：mem[addr][lane*8 +: 8] -> ctx_flat[lane*CTX_WORDS + addr]（bank-major，与 golden 一致）
    always @(posedge dc_pulse)
      for (int i = 0; i < CTX_WORDS; i++)
        for (int L = 0; L < 16; L++)
          ctx_flat[L*CTX_WORDS + i] = dut.u_ctx.mem[i][L*8 +: 8];
  endgenerate

  // ---------------- 调试探针（默认关闭：SM_DBG 置 1 开启） ----------------
  localparam bit SM_DBG = 1'b0;
  always @(posedge clk) if (SM_DBG && rst_n && dut.u_sm.ctx_we)
    $display("[sm] row=%0d j=%0d addr=%0d wdata=%h",
             dut.u_sm.row, dut.u_sm.j, dut.u_sm.ctx_waddr, dut.u_sm.ctx_wdata);
  always @(posedge clk) if (SM_DBG && rst_n && dut.eng_dma && dut.u_dma.st >= 5
                            && dut.u_dma.st <= 7)
    $display("[dmas] st=%0d wbeat=%0d raddr=%0d rd16=%h wvalid=%b wdata=%h",
             dut.u_dma.st, dut.u_dma.wbeat, dut.u_dma.ctx_raddr,
             dut.u_dma.rd16, dut.u_dma.wvalid, dut.u_dma.wdata);

  // ---------------- 一次完整运行 ----------------
  task automatic run_mode(input bit prim, input int tag);
    int f;
    // 全量复位：CTX/WRAM 清零，DDR 重载初值
    zc_pulse = 1; zw_pulse = 1; #1; zc_pulse = 0; zw_pulse = 0;
    for (int i = 0; i < DDR_BYTES; i++) ddr[i] = ddr0[i];

    rst_n = 0; repeat (4) @(posedge clk);
    hoist_en = prim;
    rst_n = 1; repeat (4) @(posedge clk);
    start = 1; @(posedge clk); start = 0;
    wait (done);
    @(posedge clk);
    $display("[tb] %s: cycles=%0d gemm=%0d dma=%0d mac_total=%0d skip_macs=%0d skip_stages=%0d",
             prim ? "PRIM" : "REF ", cycles, gemm_cycles, dma_cycles,
             mac_total, skip_macs, skip_stages);

    // dump CTX（bank-major，与 golden expected_ctx 顺序一致）与 DDR
    dc_pulse = 1; #1; dc_pulse = 0;
    f = $fopen(tag == 0 ? "dump_ctx_ref.mem"  : "dump_ctx_prim.mem", "w");
    for (int i = 0; i < 16*CTX_WORDS; i++) $fwrite(f, "%02X\n", ctx_flat[i]);
    $fclose(f);
    f = $fopen(tag == 0 ? "dump_ddr_ref.mem"  : "dump_ddr_prim.mem", "w");
    for (int i = 0; i < DDR_BYTES; i++) $fwrite(f, "%02X\n", ddr[i]);
    $fclose(f);
  endtask

  initial begin
    run_mode(1'b0, 0);   // REF：位图不置位，逐步重算
    run_mode(1'b1, 1);   // PRIMITIVE：step-invariant 跳过
    $display("[tb] 两模式 dump 完成（compare.py 比对）");
    $finish;
  end

  // 看门狗
  initial begin
    #(`V_WDG_CYC * 10);
    $display("[tb] FATAL: 超时（%0d 周期）", `V_WDG_CYC);
    $fatal(1);
  end
endmodule
