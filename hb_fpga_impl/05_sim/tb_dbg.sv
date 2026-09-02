// tb_ae_v.sv — tb_ae.sv 的 Verilator(--timing) 移植版（2026-08-30 迁移记录见 NOTES.txt）
// 与 tb_ae.sv 的三点差别（其余逐行一致，含 DDR 行为级从机与 LFSR 读延迟）：
//   1) 去掉全部跨模块层级访问（CTX/WRAM 清零脉冲、CTX dump、SM/DMA 调试探针）——
//      仿真器 Verilator 不支持 tb 引用 dut 内部信号/存储器。
//      清零语义等价性：ae_ctx_ram 的 mem 未写 initial，iverilog TB 靠层级脉冲显式清零；
//      仿真器 2 态语义下未初始化即 0，与"每次运行前清零"等价。WRAM 本身有 initial 清零。
//   2) 一个进程只跑一个模式：+MODE=0 REF / +MODE=1 PRIM（新进程=干净初始状态），
//      不再有 tb_ae.sv 的"同进程连跑两遍+中途清零"。
//   3) 只 dump DDR 终态（所有模型输出都经 OP_STORE 写回 DDR，CTX dump 只是冗余回归
//      校验；对拍用 dump_ddr_{ref,prim}.mem 与 expected_ddr_* / iverilog dump 逐字节比）。
// 参数档：默认 = 冒烟档（与 01_rtl/sim/tb_ae.sv 同值）；全参数档用
//   -DV_COLS=108 -DV_CTX_WORDS=131072 -DV_W_WORDS=4096 -DV_SEQ_N=2048 -DV_DDR_BYTES=8388608
`timescale 1ns/1ps
module tb_dbg;
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
 `define V_WDG_CYC 2000000        // 看门狗（拍）；全参数大负载用 -DV_WDG_CYC=20000000
`endif
  localparam int COLS      = `V_COLS;
  localparam int CTX_WORDS = `V_CTX_WORDS;
  localparam int W_WORDS   = `V_W_WORDS;
  localparam int SEQ_N     = `V_SEQ_N;
  localparam int DDR_BYTES = `V_DDR_BYTES;

  int mode = 0;                   // +MODE=0 REF / 1 PRIM
  int f;

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

  // ---------------- 行为级 AXI4 DDR 从机（与 tb_ae.sv 逐行一致） ----------------
  logic [7:0] ddr  [0:DDR_BYTES-1];
  initial $readmemh("ddr_init.mem", ddr);

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
        $display("[t=%0t] AR addr=%h len=%0d", $time, araddr, arlen);
        r_addr <= araddr; r_total <= {1'b0, arlen} + 9'd1; r_beat <= 9'd0;
        r_run <= 1'b1; arready <= 1'b0;
      end else if (r_run && !rvalid && stall_r) begin
        $display("[t=%0t] STALL beat=%0d lfsr=%h", $time, r_beat, lfsr);
      end else if (r_run && !rvalid && !stall_r) begin
        $display("[t=%0t] RDATA beat=%0d", $time, r_beat);
        rvalid <= 1'b1;
        rdata  <= {ddr[r_addr + r_beat*8 + 7], ddr[r_addr + r_beat*8 + 6],
                   ddr[r_addr + r_beat*8 + 5], ddr[r_addr + r_beat*8 + 4],
                   ddr[r_addr + r_beat*8 + 3], ddr[r_addr + r_beat*8 + 2],
                   ddr[r_addr + r_beat*8 + 1], ddr[r_addr + r_beat*8]};
        rlast  <= (r_beat == r_total - 9'd1);
      end else if (rvalid && rready) begin
        $display("[t=%0t] RACK beat=%0d last=%b", $time, r_beat, rlast);
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

  // ---------------- 单模式一次完整运行 ----------------
  // 复位/启动时序与 tb_ae.sv run_mode() 完全一致（仅去掉 CTX/WRAM 清零脉冲，
  // 见文件头说明 1）。CTX 终态不 dump（说明 3）。
  initial begin
    if (!$value$plusargs("MODE=%d", mode)) mode = 0;
    $display("[tb_v] MODE=%0d (%s) COLS=%0d CTX_WORDS=%0d W_WORDS=%0d SEQ_N=%0d DDR_BYTES=%0d",
             mode, mode ? "PRIM" : "REF", COLS, CTX_WORDS, W_WORDS, SEQ_N, DDR_BYTES);

    rst_n = 0; repeat (4) @(posedge clk);
    hoist_en <= (mode != 0);
    rst_n <= 1; repeat (4) @(posedge clk);  // NBA：复位释放也不能用阻塞（自由轮转计数器会被错相一拍）
    start <= 1; @(posedge clk); start <= 0;  // NBA：消除与 always_ff 的调度竞态（iverilog DUT先评估/Verilator TB先执行）
    wait (done);
    @(posedge clk);
    $display("[tb_v] %s: cycles=%0d gemm=%0d dma=%0d mac_total=%0d skip_macs=%0d skip_stages=%0d",
             mode ? "PRIM" : "REF ", cycles, gemm_cycles, dma_cycles,
             mac_total, skip_macs, skip_stages);

    f = $fopen(mode == 0 ? "dump_ddr_ref.mem" : "dump_ddr_prim.mem", "w");
    for (int i = 0; i < DDR_BYTES; i++) $fwrite(f, "%02X\n", ddr[i]);
    $fclose(f);
    $finish;
  end

  // 看门狗
  initial begin
    #(`V_WDG_CYC * 10);
    $display("[tb_v] FATAL: 超时（%0d 周期）", `V_WDG_CYC);
    $fatal(1);
  end

  // VCD 波形（iverilog=iv.vcd / Verilator=vl.vcd；只 dump dut 层级，不带 TB 的 ddr 大数组）
  initial begin
  `ifdef VERILATOR
    $dumpfile("vl.vcd"); $dumpvars(0, dut);
  `else
    $dumpfile("iv.vcd"); $dumpvars(0, dut);
  `endif
  end
endmodule
