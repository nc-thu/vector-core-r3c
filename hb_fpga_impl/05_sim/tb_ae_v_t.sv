// tb_ae_v_t.sv — tb_ae_v.sv 的功耗采迹版（2026-08-31，只加两处，语义零改动）：
//   1) 时钟半周期参数化：-DV_HALF=2 → 4 ns/拍 = 250 MHz（功耗口径，见
//      06_power/POWER_SUMMARY.txt 的 CLK_HALF 说明）；缺省 5 与 tb_ae_v 一致。
//   2) +VCD 时在 start 拉高前 $dumpfile/$dumpvars(0, dut) 开窗（窗口从段执行
//      起点起算，不含 SEQ 装载；vcd2saif.py --t-start 0）。不给 +VCD 完全不采。
// 其余逐行照抄 tb_ae_v.sv（验收过的段接口 +SEQ/+DDRIMG/+DUMP）。
// 与 tb_ae.sv 的三点差别（其余逐行一致，含 DDR 行为级从机与 LFSR 读延迟）：
//   1) 去掉全部跨模块层级访问（CTX/WRAM 清零脉冲、CTX dump、SM/DMA 调试探针）——
//      仿真器 Verilator 不支持 tb 引用 dut 内部信号/存储器。
//      清零语义等价性：ae_ctx_ram 的 mem 未写 initial，iverilog TB 靠层级脉冲显式清零；
//      仿真器 2 态语义下未初始化即 0，与"每次运行前清零"等价。WRAM 本身有 initial 清零。
//   2) 一个进程只跑一个模式：+MODE=0 REF / +MODE=1 PRIM（新进程=干净初始状态），
//      不再有 tb_ae.sv 的"同进程连跑两遍+中途清零"。
//   3) 只 dump DDR 终态（所有模型输出都经 OP_STORE 写回 DDR，CTX dump 只是冗余回归
//      校验；对拍用 dump_ddr_{ref,prim}.mem 与 expected_ddr_* / iverilog dump 逐字节比）。
//   4) 段执行接口（2026-08-31 增，兼容编译器 segment_runner 契约）：
//      +SEQ=<路径>（经 PS 端口装载指令流，不给则沿用 RTL initial 直读 ./seq.mem）
//      +DDRIMG=<路径>（缺省 ./ddr_init.mem）、+DUMP=<路径>（缺省按 MODE/PF 命名）。
//      详见下方"段执行接口"注释块。
// 参数档：默认 = 冒烟档（与 01_rtl/sim/tb_ae.sv 同值）；全参数档用
//   -DV_COLS=108 -DV_CTX_WORDS=131072 -DV_W_WORDS=4096 -DV_SEQ_N=2048 -DV_DDR_BYTES=8388608
`timescale 1ns/1ps
`ifndef V_HALF
 `define V_HALF 5
`endif
module tb_ae_v;
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
  int pf = 0;                     // +PF=1 → PRIM+权重预取（对应 tb_ae.sv 第三遍）
  int f;
  bit ok;                         // plusargs 返回值垫底

  // ---------------- 段执行接口（编译器 segment_runner 契约，2026-08-31 增） ----------
  // 一个段 = 一份指令流 + 一份 DDR 输入映像进、一份 DDR 终态 dump 出，进程无状态残留。
  //   +SEQ=<路径>     指令流映像：复位释放后经 seq_we/seq_waddr/seq_wdata 端口写满
  //                   SEQ RAM（PS 装载通道，512/2048 条）。不给则完全不动端口，沿用
  //                   RTL initial 直读 ./seq.mem 的老路径（验收门/对拍用，逐拍不变）。
  //   +DDRIMG=<路径>  DDR 输入映像（缺省 ./ddr_init.mem）
  //   +DUMP=<路径>    DDR 终态 dump（缺省按 MODE/PF 命名 dump_ddr_{ref,prim,prim2}.mem）
  // 注意：RTL 的 initial $readmemh("seq.mem") 是固定文件名，+SEQ 装载会覆盖其全部内容，
  // 但 ./seq.mem 缺失时个别仿真器会告警——段目录建议放一份占位/同内容 seq.mem。
  string seq_f  = "seq.mem";
  string ddr_f  = "ddr_init.mem";
  string dump_f = "";
  string vcd_f  = "wave.vcd";
  bit   seq_via_port = 1'b0;
  logic [255:0] seq_img [0:SEQ_N-1];

  logic clk = 0, rst_n = 0, start = 0, hoist_en = 0, pf_en = 0;
  logic busy, done;
  logic [31:0] araddr, awaddr;
  logic [7:0]  arlen, awlen;
  logic        arvalid, arready, rvalid, rready, rlast;
  logic        awvalid, awready, wvalid, wready, wlast, bvalid, bready;
  logic [63:0] rdata, wdata;
  logic [7:0]  wstrb;
  logic [31:0] cycles, gemm_cycles, dma_cycles, mac_total, skip_macs;
  logic [15:0] skip_stages;
  logic        seq_we = 1'b0;
  logic [15:0] seq_waddr = 16'd0;
  logic [255:0] seq_wdata = 256'd0;

  always #`V_HALF clk = ~clk;

  ae_core #(.COLS(COLS), .CTX_WORDS(CTX_WORDS), .W_WORDS(W_WORDS),
            .SEQ_N(SEQ_N)) dut (
    .clk(clk), .rst_n(rst_n), .start(start), .hoist_en(hoist_en), .pf_en(pf_en),
    .busy(busy), .done(done),
    .araddr(araddr), .arlen(arlen), .arvalid(arvalid), .arready(arready),
    .rdata(rdata), .rvalid(rvalid), .rlast(rlast), .rready(rready),
    .awaddr(awaddr), .awlen(awlen), .awvalid(awvalid), .awready(awready),
    .wdata(wdata), .wstrb(wstrb), .wlast(wlast), .wvalid(wvalid),
    .wready(wready), .bvalid(bvalid), .bready(bready),
    .seq_we(seq_we), .seq_waddr(seq_waddr), .seq_wdata(seq_wdata),
    .cycles(cycles), .gemm_cycles(gemm_cycles), .dma_cycles(dma_cycles),
    .mac_total(mac_total), .skip_macs(skip_macs), .skip_stages(skip_stages)
  );

  // ---------------- 行为级 AXI4 DDR 从机（与 tb_ae.sv 逐行一致） ----------------
  logic [7:0] ddr  [0:DDR_BYTES-1];
  initial $readmemh(ddr_f, ddr);

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

  // ---------------- iverilog-only：CTX/WRAM 显式清零（Verilator 2 态天然为 0） ----------
  // 新向量负载会读到少量从未写过的 CTX 字（padding 区）：基线 tb_ae.sv 靠层级脉冲
  // 清零后是 00（golden 亦 00）；Verilator 未初始化即 0。iverilog 的 fresh 进程里
  // 存储器是 X，会以 "xx" 写进 DDR dump（2026-08-31 default 用例实测 168 字节）。
  // t=0 一次性清零，三个环境（基线 / iverilog 本表 / Verilator 本表）语义对齐。
`ifndef VERILATOR
  generate
    initial begin
      for (int i = 0; i < CTX_WORDS; i++) dut.u_ctx.mem[i] = '0;
    end
    for (genvar gw = 0; gw < COLS; gw++) begin : g_zw_v
      initial begin
        for (int i = 0; i < W_WORDS; i++) dut.g_w[gw].u_bank.mem[i] = '0;
      end
    end
  endgenerate
`endif

  // ---------------- 单模式一次完整运行 ----------------
  // 复位/启动时序与 tb_ae.sv run_mode() 完全一致（仅去掉 CTX/WRAM 清零脉冲，
  // 见文件头说明 1）。CTX 终态不 dump（说明 3）。
  initial begin
    if (!$value$plusargs("MODE=%d", mode)) mode = 0;
    if (!$value$plusargs("PF=%d", pf)) pf = 0;
    ok = $value$plusargs("DDRIMG=%s", ddr_f);
    ok = $value$plusargs("DUMP=%s", dump_f);
    seq_via_port = $value$plusargs("SEQ=%s", seq_f);
    $display("[tb_v] MODE=%0d (%s%s) COLS=%0d CTX_WORDS=%0d W_WORDS=%0d SEQ_N=%0d DDR_BYTES=%0d",
             mode, mode ? "PRIM" : "REF", (mode && pf) ? "-pf1" : "",
             COLS, CTX_WORDS, W_WORDS, SEQ_N, DDR_BYTES);
    if (dump_f == "")
      $display("[tb_v] SEQ=%s%s DDRIMG=%s DUMP=<按 MODE 缺省命名>",
               seq_f, seq_via_port ? "（端口装载）" : "（RTL initial 直读）", ddr_f);
    else
      $display("[tb_v] SEQ=%s%s DDRIMG=%s DUMP=%s",
               seq_f, seq_via_port ? "（端口装载）" : "（RTL initial 直读）", ddr_f, dump_f);

    rst_n = 0; repeat (4) @(posedge clk);
    hoist_en <= (mode != 0);
    pf_en   <= (mode != 0) && (pf != 0);
    rst_n <= 1; repeat (4) @(posedge clk);  // NBA：复位释放也不能用阻塞（自由轮转计数器会被错相一拍）
    if (seq_via_port) begin                 // +SEQ：经 PS 装载通道写满 SEQ RAM 再启动
      $readmemh(seq_f, seq_img);
      for (int i = 0; i < SEQ_N; i++) begin
        seq_we <= 1'b1; seq_waddr <= i; seq_wdata <= seq_img[i];
        @(posedge clk);
      end
      seq_we <= 1'b0;
      @(posedge clk);
    end
    // +VCD：start 前开窗（vcd2saif --t-start 0 对齐段执行窗口起点）
    if ($test$plusargs("VCD")) begin
      void'($value$plusargs("VCDFILE=%s", vcd_f));
      $dumpfile(vcd_f);
      $dumpvars(0, dut);
    end
    start <= 1; @(posedge clk); start <= 0;  // NBA：消除与 always_ff 的调度竞态（iverilog DUT先评估/Verilator TB先执行）
    wait (done);
    @(posedge clk);
    $display("[tb_v] %s%s: cycles=%0d gemm=%0d dma=%0d mac_total=%0d skip_macs=%0d skip_stages=%0d",
             mode ? "PRIM" : "REF ", (mode && pf) ? "-pf1" : "",
             cycles, gemm_cycles, dma_cycles, mac_total, skip_macs, skip_stages);

    if (dump_f == "") begin
      if (mode == 0)    dump_f = "dump_ddr_ref.mem";
      else if (pf == 0) dump_f = "dump_ddr_prim.mem";
      else              dump_f = "dump_ddr_prim2.mem";
    end
    f = $fopen(dump_f, "w");
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
endmodule
