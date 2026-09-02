// tb_ae.sv — 冒烟仿真：同一 seq.mem，三遍运行 REF-pf0 / PRIM-pf0 / PRIM-pf1，
// dump CTX/DDR 终态与 golden 期望位精确比对（compare.py 比前两遍；
// 第三遍 pf_en=1 的 dump 应与 PRIM-pf0 逐位一致——预取不改终态，只省拍）。
// DDR = 行为级 AXI4 从机（64KB 字节数组，读通道含随机延迟以打 D_R2 背压路径）。
`timescale 1ns/1ps
module tb_ae;
  localparam int COLS      = 12;
  localparam int CTX_WORDS = 1024;
  localparam int W_WORDS   = 64;
  localparam int SEQ_N     = 64;
  localparam int DDR_BYTES = 65536;

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

  // 半周期（ns）。默认 5（10ns/拍，原回归节奏不变）。功耗 VCD 运行用
  // -DCLK_HALF=2（4ns/拍 = 250MHz），SAIF 翻转率才和功耗约束频率一致——
  // 否则按 100MHz 仿真速度算活动，逻辑动态功耗会被低估约 2.5 倍。
  `ifndef CLK_HALF
    `define CLK_HALF 5
  `endif
  always #`CLK_HALF clk = ~clk;

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

  // ---------------- 行为级 AXI4 DDR 从机 ----------------
  logic [7:0] ddr  [0:DDR_BYTES-1];   // 运行映像
  logic [7:0] ddr0 [0:DDR_BYTES-1];   // 初值快照
  initial $readmemh("ddr_init.mem", ddr0);

  // 伪随机延迟源（LFSR）。
  // ★ LFSR 不随复位清零、跨 run 连续演进 → 第三遍（PRIM-pf1）的停顿序列与第二遍
  //   天然错相， dma_cycles 会差 ±25 拍量级（历史实测 +23），无法做 pf 对账。
  //   对策：tag==1（PRIM-pf0）起点拍快照，tag==2（PRIM-pf1）首拍装载回放同一序列。
  logic [15:0] lfsr = 16'hACE1;
  logic [15:0] lfsr_mark;
  logic        lfsr_load = 1'b0;   // 单拍装载请求（LFSR 进程自清，避免跨进程竞争）
  wire  [15:0] lfsr_nxt = {lfsr[14:0], lfsr[15]^lfsr[13]^lfsr[12]^lfsr[10]};
  always_ff @(posedge clk) begin
    if (lfsr_load) begin lfsr <= lfsr_mark; lfsr_load <= 1'b0; end
    else lfsr <= lfsr_nxt;
  end
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

  // ---------------- 层级访问（genvar 常量索引，iverilog 不支持变量 genblk 索引） ----------------
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
  // SM16 后 softmax 每拍写一整列（16 lane）：row=行组基行号，wdata=16 行同列 P
  // DMA STORE 逐拍（wr_st: W_RD=3 W_RD2=4 W_W=5）
  localparam bit SM_DBG = 1'b0;
  always @(posedge clk) if (SM_DBG && rst_n && dut.u_sm.ctx_we)
    $display("[sm] row=%0d j=%0d addr=%0d wdata=%h",
             dut.u_sm.row, dut.u_sm.j, dut.u_sm.ctx_waddr, dut.u_sm.ctx_wdata);
  always @(posedge clk) if (SM_DBG && rst_n && dut.wr_busy
                            && dut.u_dma.wr_st >= 3 && dut.u_dma.wr_st <= 5)
    $display("[dmas] wr_st=%0d wbeat=%0d raddr=%0d rd16=%h wvalid=%b wdata=%h",
             dut.u_dma.wr_st, dut.u_dma.w_wbeat, dut.u_dma.ctx_raddr,
             dut.u_dma.w_rd16, dut.u_dma.wvalid, dut.u_dma.wdata);

  // ---------------- 预取不变量断言（PF_CHK 置 0 关闭） ----------------
  // sched 主 FSM 状态编码（ae_sched.sv st_e）：T_EXEC=3，T_RUN_CP=6
  localparam bit PF_CHK = 1'b1;
  always @(posedge clk) if (PF_CHK && rst_n) begin
    // A：pf_v=1 期间到达的 T_EXEC（未消费，pf_hit_r 尚未置位）必是 pf_pc_r 处的 OP_LOAD
    if (dut.u_sched.pf_v && dut.u_sched.st == 4'd3 && !dut.u_sched.pf_hit_r
        && !(dut.u_sched.pc == dut.u_sched.pf_pc_r
             && dut.u_sched.desc_r[255:252] == 4'd4)) begin
      $display("[pf] FATAL: pf_v=1 但 T_EXEC pc=%0d 不是预取目标 OP_LOAD(pf_pc_r=%0d)",
               dut.u_sched.pc, dut.u_sched.pf_pc_r);
      $fatal(1);
    end
    // B：后台 DMA 在飞期间不得进入 COPY（两者都写 WRAM B 口，会丢后台写数据）
    if (dut.u_sched.pf_v && dut.u_sched.st == 4'd6) begin
      $display("[pf] FATAL: pf_v=1 期间进入 T_RUN_CP（COPY 与后台 DMA WRAM 写冲突）");
      $fatal(1);
    end
  end

  // 预取活动探针（PF_DBG 置 1 开启）
  localparam bit PF_DBG = 1'b0;
  always @(posedge clk) if (PF_DBG && rst_n) begin
    if (dut.u_sched.pf_bg_start)
      $display("[pfdbg] bg_start pf_pc_r=%0d addr=%h len=%0d tag=%0d base=%0d",
               dut.u_sched.pf_pc_r, dut.pf_dmaaddr, dut.pf_dmalen,
               dut.pf_dmatag, dut.pf_dmabase);
    if (dut.u_sched.st == 4'd3 && dut.u_sched.dma_busy
        && dut.u_sched.d_op >= 4'd4 && dut.u_sched.d_op <= 4'd5
        && !(dut.u_sched.pf_v && dut.u_sched.pf_pc_r == dut.u_sched.pc))
      $display("[pfdbg] T_EXEC stall @pc=%0d op=%0d (dma_busy 串行兜底)",
               dut.u_sched.pc, dut.u_sched.d_op);
  end

  // GEMM 引擎逐次 busy 时长探针（G_DBG 置 1 开启；查引擎间干扰用）
  localparam bit G_DBG = 1'b0;
  logic g_was_busy = 1'b0;
  time  g_t0;
  always @(posedge clk) if (G_DBG && rst_n) begin
    if (dut.g_busy && !g_was_busy) g_t0 = $time;
    if (!dut.g_busy && g_was_busy)
      $display("[gdbg] pc=%0d dur=%0d", dut.u_sched.pc, ($time - g_t0) / 10);
    g_was_busy <= dut.g_busy;
  end

  // ★ R2 诊断探针：统计后台 CTX 预取写被 B 口优先级压掉的拍数
  //   条件：bg_wran=1（后台预取在飞）且 r_tag_r==0（CTX 预取）且
  //         B 口被 eng_g/eng_sm/eng_a/前台LOAD 占（bg_wran 分支进不去）
  //   这些拍 dma ctx_we=1 但 ae_core B 口 mux 选了别的引擎 → 写静默丢失
  //   DMA rd 引擎无 stall 反馈 → r_byi 照常推进 → rd_done 照常 fire
  //   → pf_done=1 但 CTX 数据残缺 → 后续 LOAD 命中预取读到旧/错数据
  localparam bit PF_DROP_DBG = 1'b1;
  logic [31:0] pf_ctx_drop_cnt;
  always @(posedge clk) if (PF_DROP_DBG && rst_n) begin
    if (pf_en && dut.bg_wran &&
        (dut.eng_g || dut.eng_sm || dut.eng_a ||
         (dut.eng_dma && !dut.dma_iswr)) &&
        dut.u_dma.r_tag_r == 3'd0 && dut.u_dma.rd_st != dut.u_dma.R_IDLE)
      pf_ctx_drop_cnt <= pf_ctx_drop_cnt + 32'd1;
  end

  // ---------------- 功耗 VCD 采样（默认关闭：编译加 -DDUMP_VCD 才生效） ----------
  // 限窗 dump：全量 $dumpvars(0, dut) 的 VCD 会到 GB 级，只在指定 run 的 start 之后
  // 延迟 DUMP_DELAY 拍打开 DUMP_LEN 拍窗口。SAIF 链路：tb_ae.vcd -> vcd2saif.py
  // -> read_saif（SAIF 树 re-root 到 dut，netlist 侧 -instance_name u_core）。
  //   iverilog -g2012 -DDUMP_VCD -DDUMP_RUN=2 -DDUMP_DELAY=200 -DDUMP_LEN=2000 ...
  //   DUMP_RUN: 0=REF / 1=PRIM-pf0 / 2=PRIM-pf1；DUMP_* 单位是 clk 拍。
  `ifdef DUMP_VCD
    `ifndef DUMP_RUN
      `define DUMP_RUN 2
    `endif
    `ifndef DUMP_DELAY
      `define DUMP_DELAY 200
    `endif
    `ifndef DUMP_LEN
      `define DUMP_LEN 2000
    `endif
    `ifndef DUMP_FILE
      `define DUMP_FILE "tb_ae.vcd"
    `endif
    int dump_wait = 0;
    int dump_left = 0;
    initial begin
      $dumpfile(`DUMP_FILE);
      $dumpvars(0, dut);
      $dumpoff;                       // 窗口外的活动不写文件
    end
    always @(posedge clk) begin
      if (dump_wait > 1) dump_wait <= dump_wait - 1;
      else if (dump_wait == 1) begin
        dump_wait <= 0;
        dump_left   <= `DUMP_LEN;
        $display("[vcd] %0t dump ON（%0d 拍窗口）", $time, `DUMP_LEN);
        $dumpon;
      end else if (dump_left > 1) dump_left <= dump_left - 1;
      else if (dump_left == 1) begin
        dump_left <= 0;
        $display("[vcd] %0t dump OFF", $time);
        $dumpoff;
      end
    end
  `endif

  // ---------------- 一次完整运行 ----------------
  task automatic run_mode(input bit prim, input bit pf, input int tag);
    int f;
    // LFSR 停顿序列对齐：tag==1 拍快照（lfsr_nxt = 本 run 首拍后的值），
    //                      tag==2 首拍装载（与 tag==1 的首拍后值对齐）
    if (tag == 1) lfsr_mark = lfsr_nxt;
    if (tag == 2) lfsr_load = 1'b1;
    // 全量复位：CTX/WRAM 清零，DDR 重载初值
    zc_pulse = 1; zw_pulse = 1; #1; zc_pulse = 0; zw_pulse = 0;
    for (int i = 0; i < DDR_BYTES; i++) ddr[i] = ddr0[i];

    rst_n = 0; repeat (4) @(posedge clk);
    hoist_en = prim;
    pf_en = pf;
    pf_ctx_drop_cnt = 0;
    rst_n = 1; repeat (4) @(posedge clk);
    start = 1; @(posedge clk); start = 0;
`ifdef DUMP_VCD
    if (tag == `DUMP_RUN) begin
      dump_wait = `DUMP_DELAY;        // DUMP_DELAY 拍后开 DUMP_LEN 拍 VCD 窗口
      $display("[vcd] %0t run tag=%0d，%0d 拍后开窗", $time, tag, `DUMP_DELAY);
    end
`endif
    wait (done);
    @(posedge clk);
    $display("[tb] %s%s: cycles=%0d gemm=%0d dma=%0d mac_total=%0d skip_macs=%0d skip_stages=%0d",
             prim ? "PRIM" : "REF ", pf ? "-pf1" : "-pf0", cycles, gemm_cycles,
             dma_cycles, mac_total, skip_macs, skip_stages);
    if (PF_DROP_DBG && pf)
      $display("[probe] pf_ctx_drop_cnt=%0d (CTX 预取写被 B 口优先级压掉的拍数)",
               pf_ctx_drop_cnt);

    // dump CTX（bank-major，与 golden expected_ctx 顺序一致）与 DDR
    dc_pulse = 1; #1; dc_pulse = 0;
    f = $fopen(tag == 0 ? "dump_ctx_ref.mem" :
               tag == 1 ? "dump_ctx_prim.mem" : "dump_ctx_prim2.mem", "w");
    for (int i = 0; i < 16*CTX_WORDS; i++) $fwrite(f, "%02X\n", ctx_flat[i]);
    $fclose(f);
    f = $fopen(tag == 0 ? "dump_ddr_ref.mem" :
               tag == 1 ? "dump_ddr_prim.mem" : "dump_ddr_prim2.mem", "w");
    for (int i = 0; i < DDR_BYTES; i++) $fwrite(f, "%02X\n", ddr[i]);
    $fclose(f);
  endtask

  initial begin
    run_mode(1'b0, 1'b0, 0);   // REF，无预取：位图不置位，逐步重算
    run_mode(1'b1, 1'b0, 1);   // PRIMITIVE，无预取：step-invariant 跳过
    run_mode(1'b1, 1'b1, 2);   // PRIMITIVE + pf_en=1：权重预取开（dump 应与上一遍逐位一致）
    $display("[tb] 三遍 dump 完成（compare.py 比 REF/PRIM；prim2 vs prim 逐位 diff）");
    $finish;
  end

  // 看门狗
  initial begin
    #20_000_000;
    $display("[tb] FATAL: 超时（2M 周期）");
    $fatal(1);
  end
endmodule
