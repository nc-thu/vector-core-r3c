// tb_ae.sv — 冒烟仿真（★ v5-R2 双读引擎版，2026-09-08 20:16）
// 相对底本 22_r3c_rtl/sim/tb_ae.sv 的 diff：
//   1. 新增第二套行为级 AXI4 读从机（AR/R 通道 2，独立单 outstanding FSM、
//      独立 LFSR 停顿源 lfsr2——与从机 1 同构，读同一个 ddr 数组），
//      接 ae_core 第二读主口（w 读引擎）。
//   2. 三遍运行（REF-pf0 / PRIM-pf0 / PRIM-pf1）与 dump/比对流程不变；
//      LFSR 对齐（tag==1 快照 / tag==2 装载）两套从机各自做。
//   3. 探针改接新信号（底本的 dut.bg_wran/dut.u_dma.rd_st 已随双引擎重构）：
//      - pf_stall_cnt：ctx 引擎后台预取被 B 口让拍冻结的拍数（原 pf_ctx_drop_cnt 语义）
//      - rd_c/rd_w/both/union busy 计数：读引擎占用与重叠度量（R2 收益读数）
//   4. 新断言（v5-R2 不变量守卫）：
//      - 前台 ctx 装载（rd_c_busy && !rd_c_bg）不得与 eng_g/eng_sm/eng_a 重叠
//        （调度器消费点等待保证；违反说明让拍覆盖面有漏洞，写会静默丢）
//      - CTX 预取让拍期间 ctx 引擎不得处于会推进的状态之外……（简化为上面的覆盖）
//   底本的 PF_CHK 预取不变量断言保留（按引擎拆成 _c/_w 两套）。
`timescale 1ns/1ps
module tb_ae;
  localparam int COLS      = 12;
  localparam int CTX_WORDS = 1024;
  localparam int W_WORDS   = 64;
  localparam int SEQ_N     = 64;
  localparam int DDR_BYTES = 65536;

  logic clk = 0, rst_n = 0, start = 0, hoist_en = 0, pf_en = 0;
  logic busy, done;
  logic [31:0] araddr, araddr2, awaddr;
  logic [7:0]  arlen, arlen2, awlen;
  logic        arvalid, arready, rvalid, rready, rlast;
  logic        arvalid2, arready2, rvalid2, rready2, rlast2;
  logic        awvalid, awready, wvalid, wready, wlast, bvalid, bready;
  logic [63:0] rdata, rdata2, wdata;
  logic [7:0]  wstrb;
  logic [31:0] cycles, gemm_cycles, dma_cycles, mac_total, skip_macs;
  logic [15:0] skip_stages;

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
    .araddr2(araddr2), .arlen2(arlen2), .arvalid2(arvalid2), .arready2(arready2),
    .rdata2(rdata2), .rvalid2(rvalid2), .rlast2(rlast2), .rready2(rready2),
    .awaddr(awaddr), .awlen(awlen), .awvalid(awvalid), .awready(awready),
    .wdata(wdata), .wstrb(wstrb), .wlast(wlast), .wvalid(wvalid),
    .wready(wready), .bvalid(bvalid), .bready(bready),
    .seq_we(1'b0), .seq_waddr(16'd0), .seq_wdata(256'd0),
    .cycles(cycles), .gemm_cycles(gemm_cycles), .dma_cycles(dma_cycles),
    .mac_total(mac_total), .skip_macs(skip_macs), .skip_stages(skip_stages)
  );

  // ---------------- 行为级 AXI4 DDR 从机（读口 1 = ctx 引擎） ----------------
  logic [7:0] ddr  [0:DDR_BYTES-1];   // 运行映像
  logic [7:0] ddr0 [0:DDR_BYTES-1];   // 初值快照
  initial $readmemh("ddr_init.mem", ddr0);

  // 伪随机延迟源（LFSR）。两套从机各一个，tag==1 快照 / tag==2 装载对齐。
  logic [15:0] lfsr = 16'hACE1;
  logic [15:0] lfsr_mark;
  logic        lfsr_load = 1'b0;
  wire  [15:0] lfsr_nxt = {lfsr[14:0], lfsr[15]^lfsr[13]^lfsr[12]^lfsr[10]};
  always_ff @(posedge clk) begin
    if (lfsr_load) begin lfsr <= lfsr_mark; lfsr_load <= 1'b0; end
    else lfsr <= lfsr_nxt;
  end
  wire stall_r = (lfsr[2:0] == 3'b000);   // ~1/8 概率晚一拍给读数据

  // AR/R 通道 1
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

  // ---------------- 行为级 AXI4 DDR 从机（读口 2 = w 引擎）★ v5-R2 ----------------
  logic [15:0] lfsr2 = 16'h31C3;      // 异种子：两从机停顿序列错相
  logic [15:0] lfsr2_mark;
  wire  [15:0] lfsr2_nxt = {lfsr2[14:0], lfsr2[15]^lfsr2[13]^lfsr2[12]^lfsr2[10]};
  always_ff @(posedge clk) begin
    if (lfsr_load) lfsr2 <= lfsr2_mark;
    else lfsr2 <= lfsr2_nxt;
  end
  wire stall_r2 = (lfsr2[2:0] == 3'b000);

  logic        r2_run;
  logic [31:0] r2_addr;
  logic [8:0]  r2_beat, r2_total;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      arready2 <= 1'b1; rvalid2 <= 1'b0; rlast2 <= 1'b0; r2_run <= 1'b0;
    end else begin
      if (arready2 && arvalid2) begin
        r2_addr <= araddr2; r2_total <= {1'b0, arlen2} + 9'd1; r2_beat <= 9'd0;
        r2_run <= 1'b1; arready2 <= 1'b0;
      end else if (r2_run && !rvalid2 && !stall_r2) begin
        rvalid2 <= 1'b1;
        rdata2  <= {ddr[r2_addr + r2_beat*8 + 7], ddr[r2_addr + r2_beat*8 + 6],
                    ddr[r2_addr + r2_beat*8 + 5], ddr[r2_addr + r2_beat*8 + 4],
                    ddr[r2_addr + r2_beat*8 + 3], ddr[r2_addr + r2_beat*8 + 2],
                    ddr[r2_addr + r2_beat*8 + 1], ddr[r2_addr + r2_beat*8]};
        rlast2  <= (r2_beat == r2_total - 9'd1);
      end else if (rvalid2 && rready2) begin
        rvalid2 <= 1'b0;
        if (rlast2) begin r2_run <= 1'b0; arready2 <= 1'b1; end
        else r2_beat <= r2_beat + 9'd1;
      end
    end
  end

  // AW/W/B 通道（与底本一致）
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
    // A：pf_v_x=1 期间到达的 T_EXEC（未消费，pf_hit_r_x 尚未置位）必是 pf_pc_x 处的 OP_LOAD
    if (dut.u_sched.pf_v_c && dut.u_sched.st == 4'd3 && !dut.u_sched.pf_hit_r_c
        && !(dut.u_sched.pc == dut.u_sched.pf_pc_c
             && dut.u_sched.desc_r[255:252] == 4'd4)) begin
      $display("[pf] FATAL: pf_v_c=1 但 T_EXEC pc=%0d 不是预取目标 OP_LOAD(pf_pc_c=%0d)",
               dut.u_sched.pc, dut.u_sched.pf_pc_c);
      $fatal(1);
    end
    if (dut.u_sched.pf_v_w && dut.u_sched.st == 4'd3 && !dut.u_sched.pf_hit_r_w
        && !(dut.u_sched.pc == dut.u_sched.pf_pc_w
             && dut.u_sched.desc_r[255:252] == 4'd4)) begin
      $display("[pf] FATAL: pf_v_w=1 但 T_EXEC pc=%0d 不是预取目标 OP_LOAD(pf_pc_w=%0d)",
               dut.u_sched.pc, dut.u_sched.pf_pc_w);
      $fatal(1);
    end
    // B：后台 DMA 在飞期间不得进入 COPY（两者都写 WRAM B 口，会丢后台写数据）
    if ((dut.u_sched.pf_v_c || dut.u_sched.pf_v_w) && dut.u_sched.st == 4'd6) begin
      $display("[pf] FATAL: pf_v=1 期间进入 T_RUN_CP（COPY 与后台 DMA WRAM 写冲突）");
      $fatal(1);
    end
    // C（★ v5-R2 新增）：前台 ctx 装载不得与串行引擎重叠——pf_ctx_stall 只覆盖
    //    后台命令；前台若重叠，B 口 mux 会静默丢写（消费点等待被破坏时才会发生）
    if (dut.rd_c_busy && !dut.rd_c_bg &&
        (dut.eng_g || dut.eng_sm || dut.eng_a)) begin
      $display("[pf] FATAL: 前台 ctx 装载与 eng_g/sm/a 重叠（t=%0t）", $time);
      $fatal(1);
    end
  end

  // 预取活动探针（PF_DBG 置 1 开启）
  localparam bit PF_DBG = 1'b0;
  always @(posedge clk) if (PF_DBG && rst_n) begin
    if (dut.u_sched.pf_bg_start)
      $display("[pfdbg] bg_start pf_pc(c=%0d,w=%0d) addr=%h len=%0d tag=%0d base=%0d",
               dut.u_sched.pf_pc_c, dut.u_sched.pf_pc_w, dut.pf_dmaaddr, dut.pf_dmalen,
               dut.pf_dmatag, dut.pf_dmabase);
    if (dut.u_sched.st == 4'd3 && dut.u_sched.dma_busy
        && dut.u_sched.d_op >= 4'd4 && dut.u_sched.d_op <= 4'd5
        && !((dut.u_sched.pf_v_c && dut.u_sched.pf_pc_c == dut.u_sched.pc)
             || (dut.u_sched.pf_v_w && dut.u_sched.pf_pc_w == dut.u_sched.pc)))
      $display("[pfdbg] T_EXEC stall @pc=%0d op=%0d (引擎忙等待)",
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

  // ★ v5-R2 探针：ctx 引擎后台预取被 B 口让拍冻结的拍数（底本 pf_ctx_drop_cnt
  //   的对应物——双引擎版该场景由 pf_ctx_stall 正确冻结，计数应为 0 丢写、
  //   >0 只是让拍等待）+ 双引擎占用/重叠计数（收益读数）
  localparam bit PF_DROP_DBG = 1'b1;
  logic [31:0] pf_stall_cnt;
  logic [31:0] rd_c_busy_cnt, rd_w_busy_cnt, rd_both_cnt, rd_union_cnt;
  always @(posedge clk) if (PF_DROP_DBG && rst_n) begin
    if (dut.pf_ctx_stall) pf_stall_cnt <= pf_stall_cnt + 32'd1;
    if (dut.rd_c_busy) rd_c_busy_cnt <= rd_c_busy_cnt + 32'd1;
    if (dut.rd_w_busy) rd_w_busy_cnt <= rd_w_busy_cnt + 32'd1;
    if (dut.rd_c_busy && dut.rd_w_busy) rd_both_cnt <= rd_both_cnt + 32'd1;
    if (dut.rd_c_busy || dut.rd_w_busy) rd_union_cnt <= rd_union_cnt + 32'd1;
  end

  // ---------------- 一次完整运行 ----------------
  task automatic run_mode(input bit prim, input bit pf, input int tag);
    int f;
    // LFSR 停顿序列对齐（两套从机一起快照/装载）
    if (tag == 1) begin lfsr_mark = lfsr_nxt; lfsr2_mark = lfsr2_nxt; end
    if (tag == 2) lfsr_load = 1'b1;
    // 全量复位：CTX/WRAM 清零，DDR 重载初值
    zc_pulse = 1; zw_pulse = 1; #1; zc_pulse = 0; zw_pulse = 0;
    for (int i = 0; i < DDR_BYTES; i++) ddr[i] = ddr0[i];

    rst_n = 0; repeat (4) @(posedge clk);
    hoist_en = prim;
    pf_en = pf;
    pf_stall_cnt = 0;
    rd_c_busy_cnt = 0; rd_w_busy_cnt = 0; rd_both_cnt = 0; rd_union_cnt = 0;
    rst_n = 1; repeat (4) @(posedge clk);
    start = 1; @(posedge clk); start = 0;
    wait (done);
    @(posedge clk);
    $display("[tb] %s%s: cycles=%0d gemm=%0d dma=%0d mac_total=%0d skip_macs=%0d skip_stages=%0d",
             prim ? "PRIM" : "REF ", pf ? "-pf1" : "-pf0", cycles, gemm_cycles,
             dma_cycles, mac_total, skip_macs, skip_stages);
    if (PF_DROP_DBG)
      $display("[probe] pf_stall=%0d rd_c=%0d rd_w=%0d both=%0d union=%0d overlap=%0.1f%%",
               pf_stall_cnt, rd_c_busy_cnt, rd_w_busy_cnt, rd_both_cnt,
               rd_union_cnt,
               rd_union_cnt > 0 ? 100.0 * rd_both_cnt / rd_union_cnt : 0.0);

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
    run_mode(1'b1, 1'b1, 2);   // PRIMITIVE + pf_en=1：预取开（dump 应与上一遍逐位一致）
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
