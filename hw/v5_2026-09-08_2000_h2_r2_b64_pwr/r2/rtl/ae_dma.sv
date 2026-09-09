// ae_dma.sv — AXI4 64-bit 主口 DMA（读装载 + 写回）★ v5-R2 双读引擎版
// ---------------------------------------------------------------------------
// 相对底本 hb_fpga_impl/22_r3c_rtl/rtl/ae_dma.sv 的 diff（2026-09-08 20:09）：
//   1. 原单台读引擎 FSM（rd_st，R_IDLE/R_AR/R_R/R_R2/R_R_FIN）逐字提取为
//      ae_rd_eng 模块（逻辑零改动，含 pf_ctx_stall 停拍与 D_R2 跨组拆拍），
//      例化两台：
//        u_rd_c —— 专职 TAG_CTX（tag==0，激活 → CTX B 口），走 AXI 读口 1
//        u_rd_w —— 专职 TAG_W  （tag!=0，权重 → WRAM B 口），走 AXI 读口 2
//      每台各自独占一组 AR/R 通道（共享一条总线做字节级交织拿不到收益，
//      见 r2/notes/findings.md 排队点二）。
//   2. 命令路由：前台 start&~cmd_is_wr 按 cmd_tag 分发；后台 bg_start 按
//      bg_tag 分发。两台引擎各自有 busy/done（rd_c_*/rd_w_*），对外保留
//      聚合 rd_busy/rd_done（= 或）向后兼容。
//   3. CTX 侧端口（ctx_we/welane/addr/wdata）只来自 u_rd_c；WRAM 侧端口
//      （wr_we/wr_addr/wr_wdata）只来自 u_rd_w。写引擎（STORE）原样未动。
//   4. rd_tag_o 改为 u_rd_c 的 tag（CTX 引擎在飞 tag，供 ae_core 判断
//      CTX 预取让拍）；新增 rd_c_bg（ctx 引擎当前命令是否后台预取）。
// 写回引擎、wr 通道、CTX A 口读、burst 切分（256 拍/2048B）全部与底本一致。
// ---------------------------------------------------------------------------
`ifndef AE_DMA_SV
`define AE_DMA_SV

// ===========================================================================
// ae_rd_eng — 单台读引擎（FSM 逻辑 = 底本 ae_dma.sv 读引擎逐字搬运）
//   命令口：go 单拍脉冲 + addr/len/tag/base；数据口按 tag 二选一：
//     tag==0 → CTX B 口字节通道写；tag!=0 → WRAM 列写（含跨 COLS 拆拍）
//   pf_ctx_stall：CTX 预取写被 B 口优先级压掉时停拍（底本 R2 修复，原样保留；
//     双引擎版只接在 ctx 引擎上，w 引擎恒 0）
// ===========================================================================
module ae_rd_eng #(
  parameter int COLS = 96
)(
  input  logic clk,
  input  logic rst_n,
  input  logic        go,
  input  logic [31:0] cmd_addr,
  input  logic [17:0] cmd_len,
  input  logic [2:0]  cmd_tag,
  input  logic [19:0] cmd_base,
  output logic busy,
  output logic done,
  output logic [2:0]  tag_o,
  output logic        is_bg,        // 当前命令来自 bg 命令口（ae_dma 包装层注入）
  input  logic        pf_ctx_stall,
  // AXI4 读通道（本引擎独占）
  output logic [31:0] araddr,
  output logic [7:0]  arlen,
  output logic        arvalid,
  input  logic        arready,
  input  logic [63:0] rdata,
  input  logic        rvalid,
  input  logic        rlast,
  output logic        rready,
  // CTX B 口（tag==0 命令写）
  output logic        ctx_we,
  output logic [15:0] ctx_welane,
  output logic [19:0] ctx_addr,
  output logic [16*8-1:0] ctx_wdata,
  // WRAM B 口（tag!=0 命令写）
  output logic [COLS-1:0] wr_we,
  output logic [11:0] wr_addr,
  output logic [COLS*8-1:0] wr_wdata
);
  typedef enum logic [2:0] {R_IDLE, R_AR, R_R, R_R2, R_FIN} rd_st_e;
  rd_st_e rd_st;

  logic [31:0] r_addr_r, r_remain;
  logic [31:0] r_chunk_b;
  logic [19:0] r_base_r;
  logic [2:0]  r_tag_r;
  logic [63:0] r_beat_buf;
  logic        r_last_rlast;
  logic [17:0] r_byi;
  logic [11:0] r_wk;
  logic [7:0]  r_wj;
  logic [3:0]  r_cross_n;
  logic        r_bg_r;              // 当前命令是否后台（is_bg 打拍）

  wire r_xing = (r_tag_r != 3'd0) && ({1'b0, r_wj} + 9'd8 > {{1'b0, COLS[7:0]}});

  assign rready  = (rd_st == R_R) && !pf_ctx_stall;
  assign arlen   = (r_chunk_b[31:11] != 21'd0) ? 8'd255 :
                   ((r_chunk_b[10:3] == 9'd0) ? 8'd0 : r_chunk_b[10:3] - 9'd1);

  assign busy = (rd_st != R_IDLE);
  assign tag_o = r_tag_r;
  assign is_bg = r_bg_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd_st <= R_IDLE; done <= 1'b0; r_last_rlast <= 1'b0;
      arvalid <= 1'b0;
      ctx_we <= 1'b0; ctx_welane <= '0; ctx_addr <= '0; ctx_wdata <= '0;
      wr_we <= '0; wr_addr <= '0; wr_wdata <= '0;
    end else begin
      done <= 1'b0; arvalid <= 1'b0;
      ctx_we <= 1'b0; ctx_welane <= '0; wr_we <= '0;
      case (rd_st)
        R_IDLE: if (go) begin
            r_addr_r <= cmd_addr; r_remain <= {14'd0, cmd_len};
            r_base_r <= cmd_base; r_tag_r <= cmd_tag;
            r_chunk_b <= (cmd_len > 18'd2048) ? 32'd2048 : {14'd0, cmd_len};
            r_byi <= '0; r_wk <= '0; r_wj <= '0;
            r_bg_r <= is_bg;
            rd_st <= R_AR;
          end
        R_AR: begin
            if (!arvalid) arvalid <= 1'b1;
            araddr <= r_addr_r;
            if (arvalid && arready) rd_st <= R_R;
          end
        R_R: if (rvalid && rready) begin
            r_remain <= r_remain - 32'd8;
            r_byi <= r_byi + 18'd8;
            if (r_tag_r == 3'd0) begin
              ctx_we <= 1'b1;
              ctx_addr <= r_base_r + {2'd0, r_byi[17:4]};
              for (int q = 0; q < 8; q++) begin
                ctx_welane[r_byi[3:0] + q[3:0]] <= 1'b1;
                ctx_wdata[(r_byi[3:0] + q[3:0])*8 +: 8] <= rdata[q*8 +: 8];
              end
            end else if (!r_xing) begin
              wr_addr <= r_base_r[11:0] + r_wk;
              for (int q = 0; q < 8; q++) begin
                wr_we[r_wj + q[7:0]] <= 1'b1;
                wr_wdata[(r_wj + q[7:0])*8 +: 8] <= rdata[q*8 +: 8];
              end
              if (r_wj + 8'd8 == COLS[7:0]) begin r_wj <= '0; r_wk <= r_wk + 12'd1; end
              else r_wj <= r_wj + 8'd8;
            end else begin
              r_cross_n <= COLS[3:0] - r_wj[3:0];
              r_beat_buf <= rdata;
              wr_addr <= r_base_r[11:0] + r_wk;
              for (int jj = 0; jj < 256; jj++) begin
                if (jj[7:0] >= r_wj && jj < COLS) begin
                  wr_we[jj] <= 1'b1;
                  wr_wdata[jj*8 +: 8] <= rdata[(jj - r_wj)*8 +: 8];
                end
              end
              r_wk <= r_wk + 12'd1;
              rd_st <= R_R2;
            end
            if (rlast) begin
              if (r_xing) r_last_rlast <= 1'b1;
              else if (r_remain <= 32'd8) rd_st <= R_FIN;
              else begin
                r_addr_r  <= r_addr_r + 32'd2048;
                r_chunk_b <= (r_remain - 32'd8 > 32'd2048) ? 32'd2048 : (r_remain - 32'd8);
                rd_st <= R_AR;
              end
            end
          end
        R_R2: begin
            wr_addr <= r_base_r[11:0] + r_wk[11:0];
            for (int jj = 0; jj < 8; jj++) begin
              if (jj[7:0] < r_wj + 8'd8 - COLS[7:0]) begin
                wr_we[jj] <= 1'b1;
                wr_wdata[jj*8 +: 8] <= r_beat_buf[(r_cross_n + jj[3:0])*8 +: 8];
              end
            end
            r_wj <= r_wj + 8'd8 - COLS[7:0];
            if (r_last_rlast) begin
              r_last_rlast <= 1'b0;
              if (r_remain == 32'd0) rd_st <= R_FIN;
              else begin
                r_addr_r  <= r_addr_r + 32'd2048;
                r_chunk_b <= (r_remain > 32'd2048) ? 32'd2048 : r_remain;
                rd_st <= R_AR;
              end
            end else rd_st <= R_R;
          end
        R_FIN: begin done <= 1'b1; rd_st <= R_IDLE; end
        default: rd_st <= R_IDLE;
      endcase
    end
  end
endmodule

// ===========================================================================
// ae_dma — 双读引擎 + 写引擎 包装层
// ===========================================================================
module ae_dma #(
  parameter int COLS = 96
)(
  input  logic clk,
  input  logic rst_n,
  // 前台命令（调度器发）：start 触发，is_wr=0 走读引擎，is_wr=1 走写引擎
  input  logic start,
  output logic busy,
  output logic done,
  input  logic [31:0] cmd_addr,
  input  logic [17:0] cmd_len,
  input  logic        cmd_is_wr,
  input  logic [2:0]  cmd_tag,
  input  logic [19:0] cmd_base,
  // 后台预取命令（pf_bg_start 脉冲，按 bg_tag 路由到对应读引擎）
  input  logic        bg_start,
  input  logic [31:0] bg_addr,
  input  logic [17:0] bg_len,
  input  logic [2:0]  bg_tag,
  input  logic [19:0] bg_base,
  // 各引擎独立状态（供调度器并发控制）
  // ★ v5-R2：rd 拆两台——rd_c（TAG_CTX）/ rd_w（TAG_W）；rd_busy/rd_done
  //   保留为聚合值（兼容旧口），新增 rd_c_*/rd_w_* 供调度器按流发射
  output logic rd_busy, rd_done,
  output logic rd_c_busy, rd_c_done,
  output logic rd_w_busy, rd_w_done,
  output logic wr_busy, wr_done,
  output logic        wr_start_o,   // 写引擎实际启动脉冲（调试用）
  // R1：CTX A 口仲裁——GEMM/softmax/ACTV/COPY 串行引擎占用 A 口时，
  //   STORE 写引擎必须停拍（否则读到 GEMM 地址的数据，写回 DDR 出错）
  input  logic        ctxa_wr_bank,
  // ★ R2 修复（底本保留）：CTX 预取写停拍反馈，只作用于 ctx 读引擎
  input  logic        pf_ctx_stall,
  // ★ v5-R2：ctx 读引擎状态（tag + 是否后台），供 ae_core 做 B 口仲裁
  output logic [2:0]  rd_tag_o,
  output logic        rd_c_bg,
  // AXI4 读通道 1（ctx 读引擎独占）
  output logic [31:0] araddr,
  output logic [7:0]  arlen,
  output logic        arvalid,
  input  logic        arready,
  input  logic [63:0] rdata,
  input  logic        rvalid,
  input  logic        rlast,
  output logic        rready,
  // AXI4 读通道 2（w 读引擎独占）★ v5-R2 新增
  output logic [31:0] araddr2,
  output logic [7:0]  arlen2,
  output logic        arvalid2,
  input  logic        arready2,
  input  logic [63:0] rdata2,
  input  logic        rvalid2,
  input  logic        rlast2,
  output logic        rready2,
  // AXI4 写通道（wr 引擎独占，与底本一致）
  output logic [31:0] awaddr,
  output logic [7:0]  awlen,
  output logic        awvalid,
  input  logic        awready,
  output logic [63:0] wdata,
  output logic [7:0]  wstrb,
  output logic        wlast,
  output logic        wvalid,
  input  logic        wready,
  input  logic        bvalid,
  output logic        bready,
  // CTX B 口（ctx 读引擎写）
  output logic        ctx_we,
  output logic [15:0] ctx_welane,
  output logic [19:0] ctx_addr,
  output logic [16*8-1:0] ctx_wdata,
  // CTX A 口（wr 引擎 STORE 读）
  output logic [19:0] ctx_raddr,
  input  logic [16*8-1:0] ctx_rdata,
  // WRAM B 口（w 读引擎写）
  output logic [COLS-1:0] wr_we,
  output logic [11:0] wr_addr,
  output logic [COLS*8-1:0] wr_wdata
);
  // =========================================================================
  // 读引擎命令路由（v5-R2）：前台按 cmd_tag、后台按 bg_tag 二选一
  //   tag==0 → u_rd_c（CTX）；tag!=0 → u_rd_w（WRAM）
  //   （前台与后台命令同拍到达时后台让前台：调度器保证不发生，保持底本语义）
  // =========================================================================
  // bg 脉冲拍让位给后台命令（底本 rd_cmd_* = bg_start ? bg_* : cmd_* 同义：
  // 调度器把 dma_start|pf_bg_start 一起 OR 进 start，bg_start 拍必须选 bg 命令组）
  wire fg_rd   = start & ~cmd_is_wr & ~bg_start;
  wire fg_to_c = fg_rd & (cmd_tag == 3'd0);
  wire fg_to_w = fg_rd & (cmd_tag != 3'd0);
  wire bg_to_c = bg_start & (bg_tag == 3'd0);
  wire bg_to_w = bg_start & (bg_tag != 3'd0);

  ae_rd_eng #(.COLS(COLS)) u_rd_c (
    .clk(clk), .rst_n(rst_n),
    .go(fg_to_c | bg_to_c),
    .cmd_addr(fg_to_c ? cmd_addr : bg_addr),
    .cmd_len (fg_to_c ? cmd_len  : bg_len),
    .cmd_tag (fg_to_c ? cmd_tag  : bg_tag),
    .cmd_base(fg_to_c ? cmd_base : bg_base),
    .busy(rd_c_busy), .done(rd_c_done),
    .tag_o(rd_tag_o), .is_bg(rd_c_bg),
    .pf_ctx_stall(pf_ctx_stall),
    .araddr(araddr), .arlen(arlen), .arvalid(arvalid), .arready(arready),
    .rdata(rdata), .rvalid(rvalid), .rlast(rlast), .rready(rready),
    .ctx_we(ctx_we), .ctx_welane(ctx_welane),
    .ctx_addr(ctx_addr), .ctx_wdata(ctx_wdata),
    .wr_we(), .wr_addr(), .wr_wdata()      // ctx 引擎不写 WRAM（悬空）
  );

  ae_rd_eng #(.COLS(COLS)) u_rd_w (
    .clk(clk), .rst_n(rst_n),
    .go(fg_to_w | bg_to_w),
    .cmd_addr(fg_to_w ? cmd_addr : bg_addr),
    .cmd_len (fg_to_w ? cmd_len  : bg_len),
    .cmd_tag (fg_to_w ? cmd_tag  : bg_tag),
    .cmd_base(fg_to_w ? cmd_base : bg_base),
    .busy(rd_w_busy), .done(rd_w_done),
    .tag_o(), .is_bg(),
    .pf_ctx_stall(1'b0),                   // w 引擎写 WRAM，无 CTX B 口让拍
    .araddr(araddr2), .arlen(arlen2), .arvalid(arvalid2), .arready(arready2),
    .rdata(rdata2), .rvalid(rvalid2), .rlast(rlast2), .rready(rready2),
    .ctx_we(), .ctx_welane(), .ctx_addr(), .ctx_wdata(),   // w 引擎不写 CTX
    .wr_we(wr_we), .wr_addr(wr_addr), .wr_wdata(wr_wdata)
  );

  // =========================================================================
  // 写引擎：STORE（AW/W/B 通道 + CTX A 口读）—— 底本逐字保留
  // =========================================================================
  typedef enum logic [3:0] {W_IDLE, W_AW, W_RD, W_RD2, W_W, W_B, W_FIN} wr_st_e;
  wr_st_e wr_st;

  logic [31:0] w_addr_r, w_remain;
  logic [31:0] w_chunk_b;
  logic [19:0] w_base_r;
  logic [31:0] w_wbeat, w_wbeat_chunk;
  logic [127:0] w_rd16;
  logic        ctxa_bank_q;   // 上一拍 A 口是否归 STORE（用于 W_RD2 停拍判定）

  assign bready  = (wr_st == W_B);
  assign wstrb   = 8'hFF;
  assign awlen   = (w_chunk_b[31:11] != 21'd0) ? 8'd255 :
                   ((w_chunk_b[10:3] == 9'd0) ? 8'd0 : w_chunk_b[10:3] - 9'd1);
  assign ctx_raddr = w_base_r + w_wbeat[19:1];   // 每 2 拍一行 16B

  assign wr_busy = (wr_st != W_IDLE);

  // 停拍条件：W_RD 拍地址需稳定 → ctxa_wr_bank=1；W_RD2 拍采样需上一拍地址有效 → ctxa_bank_q=1
  wire wr_rd_stall  = (wr_st == W_RD)  && !ctxa_wr_bank;
  wire wr_rd2_stall = (wr_st == W_RD2) && !ctxa_bank_q;

  // 前台 STORE 命令（start & is_wr）
  wire wr_go = start & cmd_is_wr;
  assign wr_start_o = wr_go;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_st <= W_IDLE; wr_done <= 1'b0;
      awvalid <= 1'b0; wvalid <= 1'b0; wlast <= 1'b0;
      w_wbeat <= '0; w_wbeat_chunk <= '0;
      ctxa_bank_q <= 1'b0;
    end else begin
      wr_done <= 1'b0; awvalid <= 1'b0; wvalid <= 1'b0; wlast <= 1'b0;
      ctxa_bank_q <= ctxa_wr_bank;   // 每拍跟踪 A 口归属
      case (wr_st)
        W_IDLE: if (wr_go) begin
            w_addr_r <= cmd_addr; w_remain <= {14'd0, cmd_len};
            w_base_r <= cmd_base;
            w_chunk_b <= (cmd_len > 18'd2048) ? 32'd2048 : {14'd0, cmd_len};
            w_wbeat <= '0; w_wbeat_chunk <= '0;
            wr_st <= W_AW;
          end
        W_AW: begin
            if (!awvalid) awvalid <= 1'b1;
            awaddr <= w_addr_r;
            if (awvalid && awready) wr_st <= W_RD;
          end
        W_RD: if (!wr_rd_stall) wr_st <= W_RD2;
        W_RD2: begin
            if (!wr_rd2_stall) begin
              w_rd16 <= ctx_rdata;
              wr_st <= W_W;
            end
            // else: 上一拍 A 口被 GEMM 抢占，原地等
          end
        W_W: begin
            if (!wvalid) wvalid <= 1'b1;
            wdata <= w_wbeat[0] ? w_rd16[127:64] : w_rd16[63:0];
            wlast <= (w_wbeat_chunk == w_chunk_b[31:3] - 32'd1);
            if (wvalid && wready) begin
              w_wbeat <= w_wbeat + 32'd1;
              w_wbeat_chunk <= w_wbeat_chunk + 32'd1;
              w_remain <= w_remain - 32'd8;
              if (w_wbeat[0] == 1'b1) begin
                wvalid <= 1'b0;
                if (w_remain <= 32'd8) wr_st <= W_B;
                else if (w_wbeat_chunk + 32'd1 == w_chunk_b[31:3]) begin
                  w_addr_r  <= w_addr_r + 32'd2048;
                  w_chunk_b <= (w_remain - 32'd8 > 32'd2048) ? 32'd2048 : (w_remain - 32'd8);
                  w_wbeat_chunk <= '0;
                  wr_st <= W_AW;
                end else wr_st <= W_RD;
              end
            end
          end
        W_B: if (bvalid) begin
            if (w_remain == 32'd0) wr_st <= W_FIN;
            else begin
              w_addr_r <= w_addr_r + 32'd2048;
              w_chunk_b <= (w_remain > 32'd2048) ? 32'd2048 : w_remain;
              w_wbeat_chunk <= '0;
              wr_st <= W_AW;
            end
          end
        W_FIN: begin wr_done <= 1'b1; wr_st <= W_IDLE; end
        default: wr_st <= W_IDLE;
      endcase
    end
  end

  // =========================================================================
  // 顶层汇总信号（向后兼容调度器接口）
  // =========================================================================
  assign rd_busy = rd_c_busy | rd_w_busy;
  assign rd_done = rd_c_done | rd_w_done;
  assign busy = rd_busy | wr_busy;
  // done：前台命令完成——STORE → wr_done；LOAD → 命令 tag 对应引擎的 done
  //   （cmd_is_wr/cmd_tag 是活线，但 T_RUN_DMA 等待期间描述符 desc_r 稳定，
  //    与底本 done 依赖 cmd_is_wr 活线同一口径）
  assign done = cmd_is_wr ? wr_done :
                (cmd_tag == 3'd0) ? rd_c_done : rd_w_done;

endmodule
`endif
