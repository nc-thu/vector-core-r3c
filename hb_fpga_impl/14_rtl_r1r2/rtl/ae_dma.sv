// ae_dma.sv — AXI4 64-bit 主口 DMA（读装载 + 写回）
// R1 改造（2026-08-31）：单 FSM 拆成两个并发 FSM：
//   读引擎 rd（AR/R 通道，服务 LOAD_CTX/LOAD_W）+ 写引擎 wr（AW/W/B 通道，服务 STORE）。
//   两引擎物理独立（AR/R 与 AW/W/B 是 AXI 全双工分离通道），可真并发。
//   CTX 端口也分离：LOAD 写 B 口、STORE 读 A 口，无端口冲突。
//   背靠背防护：两引擎的 AXI 命令通道各自独立，互不干扰。
// LOAD: DDR 突发读，每拍 8 字节一次性路由（8 lane 对齐写，1 B/cycle = 2 GB/s @250MHz）：
//   TAG_CTX: 激活 DDR 流为 [k][m16]（16B 组对齐）-> lane = byi mod 16（组内 0/8 起步，
//            8 字节不跨组），addr = base + byi/16
//   TAG_W  : 权重 DDR 流为 [k][j] -> lane = j 循环 0..COLS-1，addr = base + k；
//            8 字节 run 跨过 COLS 边界时拆 2 拍（D_R2），平均 ~1.07 拍/beat
// STORE: CTX A 口 16B 读 -> 2 个 64b W 拍 -> DDR 写（长度须 16 的倍数）
// 突发按 256 拍（2048B）切分。
`ifndef AE_DMA_SV
`define AE_DMA_SV
module ae_dma #(
  parameter int COLS = 108
)(
  input  logic clk,
  input  logic rst_n,
  // 前台命令（调度器发）：start 触发，is_wr=0 走 rd 引擎，is_wr=1 走 wr 引擎
  input  logic start,
  output logic busy,
  output logic done,
  input  logic [31:0] cmd_addr,
  input  logic [17:0] cmd_len,
  input  logic        cmd_is_wr,
  input  logic [2:0]  cmd_tag,
  input  logic [19:0] cmd_base,
  // 后台预取命令（pf_bg_start 脉冲，走 rd 引擎）
  input  logic        bg_start,
  input  logic [31:0] bg_addr,
  input  logic [17:0] bg_len,
  input  logic [2:0]  bg_tag,
  input  logic [19:0] bg_base,
  // 各引擎独立状态（供调度器并发控制）
  output logic rd_busy, rd_done,
  output logic wr_busy, wr_done,
  output logic        wr_start_o,   // 写引擎实际启动脉冲（调试用）
  // R1：CTX A 口仲裁——GEMM/softmax/ACTV/COPY 串行引擎占用 A 口时，
  //   STORE 写引擎必须停拍（否则读到 GEMM 地址的数据，写回 DDR 出错）
  input  logic        ctxa_wr_bank,
  // ★ R2 修复：CTX 预取写停拍反馈。ae_core 在后台 CTX 预取写会被
  //   B 口优先级压掉时（B 口被 GEMM/softmax/ACTV/前台LOAD 占）拉高，
  //   rd 引擎在 R_R 状态不拉 rready，等 B 口让出再继续吃 AXI R 拍。
  //   不加这条反馈时：rd 引擎照常推进 r_byi，ctx_we=1 但 mux 没选中 bg_wran
  //   分支 → 写静默丢失，DMA 仍 fire rd_done → 调度器以为预取完成但 CTX 残缺。
  input  logic        pf_ctx_stall,
  // ★ R2 修复：当前 rd 引擎锁存的 tag（供 ae_core 判断是否 CTX 预取）
  output logic [2:0]  rd_tag_o,
  // AXI4 读通道（rd 引擎独占）
  output logic [31:0] araddr,
  output logic [7:0]  arlen,
  output logic        arvalid,
  input  logic        arready,
  input  logic [63:0] rdata,
  input  logic        rvalid,
  input  logic        rlast,
  output logic        rready,
  // AXI4 写通道（wr 引擎独占）
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
  // CTX B 口（rd 引擎 LOAD 写）
  output logic        ctx_we,
  output logic [15:0] ctx_welane,
  output logic [19:0] ctx_addr,
  output logic [16*8-1:0] ctx_wdata,
  // CTX A 口（wr 引擎 STORE 读）
  output logic [19:0] ctx_raddr,
  input  logic [16*8-1:0] ctx_rdata,
  // WRAM B 口（rd 引擎写）
  output logic [COLS-1:0] wr_we,
  output logic [11:0] wr_addr,
  output logic [COLS*8-1:0] wr_wdata
);
  // =========================================================================
  // 读引擎：LOAD_CTX / LOAD_W（AR/R 通道 + CTX B 口 + WRAM）
  // =========================================================================
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

  wire r_xing = (r_tag_r != 3'd0) && ({1'b0, r_wj} + 9'd8 > {{1'b0, COLS[7:0]}});

  // ★ R2 修复：CTX 预取写停拍——B 口被占时不拉 rready，AXI R 拍不推进，
  //   r_byi/r_remain 全部冻结在原拍，等 ae_core 让出 B 口再继续。
  //   只影响 TAG_CTX 后台预取（pf_ctx_stall 只在那种条件下被 core 拉高）。
  assign rready  = (rd_st == R_R) && !pf_ctx_stall;
  assign arlen   = (r_chunk_b[31:11] != 21'd0) ? 8'd255 :
                   ((r_chunk_b[10:3] == 9'd0) ? 8'd0 : r_chunk_b[10:3] - 9'd1);

  assign rd_busy = (rd_st != R_IDLE);
  assign rd_tag_o = r_tag_r;   // ★ R2：供 ae_core 判断当前 LOAD 是否 CTX 预取

  // 命令锁存：前台 start 或后台 bg_start（两者不会同拍——调度器保证）
  wire rd_go = start & ~cmd_is_wr | bg_start;
  wire [31:0] rd_cmd_addr = bg_start ? bg_addr  : cmd_addr;
  wire [17:0] rd_cmd_len  = bg_start ? bg_len   : cmd_len;
  wire [2:0]  rd_cmd_tag  = bg_start ? bg_tag   : cmd_tag;
  wire [19:0] rd_cmd_base = bg_start ? bg_base  : cmd_base;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd_st <= R_IDLE; rd_done <= 1'b0; r_last_rlast <= 1'b0;
      arvalid <= 1'b0;
      ctx_we <= 1'b0; ctx_welane <= '0; ctx_addr <= '0; ctx_wdata <= '0;
      wr_we <= '0; wr_addr <= '0; wr_wdata <= '0;
    end else begin
      rd_done <= 1'b0; arvalid <= 1'b0;
      ctx_we <= 1'b0; ctx_welane <= '0; wr_we <= '0;
      case (rd_st)
        R_IDLE: if (rd_go) begin
            r_addr_r <= rd_cmd_addr; r_remain <= {14'd0, rd_cmd_len};
            r_base_r <= rd_cmd_base; r_tag_r <= rd_cmd_tag;
            r_chunk_b <= (rd_cmd_len > 18'd2048) ? 32'd2048 : {14'd0, rd_cmd_len};
            r_byi <= '0; r_wk <= '0; r_wj <= '0;
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
        R_FIN: begin rd_done <= 1'b1; rd_st <= R_IDLE; end
        default: rd_st <= R_IDLE;
      endcase
    end
  end

  // =========================================================================
  // 写引擎：STORE（AW/W/B 通道 + CTX A 口读）
  // R1：CTX A 口与 GEMM 分时复用。写引擎在 W_RD 拍呈现地址并等 A 口空闲，
  //   W_RD2 拍采样 ctx_rdata（上一拍 W_RD 的读结果），W_W 拍写 AXI。
  //   GEMM 占 A 口时（ctxa_wr_bank=0）停拍，避免读到 GEMM 地址的数据。
  // =========================================================================
  typedef enum logic [2:0] {W_IDLE, W_AW, W_RD, W_RD2, W_W, W_B, W_FIN} wr_st_e;
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
  // busy: 任一引擎在跑（调度器用它判断 DMA 是否空闲）
  assign busy = rd_busy | wr_busy;
  // done: 当前命令完成。前台 LOAD → rd_done；前台 STORE → wr_done；
  //       后台预取 → rd_done（pf_bg_start 走 rd 引擎）
  //       调度器在前台命令路径用 done，后台路径用 rd_done（pf_done 逻辑在调度器里）
  assign done = cmd_is_wr ? wr_done : rd_done;

endmodule
`endif
