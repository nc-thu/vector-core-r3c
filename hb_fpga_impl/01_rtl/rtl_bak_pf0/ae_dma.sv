// ae_dma.sv — AXI4 64-bit 主口 DMA（读装载 + 写回）
// LOAD: DDR 突发读，每拍 8 字节一次性路由（8 lane 对齐写，1 B/cycle = 2 GB/s @250MHz）：
//   TAG_CTX: 激活 DDR 流为 [k][m16]（16B 组对齐）-> lane = byi mod 16（组内 0/8 起步，
//            8 字节不跨组），addr = base + byi/16
//   TAG_W  : 权重 DDR 流为 [k][j] -> lane = j 循环 0..COLS-1，addr = base + k；
//            8 字节 run 跨过 COLS 边界时拆 2 拍（D_R2），平均 ~1.07 拍/beat
// STORE: CTX B 口 16B 读 -> 2 个 64b W 拍 -> DDR 写（长度须 16 的倍数）
// 突发按 256 拍（2048B）切分。v1 无读预取重叠（流量为计算受限的次要项）。
`ifndef AE_DMA_SV
`define AE_DMA_SV
module ae_dma #(
  parameter int COLS = 108
)(
  input  logic clk,
  input  logic rst_n,
  input  logic start,
  output logic busy,
  output logic done,
  input  logic [31:0] cmd_addr,
  input  logic [17:0] cmd_len,     // 字节数（LOAD 须 8 的倍数；STORE 须 16 的倍数）
  input  logic        cmd_is_wr,
  input  logic [2:0]  cmd_tag,     // 0=CTX 1=W
  input  logic [19:0] cmd_base,
  // AXI4 读通道
  output logic [31:0] araddr,
  output logic [7:0]  arlen,
  output logic        arvalid,
  input  logic        arready,
  input  logic [63:0] rdata,
  input  logic        rvalid,
  input  logic        rlast,
  output logic        rready,
  // AXI4 写通道
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
  // CTX B 口
  output logic        ctx_we,
  output logic [15:0] ctx_welane,
  output logic [19:0] ctx_addr,
  output logic [16*8-1:0] ctx_wdata,
  output logic [19:0] ctx_raddr,
  input  logic [16*8-1:0] ctx_rdata,
  // WRAM B 口（写）
  output logic [COLS-1:0] wr_we,
  output logic [11:0] wr_addr,
  output logic [COLS*8-1:0] wr_wdata
);
  typedef enum logic [3:0] {D_IDLE, D_AR, D_R, D_R2, D_AW, D_RD, D_RD2, D_W, D_B,
                            D_FIN} st_e;
  st_e st;

  logic [31:0] addr_r, remain;
  logic [31:0] chunk_b;                // 本突发字节数
  logic [19:0] base_r;
  logic [2:0]  tag_r;
  logic [63:0] beat_buf;               // D_R2 跨界残拍缓冲
  logic        last_rlast;             // 跨界拍恰为突发末拍：残拍后处理边界
  // 增量路由计数
  logic [17:0] byi;                    // CTX: 全局字节计数（lane=byi mod 16）
  logic [11:0] wk;                     // WRAM: k 计数
  logic [7:0]  wj;                     // WRAM: j 计数（0..COLS-1，COLS<=256）
  logic [3:0]  cross_n;                  // 跨界首拍字节数
  // STORE
  logic [31:0] wbeat, wbeat_chunk;
  logic [127:0] rd16;

  wire xing = (tag_r != 3'd0) && ({1'b0, wj} + 9'd8 > {{1'b0, COLS[7:0]}});

  assign busy  = (st != D_IDLE);
  assign bready = (st == D_B);
  assign wstrb  = 8'hFF;
  assign rready = (st == D_R);
  assign arlen  = (chunk_b[31:11] != 21'd0) ? 8'd255 :
                  ((chunk_b[10:3] == 9'd0) ? 8'd0 : chunk_b[10:3] - 9'd1);
  assign awlen  = arlen;
  assign ctx_raddr = base_r + wbeat[19:1];   // 每 2 拍一行 16B

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= D_IDLE; done <= 1'b0; last_rlast <= 1'b0;
      arvalid <= 1'b0; awvalid <= 1'b0; wvalid <= 1'b0; wlast <= 1'b0;
      ctx_we <= 1'b0; ctx_welane <= '0; ctx_addr <= '0; ctx_wdata <= '0;
      wr_we <= '0; wr_addr <= '0; wr_wdata <= '0;
    end else begin
      done <= 1'b0; arvalid <= 1'b0; awvalid <= 1'b0; wvalid <= 1'b0; wlast <= 1'b0;
      ctx_we <= 1'b0; ctx_welane <= '0; wr_we <= '0;
      case (st)
        D_IDLE: if (start) begin
            addr_r <= cmd_addr; remain <= {14'd0, cmd_len};
            base_r <= cmd_base; tag_r <= cmd_tag;
            chunk_b <= (cmd_len > 18'd2048) ? 32'd2048 : {14'd0, cmd_len};
            byi <= '0; wk <= '0; wj <= '0; wbeat <= '0; wbeat_chunk <= '0;
            st <= cmd_is_wr ? D_AW : D_AR;
          end
        // ---------------- LOAD ----------------
        D_AR: begin
            if (!arvalid) arvalid <= 1'b1;
            araddr <= addr_r;
            if (arvalid && arready) st <= D_R;   // 默认清 arvalid
          end
        D_R: if (rvalid && rready) begin
            remain <= remain - 32'd8;
            byi <= byi + 18'd8;
            if (tag_r == 3'd0) begin
              // CTX：8 lane 同地址一次写入（16B 组对齐 -> byi[3] 恒定）
              ctx_we <= 1'b1;
              ctx_addr <= base_r + {2'd0, byi[17:4]};
              for (int q = 0; q < 8; q++) begin
                ctx_welane[byi[3:0] + q[3:0]] <= 1'b1;
                ctx_wdata[(byi[3:0] + q[3:0])*8 +: 8] <= rdata[q*8 +: 8];
              end
            end else if (!xing) begin
              // WRAM：8 lane 同 k 一次写入
              wr_addr <= base_r[11:0] + wk;
              for (int q = 0; q < 8; q++) begin
                wr_we[wj + q[7:0]] <= 1'b1;
                wr_wdata[(wj + q[7:0])*8 +: 8] <= rdata[q*8 +: 8];
              end
              if (wj + 8'd8 == COLS[7:0]) begin wj <= '0; wk <= wk + 12'd1; end
              else wj <= wj + 8'd8;
            end else begin
              // 跨 COLS 边界：本拍写 [wj, COLS)，残拍 [0, ...) 到 k+1
              cross_n <= COLS[3:0] - wj[3:0];
              beat_buf <= rdata;
              wr_addr <= base_r[11:0] + wk;
              for (int jj = 0; jj < 256; jj++) begin
                if (jj[7:0] >= wj && jj < COLS) begin
                  wr_we[jj] <= 1'b1;
                  wr_wdata[jj*8 +: 8] <= rdata[(jj - wj)*8 +: 8];
                end
              end
              wk <= wk + 12'd1;
              st <= D_R2;
            end
            // 突发边界（跨界拍延到 D_R2 处理，避免覆盖 st）
            if (rlast) begin
              if (xing) last_rlast <= 1'b1;
              else if (remain <= 32'd8) st <= D_FIN;
              else begin
                addr_r  <= addr_r + 32'd2048;
                chunk_b <= (remain - 32'd8 > 32'd2048) ? 32'd2048 : (remain - 32'd8);
                st <= D_AR;
              end
            end
          end
        D_R2: begin  // 跨界残拍：lane [0, wj+8-COLS) @ k+1（wk 已 +1）
            wr_addr <= base_r[11:0] + wk[11:0];   // 残拍属下一 k 行
            for (int jj = 0; jj < 8; jj++) begin
              if (jj[7:0] < wj + 8'd8 - COLS[7:0]) begin
                wr_we[jj] <= 1'b1;
                wr_wdata[jj*8 +: 8] <= beat_buf[(cross_n + jj[3:0])*8 +: 8];
              end
            end
            wj <= wj + 8'd8 - COLS[7:0];
            if (last_rlast) begin
              last_rlast <= 1'b0;
              if (remain == 32'd0) st <= D_FIN;
              else begin
                addr_r  <= addr_r + 32'd2048;
                chunk_b <= (remain > 32'd2048) ? 32'd2048 : remain;
                st <= D_AR;
              end
            end else st <= D_R;
          end
        // ---------------- STORE ----------------
        D_AW: begin
            if (!awvalid) awvalid <= 1'b1;
            awaddr <= addr_r;
            if (awvalid && awready) st <= D_RD;  // 默认清 awvalid
          end
        D_RD: st <= D_RD2;              // 地址已驱动，等 RAM 回数
        D_RD2: begin
            rd16 <= ctx_rdata;          // 本行 16B 到齐
            st <= D_W;
          end
        D_W: begin
            if (!wvalid) wvalid <= 1'b1;           // 仅在无未决拍时武装
            wdata <= wbeat[0] ? rd16[127:64] : rd16[63:0];
            wlast <= (wbeat_chunk == chunk_b[31:3] - 32'd1);
            if (wvalid && wready) begin
              wbeat <= wbeat + 32'd1;
              wbeat_chunk <= wbeat_chunk + 32'd1;
              remain <= remain - 32'd8;
              if (wbeat[0] == 1'b1) begin          // 一行 16B 写完
                wvalid <= 1'b0;                    // ★ 撤下（否则 D_RD 拍冒重复拍）
                if (remain <= 32'd8) st <= D_B;
                else if (wbeat_chunk + 32'd1 == chunk_b[31:3]) begin
                  addr_r  <= addr_r + 32'd2048;    // 突发边界
                  chunk_b <= (remain - 32'd8 > 32'd2048) ? 32'd2048 : (remain - 32'd8);
                  wbeat_chunk <= '0;
                  st <= D_AW;
                end else st <= D_RD;               // 下一行
              end
            end
          end
        D_B: if (bvalid) begin
            if (remain == 32'd0) st <= D_FIN;
            else begin
              addr_r <= addr_r + 32'd2048;
              chunk_b <= (remain > 32'd2048) ? 32'd2048 : remain;
              wbeat_chunk <= '0;
              st <= D_AW;
            end
          end
        D_FIN: begin done <= 1'b1; st <= D_IDLE; end
        default: st <= D_IDLE;
      endcase
    end
  end
endmodule
`endif
