// ae_top.sv — 顶层：ae_core + AXI4-Lite 从机（控制/状态/性能/SEQ 装载）
// 寄存器映射（32b，字节地址）：
//   0x00 CTRL   bit0 start（写 1 触发，硬件自清） bit1 hoist_en  bit2 pf_en（权重预取）
//   0x04 STATUS bit0 busy bit1 done
//   0x08 cycles        0x0C gemm_cycles   0x10 dma_cycles   0x14 mac_total
//   0x18 skip_macs     0x1C skip_stages
//   0x20 SEQ_ADDR      序表项索引（0..511）
//   0x40..0x5C SEQ_W0..W7   描述符 256b 的 8 个 32b 字（W7 = [255:224]）
//   0x60 SEQ_COMMIT    写 1 -> 把 W0..W7 写入 SEQ_ADDR 项（硬件自清）
`ifndef AE_TOP_SV
`define AE_TOP_SV
module ae_top #(
  parameter int COLS      = 96,
  parameter int CTX_WORDS = 131072,
  parameter int W_WORDS   = 4096,
  parameter int SEQ_N     = 2048
)(
  input  logic clk,
  input  logic rst_n,
  // AXI4-Lite 从机（PS 控制）
  input  logic [31:0] s_axil_awaddr,
  input  logic        s_axil_awvalid,
  output logic        s_axil_awready,
  input  logic [31:0] s_axil_wdata,
  input  logic        s_axil_wvalid,
  output logic        s_axil_wready,
  output logic [1:0]  s_axil_bresp,
  output logic        s_axil_bvalid,
  input  logic        s_axil_bready,
  input  logic [31:0] s_axil_araddr,
  input  logic        s_axil_arvalid,
  output logic        s_axil_arready,
  output logic [31:0] s_axil_rdata,
  output logic [1:0]  s_axil_rresp,
  output logic        s_axil_rvalid,
  input  logic        s_axil_rready,
  // AXI4 主机（DMA，64b）
  output logic [31:0] m_axi_araddr,
  output logic [7:0]  m_axi_arlen,
  output logic        m_axi_arvalid,
  input  logic        m_axi_arready,
  input  logic [63:0] m_axi_rdata,
  input  logic        m_axi_rvalid,
  input  logic        m_axi_rlast,
  output logic        m_axi_rready,
  output logic [31:0] m_axi_awaddr,
  output logic [7:0]  m_axi_awlen,
  output logic        m_axi_awvalid,
  input  logic        m_axi_awready,
  output logic [63:0] m_axi_wdata,
  output logic [7:0]  m_axi_wstrb,
  output logic        m_axi_wlast,
  output logic        m_axi_wvalid,
  input  logic        m_axi_wready,
  input  logic        m_axi_bvalid,
  output logic        m_axi_bready
);
  // ---------------- 核 ----------------
  logic start_r, hoist_en_r, pf_en_r, core_busy, core_done;
  logic [31:0] cycles, gemm_cycles, dma_cycles, mac_total, skip_macs;
  logic [15:0] skip_stages;
  logic        seq_we;
  logic [15:0] seq_waddr;
  logic [255:0] seq_wdata;

  ae_core #(.COLS(COLS), .CTX_WORDS(CTX_WORDS), .W_WORDS(W_WORDS), .SEQ_N(SEQ_N)) u_core (
    .clk(clk), .rst_n(rst_n),
    .start(start_r), .hoist_en(hoist_en_r), .pf_en(pf_en_r),
    .busy(core_busy), .done(core_done),
    .araddr(m_axi_araddr), .arlen(m_axi_arlen), .arvalid(m_axi_arvalid),
    .arready(m_axi_arready), .rdata(m_axi_rdata), .rvalid(m_axi_rvalid),
    .rlast(m_axi_rlast), .rready(m_axi_rready),
    .awaddr(m_axi_awaddr), .awlen(m_axi_awlen), .awvalid(m_axi_awvalid),
    .awready(m_axi_awready), .wdata(m_axi_wdata), .wstrb(m_axi_wstrb),
    .wlast(m_axi_wlast), .wvalid(m_axi_wvalid), .wready(m_axi_wready),
    .bvalid(m_axi_bvalid), .bready(m_axi_bready),
    .seq_we(seq_we), .seq_waddr(seq_waddr), .seq_wdata(seq_wdata),
    .cycles(cycles), .gemm_cycles(gemm_cycles), .dma_cycles(dma_cycles),
    .mac_total(mac_total), .skip_macs(skip_macs), .skip_stages(skip_stages)
  );

  // ---------------- AXI4-Lite 从机 ----------------
  logic [15:0] seq_addr_r;
  logic [31:0] seqw [0:7];

  typedef enum logic [1:0] {XL_IDLE, XL_B, XL_R} xl_e;
  xl_e wst, rst_;

  // 写通道
  assign s_axil_awready = (wst == XL_IDLE);
  assign s_axil_wready  = (wst == XL_IDLE);
  assign s_axil_bresp   = 2'b00;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wst <= XL_IDLE; s_axil_bvalid <= 1'b0;
      start_r <= 1'b0; hoist_en_r <= 1'b0; pf_en_r <= 1'b0; seq_we <= 1'b0;
      seq_addr_r <= '0; seq_waddr <= '0; seq_wdata <= '0;
      for (int i = 0; i < 8; i++) seqw[i] <= '0;
    end else begin
      start_r <= 1'b0; seq_we <= 1'b0;      // 单拍脉冲
      case (wst)
        XL_IDLE: if (s_axil_awvalid && s_axil_wvalid) begin
            // 寄存器写
            if (s_axil_awaddr[7:0] == 8'h00) begin
              if (s_axil_wdata[0]) start_r <= 1'b1;
              hoist_en_r <= s_axil_wdata[1];
              pf_en_r    <= s_axil_wdata[2];            // ★ 权重预取使能（复位 0）
            end else if (s_axil_awaddr[7:0] == 8'h20) begin
              seq_addr_r <= s_axil_wdata[15:0];
            end else if (s_axil_awaddr[7:0] >= 8'h40 && s_axil_awaddr[7:0] <= 8'h5C) begin
              seqw[(s_axil_awaddr[5:2] - 4'd0)] <= s_axil_wdata;  // 0x40>>2=16..0x5C>>2=23
            end else if (s_axil_awaddr[7:0] == 8'h60) begin
              seq_waddr <= seq_addr_r;
              for (int i = 0; i < 8; i++) seq_wdata[i*32 +: 32] <= seqw[i];
              seq_we <= 1'b1;
            end
            wst <= XL_B; s_axil_bvalid <= 1'b1;
          end
        XL_B: if (s_axil_bready) begin s_axil_bvalid <= 1'b0; wst <= XL_IDLE; end
        default: wst <= XL_IDLE;
      endcase
    end
  end

  // 读通道
  assign s_axil_arready = (rst_ == XL_IDLE);
  assign s_axil_rresp   = 2'b00;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rst_ <= XL_IDLE; s_axil_rvalid <= 1'b0; s_axil_rdata <= '0;
    end else begin
      case (rst_)
        XL_IDLE: if (s_axil_arvalid) begin
            s_axil_rdata <= '0;
            case (s_axil_araddr[7:0])
              8'h00: s_axil_rdata <= {29'd0, pf_en_r, hoist_en_r, start_r};
              8'h04: s_axil_rdata <= {30'd0, core_done, core_busy};
              8'h08: s_axil_rdata <= cycles;
              8'h0C: s_axil_rdata <= gemm_cycles;
              8'h10: s_axil_rdata <= dma_cycles;
              8'h14: s_axil_rdata <= mac_total;
              8'h18: s_axil_rdata <= skip_macs;
              8'h1C: s_axil_rdata <= {16'd0, skip_stages};
              default: ;
            endcase
            rst_ <= XL_R; s_axil_rvalid <= 1'b1;
          end
        XL_R: if (s_axil_rready) begin s_axil_rvalid <= 1'b0; rst_ <= XL_IDLE; end
        default: rst_ <= XL_IDLE;
      endcase
    end
  end
endmodule
`endif
