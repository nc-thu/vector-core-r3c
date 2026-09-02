// ae_core.sv — 加速器核心集成：CTX 16-bank（URAM）/ WRAM COLS-bank（BRAM）/ SEQ RAM
//   + ae_gemm + ae_softmax + ae_copy + ae_dma + ae_sched（★ 原语调度器）
// 引擎串行执行（调度器 one-hot 仲裁 CTX B 口）；CTX A 口仅 GEMM 激活读，WRAM A 口仅 GEMM。
// 综合 = ZCU104 满配（COLS=108）；仿真由 tb 覆盖小参数。
`ifndef AE_CORE_SV
`define AE_CORE_SV
module ae_core #(
  parameter int COLS      = 108,
  parameter int CTX_WORDS = 131072,
  parameter int W_WORDS   = 4096,
  parameter int SEQ_N     = 2048
)(
  input  logic clk,
  input  logic rst_n,
  input  logic start,
  input  logic hoist_en,
  output logic busy,
  output logic done,
  // AXI4 主口（DMA）
  output logic [31:0] araddr,
  output logic [7:0]  arlen,
  output logic        arvalid,
  input  logic        arready,
  input  logic [63:0] rdata,
  input  logic        rvalid,
  input  logic        rlast,
  output logic        rready,
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
  // SEQ RAM 运行时装载（PS 通道；仿真用 $readmemh 预载）
  input  logic        seq_we,
  input  logic [15:0] seq_waddr,
  input  logic [255:0] seq_wdata,
  // 性能计数
  output logic [31:0] cycles, gemm_cycles, dma_cycles, mac_total, skip_macs,
  output logic [15:0] skip_stages
);
  localparam int CTX_AW = $clog2(CTX_WORDS);
  localparam int W_AW   = $clog2(W_WORDS);
  localparam int SEQ_AW = $clog2(SEQ_N);

  // ---------------- 内部信号（先声明后使用） ----------------
  logic g_start, g_busy, g_done, sm_start, sm_busy, sm_done;
  logic cp_start, cp_busy, cp_done, dma_start, dma_busy, dma_done;
  logic eng_g, eng_sm, eng_cp, eng_dma;
  logic [255:0] desc;
  logic [31:0] g_mac_cnt;

  logic [19:0] g_ctxa_addr;
  logic [16*8-1:0] g_ctxa_rdata;
  logic g_ctxb_we; logic [15:0] g_ctxb_welane; logic [19:0] g_ctxb_addr;
  logic [16*8-1:0] g_ctxb_wdata;
  logic [11:0] g_w_addr; logic [COLS*8-1:0] g_w_rdata;

  logic [19:0] sm_raddr;  logic [16*8-1:0] sm_rdata;
  // SM16：softmax 每拍写一整列（16 lane 同地址全宽写）
  logic sm_we; logic [19:0] sm_waddr; logic [127:0] sm_wdata;

  logic [19:0] cp_raddr; logic [16*8-1:0] cp_rdata;
  logic [COLS-1:0] cp_wr_we; logic [11:0] cp_wr_addr; logic [COLS*8-1:0] cp_wr_wdata;

  logic [31:0] dma_addr; logic [17:0] dma_len; logic dma_iswr; logic [2:0] dma_tag;
  logic [19:0] dma_base;
  logic dma_ctx_we; logic [15:0] dma_ctx_welane; logic [19:0] dma_ctx_addr;
  logic [16*8-1:0] dma_ctx_wdata; logic [19:0] dma_ctx_raddr;
  logic [16*8-1:0] dma_ctx_rdata;
  logic [COLS-1:0] dma_wr_we; logic [11:0] dma_wr_addr; logic [COLS*8-1:0] dma_wr_wdata;

  logic [SEQ_AW-1:0] seq_raddr;
  logic [255:0] seq_rdata;

  // CTX bank 阵互连（SDP：A 只读 / B 只写）
  logic [16*8-1:0] ctxa_rdata_bus;
  logic [CTX_AW-1:0] ctxa_addr_bank;
  logic [15:0] ctxb_welane_mux;
  logic [CTX_AW-1:0] ctxb_waddr_mux;
  logic [16*8-1:0] ctxb_wdata_mux;

  // WRAM bank 阵互连
  logic [COLS-1:0] wrb_we_mux;
  logic [11:0] wrb_addr_mux;
  logic [COLS*8-1:0] wrb_wdata_mux;

  // ---------------- 引擎 ----------------
  // 描述符字段（与 ae_sched 同切片）
  wire [3:0]  d_op     = desc[255:252];
  wire [2:0]  d_bsrc   = desc[248:246];
  wire        d_causal = desc[245];
  wire        d_ytr    = desc[244];
  wire [15:0] d_m      = desc[243:228];
  wire [15:0] d_n      = desc[227:212];
  wire [15:0] d_k      = desc[211:196];
  wire [19:0] d_abase  = desc[195:176];
  wire [19:0] d_bbase  = desc[175:156];
  wire [19:0] d_ybase  = desc[155:136];
  wire [15:0] d_spad   = desc[135:120];
  wire [15:0] d_rqm    = desc[119:104];
  wire [7:0]  d_rqs    = desc[103:96];
  wire [15:0] d_j0     = desc[77:62];   // GEMM：组全局列偏移（复用 dma_len 字段区间）

  ae_gemm #(.COLS(COLS)) u_gemm (
    .clk(clk), .rst_n(rst_n),
    .start(g_start), .busy(g_busy), .done(g_done),
    .m(d_m), .n(d_n), .n_loc(d_spad), .j0(d_j0), .k(d_k),
    .a_base(d_abase), .b_base(d_bbase), .y_base(d_ybase),
    .y_tr(d_ytr), .rq_m(d_rqm), .rq_s(d_rqs),
    .ctxa_addr(g_ctxa_addr), .ctxa_rdata(g_ctxa_rdata),
    .ctxb_we(g_ctxb_we), .ctxb_welane(g_ctxb_welane),
    .ctxb_addr(g_ctxb_addr), .ctxb_wdata(g_ctxb_wdata),
    .w_addr(g_w_addr), .w_rdata(g_w_rdata),
    .mac_cnt(g_mac_cnt)
  );

  ae_softmax u_sm (
    .clk(clk), .rst_n(rst_n),
    .start(sm_start),
    .s_base(d_ybase), .m_rows(d_m), .n_cols(d_n), .causal(d_causal),
    .ctx_raddr(sm_raddr), .ctx_rdata(sm_rdata),
    .ctx_we(sm_we), .ctx_waddr(sm_waddr), .ctx_wdata(sm_wdata),
    .busy(sm_busy), .done(sm_done)
  );

  ae_copy #(.COLS(COLS)) u_cp (
    .clk(clk), .rst_n(rst_n),
    .start(cp_start),
    .k_rows(d_k), .j_cols(d_n[7:0]), .src_base(d_bbase), .spad(d_spad),
    .src_j0(d_rqm), .wr_base(d_abase[11:0]),
    .ctx_raddr(cp_raddr), .ctx_rdata(cp_rdata),
    .wr_we(cp_wr_we), .wr_addr(cp_wr_addr), .wr_wdata(cp_wr_wdata),
    .busy(cp_busy), .done(cp_done)
  );

  ae_dma #(.COLS(COLS)) u_dma (
    .clk(clk), .rst_n(rst_n),
    .start(dma_start), .busy(dma_busy), .done(dma_done),
    .cmd_addr(dma_addr), .cmd_len(dma_len), .cmd_is_wr(dma_iswr),
    .cmd_tag(dma_tag), .cmd_base(dma_base),
    .araddr(araddr), .arlen(arlen), .arvalid(arvalid), .arready(arready),
    .rdata(rdata), .rvalid(rvalid), .rlast(rlast), .rready(rready),
    .awaddr(awaddr), .awlen(awlen), .awvalid(awvalid), .awready(awready),
    .wdata(wdata), .wstrb(wstrb), .wlast(wlast), .wvalid(wvalid), .wready(wready),
    .bvalid(bvalid), .bready(bready),
    .ctx_we(dma_ctx_we), .ctx_welane(dma_ctx_welane), .ctx_addr(dma_ctx_addr),
    .ctx_wdata(dma_ctx_wdata), .ctx_raddr(dma_ctx_raddr), .ctx_rdata(dma_ctx_rdata),
    .wr_we(dma_wr_we), .wr_addr(dma_wr_addr), .wr_wdata(dma_wr_wdata)
  );

  ae_sched #(.SEQ_AW(SEQ_AW)) u_sched (
    .clk(clk), .rst_n(rst_n),
    .start(start), .hoist_en(hoist_en), .busy(busy), .done(done),
    .seq_raddr(seq_raddr), .seq_rdata(seq_rdata),
    .g_start(g_start), .g_busy(g_busy), .g_done(g_done),
    .sm_start(sm_start), .sm_busy(sm_busy), .sm_done(sm_done),
    .cp_start(cp_start), .cp_busy(cp_busy), .cp_done(cp_done),
    .dma_start(dma_start), .dma_busy(dma_busy), .dma_done(dma_done),
    .eng_g(eng_g), .eng_sm(eng_sm), .eng_cp(eng_cp), .eng_dma(eng_dma),
    .desc_o(desc),
    .g_mac_cnt(g_mac_cnt),
    .cycles(cycles), .gemm_cycles(gemm_cycles), .dma_cycles(dma_cycles),
    .mac_total(mac_total), .skip_macs(skip_macs), .skip_stages(skip_stages)
  );

  // DMA 命令参数（从描述符）
  assign dma_addr  = desc[60:29];
  assign dma_len   = desc[78:61];
  assign dma_iswr  = (d_op == 4'd5);
  assign dma_tag   = d_bsrc;
  assign dma_base  = dma_iswr ? d_ybase : d_bbase;

  // ---------------- CTX 主存（SDP，URAM） ----------------
  // A 口（读广播）分时复用：调度器串行执行，GEMM/softmax/COPY/DMA-STORE 互斥；
  // B 口专职写（GEMM 写回 / softmax P / DMA 装载）。softmax P3 在 B 写 P[j] 的
  // 同拍用 A 预读 S[j+1] —— 不同地址，SDP（1W+1R）合法。
  always_comb begin
    ctxa_addr_bank = g_ctxa_addr[CTX_AW-1:0];
    if (eng_sm)       ctxa_addr_bank = sm_raddr[CTX_AW-1:0];
    else if (eng_cp)  ctxa_addr_bank = cp_raddr[CTX_AW-1:0];
    else if (eng_dma && dma_iswr) ctxa_addr_bank = dma_ctx_raddr[CTX_AW-1:0];
  end
  // B 口写仲裁（引擎 one-hot）
  always_comb begin
    ctxb_welane_mux = '0;
    ctxb_waddr_mux  = '0;
    ctxb_wdata_mux  = '0;
    if (eng_g) begin
      ctxb_waddr_mux  = g_ctxb_addr[CTX_AW-1:0];
      ctxb_welane_mux = g_ctxb_we ? g_ctxb_welane : '0;
      ctxb_wdata_mux  = g_ctxb_wdata;
    end else if (eng_sm) begin
      ctxb_waddr_mux  = sm_waddr[CTX_AW-1:0];
      ctxb_welane_mux = sm_we ? 16'hFFFF : '0;   // SM16 列写：16 lane 全使能
      ctxb_wdata_mux  = sm_wdata;
    end else if (eng_dma && !dma_iswr) begin
      ctxb_waddr_mux  = dma_ctx_addr[CTX_AW-1:0];
      ctxb_welane_mux = dma_ctx_we ? dma_ctx_welane : '0;
      ctxb_wdata_mux  = dma_ctx_wdata;
    end
  end

  ae_ctx_ram #(.WORDS(CTX_WORDS), .RAM_STYLE("ultra")) u_ctx (
    .clk(clk),
    .raddr(ctxa_addr_bank), .rdata(ctxa_rdata_bus),
    .we_byte(ctxb_welane_mux), .waddr(ctxb_waddr_mux), .wdata(ctxb_wdata_mux)
  );

  assign g_ctxa_rdata = ctxa_rdata_bus;
  assign sm_rdata     = ctxa_rdata_bus;   // softmax 读走 A 口
  assign cp_rdata     = ctxa_rdata_bus;   // COPY 读走 A 口（串行，GEMM 空闲）
  assign dma_ctx_rdata = ctxa_rdata_bus;  // DMA STORE 读走 A 口

  // ---------------- WRAM bank 阵（COLS lane，BRAM） ----------------
  always_comb begin
    wrb_we_mux = '0; wrb_addr_mux = '0; wrb_wdata_mux = '0;
    if (eng_cp) begin
      wrb_we_mux = cp_wr_we; wrb_addr_mux = cp_wr_addr; wrb_wdata_mux = cp_wr_wdata;
    end else if (eng_dma) begin
      wrb_we_mux = dma_wr_we; wrb_addr_mux = dma_wr_addr; wrb_wdata_mux = dma_wr_wdata;
    end
  end
  generate
  for (genvar b = 0; b < COLS; b++) begin : g_w
    ae_dpram #(.WIDTH(8), .WORDS(W_WORDS), .RAM_STYLE("block")) u_bank (
      .clk(clk),
      .a_we(1'b0), .a_addr(g_w_addr[W_AW-1:0]), .a_wdata(8'h00), .a_rdata(g_w_rdata[b*8 +: 8]),
      .b_we(wrb_we_mux[b]),
      .b_addr(wrb_addr_mux[W_AW-1:0]),
      .b_wdata(wrb_wdata_mux[b*8 +: 8]),
      .b_rdata()
    );
  end
  endgenerate

  // ---------------- SEQ RAM（512 x 256，可运行时装载 + 仿真 $readmemh） ----------------
  (* ram_style = "block" *) logic [255:0] seq_mem [0:SEQ_N-1];
  initial $readmemh("seq.mem", seq_mem);
  always_ff @(posedge clk) begin
    if (seq_we) seq_mem[seq_waddr[SEQ_AW-1:0]] <= seq_wdata;
    seq_rdata <= seq_mem[seq_raddr];
  end

endmodule
`endif
