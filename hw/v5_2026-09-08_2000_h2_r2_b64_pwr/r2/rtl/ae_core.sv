// ae_core.sv — 加速器核心集成 ★ v5-R2 双读引擎版
// ---------------------------------------------------------------------------
// 相对底本 hb_fpga_impl/22_r3c_rtl/rtl/ae_core.sv 的 diff（2026-09-08 20:15）：
//   1. 顶层新增第二组 AXI4 读主口（araddr2/arlen2/arvalid2/arready2/rdata2/
//      rvalid2/rlast2/rready2），接 ae_dma 的 w 读引擎；原读口归 ctx 读引擎。
//   2. CTX B 口写仲裁：底本的「eng_dma 前台臂 + bg_wran 后台臂」两臂合并成
//      一臂 `rd_c_busy`（ctx 读引擎在跑就在写 B 口——引擎单命令，前台/后台
//      不会同时）。优先级 eng_g > eng_sm > eng_a > rd_c_busy 不变。
//   3. pf_ctx_stall（CTX 预取写让拍反馈）：改为 `rd_c_busy && rd_c_bg &&
//      (eng_g || eng_sm || eng_a)`。只对后台命令让拍；前台装载因调度器
//      消费点等待（rd_idle）不可能与串行引擎重叠，TB 加断言守卫。
//   4. WRAM B 口写仲裁：`eng_dma || bg_wran` 臂改成 `rd_w_busy`（w 引擎
//      单命令，前台/后台同臂）。eng_cp 优先级不变。
//   5. bg_wran 触发器删除（职责移入 ae_dma 的 rd_c_bg/引擎 busy）。
//   其余（CTX A 口分时复用、SEQ RAM、引擎例化、描述符切片）逐字保留底本。
// ---------------------------------------------------------------------------
`ifndef AE_CORE_SV
`define AE_CORE_SV
module ae_core #(
  parameter int COLS      = 96,
  parameter int CTX_WORDS = 131072,
  parameter int W_WORDS   = 4096,
  parameter int SEQ_N     = 2048
)(
  input  logic clk,
  input  logic rst_n,
  input  logic start,
  input  logic hoist_en,
  input  logic pf_en,               // ★ 权重预取使能（复位 0 = 零漂移档）
  output logic busy,
  output logic done,
  // AXI4 主口 1（DMA：ctx 读引擎 + 写引擎）
  output logic [31:0] araddr,
  output logic [7:0]  arlen,
  output logic        arvalid,
  input  logic        arready,
  input  logic [63:0] rdata,
  input  logic        rvalid,
  input  logic        rlast,
  output logic        rready,
  // AXI4 主口 2（DMA：w 读引擎）★ v5-R2 新增
  output logic [31:0] araddr2,
  output logic [7:0]  arlen2,
  output logic        arvalid2,
  input  logic        arready2,
  input  logic [63:0] rdata2,
  input  logic        rvalid2,
  input  logic        rlast2,
  output logic        rready2,
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
  logic a_start, a_busy, a_done;                       // ★ AE_ACTV
  logic eng_g, eng_sm, eng_cp, eng_dma, eng_a;
  logic [255:0] desc;
  logic [31:0] g_mac_cnt;
  logic        g_wb_active;   // R2：GEMM 写回阶段标志

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

  // ★ AE_ACTV 行引擎（CTX A 口广播读 / B 口 16-lane 掩码写；不碰 WRAM）
  logic [19:0] actv_raddr; logic [127:0] actv_rdata;
  logic actv_we; logic [15:0] actv_welane; logic [19:0] actv_waddr;
  logic [127:0] actv_wdata;

  logic [31:0] dma_addr; logic [17:0] dma_len; logic dma_iswr; logic [2:0] dma_tag;
  logic [19:0] dma_base;
  logic dma_ctx_we; logic [15:0] dma_ctx_welane; logic [19:0] dma_ctx_addr;
  logic [16*8-1:0] dma_ctx_wdata; logic [19:0] dma_ctx_raddr;
  logic [16*8-1:0] dma_ctx_rdata;
  logic [COLS-1:0] dma_wr_we; logic [11:0] dma_wr_addr; logic [COLS*8-1:0] dma_wr_wdata;
  // ★ v5-R2：读引擎按流拆 + 写引擎独立
  logic rd_c_busy, rd_c_done, rd_w_busy, rd_w_done, wr_busy, wr_done, wr_start_o;
  logic ctxa_wr_bank;   // CTX A 口当前归 STORE 写引擎（GEMM 不在跑且 wr_busy）
  // R2：GEMM 写回时 CTX B 口被 GEMM 占用，DMA CTX 预取写必须让拍
  wire gemm_wb_active = g_wb_active;
  // ★ v5-R2：ctx 读引擎状态（tag + 是否后台命令）
  logic [2:0] rd_tag_o;
  logic rd_c_bg;
  logic pf_ctx_stall;

  // ★ 权重预取（后台 TAG_W / TAG_CTX LOAD）
  logic        pf_bg_start;
  logic [31:0] pf_dmaaddr;
  logic [17:0] pf_dmalen;
  logic [2:0]  pf_dmatag;
  logic [19:0] pf_dmabase;

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
  wire [15:0] d_rqm2   = desc[135:120];        // ELTWISE 第二乘子（m2，submode=3）
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
    .mac_cnt(g_mac_cnt),
    .wb_active(g_wb_active)
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

  // ★ AE_ACTV：submode 走 b_src 字段，张量基址走 y_base（与 softmax 同位），
  //   表映像基址走 b_base，表长走 k，BIAS 常数走 rq_m/rq_s。
  ae_actv u_actv (
    .clk(clk), .rst_n(rst_n),
    .start(a_start), .busy(a_busy), .done(a_done),
    .submode(d_bsrc),
    .y_base(d_ybase), .m_rows(d_m), .n_cols(d_n),
    .tbl_base(d_bbase), .tbl_len(d_k),
    .rq_m(d_rqm), .rq_s(d_rqs), .rq_m2(d_rqm2),
    .ctx_raddr(actv_raddr), .ctx_rdata(actv_rdata),
    .ctx_we(actv_we), .ctx_welane(actv_welane),
    .ctx_waddr(actv_waddr), .ctx_wdata(actv_wdata)
  );

  ae_dma #(.COLS(COLS)) u_dma (
    .clk(clk), .rst_n(rst_n),
    // 前台命令 start = dma_start（pf_bg_start 也 OR 进来触发 start 脉冲，
    //   但命令参数走 bg_* 口——见 ae_dma 内 fg_rd 的 bg 让位）
    .start(dma_start | pf_bg_start), .busy(dma_busy), .done(dma_done),
    .cmd_addr(dma_addr), .cmd_len(dma_len), .cmd_is_wr(dma_iswr),
    .cmd_tag(dma_tag), .cmd_base(dma_base),
    // 后台预取命令（pf_bg_start 脉冲时影子组已稳定，按 bg_tag 路由引擎）
    .bg_start(pf_bg_start),
    .bg_addr(pf_dmaaddr), .bg_len(pf_dmalen),
    .bg_tag(pf_dmatag), .bg_base(pf_dmabase),
    // 各引擎独立状态（v5-R2：rd 按流拆）
    .rd_busy(), .rd_done(),
    .rd_c_busy(rd_c_busy), .rd_c_done(rd_c_done),
    .rd_w_busy(rd_w_busy), .rd_w_done(rd_w_done),
    .wr_busy(wr_busy), .wr_done(wr_done), .wr_start_o(wr_start_o),
    .ctxa_wr_bank(ctxa_wr_bank),
    .pf_ctx_stall(pf_ctx_stall), .rd_tag_o(rd_tag_o), .rd_c_bg(rd_c_bg),
    // 读口 1 = ctx 引擎；读口 2 = w 引擎
    .araddr(araddr), .arlen(arlen), .arvalid(arvalid), .arready(arready),
    .rdata(rdata), .rvalid(rvalid), .rlast(rlast), .rready(rready),
    .araddr2(araddr2), .arlen2(arlen2), .arvalid2(arvalid2), .arready2(arready2),
    .rdata2(rdata2), .rvalid2(rvalid2), .rlast2(rlast2), .rready2(rready2),
    .awaddr(awaddr), .awlen(awlen), .awvalid(awvalid), .awready(awready),
    .wdata(wdata), .wstrb(wstrb), .wlast(wlast), .wvalid(wvalid), .wready(wready),
    .bvalid(bvalid), .bready(bready),
    .ctx_we(dma_ctx_we), .ctx_welane(dma_ctx_welane), .ctx_addr(dma_ctx_addr),
    .ctx_wdata(dma_ctx_wdata), .ctx_raddr(dma_ctx_raddr), .ctx_rdata(dma_ctx_rdata),
    .wr_we(dma_wr_we), .wr_addr(dma_wr_addr), .wr_wdata(dma_wr_wdata)
  );

  ae_sched #(.SEQ_AW(SEQ_AW), .W_AW(W_AW)) u_sched (
    .clk(clk), .rst_n(rst_n),
    .start(start), .hoist_en(hoist_en), .pf_en(pf_en), .busy(busy), .done(done),
    .seq_raddr(seq_raddr), .seq_rdata(seq_rdata),
    .g_start(g_start), .g_busy(g_busy), .g_done(g_done),
    .sm_start(sm_start), .sm_busy(sm_busy), .sm_done(sm_done),
    .cp_start(cp_start), .cp_busy(cp_busy), .cp_done(cp_done),
    .dma_start(dma_start), .dma_busy(dma_busy), .dma_done(dma_done),
    .rd_c_busy(rd_c_busy), .rd_c_done(rd_c_done),
    .rd_w_busy(rd_w_busy), .rd_w_done(rd_w_done),
    .wr_busy(wr_busy), .wr_done(wr_done),
    .a_start(a_start), .a_busy(a_busy), .a_done(a_done),
    .eng_g(eng_g), .eng_sm(eng_sm), .eng_cp(eng_cp), .eng_dma(eng_dma),
    .eng_a(eng_a),
    .desc_o(desc),
    .g_mac_cnt(g_mac_cnt),
    .cycles(cycles), .gemm_cycles(gemm_cycles), .dma_cycles(dma_cycles),
    .mac_total(mac_total), .skip_macs(skip_macs), .skip_stages(skip_stages),
    .pf_bg_start(pf_bg_start), .pf_dmaaddr(pf_dmaaddr), .pf_dmalen(pf_dmalen),
    .pf_dmatag(pf_dmatag), .pf_dmabase(pf_dmabase)
  );

  // DMA 命令源 2:1 mux：前台命令来自描述符，后台预取命令来自 pf_bg_start。
  assign dma_addr  = desc[60:29];
  assign dma_len   = desc[78:61];
  assign dma_iswr  = (d_op == 4'd5);
  assign dma_tag   = d_bsrc;
  assign dma_base  = dma_iswr ? d_ybase : d_bbase;

  // ---------------- CTX 主存（SDP，URAM） ----------------
  // A 口（读广播）分时复用：与底本一致（GEMM 默认，softmax/ACTV/COPY 串行，
  //   STORE 写引擎在 GEMM 不跑时拿 A 口；eng_dma&&dma_iswr 臂保留为死代码，
  //   fire-and-forget 下 STORE 不再经过 T_RUN_DMA）
  always_comb begin
    ctxa_addr_bank = g_ctxa_addr[CTX_AW-1:0];       // GEMM 默认
    if (eng_sm)       ctxa_addr_bank = sm_raddr[CTX_AW-1:0];
    else if (eng_a)   ctxa_addr_bank = actv_raddr[CTX_AW-1:0];
    else if (eng_cp)  ctxa_addr_bank = cp_raddr[CTX_AW-1:0];
    else if (eng_dma && dma_iswr) ctxa_addr_bank = dma_ctx_raddr[CTX_AW-1:0];
    // R1：STORE 并发路径——GEMM 不在跑且写引擎忙时，STORE 拿 A 口
    else if (wr_busy && !eng_g) ctxa_addr_bank = dma_ctx_raddr[CTX_AW-1:0];
  end
  // R1：CTX A 口归 STORE 写引擎的标志（供 DMA 写引擎停拍用）
  assign ctxa_wr_bank = wr_busy && !eng_g && !(eng_sm || eng_a || eng_cp ||
                                               (eng_dma && dma_iswr));
  // B 口写仲裁（★ v5-R2：DMA 侧单写者 = ctx 读引擎，前台/后台同臂 rd_c_busy）
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
    end else if (eng_a) begin                     // ★ ACTV：尾组行掩码列写
      ctxb_waddr_mux  = actv_waddr[CTX_AW-1:0];
      ctxb_welane_mux = actv_we ? actv_welane : '0;
      ctxb_wdata_mux  = actv_wdata;
    end else if (rd_c_busy) begin
      // ★ v5-R2：ctx 读引擎写 B 口（前台装载 fire-and-forget 或后台 CTX 预取；
      //   引擎单命令，两者互斥）。与串行引擎的重叠只可能来自后台预取——
      //   那种情况由 pf_ctx_stall 冻结引擎（本臂选不上时引擎不推进）；
      //   前台装载因调度器消费点等待不可能与 eng_g/sm/a 重叠（TB 断言守卫）
      ctxb_waddr_mux  = dma_ctx_addr[CTX_AW-1:0];
      ctxb_welane_mux = dma_ctx_we ? dma_ctx_welane : '0;
      ctxb_wdata_mux  = dma_ctx_wdata;
    end
  end

  // ★ R2 修复语义保留（v5-R2 改写）：CTX B 口被 eng_g/eng_sm/eng_a 占时，
  //   在飞的后台 CTX 预取写必须停拍——拉 pf_ctx_stall 冻结 ctx 读引擎
  //   （rready=0 → r_byi/r_remain 冻结，写一个不丢）。
  //   只作用于后台命令：前台装载不会与串行引擎重叠（调度器 rd_idle 消费点）。
  //   eng_g 已含 gemm_wb_active（写回期 eng_g 必为 1）。
  assign pf_ctx_stall = rd_c_busy && rd_c_bg && (eng_g || eng_sm || eng_a);

  ae_ctx_ram #(.WORDS(CTX_WORDS), .RAM_STYLE("ultra")) u_ctx (
    .clk(clk),
    .raddr(ctxa_addr_bank), .rdata(ctxa_rdata_bus),
    .we_byte(ctxb_welane_mux), .waddr(ctxb_waddr_mux), .wdata(ctxb_wdata_mux)
  );

  assign g_ctxa_rdata = ctxa_rdata_bus;
  assign sm_rdata     = ctxa_rdata_bus;   // softmax 读走 A 口
  assign cp_rdata     = ctxa_rdata_bus;   // COPY 读走 A 口（串行，GEMM 空闲）
  assign dma_ctx_rdata = ctxa_rdata_bus;  // DMA STORE 读走 A 口
  assign actv_rdata   = ctxa_rdata_bus;   // ACTV 读走 A 口（串行 one-hot）

  // ---------------- WRAM bank 阵（COLS lane，BRAM） ----------------
  // B 口写仲裁（★ v5-R2：eng_cp 最高；DMA 臂 = w 读引擎在跑（前台或后台，
  //   单命令互斥），数据通路零改动）
  always_comb begin
    wrb_we_mux = '0; wrb_addr_mux = '0; wrb_wdata_mux = '0;
    if (eng_cp) begin
      wrb_we_mux = cp_wr_we; wrb_addr_mux = cp_wr_addr; wrb_wdata_mux = cp_wr_wdata;
    end else if (rd_w_busy) begin
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
