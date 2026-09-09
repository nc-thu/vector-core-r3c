// ae_gemm.sv — GEMM 引擎：Y = requant(A * B)，B 一律来自 WRAM（COLS lane-bank）
// 一次调用 = 一个列组（ng 循环在序列层展开，每组一条描述符 + 一次 B 重驻留）：
//   A: CTX k-major（lane = m mod 16，地址 = a_base + (m div16)*K + k），CTX A 口直读
//   B: WRAM（lane = 组内局部列 j，地址 = b_base + k）—— 1 拍/k（发读-回数-喂阵流水）
//   Y: 写回 CTX B 口，全局列 c = j0 + 局部列：
//      普通 addr = y_base + mt*N + c（lane = m mod 16；N = 全局 n）
//      转置 addr = y_base + (c div16)*M16 + m（lane = c mod 16，V 给 PV；
//               一个 108 宽组跨 8 个全局 16-lane 组，按全局组迭代）
// 循环序 mt（B 一次驻留全 mt 复用）；尾 tile 用 lane-we 屏蔽；尾列用 n_loc 屏蔽。
// S = Q·Kᵀ 用「A=Q 行、B=WRAM 里的 Kᵀ」表达（K 先经 OP_COPY 转置重排进 WRAM）。
`ifndef AE_GEMM_SV
`define AE_GEMM_SV
module ae_gemm #(
  parameter int COLS = 108
)(
  input  logic clk,
  input  logic rst_n,
  input  logic start,
  output logic busy,
  output logic done,
  input  logic [15:0] m,        // token 数（行）
  input  logic [15:0] n,        // 全局输出宽（写回行步长）
  input  logic [15:0] n_loc,    // 本组列数（≤ COLS；尾组 < COLS）
  input  logic [15:0] j0,       // 本组全局列偏移
  input  logic [15:0] k,        // 归约维
  input  logic [19:0] a_base, b_base, y_base,
  input  logic        y_tr,
  input  logic signed [15:0] rq_m,
  input  logic [7:0]  rq_s,
  // CTX A 口（只读广播）
  output logic [19:0] ctxa_addr,
  input  logic [16*8-1:0] ctxa_rdata,
  // CTX B 口（写回：广播地址 + 每 lane we）
  output logic        ctxb_we,
  output logic [15:0] ctxb_welane,
  output logic [19:0] ctxb_addr,
  output logic [16*8-1:0] ctxb_wdata,
  // WRAM A 口（只读广播）
  output logic [11:0] w_addr,
  input  logic [COLS*8-1:0] w_rdata,
  output logic [31:0] mac_cnt
);
  typedef enum logic [3:0] {S_IDLE, S_INIT, S_CLR, S_FEED, S_WAITD, S_DRAIN,
                            S_DRLAT, S_WB, S_WBTR, S_NEXTMT, S_FIN} st_e;
  st_e st;

  logic [15:0] mt, kk;
  logic [15:0] mt_cnt, m16;
  logic [15:0] cgr_lo;
  logic [3:0]  wb_g;       // 转置写回的全局 16-lane 组号偏移
  logic [3:0]  tr_grps;    // 本组跨的全局 16-lane 组数（≤ 9）

  logic issue_d;
  logic [127:0] a_feed_c;
  logic [COLS*8-1:0] b_feed_c;

  logic        arr_clr, arr_busy, arr_done;
  logic [3:0]  drain_row;
  logic [COLS*32-1:0] acc_row;
  logic        rq_v;
  logic [COLS-1:0] rq_vld;
  logic [COLS*8-1:0] rq_y;
  logic [7:0]  tile_buf [0:15][0:COLS-1];
  logic signed [15:0] rq_m_r;
  logic [7:0]  rq_s_r;
  logic [3:0]  drb;
  logic [7:0]  wb_i8, wb_row8;
  logic [31:0] mac_cnt_r;

  logic        wb_we_r;
  logic [15:0] wb_lanes_r;
  (* use_dsp = "no" *) logic [19:0] wb_addr_r;  // mt*n / cgr*m16 走 LUT
  logic [16*8-1:0] wb_data_r;
  // mt*k 在一次喂入内不变：寄存基址，CTX A 口地址路径只剩加法器（同 copy 的 rbase_r）
  (* use_dsp = "no" *) logic [19:0] abase_r;
  // 乘积专用线网（防 retiming 衍生寄存器丢 use_dsp 属性）
  (* use_dsp = "no" *) logic [31:0] mtp_k, mtn, cgrm16;
  assign mtp_k  = (mt + 16'd1) * k;              // S_NEXTMT 的下一行基址
  assign mtn    = mt * n;                        // S_WB 写回行基址
  assign cgrm16 = ({12'd0, wb_g} + cgr_lo) * m16; // S_WBTR 转置组基址

  ae_sysarr #(.ROWS(16), .COLS(COLS)) u_arr (
    .clk(clk), .rst_n(rst_n),
    .clr(arr_clr), .feed_vld(issue_d),
    .a_feed(a_feed_c), .b_feed(b_feed_c),
    .busy(arr_busy), .done(arr_done),
    .drain_row(drain_row), .acc_row(acc_row)
  );

  genvar gq;
  generate
    for (gq = 0; gq < COLS; gq++) begin : g_rq
      ae_requant u_rq (
        .clk(clk), .rst_n(rst_n),
        .in_vld(rq_v),
        .x(acc_row[gq*32 +: 32]),
        .m(rq_m_r), .s(rq_s_r),
        .out_vld(rq_vld[gq]),
        .y(rq_y[gq*8 +: 8])
      );
    end
  endgenerate

  assign busy = (st != S_IDLE);
  assign mac_cnt = mac_cnt_r;

  // 写回地址随 we 走（而非 st）：最后一拍的 we 会延续到状态切换后的下一拍，
  // 若按 st 门控会把地址压成 0，丢最后一列并踩坏低地址区。
  assign ctxa_addr = abase_r + {{4'd0}, kk};
  assign w_addr    = b_base[11:0] + kk[11:0];
  assign ctxb_we    = wb_we_r;
  assign ctxb_welane = wb_lanes_r;
  assign ctxb_wdata  = wb_data_r;
  always_comb begin
    if (wb_we_r) ctxb_addr = wb_addr_r;
    else         ctxb_addr = '0;
  end

  // 喂数：CTX A 口与 WRAM A 口回数即喂入
  assign a_feed_c = ctxa_rdata;
  assign b_feed_c = w_rdata;

  // tile_buf 捕获（requant 2 拍流水，独立计数）
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)     drb <= '0;
    else if (arr_clr) drb <= '0;
    else if (rq_vld[0]) begin
      for (int jj = 0; jj < COLS; jj++) tile_buf[drb][jj] <= rq_y[jj*8 +: 8];
      drb <= drb + 4'd1;
    end
  end

  // 主 FSM
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= S_IDLE; done <= 1'b0; arr_clr <= 1'b0;
      wb_we_r <= 1'b0; wb_lanes_r <= '0; mac_cnt_r <= '0; issue_d <= 1'b0; rq_v <= 1'b0;
      drain_row <= '0;
    end else begin
      done <= 1'b0; arr_clr <= 1'b0; rq_v <= 1'b0; wb_we_r <= 1'b0; wb_lanes_r <= '0;
      case (st)
        S_IDLE: if (start) st <= S_INIT;
        S_INIT: begin
            mt <= '0; kk <= '0; mac_cnt_r <= '0;
            abase_r <= a_base;                 // mt = 0 的行基址
            mt_cnt <= (m + 16'd15) >> 4;
            m16    <= (((m + 16'd15) >> 4) << 4);
            rq_m_r <= rq_m; rq_s_r <= rq_s;
            issue_d <= 1'b0;
            cgr_lo  <= j0 >> 4;
            tr_grps <= (((j0 + n_loc - 16'd1) >> 4) - (j0 >> 4)) + 4'd1;
            st <= S_CLR;
          end
        S_CLR: begin arr_clr <= 1'b1; kk <= '0; issue_d <= 1'b0; st <= S_FEED; end
        S_FEED: begin
            if (kk < k) begin
              issue_d <= 1'b1;
              kk <= kk + 16'd1;
              mac_cnt_r <= mac_cnt_r + 16 * COLS;  // 每 k 切片 16xCOLS 个 MAC
            end else begin
              issue_d <= 1'b0;
              if (!issue_d) st <= S_WAITD;
            end
          end
        S_WAITD: if (arr_done) begin st <= S_DRAIN; drain_row <= '0; rq_v <= 1'b1; end
        S_DRAIN: begin
            rq_v <= 1'b1;
            drain_row <= drain_row + 4'd1;
            if (drain_row == 4'd15) st <= S_DRLAT;
          end
        S_DRLAT: if (drb == 4'd15) begin
            wb_i8 <= '0; wb_row8 <= '0; wb_g <= '0;
            st <= y_tr ? S_WBTR : S_WB;
          end
        // 普通写回：addr = y_base + mt*N + j0 + col；lane = m
        S_WB: begin
            if (wb_i8 < n_loc[7:0]) begin
              wb_we_r <= 1'b1;
              wb_lanes_r <= m_lanes;
              wb_addr_r <= y_base + mtn + j0 + {8'd0, wb_i8};
              for (int i = 0; i < 16; i++) wb_data_r[i*8 +: 8] <= tile_buf[i][wb_i8];
            end
            if (wb_i8 == n_loc[7:0] - 8'd1) st <= S_NEXTMT; else wb_i8 <= wb_i8 + 8'd1;
          end
        // 转置写回：全局组 cgr = cgr_lo + wb_g，lane L 的全局列 c = cgr*16 + L，
        //   c ∈ [j0, j0+n_loc) 才写；addr = y_base + cgr*M16 + m；局部列 = c - j0
        S_WBTR: begin
            if (mt*16 + wb_row8 < m) begin
              wb_we_r <= 1'b1;
              wb_addr_r <= y_base + cgrm16 + (mt*16 + wb_row8);
              for (int L = 0; L < 16; L++) begin
                if ((cgr_lo + wb_g)*16 + L >= j0 &&
                    (cgr_lo + wb_g)*16 + L < j0 + n_loc) begin
                  wb_lanes_r[L] <= 1'b1;   // lane ≡ 全局列 mod 16
                  wb_data_r[L*8 +: 8] <= tile_buf[wb_row8][(cgr_lo + wb_g)*16 + L - j0];
                end
              end
            end
            if (wb_g + 16'd1 < tr_grps) wb_g <= wb_g + 4'd1;
            else begin
              wb_g <= '0;
              wb_row8 <= wb_row8 + 8'd1;
              if (wb_row8 == 8'd15) st <= S_NEXTMT;
            end
          end
        S_NEXTMT: begin
            if (mt + 16'd1 >= mt_cnt) st <= S_FIN;
            else begin
              mt <= mt + 16'd1;
              abase_r <= a_base + mtp_k;
              st <= S_CLR;
            end
          end
        S_FIN: begin done <= 1'b1; st <= S_IDLE; end
        default: st <= S_IDLE;
      endcase
    end
  end

  // lane 屏蔽（普通写回行有效）
  logic [15:0] m_lanes;
  always_comb begin
    m_lanes = '0;
    for (int i = 0; i < 16; i++) m_lanes[i] = (mt*16 + i < m);
  end
endmodule
`endif
