// ae_gemm.sv — GEMM 引擎：Y = requant(A * B)，B 一律来自 WRAM（COLS lane-bank）
// R3C 方案 C：FSM 拆成喂数道（st_f）+ 读出道（st_r）两段并行流水。
//   喂数道：每个行组 k 切片背靠背喂数（行组周期 k+2 拍），末切片隔 1 拍发射
//     末脉冲（feed_pulse，随 A 链偏斜传播）——PE 到拍快照 acc→snap 并清零，
//     下一行组喂数随即开始（脉冲发射由 pulse_ok 门控：上一行组 drain 已推进到
//     末 4 行，快照不会再被覆盖前未读）。
//   读出道：等末脉冲扫过行 0 全列（ptap 延迟线 COLS+1 拍）→ slot 对齐 →
//     量化读出 64 拍（从快照侧）→ tile_buf → 写回（普通/转置）。单 tile_buf 串行
//     链（与 R3C 模型的 68+wb 串行口径一致；双缓冲在 COLS+50 停拍地板下无收益）。
//   行组周期 = max(k+2, 2+64+3+wb, COLS+50)；末行组读出链完整跑完才 done。
// 数据通路/寻址/写回语义与 R1+R2 完全一致（位精确）：
//   A: CTX k-major（lane = m mod 16，地址 = a_base + (m div16)*K + k），CTX A 口直读
//   B: WRAM（lane = 组内局部列 j，地址 = b_base + k）—— 1 拍/k（发读-回数-喂阵流水）
//   Y: 写回 CTX B 口，全局列 c = j0 + 局部列：
//      普通 addr = y_base + mt*N + c（lane = m mod 16；N = 全局 n）
//      转置 addr = y_base + (c div16)*M16 + m（lane = c mod 16，V 给 PV；
//               一个 COLS 宽组跨若干全局 16-lane 组，按全局组迭代）
//   循环序 mt（B 一次驻留全 mt 复用）；尾 tile 用 lane-we 屏蔽；尾列用 n_loc 屏蔽。
// requant：NGRP=COLS/4 套 rq_ms(SHARE=4, XW=27) 时分复用（二轮门 1 拍板方案）——
//   drain 从 16 拍展开成 64 拍（每行保持 4 拍，slot 0..3 各采一列；首拍对齐 slot==0）。
//   x 取 acc 低 27b：|acc| ≤ K·2^14 ≤ 2^26（K ≤ 4096 = W_WORDS）——R3C 快照 27b
//   与该口径逐位一致（ae_pe snap[26:0]，读出侧符号扩展回 32b）。COLS 必须是 4 的倍数。
`ifndef AE_GEMM_SV
`define AE_GEMM_SV
module ae_gemm #(
  parameter int COLS = 96
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
  output logic [31:0] mac_cnt,
  output logic        wb_active   // R2：写回阶段（SR_WB/SR_WBTR），供 DMA CTX 预取停拍
);
  // ---- 喂数道 / 读出道状态机 ----
  typedef enum logic [2:0] {SF_IDLE, SF_INIT, SF_FEED, SF_PWAIT, SF_TAIL, SF_FIN} sf_e;
  typedef enum logic [2:0] {SR_WAIT, SR_DALIGN, SR_DRN, SR_LAT, SR_WB, SR_WBTR, SR_DONE} sr_e;
  sf_e st_f;
  sr_e st_r;

  logic [15:0] mt_f, mt_r;        // 喂数道 / 读出道各自的行组序号
  logic [15:0] kk;
  logic [15:0] mt_cnt, m16;
  logic [15:0] cgr_lo;
  logic [3:0]  wb_g;       // 转置写回的全局 16-lane 组号偏移
  logic [3:0]  tr_grps;    // 本组跨的全局 16-lane 组数

  logic issue_d;
  logic feed_pulse;          // R3C 末脉冲（本拍为 1 = 脉冲进阵拍）
  logic arr_clr;             // 复位兜底清零（仅描述符起点发一次；行组清零靠末脉冲）
  logic [127:0] a_feed_c;
  logic [COLS*8-1:0] b_feed_c;

  logic [3:0]  drain_row;
  logic [COLS*32-1:0] acc_row;
  logic        rq_v;
  logic [COLS-1:0] rq_vld;
  logic [COLS*8-1:0] rq_y;
  logic [7:0]  tile_buf [0:15][0:COLS-1];
  logic signed [15:0] rq_m_r;
  logic [7:0]  rq_s_r;
  logic [3:0]  drb;
  // requant 时分复用参数与捕获相位
  localparam int RQ_SH = 4;             // rq_ms 复用列数（SHARE）
  localparam int RQ_XW = 27;            // acc 截位宽（|acc| ≤ 2^26）；= ae_pe snap 位宽
  localparam int NGRP  = COLS / RQ_SH;  // rq_ms 套数（96 -> 24）
  logic [1:0]  slot_grp [0:NGRP-1];
  wire  [1:0]  slot_ph = slot_grp[0];   // 各组 slot 锁步（同 rst 同自由轮转）
  logic [1:0]  slot_out;                // 本拍输出的组内列号（由 out_vld 单热译出）
  logic        cap_en, cap_done;
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
  assign mtp_k  = (mt_f + 16'd1) * k;              // 喂数道：下一行组的 A 行基址
  assign mtn    = mt_r * n;                        // SR_WB 写回行基址
  assign cgrm16 = ({12'd0, wb_g} + cgr_lo) * m16;  // SR_WBTR 转置组基址

  ae_sysarr #(.ROWS(16), .COLS(COLS)) u_arr (
    .clk(clk), .rst_n(rst_n),
    .clr(arr_clr), .feed_vld(issue_d), .feed_pulse(feed_pulse),
    .a_feed(a_feed_c), .b_feed(b_feed_c),
    .drain_row(drain_row), .acc_row(acc_row)
  );

  // requant 时分复用：一套 rq_ms 服务 4 列（in_vld 拉满 + x_bus 同摆 4 列，
  // rq_ms 内部只采 slot 对应列，天然满足驱动契约）
  genvar gq, gc;
  generate
    for (gq = 0; gq < NGRP; gq++) begin : g_rq
      logic [RQ_SH*RQ_XW-1:0] xb;
      for (gc = 0; gc < RQ_SH; gc++) begin : g_x
        assign xb[gc*RQ_XW +: RQ_XW] = acc_row[(gq*RQ_SH+gc)*32 +: RQ_XW];
      end
      // T_MAX=39：s∈[8,47] 全覆盖（HB 真实标定 s∈[21,27]，见 02_quant/hw_calib_results.json；
      // 桶形移位在 rq_v2 T1 拍，流水仍 2 拍，drain 时序不变）
      rq_ms #(.SHARE(RQ_SH), .XW(RQ_XW), .T_MAX(39)) u_ms (
        .clk(clk), .rst_n(rst_n),
        .in_vld({RQ_SH{rq_v}}),
        .x_bus(xb),
        .m(rq_m_r), .s(rq_s_r),
        .out_vld(rq_vld[gq*RQ_SH +: RQ_SH]),
        .y_bus(rq_y[gq*RQ_SH*8 +: RQ_SH*8]),
        .slot_o(slot_grp[gq])
      );
    end
  endgenerate

  assign busy = (st_f != SF_IDLE);
  assign mac_cnt = mac_cnt_r;
  // R2：GEMM 写回阶段标志（SR_WB/SR_WBTR），供 DMA CTX 预取停拍用
  //   GEMM 在写回时占 CTX B 口，CTX 预取写 B 口必须让拍
  assign wb_active = (st_r == SR_WB) || (st_r == SR_WBTR);

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

  // ---- R3C 末脉冲扫描对齐 + 快照放行 ----
  // ptap：feed_pulse 的 (COLS+1) 拍延迟线。末脉冲进阵后，行 0 的最后一个 PE
  //   （列 COLS-1）在 p+COLS 拍完成快照；ptap[COLS] 在 p+COLS+1 拍置位，
  //   读出道由此放行（DALIGN/DRN 再 +2 拍以上，行 0 读出安全裕量 ≥2 拍）。
  logic [COLS:0] ptap;
  logic          swept_r;      // 本拍起可进入 DALIGN（SR_WAIT 消费后自清）
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ptap <= '0; swept_r <= 1'b0;
    end else begin
      ptap <= {ptap[COLS-1:0], feed_pulse};
      if (ptap[COLS])           swept_r <= 1'b1;   // 末脉冲扫过行 0 全列
      else if (st_r == SR_WAIT) swept_r <= 1'b0;   // SR_WAIT 消费掉（等下一组）
    end
  end

  // pend_drain：有已发射、尚未完成 drain 的末脉冲在飞。脉冲发射拍置位，
  //   drain 末拍（15 行 × slot 3）清零。pulse_ok 门控：
  //   * 无在飞脉冲 → 放行（首组 / 上组 drain 已完）；
  //   * drain 进行到末 4 行（drain_row ≥ 12 ⇒ 当前拍 ≥ S+48，脉冲下一拍 ≥ S+49
  //     ≥ 末行最后一次读出拍 S+63 的下一拍）→ 放行；
  //   * 其余（脉冲扫描中 / drain 前 12 行 / 对齐中）→ 阻塞（SF_PWAIT 等待）。
  logic pend_drain;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)                              pend_drain <= 1'b0;
    else if (feed_pulse)                     pend_drain <= 1'b1;   // 发射优先（同拍收尾让位）
    else if (st_r == SR_DRN && drain_row == 4'd15 && slot_ph == 2'd3)
                                             pend_drain <= 1'b0;   // drain 收尾拍
  end
  wire pulse_ok = !pend_drain ||
                  ((st_r == SR_DRN) && (drain_row >= 4'd12));

  // tile_buf 捕获：rq_ms 输出逐列错拍（呈现后 +3 拍到达）。各组 slot 锁步 →
  // 每拍恰有一列（组内）输出，列相位由组 0 的 out_vld 单热译出。到达拍直写
  // tile_buf（列 ≡ slot_out mod 4，行 = drb，行内 4 拍收齐后 drb++）。
  always_comb begin
    cap_en   = |rq_vld[3:0];
    slot_out = rq_vld[1] ? 2'd1 : rq_vld[2] ? 2'd2 : rq_vld[3] ? 2'd3 : 2'd0;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      drb <= '0; cap_done <= 1'b0;
    end else if (st_r == SR_WAIT && swept_r) begin
      drb <= '0; cap_done <= 1'b0;      // 新行组读出启动：捕获游标复位
    end else if (cap_en) begin
      for (int g = 0; g < NGRP; g++)
        tile_buf[drb][g*RQ_SH + slot_out] <= rq_y[(g*RQ_SH + slot_out)*8 +: 8];
      if (slot_out == 2'd3) begin
        drb <= drb + 4'd1;
        if (drb == 4'd15) cap_done <= 1'b1;
      end
    end
  end

  // ---- 喂数道 FSM：行组背靠背喂数 + 末脉冲发射 ----
  // 行组时间线（相对 SF_FEED 首拍）：拍 0 地址引导（issue_d=0，kk=0 驱动地址，
  //   CTX/WRAM 回数寄存）→ 拍 1..k 喂 k 个切片 → 拍 k+1 = SF_TAIL（feed_pulse=1，
  //   末脉冲进阵；同拍决出续喂/收尾）→ 下一组拍 0。背靠背行组周期 = k+2。
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st_f <= SF_IDLE; done <= 1'b0; arr_clr <= 1'b0;
      mac_cnt_r <= '0;
      issue_d <= 1'b0; feed_pulse <= 1'b0;
    end else begin
      done <= 1'b0; arr_clr <= 1'b0; feed_pulse <= 1'b0;
      case (st_f)
        SF_IDLE: if (start) st_f <= SF_INIT;
        SF_INIT: begin
            mt_f <= '0; kk <= '0; mac_cnt_r <= '0;
            abase_r <= a_base;                 // mt = 0 的行基址
            mt_cnt <= (m + 16'd15) >> 4;
            m16    <= (((m + 16'd15) >> 4) << 4);
            rq_m_r <= rq_m; rq_s_r <= rq_s;
            issue_d <= 1'b0;
            cgr_lo  <= j0 >> 4;
            tr_grps <= (((j0 + n_loc - 16'd1) >> 4) - (j0 >> 4)) + 4'd1;
            arr_clr <= 1'b1;                   // 复位兜底（acc 已由上组末脉冲清零，幂等）
            st_f <= SF_FEED;                   // st_r/mt_r 复位由读出道块看 SF_INIT 完成
          end
        // 喂 k 个切片；拍 k（kk==k 且 issue_d 仍 1）决出末脉冲能否下一拍发射
        SF_FEED: begin
            if (kk < k) begin
              issue_d <= 1'b1;
              kk <= kk + 16'd1;
              mac_cnt_r <= mac_cnt_r + 16 * COLS;  // 每 k 切片 16xCOLS 个 MAC
            end else begin
              issue_d <= 1'b0;
              if (issue_d) begin
                if (pulse_ok) begin
                  feed_pulse <= 1'b1;   // 末脉冲下一拍进阵（与本组最后切片隔 1 拍）
                  st_f <= SF_TAIL;
                end else begin
                  st_f <= SF_PWAIT;     // 读出道未放行：等上一组快照消费完
                end
              end
              // kk==k 且 !issue_d（拍 k+1）不会出现在 SF_FEED：已转 SF_TAIL/SF_PWAIT
            end
          end
        // 等读出道放行再发射末脉冲（此间 issue_d=0，阵列无积累，安全）
        SF_PWAIT: begin
            if (pulse_ok) begin
              feed_pulse <= 1'b1;
              st_f <= SF_TAIL;
            end
          end
        // 末脉冲进阵拍：决出续喂下一行组（背靠背，周期 k+2）或转收尾
        SF_TAIL: begin
            if (mt_f + 16'd1 >= mt_cnt) begin
              st_f <= SF_FIN;                 // 末组：等读出道跑完
            end else begin
              mt_f <= mt_f + 16'd1;
              abase_r <= a_base + mtp_k;      // 下一行组 A 行基址
              kk <= '0;
              st_f <= SF_FEED;                // 下一拍 = 下一组地址引导拍
            end
          end
        SF_FIN: if (st_r == SR_DONE) begin done <= 1'b1; st_f <= SF_IDLE; end
        default: st_f <= SF_IDLE;
      endcase
    end
  end

  // ---- 读出道 FSM：快照 → 量化 → tile_buf → 写回（与喂数道并行）----
  // 注意：本块独占 st_r/mt_r/rq_v/drain_row/wb_* 全部读出侧寄存器
  //（SF_INIT 的复位经 st_f==SF_INIT 条件在本块内完成——跨 always 块多驱动会
  //  让默认清零每拍覆盖写回置位，CTX 一次都写不进去）。
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st_r <= SR_WAIT; rq_v <= 1'b0; drain_row <= '0;
      mt_r <= '0; wb_i8 <= '0; wb_row8 <= '0; wb_g <= '0;
      wb_we_r <= 1'b0; wb_lanes_r <= '0;
    end else begin
      wb_we_r <= 1'b0; wb_lanes_r <= '0;   // 写回默认单拍有效（本块内覆盖）
      if (st_f == SF_INIT) begin
        st_r <= SR_WAIT; mt_r <= '0;       // 新描述符：读出道回等待位
      end else begin
      case (st_r)
        // 等本组末脉冲扫过行 0 全列（swept_r；若读出链忙则顺带等链排空）
        SR_WAIT: if (swept_r) begin
            st_r <= SR_DALIGN;
            drain_row <= '0;
          end
        // slot 对齐：rq_ms 采样在拍有效当拍，首拍（rq_v 变 1 的下一拍）必须落在
        // slot==0 —— 在 slot==3 的拍置位 rq_v 即可（最多等 4 拍）。
        SR_DALIGN: if (slot_ph == 2'd3) begin st_r <= SR_DRN; rq_v <= 1'b1; end
        // drain 展开：16 行 × 4 拍 = 64 拍呈现；每行内 slot 0..3 各采一列
        //（in_vld 拉满 + x_bus 同摆 4 列，rq_ms 内部按 slot 选列）。
        SR_DRN: begin
            rq_v <= 1'b1;
            if (slot_ph == 2'd3) begin
              drain_row <= drain_row + 4'd1;
              if (drain_row == 4'd15) begin
                rq_v <= 1'b0;
                st_r <= SR_LAT;
              end
            end
          end
        SR_LAT: if (cap_done) begin
            wb_i8 <= '0; wb_row8 <= '0; wb_g <= '0;
            st_r <= y_tr ? SR_WBTR : SR_WB;
          end
        // 普通写回：addr = y_base + mt_r*N + j0 + col；lane = m
        SR_WB: begin
            if (wb_i8 < n_loc[7:0]) begin
              wb_we_r <= 1'b1;
              wb_lanes_r <= m_lanes;
              wb_addr_r <= y_base + mtn + j0 + {8'd0, wb_i8};
              for (int i = 0; i < 16; i++) wb_data_r[i*8 +: 8] <= tile_buf[i][wb_i8];
            end
            if (wb_i8 == n_loc[7:0] - 8'd1) begin
              if (mt_r + 16'd1 >= mt_cnt) st_r <= SR_DONE;
              else begin mt_r <= mt_r + 16'd1; st_r <= SR_WAIT; end
            end else wb_i8 <= wb_i8 + 8'd1;
          end
        // 转置写回：全局组 cgr = cgr_lo + wb_g，lane L 的全局列 c = cgr*16 + L，
        //   c ∈ [j0, j0+n_loc) 才写；addr = y_base + cgr*M16 + m；局部列 = c - j0
        SR_WBTR: begin
            if (mt_r*16 + wb_row8 < m) begin
              wb_we_r <= 1'b1;
              wb_addr_r <= y_base + cgrm16 + (mt_r*16 + wb_row8);
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
              if (wb_row8 == 8'd15) begin
                if (mt_r + 16'd1 >= mt_cnt) st_r <= SR_DONE;
                else begin mt_r <= mt_r + 16'd1; st_r <= SR_WAIT; end
              end
            end
          end
        SR_DONE: ;  // 驻留：SF_FIN 看到后发 done；下一次 SF_INIT 重启
        default: st_r <= SR_WAIT;
      endcase
      end
    end
  end

  // lane 屏蔽（普通写回行有效）
  logic [15:0] m_lanes;
  always_comb begin
    m_lanes = '0;
    for (int i = 0; i < 16; i++) m_lanes[i] = (mt_r*16 + i < m);
  end
endmodule
`endif
