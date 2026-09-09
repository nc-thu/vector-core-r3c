// ae_gemm_p2.sv — Pack2 GEMM 引擎（B8×R3C 集成 H1 门）
// ----------------------------------------------------------------------------
// v5/H2 时序收口（2026-09-08，目标收掉 eng48 布线 −0.169ns 的两条读出腿）：
//   ① drain_row 扇出：单寄存器一拍驱动全部逻辑列 16:1 选择 → NREP 份同 D
//      副本寄存器分组驱动（每份 REP_EACH=4 物理列），值逐拍相同，语义不变。
//   ② requant 入口：acc_rq_r → 4:1 slot mux → 27×8 乘法挤一拍 → 换 rq_ms_x
//      （mux 后加一拍 x_sel_r）。核心数据滞后 2 拍，FSM 相位整体前移一拍
//      补偿：rq_v 在 DALIGN slot==2 发（原 3）、换行 slot==1（原 2）、停发与
//      pend 走尾 slot==2（原 3）。核心消费窗口与 H1 版逐拍重合，拍数不变。
// ----------------------------------------------------------------------------
// 以 ae_gemm（R3C，hb_fpga_impl/22_r3c_rtl）为底本的集成版，FSM 结构逐态保留。
// 相对底本的全部差异（H1 门要验证的就是这些）：
//   1. PCOLS 物理列 × Pack2 = LOGICAL=2·PCOLS 逻辑列；w_rdata 每物理列 16b。
//   2. PULSE_DLY=5：FSM 仍按 R3C 节奏在末切片隔 1 拍发脉冲（拍 k+1），
//      经 5 拍延迟线才进阵列（拍 k+6）——补偿 Pack2 PE 的 6 拍积落地深度
//      （链 1 + S0 1 + MREG 1 + PREG 1 + stage1 1 + stage2 1）。
//      安全窗口 PD∈{5,6}（实测：PD=7 时脉冲拍丢弃下组首积，PD≥8 污染快照）。
//   3. requant 套数 = LOGICAL/4（PCOLS=48 → 24 套，与 R3C-96 完全相同——
//      v5 模型 B3"requant 必须加倍"在本映射下是布线问题不是吞吐问题，
//      DRAIN 保持 64 拍，无 +10.8% 惩罚）。
//   4. ptap/pend_drain：pend_drain 按原始（延迟前）脉冲置位（与 R3C 的
//      发射互锁语义逐拍一致）；ptap 延迟线 tapped 延迟后脉冲（读出放行
//      必须跟物理快照对齐）。
// 其余（CTX/WRAM 寻址、写回普通/转置、lane 屏蔽、requant 时分复用）与
// 底本逐字一致——逻辑列展平后 acc_row/tile_buf 的索引语义不变。
`ifndef AE_GEMM_P2_SV
`define AE_GEMM_P2_SV
module ae_gemm_p2 #(
  parameter int PCOLS     = 4,    // 物理列（逻辑列 = 2×PCOLS，须为 4 的倍数）
  parameter int PULSE_DLY = 5    // 末脉冲发射滞后（PE 积落地深度 − R3C 的 1 拍）
)(
  input  logic clk,
  input  logic rst_n,
  input  logic start,
  output logic busy,
  output logic done,
  input  logic [15:0] m,
  input  logic [15:0] n,
  input  logic [15:0] n_loc,    // 本组列数（≤ 2·PCOLS；尾组更小）
  input  logic [15:0] j0,
  input  logic [15:0] k,
  input  logic [19:0] a_base, b_base, y_base,
  input  logic        y_tr,
  input  logic signed [15:0] rq_m,
  input  logic [7:0]  rq_s,
  output logic [19:0] ctxa_addr,
  input  logic [16*8-1:0] ctxa_rdata,
  output logic        ctxb_we,
  output logic [15:0] ctxb_welane,
  output logic [19:0] ctxb_addr,
  output logic [16*8-1:0] ctxb_wdata,
  output logic [11:0] w_addr,
  input  logic [PCOLS*16-1:0] w_rdata,     // 每物理列 {w[2j+1], w[2j]}
  output logic [31:0] mac_cnt,
  output logic        wb_active
);
  localparam int LOGICAL = PCOLS * 2;

  typedef enum logic [2:0] {SF_IDLE, SF_INIT, SF_FEED, SF_PWAIT, SF_TAIL, SF_FIN} sf_e;
  typedef enum logic [2:0] {SR_WAIT, SR_DALIGN, SR_DRN, SR_LAT, SR_WB, SR_WBTR, SR_DONE} sr_e;
  sf_e st_f;
  sr_e st_r;

  logic [15:0] mt_f, mt_r;
  logic [15:0] kk;
  logic [15:0] mt_cnt, m16;
  logic [15:0] cgr_lo;
  logic [3:0]  wb_g;
  logic [3:0]  tr_grps;

  logic issue_d;
  logic feed_pulse_raw;       // FSM 原始末脉冲（拍 k+1，R3C 节奏）
  logic feed_pulse;           // 延迟 PULSE_DLY 后进阵列（拍 k+6）
  logic [PULSE_DLY-1:0] pulse_dly;
  logic arr_clr;
  logic [127:0] a_feed_c;
  logic [PCOLS*16-1:0] b_feed_c;

  logic [3:0]  drain_row;
  logic        r15_seen;
  // H2 ①：drain_row 副本（同一 D、同拍同值；每份只驱动 REP_EACH 个物理列的
  // 16:1 快照选择，切掉单寄存器 96 列扇出这条布线腿）
  localparam int REP_EACH = 4;
  localparam int NREP     = (PCOLS + REP_EACH - 1) / REP_EACH;
  logic [NREP*4-1:0] drain_row_rep;   // 打包向量（[ri*4 +: 4] = 第 ri 份副本）
  logic [3:0]  drain_row_d;   // 次态（本尊与副本共用，见 ptap 块后的 comb）
  logic [LOGICAL*32-1:0] acc_row;
  // 读出寄存：快照行选择 mux → requant 桶形移位原来是跨模块单拍长路径
  // （drain_row/snap_r → 16:1 mux → rq_v2 barrel → plo_r，@3.298ns 布线
  //  违例 −1.097ns）。在引擎侧加一拍 acc_rq_r（H1 时换行提前到 slot==2，
  //  H2 因 rq_ms_x 再提前到 slot==1；行 r 窗口内 acc_rq_r 恒为行 r）。
  // 代价：drain_row==15 提前一个窗口到来（第 15 行的 ph3 拍），停发条件若仍
  //  挂 drain_row==15 会漏掉第 16 行 → r15_seen（=15 后首个 ph0 拍置位）。
  logic [LOGICAL*RQ_XW-1:0] acc_rq, acc_rq_r;
  logic        rq_v;
  logic [LOGICAL-1:0] rq_vld;
  logic [LOGICAL*8-1:0] rq_y;
  logic [7:0]  tile_buf [0:15][0:LOGICAL-1];
  logic signed [15:0] rq_m_r;
  logic [7:0]  rq_s_r;
  logic [3:0]  drb;
  localparam int RQ_SH = 4;
  localparam int RQ_XW = 27;
  localparam int NGRP  = LOGICAL / RQ_SH;   // PCOLS=48 → 24 套（= R3C-96）
  logic [1:0]  slot_grp [0:NGRP-1];
  wire  [1:0]  slot_ph = slot_grp[0];
  logic [1:0]  slot_out;
  logic        cap_en, cap_done;
  logic [7:0]  wb_i8, wb_row8;
  logic [31:0] mac_cnt_r;

  logic        wb_we_r;
  logic [15:0] wb_lanes_r;
  (* use_dsp = "no" *) logic [19:0] wb_addr_r;
  logic [16*8-1:0] wb_data_r;
  (* use_dsp = "no" *) logic [19:0] abase_r;
  (* use_dsp = "no" *) logic [31:0] mtn_r, cgrm16;
  // 地址乘法换增量：mt_f/mt_r 连续递增，(mt_f+1)*k 与 mt_r*n 都是每组 +k/+n。
  // 乘法版在 3.298ns 下是全部 4 条违例路径（mt_f→abase_r −0.280ns）。
  assign cgrm16 = ({12'd0, wb_g} + cgr_lo) * m16;

  // 末脉冲延迟线：FSM 节奏不变（k+1 发射），+PULSE_DLY 进阵列
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) pulse_dly <= '0;
    else        pulse_dly <= {pulse_dly[PULSE_DLY-2:0], feed_pulse_raw};
  end
  assign feed_pulse = pulse_dly[PULSE_DLY-1];

  ae_sysarr_p2 #(.ROWS(16), .PCOLS(PCOLS)) u_arr (
    .clk(clk), .rst_n(rst_n),
    .clr(arr_clr), .feed_vld(issue_d), .feed_pulse(feed_pulse),
    .a_feed(a_feed_c), .b_feed(b_feed_c),
    .drain_row_rep(drain_row_rep), .acc_row(acc_row)
  );

  genvar gq, gc;
  generate
    for (gq = 0; gq < NGRP; gq++) begin : g_rq
      logic [RQ_SH*RQ_XW-1:0] xb;
      for (gc = 0; gc < RQ_SH; gc++) begin : g_x
        assign xb[gc*RQ_XW +: RQ_XW] = acc_rq_r[(gq*RQ_SH+gc)*RQ_XW +: RQ_XW];
      end
      rq_ms_x #(.SHARE(RQ_SH), .XW(RQ_XW), .T_MAX(39)) u_ms (
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
  assign wb_active = (st_r == SR_WB) || (st_r == SR_WBTR);

  assign ctxa_addr = abase_r + {{4'd0}, kk};
  assign w_addr    = b_base[11:0] + kk[11:0];
  assign ctxb_we    = wb_we_r;
  assign ctxb_welane = wb_lanes_r;
  assign ctxb_wdata  = wb_data_r;
  always_comb begin
    if (wb_we_r) ctxb_addr = wb_addr_r;
    else         ctxb_addr = '0;
  end

  assign a_feed_c = ctxa_rdata;
  assign b_feed_c = w_rdata;

  // 读出通路寄存器（acc_row 27b 切片 → 一拍 → requant）
  always_comb begin
    for (int c = 0; c < LOGICAL; c++)
      acc_rq[c*RQ_XW +: RQ_XW] = acc_row[c*32 +: RQ_XW];
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) acc_rq_r <= '0;
    else        acc_rq_r <= acc_rq;
  end

  // ptap：延迟后脉冲的 (PCOLS+1) 拍延迟线（读出放行对齐物理快照）
  logic [PCOLS:0] ptap;
  logic           swept_r;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ptap <= '0; swept_r <= 1'b0;
    end else begin
      ptap <= {ptap[PCOLS-1:0], feed_pulse};
      if (ptap[PCOLS])           swept_r <= 1'b1;
      else if (st_r == SR_WAIT)  swept_r <= 1'b0;
    end
  end

  // H2 ①：drain_row 次态集中化——本尊与 NREP 份副本共用同一 D，任何拍同值。
  // 换行提前到 slot==1（rq_ms_x 输入流水多一拍，核心数据滞后 2 拍）。
  always_comb begin
    if (st_r == SR_WAIT && swept_r)                                   drain_row_d = 4'd0;
    else if (st_r == SR_DRN && slot_ph == 2'd1 && drain_row != 4'd15) drain_row_d = drain_row + 4'd1;
    else                                                              drain_row_d = drain_row;
  end

  // pend_drain：按原始脉冲置位。相对 R3C 底本修了一个潜伏 bug（H1 门 k=37
  // 应力用例击穿）：脉冲在 drain(row≥12) 逃生舱发射后，同一 drain 走完时
  // 原条件照样清 pend——但那个 drain 服务的是上一个脉冲，本次脉冲的 drain
  // 还没开始 → pend 提前归零 → 下一个 feed 完成即过早发射脉冲，覆盖未读
  // 快照（实测 row 5..15 全零）。R3C 实际负载 k≥64 时喂数腿长于逃生舱窗口，
  // 脉冲总在 drain 完成后发射，此洞从未打开。
  // 修法：svc_r 标记"正在走的 drain 是否服务最新脉冲"——发射脉冲清 0、
  // 读侧消费放行起走（SR_WAIT→DALIGN 转移拍）置 1、walk 结束仅在 svc_r=1
  // 时清 pend。（第一版挂在 ptap 脉冲拍不行：放行早于旧 walk 走完，旧 walk
  // 结束时 svc 已被重新置 1，洞没关上。）
  logic pend_drain, svc_r;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      pend_drain <= 1'b0; svc_r <= 1'b0;
    end else begin
      if (feed_pulse_raw) begin
        pend_drain <= 1'b1;
        svc_r      <= 1'b0;   // 在飞的 walk 不再代表最新脉冲
      end else if (st_r == SR_WAIT && swept_r) begin
        svc_r      <= 1'b1;   // 读侧真正消费放行（起走）才重臂：本次 walk 服务当前未决脉冲
      end else if (st_r == SR_DRN && r15_seen && slot_ph == 2'd2 && svc_r) begin
        pend_drain <= 1'b0;   // 只有服务过最新脉冲的 walk 完成才清
      end
    end
  end
  wire pulse_ok = !pend_drain ||
                  ((st_r == SR_DRN) && (drain_row >= 4'd12));

  always_comb begin
    cap_en   = |rq_vld[3:0];
    slot_out = rq_vld[1] ? 2'd1 : rq_vld[2] ? 2'd2 : rq_vld[3] ? 2'd3 : 2'd0;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      drb <= '0; cap_done <= 1'b0;
    end else if (st_r == SR_WAIT && swept_r) begin
      drb <= '0; cap_done <= 1'b0;
    end else if (cap_en) begin
      for (int g = 0; g < NGRP; g++)
        tile_buf[drb][g*RQ_SH + slot_out] <= rq_y[(g*RQ_SH + slot_out)*8 +: 8];
      if (slot_out == 2'd3) begin
        drb <= drb + 4'd1;
        if (drb == 4'd15) cap_done <= 1'b1;
      end
    end
  end

  // ---- 喂数道 FSM（与底本逐态一致；脉冲在 k+1 发原始版）----
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st_f <= SF_IDLE; done <= 1'b0; arr_clr <= 1'b0;
      mac_cnt_r <= '0;
      issue_d <= 1'b0; feed_pulse_raw <= 1'b0;
    end else begin
      done <= 1'b0; arr_clr <= 1'b0; feed_pulse_raw <= 1'b0;
      case (st_f)
        SF_IDLE: if (start) st_f <= SF_INIT;
        SF_INIT: begin
            mt_f <= '0; kk <= '0; mac_cnt_r <= '0;
            abase_r <= a_base;
            mt_cnt <= (m + 16'd15) >> 4;
            m16    <= (((m + 16'd15) >> 4) << 4);
            rq_m_r <= rq_m; rq_s_r <= rq_s;
            issue_d <= 1'b0;
            cgr_lo  <= j0 >> 4;
            tr_grps <= (((j0 + n_loc - 16'd1) >> 4) - (j0 >> 4)) + 4'd1;
            arr_clr <= 1'b1;
            st_f <= SF_FEED;
          end
        SF_FEED: begin
            if (kk < k) begin
              issue_d <= 1'b1;
              kk <= kk + 16'd1;
              mac_cnt_r <= mac_cnt_r + 16 * LOGICAL;
            end else begin
              issue_d <= 1'b0;
              if (issue_d) begin
                if (pulse_ok) begin
                  feed_pulse_raw <= 1'b1;   // k+1 发射；k+6 进阵列
                  st_f <= SF_TAIL;
                end else begin
                  st_f <= SF_PWAIT;
                end
              end
            end
          end
        SF_PWAIT: begin
            if (pulse_ok) begin
              feed_pulse_raw <= 1'b1;
              st_f <= SF_TAIL;
            end
          end
        SF_TAIL: begin
            if (mt_f + 16'd1 >= mt_cnt) begin
              st_f <= SF_FIN;
            end else begin
              mt_f <= mt_f + 16'd1;
              abase_r <= abase_r + {{4'd0}, k};   // 增量替代 (mt_f+1)*k 乘法
              kk <= '0;
              st_f <= SF_FEED;
            end
          end
        SF_FIN: if (st_r == SR_DONE) begin done <= 1'b1; st_f <= SF_IDLE; end
        default: st_f <= SF_IDLE;
      endcase
    end
  end

  // ---- 读出道 FSM（与底本逐态一致；tile_buf/写回按逻辑列宽）----
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st_r <= SR_WAIT; rq_v <= 1'b0; drain_row <= '0; drain_row_rep <= '0;
      mt_r <= '0; mtn_r <= '0; wb_i8 <= '0; wb_row8 <= '0; wb_g <= '0;
      wb_we_r <= 1'b0; wb_lanes_r <= '0;
    end else begin
      wb_we_r <= 1'b0; wb_lanes_r <= '0;
      // H2 ①：本尊与 NREP 份副本共用同一 D（comb 块见 ptap 后），任何拍同值。
      // 后面 case 臂一律不写 drain_row（写了会覆盖本尊但不覆盖副本，破坏同值
      // 不变式；SR_WAIT 臂的清零与 drain_row_d 同值，可保留）。
      drain_row <= drain_row_d;
      for (int ri = 0; ri < NREP; ri++) drain_row_rep[ri*4 +: 4] <= drain_row_d;
      if (st_f == SF_INIT) begin
        st_r <= SR_WAIT; mt_r <= '0; mtn_r <= '0;
      end else begin
        case (st_r)
          SR_WAIT: if (swept_r) begin
              st_r <= SR_DALIGN;
              drain_row <= '0;
              r15_seen <= 1'b0;
            end
          // H2 相位平移：rq_ms_x 输入流水让核心数据滞后 2 拍（acc_rq_r 一拍 +
          // x_sel_r 一拍），rq_v 提前一拍在 DALIGN slot==2 发（原 slot==3），
          // 核心消费窗口与 H1 版逐拍重合。换行移进 drain_row_d（slot==1）。
          SR_DALIGN: if (slot_ph == 2'd2) begin st_r <= SR_DRN; rq_v <= 1'b1; end
          SR_DRN: begin
            rq_v <= 1'b1;
            // 第 16 行窗口的 ph0 置 r15_seen，停发挂它（换行提前后
            // drain_row==15 提前一个窗口到来，直接挂会漏第 16 行）
            if (slot_ph == 2'd0 && drain_row == 4'd15) begin
              r15_seen <= 1'b1;
            end
            if (slot_ph == 2'd2 && r15_seen) begin
              rq_v <= 1'b0;
              st_r <= SR_LAT;
            end
          end
          SR_LAT: if (cap_done) begin
            wb_i8 <= '0; wb_row8 <= '0; wb_g <= '0;
            st_r <= y_tr ? SR_WBTR : SR_WB;
          end
          SR_WB: begin
            if (wb_i8 < n_loc[7:0]) begin
              wb_we_r <= 1'b1;
              wb_lanes_r <= m_lanes;
              wb_addr_r <= y_base + mtn_r + j0 + {8'd0, wb_i8};
              for (int i = 0; i < 16; i++) wb_data_r[i*8 +: 8] <= tile_buf[i][wb_i8];
            end
            if (wb_i8 == n_loc[7:0] - 8'd1) begin
              if (mt_r + 16'd1 >= mt_cnt) st_r <= SR_DONE;
              else begin mt_r <= mt_r + 16'd1; mtn_r <= mtn_r + {{16'd0}, n}; st_r <= SR_WAIT; end
            end else wb_i8 <= wb_i8 + 8'd1;
          end
          SR_WBTR: begin
            if (mt_r*16 + wb_row8 < m) begin
              wb_we_r <= 1'b1;
              wb_addr_r <= y_base + cgrm16 + (mt_r*16 + wb_row8);
              for (int L = 0; L < 16; L++) begin
                if ((cgr_lo + wb_g)*16 + L >= j0 &&
                    (cgr_lo + wb_g)*16 + L < j0 + n_loc) begin
                  wb_lanes_r[L] <= 1'b1;
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
                else begin mt_r <= mt_r + 16'd1; mtn_r <= mtn_r + {{16'd0}, n}; st_r <= SR_WAIT; end
              end
            end
          end
          SR_DONE: ;
          default: st_r <= SR_WAIT;
        endcase
      end
    end
  end

  logic [15:0] m_lanes;
  always_comb begin
    m_lanes = '0;
    for (int i = 0; i < 16; i++) m_lanes[i] = (mt_r*16 + i < m);
  end
endmodule
`endif
