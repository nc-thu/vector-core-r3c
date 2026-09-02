// ae_sched.sv — ★ Step-Invariant Scheduler（学术原语）
// ---------------------------------------------------------------------------
// workload 观察：流匹配去噪循环里，context K/V 投影与步数无关（step-invariant），
// 参考实现每步重算（仿真器实测：全模型参考实现 22% 周期、专家 74% 周期浪费于此）。
// 硬件原语 = 三件套：
//   1) 不变位图 bitmap[16]：OP_HOIST 首次算完 K/V 投影即置位（每层 1 bit）；
//   2) early-test：循环体内 stage 命中位图且 step>0 -> 直接跳过整个 GEMM
//      （结果已在 CTX 驻留，后续 stage 读同一地址 -> 位精确等价）；
//   3) hoist_en=0 时位图永不置位 -> 同一 bit 流退化为参考实现（对照实验开关）。
// 序列表驱动（网络形状全在描述符里），ng 循环在序列层展开。
// 取指流水：T_ADV（pc 更新）-> T_FETCH（地址驱动）-> T_LATCH（RAM 回数锁存）-> T_EXEC。
// ---------------------------------------------------------------------------
`ifndef AE_SCHED_SV
`define AE_SCHED_SV
module ae_sched #(
  parameter int SEQ_AW = 9,         // log2(SEQ_N=512)
  parameter int W_AW   = 12         // log2(W_WORDS)：WRAM 半区双缓冲按地址 bit[W_AW-1] 对分
)(
  input  logic clk,
  input  logic rst_n,
  input  logic start,
  input  logic hoist_en,            // 0 = REF 参考模式
  input  logic pf_en,               // ★ 0 = 无预取（与旧 RTL 逐拍一致的零漂移档）
  output logic busy,
  output logic done,
  // SEQ RAM
  output logic [SEQ_AW-1:0] seq_raddr,
  input  logic [255:0] seq_rdata,
  // 引擎 start/done
  output logic g_start,  input logic g_busy,  input logic g_done,
  output logic sm_start, input logic sm_busy, input logic sm_done,
  output logic cp_start, input logic cp_busy, input logic cp_done,
  output logic dma_start, input logic dma_busy, input logic dma_done,
  // 引擎选择（core 端口仲裁，one-hot）
  output logic eng_g, eng_sm, eng_cp, eng_dma,
  // 本 stage 描述符（core 连到各引擎参数）
  output logic [255:0] desc_o,
  // 性能计数
  input  logic [31:0] g_mac_cnt,
  output logic [31:0] cycles, gemm_cycles, dma_cycles, mac_total, skip_macs,
  output logic [15:0] skip_stages,
  // ★ 权重预取后台发射（core 端做 DMA 命令源 2:1 mux）
  output logic        pf_bg_start,   // 单拍脉冲：u_dma 在 D_IDLE 末拍锁存 pf_dma*
  output logic [31:0] pf_dmaaddr,
  output logic [17:0] pf_dmalen,
  output logic [2:0]  pf_dmatag,
  output logic [19:0] pf_dmabase
);
  typedef enum logic [3:0] {T_IDLE, T_FETCH, T_LATCH, T_EXEC, T_RUN_G, T_RUN_SM,
                            T_RUN_CP, T_RUN_DMA, T_SKIP, T_ADV, T_FIN} st_e;
  st_e st;

  logic [SEQ_AW-1:0] pc, loop_start;
  logic [10:0] step;
  logic        loop_seen, running;
  logic [15:0] bitmap;
  logic [255:0] desc_r;
  logic [31:0] cycles_r, gemm_c_r, dma_c_r, mac_t_r;
  (* use_dsp = "no" *) logic [31:0] skip_m_r;  // d_m*d_n*d_k 统计乘走 LUT
  (* use_dsp = "no" *) logic [31:0] dm_x_dn;   // 中间积也必须具名+标记，
  (* use_dsp = "no" *) logic [31:0] dmnk;      // 否则匿名表达式仍吃 1 个 DSP
  // 统计乘三级流水：两个 16x16 LUT 乘级联一拍跑不完 250MHz，中间插寄存。
  // T_EXEC 两次 skip 间隔 ≥4 拍（T_SKIP→T_ADV→T_FETCH→T_LATCH），无碰撞
  (* use_dsp = "no" *) logic [31:0] dm_x_dn_r, dmnk_r;
  logic dm1_v, dm2_v;
  assign dm_x_dn = d_m * d_n;
  assign dmnk    = dm_x_dn_r * d_k;
  wire skip_fire = (st == T_EXEC) && skip_hit &&
                   (d_op != 4'd15) && (d_op != 4'd3) && (d_op != 4'd4) && (d_op != 4'd5);
  logic [15:0] skip_n_r;
  logic        attn_next;           // ATTN_S: GEMM 完成后接 softmax

  // ---------------------------------------------------------------------------
  // ★ 权重预取（lookahead=1，只预取 TAG_W 的 OP_LOAD）
  // 不变量（结构保证）：pf_v=1 期间（PF_ISSUE 发射 → 消费），主 FSM 到达的下一个
  //   T_EXEC 必是 pf_pc_r 处的 OP_LOAD —— 因为预取只在 T_RUN_G/T_RUN_SM 窗口发射，
  //   目标 = pc_next（与 T_ADV 同式），窗口结束后主 FSM 恰好经 T_ADV→T_FETCH→
  //   T_LATCH→T_EXEC 走到该 LOAD；LOAD 不参与 skip（skip_fire 排除 op=4）。
  // 纪律（编译器契约，硬件 issue_ok 强制）：预取目标半区（b_base bit[W_AW-1]）≠
  //   在跑 GEMM 半区，且在跑 GEMM k ≤ 半区深度 —— 违纪自动不发射（退化串行）。
  // ---------------------------------------------------------------------------
  typedef enum logic [1:0] {PF_IDLE, PF_RD, PF_LAT, PF_ISSUE} pf_st_e;
  pf_st_e pf_st;
  logic             pf_v;           // 已发射未消费（单命令在飞）
  logic             pf_done;        // 在飞预取的 DMA 已完成（dma_done 归属无歧义）
  logic             pf_hit_r;       // T_EXEC 命中：本次 LOAD 不再发前台 dma_start
  logic [SEQ_AW-1:0] pf_pc_r;       // 预取目标描述符地址
  logic [31:0]      pf_desc_addr_r; // 影子命令组（PF_LAT 拍从 seq_rdata 切片锁存）
  logic [17:0]      pf_desc_len_r;
  logic [2:0]       pf_desc_tag_r;
  logic [19:0]      pf_desc_base_r;
  logic [3:0]       pf_desc_op_r;

  localparam int W_HALF = 2 ** (W_AW - 1);   // WRAM 半区深度 = W_WORDS/2

  // pc_next：与 T_ADV 完全同式（同一批输入寄存器），窗口内即「下一条描述符」
  wire loop_back = d_isend && d_inloop && (step + 11'd1 < d_steps);
  wire [SEQ_AW-1:0] pc_next = loop_back ? loop_start
                              : pc + {{(SEQ_AW-1){1'b0}}, 1'b1};
  wire pf_win      = pf_en && ((st == T_RUN_G) || (st == T_RUN_SM));
  wire pf_idle_ok  = pf_win && (pf_st == PF_IDLE) && !pf_v && !dma_busy;
  wire pf_issue_ok = pf_en && (pf_desc_op_r == 4'd4) && (pf_desc_tag_r == 3'd1)
                     && (pf_desc_base_r[W_AW-1] != d_bbase[W_AW-1])
                     && (d_k <= W_HALF);

  assign busy = running;
  assign desc_o = desc_r;
  assign cycles = cycles_r; assign gemm_cycles = gemm_c_r; assign dma_cycles = dma_c_r;
  assign mac_total = mac_t_r; assign skip_macs = skip_m_r; assign skip_stages = skip_n_r;
  assign pf_dmaaddr = pf_desc_addr_r;
  assign pf_dmalen  = pf_desc_len_r;
  assign pf_dmatag  = pf_desc_tag_r;
  assign pf_dmabase = pf_desc_base_r;

  // 描述符字段切片（packed struct 首字段在 MSB）
  wire [3:0]  d_op    = desc_r[255:252];
  wire [2:0]  d_bsrc  = desc_r[248:246];
  wire        d_causal = desc_r[245];
  wire        d_ytr   = desc_r[244];
  wire [15:0] d_m    = desc_r[243:228];
  wire [15:0] d_n    = desc_r[227:212];
  wire [15:0] d_k    = desc_r[211:196];
  wire [19:0] d_abase= desc_r[195:176];
  wire [19:0] d_bbase= desc_r[175:156];
  wire [19:0] d_ybase= desc_r[155:136];
  wire [15:0] d_spad = desc_r[135:120];
  wire [15:0] d_rqm  = desc_r[119:104];
  wire [7:0]  d_rqs  = desc_r[103:96];
  wire [3:0]  d_inv  = desc_r[95:92];
  wire [10:0] d_steps= desc_r[91:81];
  wire        d_inloop = desc_r[80];
  wire        d_isend  = desc_r[79];
  wire [17:0] d_dmalen = desc_r[78:61];
  wire [31:0] d_dmaaddr= desc_r[60:29];

  wire skip_hit = hoist_en & (d_inv != 4'hF) & d_inloop & (step != 11'd0) & bitmap[d_inv];

  assign eng_g  = (st == T_RUN_G);
  assign eng_sm = (st == T_RUN_SM);
  assign eng_cp = (st == T_RUN_CP);
  assign eng_dma= (st == T_RUN_DMA);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= T_IDLE; running <= 1'b0; done <= 1'b0;
      g_start <= 1'b0; sm_start <= 1'b0; cp_start <= 1'b0; dma_start <= 1'b0;
      pc <= '0; loop_start <= '0; step <= '0; loop_seen <= 1'b0; bitmap <= '0;
      cycles_r <= '0; gemm_c_r <= '0; dma_c_r <= '0; mac_t_r <= '0; skip_m_r <= '0; skip_n_r <= '0;
      dm1_v <= 1'b0; dm2_v <= 1'b0; dm_x_dn_r <= '0; dmnk_r <= '0;
      attn_next <= 1'b0;
      pf_st <= PF_IDLE; pf_bg_start <= 1'b0; pf_v <= 1'b0; pf_done <= 1'b0;
      pf_hit_r <= 1'b0; pf_pc_r <= '0;
      pf_desc_addr_r <= '0; pf_desc_len_r <= '0; pf_desc_tag_r <= '0;
      pf_desc_base_r <= '0; pf_desc_op_r <= '0;
    end else begin
      done <= 1'b0; g_start <= 1'b0; sm_start <= 1'b0; cp_start <= 1'b0; dma_start <= 1'b0;
      pf_bg_start <= 1'b0;
      if (running) cycles_r <= cycles_r + 32'd1;
      if (g_busy)  gemm_c_r <= gemm_c_r + 32'd1;
      if (dma_busy) dma_c_r <= dma_c_r + 32'd1;
      // skip 统计流水（T_EXEC 打拍 dm1_v → dm2_v → 累加，值延迟 2 拍，done 前必已冲刷）
      dm_x_dn_r <= dm_x_dn;  dmnk_r <= dmnk;
      dm1_v <= skip_fire;    dm2_v <= dm1_v;
      if (dm2_v) skip_m_r <= skip_m_r + dmnk_r;

      case (st)
        T_IDLE: if (start) begin
            pc <= '0; step <= '0; loop_seen <= 1'b0; bitmap <= '0; running <= 1'b1;
            cycles_r <= '0; gemm_c_r <= '0; dma_c_r <= '0; mac_t_r <= '0;
            skip_m_r <= '0; skip_n_r <= '0;
            pf_st <= PF_IDLE; pf_v <= 1'b0; pf_done <= 1'b0; pf_hit_r <= 1'b0;
            st <= T_FETCH;
          end
        T_FETCH: st <= T_LATCH;      // 地址已按 pc 驱动，等 SEQ RAM 回数
        T_LATCH: begin
            desc_r <= seq_rdata;
            if (seq_rdata[80] && !loop_seen) begin
              loop_start <= pc; loop_seen <= 1'b1;
            end
            st <= T_EXEC;
          end
        T_EXEC: begin
            attn_next <= 1'b0;
            case (d_op)
              4'd15: st <= T_FIN;                       // OP_DONE
              4'd3: begin cp_start <= 1'b1; st <= T_RUN_CP; end
              4'd4, 4'd5: begin
                if ((d_op == 4'd4) && pf_v && (pf_pc_r == pc)) begin
                  // ★ 命中预取：后台 DMA 已在飞/已完成，不发 dma_start，直接等 pf_done
                  pf_hit_r <= 1'b1;
                  st <= T_RUN_DMA;
                end else if (!dma_busy) begin
                  dma_start <= 1'b1;
                  pf_hit_r <= 1'b0;
                  st <= T_RUN_DMA;
                end
                // else：dma_busy=1 且未命中（不变量破坏时的保险路径，正常不可达）
                //       —— 原地等 DMA 空闲再串行发射，避免双命令压进单引擎
              end
              default: begin                            // GEMM / ATTN_S / HOIST
                if (skip_hit) begin
                  skip_n_r <= skip_n_r + 16'd1;
                  st <= T_SKIP;
                end else begin
                  g_start <= 1'b1;
                  attn_next <= (d_op == 4'd1);
                  st <= T_RUN_G;
                end
              end
            endcase
          end
        T_RUN_G: if (g_done) begin
            mac_t_r <= mac_t_r + g_mac_cnt;
            if (attn_next) begin sm_start <= 1'b1; st <= T_RUN_SM; end
            else if (d_op == 4'd2 && hoist_en && d_inv != 4'hF) begin
              bitmap[d_inv] <= 1'b1;                    // ★ 置不变位
              st <= T_ADV;
            end else st <= T_ADV;
          end
        T_RUN_SM: if (sm_done) st <= T_ADV;
        T_RUN_CP: if (cp_done) st <= T_ADV;
        T_RUN_DMA: begin
            if (pf_hit_r) begin
              // ★ 命中路径：等后台 DMA 完成（pf_done）再推进；pf_pc_r!=pc 为
              //   不变量破坏的保险出口（正常不可达，TB 断言会在更早处报警）
              if (pf_done || (pf_pc_r != pc)) begin
                pf_v <= 1'b0; pf_done <= 1'b0; pf_hit_r <= 1'b0;   // 消费
                st <= T_ADV;
              end
            end else if (dma_done) begin
              pf_v <= 1'b0; pf_done <= 1'b0;   // 串行兜底路径的残留清理（正常为 0）
              st <= T_ADV;
            end
          end
        T_SKIP: st <= T_ADV;
        T_ADV: begin
            if (loop_back) begin
              step <= step + 11'd1;
              pc <= loop_start;
            end else begin
              if (d_isend && d_inloop) step <= '0;      // 退出循环
              pc <= pc_next;                            // !loop_back 时 == pc+1
            end
            st <= T_FETCH;
          end
        T_FIN: begin
            done <= 1'b1; running <= 1'b0; st <= T_IDLE;
            pf_st <= PF_IDLE; pf_v <= 1'b0; pf_done <= 1'b0; pf_hit_r <= 1'b0;
          end
        default: st <= T_IDLE;
      endcase

      // ---- ★ pf 子状态机（SEQ 读口只在 PF_RD 一拍借给 pc_next）----
      case (pf_st)
        PF_IDLE: if (pf_idle_ok) pf_st <= PF_RD;
        PF_RD:   pf_st <= pf_win ? PF_LAT : PF_IDLE;    // 窗口已关：放弃（永远安全）
        PF_LAT: begin
            // seq_rdata 在本拍仍是 mem[pc_next]（PF_RD 拍寄存的读数），错过即失
            pf_desc_op_r   <= seq_rdata[255:252];
            pf_desc_tag_r  <= seq_rdata[248:246];
            pf_desc_base_r <= seq_rdata[175:156];
            pf_desc_len_r  <= seq_rdata[78:61];         // LOAD 用整个 18b（j0 复用段即此）
            pf_desc_addr_r <= seq_rdata[60:29];
            pf_pc_r        <= pc_next;
            pf_st <= pf_win ? PF_ISSUE : PF_IDLE;       // 窗口已关：放弃（影子锁存无害）
          end
        PF_ISSUE: begin
            if (pf_issue_ok) begin
              pf_bg_start <= 1'b1;                      // 下一拍为脉冲拍，影子组已稳定
              pf_v        <= 1'b1;
            end
            pf_st <= PF_IDLE;
          end
        default: pf_st <= PF_IDLE;
      endcase

      // pf_done：pf_v=1 期间唯一在飞的 DMA 就是预取命令（前台 DMA 在消费前不可能出现）
      if (pf_v && !pf_done && dma_done) pf_done <= 1'b1;
    end
  end

  // SEQ 读口复用：PF_RD 一拍切到 pc_next（预取取指），其余时间归主 FSM（pc）
  always_comb begin
    seq_raddr = (pf_st == PF_RD) ? pc_next : pc;
  end
endmodule
`endif
