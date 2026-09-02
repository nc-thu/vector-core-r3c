// ae_actv.sv — AE_ACTV 片上算子引擎（MVP：ACTV 直查表 + BIAS 两模式）
// ---------------------------------------------------------------------------
// 照 SM16 的接入方式（07_onchip_ops/NOTES §二 结构模板）：
//   CTX A 口广播读（一拍 16 行同列字节）→ 数据通路 → B 口 16-lane 写；
//   调度器串行 one-hot（新状态 T_RUN_A），不碰 WRAM 写交叉（WNS −1.038 最差路径）。
//   读数据寄存后再进组合级（softmax v2 教训：URAM 级联读出一拍走不完）。
// 描述符编码（256b 布局不动，op=6；golden gen_vectors.py 同位切片）：
//   b_src[2:0] = 子模式：0=ACTV（LUT 直查） 1=BIAS（y'=sat8((y·m+b_j)>>>s)）
//   m=行数  n=列数（=行 stride，同 softmax 的 n_cols 口径）
//   a_base = 张量 CTX 基址（原地变换：读列 j、发读后第 3 拍写回列 j）
//   b_base = 表映像 CTX 基址（编译器先用 OP_LOAD(TAG_CTX) 把表 DMA 进 CTX）
//   k      = BIAS 表项数（ACTV 固定 256，不读此字段）
//   rq_m/rq_s = BIAS 的 m（Q8.8 有符号）与右移 s（requant 同口径）
// 表映像布局（DMA TAG_CTX 装进 CTX 后由引擎自取）：
//   ACTV：★表项 x 在字 b_base+x 的全部 16 个 lane 槽位各放一份（映像 256 字，
//         512B）。为什么必须复制：CTX 广播读是"按槽位对号"的——字 W 的槽 L
//         字节只会送进 lane L 的数据通路，所以想让 16 个 lane 各自装满一整份
//         256 项的表，唯一办法是让每个字把同一字节放到 16 个槽里（第一版按
//         "项 x 在 lane x%16 @ b_base+x/16" 的位置图装载是错的：那样每个
//         lane 只能收到 x%16==L 的 16 项，其余 240 项永远到不了它）。装载
//         260 拍：稳态每拍发一个字地址、隔 2 捕获 rd_r 并武装 tbl 写。
//   BIAS：项 j 拆 lo/hi 两字节：lo 在 lane j%16 @ b_base + j/16，hi 同相位
//         @ b_base + NLO + j/16（NLO=ceil(k/16)，映像两区各按 16B 对齐）。
//         bias 只依赖列 j、全 lane 共一张表，槽位对号恰好正确。装载逐组
//         20 拍：发 lo 读 → 发 hi 读 → 捕 lo(rd_r) → 捕 hi(rd_r)
//         → 16 拍写共享 BRAM（写口 1 项/拍）。
// 数值语义（golden 逐位镜像；尾组 pad 行不写、原地不变）：
//   ACTV：y' = LUT[y&0xFF]
//   BIAS：y' = sat8(((y·m) + b_j) >>> s)，b_j 16b 有符号
// RUN 时序：发读列 jr（每拍 +1）→ rd_r / b_r 两拍后对齐（ctx_rdata 与
//   bias_rd 各打一拍）→ 组合出 wbyte → 寄存写（发读后第 3 拍落 B 口），
//   同 softmax P3 的 jr/jw 双指针口径。
// ---------------------------------------------------------------------------
`ifndef AE_ACTV_SV
`define AE_ACTV_SV
module ae_actv #(
  parameter int BIAS_D = 4096                 // bias 表深度（部署档最长 3072）
)(
  input  logic clk,
  input  logic rst_n,
  input  logic start,
  input  logic [2:0]  submode,                // 0=ACTV 1=BIAS
  input  logic [19:0] y_base,                 // 张量 CTX 基址（原地）
  input  logic [15:0] m_rows,                 // 有效行数（尾组不满按 lane 掩码跳写）
  input  logic [15:0] n_cols,                 // 列数 = 行 stride
  input  logic [19:0] tbl_base,               // 表映像 CTX 基址
  input  logic [15:0] tbl_len,                // BIAS 表项数
  input  logic [15:0] rq_m,                   // BIAS 乘子（Q8.8 有符号）
  input  logic [7:0]  rq_s,                   // BIAS 右移
  // CTX A 口（广播读）
  output logic [19:0] ctx_raddr,
  input  logic [127:0] ctx_rdata,
  // CTX B 口（16-lane 写）
  output logic        ctx_we,
  output logic [15:0] ctx_welane,
  output logic [19:0] ctx_waddr,
  output logic [127:0] ctx_wdata,
  output logic busy,
  output logic done
);
  localparam int BAW = (BIAS_D <= 2) ? 1 : $clog2(BIAS_D);

  typedef enum logic [3:0] {A_IDLE, A_LD, A_DRAIN, A_B_LO, A_B_HI,
                            A_B_C1, A_B_C2, A_B_WR, A_RUN, A_NEXT, A_FIN} st_e;
  st_e st;

  // ---- 参数锁存 ----
  logic        md_bias;
  logic [19:0] tbase_r, ybase_r, grp_base;
  logic [15:0] m_r, n_r, k_r, rq_m_r;
  logic [7:0]  rq_s_r;
  logic [11:0] nlo_r;                         // BIAS：lo/hi 区各占字数 ceil(k/16)

  // ---- 装载计数 ----
  logic [8:0]  li;                            // ACTV：发读字指针 0..256
  logic [11:0] bg_r;                          // BIAS：当前组号
  logic [3:0]  bwr_i;                         // BIAS：组内写序号
  logic [127:0] lo16, hi16;                   // 本组 lo/hi 捕获（lane L = 项 j&15）

  // ---- RUN 指针与流水 ----
  logic [16:0] jr, jw;                        // 发读列 / 写列（滞后 3 拍）
  logic        run_v1, run_v2;                // 读数据 2 拍有效流水
  logic [15:0] row;                           // 行组基行号
  logic [15:0] lane_mask;                     // 本组有效行掩码

  // ---- 读数据寄存（rd_r 与 b_r 同相位）----
  logic [127:0] rd_r;
  logic signed [15:0] b_r;

  // ---- bias 共享 BRAM（每列一项 16b，全 lane 同 j）----
  logic         bias_we;
  logic [BAW-1:0] bias_wa;
  logic [15:0]  bias_wd;
  logic [BAW-1:0] bias_ra;
  logic [15:0]  bias_rd;
  (* ram_style = "block" *) logic [15:0] bias_mem [0:BIAS_D-1];
  always_ff @(posedge clk) begin
    if (bias_we) bias_mem[bias_wa] <= bias_wd;
    bias_rd <= bias_mem[bias_ra];
  end

  // ---- 每 lane 256x8 直查表（分布式 RAM：装载期写、RUN 期异步读）----
  logic         lut_we;
  logic [7:0]   lut_wa;
  logic [127:0] lut_wd;
  logic [127:0] wbyte;
  for (genvar g = 0; g < 16; g++) begin : g_lut
    (* ram_style = "distributed" *) logic [7:0] tbl [0:255];
    always_ff @(posedge clk) if (lut_we) tbl[lut_wa] <= lut_wd[8*g +: 8];
    assign wbyte[8*g +: 8] = tbl[rd_r[8*g +: 8]];
  end

  // ---- BIAS 数据通路：16 份 8x16 LUT 乘 + 加 b_j + 桶移 + sat8 ----
  logic [127:0] bwbyte;
  for (genvar g = 0; g < 16; g++) begin : g_bias
    (* use_dsp = "no" *) logic signed [23:0] prod;    // 具名线网强制纯 LUT
    logic signed [24:0] accb;
    logic signed [24:0] p_sh;
    logic [7:0]         sat;
    assign prod = $signed(rd_r[8*g +: 8]) * $signed(rq_m_r);
    assign accb = prod + $signed({{9{b_r[15]}}, b_r});
    assign p_sh = accb >>> rq_s_r;
    always_comb begin
      if      (p_sh > 25'sd127)  sat = 8'd127;
      else if (p_sh < -25'sd128) sat = -8'sd128;
      else                       sat = p_sh[7:0];
    end
    assign bwbyte[8*g +: 8] = sat;
  end

  // ---- 组基行号 → 有效行掩码（尾组不满；m 显式传参——start 拍 m_r 尚未锁存，
  //      读 m_r 会拿到上一条描述符的 m，尾组掩码全错）----
  function automatic logic [15:0] row_mask(logic [15:0] r, logic [15:0] mv);
    for (int L = 0; L < 16; L++)
      row_mask[L] = ({1'b0, r} + L < {1'b0, mv});
  endfunction

  // ---- BIAS 写地址/末项判断的组合式 ----
  logic [15:0] bbase;                          // 16*组号
  assign bbase = {4'd0, bg_r} << 4;
  wire [16:0] bnext = {1'b0, bbase} + {13'd0, bwr_i} + 17'd1;   // 本写之后项数
  wire        blast = ({1'b0, bbase} + 17'd16 >= {1'b0, k_r});  // 本组是末组
  wire [15:0] bwa_full = bbase + {12'd0, bwr_i};                // 本写 bias 地址
  wire [16:0] nlo_w = {1'b0, tbl_len} + 17'd15;                 // ceil(k/16) 前置

  // ---- 读地址（每状态一个来源；RUN 用预读指针 jr）----
  always_comb begin
    ctx_raddr = grp_base;                     // 默认无害
    case (st)
      A_LD:      ctx_raddr = tbase_r + {11'd0, li};
      A_B_LO:    ctx_raddr = tbase_r + {8'd0, bg_r};
      A_B_HI:    ctx_raddr = tbase_r + {8'd0, nlo_r + bg_r};
      A_RUN:     ctx_raddr = grp_base + {3'd0, jr};
      default:   ctx_raddr = grp_base;
    endcase
  end
  assign bias_ra = jr[BAW-1:0];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= A_IDLE; busy <= 1'b0; done <= 1'b0;
      md_bias <= 1'b0; tbase_r <= '0; ybase_r <= '0; grp_base <= '0;
      m_r <= '0; n_r <= '0; k_r <= '0; rq_m_r <= '0; rq_s_r <= '0; nlo_r <= '0;
      li <= '0; lut_wa <= '0; bg_r <= '0; bwr_i <= '0;
      lo16 <= '0; hi16 <= '0;
      jr <= '0; jw <= '0; run_v1 <= 1'b0; run_v2 <= 1'b0;
      row <= '0; lane_mask <= '0;
      rd_r <= '0; b_r <= '0;
      bias_we <= 1'b0; bias_wa <= '0; bias_wd <= '0;
      lut_we <= 1'b0; lut_wd <= '0;
      ctx_we <= 1'b0; ctx_welane <= '0; ctx_waddr <= '0; ctx_wdata <= '0;
    end else begin
      done <= 1'b0; ctx_we <= 1'b0; ctx_welane <= '0;
      bias_we <= 1'b0; lut_we <= 1'b0;
      rd_r <= ctx_rdata;                      // 读流水无条件推进
      b_r  <= bias_rd;

      case (st)
        A_IDLE: if (start) begin
            busy    <= 1'b1;
            md_bias <= (submode == 3'd1);
            tbase_r <= tbl_base;
            ybase_r <= y_base;
            grp_base<= y_base;
            m_r <= m_rows; n_r <= n_cols; k_r <= tbl_len;
            rq_m_r <= rq_m; rq_s_r <= rq_s;
            nlo_r <= nlo_w[15:4];                               // ceil(k/16)
            li <= '0; lut_wa <= 8'hFF; bg_r <= '0; bwr_i <= '0;  // FF 起步：首拍武装写址 0
            row <= '0; lane_mask <= row_mask(16'd0, m_rows);
            st <= (submode == 3'd1) ? A_B_LO : A_LD;
          end
        // ---------------- 装载：ACTV 直查表（260 拍，稳态 1 字/拍）----------------
        //   发读指针 li 每拍 +1；rd_r 两拍后对齐（run_v1/run_v2 流水）；
        //   捕获拍把 rd_r 存进 lut_wd 并武装下一拍的 tbl 写（16 lane 同写 lw）。
        A_LD: begin
          run_v2 <= run_v1;
          if (li < 9'd256) begin
            li <= li + 9'd1;
            run_v1 <= 1'b1;
          end else run_v1 <= 1'b0;
          if (run_v2) begin                   // 本拍 rd_r = 表项 lut_wa+1（16 槽同值）
            lut_we <= 1'b1;
            lut_wd <= rd_r;
            lut_wa <= lut_wa + 8'd1;          // 写址 = 本拍表项号（下一拍写落盘）
            if (lut_wa == 8'd254) begin       // 正在武装最后一项（255）
              run_v1 <= 1'b0; run_v2 <= 1'b0;
              jr <= '0; jw <= '0;
              st <= A_DRAIN;
            end
          end
        end
        A_DRAIN: st <= A_RUN;                 // 末项 tbl 写在本拍末落盘
        // ---------------- 装载：BIAS 表（逐组 20 拍）----------------
        A_B_LO: st <= A_B_HI;                 // raddr = lo 组 bg_r
        A_B_HI: st <= A_B_C1;                 // raddr = hi 组
        A_B_C1: begin lo16 <= rd_r; st <= A_B_C2; end   // rd_r = lo 组
        A_B_C2: begin hi16 <= rd_r; st <= A_B_WR; end   // rd_r = hi 组
        A_B_WR: begin
          bias_we <= 1'b1;
          bias_wa <= bwa_full[BAW-1:0];
          bias_wd <= {hi16[8*bwr_i +: 8], lo16[8*bwr_i +: 8]};
          if (bwr_i == 4'd15 || bnext >= {1'b0, k_r}) begin
            bwr_i <= '0;
            if (blast) begin
              jr <= '0; jw <= '0; run_v1 <= 1'b0; run_v2 <= 1'b0;
              st <= A_RUN;
            end else begin
              bg_r <= bg_r + 12'd1;
              st <= A_B_LO;
            end
          end else bwr_i <= bwr_i + 4'd1;
        end
        // ---------------- 原地变换：稳态每拍写一整列（16 行）----------------
        A_RUN: begin
          if (jr < {1'b0, n_r}) begin
            jr <= jr + 17'd1;
            run_v1 <= 1'b1;
          end else run_v1 <= 1'b0;
          run_v2 <= run_v1;
          if (run_v2) begin                   // rd_r/b_r = 列 jw 数据
            ctx_we     <= 1'b1;
            ctx_welane <= lane_mask;
            ctx_waddr  <= grp_base + {4'd0, jw[15:0]};
            ctx_wdata  <= md_bias ? bwbyte : wbyte;
            if (jw + 17'd1 >= {1'b0, n_r}) st <= A_NEXT;
            else jw <= jw + 17'd1;
          end
        end
        A_NEXT: begin
          if ({1'b0, row} + 17'd16 >= {1'b0, m_r}) begin
            st <= A_FIN; busy <= 1'b0;
          end else begin
            row      <= row + 16'd16;
            grp_base <= grp_base + {4'd0, n_r};
            jr <= '0; jw <= '0; run_v1 <= 1'b0; run_v2 <= 1'b0;
            lane_mask <= row_mask(row + 16'd16, m_r);
            st <= A_RUN;
          end
        end
        A_FIN: begin done <= 1'b1; st <= A_IDLE; end
        default: st <= A_IDLE;
      endcase
    end
  end
endmodule
`endif
