// ae_softmax.sv — 两遍(读)+整除 softmax，S/P 物化在 CTX 的同一行内
// 布局：S 存于 CTX，lane = i mod 16（token 行 i），地址 = s_base + (i div 16)*NPAD + j。
// 每行 i：P1 求 max -> P2 求 Σexp -> 恢复除法 r = floor(127*2^30/Σe) -> P3 写 P_j = sat127((e*r)>>>30)
// 因果掩码：j <= i 有效；j ∈ [vlen, n) 一律写 0（PV 读全宽，不能留 S 残值）。
// e 由 exp2 LUT 给出（Q12），全程整数运算，与黄金模型位精确一致。
`ifndef AE_SOFTMAX_SV
`define AE_SOFTMAX_SV
module ae_softmax (
  input  logic clk,
  input  logic rst_n,
  input  logic start,
  input  logic [19:0] s_base,     // CTX bank 内字地址
  input  logic [15:0] m_rows,     // 行数
  input  logic [15:0] n_cols,     // 列数（含因果掩码前的全宽）
  input  logic        causal,
  // CTX 读口（广播地址，16 lane 数据全回，选 lane）
  output logic [19:0] ctx_raddr,
  input  logic [16*8-1:0] ctx_rdata,
  // CTX 写口（单 lane）
  output logic        ctx_we,
  output logic [3:0]  ctx_wlane,
  output logic [19:0] ctx_waddr,
  output logic [7:0]  ctx_wdata,
  output logic        busy,
  output logic        done
);
  typedef enum logic [3:0] {ST_IDLE, ST_P1, ST_W1, ST_P2, ST_DIV, ST_P3W, ST_P3R,
                            ST_NEXT, ST_FIN} state_e;
  state_e st;

  logic [15:0] row, j;
  (* use_dsp = "no" *) logic [19:0] row_base;  // (row>>4)*n_cols 走 LUT
  (* use_dsp = "no" *) logic [31:0] rgrp_nc;   // 乘积专用线网（防 retiming 丢属性）
  assign rgrp_nc = ((row + 16'd1) >> 4) * n_cols;
  logic [15:0] vlen;
  logic signed [7:0] mx;
  logic [31:0]  se;
  logic [15:0]  issued, captured;
  logic signed [7:0] rd;        // 选中的 lane 字节

  // 除法器
  logic [37:0] dv_num, dv_rem;
  logic [5:0]  dv_i;
  logic [25:0] quo;

  // exp LUT 与 P 乘法
  logic [7:0]       lut_d;
  logic signed [12:0] lut_e;
  logic signed [12:0] e_val;
  logic signed [38:0] epr;      // e * r
  (* use_dsp = "no" *) logic signed [38:0] epr_c;
  assign epr_c = e_val * quo;
  logic signed [38:0] p_sh;

  ae_exp_lut u_lut (.d(lut_d), .e(lut_e));

  assign ctx_raddr = row_base + j;
  assign lut_d = mx - rd;      // 0..128（>128 在 LUT 内钳位）
  always_comb begin
    e_val = lut_e;
    p_sh  = epr_c >>> 30;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= ST_IDLE; busy <= 1'b0; done <= 1'b0;
      row <= '0; j <= '0; mx <= '0; se <= '0; quo <= '0;
      ctx_we <= 1'b0; issued <= '0; captured <= '0; dv_i <= '0; dv_rem <= '0; dv_num <= '0;
    end else begin
      done <= 1'b0; ctx_we <= 1'b0;
      case (st)
        ST_IDLE: if (start) begin
            row <= '0; busy <= 1'b1; j <= '0; issued <= '0; captured <= '0;
            row_base <= s_base; mx <= -8'sd128; st <= ST_P1;
          end
        // ---- P1: max ----
        ST_P1: begin
          if (j < vlen) begin
            j <= j + 16'd1;
            if (j != 16'd0 && rd > mx) mx <= rd;   // 捕获上一次读数
          end else begin
            if (rd > mx) mx <= rd;                 // 最后一个读数
            st <= ST_W1; j <= '0; issued <= '0; captured <= '0;
          end
        end
        ST_W1: begin // 读口排空一拍
          j <= '0; issued <= '0; captured <= '0; se <= '0; st <= ST_P2;
        end
        // ---- P2: Σexp ----
        ST_P2: begin
          if (j < vlen) begin
            j <= j + 16'd1;
            if (j != 16'd0) se <= se + {{19{1'b0}}, lut_e};
          end else begin
            se <= se + {{19{1'b0}}, lut_e};
            dv_num <= 38'd136365211648;  // 127 * 2^30
            dv_rem <= '0; dv_i <= 6'd37; j <= '0; st <= ST_DIV;
          end
        end
        // ---- 恢复除法: quo = floor(num / se) ----
        ST_DIV: begin
          // 组合判断 {rem, num[dv_i]} >= se；期间 raddr=base+0 预取 S[0]
          if (dv_i != 6'd0) dv_i <= dv_i - 6'd1;
          else              st <= ST_P3W;
        end
        // ---- P3: 写 P（j ∈ [vlen, n) 写 0）----
        // B 口单端口：W(j) 拍写 P[j]（rd=S[j] 已由前一拍 R 或 DIV 预取），
        // R 拍发读地址 base+j+1 供下一写拍用 —— 两拍一列。
        ST_P3W: begin
          ctx_we   <= 1'b1;
          ctx_wlane <= row[3:0];
          ctx_waddr <= row_base + j;
          ctx_wdata <= (j < vlen)
                        ? ((p_sh > 39'sd127) ? 8'd127 : p_sh[7:0])
                        : 8'd0;
          if (j + 16'd1 >= n_cols) st <= ST_NEXT;
          else begin j <= j + 16'd1; st <= ST_P3R; end
        end
        ST_P3R: st <= ST_P3W;   // 读地址 = row_base + j（已 +1），等回数
        ST_NEXT: begin
          if (row == m_rows - 16'd1) begin
            st <= ST_FIN; busy <= 1'b0;
          end else begin
            row <= row + 16'd1;
            row_base <= s_base + rgrp_nc;
            j <= '0; mx <= -8'sd128; st <= ST_P1;
          end
        end
        ST_FIN: begin done <= 1'b1; st <= ST_IDLE; end
        default: st <= ST_IDLE;
      endcase
    end
  end

  // 读数据 lane 选择 + vlen 计算
  assign rd = ctx_rdata[ (row[3:0])*8 +: 8 ];
  always_comb begin
    if (causal) vlen = (n_cols < row + 16'd1) ? n_cols : (row + 16'd1);
    else        vlen = n_cols;
  end

  // 除法器数据通路（在 ST_DIV 拍完成移位-比较-减法）
  logic [38:0] rem_sh;
  logic ge;
  assign rem_sh = {dv_rem, dv_num[dv_i]};
  assign ge = (rem_sh >= {7'd0, se});
  always_ff @(posedge clk) begin
    if (st == ST_DIV) begin
      dv_rem <= ge ? (rem_sh - {7'd0, se}) : rem_sh[37:0];
      quo    <= {quo[24:0], ge};
    end
  end
endmodule
`endif
