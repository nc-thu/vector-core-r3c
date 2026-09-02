// ae_copy.sv — OP_COPY 引擎：B 矩阵 CTX 16-lane -> WRAM COLS-lane 重排
// 全局列 j ∈ [src_j0, src_j0+j_cols)（本 GEMM 组的 B 列片）：
//   源（CTX）：lane = j mod 16，addr = src_base + (j div 16)*spad + k
//   目的（WRAM）：lane = j - src_j0（局部列，0..COLS-1），addr = k —— GEMM 的 B 喂入口径
//   列片跨 2 个源 16-lane 组时 cg 遍历 [src_j0>>4, (src_j0+j_cols-1)>>4]。
// C_RD 地址驱动 -> C_RD2 回数锁存 -> C_WR 一拍写本组覆盖的 lane。
`ifndef AE_COPY_SV
`define AE_COPY_SV
module ae_copy #(
  parameter int COLS = 108
)(
  input  logic clk,
  input  logic rst_n,
  input  logic start,
  output logic busy,
  output logic done,
  input  logic [15:0] k_rows,   // 归约维（WRAM 地址维）
  input  logic [7:0]  j_cols,   // 本组列数（≤ COLS）
  input  logic [19:0] src_base,
  input  logic [15:0] spad,     // 源 16-lane 组步长
  input  logic [15:0] src_j0,   // 全局起始列
  input  logic [11:0] wr_base,  // WRAM 目的基址（GEMM 的 b_base 同值）
  // CTX B 口（读）
  output logic [19:0] ctx_raddr,
  input  logic [16*8-1:0] ctx_rdata,
  // WRAM B 口（写）
  output logic [COLS-1:0] wr_we,
  output logic [11:0] wr_addr,
  output logic [COLS*8-1:0] wr_wdata
);
  typedef enum logic [2:0] {C_IDLE, C_RD, C_RD2, C_WR, C_FIN} st_e;
  st_e st;
  logic [15:0] kk, cg;
  logic [15:0] cg_lo, cg_hi;
  logic [127:0] lat;
  // cg*spad 在源组内不变：提升为寄存基址，URAM 读地址路径只剩加法器
  // （乘法走 LUT 且有整拍预算；1728 DSP 全部留给 MAC 阵列）
  (* use_dsp = "no" *) logic [19:0] rbase_r;
  // 乘积放专用线网统一标记——若直接写在赋值右侧，Vivado retiming 出的
  // 衍生寄存器（rbase_r1 等）不带属性，仍会吃 DSP
  (* use_dsp = "no" *) logic [31:0] j0g_spad, cgn_spad;
  assign j0g_spad = (src_j0 >> 4) * spad;
  assign cgn_spad = (cg + 16'd1) * spad;

  assign busy = (st != C_IDLE);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= C_IDLE; done <= 1'b0; wr_we <= '0;
    end else begin
      done <= 1'b0; wr_we <= '0;
      case (st)
        C_IDLE: if (start) begin
            kk <= '0;
            cg_lo <= src_j0 >> 4;
            cg_hi <= (src_j0 + {8'd0, j_cols} - 16'd1) >> 4;
            cg <= src_j0 >> 4;
            rbase_r <= src_base + j0g_spad;
            st <= C_RD;
          end
        C_RD: st <= C_RD2;       // 地址（cg,kk）已驱动，等 CTX 回数
        C_RD2: begin
            lat <= ctx_rdata;    // 本源组 16 列字节到齐
            st <= C_WR;
          end
        C_WR: begin
          wr_addr <= wr_base + kk[11:0];
          for (int s = 0; s < 16; s++) begin
            int l;
            l = cg*16 + s - src_j0;          // 局部目的 lane（可为负 -> 不写）
            if (l >= 0 && l < int'(j_cols) && l < COLS) begin
              wr_we[l] <= 1'b1;
              wr_wdata[l*8 +: 8] <= lat[s*8 +: 8];
            end
          end
          if (kk + 16'd1 >= k_rows) begin
            kk <= '0;
            if (cg + 16'd1 > cg_hi) st <= C_FIN;
            else begin
              cg <= cg + 16'd1;
              rbase_r <= src_base + cgn_spad;
              st <= C_RD;
            end
          end else begin
            kk <= kk + 16'd1; st <= C_RD;
          end
        end
        C_FIN: begin done <= 1'b1; st <= C_IDLE; end
        default: st <= C_IDLE;
      endcase
    end
  end

  assign ctx_raddr = rbase_r + {{4'd0}, kk};
endmodule
`endif
