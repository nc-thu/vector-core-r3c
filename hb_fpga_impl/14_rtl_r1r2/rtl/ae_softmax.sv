// ae_softmax.sv — SM16：16 行并行走 softmax（CTX 广播读 + 全 16 lane 写）
// 布局与 v1（串行版，备份于 rtl/bak_v1_serial/）完全一致：S/P 物化在 CTX 同一行内，
//   lane = i mod 16（token 行 i），地址 = s_base + (i>>4)*n_cols + j。
// 一次广播读 = 同列 16 行（一个行组 rg*16..rg*16+15，尾组可不满）→ 三段全部并行：
//   P1 每拍取 16 行同列 S 做 max → P2 每拍累加 16 行 Σexp → DIV 38 拍恢复除法
//   （16 路锁步，迭代结构不变）→ P3 每拍写一整列 P[:,j]（16 lane 同地址，B 口
//   全宽写 + A 口预读，SDP 合法）。
// 位精确：每行仍是同一集合的 max、同一整数和（加法序无关）、同一恢复除法序列、
//   同一 e*quo>>>30 与 sat127 —— 与 v1 逐位一致，只有周期数变少。
//
// ★ 时序流水（v2 综合教训）：CTX URAM 级联读出（≈3ns）→ exp LUT → 13×26 乘 →
//   饱和比较 一拍内走不完（首版综合 WNS −4.462ns，Fmax≈118MHz）。故读数据两级
//   寄存：ctx_rdata → rd_r → (mx−rd_r → exp LUT) → e_v_r → (×quo → sat) → ctx_wdata。
//   P1/P2 捕获列 = j−2（两级读延迟），P3 读指针 jr 预读、写指针 jw 滞后 3 拍起流，
//   稳态仍每拍写一整列。每行组周期 ≈ 2*glen + n_cols + 45（v1 为每行 2*vlen + 2*n + 42）。
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
  // CTX 读口（广播地址，16 lane 数据全回）
  output logic [19:0] ctx_raddr,
  input  logic [127:0] ctx_rdata,
  // CTX 写口（16 lane 全宽写：一组 16 行共享同一列地址）
  output logic        ctx_we,
  output logic [19:0] ctx_waddr,
  output logic [127:0] ctx_wdata,
  output logic        busy,
  output logic        done
);
  typedef enum logic [2:0] {ST_IDLE, ST_P1, ST_P2, ST_DIV, ST_P3, ST_NEXT,
                            ST_FIN} state_e;
  state_e st;

  logic [15:0] row;          // 行组基行号（= rg*16；lane L 的绝对行 = row + L）
  logic [16:0] j;            // 列指针（P1/P2 = 发地址口径，跑到 glen+1 排空读流水）
  logic [19:0] grp_base;     // = s_base + rg*n_cols（模 2^20 累加，同 v1 截断口径）
  logic [16:0] glen;         // 本组扫描列数（P1/P2 用）
  logic [15:0][16:0] vlen;   // 每行有效列数（j < vlen 参与；不存在行 = 0）
  logic [15:0][7:0]  mx;     // 每行 max（int8 原码存放）
  logic [15:0][31:0] se;     // 每行 Σexp（Q12）

  // 读数据两级流水寄存器（时序关键，见文件头）
  logic [127:0] rd_r;            // ctx_rdata 打一拍
  logic [15:0][12:0] e_v_r;      // exp LUT 输出打一拍（P2 累加 / P3 乘法用）

  // P3 读/写双指针：jr = 下一个发读的列（每拍 +1 预读），jw = 下一个写的列。
  // 列 c 的写在发读后第 3 拍（c=0 由 DIV 期间预取提前就绪），稳态每拍写一列。
  logic [16:0] jr;
  logic [15:0] jw;
  logic [1:0]  warm;             // P3 起流气泡计数（首写后 2 拍不写）

  // 除法器（16 路锁步；num = 127*2^30 共享常量）
  localparam logic [37:0] DV_NUM = 38'd136365211648;
  logic [5:0]  dv_i;
  logic [15:0][37:0] dv_rem;
  logic [15:0][25:0] quo;

  // 读数 lane 视图 + exp LUT（每行一份，16 份同表）；LUT 入口用 rd_r（已寄存）
  logic [15:0][7:0] rd;
  logic [15:0][12:0] e_v;
  assign rd   = rd_r;

  // 行组几何（进组时锁存）：vlen[L] = 行存在 ? (causal ? min(n,row+L+1) : n) : 0
  logic [16:0] row_nx, rowa, v1t, glen_nx;
  logic [15:0][16:0] vlen_nx;
  assign row_nx = (st == ST_IDLE) ? 17'd0 : {1'b0, row} + 17'd16;
  always_comb begin
    glen_nx = {1'b0, n_cols};
    for (int L = 0; L < 16; L++) begin
      rowa = row_nx + L;
      v1t  = rowa + 17'd1;
      if (rowa >= {1'b0, m_rows})
        vlen_nx[L] = 17'd0;
      else if (causal)
        vlen_nx[L] = (v1t >= {1'b0, n_cols}) ? {1'b0, n_cols} : v1t;
      else
        vlen_nx[L] = {1'b0, n_cols};
    end
    if (causal) begin
      // 组内最长 vlen = min(n, min(m, rg*16+16)) —— 截到尾组实际行数
      rowa    = {1'b0, m_rows};
      v1t     = row_nx + 17'd16;
      glen_nx = (v1t >= {1'b0, m_rows}) ? ((rowa >= {1'b0, n_cols}) ? {1'b0, n_cols} : rowa)
                                        : ((v1t >= {1'b0, n_cols}) ? {1'b0, n_cols} : v1t);
    end
  end

  // 除法组合级（temp 平面向量绕开 iverilog 不支持的可变索引二次切片）
  logic [15:0][38:0] rem_sh;
  logic [15:0]       ge;
  logic [15:0][37:0] dv_rn;    // 每路 rem 新值
  logic [15:0][25:0] dv_qn;    // 每路 quo 新值
  always_comb begin
    logic [37:0] rm, sm_;
    logic [38:0] rs;
    logic [25:0] qs;
    for (int L = 0; L < 16; L++) begin
      rm = dv_rem[L];
      rs = {rm, DV_NUM[dv_i]};
      sm_ = se[L];
      rem_sh[L] = rs;
      ge[L]     = (rs >= {7'd0, sm_});
      dv_rn[L]  = ge[L] ? (rs - {7'd0, sm_}) : rs[37:0];
      qs = quo[L];
      dv_qn[L]  = {qs[24:0], ge[L]};
    end
  end

  // P3 乘积与饱和（每行一份 13x26 LUT 乘；具名线网强制纯 LUT，与 requant 同法）
  //   乘法入口 = e_v_r（已寄存），列号用写指针 jw
  logic [127:0] wbyte;
  for (genvar g = 0; g < 16; g++) begin : g_lane
    logic [7:0] dmx;
    (* use_dsp = "no" *) logic signed [38:0] epr;
    logic signed [38:0] p_sh;
    assign dmx   = mx[g] - rd[g];            // 数值同 v1 的 lut_d（无效列钳位在表内）
    assign epr   = e_v_r[g] * quo[g];        // e_v>0、quo≥0：正×正，39b 内
    assign p_sh  = epr >>> 30;
    assign wbyte[8*g +: 8] =
        ({1'b0, jw} < vlen[g]) ? ((p_sh > 39'sd127) ? 8'd127 : p_sh[7:0]) : 8'd0;
    ae_exp_lut u_elut (.d(dmx), .e(e_v[g]));
  end

  // 读地址：P3 用预读指针 jr（每拍 +1），其余状态用 j（DIV 期间 j=0 → 预取 S[:,0]）
  //   （17b 指针零扩到 20b，与 grp_base 模 2^20 相加 —— 切勿用越界位切片 jr[19:0]）
  logic [19:0] rcol;
  assign rcol = (st == ST_P3) ? {3'd0, jr} : {3'd0, j};
  assign ctx_raddr = grp_base + rcol;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= ST_IDLE; busy <= 1'b0; done <= 1'b0;
      row <= '0; j <= '0; grp_base <= '0; glen <= '0;
      vlen <= '0; mx <= '0; se <= '0; quo <= '0; dv_rem <= '0; dv_i <= '0;
      rd_r <= '0; e_v_r <= '0; jr <= '0; jw <= '0; warm <= '0;
      ctx_we <= 1'b0; ctx_waddr <= '0; ctx_wdata <= '0;
    end else begin
      done <= 1'b0; ctx_we <= 1'b0;
      rd_r <= ctx_rdata;                      // 读数据两级流水（无条件推进）
      e_v_r <= e_v;
      case (st)
        ST_IDLE: if (start) begin
            busy <= 1'b1; row <= '0; j <= '0;
            grp_base <= s_base; glen <= glen_nx; vlen <= vlen_nx;
            mx <= '0; se <= '0;                     // mx 初值 -128 装载见下
            for (int L = 0; L < 16; L++) mx[L] <= -8'sd128;
            dv_rem <= '0; quo <= '0;                // 除法器状态必须归零（余数不清会污染下一组）
            st <= ST_P1;
          end
        // ---- P1: 16 行同列 max（rd_r = 上上拍所发列；捕获列 = j-2）----
        //   发址 j = 0..glen，排空到 j = glen+1 拍捕获末列后转 P2，共 glen+2 拍。
        ST_P1: begin
          if ({1'b0, j} >= glen + 17'd1) begin
            // 此拍 rd_r = 列 glen-1（最后一个有效读），捕获后转 P2
            for (int L = 0; L < 16; L++)
              if ({1'b0, j} - 17'd2 < vlen[L] && $signed(rd[L]) > $signed(mx[L]))
                mx[L] <= rd[L];
            j <= '0; se <= '0; st <= ST_P2;
          end else begin
            j <= j + 17'd1;
            if (j >= 17'd2)
              for (int L = 0; L < 16; L++)
                if ({1'b0, j} - 17'd2 < vlen[L] && $signed(rd[L]) > $signed(mx[L]))
                  mx[L] <= rd[L];
          end
        end
        // ---- P2: 16 行同列 Σexp（e_v_r 比发址晚 3 拍：rd_r + exp 寄存；捕获列 = j-3）----
        ST_P2: begin
          if ({1'b0, j} >= glen + 17'd2) begin
            for (int L = 0; L < 16; L++)
              if ({1'b0, j} - 17'd3 < vlen[L])
                se[L] <= se[L] + {{19{1'b0}}, e_v_r[L]};
            dv_i <= 6'd37; j <= '0; jr <= 17'd1; jw <= '0; warm <= 2'd0;
            st <= ST_DIV;                      // raddr=base+0 预取 S[:,0]
          end else begin
            j <= j + 17'd1;
            if (j >= 17'd3)
              for (int L = 0; L < 16; L++)
                if ({1'b0, j} - 17'd3 < vlen[L])
                  se[L] <= se[L] + {{19{1'b0}}, e_v_r[L]};
          end
        end
        // ---- 恢复除法（与 v1 同构，16 路锁步 38 拍）----
        ST_DIV: begin
          for (int L = 0; L < 16; L++) begin
            dv_rem[L] <= dv_rn[L];
            quo[L]    <= dv_qn[L];
          end
          if (dv_i != 6'd0) dv_i <= dv_i - 6'd1;
          else              st <= ST_P3;
        end
        // ---- P3: 稳态每拍写一整列（jr 预读，jw 滞后 3 拍起流，首 2 拍气泡）----
        ST_P3: begin
          jr <= jr + 17'd1;
          if (warm == 2'd3 || warm == 2'd0) begin
            ctx_we    <= 1'b1;
            ctx_waddr <= grp_base + {4'd0, jw};
            ctx_wdata <= wbyte;
            if ({1'b0, jw} + 17'd1 >= {1'b0, n_cols}) st <= ST_NEXT;
            else begin
              jw <= jw + 16'd1;
              if (warm == 2'd0) warm <= 2'd1;
            end
          end else warm <= warm + 2'd1;
        end
        ST_NEXT: begin
          if (row_nx >= {1'b0, m_rows}) begin
            st <= ST_FIN; busy <= 1'b0;
          end else begin
            row      <= row_nx[15:0];
            grp_base <= grp_base + {4'd0, n_cols};
            j <= '0; glen <= glen_nx; vlen <= vlen_nx; se <= '0;
            dv_rem <= '0; quo <= '0;                // 同上：每组除法从 0 余数起步
            for (int L = 0; L < 16; L++) mx[L] <= -8'sd128;
            st <= ST_P1;
          end
        end
        ST_FIN: begin done <= 1'b1; st <= ST_IDLE; end
        default: st <= ST_IDLE;
      endcase
    end
  end
endmodule
`endif
