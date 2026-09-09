# -*- coding: utf-8 -*-
# 一次性补丁 v2：ae_actv.sv 加 AdaRMS（t 表 + S4 t 乘级）与 ELTWISE（submode=3）。
# 锚点全部逐字拷贝自当前文件。用后即删。
p = 'ae_actv.sv'
s = open(p, encoding='utf-8').read().replace('\r\n', '\n')

def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, f"anchor cnt={s.count(old)} (want {cnt}): {old[:70]!r}"
    s = s.replace(old, new)

# 1) 端口：rq_m2
rep("""  input  logic [7:0]  rq_s,                   // BIAS 右移
""",
"""  input  logic [7:0]  rq_s,                   // BIAS 右移
  input  logic [15:0] rq_m2,                  // ELTWISE 第二乘子（desc[135:120]）
""")

# 2) 枚举加 A_E_LT
rep("""                            A_N_C0, A_N_C1, A_N_C2, A_N_P1, A_N_ST,
                            A_N_P2} st_e;""",
"""                            A_N_C0, A_N_C1, A_N_C2, A_N_P1, A_N_ST,
                            A_N_P2, A_E_LT} st_e;""")

# 3) 新寄存器声明（放 ln_r 后）
rep("""  logic        ln_r;                          // 1=LayerNorm（减均值）
""",
"""  logic        ln_r;                          // 1=LayerNorm（减均值）
  logic        ada_r;                         // word0 bit83：AdaRMS t 区存在
  logic [127:0] t16;                          // 本行组 16 行 t6（lane L=行 r+L）

  // ---- ELTWISE（submode=3）参数与读流 ----
  logic        md_elt;                        // submode=3
  logic signed [15:0] m2_r;                   // 第二乘子（rq_m 锁存为 m1）
  logic [19:0] x2g_r;                         // x2 当前行组基址（随组推进）
  logic        ph_q;                          // 发址相位：0=x1 1=x2
  logic        iv1, iv2;                      // 发址有效流水（2 拍到数据）
  logic        pd1, pd2;                      // rd_r 当前数据的相位
  logic [15:0] jc;                            // 发址列
  logic        isu_done, lastw;               // 发址完 / 末列写已发
  logic [127:0] x1_q;                         // x1 列寄存（等 x2 到合并）
""")

# 4) ELTWISE 数据通路 generate（放 g_norm 块前）
rep("""  // ================= NORM：一遍扫描累加器（每 lane 一行，精确）=================""",
"""  // ---- ELTWISE 数据通路：y = sat8(((x1·m1 + x2·m2) + 2^(s-1))>>>s) ----
  //   合并拍：rd_r = 本列 x2（刚下总线），x1_q = 上一拍存的本列 x1；
  //   m1 走 BIAS 的 rq_m_r、m2 走 rq_m2（ae_core 从 desc[135:120] 接线）
  logic [127:0] ewbyte;
  for (genvar g = 0; g < 16; g++) begin : g_elt
    (* use_dsp = "no" *) logic signed [23:0] ea, eb;  // 8×16 乘（x1·m1 / x2·m2）
    logic signed [24:0] eacc, esh;
    logic [7:0]         esat;
    assign ea   = $signed(x1_q[8*g +: 8]) * $signed(rq_m_r);
    assign eb   = $signed(rd_r [8*g +: 8]) * m2_r;
    assign eacc = ea + eb;
    always_comb begin
      if (rq_s_r == 8'd0) esh = eacc;
      else begin
        logic signed [24:0] ehr;
        ehr = eacc + (25'sd1 <<< (rq_s_r[4:0] - 5'd1));
        esh = ehr >>> rq_s_r[4:0];
      end
      if      (esh > 25'sd127)  esat = 8'd127;
      else if (esh < -25'sd128) esat = -8'sd128;
      else                      esat = esh[7:0];
    end
    assign ewbyte[8*g +: 8] = esat;
  end

  // ================= NORM：一遍扫描累加器（每 lane 一行，精确）=================""")

# 5a) 注释头改 5 级
rep("""  // ============ NORM 二遍：每 lane 4 级流水（吞吐 1 列/拍）================
  // 原单周期组合链（乘→减→桶移→饱和→乘→加→桶移→饱和→写）OOC 时序
  // WNS −4.9ns；切 4 级：S1=x·inv乘+61b减  S2=桶移+sat9  S3=9×16乘+两级加
  // S4=out_shift桶移+sat8+写回。列号随流水 3 级跟踪（p2j1..p2j3），
  // S4 的寄存器即写口 ctx_wdata（共 4 级）。
  // 输出终值与原组合链逐位一致（只是重新切拍），微观位精确门不变。""",
"""  // ============ NORM 二遍：每 lane 5 级流水（吞吐 1 列/拍）================
  // 原单周期组合链 OOC 时序 WNS −4.9ns，切流水：
  //   S1=x·inv乘(8×27)+61b减  S2=桶移+sat9  S3=9×16乘+>>>8
  //   S4=AdaRMS t 乘(17×8)+>>>6+加β  S5=out_shift 桶移+sat8（写口寄存）
  // 列号随流水 4 级跟踪（p2j1..p2j4）。AdaRMS t6=round(t·64) 缺省 64：
  // (64·x+32)>>>6 == x 对一切整数 x 成立，非 ada 路径与 v1.1 逐位相同。""")

# 5b) 流水寄存器声明
rep("""  // 流水有效/列号（st==A_N_P2 内推进）；g/b 表值随流水对齐两级
  logic       p2v1, p2v2, p2v3;
  logic [15:0] p2j1, p2j2, p2j3;
  wire  p2e1 = (st == A_N_P2) && run_v2;      // stage1 装载使能
  wire  p2e2 = (st == A_N_P2) && p2v1;
  wire  p2e3 = (st == A_N_P2) && p2v2;
  logic [15:0] g_q, b_q, g_q2, b_q2;""",
"""  // 流水有效/列号（st==A_N_P2 内推进）；g/b 表值随流水对齐两级
  logic       p2v1, p2v2, p2v3, p2v4;
  logic [15:0] p2j1, p2j2, p2j3, p2j4;
  wire  p2e1 = (st == A_N_P2) && run_v2;      // stage1 装载使能
  wire  p2e2 = (st == A_N_P2) && p2v1;
  wire  p2e3 = (st == A_N_P2) && p2v2;
  wire  p2e4 = (st == A_N_P2) && p2v3;
  logic [15:0] g_q, b_q, g_q2, b_q2, b_q3;""")

# 5c) genvar 内声明
rep("""    (* use_dsp = "no" *) logic signed [24:0] wg;    // w·g_j（S3）
    logic signed [16:0] t17;
    logic signed [17:0] tb18, tb_q;
    logic [7:0]         y8;""",
"""    (* use_dsp = "no" *) logic signed [24:0] wg;    // w·g_j（S3）
    logic signed [16:0] t17, t17_q;
    (* use_dsp = "no" *) logic signed [24:0] tt;    // t17·t6（S4，AdaRMS）
    logic signed [18:0] ta, tb19, tb_q;
    logic [7:0]         t8v, y8;""")

# 5d) 组合链：S3 后拆两级（t 乘 + β 加挪到 S4）；y8 饱和加宽 19b
rep("""    assign wg   = w9_q * $signed(g_q2);
    assign t17  = (wg + 25'sd128) >>> 8;
    assign tb18 = t17 + $signed({{2{b_q2[15]}}, b_q2});
    always_ff @(posedge clk) if (p2e3) tb_q <= tb18;""",
"""    assign wg   = w9_q * $signed(g_q2);
    assign t17  = (wg + 25'sd128) >>> 8;
    always_ff @(posedge clk) if (p2e3) t17_q <= t17;
    // AdaRMS 逐行缩放：t6 = round(t·64)（t8v）；非 ada 恒 64，
    // (64·x+32)>>>6 == x 对一切整数 x 成立 → 非 ada 路径逐位不变
    assign t8v  = ada_r ? t16[8*g +: 8] : 8'd64;
    assign tt   = t17_q * $signed(t8v);
    assign ta   = (tt + 25'sd32) >>> 6;
    assign tb19 = ta + $signed({{3{b_q3[15]}}, b_q3});
    always_ff @(posedge clk) if (p2e4) tb_q <= tb19;""")

rep("""        if      (tb_q > 18'sd127)   y8 = 8'd127;
        else if (tb_q < -18'sd128)  y8 = -8'sd128;
        else                        y8 = tb_q[7:0];
      end else begin
        logic signed [18:0] tr;
        logic signed [18:0] tsh;
        tr  = tb_q + (19'sd1 <<< (osh_r - 4'd1));""",
"""        if      (tb_q > 19'sd127)   y8 = 8'd127;
        else if (tb_q < -19'sd128)  y8 = -8'sd128;
        else                        y8 = tb_q[7:0];
      end else begin
        logic signed [19:0] tr;
        logic signed [19:0] tsh;
        tr  = tb_q + (20'sd1 <<< (osh_r - 4'd1));""")

# 6) 读地址：A_N_ST 的 t 字覆盖（c7 发读 → rd_r 恰在 A_N_P2 首拍有效）+ A_E_LT
rep("""      A_RUN:     ctx_raddr = grp_base + {3'd0, jr};
      A_N_P1:    ctx_raddr = grp_base + {3'd0, jr};
      A_N_P2:    ctx_raddr = grp_base + {3'd0, jr};
      default:   ctx_raddr = grp_base;""",
"""      A_RUN:     ctx_raddr = grp_base + {3'd0, jr};
      A_N_P1:    ctx_raddr = grp_base + {3'd0, jr};
      A_N_P2:    ctx_raddr = grp_base + {3'd0, jr};
      // AdaRMS：t 字（tbl+1+4·NLO+组号）趁统计 c7 拍 A 口空闲读，
      // 2 拍延迟后恰在 A_N_P2 首拍落进 rd_r（零额外拍数；非 ada 地址不动）
      A_N_ST:    ctx_raddr = (ada_r && st_cnt == 4'd7)
                  ? tbase_r + 20'd1 + {nlo_r, 2'b00} + {8'd0, row[15:4]}
                  : grp_base;
      A_E_LT:    ctx_raddr = ph_q ? (x2g_r + {4'd0, jc})
                                  : (grp_base + {4'd0, jc});
      default:   ctx_raddr = grp_base;""")

# 7) 复位区
rep("""      invn_r <= '0; eps_r <= '0; gsh_r <= '0; osh_r <= '0; ln_r <= 1'b0;
      li <= '0; lut_wa <= '0; bg_r <= '0; bwr_i <= '0; ld_reg <= '0;""",
"""      invn_r <= '0; eps_r <= '0; gsh_r <= '0; osh_r <= '0; ln_r <= 1'b0;
      ada_r <= 1'b0; t16 <= '0;
      md_elt <= 1'b0; m2_r <= '0; x2g_r <= '0;
      ph_q <= 1'b0; iv1 <= 1'b0; iv2 <= 1'b0; pd1 <= 1'b0; pd2 <= 1'b0;
      jc <= '0; isu_done <= 1'b0; lastw <= 1'b0; x1_q <= '0;
      li <= '0; lut_wa <= '0; bg_r <= '0; bwr_i <= '0; ld_reg <= '0;""")

rep("""      p2v1 <= 1'b0; p2v2 <= 1'b0; p2v3 <= 1'b0;
      p2j1 <= '0; p2j2 <= '0; p2j3 <= '0;
      g_q <= '0; b_q <= '0; g_q2 <= '0; b_q2 <= '0;""",
"""      p2v1 <= 1'b0; p2v2 <= 1'b0; p2v3 <= 1'b0; p2v4 <= 1'b0;
      p2j1 <= '0; p2j2 <= '0; p2j3 <= '0; p2j4 <= '0;
      g_q <= '0; b_q <= '0; g_q2 <= '0; b_q2 <= '0; b_q3 <= '0;""")

# 8) A_IDLE start 块
rep("""            md_bias <= (submode == 3'd1);
            md_norm <= (submode == 3'd2);""",
"""            md_bias <= (submode == 3'd1);
            md_norm <= (submode == 3'd2);
            md_elt  <= (submode == 3'd3);
            m2_r    <= rq_m2;
            x2g_r   <= tbl_base;              // ELTWISE：x2 基址走 tbl_base
            ph_q <= 1'b0; iv1 <= 1'b0; iv2 <= 1'b0;
            jc <= '0; isu_done <= 1'b0; lastw <= 1'b0;""")

rep("""            case (submode)
              3'd1: st <= A_B_LO;
              3'd2: st <= A_N_C0;
              default: st <= A_LD;
            endcase""",
"""            case (submode)
              3'd1: st <= A_B_LO;
              3'd2: st <= A_N_C0;
              3'd3: st <= A_E_LT;             // ELTWISE 无表装载，直入读流
              default: st <= A_LD;
            endcase""")

# 9) word0 解码加 ada
rep("""          ln_r   <= rd_r[82];""",
"""          ln_r   <= rd_r[82];
          ada_r  <= rd_r[83];                 // bit83：AdaRMS t 区存在""")

# 10) 统计级转 P2 清 p2v4
rep("""                p2v1 <= 1'b0; p2v2 <= 1'b0; p2v3 <= 1'b0;
                st <= A_N_P2;""",
"""                p2v1 <= 1'b0; p2v2 <= 1'b0; p2v3 <= 1'b0; p2v4 <= 1'b0;
                st <= A_N_P2;""")

# 11) A_N_P2 流水推进 + t16 锁存 + 写回 p2v4 + 退出；后插 A_E_LT 状态
rep("""        A_N_P2: begin
          if (jr < {1'b0, n_r}) begin
            jr <= jr + 17'd1;
            run_v1 <= 1'b1;
          end else run_v1 <= 1'b0;
          run_v2 <= run_v1;
          p2v1 <= run_v2;  p2j1 <= jw[15:0];
          p2v2 <= p2v1;    p2j2 <= p2j1;
          p2v3 <= p2v2;    p2j3 <= p2j2;
          if (run_v2) begin
            g_q <= g_r;  b_q <= b_r;          // S1 随 prh_q 同拍对齐
            if (jw + 17'd1 >= {1'b0, n_r}) jw <= jw;
            else jw <= jw + 17'd1;
          end
          if (p2v1) begin g_q2 <= g_q; b_q2 <= b_q; end
          if (p2v3) begin                       // S4 写回：本拍 nwbyte 已是 tb_q 的
            ctx_we     <= 1'b1;                 // 桶移+sat8 结果，S4 的流水寄存器
            ctx_welane <= lane_mask;            // 就是写口的 ctx_wdata（滞后 3 级）
            ctx_waddr  <= grp_base + {4'd0, p2j3};
            ctx_wdata  <= nwbyte;
          end
          if (jr >= {1'b0, n_r} && !run_v1 && !run_v2 &&
              !p2v1 && !p2v2 && !p2v3) st <= A_NEXT;
        end""",
"""        A_N_P2: begin
          if (jr == 17'd0 && !run_v2) t16 <= rd_r;  // 首拍 rd_r = t 字（stats c7 发读）
          if (jr < {1'b0, n_r}) begin
            jr <= jr + 17'd1;
            run_v1 <= 1'b1;
          end else run_v1 <= 1'b0;
          run_v2 <= run_v1;
          p2v1 <= run_v2;  p2j1 <= jw[15:0];
          p2v2 <= p2v1;    p2j2 <= p2j1;
          p2v3 <= p2v2;    p2j3 <= p2j2;
          p2v4 <= p2v3;    p2j4 <= p2j3;
          if (run_v2) begin
            g_q <= g_r;  b_q <= b_r;          // S1 随 prh_q 同拍对齐
            if (jw + 17'd1 >= {1'b0, n_r}) jw <= jw;
            else jw <= jw + 17'd1;
          end
          if (p2v1) begin g_q2 <= g_q; b_q2 <= b_q; end
          if (p2e3) b_q3 <= b_q2;             // b 随 t17_q 再骑一级
          if (p2v4) begin                      // S5 写回：写口寄存即第 5 级
            ctx_we     <= 1'b1;
            ctx_welane <= lane_mask;
            ctx_waddr  <= grp_base + {4'd0, p2j4};
            ctx_wdata  <= nwbyte;
          end
          if (jr >= {1'b0, n_r} && !run_v1 && !run_v2 &&
              !p2v1 && !p2v2 && !p2v3 && !p2v4) st <= A_NEXT;
        end
        // ------- ELTWISE：双输入残差加，每列两拍（x1 存寄存 → x2 合并写）-------
        //   发址奇偶交替：偶拍 grp_base+jc 读 x1、奇拍 x2g_r+jc 读 x2；
        //   2 拍延迟后按 pd2 相位分流：x1 进 x1_q，x2 到达拍合并写回列 jw。
        //   吞吐 1 列/2 拍（CTX 单读口，双输入只能交替读，如实计费）。
        A_E_LT: begin
          if (!isu_done) begin
            iv1 <= 1'b1;  pd1 <= ph_q;  ph_q <= ~ph_q;
            if (ph_q) begin                       // 本拍发 x2 → 本列发址完成
              if ({1'b0, jc} + 17'd1 >= {1'b0, n_r}) isu_done <= 1'b1;
              else jc <= jc + 16'd1;
            end
          end else iv1 <= 1'b0;                   // 多保 1 拍：末列 x2 数据在途
          iv2 <= iv1;  pd2 <= pd1;
          if (iv2 && !pd2) x1_q <= rd_r;          // rd_r = 列 jw 的 x1
          if (iv2 && pd2) begin                   // rd_r = 列 jw 的 x2 → 合并写
            ctx_we     <= 1'b1;
            ctx_welane <= lane_mask;
            ctx_waddr  <= grp_base + {4'd0, jw[15:0]};
            ctx_wdata  <= ewbyte;
            if ({1'b0, jw} + 17'd1 >= {1'b0, n_r}) lastw <= 1'b1;
            else jw <= jw + 17'd1;
          end
          if (lastw && !iv1 && !iv2) st <= A_NEXT;
        end""")

# 12) A_NEXT：eltwise 每组复位 + 分发
rep("""            jr <= '0; jw <= '0; run_v1 <= 1'b0; run_v2 <= 1'b0;
            lane_mask <= row_mask(row + 16'd16, m_r);
            if (md_norm)
              for (int L = 0; L < 16; L++) begin
                s1[L] <= '0; s2[L] <= '0;         // 下一行组的一遍累加器清零
              end
            st <= md_norm ? A_N_P1 : A_RUN;""",
"""            jr <= '0; jw <= '0; run_v1 <= 1'b0; run_v2 <= 1'b0;
            lane_mask <= row_mask(row + 16'd16, m_r);
            x2g_r <= x2g_r + {4'd0, n_r};         // ELTWISE x2 组基址随组推进
            ph_q <= 1'b0; iv1 <= 1'b0; iv2 <= 1'b0;
            jc <= '0; isu_done <= 1'b0; lastw <= 1'b0;
            if (md_norm)
              for (int L = 0; L < 16; L++) begin
                s1[L] <= '0; s2[L] <= '0;         // 下一行组的一遍累加器清零
              end
            st <= md_norm ? A_N_P1 : (md_elt ? A_E_LT : A_RUN);""")

open(p, 'w', encoding='utf-8', newline='\n').write(s)
print("RTL patch ok, lines:", s.count('\n') + 1)
