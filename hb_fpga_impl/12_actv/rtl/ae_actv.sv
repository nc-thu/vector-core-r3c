// ae_actv.sv — AE_ACTV 片上算子引擎（ACTV 直查表 + BIAS + NORM 三模式）
// ---------------------------------------------------------------------------
// 照 SM16 的接入方式（07_onchip_ops/NOTES §二 结构模板）：
//   CTX A 口广播读（一拍 16 行同列字节）→ 数据通路 → B 口 16-lane 写；
//   调度器串行 one-hot（状态 T_RUN_A），不碰 WRAM 写交叉（WNS −1.038 最差路径）。
//   读数据寄存后再进组合级（softmax v2 教训：URAM 级联读出一拍走不完）。
// 描述符编码（256b 布局不动，op=6；golden gen_vectors.py 同位切片）：
//   b_src[2:0] = 子模式：0=ACTV（LUT 直查） 1=BIAS（y'=sat8((y·m+b_j)>>>s)）
//                          2=NORM（LayerNorm/RMSNorm，见 12_actv/spec/norm_spec.json）
//   m=行数  n=列数（=行 stride）  k=表项数（NORM 时 k=n）
//   a_base = 张量 CTX 基址（原地变换）  b_base = 表映像 CTX 基址
//   rq_m/rq_s = BIAS 的 m（Q8.8 有符号）与右移 s（NORM 不用）
// 表映像布局（DMA TAG_CTX 装进 CTX 后由引擎自取）：
//   ACTV：表项 x 在字 b_base+x 的全部 16 个 lane 槽位各放一份（映像 256 字）。
//         为什么复制：CTX 广播读按槽位对号——字 W 的槽 L 字节只送 lane L，
//         想让 16 个 lane 各自装满 256 项的表，只能每字 16 槽同值。装载 260 拍。
//   BIAS：项 j 拆 lo/hi：lo 在 lane j%16 @ b_base+j/16，hi @ b_base+NLO+j/16
//         （NLO=ceil(k/16)）。装载逐组 20 拍（发 lo 读→发 hi 读→捕两次→
//         16 拍写共享 BRAM）。
//   NORM：字 b_base+0 常数区（invn24/eps_q24 48/g_shift 3/out_shift 4/flags 1），
//         随后 g 表 lo/hi、b 表 lo/hi 四区各 NLO=ceil(n/16) 字（同 BIAS 装载）。
//         串行装载（A 口被一遍扫描占用，B 口只写，无法重叠），如实计费。
// NORM 数值管线（语义唯一来源 spec/norm_gold.py，逐位一致）：
//   一遍扫描每 lane 精确累加 S1=Σx(24b)/S2=Σx²(30b)；
//   统计级 μ=S1·invn(32b)、ms=S2·invn(48b)、var=max(0,ms−(μ²>>>24))（RMS 跳过）、
//   v=max(var+eps,2^13)；共享 rsqrt 单元逐 lane 串行：LOD 偶化 → 512×15b LUT
//   →一次牛顿 → invstat_q20=sat27(r1<<<(18−f))；
//   二遍重读：u=(x<<<24)−μ，prod=x·(inv<<<24)−q（q=μ·inv 预计算），
//   w=sat9((prod+2^(S-1))>>>S)（S=44−g_shift，round-half-up），
//   y=sat8(((w·g_j+128)>>>8 + b_j ±半)>>>out_shift)。
// 拍数（见 spec fsm_cycles）：
//   ACTV 260+ceil16(m)·(n+3)   BIAS ceil16(k)·20+ceil16(m)·(n+4)
//   NORM 3+2·ceil16(n)·20+ceil16(m)·(2n+185)
//   ELTWISE 2+ceil16(m)·(2n+6)
//     （v1.2.2 统计级 8→11 拍/lane：拆 var/LOD/查表三拍修 WNS，+32 拍/组；
//      v1.2.3 写口数据落一拍修 WNS：BIAS/ELT/NORM 各 +1 拍排空，ACTV 不变）
// ---------------------------------------------------------------------------
`ifndef AE_ACTV_SV
`define AE_ACTV_SV
module ae_actv #(
  parameter int BIAS_D = 4096                 // bias 表深度（部署档最长 3072）
)(
  input  logic clk,
  input  logic rst_n,
  input  logic start,
  input  logic [2:0]  submode,                // 0=ACTV 1=BIAS 2=NORM
  input  logic [19:0] y_base,                 // 张量 CTX 基址（原地）
  input  logic [15:0] m_rows,                 // 有效行数（尾组不满按 lane 掩码跳写）
  input  logic [15:0] n_cols,                 // 列数 = 行 stride
  input  logic [19:0] tbl_base,               // 表映像 CTX 基址
  input  logic [15:0] tbl_len,                // BIAS 表项数（NORM 时 =n）
  input  logic [15:0] rq_m,                   // BIAS 乘子（Q8.8 有符号）
  input  logic [7:0]  rq_s,                   // BIAS 右移
  input  logic [15:0] rq_m2,                  // ELTWISE 第二乘子（desc[135:120]）
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

  typedef enum logic [4:0] {A_IDLE, A_LD, A_DRAIN, A_B_LO, A_B_HI,
                            A_B_C1, A_B_C2, A_B_WR, A_RUN, A_NEXT, A_FIN,
                            A_N_C0, A_N_C1, A_N_C2, A_N_P1, A_N_ST,
                            A_N_P2, A_E_LT} st_e;
  st_e st;

  // ---- 参数锁存 ----
  logic        md_bias, md_norm;
  logic [19:0] tbase_r, ybase_r, grp_base;
  logic [15:0] m_r, n_r, k_r, rq_m_r;
  logic [7:0]  rq_s_r;
  logic [11:0] nlo_r;                         // lo/hi 区各占字数 ceil(k/16)

  // ---- NORM 常数（表映像 word0）----
  logic [23:0] invn_r;                        // floor(2^24/n)
  logic [47:0] eps_r;                         // eps_q24
  logic [2:0]  gsh_r;                         // g_shift 0..7
  logic [3:0]  osh_r;                         // out_shift 0..15
  logic        ln_r;                          // 1=LayerNorm（减均值）
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

  // ---- 装载计数 ----
  logic [8:0]  li;                            // ACTV：发读字指针 0..256
  logic [11:0] bg_r;                          // BIAS/NORM：当前组号
  logic [3:0]  bwr_i;                         // 组内写序号
  logic [1:0]  ld_reg;                        // NORM 装载区：1=g 表 2=b 表（0=BIAS）
  logic [127:0] lo16, hi16;                   // 本组 lo/hi 捕获（lane L = 项 j&15）

  // ---- RUN 指针与流水 ----
  logic [16:0] jr, jw;                        // 发读列 / 写列（滞后 3 拍）
  logic        run_v1, run_v2;                // 读数据 2 拍有效流水
  logic        r_last;                        // A_RUN 末列已发（v1.2.3 排空用）
  logic [15:0] row;                           // 行组基行号
  logic [15:0] lane_mask;                     // 本组有效行掩码

  // ---- v1.2.3 写口流水寄存（时序修复，见 g_bias 注释与 FSM 各写口）----
  logic        bwv;                           // BIAS：写请求（数据晚 1 拍）
  logic [19:0] bwa;                           //        写地址
  logic        ewv;                           // ELTWISE：写请求（晚 1 拍）
  logic [19:0] ewa;
  logic        p2v5;                          // NORM：S5 结果写口（第 6 级）
  logic [15:0] p2j5;                          //        写列号
  logic [127:0] p2y_q;                        //        整列输出字节

  // ---- 读数据寄存（rd_r 与 b_r/g_r 同相位）----
  logic [127:0] rd_r;
  logic signed [15:0] b_r;
  logic signed [15:0] g_r;

  // ---- bias 共享 BRAM（BIAS 的 b_j + NORM 的 b_j）----
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

  // ---- g 表共享 BRAM（NORM 的 g_j；BIAS/ACTV 不用）----
  logic         g_we;
  logic [BAW-1:0] g_wa;
  logic [15:0]  g_wd;
  logic [BAW-1:0] g_ra;
  logic [15:0]  g_rd;
  (* ram_style = "block" *) logic [15:0] g_mem [0:BIAS_D-1];
  always_ff @(posedge clk) begin
    if (g_we) g_mem[g_wa] <= g_wd;
    g_rd <= g_mem[g_ra];
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

  // ---- 共享数据通路（v1.2 面积收敛）：BIAS / ELTWISE / NORM-pass1 三模式复用
  //      同一组 16 份 8×16 乘法 + 25b 加 + 桶移 + sat8（三模式串行运行，绝不并发）----
  //   BIAS：    y = sat8((x·rq_m + b_j) >>> rq_s)
  //   ELTWISE：y = sat8(((x1·m1 + x2·m2) + 2^(s-1)) >>> s)。x1 拍乘 m1 并把
  //             +2^(s-1) 折进保持项 ea_q，x2 拍乘 m2 后走同一条 加>>>sat 出口
  //             ——(ea+half)+eb ≡ ea+eb+half，整数精确，逐位不变；
  //             m1 就是 BIAS 的 rq_m_r、m2 走 rq_m2（desc[135:120] 接线）。
  //   NORM 一遍：同阵列算 x²（B 口选 rd_r 字节自乘；此拍 BIAS 阵列本来空闲），
  //             乘积从 pmul 出口给 s2 累加，省掉独立的 16 份 8×8。
  logic [127:0] bwbyte, ewbyte;
  wire  elt_ph2 = md_elt && pd2;                    // ELT 合并拍（rd_r = x2）
  wire  p1_sq   = md_norm && (st == A_N_P1);        // NORM pass1：算 x²
  logic signed [23:0] pmul [0:15];                  // 共享乘积出口（pass1 取用）
  for (genvar g = 0; g < 16; g++) begin : g_bias
    (* use_dsp = "no" *) logic signed [23:0] prod;  // 具名线网强制纯 LUT
    logic signed [24:0] accb, p_sh;
    logic signed [24:0] accb_q;                     // v1.2.3：加法输出落一拍
    logic signed [24:0] ea_q;                       // ELT x1 拍乘积（已折入 half）
    logic [7:0]         sat;
    // 注意：$signed(变基部位选) 嵌在三元里 iverilog 会按无符号零扩展（LRM 应
    // 符号扩展）——先落 signed 线网再显式 {{}} 扩展，两个工具行为一致（坑 #13）
    wire signed [7:0]  xb = rd_r[8*g +: 8];
    wire signed [16:0] mb = elt_ph2 ? m2_r :
                            p1_sq   ? {{9{xb[7]}}, xb} : rq_m_r;
    assign prod  = $signed(rd_r[8*g +: 8]) * mb;
    assign pmul[g] = prod;
    assign accb  = prod + (md_elt ? ea_q : $signed({{9{b_r[15]}}, b_r}));
    // v1.2.3 时序修复：原组合链 md_norm→mb→乘法→accb→桶移→sat→ctx_wdata
    // 24 级 5.36ns（全片最差 −1.359）。切成两段：乘+加 → accb_q（本地寄存
    // 器），桶移+sat 改吃 accb_q——数值逐位不变，写口数据整体晚一拍，
    // BIAS/ELT 各 +1 拍排空（见 FSM 写口 ewv/bwv）。
    always_ff @(posedge clk) accb_q <= accb;
    assign p_sh  = accb_q >>> rq_s_r;               // ELT 的 s ≤ 15，同一条桶移
    always_comb begin
      if      (p_sh > 25'sd127)  sat = 8'd127;
      else if (p_sh < -25'sd128) sat = -8'sd128;
      else                       sat = p_sh[7:0];
    end
    assign bwbyte[8*g +: 8] = sat;
    assign ewbyte[8*g +: 8] = sat;                  // 同一出口，两个名字（写口各自取用）
    // ELT x1 拍：rd_r = 本列 x1，乘 m1 并折入舍入半拍（s=0 时 half=0，>>>0 不变）
    always_ff @(posedge clk) if (md_elt && st == A_E_LT && iv2 && !pd2)
      ea_q <= prod + ((rq_s_r == 8'd0) ? 25'sd0
                      : (25'sd1 <<< (rq_s_r[4:0] - 5'd1)));
  end

  // ================= NORM：一遍扫描累加器（每 lane 一行，精确）=================
  logic signed [23:0] s1 [0:15];
  logic signed [29:0] s2 [0:15];

  // ---- 统计级寄存器堆（每 lane：μ 32b + invstat 27b + q=μ·inv 59b）----
  logic signed [31:0] mu_rf  [0:15];
  logic [26:0]        inv_rf [0:15];
  // qn_rf = μ·inv − 2^(S-1)（把二遍的 +2^(S-1) 折进统计期预计算：
  // prh = xm − qn，省一级 61b 加法，时序友好；值等价 xm − q + half，整数精确）
  logic signed [61:0] qn_rf  [0:15];

  // ---- 共享统计单元（逐 lane 串行，st_cnt 0..7 微步）----
  logic [3:0]  st_lane;                       // 当前 lane 0..15
  logic [3:0]  st_cnt;
  logic signed [31:0] mu_c;                   // S1·invn
  logic [47:0]        ms_c;                   // S2·invn
  logic signed [38:0] sq_c;                   // μ²>>>24
  logic [12:0]        mq_c;                   // m_q11 ∈ [2048,8192)
  logic [4:0]         f_c;                    // E/2
  logic [14:0]        r0_c;
  logic signed [29:0] r0sq_c;
  logic signed [42:0] mrs_c;                  // m_q11·r0²
  logic signed [57:0] nw_c;                   // r0·(3·2^39 − m_q11·r0²)

  // rsqrt LUT：512×15b 分布式 ROM（有效 384 项；rsqrt_lut.mem 由
  // spec/norm_gold.py --dump-lut 生成，与 Python 黄金同一张表）
  (* rom_style = "distributed" *) logic [14:0] rtab [0:511];
  initial $readmemh("rsqrt_lut.mem", rtab);

  // μ²>>>24 的精确分解（避免 32×32 大乘）：
  //   μ = A_hi·2^12 + A_lo；μ² = A_hi²·2^24 + 2·A_hi·A_lo·2^12 + A_lo²
  //   μ²>>>24 = A_hi² + ((2·A_hi·A_lo·2^12 + A_lo²)>>>24)   （逐位等价）
  //   注意 A_lo 是低 12 位的无符号值（0..4095）——声明成 signed 会让 ≥2048 的
  //   低位变负、恒等式破坏；且 Verilog 有符号×无符号混乘整体退化无符号，
  //   A_lo 必须显式零扩展成有符号再乘。
  wire signed [19:0] mu_hi = mu_c[31:12];
  wire        [11:0] mu_lo = mu_c[11:0];
  (* use_dsp = "no" *) logic signed [39:0] sq_hi;   // A_hi²
  (* use_dsp = "no" *) logic signed [31:0] sqm;     // A_hi·A_lo（v1.2.2：先乘再移）
  (* use_dsp = "no" *) logic signed [25:0] sq_lo;   // A_lo²（13×13 要 26b）
  assign sq_hi = mu_hi * mu_hi;
  assign sq_lo = $signed({1'b0, mu_lo}) * $signed({1'b0, mu_lo});
  // v1.2.2 DSP 修复：原来写 sq_mid = (mu_hi*mu_lo) <<< 1 再 <<<12——乘积带
  // 后移位会让 use_dsp=no 失配、整个乘法被提成 DSP48（首综合 DSP=1 的根因）。
  // 拆成独立乘积网线 sqm，把 <<<13 并进 mid_sh，数值逐位不变。
  assign sqm = mu_hi * $signed({1'b0, mu_lo});
  // mid_sh：2·A_hi·A_lo·2^12 + A_lo²。必须用有符号表达式加（赋值 46b 上下文
  // 会先把两边符号扩展到 46b 再加）——若用 {拼接} 做，两个 45b 无符号加法会把
  // 负结果的符号进位丢在 bit45 之外（差恰好 2^45，肉眼极难查）。
  wire signed [45:0] mid_sh = (sqm <<< 13) + sq_lo;
  wire signed [38:0] mu2_p24 = sq_hi + $signed(mid_sh[45:24]);

  // 统计乘法器（v1.2.2：S1·invn 与 S2·invn 共享一个 32b×24b 乘法器，c0/c1
  //   两拍换操作数——s2 是非负数，零扩展成有符号乘值不变，逐位等价；
  //   两个独立乘法器省约 0.2k LUT）
  wire signed [31:0] msA = (st_cnt == 4'd0)
                         ? {{8{s1[st_lane][23]}}, s1[st_lane]}   // c0 拍：S1
                         : {2'b00, s2[st_lane]};                 // c1 拍：S2
  (* use_dsp = "no" *) logic signed [55:0] m_s;    // A·invn ≤ 2^31·2^24 = 2^55
  assign m_s = msA * $signed({8'd0, invn_r});
  wire signed [47:0] m_s1 = m_s[47:0];             // 兼容名（tb 探针引用）
  wire        [53:0] m_s2 = m_s[47:0];

  // rsqrt 牛顿乘法器（共享）
  (* use_dsp = "no" *) logic signed [29:0] r0r0;
  (* use_dsp = "no" *) logic signed [42:0] mmul;
  (* use_dsp = "no" *) logic signed [57:0] nmul;
  assign r0r0 = $signed({1'b0, r0_c}) * $signed({1'b0, r0_c});
  assign mmul = $signed({3'd0, mq_c}) * r0sq_c;
  assign nmul = $signed({1'b0, r0_c}) *
                ($signed(43'sd1649267441664) - mrs_c);   // 3·2^39 = 1649267441664

  // ---- LOD：49b 优先编码（E = MSB 偶化；v ≥ 2^13 保证 E ≥ 12）----
  function automatic logic [5:0] even_msb49(logic [48:0] v);
    logic [5:0] e;
    e = 6'd0;
    for (int b = 48; b >= 0; b--)
      if (e == 6'd0 && v[b]) e = b[5:0];
    if (e[0]) e = e - 6'd1;
    if (e < 6'd12) e = 6'd12;
    return e;
  endfunction

  // ---- 统计前段（v1.2.2 拆成三拍修 WNS：减/加/钳 → LOD+桶移 → 查表）----
  //   原 c3 一拍干完 var→+eps→max→49b LOD→桶移→查表，35 级逻辑 7.7ns，是
  //   首综合 WNS −3.733 的最差路径。拆拍后各段 ≤ 14 级，数值逐位不变。
  wire signed [48:0] var_w = ln_r ? ((ms_c >= {10'd0, sq_c})
                                       ? ({1'b0, ms_c} - {10'd0, sq_c})
                                       : 49'sd0)
                                  : {1'b0, ms_c};
  wire signed [48:0] vraw_w = var_w + $signed({1'b0, eps_r});
  wire [48:0] v_eff_w = (vraw_w < 49'sd8192) ? 49'sd8192 : vraw_w;
  logic [48:0] ve_c;                            // c3 拍存 v_eff，c4 拍做 LOD
  wire [5:0]  ve_w  = even_msb49(ve_c);
  wire [12:0] mq_w  = ve_c >>> (ve_w - 6'd11);
  wire [8:0]  rt_ix = mq_w[12:4] - 9'd128;    // (m_q11−2048)>>>4

  // ---- invstat 收尾：r1 = sat15(nw>>>40)；inv = sat27(r1<<<(18−f)) ----
  function automatic logic [14:0] sat15(logic signed [57:0] x);
    logic signed [57:0] r;
    r = (x > 57'sd16383) ? 57'sd16383 :
        (x < -57'sd16384) ? -57'sd16384 : x;
    sat15 = r[14:0];
  endfunction

  // 统计级 c7 收尾组合线：nw_c/f_c → inv27（桶移+sat+max1）→ qm=μ·inv
  //   （乘法独立成线网才能挂 use_dsp=no；c7 存 inv、c8 存 qn，各一拍）
  logic [26:0] inv27_w;
  always_comb begin
    logic signed [57:0] r1s, sh;
    r1s = nw_c >>> 40;
    if      (r1s > 57'sd16383)  r1s = 57'sd16383;
    else if (r1s < -57'sd16384) r1s = -57'sd16384;
    sh = (f_c <= 5'd18) ? (r1s <<< (5'd18 - f_c)) : (r1s >>> (f_c - 5'd18));
    if      (sh > 57'sd134217727) inv27_w = 27'd134217727;
    else if (sh < 57'sd0)         inv27_w = 27'd0;
    else                          inv27_w = sh[26:0];
    if (inv27_w == 27'd0) inv27_w = 27'd1;
  end
  (* use_dsp = "no" *) logic signed [58:0] qm_w;   // μ·inv 需 59b（32×27）
  logic [26:0] invq_c;                          // c9 拍锁存 inv27_w（v1.2.2：
                                                // qn 乘法独立成拍，吃寄存后的 inv）
  assign qm_w = mu_rf[st_lane] * $signed({5'd0, invq_c});

  // 一遍累加用的 x² 不再独立做 8×8：p1_sq 拍共享乘法器阵列 B 口选 rd_r 字节
  // 自乘（见 g_bias 的 mb），乘积走 pmul 出口——值与 x·x 完全一致

  // ============ NORM 二遍：每 lane 5 级流水（吞吐 1 列/拍）================
  // 原单周期组合链 OOC 时序 WNS −4.9ns，切流水：
  //   S1=x·inv乘(8×27)+61b减  S2=桶移+sat9  S3=9×16乘+>>>8
  //   S4=AdaRMS t 乘(17×8)+>>>6+加β  S5=out_shift 桶移+sat8（写口寄存）
  // 列号随流水 4 级跟踪（p2j1..p2j4）。AdaRMS t6=round(t·64) 缺省 64：
  // (64·x+32)>>>6 == x 对一切整数 x 成立，非 ada 路径与 v1.1 逐位相同。
  logic signed [61:0] qn_rf_lane [0:15];      // 综合期拷贝（避免在 generate
  for (genvar g = 0; g < 16; g++) begin : g_qcp   // 里引用变址寄存器堆）
    assign qn_rf_lane[g] = qn_rf[g];
  end
  logic [127:0] nwbyte;
  logic [5:0]  nS;                            // S = 44 − g_shift ∈ [37,44]
  assign nS = 6'd44 - {3'd0, gsh_r};
  logic signed [60:0] nhalf;
  assign nhalf = 61'sd1 <<< (nS - 6'd1);
  // 流水有效/列号（st==A_N_P2 内推进）；g/b 表值随流水对齐两级
  logic       p2v1, p2v2, p2v3, p2v4;
  logic [15:0] p2j1, p2j2, p2j3, p2j4;
  wire  p2e1 = (st == A_N_P2) && run_v2;      // stage1 装载使能
  wire  p2e2 = (st == A_N_P2) && p2v1;
  wire  p2e3 = (st == A_N_P2) && p2v2;
  wire  p2e4 = (st == A_N_P2) && p2v3;
  logic [15:0] g_q, b_q, g_q2, b_q2, b_q3;
  for (genvar g = 0; g < 16; g++) begin : g_norm
    (* use_dsp = "no" *) logic signed [34:0] xinv;  // x·inv（8×27，35b）
    // 宽路径收窄（v1.2 面积收敛，逐位等价）：原 61b 的 prh=(xinv<<<24)−qn 桶移
    //   >>>S（S=44−gsh ∈ [37,44]）里，低 24 位只贡献一个借位——
    //   令 D = xinv − qn[61:24] − (qn[23:0]≠0)，则 prh = D·2^24 + R（0≤R<2^24），
    //   对 S≥24 有恒等式 floor(prh/2^S) = D >>>(S−24)（D 负时按补码余数同样成立）。
    //   于是 S1 只需 38b 减法，S2 桶移 = 固定 >>>13（纯接线）+ 3 级桶移(7−gsh)，
    //   sat9 比较从 61b 缩到 25b——数值与原 61b 实现逐位相同。
    logic signed [37:0] prh, prh_q;            // D（|D| < 2^35，38b 富余）
    logic signed [24:0] psh13, psh;            // D>>>13 接线截位 / 再 >>>(7−gsh)
    logic signed [8:0]  w9, w9_q;
    (* use_dsp = "no" *) logic signed [24:0] wg;    // w·g_j（S3）
    logic signed [16:0] t17, t17_q;
    (* use_dsp = "no" *) logic signed [24:0] tt;    // t17·t6（S4，AdaRMS）
    logic signed [18:0] ta, tb19, tb_q;
    logic [7:0]         t8v, y8;
    assign xinv  = $signed(rd_r[8*g +: 8]) * $signed({5'b0, inv_rf[g]});
    assign prh   = xinv - $signed(qn_rf_lane[g][61:24])
                 - $signed({37'd0, (|qn_rf_lane[g][23:0])});
    always_ff @(posedge clk) if (p2e1) prh_q <= prh;
    assign psh13 = prh_q[37:13];               // 算术 >>>13：取顶 25 位即补码值
    assign psh   = psh13 >>> (3'd7 - gsh_r);   // 总移位 20−gsh = S−24
    always_comb begin
      if      (psh > 25'sd255)   w9 = 9'sd255;
      else if (psh < -25'sd256)  w9 = -9'sd256;
      else                       w9 = psh[8:0];
    end
    always_ff @(posedge clk) if (p2e2) w9_q <= w9;
    // g_q2 是无符号声明的 16b 寄存器，乘法必须显式 $signed——
    // 有符号×无符号混乘整体退化无符号（w9_q 的负值会变成 512+|w|，坑清单 #2）
    assign wg   = w9_q * $signed(g_q2);
    assign t17  = (wg + 25'sd128) >>> 8;
    always_ff @(posedge clk) if (p2e3) t17_q <= t17;
    // AdaRMS 逐行缩放：t6 = round(t·64)（t8v）；非 ada 恒 64，
    // (64·x+32)>>>6 == x 对一切整数 x 成立 → 非 ada 路径逐位不变
    assign t8v  = ada_r ? t16[8*g +: 8] : 8'd64;
    assign tt   = t17_q * $signed(t8v);
    assign ta   = (tt + 25'sd32) >>> 6;
    assign tb19 = ta + $signed({{3{b_q3[15]}}, b_q3});
    always_ff @(posedge clk) if (p2e4) tb_q <= tb19;
    always_comb begin
      if (osh_r == 4'd0) begin
        if      (tb_q > 19'sd127)   y8 = 8'd127;
        else if (tb_q < -19'sd128)  y8 = -8'sd128;
        else                        y8 = tb_q[7:0];
      end else begin
        logic signed [19:0] tr;
        logic signed [19:0] tsh;
        tr  = tb_q + (20'sd1 <<< (osh_r - 4'd1));
        tsh = tr >>> osh_r;
        if      (tsh > 19'sd127)   y8 = 8'd127;
        else if (tsh < -19'sd128)  y8 = -8'sd128;
        else                        y8 = tsh[7:0];
      end
    end
    assign nwbyte[8*g +: 8] = y8;
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

  // ---- NORM/BIAS 装载区 lo 基址（NORM：g 表 b+1、b 表 b+1+2NLO）----
  logic [19:0] ld_lo_base, ld_hi_base;
  always_comb begin
    if (md_norm) begin
      case (ld_reg)
        2'd2: begin ld_lo_base = tbase_r + 20'd1 + {8'd0, nlo_r, 1'b0};
                     ld_hi_base = ld_lo_base + {8'd0, nlo_r}; end
        default: begin ld_lo_base = tbase_r + 20'd1;
                       ld_hi_base = ld_lo_base + {8'd0, nlo_r}; end
      endcase
    end else begin
      ld_lo_base = tbase_r;
      ld_hi_base = tbase_r + {8'd0, nlo_r};
    end
  end

  // ---- 读地址（每状态一个来源；RUN/P1/P2 用预读指针 jr）----
  always_comb begin
    ctx_raddr = grp_base;                     // 默认无害
    case (st)
      A_LD:      ctx_raddr = tbase_r + {11'd0, li};
      A_B_LO:    ctx_raddr = ld_lo_base + {8'd0, bg_r};
      A_B_HI:    ctx_raddr = ld_hi_base + {8'd0, bg_r};
      A_N_C0:    ctx_raddr = tbase_r;
      A_RUN:     ctx_raddr = grp_base + {3'd0, jr};
      A_N_P1:    ctx_raddr = grp_base + {3'd0, jr};
      A_N_P2:    ctx_raddr = grp_base + {3'd0, jr};
      // AdaRMS：t 字（tbl+1+4·NLO+组号）趁统计 c9 拍 A 口空闲读，
      // 2 拍延迟后恰在 A_N_P2 首拍落进 rd_r（零额外拍数；非 ada 地址不动）
      A_N_ST:    ctx_raddr = (ada_r && st_cnt == 4'd9)
                  ? tbase_r + 20'd1 + {nlo_r, 2'b00} + {8'd0, row[15:4]}
                  : grp_base;
      A_E_LT:    ctx_raddr = ph_q ? (x2g_r + {4'd0, jc})
                                  : (grp_base + {4'd0, jc});
      default:   ctx_raddr = grp_base;
    endcase
  end
  assign bias_ra = jr[BAW-1:0];
  assign g_ra    = jr[BAW-1:0];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= A_IDLE; busy <= 1'b0; done <= 1'b0;
      md_bias <= 1'b0; md_norm <= 1'b0; tbase_r <= '0; ybase_r <= '0; grp_base <= '0;
      m_r <= '0; n_r <= '0; k_r <= '0; rq_m_r <= '0; rq_s_r <= '0; nlo_r <= '0;
      invn_r <= '0; eps_r <= '0; gsh_r <= '0; osh_r <= '0; ln_r <= 1'b0;
      ada_r <= 1'b0; t16 <= '0;
      md_elt <= 1'b0; m2_r <= '0; x2g_r <= '0;
      ph_q <= 1'b0; iv1 <= 1'b0; iv2 <= 1'b0; pd1 <= 1'b0; pd2 <= 1'b0;
      jc <= '0; isu_done <= 1'b0; lastw <= 1'b0;
      li <= '0; lut_wa <= '0; bg_r <= '0; bwr_i <= '0; ld_reg <= '0;
      lo16 <= '0; hi16 <= '0;
      jr <= '0; jw <= '0; run_v1 <= 1'b0; run_v2 <= 1'b0;
      r_last <= 1'b0; bwv <= 1'b0; bwa <= '0; ewv <= 1'b0; ewa <= '0;
      p2v5 <= 1'b0; p2j5 <= '0; p2y_q <= '0;
      row <= '0; lane_mask <= '0;
      rd_r <= '0; b_r <= '0; g_r <= '0;
      bias_we <= 1'b0; bias_wa <= '0; bias_wd <= '0;
      g_we <= 1'b0; g_wa <= '0; g_wd <= '0;
      lut_we <= 1'b0; lut_wd <= '0;
      st_lane <= '0; st_cnt <= '0;
      mu_c <= '0; ms_c <= '0; sq_c <= '0; mq_c <= '0;
      f_c <= '0; r0_c <= '0; r0sq_c <= '0; mrs_c <= '0; nw_c <= '0;
      ve_c <= '0; invq_c <= '0;
      ctx_we <= 1'b0; ctx_welane <= '0; ctx_waddr <= '0; ctx_wdata <= '0;
      for (int L = 0; L < 16; L++) begin
        s1[L] <= '0; s2[L] <= '0; mu_rf[L] <= '0; inv_rf[L] <= '0; qn_rf[L] <= '0;
      end
      p2v1 <= 1'b0; p2v2 <= 1'b0; p2v3 <= 1'b0; p2v4 <= 1'b0;
      p2j1 <= '0; p2j2 <= '0; p2j3 <= '0; p2j4 <= '0;
      g_q <= '0; b_q <= '0; g_q2 <= '0; b_q2 <= '0; b_q3 <= '0;
    end else begin
      done <= 1'b0; ctx_we <= 1'b0; ctx_welane <= '0;
      bias_we <= 1'b0; g_we <= 1'b0; lut_we <= 1'b0;
      rd_r <= ctx_rdata;                      // 读流水无条件推进
      b_r  <= bias_rd;
      g_r  <= g_rd;

      case (st)
        A_IDLE: if (start) begin
            busy    <= 1'b1;
            md_bias <= (submode == 3'd1);
            md_norm <= (submode == 3'd2);
            md_elt  <= (submode == 3'd3);
            m2_r    <= rq_m2;
            x2g_r   <= tbl_base;              // ELTWISE：x2 基址走 tbl_base
            ph_q <= 1'b0; iv1 <= 1'b0; iv2 <= 1'b0;
            jc <= '0; isu_done <= 1'b0; lastw <= 1'b0;
            tbase_r <= tbl_base;
            ybase_r <= y_base;
            grp_base<= y_base;
            m_r <= m_rows; n_r <= n_cols; k_r <= tbl_len;
            rq_m_r <= rq_m; rq_s_r <= rq_s;
            nlo_r <= nlo_w[15:4];                               // ceil(k/16)
            li <= '0; lut_wa <= 8'hFF; bg_r <= '0; bwr_i <= '0;  // FF 起步：首拍武装写址 0
            ld_reg <= '0;
            row <= '0; lane_mask <= row_mask(16'd0, m_rows);
            jr <= '0; jw <= '0; run_v1 <= 1'b0; run_v2 <= 1'b0;
            // v1.2.3 写口流水/排空标志——上一用例残值会伪造写请求，顺手清
            r_last <= 1'b0; bwv <= 1'b0; ewv <= 1'b0; p2v5 <= 1'b0;
            // ELTWISE 直入 A_E_LT，无装载级可顺手清 jw——上一用例的残值
            // 会让整组写进同一个字（其余子模式的装载级本来就会再清一遍）
            case (submode)
              3'd1: st <= A_B_LO;
              3'd2: st <= A_N_C0;
              3'd3: st <= A_E_LT;             // ELTWISE 无表装载，直入读流
              default: st <= A_LD;
            endcase
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
        // ---------------- 装载：BIAS 表 / NORM g、b 表（逐组 20 拍）----------------
        //   md_norm 时按 ld_reg 依次装 g 表（写 g_mem）与 b 表（写 bias_mem）。
        A_B_LO: st <= A_B_HI;                 // raddr = 本区 lo 组 bg_r
        A_B_HI: st <= A_B_C1;                 // raddr = 本区 hi 组
        A_B_C1: begin lo16 <= rd_r; st <= A_B_C2; end   // rd_r = lo 组
        A_B_C2: begin hi16 <= rd_r; st <= A_B_WR; end   // rd_r = hi 组
        A_B_WR: begin
          if (ld_reg == 2'd1) begin
            g_we <= 1'b1;  g_wa <= bwa_full[BAW-1:0];
            g_wd <= {hi16[8*bwr_i +: 8], lo16[8*bwr_i +: 8]};
          end else begin
            bias_we <= 1'b1;  bias_wa <= bwa_full[BAW-1:0];
            bias_wd <= {hi16[8*bwr_i +: 8], lo16[8*bwr_i +: 8]};
          end
          if (bwr_i == 4'd15 || bnext >= {1'b0, k_r}) begin
            bwr_i <= '0;
            if (blast) begin
              if (md_norm && ld_reg == 2'd1) begin
                ld_reg <= 2'd2;               // g 表装完 → b 表
                bg_r <= '0;
                st <= A_B_LO;
              end else begin
                jr <= '0; jw <= '0; run_v1 <= 1'b0; run_v2 <= 1'b0;
                for (int L = 0; L < 16; L++) begin
                  s1[L] <= '0; s2[L] <= '0;       // 一遍累加器每组清零
                end
                st <= md_norm ? A_N_P1 : A_RUN;
              end
            end else begin
              bg_r <= bg_r + 12'd1;
              st <= A_B_LO;
            end
          end else bwr_i <= bwr_i + 4'd1;
        end
        // ---------------- NORM 常数区装载（3 拍）----------------
        A_N_C0: st <= A_N_C1;                 // raddr = tbase（常数字）
        A_N_C1: st <= A_N_C2;
        A_N_C2: begin                         // rd_r = 常数字，切字节解码
          invn_r <= rd_r[23:0];
          eps_r  <= rd_r[71:24];
          gsh_r  <= rd_r[74:72];
          osh_r  <= rd_r[81:78];
          ln_r   <= rd_r[82];
          ada_r  <= rd_r[83];                 // bit83：AdaRMS t 区存在
          ld_reg <= 2'd1; bg_r <= '0; bwr_i <= '0;
          st <= A_B_LO;
        end
        // ---------------- NORM 一遍扫描：每 lane 精确累加 S1/S2 ----------------
        A_N_P1: begin
          if (jr < {1'b0, n_r}) begin
            jr <= jr + 17'd1;
            run_v1 <= 1'b1;
          end else run_v1 <= 1'b0;
          run_v2 <= run_v1;
          if (run_v2) begin                   // rd_r = 列 jw 的 16 行字节
            for (int L = 0; L < 16; L++) begin
              s1[L] <= s1[L] + $signed(rd_r[8*L +: 8]);
              s2[L] <= s2[L] + pmul[L];      // 共享阵列算的 x²（g_bias.pmul）
            end
            if (jw + 17'd1 >= {1'b0, n_r}) begin
              st_lane <= '0; st_cnt <= '0;
              st <= A_N_ST;
            end else jw <= jw + 17'd1;
          end
        end
        // ---------------- NORM 统计级：共享单元逐 lane 11 拍 ----------------
        //   c0 μ  c1 ms+存μ  c2 μ²>>>24  c3 var+eps+钳  c4 LOD+桶移
        //   c5 查 rtab  c6 r0²  c7 m·r0²  c8 牛顿乘  c9 存 inv  c10 存 qn
        //   （v1.2.2：c3 三段拆拍修 WNS，见上方"统计前段"注释；qn 乘法改吃
        //    c9 锁存的 invq_c，桶移与 32×27 乘不再串在同一拍）
        A_N_ST: begin
          st_cnt <= st_cnt + 4'd1;
          case (st_cnt)
            4'd0: mu_c <= m_s[31:0];
            4'd1: begin ms_c <= m_s[47:0];
                         // RMS 不减均值：μ 存 0（二遍 u = x<<<24，q 也随之 0）
                         mu_rf[st_lane] <= ln_r ? mu_c : '0; end
            4'd2: sq_c <= mu2_p24;
            4'd3: ve_c <= v_eff_w;            // var+eps+max（减/加各一段）
            4'd4: begin mq_c <= mq_w;         // LOD 偶化 + 桶移（输入 ve_c）
                         f_c  <= ve_w[5:1]; end
            4'd5: r0_c <= rtab[rt_ix];        // 512×15 ROM 查表（输入 mq_c）
            4'd6: r0sq_c <= r0r0;
            4'd7: mrs_c  <= mmul;
            4'd8: nw_c   <= nmul;
            4'd9: begin inv_rf[st_lane] <= inv27_w;    // 桶移+sat27 收尾
                         invq_c <= inv27_w; end
            default: begin  // c10：qn = μ·inv − 2^(S-1)（乘法吃 invq_c）
              qn_rf[st_lane] <= {{3{qm_w[58]}}, qm_w} - {{3{nhalf[60]}}, nhalf};
              st_cnt <= '0;
              if (st_lane == 4'd15) begin
                jr <= '0; jw <= '0; run_v1 <= 1'b0; run_v2 <= 1'b0;
                p2v1 <= 1'b0; p2v2 <= 1'b0; p2v3 <= 1'b0; p2v4 <= 1'b0;
                p2v5 <= 1'b0;
                st <= A_N_P2;
              end else st_lane <= st_lane + 4'd1;
            end
          endcase
        end
        // ---------------- 原地变换：稳态每拍写一整列（16 行）----------------
        //   v1.2.3：ACTV 直写不动（LUT 异步读路径短，公式 n+3 不变）；
        //   BIAS 写口数据晚一拍（accb_q），ewbyte/bwbyte 走 bwv 寄存拍，
        //   每组 +1 拍排空（n+4）。
        A_RUN: begin
          if (jr < {1'b0, n_r}) begin
            jr <= jr + 17'd1;
            run_v1 <= 1'b1;
          end else run_v1 <= 1'b0;
          run_v2 <= run_v1;
          if (run_v2) begin                   // rd_r/b_r = 列 jw 数据
            if (!md_bias) begin               // ACTV：直写
              ctx_we     <= 1'b1;
              ctx_welane <= lane_mask;
              ctx_waddr  <= grp_base + {4'd0, jw[15:0]};
              ctx_wdata  <= wbyte;
            end else begin                    // BIAS：先武装，下一拍写
              bwv <= 1'b1;
              bwa <= grp_base + {4'd0, jw[15:0]};
            end
            if (jw + 17'd1 >= {1'b0, n_r}) r_last <= 1'b1;
            else jw <= jw + 17'd1;
          end else bwv <= 1'b0;
          if (bwv) begin                      // BIAS 写拍（数据 = accb_q 链）
            ctx_we     <= 1'b1;
            ctx_welane <= lane_mask;
            ctx_waddr  <= bwa;
            ctx_wdata  <= bwbyte;
          end
          if (r_last && !run_v2 && !bwv) st <= A_NEXT;
        end
        // ------- NORM 二遍：4 级流水，发址每拍一列，写回滞后 4 拍 -------
        //   t0 发址 j → t+2 rd_r/g_r=列j（S1 装载 prh_q/g_q/b_q）
        //   → t+3 S2 装载 w9_q/g_q2/b_q2 → t+4 S3 装载 tb_q → t+4 S4 写回
        A_N_P2: begin
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
          p2v5 <= p2v4;    p2j5 <= p2j4;       // v1.2.3：写口再骑一级
          if (p2v4) p2y_q <= nwbyte;           // S5 结果先落本地寄存（第 6 级）
          if (run_v2) begin
            g_q <= g_r;  b_q <= b_r;          // S1 随 prh_q 同拍对齐
            if (jw + 17'd1 >= {1'b0, n_r}) jw <= jw;
            else jw <= jw + 17'd1;
          end
          if (p2v1) begin g_q2 <= g_q; b_q2 <= b_q; end
          if (p2e3) b_q3 <= b_q2;             // b 随 t17_q 再骑一级
          if (p2v5) begin                      // 写口寄存源（S5 之后一拍）
            ctx_we     <= 1'b1;
            ctx_welane <= lane_mask;
            ctx_waddr  <= grp_base + {4'd0, p2j5};
            ctx_wdata  <= p2y_q;
          end
          if (jr >= {1'b0, n_r} && !run_v1 && !run_v2 &&
              !p2v1 && !p2v2 && !p2v3 && !p2v4 && !p2v5) st <= A_NEXT;
        end
        // ------- ELTWISE：双输入残差加，每列两拍（x1 乘 → x2 乘合并写）-------
        //   发址奇偶交替：偶拍 grp_base+jc 读 x1、奇拍 x2g_r+jc 读 x2；
        //   2 拍延迟后按 pd2 相位分流：x1 拍共享阵列乘 m1 折 half 存 ea_q，
        //   x2 拍乘 m2 相加后走同一条 >>>s/sat8 出口写回列 jw。
        //   吞吐 1 列/2 拍（CTX 单读口，双输入只能交替读，如实计费）。
        //   v1.2.3：写口数据（accb_q 链的 ewbyte）晚一拍落写，每组 +1 排空。
        A_E_LT: begin
          if (!isu_done) begin
            iv1 <= 1'b1;  pd1 <= ph_q;  ph_q <= ~ph_q;
            if (ph_q) begin                       // 本拍发 x2 → 本列发址完成
              if ({1'b0, jc} + 17'd1 >= {1'b0, n_r}) isu_done <= 1'b1;
              else jc <= jc + 16'd1;
            end
          end else iv1 <= 1'b0;                   // 多保 1 拍：末列 x2 数据在途
          iv2 <= iv1;  pd2 <= pd1;
          if (iv2 && pd2) begin                   // rd_r = 列 jw 的 x2 → 武装写
            ewv <= 1'b1;
            ewa <= grp_base + {4'd0, jw[15:0]};
            if ({1'b0, jw} + 17'd1 >= {1'b0, n_r}) lastw <= 1'b1;
            else jw <= jw + 17'd1;
          end else ewv <= 1'b0;
          if (ewv) begin                          // 合并写（数据 = accb_q 链）
            ctx_we     <= 1'b1;
            ctx_welane <= lane_mask;
            ctx_waddr  <= ewa;
            ctx_wdata  <= ewbyte;
          end
          if (lastw && !iv1 && !iv2 && !ewv) st <= A_NEXT;
        end
        A_NEXT: begin
          if ({1'b0, row} + 17'd16 >= {1'b0, m_r}) begin
            st <= A_FIN; busy <= 1'b0;
          end else begin
            row      <= row + 16'd16;
            grp_base <= grp_base + {4'd0, n_r};
            jr <= '0; jw <= '0; run_v1 <= 1'b0; run_v2 <= 1'b0;
            r_last <= 1'b0; bwv <= 1'b0; ewv <= 1'b0; p2v5 <= 1'b0;
            lane_mask <= row_mask(row + 16'd16, m_r);
            x2g_r <= x2g_r + {4'd0, n_r};         // ELTWISE x2 组基址随组推进
            ph_q <= 1'b0; iv1 <= 1'b0; iv2 <= 1'b0;
            jc <= '0; isu_done <= 1'b0; lastw <= 1'b0;
            if (md_norm)
              for (int L = 0; L < 16; L++) begin
                s1[L] <= '0; s2[L] <= '0;         // 下一行组的一遍累加器清零
              end
            st <= md_norm ? A_N_P1 : (md_elt ? A_E_LT : A_RUN);
          end
        end
        A_FIN: begin done <= 1'b1; st <= A_IDLE; end
        default: st <= A_IDLE;
      endcase
    end
  end
endmodule
`endif
