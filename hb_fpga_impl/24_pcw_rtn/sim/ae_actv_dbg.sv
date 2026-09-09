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
//   ACTV 260+ceil16(m)·(n+3)   BIAS ceil16(k)·20+ceil16(m)·(n+3)
//   NORM 3+2·ceil16(n)·20+ceil16(m)·(2n+132)
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
                            A_N_P2} st_e;
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

  // ---- 装载计数 ----
  logic [8:0]  li;                            // ACTV：发读字指针 0..256
  logic [11:0] bg_r;                          // BIAS/NORM：当前组号
  logic [3:0]  bwr_i;                         // 组内写序号
  logic [1:0]  ld_reg;                        // NORM 装载区：1=g 表 2=b 表（0=BIAS）
  logic [127:0] lo16, hi16;                   // 本组 lo/hi 捕获（lane L = 项 j&15）

  // ---- RUN 指针与流水 ----
  logic [16:0] jr, jw;                        // 发读列 / 写列（滞后 3 拍）
  logic        run_v1, run_v2;                // 读数据 2 拍有效流水
  logic [15:0] row;                           // 行组基行号
  logic [15:0] lane_mask;                     // 本组有效行掩码

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
  (* use_dsp = "no" *) logic signed [32:0] sq_mid;  // 2·A_hi·A_lo
  (* use_dsp = "no" *) logic signed [25:0] sq_lo;   // A_lo²（13×13 要 26b）
  assign sq_hi = mu_hi * mu_hi;
  assign sq_mid = (mu_hi * $signed({1'b0, mu_lo})) <<< 1;
  assign sq_lo = $signed({1'b0, mu_lo}) * $signed({1'b0, mu_lo});
  // mid_sh：2·A_hi·A_lo·2^12 + A_lo²。必须用有符号表达式加（赋值 46b 上下文
  // 会先把两边符号扩展到 46b 再加）——若用 {拼接} 做，两个 45b 无符号加法会把
  // 负结果的符号进位丢在 bit45 之外（差恰好 2^45，肉眼极难查）。
  wire signed [45:0] mid_sh = (sq_mid <<< 12) + sq_lo;
  wire signed [38:0] mu2_p24 = sq_hi + $signed(mid_sh[45:24]);

  // 统计乘法器（共享，一拍一个）
  (* use_dsp = "no" *) logic signed [47:0] m_s1;    // S1·invn（值 ≤ 2^31）
  (* use_dsp = "no" *) logic [53:0]    m_s2;        // S2·invn（无符号，≤2^38）
  assign m_s1 = s1[st_lane] * $signed({1'b0, invn_r});
  assign m_s2 = s2[st_lane] * {1'b0, invn_r};

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

  // ---- c3 拍组合：v_eff / E / m_q11（输入是本 lane 的 ms_c/sq_c 寄存器）----
  wire signed [48:0] var_w = ln_r ? ((ms_c >= {10'd0, sq_c})
                                       ? ({1'b0, ms_c} - {10'd0, sq_c})
                                       : 49'sd0)
                                  : {1'b0, ms_c};
  wire signed [48:0] vraw_w = var_w + $signed({1'b0, eps_r});
  wire [48:0] v_eff_w = (vraw_w < 49'sd8192) ? 49'sd8192 : vraw_w;
  wire [5:0]  ve_w  = even_msb49(v_eff_w);
  wire [12:0] mq_w  = v_eff_w >>> (ve_w - 6'd11);
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
  assign qm_w = mu_rf[st_lane] * $signed({5'd0, inv27_w});

  // 一遍累加用的每 lane x²（8×8，显式禁 DSP）
  (* use_dsp = "no" *) logic signed [15:0] xsq [0:15];
  for (genvar g = 0; g < 16; g++) begin : g_xsq
    assign xsq[g] = $signed(rd_r[8*g +: 8]) * $signed(rd_r[8*g +: 8]);
  end

  // ============ NORM 二遍：每 lane 4 级流水（吞吐 1 列/拍）================
  // 原单周期组合链（乘→减→桶移→饱和→乘→加→桶移→饱和→写）OOC 时序
  // WNS −4.9ns；切 4 级：S1=x·inv乘+61b减  S2=桶移+sat9  S3=9×16乘+两级加
  // S4=out_shift桶移+sat8+写回。列号随流水 3 级跟踪（p2j1..p2j3），
  // S4 的寄存器即写口 ctx_wdata（共 4 级）。
  // 输出终值与原组合链逐位一致（只是重新切拍），微观位精确门不变。
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
  logic       p2v1, p2v2, p2v3;
  logic [15:0] p2j1, p2j2, p2j3;
  wire  p2e1 = (st == A_N_P2) && run_v2;      // stage1 装载使能
  wire  p2e2 = (st == A_N_P2) && p2v1;
  wire  p2e3 = (st == A_N_P2) && p2v2;
  logic [15:0] g_q, b_q, g_q2, b_q2;
  logic       dbg_en;
  logic [8:0] dbg_n;
  always_ff @(posedge clk) begin
    if (st == A_N_P2) begin if (dbg_n < 9'd80) begin dbg_en <= 1'b1; dbg_n <= dbg_n + 9'd1; end else dbg_en <= 1'b0; end
    else if (st == A_N_ST || st == A_N_P1) begin dbg_n <= '0; dbg_en <= 1'b0; end
  end
  for (genvar g = 0; g < 16; g++) begin : g_norm
    (* use_dsp = "no" *) logic signed [34:0] xinv;  // x·inv（8×27，35b）
    logic signed [58:0] xm;                    // (x·inv)<<<24，与 x·(inv<<<24) 精确等价
    logic signed [61:0] prh;                   // xm − qn（qn 已折进 +2^(S-1)）
    logic signed [61:0] prh_q;                 // S1 流水寄存器
    logic signed [61:0] psh;                   // S2：桶移
    logic signed [8:0]  w9, w9_q;
    (* use_dsp = "no" *) logic signed [24:0] wg;    // w·g_j（S3）
    logic signed [16:0] t17;
    logic signed [17:0] tb18, tb_q;
    logic [7:0]         y8;
    assign xinv = $signed(rd_r[8*g +: 8]) * $signed({5'b0, inv_rf[g]});
    assign xm   = xinv <<< 24;
    assign prh  = xm - qn_rf_lane[g];
    always_ff @(posedge clk) if (p2e1) prh_q <= prh;
    assign psh  = prh_q >>> nS;
    always_comb begin
      if      (psh > 61'sd255)   w9 = 9'sd255;
      else if (psh < -61'sd256)  w9 = -9'sd256;
      else                       w9 = psh[8:0];
    end
    always_ff @(posedge clk) if (p2e2) w9_q <= w9;
    assign wg   = w9_q * g_q2;
    assign t17  = (wg + 25'sd128) >>> 8;
    assign tb18 = t17 + $signed({{2{b_q2[15]}}, b_q2});
    always_ff @(posedge clk) if (p2e3) tb_q <= tb18;
    always_comb begin
      if (osh_r == 4'd0) begin
        if      (tb_q > 18'sd127)   y8 = 8'd127;
        else if (tb_q < -18'sd128)  y8 = -8'sd128;
        else                        y8 = tb_q[7:0];
      end else begin
        logic signed [18:0] tr;
        logic signed [18:0] tsh;
        tr  = tb_q + (19'sd1 <<< (osh_r - 4'd1));
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
      li <= '0; lut_wa <= '0; bg_r <= '0; bwr_i <= '0; ld_reg <= '0;
      lo16 <= '0; hi16 <= '0;
      jr <= '0; jw <= '0; run_v1 <= 1'b0; run_v2 <= 1'b0;
      row <= '0; lane_mask <= '0;
      rd_r <= '0; b_r <= '0; g_r <= '0;
      bias_we <= 1'b0; bias_wa <= '0; bias_wd <= '0;
      g_we <= 1'b0; g_wa <= '0; g_wd <= '0;
      lut_we <= 1'b0; lut_wd <= '0;
      st_lane <= '0; st_cnt <= '0;
      mu_c <= '0; ms_c <= '0; sq_c <= '0; mq_c <= '0;
      f_c <= '0; r0_c <= '0; r0sq_c <= '0; mrs_c <= '0; nw_c <= '0;
      ctx_we <= 1'b0; ctx_welane <= '0; ctx_waddr <= '0; ctx_wdata <= '0;
      for (int L = 0; L < 16; L++) begin
        s1[L] <= '0; s2[L] <= '0; mu_rf[L] <= '0; inv_rf[L] <= '0; qn_rf[L] <= '0;
      end
      p2v1 <= 1'b0; p2v2 <= 1'b0; p2v3 <= 1'b0;
      p2j1 <= '0; p2j2 <= '0; p2j3 <= '0;
      g_q <= '0; b_q <= '0; g_q2 <= '0; b_q2 <= '0;
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
            tbase_r <= tbl_base;
            ybase_r <= y_base;
            grp_base<= y_base;
            m_r <= m_rows; n_r <= n_cols; k_r <= tbl_len;
            rq_m_r <= rq_m; rq_s_r <= rq_s;
            nlo_r <= nlo_w[15:4];                               // ceil(k/16)
            li <= '0; lut_wa <= 8'hFF; bg_r <= '0; bwr_i <= '0;  // FF 起步：首拍武装写址 0
            ld_reg <= '0;
            row <= '0; lane_mask <= row_mask(16'd0, m_rows);
            case (submode)
              3'd1: st <= A_B_LO;
              3'd2: st <= A_N_C0;
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
              s2[L] <= s2[L] + xsq[L];      // xsq 线网带 use_dsp=no
            end
            if (jw + 17'd1 >= {1'b0, n_r}) begin
              st_lane <= '0; st_cnt <= '0;
              st <= A_N_ST;
            end else jw <= jw + 17'd1;
          end
        end
        // ---------------- NORM 统计级：共享单元逐 lane 8 拍 ----------------
        //   c0 μ=S1·invn  c1 ms=S2·invn+存μ  c2 μ²>>>24  c3 var+v+LOD+LUT
        //   c4 r0²  c5 m·r0²  c6 牛顿乘  c7 收尾+存 inv/q，lane++
        A_N_ST: begin
          st_cnt <= st_cnt + 4'd1;
          case (st_cnt)
            3'd0: mu_c <= m_s1[31:0];
            3'd1: begin ms_c <= m_s2[47:0];
                         // RMS 不减均值：μ 存 0（二遍 u = x<<<24，q 也随之 0）
                         mu_rf[st_lane] <= ln_r ? mu_c : '0; end
            3'd2: sq_c <= mu2_p24;
            3'd3: begin                 // v/E/m_q11 全部用本 lane 的 ms_c/sq_c 组合算
              mq_c <= mq_w;
              f_c  <= ve_w[5:1];
              r0_c <= rtab[rt_ix];
            end
            3'd4: r0sq_c <= r0r0;
            3'd5: mrs_c  <= mmul;
            3'd6: nw_c   <= nmul;
            4'd7: begin   // 收尾拍：桶移+sat27 存 inv（组合线 inv27_w/qm_w）
              inv_rf[st_lane] <= inv27_w;
            end
            default: begin  // 4'd8：qn = μ·inv − 2^(S-1)（独立拍，乘法带 use_dsp=no）
              qn_rf[st_lane] <= {{3{qm_w[58]}}, qm_w} - {{3{nhalf[60]}}, nhalf};
              st_cnt <= '0;
              if (st_lane == 4'd15) begin
                jr <= '0; jw <= '0; run_v1 <= 1'b0; run_v2 <= 1'b0;
                p2v1 <= 1'b0; p2v2 <= 1'b0; p2v3 <= 1'b0;
                st <= A_N_P2;
              end else st_lane <= st_lane + 4'd1;
            end
          endcase
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
        // ------- NORM 二遍：4 级流水，发址每拍一列，写回滞后 4 拍 -------
        //   t0 发址 j → t+2 rd_r/g_r=列j（S1 装载 prh_q/g_q/b_q）
        //   → t+3 S2 装载 w9_q/g_q2/b_q2 → t+4 S3 装载 tb_q → t+4 S4 写回
        A_N_P2: begin
          if (dbg_n == 9'd0)
            $display("[ST] lane0 mu=%0d inv=%0d qn=%0d  lane1 mu=%0d inv=%0d qn=%0d",
                     mu_rf[0], inv_rf[0], qn_rf[0], mu_rf[1], inv_rf[1], qn_rf[1]);
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
          // ---- DEBUG 探针（仅 lane0；posedge 显示的是本拍输入侧值）----
          if (dbg_en) $display("[P2] t=%0t jr=%0d jw=%0d r1=%b r2=%b v1=%b v2=%b v3=%b j1=%0d j2=%0d j3=%0d prhq=%0d w9q=%0d tbq=%0d gq=%0d gq2=%0d bq2=%0d nw0=%02X rd0=%0d",
                $time, jr, jw, run_v1, run_v2, p2v1, p2v2, p2v3, p2j1, p2j2, p2j3,
                g_norm[0].prh_q, g_norm[0].w9_q, g_norm[0].tb_q, g_q, g_q2, b_q2, nwbyte[7:0], $signed(rd_r[7:0]));
          if (p2v3) begin                       // S4 写回：本拍 nwbyte 已是 tb_q 的
            ctx_we     <= 1'b1;                 // 桶移+sat8 结果，S4 的流水寄存器
            ctx_welane <= lane_mask;            // 就是写口的 ctx_wdata（滞后 3 级）
            ctx_waddr  <= grp_base + {4'd0, p2j3};
            ctx_wdata  <= nwbyte;
          end
          if (jr >= {1'b0, n_r} && !run_v1 && !run_v2 &&
              !p2v1 && !p2v2 && !p2v3) st <= A_NEXT;
        end
        A_NEXT: begin
          if ({1'b0, row} + 17'd16 >= {1'b0, m_r}) begin
            st <= A_FIN; busy <= 1'b0;
          end else begin
            row      <= row + 16'd16;
            grp_base <= grp_base + {4'd0, n_r};
            jr <= '0; jw <= '0; run_v1 <= 1'b0; run_v2 <= 1'b0;
            lane_mask <= row_mask(row + 16'd16, m_r);
            if (md_norm)
              for (int L = 0; L < 16; L++) begin
                s1[L] <= '0; s2[L] <= '0;         // 下一行组的一遍累加器清零
              end
            st <= md_norm ? A_N_P1 : A_RUN;
          end
        end
        A_FIN: begin done <= 1'b1; st <= A_IDLE; end
        default: st <= A_IDLE;
      endcase
    end
  end
endmodule
`endif
