// ae_pe_pack_dsp.sv — packed INT8 PE，手工例化 DSP48E2 原语版（R3 证伪后的第 4 次尝试）
// ----------------------------------------------------------------------------
// 与 ae_pe_pack.sv（行为级，位精确已证）端口/协议/拍数完全一致；唯一差别是 win
// 累加器不再靠行为级 RTL 等综合器推断，而是直接例化一颗 DSP48E2 原语，绕过
// Vivado 2021.2 推断器（行为级三次尝试均被拆成 2×DSP，见 ROUND3_INTEGRATION §2）。
//
// 数学方案与行为级逐字相同（推导全文见 ae_pe_pack.sv 注释）：
//   A 口 27b：Ã = ã1·2^18 + ã0，ã = a+128（符号位取反，纯线网零逻辑）
//   B 口 18b：共享 b 符号扩展
//   低场 P[17:0] = Σ4(ã0·b)，高场 P[35:18] = Σ4(ã1·b) + floor，两场不重叠
//   抽取校正：acc0 += P[17:0] − (sb<<<7)；acc1 += P[35:18] + P[17] − (sb<<<7)
//
// DSP48E2 映射（opmode 位段编码逐一核对自 unisims/DSP48E2.v 的 xmux/ymux/zmux/
// wmux 四段 always 块，非凭记忆）：
//   行为级 win 方程（非 clr 分支）：
//     win' = (flush&&!en) ? 0 : (full ? 0 : win) + (en ? M : 0)
//   DSP48E2 的 ALU 是 X+Y+Z+W 四操作数加法（ALUMODE=0000，W 恒 0，cin=0）：
//   · 乘法器输出 M 在片上拆成两个 Booth 部分和：X mux（OPMODE[1:0]=01）选 U
//     （符号扩展进 48b）、Y mux（OPMODE[3:2]=01）选 V（零扩展），U+V ≡ M
//     (mod 2^48)。M 没有单口可取——所以「乘积门控」必须同时压 X、Y 两个 mux
//     （只压一个会把残缺部分和加进 P）。X mux 的 10 档虽然也能选 P，但那样 U
//     就没有入口——累加反馈必须走 Z mux（OPMODE[6:4]=010 选 P 反馈）。
//   · Z 口平时选 P（累加），翻窗/清零拍选 0：z_fb = ~(clr|full|(flush&~en))
//   · X/Y 口乘积使能拍选 U/V，否则选 0：    m_en = en & ~clr
//   · OPMODE = {2'b00(W=0), z_fb ? 3'b010 : 3'b000, {2{m_en ? 2'b01 : 2'b00}}}
//   六种拍型逐一核对（均与行为级 win 逐拍一致）：
//     clr            → P' = 0            （X=Y=Z=0）
//     flush && !en   → P' = 0            （Z=0，X=Y=0）
//     full && en     → P' = M            （新窗以本拍积起步，背靠背不断流）
//     full && !en    → P' = 0
//     !full && en    → P' = P + M
//     !full && !en   → P' = P
//
// 参数：AREG=0/BREG=0/MREG=0/PREG=1/OPMODEREG=0/ALUMODEREG=0/INMODEREG=0，
//   INMODE=00000（A/B 直进乘法器，不用预加器），USE_MULT=MULTIPLY，USE_SIMD=
//   ONE48，ALUMODE=0000（加法），CARRYIN=0/CARRYINSEL=000 ⇒ cin=0。DSP48E2
//   没有 USE_DPORT 参数（那是 E1 的）——D 口关闭由 AMULTSEL="A"/BMULTSEL="B"
//   表达。A 口 30b 中乘法器只用 [26:0]，[29:27] 接 0（A:B 拼接档才用高位）。
//   C/D/PCIN/ACIN/BCIN/级联口全接死。
//
// P 没有异步复位（DSP48E2 的 RSTP 是同步口，这里全部 RST 管脚接 0，翻窗/清零
//   全走 Z/X/Y mux）：上电配置后 P 初值为 0；rst_n 低电平期间 fabric 状态被
//   异步复位，但 P 若在此期间遇到 en=1 会照常累乘——协议保证 rst_n 之后第一个
//   clr 拍把 P 归零（z_fb=0 且 m_en=0 ⇒ P'=0），tile 级结果不受影响。外部
//   使用必须保持「先 clr 后送数」时序（ae_gemm 的 tile 开始拍正是如此）。
//
// 拍数语义与行为级完全一致：直通链 1 拍寄存、窗口 4 拍抽取、acc 驻留读出，
//   零周期代价。位精确对拍：tb_pe_pack_dsp.sv（2216 tiles vs 2×ae_pe）。
`ifndef AE_PE_PACK_DSP_SV
`define AE_PE_PACK_DSP_SV
module ae_pe_pack_dsp (
  input  logic              clk,
  input  logic              rst_n,
  input  logic              clr,       // 累加器清零（tile 开始）
  input  logic              flush,     // 收尾：残窗折入外部累加器（单拍脉冲）
  input  logic              av_in,     // a 对波前有效（两行共享，结构上同相）
  input  logic              bv_in,     // b 波前有效
  input  logic signed [7:0] a0_in,     // 行 2i（西侧进入）
  input  logic signed [7:0] a1_in,     // 行 2i+1（与 a0 同 κ 对齐）
  input  logic signed [7:0] b_in,      // 北侧进入（列共享）
  output logic              av_out,
  output logic              bv_out,
  output logic signed [7:0] a0_out,    // 东侧出（延迟 1 拍）
  output logic signed [7:0] a1_out,
  output logic signed [7:0] b_out,     // 南侧出（延迟 1 拍，给下一个行对）
  output logic signed [27:0] acc0,     // 行 2i 累加（驻留读出，|·| ≤ 2^26）
  output logic signed [27:0] acc1      // 行 2i+1 累加
);
  localparam int WLOG = 2;             // 窗口 = 4 个有效积
  localparam int WIN  = 1 << WLOG;

  logic signed [7:0] a0_r, a1_r, b_r;
  logic              av_r, bv_r;
  logic              en;

  // ã = a + 128：符号位取反即加 128（补码恒等，纯线网）
  wire [7:0] at0 = {~a0_in[7], a0_in[6:0]};
  wire [7:0] at1 = {~a1_in[7], a1_in[6:0]};

  // Ã = ã1·2^18 + ã0（正数，bit26=0），27b
  wire [26:0] ap = {1'b0, at1, 10'd0, at0};
  wire [17:0] bx = {{10{b_in[7]}}, b_in};
  assign en = av_in & bv_in;

  logic        [WLOG-1:0] wc;              // 窗口内已积个数
  logic                   full;            // 本拍 win 已含完整 4 积（可抽取）
  logic signed [10:0]    sb;               // 窗口 Σb（|·| ≤ 512）
  logic signed [17:0]    corr;             // −(sb <<< 7)：偏置校正项
  logic signed [18:0]    hi_f;             // 高场 + floor 修复（全有符号运算）

  // DSP 控制（组合，直进原语 OPMODE）：
  wire       z_fb = ~(clr | full | (flush & ~en));  // Z mux：1=P 反馈 0=翻窗/清零
  wire       m_en = en & ~clr;                       // X/Y mux：1=乘积部分和 0=0
  wire [8:0] opmode = {2'b00, z_fb ? 3'b010 : 3'b000, {2{m_en ? 2'b01 : 2'b00}}};

  wire [47:0] win;                        // DSP P 输出，即行为级的 win[47:0]

  DSP48E2 #(
    .ACASCREG            (0),
    .ADREG               (0),
    .ALUMODEREG          (0),
    .AMULTSEL            ("A"),           // 乘法器 A 口取 A（不用预加器）
    .AREG                (0),
    .AUTORESET_PATDET    ("NO_RESET"),
    .AUTORESET_PRIORITY  ("RESET"),
    .A_INPUT             ("DIRECT"),
    .BCASCREG            (0),
    .BMULTSEL            ("B"),           // 乘法器 B 口取 B（D 口等效关闭）
    .BREG                (0),
    .B_INPUT             ("DIRECT"),
    .CARRYINREG          (0),
    .CARRYINSELREG       (0),
    .CREG                (0),
    .DREG                (0),
    .INMODEREG           (0),
    .IS_ALUMODE_INVERTED (4'b0000),
    .IS_CARRYIN_INVERTED (1'b0),
    .IS_CLK_INVERTED     (1'b0),
    .IS_INMODE_INVERTED  (5'b00000),
    .IS_OPMODE_INVERTED  (9'b000000000),
    .IS_RSTALLCARRYIN_INVERTED (1'b0),
    .IS_RSTALUMODE_INVERTED    (1'b0),
    .IS_RSTA_INVERTED          (1'b0),
    .IS_RSTB_INVERTED          (1'b0),
    .IS_RSTCTRL_INVERTED       (1'b0),
    .IS_RSTC_INVERTED          (1'b0),
    .IS_RSTD_INVERTED          (1'b0),
    .IS_RSTINMODE_INVERTED     (1'b0),
    .IS_RSTM_INVERTED          (1'b0),
    .IS_RSTP_INVERTED          (1'b0),
    .MASK                (48'h3FFFFFFFFFFF),
    .MREG                (0),
    .OPMODEREG           (0),
    .PATTERN             (48'h000000000000),
    .PREADDINSEL         ("A"),
    .PREG                (1),
    .RND                 (48'h000000000000),
    .SEL_MASK            ("MASK"),
    .SEL_PATTERN         ("PATTERN"),
    .USE_MULT            ("MULTIPLY"),
    .USE_PATTERN_DETECT  ("NO_PATDET"),
    .USE_SIMD            ("ONE48"),
    .USE_WIDEXOR         ("FALSE"),
    .XORSIMD             ("XOR24_48_96")
  ) u_dsp (
    .CLK          (clk),
    // [26:0] = Ã；[29:27] 乘法器不使用
    .A            ({3'b000, ap}),
    .B            (bx),
    .C            (48'b0),
    .D            (27'b0),
    .ACIN         (30'b0),
    .BCIN         (18'b0),
    .PCIN         (48'b0),
    .ALUMODE      (4'b0000),              // X+Y+Z（加法）
    .INMODE       (5'b00000),             // A/B 直进乘法器
    .OPMODE       (opmode),
    .CARRYIN      (1'b0),
    .CARRYINSEL   (3'b000),
    .CARRYCASCIN  (1'b0),
    .MULTSIGNIN   (1'b0),
    .CEA1         (1'b1),
    .CEA2         (1'b1),
    .CEAD         (1'b1),
    .CEALUMODE    (1'b1),
    .CEB1         (1'b1),
    .CEB2         (1'b1),
    .CEC          (1'b1),
    .CECARRYIN    (1'b1),
    .CECTRL       (1'b1),
    .CED          (1'b1),
    .CEINMODE     (1'b1),
    .CEM          (1'b1),
    .CEP          (1'b1),
    // 全部 RST 接 0：P 无异步复位，翻窗/清零走 Z/X/Y mux（见文件头注释）
    .RSTA         (1'b0),
    .RSTALLCARRYIN(1'b0),
    .RSTALUMODE   (1'b0),
    .RSTB         (1'b0),
    .RSTC         (1'b0),
    .RSTCTRL      (1'b0),
    .RSTD         (1'b0),
    .RSTINMODE    (1'b0),
    .RSTM         (1'b0),
    .RSTP         (1'b0),
    .P            (win)
    // 未用输出（ACOUT/BCOUT/PCOUT/级联/模式检测/SIMD）悬空
  );

  assign corr = -(sb <<< 7);
  assign hi_f = $signed(win[35:18]) + (win[17] ? 19'sd1 : 19'sd0);

  // 抽取/直通逻辑与行为级逐字相同（win 换成 DSP P 输出；win 本体不再在
  // fabric 复位清单里——见文件头「P 无异步复位」说明）
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a0_r <= '0; a1_r <= '0; b_r <= '0; av_r <= 1'b0; bv_r <= 1'b0;
      wc <= '0; full <= 1'b0; sb <= '0;
      acc0 <= '0; acc1 <= '0;
    end else begin
      a0_r <= a0_in;  a1_r <= a1_in;  b_r <= b_in;
      av_r <= av_in;  bv_r <= bv_in;
      full <= 1'b0;
      if (clr) begin
        wc <= '0; sb <= '0;
        acc0 <= '0; acc1 <= '0;
      end else begin
        if (en) begin
          if (wc == WIN-1) begin wc <= '0; full <= 1'b1; end
          else wc <= wc + 1'b1;
          // wc==0 即新窗口首个积：sb 从 b 起算（e4 拍 sb 恰为 Σ4，供抽取拍用）
          if (wc == '0) sb <= b_in;
          else          sb <= sb + b_in;
        end
        // 抽取：低场 −偏置 = Σ4(a0·b)；高场 −偏置 + win[17]（floor 修复）。
        // 沿用行为级两条教训：① 全有符号中间量 hi_f（拼接会污染符号语境）；
        // ② flush 残窗守卫 wc!=0（整窗抽完后 sb 是 stale 值，不能白加 corr）。
        if (full || (flush && wc != '0)) begin
          acc0 <= acc0 + $signed(win[17:0]) + corr;
          acc1 <= acc1 + hi_f + corr;
        end
      end
    end
  end

  assign a0_out = a0_r;  assign a1_out = a1_r;  assign av_out = av_r;
  assign b_out  = b_r;   assign bv_out  = bv_r;
endmodule
`endif
