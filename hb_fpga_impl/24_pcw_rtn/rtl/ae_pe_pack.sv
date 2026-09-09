// ae_pe_pack.sv — packed INT8 PE：一个 DSP48E2 承载两行共享 b 的 MAC（门 E2a）
// ----------------------------------------------------------------------------
// 服务对象：同一列、相邻两行（行对 2i/2i+1）——脉动阵列里同一列的行共享 b 链，
//   a0/a1 对齐到同一 κ 后与同一 b 样本相乘（行 i 多延迟 1 拍），和式不变。
//
// 打包（推导见 ROUND3_INTEGRATION.md）：
//   A 口 27b：Ã = ã1·2^18 + ã0，ã = a + 128（无符号偏置场；符号位取反即 +128，
//             纯线网零逻辑）。不能直接拼有符号 a：低场符号位是内部位，位向量
//             值会多出 256·[a0<0]·b 的污染项，混进两场无法分离。
//   B 口 18b：共享 b（符号扩展）。
//   Ã·b = [a1·b·2^18 + a0·b] + 128·b·(2^18 + 1)     （偏置项）
//   低场 win[17:0] = Σ4(ã0·b)，高场 win[35:18] = Σ4(ã1·b) + floor——两场互不
//   重叠（|Σ4(ã·b)| ≤ 4·255·128 = 130560 ⊂ s18，余量 512）。
// 窗口累加：DSP PCRE 只累 4 个有效积（场不溢出），每 4 拍抽取到外部累加器；
//   残窗（<4 积）由 flush 脉冲收尾。抽取校正：
//   acc0 += win[17:0] − (sb<<<7)                       sb = 窗口 Σb（|·|≤512）
//   acc1 += win[35:18] − (sb<<<7) + win[17]            win[17] = floor 的 −1 修复
//   （sb 加法器 11b；校正项作 3 输入加法器的小操作数，进位链吸收）
// 位精确：与 2 个独立 ae_pe 在合法域（K ≤ 4096 ⇒ |acc| ≤ 2^26 ⊂ s28）逐位一致；
//   ae_pe 的 32b 回绕在该域内永不触发，28b 累加器等价（tb_pe_pack.sv 对拍，
//   含 ±2^26 极值 / 混符号窗口 / 稀疏有效 / 残窗用例）。
`ifndef AE_PE_PACK_SV
`define AE_PE_PACK_SV
module ae_pe_pack (
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

  // Ã = ã1·2^18 + ã0（正数，bit26=0），27b 有符号
  logic signed [26:0] ap;
  assign ap = {1'b0, at1, 10'd0, at0};
  logic signed [17:0] bx;
  assign bx = {{10{b_in[7]}}, b_in};
  assign en = av_in & bv_in;

  // DSP48E2：27×18 有符号乘 + 48b PCRE 窗口累加（use_dsp 同 ae_pe 三坑教训）
  (* use_dsp = "yes" *) logic signed [47:0] win;
  logic        [WLOG-1:0] wc;              // 窗口内已积个数
  logic                   full;            // 本拍 win 已含完整 4 积（可抽取）
  logic signed [10:0]    sb;               // 窗口 Σb（|·| ≤ 512）
  logic signed [17:0]    corr;             // −(sb <<< 7)：偏置校正项
  logic signed [18:0]    hi_f;             // 高场 + floor 修复（全有符号运算）

  assign corr = -(sb <<< 7);
  assign hi_f = $signed(win[35:18]) + (win[17] ? 19'sd1 : 19'sd0);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a0_r <= '0; a1_r <= '0; b_r <= '0; av_r <= 1'b0; bv_r <= 1'b0;
      win <= '0; wc <= '0; full <= 1'b0; sb <= '0;
      acc0 <= '0; acc1 <= '0;
    end else begin
      a0_r <= a0_in;  a1_r <= a1_in;  b_r <= b_in;
      av_r <= av_in;  bv_r <= bv_in;
      full <= 1'b0;
      if (clr) begin
        win <= '0; wc <= '0; sb <= '0;
        acc0 <= '0; acc1 <= '0;
      end else begin
        if (en) begin
          if (wc == WIN-1) begin wc <= '0; full <= 1'b1; end
          else wc <= wc + 1'b1;
          // wc==0 即新窗口首个积：sb 从 b 起算（e4 拍 sb 恰为 Σ4，供抽取拍用）
          if (wc == '0) sb <= b_in;
          else          sb <= sb + b_in;
        end
        // 窗口累加/翻窗（full && en 时以新积起步，背靠背不断流）。
        // 全次态只允许出现一次乘法：写两次（累加分支 + 翻窗分支各一）综合器会
        // 生成两颗 DSP（win2=C+A*B 与 win3=A*B，两轮实测的根源）。现在的形态
        // 是 DSP 原生的 "(0 or C)+(A*B or 0)" 模式：C 口 mux 承担翻窗清零，
        // 乘积门控承担 en。
        win <= (flush && !en) ? 48'sd0
             : (full ? 48'sd0 : win) + (en ? ap * bx : 48'sd0);
        // 抽取：低场 −偏置 = Σ4(a0·b)；高场 −偏置 + win[17]（floor 修复）。
        // 两条首版教训：① {27'd0, win[17]} 拼接是无符号数，会把整条加法拉进
        // 无符号语境、符号扩展失效（acc1 差 k·2^18 的根源）——改用全有符号
        // 中间量 hi_f；② flush 必须有残窗守卫：整窗抽完后 win 已清零但 sb 仍
        // 是旧窗口值，再抽会把 stale corr 白加进 acc（差 −72·128 的根源）。
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
