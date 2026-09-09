// ============================================================================
// rq_v2.sv — requant 二代核心（门 1）：m 字节拆分 + 移位吸收，精确等价改写
// ----------------------------------------------------------------------------
// v1 语义: y = sat8((x·m) >>> s)，m 为 Q8.8 有符号 16b，乘法 32×16 LUT + 48b 桶形移位。
// 精确等价（整数恒等式，对全部 x∈Z、mh∈Z、ml∈Z 成立，含负数）：
//     m = mh·2^8 + ml   (mh = m[15:8] 有符号, ml = m[7:0] 无符号)
//     (x·m) >>> (8+t) = ( x·mh + ((x·ml) >>> 8) ) >>> t      [证明见 ROUND2_MICRO.md]
// 即「先各乘 8b、低半积先截 8 位、再相加、再移 t」。两个 XW×8 窄乘 + 一次加法
// 替代 XW×16 宽乘；t=0（s=8）时桶形移位整个消失。
// 参数 : XW   x 位宽。27 = GEMM 累加器可证上界（|x| ≤ K·128·128 ≤ 2^26, K≤4096），
//             32 = 与 v1 完全同域（对照实测）。
//        T_MAX 支持的最大 t=s-8。0 = 只支持 s=8（无移位器）；39 = s∈[8,47] 全覆盖。
// 口径 : s≥8 且 t≤T_MAX 时与 v1 逐位一致（TB 对拍）。s<8 / s>8+T_MAX 无定义（v1
//        驱动器只发 s=8，见 gen_vectors.py RQ 表）。
// 流水 : 2 拍，与 v1 延迟逐拍相同（T0 乘法寄存，T1 加/移寄存，y 组合出）。
// ----------------------------------------------------------------------------
// ★ 24_pcw_rtn（2026-09-04）：RTN 就近舍入（research_w8a8_error/REPORT.md §4 方案 A）
//   rn_en=1 时 y = sat8((sum + rn) >>> t)，rn = 2^(t-1) = 2^(s-9)（t≥1）。
//   与 v1 级软件公式 y = sat8((x·m + 2^(s-1)) >>> s)（s≥9）逐位等价：
//     sum = floor((x·m)/2^8) 精确成立（>>>8 即 floor 除），写 p = 256·sum + r，
//     r∈[0,256)，则 floor((p + 2^(s-1))/2^s) = floor((256·(sum + 2^(s-9)) + r)/2^s)
//     = (sum + 2^(s-9) + r/256) >>> (s-8)，而 s≥9 时 2^(s-9) 与 sum 均为整数、
//     r/256 < 1，故取整后与 (sum + rn) >>> t 完全相等。
//   s=8（t=0）：2^(s-9) 非整数 → 本核退化为截断（rn=0），与 REPORT §4「s=8 退化
//   为截断或把该列编到 s=9」的口径一致；编译器契约 = RTG 列一律编 s≥9。
//   rn_en=0（floor 模式）：rn 恒 0，与 R3C 基线逐位一致（tb_rq 60k 向量回归锚）。
//   位宽论证（rn 加入后不需再加宽容器）：|sum| = |x·mh + ((x·ml)>>>8)| ≤
//   2^(XW-1)·2^7 + 2^(XW-1)·(255/256 + 1) < 2^(XW+6)·1.032 < 2^(PW-1)；rn 经钳位
//   ≤ 2^(PW-1)，故 sum + rn ∈ (-2^(PW-1), 2^PW) —— 恰装入 PW+1 位有符号，不回卷。
//   t-1 ≥ PW-1（XW=27 时 t≥35）时 rn 钳在 2^(PW-1)：|sum| < 2^(PW-1) 保证
//   (sum + 2^(t-1)) >>> t 与 (sum + 2^(PW-1)) >>> t 同为 0，钳位无损。
//   rn 由 s 译码、T0 与 t_r 同拍寄存（不在移位那拍组合输入上拉加法器）；
//   T_MAX=0 配置（s=8 专用）不支持 RTN（t≡0 → rn≡0），rn_en 被忽略。
// ============================================================================
`ifndef RQ_V2_SV
`define RQ_V2_SV
module rq_v2 #(
  parameter int XW    = 27,
  parameter int T_MAX = 0
)(
  input  logic              clk,
  input  logic              rst_n,
  input  logic              in_vld,
  input  logic signed [XW-1:0] x,
  input  logic signed [15:0] m,
  input  logic        [7:0]  s,
  input  logic               rn_en,     // 0=floor（R3C 位精确）；1=RTN 就近舍入
  output logic              out_vld,
  output logic signed [7:0]  y
);
  localparam int PW = XW + 8;   // 乘积位宽

  logic signed [7:0]  mh;
  logic        [7:0]  ml;
  assign mh = m[15:8];
  assign ml = m[7:0];

  // 乘积专用具名线网（防 retiming 丢 use_dsp 属性 —— 三坑之一）
  (* use_dsp = "no" *) logic signed [PW-1:0] phi;    // x·mh  （有符号×有符号）
  (* use_dsp = "no" *) logic signed [PW-1:0] plo_f;  // x·ml  （有符号×无符号）
  assign phi   = x * mh;
  assign plo_f = x * $signed({1'b0, ml});
  // (x·ml)>>>8：常数截取 = 算术右移，位切片即精确 floor
  logic signed [XW-1:0] plo;
  assign plo = plo_f[PW-1:8];

  // T0 寄存
  logic signed [PW-1:0] phi_r;
  logic signed [XW-1:0] plo_r;
  logic v1_r, v2_r;
  // ★ PW+1 位和容器：sext(sum) + rn 不回卷（见头部位宽论证）；rn=0 时与
  //   旧版 PW 位容器逐位等价（sext 后 sat8 检测位一一对应）
  logic signed [PW:0] sum, sum_r;
  assign sum = {{1{phi_r[PW-1]}}, phi_r} + {{(PW+1-XW){plo_r[XW-1]}}, plo_r};

  logic [5:0] t;
  assign t = s[5:0] - 6'd8;      // s≥8 时 = s-8 ∈ [0,31]；本设计口径 t≤T_MAX

  // ★ RTN 舍入常数（T0 组合译码，下一拍随乘积一起寄存进 rn_r）
  logic [PW-1:0] rn_c;
  always_comb begin
    rn_c = '0;
    if (rn_en && (t >= 6'd1)) begin
      if (t >= PW) rn_c[PW-1] = 1'b1;      // 钳位：t-1 ≥ PW-1，无损（见头部论证）
      else         rn_c[t-1]  = 1'b1;      // rn = 2^(t-1) = 2^(s-9)
    end
  end

  generate
    if (T_MAX == 0) begin : g_nosh
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          phi_r <= '0; plo_r <= '0; sum_r <= '0; v1_r <= 1'b0; v2_r <= 1'b0;
        end else begin
          phi_r <= phi;  plo_r <= plo;
          v1_r  <= in_vld;
          sum_r <= sum;                 // t≡0（s=8）→ rn≡0，RTN 不可用（口径见头部）
          v2_r  <= v1_r;
        end
      end
    end else begin : g_barrel
      // t 与乘积同拍寄存（t_r）：移位量必须与被移的积流水对齐。v1 原版在 T1
      // 采当前 s —— s 逐 GEMM 静态时两者等价；本设计按逐向量正确实现。
      // rn_r 同拍流水（照抄 t_r 的对齐处理），T1 的加法器吃的是寄存器输出。
      logic [5:0] t_r;
      logic [PW-1:0] rn_r;
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          phi_r <= '0; plo_r <= '0; sum_r <= '0; t_r <= '0; rn_r <= '0;
          v1_r <= 1'b0; v2_r <= 1'b0;
        end else begin
          phi_r <= phi;  plo_r <= plo;  t_r <= t;  rn_r <= rn_c;
          v1_r  <= in_vld;
          // $signed 使加法保持有符号（裸拼接是无符号，会把 >>> 降级成逻辑移位）
          sum_r <= (sum + $signed({1'b0, rn_r})) >>> t_r;
          v2_r  <= v1_r;
        end
      end
    end
  endgenerate

  // sat8：上/下溢检测代替宽比较器（PW+1 位口径：符号位 sum_r[PW]，
  // 其下 PW 位与旧版 PW 位容器同位 —— rn=0 时两口径逐位等价）
  logic pos_ov, neg_ov;
  assign pos_ov = ~sum_r[PW] & (|sum_r[PW-1:7]);
  assign neg_ov =  sum_r[PW] & (~&sum_r[PW-1:7]);
  assign y = pos_ov ? 8'sd127 : neg_ov ? -8'sd128 : sum_r[7:0];
  assign out_vld = v2_r;
endmodule
`endif
