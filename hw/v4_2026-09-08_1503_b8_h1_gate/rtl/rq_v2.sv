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
  logic signed [PW-1:0] sum, sum_r;
  assign sum = phi_r + plo_r;

  logic [5:0] t;
  assign t = s[5:0] - 6'd8;      // s≥8 时 = s-8 ∈ [0,31]；本设计口径 t≤T_MAX

  generate
    if (T_MAX == 0) begin : g_nosh
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          phi_r <= '0; plo_r <= '0; sum_r <= '0; v1_r <= 1'b0; v2_r <= 1'b0;
        end else begin
          phi_r <= phi;  plo_r <= plo;
          v1_r  <= in_vld;
          sum_r <= sum;
          v2_r  <= v1_r;
        end
      end
    end else begin : g_barrel
      // t 与乘积同拍寄存（t_r）：移位量必须与被移的积流水对齐。v1 原版在 T1
      // 采当前 s —— s 逐 GEMM 静态时两者等价；本设计按逐向量正确实现。
      logic [5:0] t_r;
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          phi_r <= '0; plo_r <= '0; sum_r <= '0; t_r <= '0;
          v1_r <= 1'b0; v2_r <= 1'b0;
        end else begin
          phi_r <= phi;  plo_r <= plo;  t_r <= t;
          v1_r  <= in_vld;
          sum_r <= sum >>> t_r;
          v2_r  <= v1_r;
        end
      end
    end
  endgenerate

  // sat8：上/下溢检测代替宽比较器
  logic pos_ov, neg_ov;
  assign pos_ov = ~sum_r[PW-1] & (|sum_r[PW-2:7]);
  assign neg_ov =  sum_r[PW-1] & (~&sum_r[PW-2:7]);
  assign y = pos_ov ? 8'sd127 : neg_ov ? -8'sd128 : sum_r[7:0];
  assign out_vld = v2_r;
endmodule
`endif
