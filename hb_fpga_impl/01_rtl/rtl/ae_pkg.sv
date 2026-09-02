// ae_pkg.sv — 全模型 INT8 加速器公共参数与序列描述符
// ZCU104 (XCZU7EV-2) · 16x108 输出驻留脉冲阵列 · 1728 DSP48E2 · 250 MHz（综合档）
// 仿真冒烟档由 ae_core 参数覆盖（COLS=12 等）。
// 存储布局纪律：所有激活以 k-major 存 CTX（16 lane-bank，lane = m mod 16，
//   bank 内地址 = (m div 16)*KPAD + k）；B 矩阵一律先重排进 WRAM（COLS lane-bank，
//   lane = 列组内 j，地址 = 归约维 k）——OP_COPY 负责 CTX 16-lane -> WRAM COLS-lane。
`ifndef AE_PKG_SV
`define AE_PKG_SV

package ae_pkg;

  parameter int ROWS = 16;            // M 维 tile 行数（token）
  parameter int COLS_DFLT = 108;      // N 维列组宽度（synth 档；sim 覆盖为 12）
  parameter int CTX_BANKS = 16;       // 激活池 lane bank 数
  parameter int CTX_WORDS = 131072;   // 每 bank 深度（64 URAM，共 1 MB）
  parameter int W_WORDS   = 4096;     // WRAM 每 bank 深度（归约维上限 4096）
  parameter int SEQ_N     = 2048;     // 序列表项数（全模型 ViT+LLM+head 单层展开）

  typedef enum logic [2:0] {
    TAG_CTX = 3'd0,
    TAG_W   = 3'd1
  } dma_tag_e;

  typedef enum logic [3:0] {
    OP_GEMM    = 4'd0,  // Y = requant(A·B)，B=WRAM
    OP_ATTN_S  = 4'd1,  // GEMM(S=Q·Kᵀ) + 两遍 softmax 原地 -> P(int8)
    OP_HOIST   = 4'd2,  // 一次性 K/V 投影写 CTX（驻留）并置不变位图 —— primitive 核心
    OP_COPY    = 4'd3,  // CTX -> WRAM 的 B 矩阵 108-lane 重排（S/PV 前置）
    OP_LOAD    = 4'd4,  // DMA: DDR -> 片上（TAG 由 b_src 选）
    OP_STORE   = 4'd5,  // DMA: CTX -> DDR
    OP_DONE    = 4'd15
  } op_e;

  // 序列描述符（256 bit，PS 经 AXI-Lite 写入 SEQ RAM；GEMM 列组循环在序列层展开）
  // 字段复用（不同 op 读同一字段的不同含义）：
  //   b_spad : GEMM = n_loc（本组列数）   / COPY = 源 16-lane 组步长
  //   rq_m   : GEMM = requant 乘子 Q8.8  / COPY = src_j0（全局起始列）
  //   dma_len 低 16 位：GEMM = j0（组全局列偏移）/ DMA = 字节数
  //   a_base 低 12 位：COPY = WRAM 目的基址（= 后续 GEMM 的 b_base）
  typedef struct packed {
    logic [3:0]  op;
    logic [2:0]  a_src;      // 0=CTX
    logic [2:0]  b_src;      // 0=WRAM 1=CTX（OP_COPY 源 / OP_LOAD 的 TAG）
    logic        sm_causal;  // OP_ATTN_S 因果掩码（j <= i 有效）
    logic        y_tr;       // 写回朝向：0=[n][m-lane] 1=转置 [m][n-lane]（V 给 PV）
    logic [15:0] m;          // token 数（行）
    logic [15:0] n;          // 全局输出宽（S 阶段 = K 的 token 数）
    logic [15:0] k;          // 归约维
    logic [19:0] a_base;     // CTX 字地址（COPY 时低 12 位 = WRAM 目的基址）
    logic [19:0] b_base;     // WRAM / CTX 字地址（OP_COPY 源 / DMA 目的）
    logic [19:0] y_base;     // CTX 目的（OP_STORE 源）
    logic [15:0] b_spad;     // GEMM n_loc / COPY 源步长
    logic [15:0] rq_m;       // GEMM requant 乘子 / COPY src_j0
    logic [7:0]  rq_s;       // requant 右移
    logic [3:0]  inv_idx;    // 不变位图索引（0xF = 非不变）
    logic [10:0] steps;      // denoise 循环步数
    logic        in_loop;
    logic        is_loop_end;
    logic [17:0] dma_len;    // DMA 字节数（LOAD 8 倍数 / STORE 16 倍数；最大 256KB）
    logic [31:0] dma_addr;   // DDR 地址
    logic [28:0] pad;        // 凑 256 bit（4+3+3+1+1+16*3+20*3+16*2+8+4+11+1+1+18+32+29=256）
  } seq_desc_t;

  parameter int DESC_W = 256;

endpackage : ae_pkg
`endif
