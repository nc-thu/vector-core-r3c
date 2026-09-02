# -*- coding: utf-8 -*-
"""emit_big.py — 大负载吞吐基准向量（COLS=108 全参数档），2026-08-30 新增。

目标：~100 万拍的 GEMM 密集序列，测 iverilog vs Verilator 的 cycles/秒（墙钟），
外推真实模型级负载（约 5400 万拍/次推理）的仿真时长。

负载形状（与任务书的建议形状的差别只在一处）：
  50 条 m=1024/k=2048 的 GEMM 在本设计里放不下：
    - CTX 装不下 A：m=1024 行 × k=2048 需要 64×2048=131072 字 = 整个 CTX，
      Y 写回就没有落点（本表无 golden 对拍，但读写重叠会让数据无意义）；
    - 更硬的约束是描述符 dma_len 只有 18 位（≤262143B），A 的 2MB 装载要拆 8 条以上。
  所以取 k=1024（A 占 65536 字，恰好半个 CTX，Y 放高半区，无重叠）：
    10 组 [LOAD_W + GEMM] + 前置 8 条 LOAD_CTX + 收尾 1 条 STORE，
    按周期模型估算 ≈ 117 万拍（脚本末尾打印分项估算，可与 RTL 计数器对账）。

产出（写在本目录）：seq.mem / ddr_init.mem / exp2_lut.mem / big_report.json
本表不做数值对拍（没有 golden），只看 cycles 与墙钟；两工具跑同一 seq/ddr
即可互为周期数对照（cycles 计数必须一致，DDR dump 也顺带可比）。
"""
import numpy as np
import json, os, sys

SIM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SIM, '..', 'sim108'))
from gen_vectors_108 import desc, EXP, wmem   # 复用描述符打包 / LUT / 写盘

COLS      = 108
CTX_WORDS = 131072
W_WORDS   = 4096
DDR_SIZE  = 0x800000
M   = 1024                       # GEMM 行数（64 个 m-tile）
K   = 1024                       # GEMM 缩减深度 = 权重 k 行数
N   = COLS                       # 输出列（1 个列组）
PAIRS = 10                       # [LOAD_W + GEMM] 组数

M16  = ((M + 15) // 16) * 16
A_BASE = 0                       # CTX：A 占 M16/16*K = 65536 字
Y_BASE = M16 // 16 * K           # CTX：Y 占 M16/16*N = 6912 字（无重叠）
assert Y_BASE + (M16 // 16) * N <= CTX_WORDS

DDR_A     = 0x10000              # 激活 1MB
DDR_W0    = 0x110000             # 权重槽基址（槽间距 0x20000 > K*COLS=110592B）
DDR_OUT   = 0x400000
A_BYTES   = M16 * K              # 1048576
W_BYTES   = K * COLS             # 110592
OUT_BYTES = M16 * N
assert DDR_A + A_BYTES <= 0x110000
assert DDR_OUT + OUT_BYTES <= DDR_SIZE
CHUNK = 0x20000                  # LOAD_CTX 分块（dma_len 18 位 < 0x40000）
assert A_BYTES % CHUNK == 0 and CHUNK <= 0x3FFFF

rng = np.random.default_rng(20260830)
X = rng.integers(-24, 25, size=(M16, K)).astype(np.int64)   # 激活 [m][k]
Ws = [rng.integers(-24, 25, size=(K, N)).astype(np.int64)   # 权重 [k][j]
      for _ in range(PAIRS)]

ddr = np.zeros(DDR_SIZE, dtype=np.int64)
# 激活 [k][m16] k-major（DMA TAG_CTX 路由契约）：byte b -> lane=b%16, addr=b//16
b = 0
for k in range(K):
    for m in range(M16):
        ddr[DDR_A + b] = X[m, k]
        b += 1
# 权重 [k][j] 行主序按 COLS 补齐（TAG_W 契约：每 k 恰好 COLS 字节）
for i, W in enumerate(Ws):
    for k in range(K):
        for j in range(COLS):
            ddr[DDR_W0 + i * 0x20000 + k * COLS + j] = W[k, j]

seq = []
for c in range(A_BYTES // CHUNK):          # 8 条 LOAD_CTX
    seq.append(desc(op=4, b_src=0, b_base=c * (CHUNK // 16),
                    dma_len=CHUNK, dma_addr=DDR_A + c * CHUNK))
for i in range(PAIRS):                     # 10 组 LOAD_W + GEMM
    seq.append(desc(op=4, b_src=1, b_base=0,
                    dma_len=W_BYTES, dma_addr=DDR_W0 + i * 0x20000))
    seq.append(desc(op=0, m=M, n=N, b_spad=N, j0=0, k=K,
                    a_base=A_BASE, b_base=0, y_base=Y_BASE,
                    rq_m=64, rq_s=8))
seq.append(desc(op=5, y_base=Y_BASE, dma_len=OUT_BYTES, dma_addr=DDR_OUT))
seq.append(desc(op=15))                    # OP_DONE

# ---------------- 周期模型估算（常数取自 gem_cycles.py，对账用） ----------------
def waitd(cols):  return 16 + cols + 3
def gemm_cyc(m, k, n_loc):
    mt = (m + 15) // 16
    tile = 1 + (k + 2) + waitd(COLS) + 64 + 2 + 2 + n_loc + 1
    return 2 + 2 + mt * tile
def load_w_cyc(k, cols):
    beats = k * cols // 8
    xings = beats - beats // (cols // 8)   # 近似：每 cols/8 拍一次跨界
    import math
    return beats + xings + math.ceil(k * cols / 2048) * 2 + 5
def load_ctx_cyc(nb):  return nb // 8 + (nb // 2048 + 1) * 2 + 5
def store_cyc(nb):     return (nb + 15) // 16 * 5 + 5

est = (A_BYTES // CHUNK) * load_ctx_cyc(CHUNK) \
    + PAIRS * (load_w_cyc(K, COLS) + gemm_cyc(M, K, N)) + store_cyc(OUT_BYTES)

wmem('exp2_lut.mem', EXP, 13)
wmem('seq.mem', seq, 256)
wmem('ddr_init.mem', ddr, 8)
# wmem 写到 gen_vectors_108 所在目录（其内部 SIM 常量），搬回本目录，避免覆盖
# sim108/ 的冒烟档向量（2026-08-31 踩坑：不搬会把 sim108 的 seq/ddr 换成大负载档）
for _n in ('exp2_lut.mem', 'seq.mem', 'ddr_init.mem'):
    _src = os.path.join(SIM, '..', 'sim108', _n)
    if os.path.exists(_src):
        os.replace(_src, os.path.join(SIM, _n))
report = dict(cols=COLS, ctx_words=CTX_WORDS, w_words=W_WORDS, ddr_size=DDR_SIZE,
              M=M, K=K, N=N, pairs=PAIRS, seq_len=len(seq),
              est_cycles_total=est,
              est=dict(load_ctx=(A_BYTES // CHUNK) * load_ctx_cyc(CHUNK),
                       per_pair_w=load_w_cyc(K, COLS),
                       per_pair_gemm=gemm_cyc(M, K, N),
                       store=store_cyc(OUT_BYTES)))
with open(os.path.join(SIM, 'big_report.json'), 'w') as f:
    json.dump(report, f, indent=2)
print(f"[big] seq={len(seq)} 条  估算总周期 ≈ {est} ({est/1e6:.2f}M)")
print(f"[big]   LOAD_CTX x{A_BYTES//CHUNK} ≈ {(A_BYTES//CHUNK)*load_ctx_cyc(CHUNK)}"
      f"  LOAD_W ≈ {load_w_cyc(K,COLS)}  GEMM ≈ {gemm_cyc(M,K,N)}"
      f"  STORE ≈ {store_cyc(OUT_BYTES)}")
print(f"[big] 写出 seq.mem / ddr_init.mem / exp2_lut.mem")
