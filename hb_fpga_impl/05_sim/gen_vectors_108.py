# -*- coding: utf-8 -*-
"""gen_vectors_108.py — gen_vectors.py 的全参数档（COLS=108）版本，2026-08-30 新增。

与 gen_vectors.py 的全部差别（其余逐行一致，黄金模型本身与列数无关）：
  1) 档位常数：COLS=108 / CTX_WORDS=131072 / W_WORDS=4096 / SEQ_N=2048 /
     DDR_SIZE=0x800000（8MB，与 ae_core 综合默认参数一致）
  2) DDR 权重槽间距 0x100 -> 0x800：d=8 行 × 108 列 = 864 字节 > 256，
     原间距放不下一个权重矩阵（gen_vectors.py 的 0x100 栅栏断言随之改 0x800）
  3) --frames 默认 1：帧二（不复位帧链）是 COLS=12 冒烟的回归项，
     全参数档只做 REF/PRIM 单帧 golden 对拍
用法：
  python gen_vectors_108.py              # 全参数档冒烟向量（seq/ddr_init/expected_*）
  python gen_vectors_108.py --case tail  # 尾部边界（MQ=17/MC=13/DK=25）
产出文件名与 gen_vectors.py 相同（在本目录内使用，不要与 COLS=12 的目录混放）。
"""
import numpy as np
import json, os, argparse

SIM = os.path.dirname(os.path.abspath(__file__))
COLS = 108         # 全参数档阵列宽
CTX_WORDS = 131072
W_WORDS = 4096
DDR_SIZE = 0x800000
SEQ_N = 2048

# ---------------- 基础算术 ----------------
def sat8(x):
    return np.clip(x, -128, 127).astype(np.int64)

def requant(acc, m, s):
    prod = acc.astype(np.int64) * np.int64(m)
    shr = prod >> np.int64(s)          # 算术右移
    return sat8(shr)

EXP = [int(np.floor((2.0 ** (-d / 16.0)) * 4096 + 0.5)) for d in range(129)]

def softmax_rows(S, n_cols, causal):
    """S: [rows, n_cols] int8；原位返回 P（int8）"""
    P = np.zeros_like(S)
    m = S.shape[0]
    for i in range(m):
        vlen = min(i + 1, n_cols) if causal else n_cols
        row = S[i, :vlen].astype(np.int64)
        mx = int(row.max())
        e = np.array([EXP[min(mx - int(v), 128)] for v in row], dtype=np.int64)
        se = int(e.sum())
        quo = (127 << 30) // se                       # floor 除法
        p = (e * quo) >> 30
        p = np.minimum(p, 127)
        P[i, :vlen] = p.astype(np.int8)
    return P

# ---------------- CTX / WRAM / DDR 容器 ----------------
class CTX:
    def __init__(self):
        self.b = np.zeros((16, CTX_WORDS), dtype=np.int64)
    def wr_act(self, base, kpad, X):
        """X: [M,K] int -> lane = m mod16, addr = base + (m div16)*kpad + k"""
        M, K = X.shape
        for m in range(M):
            for k in range(K):
                self.b[m % 16, base + (m // 16) * kpad + k] = X[m, k]
    def rd_act(self, base, kpad, M, K):
        X = np.zeros((M, K), dtype=np.int64)
        for m in range(M):
            for k in range(K):
                X[m, k] = self.b[m % 16, base + (m // 16) * kpad + k]
        return X
    def wr_norm(self, base, n, Y, Mvalid, j0=0):
        """普通写回: addr = base + mt*n + j0 + col（全局列），lane = m mod 16; Y [M16pad, N] 局部列"""
        M16, N = Y.shape
        for i in range(M16):
            if i >= Mvalid: continue
            for c in range(N):
                self.b[i % 16, base + (i // 16) * n + j0 + c] = Y[i, c]
    def rd_norm(self, base, n, M, N):
        Y = np.zeros((M, N), dtype=np.int64)
        for i in range(M):
            for c in range(N):
                Y[i, c] = self.b[i % 16, base + (i // 16) * n + c]
        return Y
    def wr_tr(self, base, m16, Y, Nvalid, Mvalid, j0=0, nglob=0):
        """转置写回: 全局列 c = j0+局部列; addr = base + (c div16)*m16 + m, lane = c mod16"""
        M16, Nloc = Y.shape
        for i in range(M16):
            if i >= Mvalid: continue
            for lc in range(Nloc):
                c = j0 + lc
                if c >= nglob: continue
                self.b[c % 16, base + (c // 16) * m16 + i] = Y[i, lc]
    def rd_tr(self, base, m16, M16, Nglob):
        """读转置布局 [M16 x Nglob]（PV 的 B = V: B[k=m][j=f]）"""
        Y = np.zeros((M16, Nglob), dtype=np.int64)
        for f in range(Nglob):
            for m in range(M16):
                Y[m, f] = self.b[f % 16, base + (f // 16) * m16 + m]
        return Y

# ---------------- 描述符 ----------------
def desc(op=0, a_src=0, b_src=0, sm_causal=0, y_tr=0, m=0, n=0, k=0,
         a_base=0, b_base=0, y_base=0, b_spad=0, rq_m=0, rq_s=0,
         inv_idx=0xF, steps=0, in_loop=0, is_loop_end=0,
         dma_len=0, dma_addr=0, j0=0):
    # inv 契约：不变槽索引只允许出现在 GEMM 族（OP_GEMM/ATTN_S/HOIST）
    assert inv_idx == 0xF or op in (0, 1, 2), \
        f"inv_idx={inv_idx} 只能配 GEMM 族 op，当前 op={op}"
    v = 0
    v |= op << 252
    v |= a_src << 249
    v |= b_src << 246
    v |= sm_causal << 245
    v |= y_tr << 244
    v |= m << 228
    v |= n << 212
    v |= k << 196
    v |= a_base << 176
    v |= b_base << 156
    v |= y_base << 136
    v |= b_spad << 120          # GEMM n_loc / COPY spad
    v |= (rq_m & 0xFFFF) << 104 # GEMM requant / COPY src_j0
    v |= rq_s << 96
    v |= inv_idx << 92
    v |= steps << 81
    v |= in_loop << 80
    v |= is_loop_end << 79
    v |= dma_len << 61
    v |= dma_addr << 29
    v |= (j0 & 0xFFFF) << 62    # 复用 dma_len 字段区间 [77:62]
    assert v < (1 << 256)
    return v

# ---------------- 工作负载构建 ----------------
def build(mq=18, mc=16, dk=20, dv=8, d=8, steps=2, frames=1, seed=20260826):
    """与 gen_vectors.build 相同的冒烟负载形状（mq/mc/dk/dv/d/steps 默认值一致），
    仅档位常数与 DDR 权重槽间距不同（见文件头）。"""
    M16C = ((mc + 15) // 16) * 16
    M16Q = ((mq + 15) // 16) * 16

    # ---- CTX 布局（与 gen_vectors 相同——冒烟负载很小，131072 深度绰绰有余） ----
    A_XCTX, A_XT = 0, 16
    A_KCTX, A_VCTX = 64, 96
    A_K2CT, A_V2CT = 128, 144
    A_K1T, A_V1T, A_Q1T = 176, 192, 224
    A_S1T, A_O1T = 256, 292
    A_S2T, A_O2T = 308, 340
    A_FT = 356
    regions = [                      # (名, base, end)——写回真实占用
        ('XCTX', A_XCTX, A_XCTX + d), ('XT', A_XT, A_XT + d),
        ('KCTX', A_KCTX, A_KCTX + M16C // 16 * dk),
        ('VCTX', A_VCTX, A_VCTX + M16C),
        ('K2CT', A_K2CT, A_K2CT + M16C), ('V2CT', A_V2CT, A_V2CT + M16C),
        ('K1T', A_K1T, A_K1T + M16Q), ('V1T', A_V1T, A_V1T + M16Q),
        ('Q1T', A_Q1T, A_Q1T + M16Q // 16 * dv),
        ('S1T', A_S1T, A_S1T + M16Q // 16 * mq),
        ('O1T', A_O1T, A_O1T + M16Q // 16 * dv),
        ('S2T', A_S2T, A_S2T + M16Q // 16 * mc),
        ('O2T', A_O2T, A_O2T + M16Q // 16 * dv),
        ('FT', A_FT, A_FT + M16Q // 16 * COLS),
    ]
    for name, _, end in regions:
        assert end <= CTX_WORDS, f"CTX 溢出：{name} end={end}"
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            n1, b1, e1 = regions[i]
            n2, b2, e2 = regions[j]
            if min(e1, e2) > max(b1, b2):
                assert {n1, n2} == {'K1T', 'V1T'} and d <= 16, \
                    f"意外 CTX 重叠 {n1}/{n2}（K1T/V1T 为文档化例外）"

    # ---- DDR 布局：激活 0x1000/0x1100；权重槽 0x2000+slot*0x800（★唯一布局改动：
    #      108 列 × d=8 行 = 864B/槽 > 256B，间距从 0x100 放大到 0x800）----
    DDR_XCTX, DDR_XT = 0x1000, 0x1100
    DDR_OUT = 0x8000
    assert d * M16C <= 0x100 and d * M16Q <= 0x100, "激活窗口越过 0x100 栅栏"
    slot = 0
    def next_slot():
        nonlocal slot
        a = 0x2000 + slot * 0x800
        slot += 1
        return a
    W_SLOTS = {}                     # (矩阵, 组号) -> DDR 地址
    for g in range((dk + COLS - 1) // COLS):
        W_SLOTS[('K', g)] = next_slot()
    for mat in ('V', 'K1', 'V1', 'Q1', 'K2C', 'V2C', 'F'):
        W_SLOTS[(mat, 0)] = next_slot()
    assert d * COLS <= 0x800, "权重槽 0x800 栅栏"
    assert max(W_SLOTS.values()) + d * COLS <= DDR_OUT, "权重区侵入输出区"
    assert DDR_OUT + M16Q * COLS <= DDR_SIZE, "DDR 溢出"

    # ---- 随机数据（抽取顺序即 gen_vectors 历史顺序；seed 相同则重叠字段同值） ----
    rng = np.random.default_rng(seed)
    def rint8(*shape, lo=-24, hi=24):
        return rng.integers(lo, hi + 1, size=shape).astype(np.int64)

    X_ctx = rint8(M16C, d)           # context 激活（含 pad 行，pad 不被读）
    X_q   = rint8(M16Q, d)
    W_K   = rint8(d, dk)             # [k][j] 行主序 = DDR 布局
    W_V   = rint8(d, dv)
    W_K2C = rint8(d, dv)
    W_V2C = rint8(d, dv)
    W_K1  = rint8(d, dv)
    W_V1  = rint8(d, dv)
    W_Q1  = rint8(d, dv)
    W_F   = rint8(d, COLS)
    X_ctx2 = rint8(M16C, d) if frames >= 2 else None
    X_q2   = rint8(M16Q, d) if frames >= 2 else None

    # ---- DDR 初始映像 ----
    ddr = np.zeros(DDR_SIZE, dtype=np.int64)
    def wr_act_ddr(addr, X):
        """激活 [k][m16] k-major：byte b -> lane = b mod16, ctx addr = base + b div16"""
        M16, K = X.shape
        b = 0
        for k in range(K):
            for m in range(M16):
                ddr[addr + b] = X[m, k]
                b += 1
    def wr_w_ddr(addr, W):
        """权重 [k][j] 行主序，按 COLS 列补齐（DMA 路由契约：每 k 恰好 COLS 字节）"""
        K, N = W.shape
        for k in range(K):
            for j in range(COLS):
                ddr[addr + k * COLS + j] = W[k, j] if j < N else 0

    wr_act_ddr(DDR_XCTX, X_ctx)
    wr_act_ddr(DDR_XT, X_q)
    for g, j0 in enumerate(range(0, dk, COLS)):
        wr_w_ddr(W_SLOTS[('K', g)], W_K[:, j0:j0 + COLS])
    for mat, W in [('V', W_V), ('K1', W_K1), ('V1', W_V1), ('Q1', W_Q1),
                   ('K2C', W_K2C), ('V2C', W_V2C), ('F', W_F)]:
        wr_w_ddr(W_SLOTS[(mat, 0)], W)
    ddr2 = None
    if frames >= 2:                  # 帧二：仅激活换新，权重同帧一
        ddr2 = ddr.copy()
        b = 0
        for k in range(d):
            for m in range(M16C):
                ddr2[DDR_XCTX + b] = X_ctx2[m, k]
                b += 1
        b = 0
        for k in range(d):
            for m in range(M16Q):
                ddr2[DDR_XT + b] = X_q2[m, k]
                b += 1

    # ---- requant 参数（与 gen_vectors 相同，故意含饱和情形） ----
    RQ = {
        'kc':  (64, 8), 'vc':  (48, 8), 'k2c': (40, 8), 'v2c': (40, 8),
        'k1':  (56, 8), 'v1':  (56, 8), 'q1':  (56, 8),
        's1':  (16, 8), 's2':  (16, 8),
        'pv1': (32, 8), 'pv2': (32, 8), 'f': (96, 8),
    }

    # ---- 序列表 ----
    seq = []
    def D_LOAD(ddr_addr, nbytes, tag, ctx_base, **kw):
        seq.append(desc(op=4, b_src=tag, b_base=ctx_base,
                        dma_len=nbytes, dma_addr=ddr_addr, **kw))
    def D_STORE(ddr_addr, nbytes, src_base, **kw):
        seq.append(desc(op=5, b_src=0, y_base=src_base,
                        dma_len=nbytes, dma_addr=ddr_addr, **kw))
    def D_COPY(k_rows, j_cols, src_base, spad, src_j0, wr_base, **kw):
        seq.append(desc(op=3, k=k_rows, n=j_cols, b_base=src_base, b_spad=spad,
                        rq_m=src_j0, a_base=wr_base, **kw))
    def D_GEMM(m, n, n_loc, j0, k, a_base, b_base, y_base, rq, y_tr=0,
               op=0, **kw):
        seq.append(desc(op=op, m=m, n=n, k=k, a_base=a_base, b_base=b_base,
                        y_base=y_base, b_spad=n_loc, rq_m=rq[0], rq_s=rq[1],
                        y_tr=y_tr, j0=j0, **kw))

    # ---- setup（循环外）：context K/V 投影，K 宽 DK 逐列组 ----
    D_LOAD(DDR_XCTX, d * M16C, 0, A_XCTX)
    D_LOAD(DDR_XT, d * M16Q, 0, A_XT)
    for j0 in range(0, dk, COLS):
        n_loc = min(COLS, dk - j0)
        D_LOAD(W_SLOTS[('K', j0 // COLS)], d * COLS, 1, 0)
        D_GEMM(mc, dk, n_loc, j0, d, A_XCTX, 0, A_KCTX, RQ['kc'])
    D_LOAD(W_SLOTS[('V', 0)], d * COLS, 1, 0)
    D_GEMM(mc, dv, dv, 0, d, A_XCTX, 0, A_VCTX, RQ['vc'], y_tr=1)

    # ---- denoise 循环体（steps 可参；HOIST 对 = context K2/V2 投影） ----
    def emit_loop_body(in_loop, is_end):
        # ★ step-invariant：context K2/V2（inv 0/1）
        D_LOAD(W_SLOTS[('K2C', 0)], d * COLS, 1, 0, in_loop=in_loop, is_loop_end=0)
        D_GEMM(mc, dv, dv, 0, d, A_XCTX, 0, A_K2CT, RQ['k2c'], op=2,
               inv_idx=0, steps=steps, in_loop=in_loop, is_loop_end=0)
        D_LOAD(W_SLOTS[('V2C', 0)], d * COLS, 1, 0, in_loop=in_loop, is_loop_end=0)
        D_GEMM(mc, dv, dv, 0, d, A_XCTX, 0, A_V2CT, RQ['v2c'], op=2, y_tr=1,
               inv_idx=1, steps=steps, in_loop=in_loop, is_loop_end=0)
        # 每步重算：Q1/K1/V1（step-dependent）
        D_LOAD(W_SLOTS[('Q1', 0)], d * COLS, 1, 0, in_loop=in_loop, is_loop_end=0)
        D_GEMM(mq, dv, dv, 0, d, A_XT, 0, A_Q1T, RQ['q1'],
               in_loop=in_loop, is_loop_end=0)
        D_LOAD(W_SLOTS[('K1', 0)], d * COLS, 1, 0, in_loop=in_loop, is_loop_end=0)
        D_GEMM(mq, dv, dv, 0, d, A_XT, 0, A_K1T, RQ['k1'],
               in_loop=in_loop, is_loop_end=0)
        D_LOAD(W_SLOTS[('V1', 0)], d * COLS, 1, 0, in_loop=in_loop, is_loop_end=0)
        D_GEMM(mq, dv, dv, 0, d, A_XT, 0, A_V1T, RQ['v1'], y_tr=1,
               in_loop=in_loop, is_loop_end=0)
        # 注意力 1（自注意，causal）：S1 = Q1·K1ᵀ [mq x mq]，逐列组
        def emit_attn(n_cols, k_src, a_s, y_s, rq):
            for j0 in range(0, n_cols, COLS):
                n_loc = min(COLS, n_cols - j0)
                last = j0 + COLS >= n_cols
                D_COPY(d, n_loc, k_src, dv, j0, 0, in_loop=in_loop, is_loop_end=0)
                D_GEMM(mq, n_cols, n_loc, j0, d, a_s, 0, y_s, rq,
                       op=1 if last else 0, sm_causal=1 if last else 0,
                       in_loop=in_loop, is_loop_end=0)  # 末组 GEMM 后接 softmax
        def emit_xattn(n_cols, k_src, a_s, y_s, rq):
            for j0 in range(0, n_cols, COLS):   # 交叉注意：非 causal
                n_loc = min(COLS, n_cols - j0)
                last = j0 + COLS >= n_cols
                D_COPY(d, n_loc, k_src, dv, j0, 0, in_loop=in_loop, is_loop_end=0)
                D_GEMM(mq, n_cols, n_loc, j0, d, a_s, 0, y_s, rq,
                       op=1 if last else 0, sm_causal=0,
                       in_loop=in_loop, is_loop_end=0)
        emit_attn(mq, A_K1T, A_Q1T, A_S1T, RQ['s1'])
        D_COPY(mq, dv, A_V1T, M16Q, 0, 0, in_loop=in_loop, is_loop_end=0)
        D_GEMM(mq, dv, dv, 0, mq, A_S1T, 0, A_O1T, RQ['pv1'],
               in_loop=in_loop, is_loop_end=0)
        # 注意力 2（交叉注意，使用 ★驻留 的 K2c/V2c）：S2 = Q1·K2cᵀ [mq x mc]
        emit_xattn(mc, A_K2CT, A_Q1T, A_S2T, RQ['s2'])
        D_COPY(mc, dv, A_V2CT, M16C, 0, 0, in_loop=in_loop, is_loop_end=0)
        D_GEMM(mq, dv, dv, 0, mc, A_S2T, 0, A_O2T, RQ['pv2'],
               in_loop=in_loop, is_loop_end=0)
        # 尾部投影 + 回写
        D_LOAD(W_SLOTS[('F', 0)], d * COLS, 1, 0, in_loop=in_loop, is_loop_end=0)
        D_GEMM(mq, COLS, COLS, 0, dv, A_O2T, 0, A_FT, RQ['f'],
               in_loop=in_loop, is_loop_end=0)
        D_STORE(DDR_OUT, M16Q * COLS, A_FT,
                in_loop=in_loop, is_loop_end=1 if is_end else 0, steps=steps)

    emit_loop_body(1, 1)
    seq.append(desc(op=15))
    assert len(seq) <= SEQ_N, f"序列 {len(seq)} 条超 SEQ_N={SEQ_N}"
    assert max(d, mq, mc) <= W_WORDS, "WRAM 深度不足"

    return dict(seq=seq, ddr=ddr, ddr2=ddr2, RQ=RQ, mq=mq, mc=mc, dk=dk, dv=dv,
                d=d, steps=steps, frames=frames, M16C=M16C, M16Q=M16Q,
                DDR_XCTX=DDR_XCTX, DDR_XT=DDR_XT)

# ---------------- 黄金执行（REF 与 PRIMITIVE 两种模式，可链帧）----------------
def run(wl, mode, ctx0=None, dram0=None, wram0=None):
    """执行 wl['seq']（与 gen_vectors.run 逐行一致）。"""
    SEQ = wl['seq']
    ctx = ctx0 if ctx0 is not None else CTX()
    dram = dram0.copy() if dram0 is not None else wl['ddr'].copy()
    wram = wram0.copy() if wram0 is not None else \
        np.zeros((COLS, W_WORDS), dtype=np.int64)
    bitmap = [False] * 16          # 硬件 T_IDLE 收 start 时清位图
    step = 0
    macs = 0
    skip_stages = 0
    skip_macs = 0
    pc, loop_start, loop_seen = 0, None, False
    n_exec = 0
    while True:
        d = SEQ[pc]
        op = (d >> 252) & 0xF
        n = (d >> 212) & 0xFFFF
        k = (d >> 196) & 0xFFFF
        m = (d >> 228) & 0xFFFF
        a_base = (d >> 176) & 0xFFFFF
        b_base = (d >> 156) & 0xFFFFF
        y_base = (d >> 136) & 0xFFFFF
        b_spad = (d >> 120) & 0xFFFF
        rq_m = (d >> 104) & 0xFFFF
        if rq_m >= 0x8000: rq_m -= 0x10000
        rq_s = (d >> 96) & 0xFF
        inv = (d >> 92) & 0xF
        steps = (d >> 81) & 0x7FF
        in_loop = (d >> 80) & 1
        is_end = (d >> 79) & 1
        dma_len = (d >> 61) & 0x3FFFF
        dma_addr = (d >> 29) & 0xFFFFFFFF
        j0 = (d >> 62) & 0xFFFF
        sm_causal = (d >> 245) & 1
        y_tr = (d >> 244) & 1
        b_src = (d >> 246) & 7
        n_loc = b_spad

        if op == 15:
            break
        assert inv == 0xF or op in (0, 1, 2), \
            f"pc={pc}: op={op} 携带 inv={inv}（违反编译器契约）"
        skip = (mode == 'PRIM' and inv != 0xF and in_loop and step > 0
                and op not in (15, 3, 4, 5) and bitmap[inv])
        gemmish = op in (0, 1, 2)

        if skip:
            skip_stages += 1
            skip_macs += m * n * k
        elif op == 4:  # LOAD
            if b_src == 0:  # CTX [k][m16]
                for b in range(dma_len):
                    ctx.b[b % 16, b_base + b // 16] = dram[dma_addr + b]
            else:          # W [k][j]
                wj, wk = 0, 0
                for b in range(dma_len):
                    wram[wj, b_base + wk] = dram[dma_addr + b]
                    wj += 1
                    if wj == COLS:
                        wj = 0; wk += 1
        elif op == 5:  # STORE
            for w in range(dma_len // 16):
                for half in range(2):
                    for q in range(8):
                        lane = half * 8 + q
                        dram[dma_addr + w * 16 + half * 8 + q] = ctx.b[lane, y_base + w]
        elif op == 3:  # COPY: CTX -> WRAM 重排
            src_j0 = rq_m
            for j in range(j_cols := ((d >> 212) & 0xFF)):
                gcol = src_j0 + j
                for kk in range(k):
                    wram[j, a_base + kk] = ctx.b[gcol % 16,
                                                 b_base + (gcol // 16) * b_spad + kk]
        elif gemmish:
            macs += ((m + 15) // 16) * 16 * COLS * k
            A = ctx.rd_act(a_base, k, m, k)          # [m x k]
            B = wram[:, b_base:b_base + k].T         # [k x COLS] -> 取 n_loc 列
            B = B[:, :n_loc]
            Y = requant(A @ B, rq_m, rq_s)           # [m x n_loc] int8
            m16 = ((m + 15) // 16) * 16
            Yp = np.zeros((m16, n_loc), dtype=np.int64)
            Yp[:m, :] = Y
            if y_tr:
                ctx.wr_tr(y_base, m16, Yp, n_loc, m, j0=j0, nglob=n)
            else:
                ctx.wr_norm(y_base, n, Yp, m, j0=j0)
            if op == 2 and mode == 'PRIM' and inv != 0xF:
                bitmap[inv] = True
            if op == 1:  # softmax（作用于 y_base 处 [m x n]）
                S = ctx.rd_norm(y_base, n, m, n).astype(np.int64)
                P = softmax_rows(S, n, sm_causal)
                ctx.wr_norm(y_base, n, P, m)
        else:
            assert False, f"未定义 op={op} @pc={pc}"
        n_exec += 1
        # 循环推进
        if in_loop and loop_start is None:
            loop_start = pc
        if is_end and in_loop and step + 1 < steps:
            step += 1
            pc = loop_start
        else:
            pc += 1
    return ctx, dram, wram, dict(macs=macs, skip_stages=skip_stages,
                                 skip_macs=skip_macs, n_exec=n_exec)

# ---------------- 主流程 ----------------
def wmem(name, arr, width):
    """与 gen_vectors.wmem 输出逐字节一致；8M 行的 DDR 映像改为分块拼接写出（快 ~20 倍）。"""
    nd = (width + 3) // 4
    mask = (1 << width) - 1
    path = os.path.join(SIM, name)
    with open(path, 'w') as f:
        CH = 1 << 16
        for i in range(0, len(arr), CH):
            chunk = arr[i:i + CH]
            f.write('\n'.join(f"{int(v) & mask:0{nd}X}" for v in chunk) + '\n')

def main():
    ap = argparse.ArgumentParser(description='ae 黄金模型 + 向量生成（COLS=108 全参数档）')
    ap.add_argument('--mq', type=int, default=18, help='query token 数')
    ap.add_argument('--mc', type=int, default=16, help='context token 数')
    ap.add_argument('--dk', type=int, default=20, help='K_ctx 特征数')
    ap.add_argument('--dv', type=int, default=8, help='V/投影特征数')
    ap.add_argument('--steps', type=int, default=2, help='去噪步数')
    ap.add_argument('--frames', type=int, default=1,
                    help='推理帧数（全参数档默认 1；2=含不复位帧链）')
    ap.add_argument('--case', choices=['default', 'tail'], default='default')
    ap.add_argument('--seed', type=int, default=20260826)
    args = ap.parse_args()
    if args.case == 'tail':       # 尾 tile 边界：M 差一 / n_loc=1 / 尾列组
        args.mq, args.mc, args.dk = 17, 13, 25

    wl = build(mq=args.mq, mc=args.mc, dk=args.dk, dv=args.dv,
               steps=args.steps, frames=args.frames, seed=args.seed)
    wmem('exp2_lut.mem', EXP, 13)
    wmem('seq.mem', wl['seq'], 256)
    wmem('ddr_init.mem', wl['ddr'], 8)

    report = dict(cols=COLS, ctx_words=CTX_WORDS, w_words=W_WORDS,
                  seq_len=len(wl['seq']), params=dict(mq=args.mq, mc=args.mc,
                  dk=args.dk, dv=args.dv, steps=args.steps, frames=args.frames,
                  case=args.case, seed=args.seed))
    for mode, tag in (('REF', 'ref'), ('PRIM', 'prim')):
        ctx, dram, wram, perf = run(wl, mode)
        print(f"[golden] {mode:4s} exec={perf['n_exec']} macs={perf['macs']} "
              f"skip_stages={perf['skip_stages']} skip_macs={perf['skip_macs']}")
        wmem(f'expected_ctx_{tag}.mem', ctx.b.reshape(-1), 8)
        wmem(f'expected_ddr_{tag}.mem', dram, 8)
        report[f'perf_{tag}'] = perf
        if wl['frames'] >= 2 and wl['ddr2'] is not None:
            dram2 = dram.copy()
            M16C, M16Q = wl['M16C'], wl['M16Q']
            dram2[wl['DDR_XCTX']:wl['DDR_XCTX'] + wl['d'] * M16C] = \
                wl['ddr2'][wl['DDR_XCTX']:wl['DDR_XCTX'] + wl['d'] * M16C]
            dram2[wl['DDR_XT']:wl['DDR_XT'] + wl['d'] * M16Q] = \
                wl['ddr2'][wl['DDR_XT']:wl['DDR_XT'] + wl['d'] * M16Q]
            ctx2, dram_f2, _, perf2 = run(wl, mode, ctx0=ctx, dram0=dram2,
                                          wram0=wram)
            print(f"[golden] {mode:4s} 帧二 exec={perf2['n_exec']} "
                  f"macs={perf2['macs']} skip_stages={perf2['skip_stages']}")
            wmem(f'expected_ctx_{tag}_f2.mem', ctx2.b.reshape(-1), 8)
            wmem(f'expected_ddr_{tag}_f2.mem', dram_f2, 8)
            report[f'perf_{tag}_f2'] = perf2
            if mode == 'REF':
                ref_f2 = (ctx2, dram_f2)
            else:
                report['eq_ctx_f2'] = bool(np.array_equal(ref_f2[0].b, ctx2.b))
                report['eq_ddr_f2'] = bool(np.array_equal(ref_f2[1], dram_f2))
    if wl['frames'] >= 2 and wl['ddr2'] is not None:
        wmem('ddr_init2.mem', wl['ddr2'], 8)
        print(f"[golden] REF==PRIM 帧二 ctx:{report['eq_ctx_f2']} "
              f"ddr:{report['eq_ddr_f2']}")
        assert report['eq_ctx_f2'] and report['eq_ddr_f2'], "帧二不一致（模型 bug）"
    if args.frames < 2:
        for f in ('ddr_init2.mem', 'expected_ctx_ref_f2.mem',
                  'expected_ctx_prim_f2.mem', 'expected_ddr_ref_f2.mem',
                  'expected_ddr_prim_f2.mem'):
            p = os.path.join(SIM, f)
            if os.path.exists(p):
                os.remove(p)

    # 帧一等价自检
    r = run(wl, 'REF')
    p = run(wl, 'PRIM')
    ok_ctx = np.array_equal(r[0].b, p[0].b)
    ok_dram = np.array_equal(r[1], p[1])
    print(f"[golden] REF==PRIM 帧一 ctx:{ok_ctx} ddr:{ok_dram}")
    assert ok_ctx and ok_dram, "PRIMITIVE 与 REF 终态不一致（模型 bug）"
    report['eq_ctx'], report['eq_ddr'] = bool(ok_ctx), bool(ok_dram)

    with open(os.path.join(SIM, 'golden_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[golden] 写出 seq.mem ({len(wl['seq'])} 条) / ddr_init.mem / "
          f"expected_* (frames={wl['frames']})")

if __name__ == '__main__':
    main()
