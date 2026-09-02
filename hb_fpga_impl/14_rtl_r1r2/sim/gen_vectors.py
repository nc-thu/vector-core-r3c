# -*- coding: utf-8 -*-
"""gen_vectors.py — ae 加速器黄金模型（位精确）+ 冒烟向量生成
产出（写到本目录）：
  exp2_lut.mem  129 x 13bit 指数表
  seq.mem       256bit 描述符序列（hex，一行一条）
  ddr_init.mem  DDR 初始映像（一行一字节，64KB）
  ddr_init2.mem 第二帧初始映像（仅激活窗口与帧一不同；帧二不复位链路用）
  expected_{ctx,ddr}_{ref,prim}[_f2].mem  两模式 × 两帧的期望终态
  golden_report.json  周期/一致性自检结果
位精确约定（与 rtl/ 完全一致）：
  requant  y = sat8((x*m) >>> s)，m 为 Q8.8 有符号
  softmax  e = EXP[min(max-row,128)]（Q12），r = floor(127*2^30/Σe)，
           P = min(floor(e*r/2^30),127)，因果行 j>i 写 0
  布局     CTX lane = 行 mod 16，addr = (行 div16)*KPAD + k（k-major）
           WRAM lane = 列，addr = k
跳过语义（与 ae_sched.sv skip_fire 一致）：inv≠F && in_loop && step>0 &&
  bitmap[inv]，且 op∉{15,3,4,5}；inv 契约：只有 GEMM 族 op∈{0,1,2} 可携带
  inv≠F（desc()/run() 双侧断言）。op=6（OP_ACTV）同理恒 inv=F，永不 skip。
op=6 AE_ACTV（★ 本目录新增，09_onchip_rtl 专用；字段复用同 ae_core 接线）：
  b_src  = 子模式：0=ACTV（y'=LUT[y]） 1=BIAS（y'=sat8((y·m+b_j)>>>s)）
  m/n    = 行数/列数（n=行 stride，同 softmax 口径）  y_base = 原地张量基址
  b_base = 表映像 CTX 基址（编译器先 OP_LOAD TAG_CTX 把表 DMA 进 CTX）
  k      = BIAS 表项数  rq_m/rq_s = BIAS 乘移常数（Q8.8 / 右移）
  映像布局（= DMA TAG_CTX 字节路由 lane=b%16, addr=base+b/16）：
    ACTV 表项 x 复制到字 b_base+x 的全部 16 lane 槽（映像 256 字）——CTX 广播读
    按槽位对号，每 lane 装满整份 256 项表的唯一办法；
    BIAS 项 j 拆 lo/hi：lo 在 lane j%16 @ b_base+j//16，hi 同相位 @ b_base+NLO+j//16
    （NLO=ceil(k/16)）。
用法：
  python gen_vectors.py                 # 默认冒烟（与历史产物逐字节一致）
  python gen_vectors.py --case tail     # 尾部边界（MQ=17/MC=13/DK=25，3 列组）
  python gen_vectors.py --case actv     # 全链路 + op=6（ACTV×2 表/BIAS）冒烟
  python gen_vectors.py --frames 1      # 单帧（不产 ddr_init2/f2 期望）
"""
import numpy as np
import json, os, argparse

SIM = os.path.dirname(os.path.abspath(__file__))
COLS = 12          # 冒烟档阵列宽
CTX_WORDS = 1024   # 每 bank 深度（RTL 参数同值）
W_WORDS = 64
DDR_SIZE = 0x10000
SEQ_N = 64         # tb_ae.sv 序表深度

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
def build(mq=18, mc=16, dk=20, dv=8, d=8, steps=2, frames=2, seed=20260826,
          actv=False):
    """返回 dict：seq / ddr_f1 / ddr_f2 / 形状参数。默认参数与历史冒烟逐字节一致。
    actv=True 时追加 op=6 三处（setup KCTX-ACTV / 循环内 O2T-BIAS + FT-ACTV）
    与三张表映像的 TAG_CTX 装载——随机抽取顺序不变（新表追加在既有抽取之后）。"""
    M16C = ((mc + 15) // 16) * 16
    M16Q = ((mq + 15) // 16) * 16

    # ---- CTX 布局（字地址；沿用手工布局，K1T/V1T 的 [192,208) 重叠是有意的：
    #      S1 归约 D≤16，K1T 仅前 D 行被 COPY 读，重叠区（行≥16）从不被读）----
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
    # ---- op=6 表映像区（actv 用例）：ACTV 表各 256 字（16 lane 复制），BIAS 表 2 字 ----
    A_TB1, A_TB2, A_TBB = 400, 660, 920
    if actv:
        regions += [('TB1', A_TB1, A_TB1 + 256), ('TB2', A_TB2, A_TB2 + 256),
                    ('TBB', A_TBB, A_TBB + (dv + 15) // 16 * 2)]
    for name, _, end in regions:
        assert end <= CTX_WORDS, f"CTX 溢出：{name} end={end}"
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            n1, b1, e1 = regions[i]
            n2, b2, e2 = regions[j]
            if min(e1, e2) > max(b1, b2):
                assert {n1, n2} == {'K1T', 'V1T'} and d <= 16, \
                    f"意外 CTX 重叠 {n1}/{n2}（K1T/V1T 为文档化例外）"

    # ---- DDR 布局：激活 0x1000/0x1100；权重槽按固定次序 0x2000+slot*0x100 ----
    DDR_XCTX, DDR_XT = 0x1000, 0x1100
    DDR_OUT = 0x8000
    assert d * M16C <= 0x100 and d * M16Q <= 0x100, "激活窗口越过 0x100 栅栏"
    assert d * M16C == 128 and d * M16Q == 256, \
        "tb_ae.sv run_mode_noreset 硬编码窗口长 128/256，需同步改 TB"
    slot = 0
    def next_slot():
        nonlocal slot
        a = 0x2000 + slot * 0x100
        slot += 1
        return a
    W_SLOTS = {}                     # (矩阵, 组号) -> DDR 地址
    for g in range((dk + COLS - 1) // COLS):
        W_SLOTS[('K', g)] = next_slot()
    for mat in ('V', 'K1', 'V1', 'Q1', 'K2C', 'V2C', 'F'):
        W_SLOTS[(mat, 0)] = next_slot()
    assert d * COLS <= 0x100, "权重槽 0x100 栅栏"
    assert max(W_SLOTS.values()) + d * COLS <= DDR_OUT, "权重区侵入输出区"
    assert DDR_OUT + M16Q * COLS <= DDR_SIZE, "DDR 溢出"
    DDR_TB1, DDR_TB2, DDR_TBB = 0x8200, 0x9300, 0xA400      # op=6 表映像（actv 用例）
    if actv:
        assert DDR_TBB + 2 * ((dv + 15) // 16) * 16 <= DDR_SIZE, "表映像 DDR 溢出"

    # ---- 随机数据（抽取顺序即历史顺序；帧二输入追加在最后）----
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
    # op=6 表（actv 用例；追加在既有抽取之后，帧一/帧二数据序列不变）
    LUT1 = rng.integers(-128, 128, size=256).astype(np.int64) if actv else None
    LUT2 = rng.integers(-128, 128, size=256).astype(np.int64) if actv else None
    BJ8  = rng.integers(-3000, 3001, size=dv).astype(np.int64) if actv else None
    RQ_BIAS = (321, 4)                   # O2T 偏置：常饱和（y·321±3000>>4 → ±670）

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
    if actv:
        # ACTV 表映像：项 x 复制到 16 lane（字节 16x+L → lane L，字 b_base+x）
        for x in range(256):
            for L in range(16):
                ddr[DDR_TB1 + 16 * x + L] = LUT1[x]
                ddr[DDR_TB2 + 16 * x + L] = LUT2[x]
        # BIAS 表映像（k=dv≤16）：lo 在字节 j（lane j @ 字 0），hi 在字节 16+j（字 1）
        for j in range(dv):
            ddr[DDR_TBB + j] = int(BJ8[j]) & 0xFF
            ddr[DDR_TBB + 16 + j] = (int(BJ8[j]) >> 8) & 0xFF
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

    # ---- requant 参数（故意含饱和情形）----
    RQ = {
        'kc':  (64, 8), 'vc':  (48, 8), 'k2c': (40, 8), 'v2c': (40, 8),
        'k1':  (56, 8), 'v1':  (56, 8), 'q1':  (56, 8),
        's1':  (16, 8), 's2':  (16, 8),
        'pv1': (32, 8), 'pv2': (32, 8), 'f': (96, 8),
    }
    if actv:
        RQ['o2b'] = RQ_BIAS                    # O2T 偏置的 m/s（Q8.8 / 右移）

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
    def D_ACTV(m, n, y_base, tbl_base, **kw):        # op=6 子模式 0：LUT 直查
        seq.append(desc(op=6, b_src=0, m=m, n=n, y_base=y_base,
                        b_base=tbl_base, **kw))
    def D_BIAS(m, n, y_base, tbl_base, klen, rq, **kw):   # op=6 子模式 1：偏置
        seq.append(desc(op=6, b_src=1, m=m, n=n, k=klen, y_base=y_base,
                        b_base=tbl_base, rq_m=rq[0], rq_s=rq[1], **kw))

    # ---- setup（循环外）：context K/V 投影，K 宽 DK 逐列组 ----
    D_LOAD(DDR_XCTX, d * M16C, 0, A_XCTX)
    D_LOAD(DDR_XT, d * M16Q, 0, A_XT)
    for j0 in range(0, dk, COLS):
        n_loc = min(COLS, dk - j0)
        D_LOAD(W_SLOTS[('K', j0 // COLS)], d * COLS, 1, 0)
        D_GEMM(mc, dk, n_loc, j0, d, A_XCTX, 0, A_KCTX, RQ['kc'])
    D_LOAD(W_SLOTS[('V', 0)], d * COLS, 1, 0)
    D_GEMM(mc, dv, dv, 0, d, A_XCTX, 0, A_VCTX, RQ['vc'], y_tr=1)
    if actv:
        # 表映像装载（4096B×2 + 32B，TAG_CTX）+ setup 段 ACTV：KCTX（stride=dk）
        D_LOAD(DDR_TB1, 4096, 0, A_TB1)
        D_LOAD(DDR_TB2, 4096, 0, A_TB2)
        D_LOAD(DDR_TBB, 2 * ((dv + 15) // 16) * 16, 0, A_TBB)
        D_ACTV(mc, dk, A_KCTX, A_TB1)

    # ---- denoise 循环体（steps 可参；HOIST 对 = context K2/V2 投影）----
    def emit_loop_body(in_loop, is_end):
        # ★ step-invariant：context K2/V2（inv 0/1）——参考实现每步重算
        #   （WRAM 单缓冲：B 权重须在使用前现载——这正是 REF 每步付出的 DMA+MAC 成本）
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
        if actv:
            # ★ op=6 BIAS：O2T 逐列偏置（后继 F 投影消费偏置后的值）
            D_BIAS(mq, dv, A_O2T, A_TBB, dv, RQ['o2b'],
                   in_loop=in_loop, is_loop_end=0)
        # 尾部投影 + 回写
        D_LOAD(W_SLOTS[('F', 0)], d * COLS, 1, 0, in_loop=in_loop, is_loop_end=0)
        D_GEMM(mq, COLS, COLS, 0, dv, A_O2T, 0, A_FT, RQ['f'],
               in_loop=in_loop, is_loop_end=0)
        if actv:
            # ★ op=6 ACTV：输出投影后查表非线性，再回写 DDR
            D_ACTV(mq, COLS, A_FT, A_TB2, in_loop=in_loop, is_loop_end=0)
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
    """执行 wl['seq']。ctx0/dram0/wram0 非 None 时为帧链（不复位的延续状态）；
    bitmap/step/pc 语义同硬件：每次 start 全清。"""
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
        # inv 契约（与 desc() 断言同源）：非 GEMM 族不得携带 inv≠F
        assert inv == 0xF or op in (0, 1, 2), \
            f"pc={pc}: op={op} 携带 inv={inv}（违反编译器契约）"
        # 跳过语义 = ae_sched.sv skip_fire：GEMM 族之外不跳
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
        elif op == 6:  # AE_ACTV：原地行变换（镜像 ae_actv.sv 两子模式）
            # 表映像在执行时刻从 CTX 现读（引擎先装表再跑，同序）
            if b_src == 0:      # ACTV：y' = LUT[y&0xFF]，映像 16 lane 复制
                lut = [int(ctx.b[0, b_base + x]) & 0xFF for x in range(256)]
                for row in range(m):
                    lane, base = row % 16, y_base + (row // 16) * n
                    for j in range(n):
                        ctx.b[lane, base + j] = lut[int(ctx.b[lane, base + j]) & 0xFF]
            else:               # BIAS：y' = sat8((y·m + b_j) >>> s)
                nlo = (k + 15) // 16
                bj = []
                for j in range(k):
                    lo = int(ctx.b[j % 16, b_base + j // 16]) & 0xFF
                    hi = int(ctx.b[j % 16, b_base + nlo + j // 16]) & 0xFF
                    v = lo | (hi << 8)
                    bj.append(v - 0x10000 if v >= 0x8000 else v)
                for row in range(m):
                    lane, base = row % 16, y_base + (row // 16) * n
                    for j in range(n):
                        y = int(ctx.b[lane, base + j])
                        ctx.b[lane, base + j] = sat8((y * rq_m + bj[j]) >> rq_s)
        elif gemmish:
            # RTL 计数口径：mt_cnt * 16 * COLS * k（tile/列宽补齐）
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

# ---------------- pf 用例：权重双缓冲半区交替 ----------------
def build_pf(mq=18, mc=16, d=8, dv=8, steps=2, seed=20260827, rq_tab=None):
    """权重预取（lookahead=1）专用 workload。WRAM 半区 = W_WORDS/2（冒烟档 32 词）。

    结构（★ = 可被预取的 LOAD，发射窗口=前一 GEMM/ATTN_S）：
      setup:  LOAD X_ctx/X_q/X_big(CTX) → K/V 投影（W 半0→半1 边界组 k=32→半0）
      loop :  K2C/V2C（inv 0/1，step>0 GEMM 被 skip、LOAD 照跑 = 死装载）
              Q1/K1/V1 投影（半区交替）→ 自注意 S1（COPY→GEMM×2，末组 ATTN_S
              ★接 W_F 装载，打 T_RUN_SM 窗口发射路径）→ PV1 → 交叉注意 S2 → PV2
              → F 投影（读 SM 窗口预取的 W_F）→ STORE（is_end, steps=2）
    纪律断言（编译器契约，对应 ae_sched pf_issue_ok）：每个可预取 LOAD 的目标
    半区 ≠ 在跑 GEMM 半区；所有 GEMM k ≤ 半区深度；TAG_W LOAD len = k×COLS 且
    不越半区边界。run() 数据语义零改动（预取不改变终态）。"""
    M16Q = ((mq + 15) // 16) * 16
    M16C = ((mc + 15) // 16) * 16
    HALF = W_WORDS // 2
    assert d <= HALF and dv <= HALF and mq <= HALF, "归约维越半区"
    dbig = HALF                       # k=W_WORDS/2 边界组

    # ---- CTX 布局（手工，互不重叠）----
    A_XCTX, A_XT, A_XBIG = 0, 16, 64
    A_KCTX, A_VCTX, A_KB = 128, 160, 544
    A_K2CT, A_V2CT = 192, 224
    A_Q1T, A_K1T, A_V1T = 256, 288, 320
    A_S1T, A_O1T, A_S2T, A_O2T, A_FT = 384, 432, 448, 488, 512
    regions = [('XCTX', 0, d), ('XT', 16, 32), ('XBIG', 64, 96),
               ('KCTX', 128, 140), ('VCTX', 160, 176), ('K2CT', 192, 200),
               ('V2CT', 224, 240), ('Q1T', 256, 272), ('K1T', 288, 304),
               ('V1T', 320, 352), ('S1T', 384, 420), ('O1T', 432, 448),
               ('S2T', 448, 480), ('O2T', 488, 504), ('FT', 512, 536),
               ('KB', 544, 556)]
    for name, b, e in regions:
        assert e <= CTX_WORDS, f"CTX 溢出：{name} end={e}"
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            n1, b1, e1 = regions[i]
            n2, b2, e2 = regions[j]
            assert min(e1, e2) <= max(b1, b2), f"CTX 重叠 {n1}/{n2}"

    # ---- DDR 布局：激活 0x1000+；权重槽 0x2000+slot*0x400（最大 384B）；输出 0x8000 ----
    DDR_XCTX, DDR_XT, DDR_XBIG, DDR_OUT = 0x1000, 0x1100, 0x1200, 0x8000
    W_ADDR = {}                        # 矩阵名 -> DDR 地址（半区由 b_base 决定）
    for i, mat in enumerate(['K', 'KB', 'V', 'K2C', 'V2C', 'Q1', 'K1', 'V1', 'F']):
        W_ADDR[mat] = 0x2000 + i * 0x400
    assert DDR_OUT + M16Q * COLS <= DDR_SIZE, "DDR 溢出"

    # ---- 随机数据 ----
    rng = np.random.default_rng(seed)
    def rint8(*shape):
        return rng.integers(-24, 25, size=shape).astype(np.int64)
    X_ctx = rint8(M16C, d)
    X_q   = rint8(M16Q, d)
    X_big = rint8(M16C, dbig)
    W = dict(K=rint8(d, COLS), KB=rint8(dbig, COLS), V=rint8(d, COLS),
             K2C=rint8(d, COLS), V2C=rint8(d, COLS), Q1=rint8(d, COLS),
             K1=rint8(d, COLS), V1=rint8(d, COLS), F=rint8(d, COLS))

    ddr = np.zeros(DDR_SIZE, dtype=np.int64)
    def wr_act_ddr(addr, X):
        M16, K = X.shape
        b = 0
        for k in range(K):
            for m in range(M16):
                ddr[addr + b] = X[m, k]
                b += 1
    def wr_w_ddr(addr, Wm):
        K, N = Wm.shape
        for k in range(K):
            for j in range(COLS):
                ddr[addr + k * COLS + j] = Wm[k, j] if j < N else 0
    wr_act_ddr(DDR_XCTX, X_ctx)
    wr_act_ddr(DDR_XT, X_q)
    wr_act_ddr(DDR_XBIG, X_big)
    for mat, Wm in W.items():
        wr_w_ddr(W_ADDR[mat], Wm)

    RQ = {'kc': (64, 8), 'kb': (64, 8), 'vc': (48, 8), 'k2c': (40, 8),
          'v2c': (40, 8), 'q1': (56, 8), 'k1': (56, 8), 'v1': (56, 8),
          's1': (16, 8), 's2': (16, 8), 'pv1': (32, 8), 'pv2': (32, 8),
          'f': (96, 8)}
    if rq_tab is not None:                 # rqs 用例：s>8 requant 扫描（T_MAX=39 门）
        RQ.update(rq_tab)

    # ---- 序列表（H0/H1 = WRAM 半区基址）----
    H0, H1 = 0, HALF
    seq = []
    def D_LOAD(ddr_addr, nbytes, tag, base, **kw):
        seq.append(desc(op=4, b_src=tag, b_base=base,
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
    def WLOAD(mat, base, **kw):                 # 权重装载：len = k×COLS
        kk = W[mat].shape[0]
        D_LOAD(W_ADDR[mat], kk * COLS, 1, base, **kw)

    # setup（循环外）
    D_LOAD(DDR_XCTX, d * M16C, 0, A_XCTX)
    D_LOAD(DDR_XT, d * M16Q, 0, A_XT)
    D_LOAD(DDR_XBIG, dbig * M16C, 0, A_XBIG)
    WLOAD('K', H0)
    D_GEMM(mc, COLS, COLS, 0, d, A_XCTX, H0, A_KCTX, RQ['kc'])          # ★KB
    WLOAD('KB', H1)                                                    # k=32 边界组
    D_GEMM(mc, COLS, COLS, 0, dbig, A_XBIG, H1, A_KB, RQ['kb'])        # ★V
    WLOAD('V', H0)
    D_GEMM(mc, dv, dv, 0, d, A_XCTX, H0, A_VCTX, RQ['vc'], y_tr=1)     # ★K2C

    # denoise 循环体（steps=2）
    def emit_loop_body(in_loop, is_end):
        WLOAD('K2C', H1, in_loop=in_loop)                               # ★V2C；step>0 死装载
        D_GEMM(mc, dv, dv, 0, d, A_XCTX, H1, A_K2CT, RQ['k2c'], op=2,
               inv_idx=0, steps=steps, in_loop=in_loop)                 # ★V2C
        WLOAD('V2C', H0, in_loop=in_loop)                               # step>0 死装载
        D_GEMM(mc, dv, dv, 0, d, A_XCTX, H0, A_V2CT, RQ['v2c'], op=2,
               y_tr=1, inv_idx=1, steps=steps, in_loop=in_loop)         # ★Q1
        WLOAD('Q1', H1, in_loop=in_loop)
        D_GEMM(mq, dv, dv, 0, d, A_XT, H1, A_Q1T, RQ['q1'],
               in_loop=in_loop)                                         # ★K1
        WLOAD('K1', H0, in_loop=in_loop)
        D_GEMM(mq, dv, dv, 0, d, A_XT, H0, A_K1T, RQ['k1'],
               in_loop=in_loop)                                         # ★V1
        WLOAD('V1', H1, in_loop=in_loop)
        D_GEMM(mq, dv, dv, 0, d, A_XT, H1, A_V1T, RQ['v1'], y_tr=1,
               in_loop=in_loop)
        # 自注意 1（causal）：S1 = Q1·K1ᵀ [mq×mq]，逐列组 COPY→GEMM，末组接 softmax
        for j0 in range(0, mq, COLS):
            n_loc = min(COLS, mq - j0)
            last = j0 + COLS >= mq
            D_COPY(d, n_loc, A_K1T, dv, j0, H0, in_loop=in_loop)
            D_GEMM(mq, mq, n_loc, j0, d, A_Q1T, H0, A_S1T, RQ['s1'],
                   op=1 if last else 0, sm_causal=1 if last else 0,
                   in_loop=in_loop)
            if last:                                                    # ★F（SM 窗口路径）
                WLOAD('F', H1, in_loop=in_loop)
        D_COPY(mq, dv, A_V1T, M16Q, 0, H0, in_loop=in_loop)
        D_GEMM(mq, dv, dv, 0, mq, A_S1T, H0, A_O1T, RQ['pv1'], in_loop=in_loop)
        # 交叉注意 2（驻留 K2c/V2c，非 causal）：S2 = Q1·K2cᵀ [mq×mc]
        for j0 in range(0, mc, COLS):
            n_loc = min(COLS, mc - j0)
            last = j0 + COLS >= mc
            D_COPY(d, n_loc, A_K2CT, dv, j0, H0, in_loop=in_loop)
            D_GEMM(mq, mc, n_loc, j0, d, A_Q1T, H0, A_S2T, RQ['s2'],
                   op=1 if last else 0, sm_causal=0, in_loop=in_loop)
        D_COPY(mc, dv, A_V2CT, M16C, 0, H0, in_loop=in_loop)
        D_GEMM(mq, dv, dv, 0, mc, A_S2T, H0, A_O2T, RQ['pv2'], in_loop=in_loop)
        D_GEMM(mq, COLS, COLS, 0, dv, A_O2T, H1, A_FT, RQ['f'], in_loop=in_loop)
        D_STORE(DDR_OUT, M16Q * COLS, A_FT,
                in_loop=in_loop, is_loop_end=1 if is_end else 0, steps=steps)

    emit_loop_body(1, 1)
    seq.append(desc(op=15))
    assert len(seq) <= SEQ_N, f"序列 {len(seq)} 条超 SEQ_N={SEQ_N}"

    # ---- 预取纪律断言（编译器契约 ←→ ae_sched pf_issue_ok 硬件守卫）----
    n_pf = 0
    for i, s in enumerate(seq[:-1]):
        op = (s >> 252) & 0xF
        k = (s >> 196) & 0xFFFF
        b_base = (s >> 156) & 0xFFFFF
        nxt = seq[i + 1]
        nop = (nxt >> 252) & 0xF
        if op in (0, 1, 2):
            assert k <= HALF, f"pc={i}: GEMM k={k} 越半区 {HALF}"
        if op in (0, 1, 2) and nop == 4:
            ntag = (nxt >> 246) & 7
            nbase = (nxt >> 156) & 0xFFFFF
            nlen = (nxt >> 61) & 0x3FFFF
            if ntag == 1:                      # TAG_W LOAD 才会被预取
                n_pf += 1
                assert nbase // HALF != b_base // HALF, \
                    f"pc={i+1}: 预取目标半区 {nbase//HALF} 与在跑 GEMM 半区 {b_base//HALF} 相同"
                assert nlen == (nlen // COLS) * COLS and nlen // COLS <= HALF \
                    and nbase % HALF + nlen // COLS <= HALF, \
                    f"pc={i+1}: 权重装载越半区（len={nlen} base={nbase}）"
    print(f"[golden] pf 用例：{len(seq)} 条描述符，可预取权重装载 {n_pf} 处/步")

    return dict(seq=seq, ddr=ddr, ddr2=None, RQ=RQ, mq=mq, mc=mc, dk=dv, dv=dv,
                d=d, steps=steps, frames=1, M16C=M16C, M16Q=M16Q,
                DDR_XCTX=DDR_XCTX, DDR_XT=DDR_XT,
                case='rqs' if rq_tab else 'pf', n_pf=n_pf)


# ---------------- 主流程 ----------------
def wmem(name, arr, width):
    nd = (width + 3) // 4
    with open(os.path.join(SIM, name), 'w') as f:
        mask = (1 << width) - 1
        for v in arr:
            f.write(f"{v & mask:0{nd}X}\n")

def main():
    ap = argparse.ArgumentParser(description='ae 黄金模型 + 向量生成')
    ap.add_argument('--mq', type=int, default=18, help='query token 数')
    ap.add_argument('--mc', type=int, default=16, help='context token 数')
    ap.add_argument('--dk', type=int, default=20, help='K_ctx 特征数（>COLS 多列组）')
    ap.add_argument('--dv', type=int, default=8, help='V/投影特征数')
    ap.add_argument('--steps', type=int, default=2, help='去噪步数')
    ap.add_argument('--frames', type=int, default=2, help='推理帧数（2=含不复位帧链）')
    ap.add_argument('--case', choices=['default', 'tail', 'pf', 'rqs', 'actv'],
                    default='default')
    ap.add_argument('--seed', type=int, default=20260826)
    args = ap.parse_args()
    if args.case == 'tail':       # 尾 tile 边界：M 差一 / n_loc=1 / 3 列组
        args.mq, args.mc, args.dk = 17, 13, 25
    if args.case == 'pf':         # 权重预取：半区交替（seed 自带默认，忽略 --seed）
        wl = build_pf(mq=args.mq, mc=args.mc, d=8, dv=args.dv,
                      steps=args.steps, seed=20260827)
        args.frames = 1
    elif args.case == 'rqs':      # s>8 requant 扫描：HB 标定域 m∈[16424,32754] s∈[21,27]
        #                    （ae_gemm rq_ms T_MAX=39 门；golden y=sat8((acc·m)>>>s)）
        RQS = {'kc': (20000, 23), 'kb': (16424, 21), 'vc': (24000, 24),
               'k2c': (30000, 25), 'v2c': (32754, 27), 'q1': (18000, 22),
               'v1': (22000, 24), 's1': (9000, 21), 's2': (12000, 22),
               'pv1': (32000, 26), 'pv2': (15000, 21), 'f': (20000, 23)}
        wl = build_pf(mq=args.mq, mc=args.mc, d=8, dv=args.dv,
                      steps=args.steps, seed=20260827, rq_tab=RQS)
        args.frames = 1
    else:
        wl = build(mq=args.mq, mc=args.mc, dk=args.dk, dv=args.dv,
                   steps=args.steps, frames=args.frames, seed=args.seed,
                   actv=(args.case == 'actv'))
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
            # 帧二：CTX/WRAM/DDR 延续帧一终态，仅激活窗口换新（同 TB noreset 路径）
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
