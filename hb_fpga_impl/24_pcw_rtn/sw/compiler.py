# -*- coding: utf-8 -*-
"""compiler.py — HB-GD 模型 → ae 加速器分段指令流编译器 v0

输入：
  ops_trace.json    执行序 IR（父模块 hook 晚于子模块 → 注意力的 q/k/v/
                    输出投影/内部 norm 记录都出现在注意力 summary 记录之前，
                    编译器统一吞并重排）
  manifest.json     w8_export 权重清单（key → file/shape/dtype/extra）
  w8_export/        int8 权重文件
  hw_calib_table.json  02_quant 校准表（sa/sw/so/m_requant/s_shift/
                    bias_aug_c/w_bias_int8/bias_fp_fallback）
输出（--out 目录）：
  segments/seg_NNNN/{seq.mem, manifest.json}
  weights_blob.bin   全部段权重字节拼接（manifest 记偏移）
  host_plan.json     段执行序 + 段间 host 步骤
  model_summary.json 统计

位切片与执行语义唯一权威 = hw_zcu104/sim/gen_vectors.py（desc()/run()），
此处照抄。DMA 事实（ae_dma.sv）：单条描述符最长 262128B（内部按 2048B 突发
切分），LOAD 长度须 8 倍数、STORE 须 16 倍数；LOAD W 字节流按「每 k 恰好
COLS 字节」路由进 WRAM（跨 COLS 边界拆残拍），超长 W 装载按 k 行对齐拆条。

v0 关键决策（详见 NOTES.txt）：
  * PL = 全部 Linear/Conv 权重 GEMM + 注意力 QK^T/PV 合成 GEMM（COPY 把
    K/V^T 转进 WRAM B 阵）；host = norm/actv/softmax/rotary/窗口重排/
    im2col/deformable/PSE/豁免层。
  * 注意力路径：BertAttention 走 OP_ATTN_S（QK^T 后硬件 SM16 softmax，段内
    一气呵成到 out_proj）；WindowMSA/Rotary/TemporalJoint/JointGraph/MHA/
    BiMHA 走 QK^T/PV GEMM + 段间 host softmax；MSDeformable 整块 host。
  * K+1 增广：统一 k' = k+1；calib.bias_aug_c 有效且 k+1 ≤ W_HALF 时增广
    （A 图常数词由 host 写进输入图；权重末行 = w_bias_int8）。否则 host-bias
    （PL 出 requant int8，host 反量化加 fp bias 再给下一段）。k > W_WORDS
    整层 host GEMM；豁免层 spatial_enhancer.pts_prob_fc.layers.1 整层 host。
  * 段内直连（注意力 O → 输出投影）：PV 直接按消费者 k' 步长写 O，常数词列
    用每行组一条 16B 小 LOAD 从常数图补写，免二次整图物化。
  * STORE 会读到的 pad 格子（GEMM 从不写的行/列）在计算前从零槽预清零，
    保证 DDR dump 与黄金模型（CTX 零初始化）逐字节一致，segment_runner
    因此不需要层次引用去清 CTX。
  * WRAM 半区交替断言：段内相邻两次（逻辑）LOAD W 目标半区必须不同；
    k' > W_HALF 的层全 WRAM 单缓冲（b_base=0，不断言）。COPY 恒写 WRAM 基址 0。
"""
import itertools
import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

PROFILES = {
    # 与 hw_zcu104 仿真档同参数（CTX 放大到 2048 才装得下迷你 BERT 段）
    'smoke': dict(COLS=12, CTX_WORDS=2048, W_WORDS=64, SEQ_N=64,
                  DDR_BYTES=65536, ZERO_SLOT=256),
    # ZCU104 满配
    'full': dict(COLS=108, CTX_WORDS=131072, W_WORDS=4096, SEQ_N=2048,
                 DDR_BYTES=8388608, ZERO_SLOT=4096),
}

OP_GEMM, OP_ATTN_S, OP_HOIST, OP_COPY, OP_LOAD, OP_STORE, OP_DONE = \
    0, 1, 2, 3, 4, 5, 15
OP_SF = 14                  # pcW 逐列 requant 系数装载（24_pcw_rtn）
F_SYN = 198.5e6
DMA_MAX = 262128            # 0x3FFF0（16 倍数），单描述符 DMA 上限

# pcW+RTN（2026-09-04，24_pcw_rtn）：
#   * requant 就近舍入（RTN）：GEMM 描述符 rq_m bit15=1 时
#     y = sat8((acc·m + 2^(s-1)) >>> s)；bit15=0 保持旧 floor 语义
#     （floor 模式与旧版逐位一致，是 RTL 回归对照的锚）。
#   * 逐列系数：rq_s bit7=1 时 m/s 取 sf 寄存器组（本 GEMM 的前 n_loc 列），
#     由 OP_SF（op=14）描述符字装载：每字 10 个 24b 槽（槽 i =
#     bits[24i+23:24i]，值 = m<<8 | s），bits[251:240] = 起始槽号 slot0。
#     COLS=108 → 每个列组 11 个字；RTL COLS=96 → 10 个字。
#   * s 下限 9：RTL rq_v2 内部 t=s-8，RTN 舍入常数 rn=2^(s-9) 要求 t>=1。
#     软件 rn=2^(s-1) 与 RTL rn=2^(s-9)>>>>(s-8) 逐位等价（整数恒等式，
#     见 24_pcw_rtn/REPORT_SW.md）。
S_MIN_RT = 9

# 头数表：trace 里读不出的家族写假设值（NOTES 有说明，改这里即可）
HEADS = {
    'BertAttention': 12,             # BERT-base（mask [1,1,8,8] 佐证）
    'MultiheadAttention': None,      # 从 attn_mask [H,T,T] 读
    'JointGraphAttention': 8,        # 假设（与 temporal 同族 H=8/d=32）
    'RotaryAttention': 8,            # RotaryEmbedding [1,8,T,32] 佐证
    'TemporalJointGraphAttention': 8,  # 同上
    'BiMultiHeadAttention': 4,       # 假设（E=1024 → d=256）
}


def ceil16(x):
    return (x + 15) // 16


def ceil_div(a, b):
    return -(-a // b)


# ---------------------------------------------------------------------------
# 描述符编码（与 gen_vectors.desc() 逐位一致）
# ---------------------------------------------------------------------------
def desc(op=0, a_src=0, b_src=0, sm_causal=0, y_tr=0, m=0, n=0, k=0,
         a_base=0, b_base=0, y_base=0, b_spad=0, rq_m=0, rq_s=0,
         inv_idx=0xF, steps=0, in_loop=0, is_loop_end=0,
         dma_len=0, dma_addr=0, j0=0):
    assert inv_idx == 0xF or op in (0, 1, 2), 'inv 只能配 GEMM 族'
    # 防回归：dma_len 只能占 18 位字段 [78:61]，超长会把高位溢进
    # is_loop_end(79)/in_loop(80)/steps(81+)，RTL/黄金只读 18 位——
    # 非零窄化静默少搬数据、归零窄化 DMA 死循环（2026-08-31 修）。
    # 超长 LOAD/STORE 必须在发射点按 DMA_MAX 拆分，不许整条编码。
    assert dma_len <= 0x3FFFF, \
        f'dma_len={dma_len} (0x{dma_len:X}) 超 18 位字段，须拆分后再编码'
    assert dma_len == 0 or j0 == 0, \
        'dma_len 与 j0 共用位段 [77:62]，同一条描述符不能同时非零'
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
    v |= b_spad << 120
    v |= (rq_m & 0xFFFF) << 104
    v |= rq_s << 96
    v |= inv_idx << 92
    v |= steps << 81
    v |= in_loop << 80
    v |= is_loop_end << 79
    v |= dma_len << 61
    v |= dma_addr << 29
    v |= (j0 & 0xFFFF) << 62
    assert v < (1 << 256)
    return v


# ---------------------------------------------------------------------------
# 引擎拍数估计（抄 hw_zcu104/sim/gem_cycles.py 口径；softmax 为 SM16 公式）
# ---------------------------------------------------------------------------
ROWS, RQ_SH, DRAIN, DALIGN = 16, 4, 64, 2
BURST_B, AR_OVH, CMD_OVH, GEMM_CMD_OVH = 2048, 2, 5, 2


def _wb_cycles(n_loc, j0, y_tr):
    if not y_tr:
        return n_loc
    return 16 * (((j0 + n_loc - 1) >> 4) - (j0 >> 4) + 1)


def gemm_cycles(m, k, n_loc, j0, y_tr, cols):
    mt = ceil16(m) // 16
    tile = (1 + (k + 2) + (ROWS + cols + 3) + DRAIN + DALIGN + 2
            + _wb_cycles(n_loc, j0, y_tr) + 1)
    return 2 + GEMM_CMD_OVH + mt * tile


def load_w_ideal(k, cols):
    nbytes = k * cols
    return nbytes // 8 + (nbytes // cols) * max(0, (cols - 1) // 8) \
        + math.ceil(nbytes / BURST_B) * AR_OVH + CMD_OVH


def load_ctx_ideal(nbytes):
    return nbytes // 8 + math.ceil(nbytes / BURST_B) * AR_OVH + CMD_OVH


def store_cycles(nbytes):
    return ceil_div(nbytes, 16) * 5 + CMD_OVH


def copy_cycles(k_rows, j_cols, src_j0):
    grps = ((src_j0 + j_cols - 1) >> 4) - (src_j0 >> 4) + 1
    return 2 + 3 * k_rows * grps


def softmax_cycles(m_rows, n_cols, causal):
    """SM16：每行组 P1(glen+2)+P2(glen+3)+DIV(38)+P3(n_cols+4)+NEXT(2)。"""
    tot = 2
    rg = 0
    while rg * 16 < m_rows:
        g_end = min(rg * 16 + 16, m_rows)
        glen = min(n_cols, g_end) if causal else n_cols
        tot += 2 * glen + n_cols + 49
        rg += 1
    return tot


def est_desc_cycles(d, cols):
    op = (d >> 252) & 0xF
    m = (d >> 228) & 0xFFFF
    n = (d >> 212) & 0xFFFF
    k = (d >> 196) & 0xFFFF
    n_loc = (d >> 120) & 0xFFFF
    j0 = (d >> 62) & 0xFFFF
    y_tr = (d >> 244) & 1
    sm_causal = (d >> 245) & 1
    dma_len = (d >> 61) & 0x3FFFF
    b_src = (d >> 246) & 7
    if op in (0, 1, 2):
        g = gemm_cycles(m, k, n_loc, j0, y_tr, cols)
        if op == 1:
            g += softmax_cycles(m, n, sm_causal)
        return g
    if op == 4:
        return load_ctx_ideal(dma_len) if b_src == 0 else \
            load_w_ideal(dma_len // cols, cols)
    if op == 5:
        return store_cycles(dma_len)
    if op == 3:
        return copy_cycles(k, n & 0xFF, j0)
    if op == 14:                          # OP_SF：256b 系数字写入 sf 寄存器组
        return 8
    return 0


# ---------------------------------------------------------------------------
# IR 节点
# ---------------------------------------------------------------------------
class HostOp:
    """host 步骤（段边界）。kind: norm/actv/softmax/rotary/rearrange/
    im2col/host_gemm/exempt/deform_host/other"""

    def __init__(self, tag, module, cls, kind, note='', seq=None):
        self.tag, self.module, self.cls, self.kind = tag, module, cls, kind
        self.note, self.seq = note, seq


class WGemm:
    """权重 GEMM（trace Linear/Conv、MHA 合成 in_proj、注意力重发的投影）。"""

    def __init__(self, tag, module, a, y, m, k, n, wkey, has_bias,
                 kind='linear', src_seq=None):
        self.tag, self.module = tag, module
        self.a, self.y = a, y
        self.m, self.k, self.n = m, k, n
        self.wkey, self.has_bias, self.kind = wkey, has_bias, kind
        self.src_seq = src_seq
        self.store = True        # 段内直连消费时置 False
        self.vt_of = False       # True = 主 GEMM 后追加 y_tr 孪生（V^T 供 PV）
        self.vt_span = None      # 孪生只算的列区间 [lo, hi)；None = 全列
        self.vt_store = True     # 孪生 VT 是否落 DDR（BERT 段内直连则 False）
        self.rq = (64, 8)
        self.rq_pc = None        # 逐列 requant [[m,s]]*n（全局输出列索引）
        self.aug = False
        self.c_val = 0
        self.w_bias = None       # int64 [n]
        self.host_bias = False
        self.sa = self.sw = self.so = 1.0

    @property
    def k_eff(self):
        return self.k + 1


class Attn:
    """注意力实例（family 决定 lowering 路径）。"""

    def __init__(self, tag, module, family, seq):
        self.tag, self.module, self.family, self.seq = tag, module, family, seq
        self.H = 0
        self.d = 0
        self.C = 0
        self.mq = self.mk = 0
        self.units = 1
        self.q = self.k = self.v = None      # 生产者 WGemm
        self.qkv = None                      # swin 打包 qkv
        self.out_recs = []                   # 吞并重发的输出投影 trace 记录
        self.relay_norm = None               # BERT：attention.output.LayerNorm
        self.geo = {}


# ---------------------------------------------------------------------------
# 段构建器
# ---------------------------------------------------------------------------
class Seg:
    def __init__(self, idx, P):
        self.P = P
        self.idx = idx
        self.name = f'seg_{idx:04d}'
        self.descs = []          # (desc_int, note)
        self.ctx = {}            # name -> [base, words, live]
        self.ctx_free = []
        self.ctx_top = 0
        self.ddr_top = 0
        self.inputs = []         # 段输入图（host 写入 DDR）
        self.outputs = []        # 段输出图（STORE 出）
        self.weights = []        # 权重放置
        self.zero_addr = None
        self.w_bytes = 0
        self.macs = 0
        self.notes = []
        self._last_whalf = None
        self.sf_state = None      # 已装载进 sf 寄存器组的系数集（OP_SF 去重）

    def ctx_alloc(self, name, words):
        for i, (b, w) in enumerate(self.ctx_free):
            if w >= words:
                if w > words:
                    self.ctx_free[i] = (b + words, w - words)
                else:
                    self.ctx_free.pop(i)
                self.ctx[name] = [b, words, True]
                return b
        b = self.ctx_top
        if b + words > self.P['CTX_WORDS']:
            return None
        self.ctx_top = b + words
        self.ctx[name] = [b, words, True]
        return b

    def ctx_free_region(self, name):
        b, w, live = self.ctx[name]
        if live:
            self.ctx_free.append((b, w))
            self.ctx[name][2] = False

    def ctx_base(self, name):
        return self.ctx[name][0]

    def ddr_alloc(self, nbytes, align=16):
        a = (self.ddr_top + align - 1) // align * align
        if a + nbytes > self.P['DDR_BYTES']:
            return None
        self.ddr_top = a + nbytes
        return a

    def emit(self, d, note):
        assert len(self.descs) < self.P['SEQ_N'], \
            f'{self.name}: SEQ 溢出（{len(self.descs) + 1} > {self.P["SEQ_N"]}）'
        self.descs.append((d, note))
        op = (d >> 252) & 0xF
        m = (d >> 228) & 0xFFFF
        k = (d >> 196) & 0xFFFF
        if op in (0, 1, 2):
            self.macs += (ceil16(m) // 16) * 16 * self.P['COLS'] * k
        return d

    def wram_half(self, k_len, tag):
        """逻辑 LOAD W 的目标半区（相邻逻辑装载交替；k' > 半区 → 全 WRAM）。"""
        half = self.P['W_WORDS'] // 2
        if k_len > half:
            return 0, 'full'
        h = 0 if self._last_whalf in (None, 1, 'full') else 1
        assert self._last_whalf != h, \
            f'{self.name}@{tag}: WRAM 半区交替断言失败（连续 {h}）'
        self._last_whalf = h
        return h * half, h

    def note_copy_uses_wram(self):
        """COPY 写 WRAM 基址 0：此后 LOAD W 必须去另一半区。"""
        if self._last_whalf is None:
            self._last_whalf = 0


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------
FAMILIES = {'WindowMSA', 'BertAttention', 'RotaryAttention',
            'TemporalJointGraphAttention', 'JointGraphAttention',
            'BiMultiHeadAttention', 'MultiheadAttention'}
CONTAINER_SKIP = {'SwinBlock', 'BertLayer', 'SingleScaleBiAttentionBlock',
                  'DeformableDetrTransformerEncoderLayer',
                  'DetrTransformerEncoderLayer'}
OUT_PROJ_LEAVES = {'dense', 'proj', 'out_proj', 'out_v_proj', 'out_l_proj'}
PRODUCER_LEAVES = {'query', 'key', 'value', 'qkv', 'q_proj', 'k_proj',
                   'v_proj', 'l_proj', 'values_v_proj', 'values_l_proj'}
EXEMPT_PREFIX = 'spatial_enhancer.pts_prob_fc.layers.1'
# 杠杆 F（BRIEF_SW）：额外豁免前缀（--exempt-extra 逗号表填充）。
# 命中的层整层走 host fp（权重不进 PL），零 RTL 改动。
EXEMPT_EXTRA = []
KIND_MAP = {'SiLU': 'actv', 'GELU': 'actv', 'GELUActivation': 'actv',
            'ReLU': 'actv', 'LayerNorm': 'norm', 'RMSNorm': 'norm',
            'AdaRMSNorm': 'norm', 'GroupNorm': 'norm', 'Softmax': 'softmax',
            'RotaryEmbedding': 'rotary'}


def comp_prefix(mod, base):
    """mod 的 '.' 分量前缀恰为 base（避免 layers.1 匹配 layers.10）。"""
    a, b = mod.split('.'), base.split('.')
    return len(a) > len(b) and a[:len(b)] == b


def flat_len(sh):
    p = 1
    for x in sh:
        p *= x
    return p


class Compiler:
    def __init__(self, trace, manifest, w8dir, calib, P, out_dir, acal=None,
                 rq_max_s=47, pcw_w8dir=None, rtn=True):
        self.T = trace['ops']
        self._by_seq = {r['seq']: r for r in self.T}
        self.man = {t['key']: t for t in manifest['tensors']}
        self.w8dir = w8dir
        self.cal = (calib or {}).get('gemms', {})
        self.acal = (acal or {})   # {module: {family, s_absmax, calls}}
        self.rq_max_s = rq_max_s   # requant 移位上限（现网 RTL T_MAX=0 只吃 s=8）
        self.pcw_w8dir = pcw_w8dir  # 逐通道 INT8 权重目录（pcw_export.py 产物）
        self.rtn = rtn             # False = 纯 floor（A/B 对照锚，旧语义）
        self.pcw_wkeys = set()     # 走 pcw 权重文件的 wkey（含 .weight 后缀）
        self.P = P
        self.out = out_dir
        self.col_half = P['W_WORDS'] // 2
        self.nodes = []          # HostOp / WGemm / Attn 混合执行序
        self.segments = []
        self.host_steps = []
        self.blob = bytearray()
        self.blob_map = {}       # (wkey,j0,nloc,aug) -> (off, nbytes)
        self.stats = defaultdict(int)
        self.warnings = []

    # ---------------- 校准 ----------------
    def apply_calib(self, g):
        key = g.wkey[:-len('.weight')] if g.wkey.endswith('.weight') else g.wkey
        self.stats['gemm_total'] += 1
        if g.module.startswith(EXEMPT_PREFIX) or \
                any(g.module.startswith(p) for p in EXEMPT_EXTRA):
            self.stats['exempt_host_gemm'] += 1
            return False                      # 整层 host
        c = self.cal.get(key)
        if c is None:
            # 占位 requant：r = m·2^-s，选 s 使 m 落在 [256, 32767]
            r = 2048.0 / (g.k_eff * 127.0 * 127.0)
            s = 8
            while s < 47 and round(r * (1 << s)) < 256:
                s += 1
            g.rq = (max(1, min(32767, int(round(r * (1 << s))))), s)
            self.stats['gemm_no_calib'] += 1
            self.warnings.append(f'无校准占位 requant: {g.module}')
            return True
        g.sa, g.sw, g.so = c['sa'], c['sw'], c['so']
        g.rq = (int(c['m_requant']), int(c['s_shift']))
        if c.get('pcw') and c.get('rq_ms_col'):
            # 逐通道权重 + 逐列 requant（语义权威 V2c_pcW/V2_rn_pcW：激活
            # 仍静态逐 tensor，sa/so 沿用 v2 表）。不提前 return——偏置继续
            # 走下面的 K+1 增广判定：qkv 等 PL 内消费层的 int8 输出不回
            # host，fp 偏置插不进去，必须进累加器（表里已是逐通道单位
            # w_bias_j=round(b_j/(sa·swc_j·c))，2026-09-04 修正：首版
            # 全 fp 偏置把所有注意力的 q/k/v 偏置静默丢掉了）。
            ms = c['rq_ms_col']
            assert len(ms) == g.n, \
                f'{g.module}: rq_ms_col {len(ms)} != n {g.n}'
            g.rq_pc = [[int(m), int(s)] for m, s in ms]
            self.stats['pcw_layers'] += 1
            wkey = key + ('' if key.endswith('_weight') else '.weight')
            if wkey in self.man:
                self.pcw_wkeys.add(wkey)
        if self.rtn and g.rq[1] < S_MIN_RT:
            # RTN 需要 s>=9（RTL t=s-8>=1）；现有表 s∈[21,27]，此为防御
            m, s = g.rq
            m2 = max(1, min(32767, int(round(m * (1 << (S_MIN_RT - s))))))
            g.rq = (m2, S_MIN_RT)
            self.stats['rq_s_bumped'] += 1
            self.warnings.append(
                f'{g.module}: s={s}<{S_MIN_RT}（RTN 下限），抬到 9 重编码 m')
        half = self.col_half
        aug_ok = (g.has_bias and not c.get('bias_fp_fallback')
                  and c.get('bias_aug_c') is not None
                  and c.get('w_bias_int8') and g.k_eff <= half)
        if aug_ok:
            wb = np.asarray(c['w_bias_int8'], dtype=np.int64)
            if len(wb) == g.n:
                g.aug = True
                g.c_val = int(c['bias_aug_c'])
                g.w_bias = wb
                self.stats['aug_layers'] += 1
                return True
        if g.has_bias:
            g.host_bias = True
            if c.get('bias_fp_fallback'):
                self.stats['host_bias_fp_fallback'] += 1
                if g.rq_pc is not None:
                    self.stats['host_bias_pcw'] += 1
            else:
                self.stats['host_bias_k_too_big'] += 1
        else:
            self.stats['no_bias_layers'] += 1
        return True

    # ---------------- 权重图（blob 缓存） ----------------
    def _load_w_raw(self, wkey):
        e = self.man[wkey]
        d = self.pcw_w8dir if (wkey in self.pcw_wkeys
                               and self.pcw_w8dir) else self.w8dir
        a = np.fromfile(os.path.join(d, e['file']), dtype=np.int8)
        n = flat_len(e['shape'])
        assert len(a) >= n, f'{wkey}: {len(a)} < {n}'
        return a[:n].reshape(e['shape']).astype(np.int64), e

    def weight_image(self, g, j0, n_loc):
        """返回 [k_eff][COLS] int8 的字节串（末行 = w_bias 或 0）。"""
        ck = (g.wkey, j0, n_loc, g.aug)
        if ck in self.blob_map:
            return self.blob_map[ck]
        W, _ = self._load_w_raw(g.wkey)          # [n, k...]（k 维在尾）
        if g.kind in ('conv', 'conv1d', 'conv2d'):
            W = W.reshape(W.shape[0], -1)        # [n, k]
        img = np.zeros((g.k_eff, self.P['COLS']), dtype=np.int8)
        img[:g.k, :n_loc] = W[j0:j0 + n_loc, :g.k].T
        if g.aug:
            img[g.k, :n_loc] = g.w_bias[j0:j0 + n_loc].astype(np.int8)
        raw = img.tobytes()
        pad = (-len(raw)) % 8
        raw += b'\x00' * pad                     # DMA 8 倍数
        off = len(self.blob)
        self.blob += raw
        self.blob_map[ck] = (off, len(raw))
        return off, len(raw)

    # ---------------- 预扫描：注意力实例 / 吞并表 / 几何 ----------------
    def pre_scan(self):
        T = self.T
        attns, by_seq = {}, {}
        for r in T:
            if r['op'] == 'attn':
                if r['cls'] in FAMILIES:
                    A = Attn(r['module'], r['module'], r['cls'], r['seq'])
                    attns[r['seq']] = A
                    by_seq[r['seq']] = A
        # 每条记录归属「时间上随后第一条前缀匹配的注意力实例」。
        # trace 按执行序记录：成员记录（q_proj/proj/softmax…）先于其
        # attention summary 记录；同一 module 在 rollout/去噪多步中反复
        # 执行（如 decoder.layers.1 出现 10 次），必须按时间就近归属，
        # 否则第一个实例会吞掉所有重复调用的成员，后面的实例拿不到叶子。
        owner = {}
        pend = []
        for r in T:
            if r['op'] == 'attn' and r['cls'] in FAMILIES:
                for p in pend:
                    if comp_prefix(p['module'], r['module']):
                        owner[p['seq']] = attns[r['seq']]
                pend = []
            else:
                pend.append(r)
        self.owner = owner
        self.attns = attns
        supp = set()
        for A in attns.values():
            fam = A.family
            kids = defaultdict(list)
            for r in T:
                if owner.get(r['seq']) is A and r['seq'] != A.seq:
                    kids[r['cls']].append(r)
            if fam == 'WindowMSA':
                sm = [r for r in kids.get('Softmax', [])
                      if len(r['in_shapes'][0]) == 4]
                assert sm, f'{A.module}: 找不到窗口 Softmax 几何'
                Wn, H, Tt, _ = sm[0]['in_shapes'][0]
                A.units, A.H, A.mq, A.mk = Wn, H, Tt, Tt
                A.C = None            # 从 qkv 权重宽度补
            elif fam == 'BertAttention':
                sh = self._by_seq[A.seq]['in_shapes'][0]
                _, A.mq, A.C = sh
                A.mk = A.mq
                A.H = HEADS[fam]
                A.d = A.C // A.H
            elif fam == 'MultiheadAttention':
                kw = self._by_seq[A.seq]['kwargs_shapes']
                A.mq = A.mk = kw['query'][0]
                A.C = kw['query'][-1]
                A.H = kw['attn_mask'][0] if kw.get('attn_mask') else \
                    HEADS[fam]
                A.d = A.C // A.H
            elif fam == 'JointGraphAttention':
                kw = self._by_seq[A.seq]['kwargs_shapes']
                A.mq = A.mk = kw['query'][1]
                A.C = kw['query'][2]
                A.H = HEADS[fam]
                A.d = A.C // A.H
            elif fam == 'RotaryAttention':
                kw = self._by_seq[A.seq]['kwargs_shapes']
                A.mq, A.mk, A.C = kw['query'][1], kw['key'][1], kw['query'][2]
                enc = [r for r in kids.get('RotaryEmbedding', [])]
                sh = enc[0]['in_shapes'][0] if enc else [1, 8, A.mq, 32]
                A.H, A.d = sh[1], sh[3]
            elif fam == 'TemporalJointGraphAttention':
                kw = self._by_seq[A.seq]['kwargs_shapes']
                A.units, A.mq, A.C = kw['query'][1], kw['query'][2], kw['query'][3]
                # 全连接：行 (n,tq)、列 (m,tk)——键数是 M*T_k，不是 T_k
                A.mk = kw['key'][1] * kw['key'][2]
                enc = [r for r in kids.get('RotaryEmbedding', [])]
                sh = enc[0]['in_shapes'][0] if enc else [1, 8, A.units, A.mq, 32]
                A.H, A.d = sh[1], sh[-1]
            elif fam == 'BiMultiHeadAttention':
                sh = self._by_seq[A.seq]['in_shapes']
                A.mq, A.C = sh[0][1], sh[0][2]
                A.mk = sh[1][1]
                A.H = HEADS[fam]
                A.d = None        # 由 v_proj 的 n（E）定
            # 吞并：成员叶子分类
            for r in T:
                if owner.get(r['seq']) is not A or r['seq'] == A.seq:
                    continue
                leaf = r['module'].split('.')[-1]
                if r['op'] == 'gemm' and leaf in OUT_PROJ_LEAVES:
                    supp.add(r['seq'])
                    A.out_recs.append(r)
                elif r['op'] == 'gemm' and fam == 'BertAttention' and \
                        leaf in ('query', 'key', 'value'):
                    supp.add(r['seq'])     # 折进注意力段逐头重发
                elif r['op'] == 'elem_norm' and r['cls'] == 'Softmax':
                    supp.add(r['seq'])
                elif r['op'] == 'custom' and r['cls'] == 'RotaryEmbedding':
                    supp.add(r['seq'])
                elif (r['op'] == 'elem_norm' and r['cls'] in
                      ('LayerNorm', 'RMSNorm')
                      and fam == 'BertAttention' and leaf == 'LayerNorm'):
                    supp.add(r['seq'])
                    A.relay_norm = r
        # swin 的 C 从 qkv 权重宽度补
        for A in attns.values():
            if A.family == 'WindowMSA' and A.C is None:
                for r in T:
                    if owner.get(r['seq']) is A and \
                            r['module'].endswith('.qkv'):
                        A.C = r['w_shape'][1]
                        break
        # swin 壳（ShiftWindowMSA 记录）→ reverse host 步骤（挂到 w_msa 实例）
        self.shift_shell = {}
        for r in T:
            if r['op'] == 'attn' and r['cls'] == 'ShiftWindowMSA':
                self.shift_shell[r['module']] = r['seq']
        self.supp = supp
        return attns, supp

    # ---------------- IR 构建 ----------------
    def build_ir(self):
        attns, supp = self.pre_scan()
        self._by_seq = {r['seq']: r for r in self.T}
        # swin 前置 partition host 步骤：插到实例第一个成员记录前
        first_child = {}
        for r in self.T:
            A = self.owner.get(r['seq'])
            if A is not None and A.family == 'WindowMSA' and \
                    r['seq'] != A.seq and A.seq not in first_child:
                first_child[A.seq] = r['seq']
        nodes = self.nodes
        for r in self.T:
            s, A = r['seq'], self.owner.get(r['seq'])
            if A is not None and A.seq != s and first_child.get(A.seq) == s:
                nodes.append(HostOp(f'h{len(nodes)}', A.module, 'WindowMSA',
                                    'swin_partition',
                                    note='窗口切分（含 shift roll）', seq=s))
            if s in supp:
                continue
            if s in attns:                      # 注意力实例自身的 summary 记录
                nodes.append(attns[s])
                continue
            op, cls = r['op'], r['cls']
            if op == 'gemm':
                self._add_gemm(r)
            elif op == 'attn':
                if cls == 'ShiftWindowMSA':
                    wrec = r
                    nodes.append(HostOp(f'h{len(nodes)}', wrec['module'],
                                        cls, 'swin_reverse',
                                        note='窗口 reverse（+shift 还原）',
                                        seq=s))
                elif cls == 'MultiScaleDeformableAttention':
                    if r.get('in_shapes'):
                        nodes.append(HostOp(f'h{len(nodes)}', r['module'], cls,
                                            'deform_host',
                                            note='deformable 采样 host', seq=s))
                        self.stats['host_attn_msdeform'] += 1
                else:
                    pass        # 壳记录（in_shapes 空）跳过
            elif op == 'elem_norm':
                nodes.append(HostOp(f'h{len(nodes)}', r['module'], cls,
                                    KIND_MAP.get(cls, 'other'), seq=s))
            elif op == 'custom':
                if cls in CONTAINER_SKIP:
                    continue
                nodes.append(HostOp(f'h{len(nodes)}', r['module'], cls,
                                    'other', note=cls, seq=s))
        # 生产者链接
        w_by_seq = {n.src_seq: n for n in nodes if isinstance(n, WGemm)}
        for A in attns.values():
            fam = A.family
            kids = [r for r in self.T if self.owner.get(r['seq']) is A
                    and r['seq'] != A.seq and r['op'] == 'gemm']
            leaf_of = {}
            for r in kids:
                leaf = r['module'].split('.')[-1]
                leaf_of.setdefault(leaf, r)
            if fam == 'BertAttention':
                # q/k/v 折进注意力段：从记录造 WGemm 元数据（不进 nodes）
                A.q = self._mk_gemm(leaf_of['query'], False)
                A.k = self._mk_gemm(leaf_of['key'], False)
                A.v = self._mk_gemm(leaf_of['value'], False)
            elif fam in ('JointGraphAttention', 'RotaryAttention',
                         'TemporalJointGraphAttention'):
                A.q = w_by_seq[leaf_of['q_proj']['seq']]
                A.k = w_by_seq[leaf_of['k_proj']['seq']]
                A.v = w_by_seq[leaf_of['v_proj']['seq']]
                A.q.heads_mode = (A.H, A.d, False)
                A.k.heads_mode = (A.H, A.d, False)
                A.v.heads_mode = (A.H, A.d, True)
                A.q.store = A.k.store = A.v.store = False
            elif fam == 'BiMultiHeadAttention':
                A.q = w_by_seq[leaf_of['v_proj']['seq']]     # Q_v / K_v 源
                A.k = w_by_seq[leaf_of['l_proj']['seq']]     # Q_l / K_l 源
                A.v = w_by_seq[leaf_of['values_v_proj']['seq']]
                A.geo['v_l'] = w_by_seq[leaf_of['values_l_proj']['seq']]
                E = A.v.n
                A.d = E // A.H
            elif fam == 'WindowMSA':
                A.qkv = w_by_seq[leaf_of['qkv']['seq']]
        return attns

    def _mk_gemm(self, r, append=True):
        wsh = r['w_shape']
        kind = 'linear'
        sh = r['in_shapes'][0]
        # 行数 = 除最后一维（k 维）外的乘积：[N,K]→N、[Wn,T,C]→Wn·T；
        # 一维 [K] 按 K 行处理（迷你冒烟 trace 的口径）
        m = flat_len(sh) if len(sh) == 1 else flat_len(sh[:-1])
        if r['cls'] in ('Conv1d', 'Conv2d'):
            kind = 'conv1d' if r['cls'] == 'Conv1d' else 'conv2d'
            # m = B·∏(L_out)。优先取 hook 记录的真实输出形状（bringup
            # trace 核对发现 21/30 conv 与「stride=kernel≥4?:1、pad 0」
            # 假设不符：UpsampleHead Conv1d k=3 pad=1 stride=1 等）。
            outs = r.get('out_shapes') or []
            if outs:
                m = flat_len(outs[0]) // wsh[0]
            else:
                ins = r['in_shapes'][0]
                m = ins[0]
                for ax, L in enumerate(ins[2:]):
                    kk = wsh[2] if r['cls'] == 'Conv1d' else wsh[2 + ax]
                    st = kk if kk >= 4 else 1
                    m *= (L - kk) // st + 1
        g = WGemm(f'g{r["seq"]}', r['module'], f't{r["in_ids"][0]}',
                  f'{r["module"]}#{r["seq"]}', m, flat_len(wsh[1:]), wsh[0],
                  r['weight_key'], bool(r.get('has_bias')), kind=kind,
                  src_seq=r['seq'])
        ok = self.apply_calib(g)
        if append:
            if r['cls'] in ('Conv1d', 'Conv2d'):
                self.nodes.append(HostOp(f'h{len(self.nodes)}', r['module'],
                                         r['cls'], 'im2col', seq=r['seq']))
            if not ok:
                self.nodes.append(HostOp(f'h{len(self.nodes)}', r['module'],
                                         r['cls'], 'exempt',
                                         note='豁免层整层 host', seq=r['seq']))
                return None
            if g.k_eff > self.P['W_WORDS']:
                self.stats['host_gemm_k_too_big'] += 1
                self.nodes.append(HostOp(f'h{len(self.nodes)}', r['module'],
                                         r['cls'], 'host_gemm',
                                         note=f'k={g.k} 超 W_WORDS',
                                         seq=r['seq']))
                return None
            self.nodes.append(g)
        return g

    def _add_gemm(self, r):
        return self._mk_gemm(r, append=True)


    # ================= 段发射辅助 =================
    def _fresh_seg(self):
        seg = Seg(len(self.segments), self.P)
        self.segments.append(seg)
        return seg

    def _finalize(self, seg):
        seg.emit(desc(op=OP_DONE), 'DONE')
        cyc = sum(est_desc_cycles(d, self.P['COLS']) for d, _ in seg.descs)
        seg.est_cycles = cyc
        self.stats['total_cycles'] += cyc
        self.stats['d_total'] += len(seg.descs)
        self.stats['max_seq'] = max(self.stats['max_seq'], len(seg.descs))
        self.stats['max_seg_wbytes'] = max(self.stats['max_seg_wbytes'],
                                           seg.w_bytes)

    def _close(self, seg):
        if seg.descs:
            self._finalize(seg)
            return self._fresh_seg()
        return seg              # 空段复用

    def _seg_input(self, seg, name, words, note='', **kw):
        for e in seg.inputs:
            if e['name'] == name:
                return e
        ddr = seg.ddr_alloc(words * 16)
        assert ddr is not None, \
            f'{seg.name}: 输入图 {name}（{words} 字）超 DDR 预算'
        e = dict(name=name, ddr=ddr, words=words, note=note, **kw)
        seg.inputs.append(e)
        return e

    def _seg_output(self, seg, name, base, words, note='', **kw):
        e = dict(name=name, ddr=None, ctx_base=base, words=words, note=note,
                 **kw)
        e['ddr'] = seg.ddr_alloc(words * 16)
        assert e['ddr'] is not None, \
            f'{seg.name}: 输出图 {name}（{words} 字）超 DDR 预算'
        seg.outputs.append(e)
        return e

    def _emit_store(self, seg, out, ctx_base, nbytes, note='', byte0=0):
        """STORE 按 DMA_MAX 拆分：STORE 长度须 16 倍数，DMA_MAX 本身是
        16 倍数，分片天然保持 16 对齐；y_base（CTX 字，16B/字）与
        dma_addr 按分片字节偏移推进。拆分与整条搬运逐字节等价
        （STORE 字节流 word-major，分片边界在字边界上）。
        byte0 = 本次搬运在输出图内的起始字节偏移（行 tile 循环时
        (r0//16)*width*16：word-major 图按 [行块][列] 排，tile 起始
        行块决定偏移；不传则从图首写——多 tile 共用一个图时必须传，
        否则后 tile 覆盖前 tile，2026-08-31 修）。"""
        assert nbytes % 16 == 0
        assert byte0 % 16 == 0
        base_note = note or f'STORE {out["name"]}'
        off = 0
        while off < nbytes:
            n = min(DMA_MAX, nbytes - off)
            seg.emit(desc(op=OP_STORE, y_base=ctx_base + off // 16, dma_len=n,
                          dma_addr=out['ddr'] + byte0 + off),
                     base_note if off == 0 else f'{base_note} +{off}B')
            self.stats['d_store'] += 1
            off += n

    def _emit_load_ctx(self, seg, ddr, off, nbytes, ctx_base, note=''):
        """LOAD CTX 按 DMA_MAX 拆分。字节流 k-major（byte b → lane b%16、
        字 b//16），中间分片必须 16 字节对齐才能让 lane 相位与整条搬运
        一致、b_base 推进取整——DMA_MAX 是 16 倍数所以自动满足；只有
        8 倍数尾巴会落在最后一个分片（无需再推进）。"""
        assert nbytes % 8 == 0
        off_b = 0
        while off_b < nbytes:
            n = min(DMA_MAX, nbytes - off_b)
            if n % 16 and nbytes - off_b - n > 0:
                n = max(8, n - 8)          # 中间分片退到 16 对齐
            seg.emit(desc(op=OP_LOAD, b_src=0, b_base=ctx_base + off_b // 16,
                          dma_len=n, dma_addr=ddr + off + off_b),
                     note if off_b == 0 else f'{note} +{off_b}B')
            self.stats['d_load_ctx'] += 1
            off_b += n

    def _emit_copy(self, seg, k_rows, j_cols, src_base, spad, src_j0, note):
        seg.emit(desc(op=OP_COPY, k=k_rows, n=j_cols, b_base=src_base,
                      b_spad=spad, rq_m=src_j0, a_base=0), note)
        self.stats['d_copy'] += 1
        seg.note_copy_uses_wram()

    def _emit_sf(self, seg, slots):
        """OP_SF 装载逐列 requant 系数：每字 10 个 24b 槽（m<<8|s），
        bits[251:240]=slot0。slots = 本列组连续列的 [(m,s)]。"""
        for w0 in range(0, len(slots), 10):
            v = OP_SF << 252
            v |= (w0 & 0xFFF) << 240
            for i, (m, s) in enumerate(slots[w0:w0 + 10]):
                assert 1 <= m <= 32767 and S_MIN_RT <= s <= self.rq_max_s, \
                    f'sf 槽非法 (m={m},s={s})'
                v |= ((m << 8) | s) << (24 * i)
            seg.emit(v, f'SF cols +{w0} ({len(slots)} 列)')
            self.stats['d_sf'] += 1

    def _emit_gemm(self, seg, m, n, n_loc, j0, k, a_base, b_base, y_base,
                   rq, note, op=OP_GEMM, y_tr=0, sm_causal=0, rq_pc=None,
                   col0=None):
        """rq_pc 给定时按列发系数（OP_SF + rq_s bit7）。col0 = 本 GEMM 首列
        在层的全局输出列里的下标——描述符 j0 是 Y 图散布偏移，两者在
        chunk 打包 / 逐头块两种布局下不同（col0 缺省 = j0，全宽 Y 布局）。"""
        rq_m, rq_s = rq[0], rq[1]
        if rq_pc is not None:
            c0 = j0 if col0 is None else col0
            slots = rq_pc[c0:c0 + n_loc]
            state = (id(rq_pc), c0, n_loc)
            if seg.sf_state != state:
                self._emit_sf(seg, slots)
                seg.sf_state = state
            rq_s |= 0x80                     # 逐列系数模式
        if self.rtn:
            rq_m |= 0x8000                   # RTN 就近舍入
        seg.emit(desc(op=op, m=m, n=n, k=k, a_base=a_base, b_base=b_base,
                      y_base=y_base, b_spad=n_loc, rq_m=rq_m, rq_s=rq_s,
                      j0=j0, y_tr=y_tr, sm_causal=sm_causal), note)
        self.stats['d_gemm'] += 1

    def _prezero(self, seg, ctx_base, words):
        """从零槽 LOAD 预清零（必须在写入该区的 GEMM 之前；
        每条从零槽头部读 ≤ 槽长的块，避免越槽）。"""
        P = self.P
        if seg.zero_addr is None:
            seg.zero_addr = seg.ddr_alloc(P['ZERO_SLOT'] * 16, align=2048)
            assert seg.zero_addr is not None, f'{seg.name}: 零槽分配失败'
        total = words * 16
        off = 0
        while off < total:
            n = min(DMA_MAX, total - off, P['ZERO_SLOT'] * 16) // 16 * 16
            seg.emit(desc(op=OP_LOAD, b_src=0, b_base=ctx_base + off // 16,
                          dma_len=n, dma_addr=seg.zero_addr),
                     f'prezero {n}B @{ctx_base + off // 16}')
            self.stats['d_load_ctx'] += 1
            off += n

    def _const_stripe(self, seg, base, pitch, groups, cval, tag):
        """给 A 图的常数词列（第 k 列）补 16B：每组一条 LOAD。"""
        P = self.P
        img = self._seg_input(seg, f'const:{cval}', 1, note=f'常数 {cval}×16B',
                              kind='const', cval=cval)
        for gidx in range(groups):
            seg.emit(desc(op=OP_LOAD, b_src=0, b_base=base + gidx * pitch
                          + pitch - 1, dma_len=16, dma_addr=img['ddr']),
                     f'const stripe {tag} g{gidx}')
            self.stats['d_load_ctx'] += 1

    def _emit_load_w(self, seg, g, j0, n_loc):
        """装载一组权重进 WRAM 半区（超长按 k 行拆条）。返回 b_base。
        同段内同 (wkey,j0,n_loc) 复用 DDR 放置（重发 LOAD 无害）——
        swin 每窗现载 proj，不复用会被同名图撑爆段 DDR。"""
        P = self.P
        cols = P['COLS']
        off, nbytes = self.weight_image(g, j0, n_loc)
        we = None
        for we0 in seg.weights:
            if we0['key'] == g.wkey and we0['j0'] == j0 and \
                    we0['n_loc'] == n_loc and we0['blob_len'] == nbytes:
                we = we0
                break
        if we is None:
            we = dict(key=g.wkey, j0=j0, n_loc=n_loc, blob_off=off,
                      blob_len=nbytes, aug=g.aug)
            we['ddr'] = seg.ddr_alloc(nbytes, align=2048)
            assert we['ddr'] is not None, f'{seg.name}: 权重图超 DDR'
            seg.weights.append(we)
            seg.w_bytes += nbytes
        ddr = we['ddr']
        b_base, half = seg.wram_half(g.k_eff, f'{g.tag}@{j0}')
        we['b_base'], we['half'] = b_base, half
        rows = g.k_eff
        r = 0
        while r < rows:
            rem = nbytes - r * cols          # 到图尾（含 8 倍数 pad）的字节数
            if rem <= DMA_MAX:
                seg.emit(desc(op=OP_LOAD, b_src=1, b_base=b_base + r,
                              dma_len=rem, dma_addr=ddr + r * cols),
                         f'LOAD W {g.wkey}[{j0}:{j0 + n_loc}] rows {r}:'
                         f'{rows}+pad')
                self.stats['d_load_w'] += 1
                break
            nr = DMA_MAX // cols             # 整段装载：cols·nr 须 8 倍数
            while (nr * cols) % 8:
                nr -= 1
            seg.emit(desc(op=OP_LOAD, b_src=1, b_base=b_base + r,
                          dma_len=nr * cols, dma_addr=ddr + r * cols),
                     f'LOAD W {g.wkey}[{j0}:{j0 + n_loc}] rows {r}+{nr}')
            self.stats['d_load_w'] += 1
            r += nr
        return b_base

    # ================= 通用权重 GEMM lowering =================
    def _lower_wgemm(self, seg, g, a_direct=None):
        """列组打包 → 行 tile 循环。a_direct 给定时 A 直接用段内 O 区。"""
        P = self.P
        cols = P['COLS']
        k = g.k_eff
        m16 = ceil16(g.m)
        groups = [(j0, min(cols, g.n - j0)) for j0 in range(0, g.n, cols)]
        lo, hi = g.vt_span if g.vt_of and g.vt_span else (
            (0, g.n) if g.vt_of else (0, 0))
        vt_words = (ceil_div(hi, 16) - lo // 16) * m16 if g.vt_of else 0
        wbytes = (-(-(k * cols) // 8) * 8)
        a_bytes = m16 * k * 16
        budget = P['DDR_BYTES'] - P['ZERO_SLOT'] * 16 - 2048
        # ---- 列组打包成 chunk（Y 图按 chunk 紧凑，host 拼接）----
        chunks, cur, ysum = [], [], 0
        for (j0, n_loc) in groups:
            need = a_bytes + ysum + m16 * n_loc * 16 \
                + wbytes * (len(cur) + 1) + vt_words * 16
            if cur and need > budget:
                chunks.append(cur)
                cur, ysum = [], 0
            cur.append((j0, n_loc))
            ysum += m16 * n_loc * 16
        chunks.append(cur)
        out_segs = []
        for ci, chunk in enumerate(chunks):
            width = sum(nl for _, nl in chunk)
            fixed = wbytes * len(chunk) + vt_words * 16 \
                + P['ZERO_SLOT'] * 16
            need = a_bytes + m16 * width * 16 + fixed
            rblks = [(0, g.m)]
            if need > budget and a_direct is None and not g.vt_of:
                # 单列组也超预算（激活本身接近 DDR 上限）→ 按行分块，
                # 每块用切片输入/输出图独立落段，host 按行段拼回整张
                grp = max(1, (budget - fixed) // ((k + width) * 16))
                rstep = max(16, min(m16, grp * 16))
                rblks = [(r, min(rstep, g.m - r))
                         for r in range(0, g.m, rstep)]
            for (br0, rows) in rblks:
                sliced = rows != g.m
                est = 4 + 2 * len(chunk) + 3 * (ceil16(rows) // 16 + 1)
                b_need = fixed + (ceil16(rows) * (k + width) * 16
                                  if sliced else need)
                if seg is None or len(seg.descs) + est > P['SEQ_N'] - 2 or \
                        (ci > 0 and seg.descs) or \
                        (sliced and seg.descs) or \
                        (seg.descs and seg.ddr_top + b_need > P['DDR_BYTES']):
                    seg = self._close(seg)
                seg = self._wgemm_chunk(seg, g, chunk, a_direct,
                                        br0=br0, rows=rows)
                out_segs.append(seg)
        return seg

    def _ctx_mtg(self, seg, k, width, reserve=0):
        """行 tile 组数：扣除段内 CTX 已占用量（A/Y 两块同时活）。"""
        live = seg.ctx_top - sum(w for _, w in seg.ctx_free)
        return max(1, (int(self.P['CTX_WORDS'] * 0.9) - live - reserve)
                   // (k + width))

    def _wgemm_chunk(self, seg, g, chunk, a_direct=None, br0=0, rows=None):
        P = self.P
        cols = P['COLS']
        k = g.k_eff
        j0_lo = chunk[0][0]
        width = sum(nl for _, nl in chunk)
        lo, hi = g.vt_span if g.vt_of and g.vt_span else (
            (0, g.n) if g.vt_of else (0, 0))
        if rows is None:
            rows = g.m
        m16 = ceil16(rows)
        sliced = rows != g.m
        # ---- 图（行分块时用切片名，host 按行段供给/拼回） ----
        if a_direct is None:
            aimg = self._seg_input(seg, f'{g.a}@{br0}' if sliced else g.a,
                                   m16 * k, kind='act_in',
                                   m=rows, k=k, pitch=k, sa=g.sa,
                                   row_lo=br0, row_hi=br0 + rows,
                                   module=g.module, layout='kact',
                                   note=f'A 图 {g.module}[{br0}:+{rows}]')
        yimg = None
        if g.store:
            yimg = self._seg_output(seg, f'{g.y}@{br0}' if sliced else g.y,
                                    0, m16 * width,
                                    kind='act_out', m=rows, n=width,
                                    col_lo=j0_lo, col_hi=j0_lo + width,
                                    row_lo=br0, row_hi=br0 + rows,
                                    pitch=width, so=g.so, layout='wm',
                                    host_bias=g.host_bias,
                                    bias_key=g.wkey, module=g.module,
                                    note=f'Y 图 {g.module}[{br0}:+{rows}]')
        vt = None
        if g.vt_of:
            clo = max(lo, j0_lo)
            chi = min(hi, j0_lo + width)
            if chi > clo:
                m16v = m16
                glo, ghi = clo // 16, ceil_div(chi, 16)
                vwords = (ghi - glo) * m16v
                vt = self._seg_output(seg, f'VT:{g.y}', 0, vwords,
                                      kind='vt_out', m=g.m,
                                      col_lo=clo, col_hi=chi, pitch=m16v,
                                      glo=glo, so=g.so, layout='wm',
                                      module=g.module, note=f'VT {g.module}')
        # ---- 行 tile 循环（Y/A 放不下时组数减半重试，抗碎片化） ----
        mtg = self._ctx_mtg(seg, k, width, reserve=(vt['words'] if vt else 0))
        vt_ct = None
        r0 = 0
        while r0 < rows:
            mt = min(mtg * 16, rows - r0)
            yb = ab = None
            while yb is None:
                yb = seg.ctx_alloc(f'Y:{g.tag}:{br0}+{r0}',
                                   ceil16(mt) * width)
                if yb is not None and a_direct is None:
                    ab = seg.ctx_alloc(f'A:{g.tag}:{br0}+{r0}',
                                       ceil16(mt) * k)
                    if ab is None:
                        seg.ctx_free_region(f'Y:{g.tag}:{br0}+{r0}')
                        yb = None
                if yb is None:
                    if mt <= 16:
                        break
                    mt = max(16, (mt // 2 // 16) * 16)
            if yb is not None and a_direct is not None:
                ab = a_direct + (r0 // 16) * k
            assert yb is not None, f'{seg.name}: Y/A 工作区溢出（碎片化）'
            if a_direct is None:
                nb = ceil16(mt) * k * 16
                self._emit_load_ctx(seg, aimg['ddr'], (r0 // 16) * k * 16,
                                    nb, ab, f'LOAD A[{br0 + r0}:'
                                            f'{br0 + r0 + mt}]')
            # 末行组 pad 预清零（本 tile 拥有全局末组时）
            if g.m % 16 and br0 + r0 + mt >= g.m:
                gidx = g.m // 16 - (br0 + r0) // 16
                self._prezero(seg, yb + gidx * width, width)
            for (j0, n_loc) in chunk:
                b_base = self._emit_load_w(seg, g, j0, n_loc)
                self._emit_gemm(seg, mt, width, n_loc, j0 - j0_lo, k, ab,
                                b_base, yb, g.rq,
                                f'GEMM {g.module}[{r0}:+{mt}] '
                                f'cols {j0}+{n_loc}', rq_pc=g.rq_pc,
                                col0=j0)
                if vt:
                    clo = max(lo, j0)
                    chi = min(hi, j0 + n_loc)
                    if chi > clo:
                        if vt_ct is None:
                            vt_ct = seg.ctx_alloc(f'VT:{g.tag}', vt['words'])
                            assert vt_ct is not None
                            self._prezero(seg, vt_ct, vt['words'])
                        self._emit_gemm(seg, mt, hi, chi - clo, clo, k, ab,
                                        b_base, vt_ct, g.rq, y_tr=1,
                                        note=f'VT twin {g.module}',
                                        rq_pc=g.rq_pc, col0=clo)
            if yimg:
                self._emit_store(seg, yimg, yb, ceil16(mt) * width * 16,
                                 f'STORE Y {g.module}[{br0 + r0}:+{mt}]',
                                 byte0=(r0 // 16) * width * 16)
            seg.ctx_free_region(f'Y:{g.tag}:{br0}+{r0}')
            if a_direct is None:
                seg.ctx_free_region(f'A:{g.tag}:{br0}+{r0}')
            r0 += mt
        if vt and vt_ct is not None:
            self._emit_store(seg, vt, vt_ct, vt['words'] * 16,
                             f'STORE VT {g.module}')
            seg.ctx_free_region(f'VT:{g.tag}')
        return seg


    # ================= 注意力 lowering =================
    def _host(self, seg, module, cls, kind, note=''):
        if seg.descs:
            self._finalize(seg)
            seg = self._fresh_seg()
        self.host_steps.append(dict(after_seg=len(self.segments) - 1,
                                    module=module, cls=cls, kind=kind,
                                    note=note))
        return seg

    def _ph_rq(self, k):
        """合成 GEMM（QK^T/PV/无校准层）的占位 requant。"""
        r = 2048.0 / (k * 127.0 * 127.0)
        s = 8
        while s < 47 and int(round(r * (1 << s))) < 1:
            s += 1
        return (max(1, min(32767, int(round(r * (1 << s))))), s)

    def _enc_r(self, r):
        """r = m·2^-s 编码（hw_calib v1_encode 同款：s=floor(log2(32767/r)),
        m=round(r·2^s)∈[1,32767]）。s 再夹到 self.rq_max_s：现网 RTL 的
        GEMM requant 是 rq_v2 T_MAX=0（无桶形移位，只支持 s=8，见
        hw_calib 模式 B_s8 的已知失败模式），对拍现网 RTL 时用
        --rq-max-s 8；桶形版（v1 gate 形态）用默认 47。
        RTN 开启时下限抬到 S_MIN_RT=9（rn=2^(s-9) 需 t=s-8>=1）。"""
        if not (r > 0):
            return (1, S_MIN_RT if self.rtn else 0)
        s = int(np.floor(np.log2(32767.0 / r)))
        s = max(S_MIN_RT if self.rtn else 0, min(self.rq_max_s, s))
        m = max(1, min(32767, int(round(r * (1 << s)))))
        if r * (1 << s) < 0.5 and m == 1:
            self.stats['rq_s_clamp_starved'] += 1
        return (m, s)

    def _attn_rq(self, A, so_q, so_k, so_v, so_o, exact_temp=False):
        """两相注意力的 QK^T / PV requant（替占位 _ph_rq）。

        QK^T: acc = q_int@k_int 代表 S_fp/(so_q·so_k)， requant 到
              S_int = S_fp/σ_S ⇒ r_qk = so_q·so_k/σ_S，
              σ_S = absmax(plain q@k)/127（attn_calib.py 量）。
        BERT 例外（exact_temp）：SM16 硬 softmax 直接吃 S_int，温度必须
              精确 ⇒ r_qk = so_q·so_k/√d，不做 absmax 归一（S/√d 典型
              ±20，int8 不饱和）。
        PV:   acc = P_int@v_int，P_int=P·127 ⇒ r_pv = so_v/(127·so_o)，
              so_o = out 投影 GEMM 的 sa（PV 结果直连其输入）。
        """
        ac = self.acal.get(A.module)
        if exact_temp:
            rq_qk = self._enc_r(so_q * so_k * (A.d ** -0.5))
            sigma_s = so_q * so_k * (A.d ** -0.5)
            if ac and ac.get('s_absmax'):
                sat = ac['s_absmax'] * sigma_s
                if sat > 120:
                    self.warnings.append(
                        f'BERT 精确温度接近饱和: {A.module} '
                        f'max|S/√d|≈{sat:.0f}')
            rq_pv = self._enc_r(so_v / (127.0 * so_o)) if so_o > 0 else \
                self._ph_rq(A.mq)
            self.stats['attn_rq_calibrated'] += 1
            return rq_qk, rq_pv, sigma_s
        if (ac is None or not ac.get('s_absmax')
                or min(so_q, so_k, so_v, so_o) <= 0):
            self.stats['attn_rq_placeholder'] += 1
            self.warnings.append(
                f'注意力缺 σ_S/so，退占位 requant: {A.module}')
            return (self._ph_rq(A.d), self._ph_rq(A.mk or A.mq), None)
        sigma_s = ac['s_absmax'] / 127.0
        rq_qk = self._enc_r(so_q * so_k / sigma_s)
        rq_pv = self._enc_r(so_v / (127.0 * so_o))
        self.stats['attn_rq_calibrated'] += 1
        return rq_qk, rq_pv, sigma_s

    def _attn_meta(self, A, rq_qk, rqpv, sigma_s, so_q, so_k, so_vs, so_os,
                   extra=None):
        """把注意力数值链常数挂到最近一条 host 步骤（host 驱动直接读）。"""
        meta = dict(family=A.family, module=A.module, seq=A.seq,
                    H=A.H, d=A.d, mq=A.mq, mk=A.mk, units=A.units,
                    sigma_s=sigma_s, sigma_q=so_q, sigma_k=so_k,
                    sigma_vs=so_vs, sigma_os=so_os,
                    rq_qk=list(rq_qk),
                    rq_pvs=[list(r) for r in (rqpv if isinstance(rqpv, list)
                                              else [rqpv])])
        if extra:
            meta.update(extra)
        if self.host_steps:
            self.host_steps[-1].setdefault('attn', meta)
        return meta

    def _blk_in(self, seg, name, rows16, cols, note='', kind='blk_in'):
        return self._seg_input(seg, name, rows16 * cols, kind=kind,
                               rows16=rows16, cols=cols, layout='kact',
                               note=note)

    def _blk_out(self, seg, name, rows16, cols, note='', kind='blk_out'):
        return self._seg_output(seg, name, 0, rows16 * cols, kind=kind,
                                rows16=rows16, cols=cols, layout='wm',
                                note=note)

    def _load_blk(self, seg, img, ctx_base, note=''):
        nb = img['words'] * 16
        self._emit_load_ctx(seg, img['ddr'], 0, nb, ctx_base, note)

    # ---------- per-head 权重 GEMM（q/k 紧凑块 / v 孪生 VT） ----------
    def _lower_wgemm_heads(self, seg, g, H, d, twin, tag):
        P = self.P
        k = g.k_eff
        m16 = ceil16(g.m)
        aimg = self._seg_input(seg, g.a, m16 * k, kind='act_in', m=g.m, k=k,
                               pitch=k, sa=g.sa, layout='kact',
                               module=g.module, note=f'A 图 {g.module}')
        d16 = ceil16(d)
        m16r = m16 * 16          # y_tr 的列距 = 行补齐 16 的行数（不是组数！）
        for h in range(H):
            vt_ct = None
            if twin:
                vt_ct = seg.ctx_alloc(f'{tag}:VT{h}', d16 * m16r)
                assert vt_ct is not None, f'{seg.name}: VT{h} 溢出'
                if g.m % 16:
                    self._prezero(seg, vt_ct, d16 * m16r)
                vimg = self._blk_out(seg, f'{g.y}#vt{h}', d16, m16r,
                                     kind='vt_out', note=f'VT {g.module} h{h}')
            mtg = self._ctx_mtg(seg, k, d, reserve=(d16 * m16r if twin else 0))
            if twin:
                assert mtg * 16 >= g.m, \
                    f'{g.module}: VT 孪生需单 tile（m={g.m} > {mtg * 16}）'
            for r0 in range(0, g.m, mtg * 16):
                mt = min(mtg * 16, g.m - r0)
                if twin:
                    yb = vt_ct
                else:
                    yb = seg.ctx_alloc(f'{tag}:Y{h}:{r0}', ceil16(mt) * d)
                    assert yb is not None, f'{seg.name}: Y{h} 溢出'
                    if g.m % 16 and r0 + mt >= g.m:
                        gi = g.m // 16 - r0 // 16
                        self._prezero(seg, yb + gi * d, d)
                ab = seg.ctx_alloc(f'{tag}:A{h}:{r0}', ceil16(mt) * k)
                assert ab is not None, f'{seg.name}: A{h} 溢出'
                self._emit_load_ctx(seg, aimg['ddr'], (r0 // 16) * k * 16,
                                    ceil16(mt) * k * 16, ab,
                                    f'LOAD A {g.module} h{h}[{r0}:]')
                b_base = self._emit_load_w(seg, g, h * d, d)
                self._emit_gemm(seg, mt, d, d, 0, k, ab, b_base, yb, g.rq,
                                f'GEMM {g.module} h{h}[{r0}:+{mt}]',
                                y_tr=1 if twin else 0, rq_pc=g.rq_pc,
                                col0=h * d)
                if not twin:
                    yimg = self._blk_out(seg, f'{g.y}#h{h}', ceil16(mt), d,
                                         note=f'{g.module} h{h} blk')
                    self._emit_store(seg, yimg, yb, ceil16(mt) * d * 16)
                    seg.ctx_free_region(f'{tag}:Y{h}:{r0}')
                seg.ctx_free_region(f'{tag}:A{h}:{r0}')
            if twin:
                self._emit_store(seg, vimg, vt_ct, d16 * m16r * 16)
                seg.ctx_free_region(f'{tag}:VT{h}')
        return seg

    # ---------- BertAttention：OP_ATTN_S 单段直通 ----------
    def _attn_bert(self, seg, A):
        P = self.P
        cols = P['COLS']
        T, C, H, d = A.mq, A.C, A.H, A.d
        kq = A.q.k_eff
        ke = ceil16(T)
        if seg.descs:
            seg = self._close(seg)
        relays = [g for g in (self._relay_gemm(r) for r in A.out_recs)
                  if g is not None]
        cval = relays[0].c_val if relays and relays[0].aug else 0
        rq_qk, rqpv, sig_s = self._attn_rq(
            A, A.q.so, A.k.so, A.v.so,
            relays[0].sa if relays else 1.0, exact_temp=True)
        aimg = self._seg_input(seg, A.q.a, ke * kq, kind='act_in', m=T,
                               k=kq, pitch=kq, sa=A.q.sa, layout='kact',
                               module=A.module, note='BERT X 图')
        ab = seg.ctx_alloc('bert:X', ke * kq)
        assert ab is not None
        self._emit_load_ctx(seg, aimg['ddr'], 0, ke * kq * 16, ab, 'LOAD X')
        o_base = seg.ctx_alloc('bert:O', ke * kq)
        assert o_base is not None
        self._prezero(seg, o_base, ke * kq)     # pad 行 + 常数词列
        for h in range(H):
            qb = seg.ctx_alloc(f'bert:Q{h}', ke * d)
            kb = seg.ctx_alloc(f'bert:K{h}', ke * d)
            vt = seg.ctx_alloc(f'bert:VT{h}', ceil16(d) * ke * 16)
            assert None not in (qb, kb, vt)
            for nm, gg, yb, tr in (('q', A.q, qb, 0), ('k', A.k, kb, 0),
                                   ('v', A.v, vt, 1)):
                b_base = self._emit_load_w(seg, gg, h * d, d)
                self._emit_gemm(seg, T, d, d, 0, kq, ab, b_base, yb, gg.rq,
                                f'BERT {nm} h{h}', y_tr=tr, rq_pc=gg.rq_pc,
                                col0=h * d)
            self._emit_copy(seg, d, T, kb, d, 0, f'BERT COPY K^T h{h}')
            sb = seg.ctx_alloc(f'bert:S{h}', ke * T)
            self._emit_gemm(seg, T, T, T, 0, d, qb, 0, sb, rq_qk,
                            f'BERT QK^T+SM h{h}', op=OP_ATTN_S)
            self._emit_copy(seg, T, d, vt, ke * 16, 0, f'BERT COPY V h{h}')
            self._emit_gemm(seg, T, kq, d, h * d, T, sb, 0, o_base, rqpv,
                            f'BERT PV h{h}')
            for nm in (f'bert:Q{h}', f'bert:K{h}', f'bert:S{h}',
                       f'bert:VT{h}'):
                seg.ctx_free_region(nm)
        self._const_stripe(seg, o_base, kq, ke, cval, 'bertO')
        for g in relays:
            yimg = self._seg_output(seg, g.y, 0, ke * g.n, kind='act_out',
                                    m=T, n=g.n, col_lo=0, col_hi=g.n,
                                    pitch=g.n, so=g.so, layout='wm',
                                    host_bias=g.host_bias, bias_key=g.wkey,
                                    module=g.module, note=f'Y {g.module}')
            yb = seg.ctx_alloc('bert:Yout', ke * g.n)
            if T % 16:
                self._prezero(seg, yb + (T // 16) * g.n, g.n)
            for j0 in range(0, g.n, cols):
                nl = min(cols, g.n - j0)
                b_base = self._emit_load_w(seg, g, j0, nl)
                self._emit_gemm(seg, T, g.n, nl, j0, kq, o_base, b_base, yb,
                                g.rq, f'BERT out.dense cols {j0}+{nl}',
                                rq_pc=g.rq_pc)
            self._emit_store(seg, yimg, yb, ke * g.n * 16, 'STORE out.dense')
            seg.ctx_free_region('bert:Yout')
        seg.notes.append('BERT OP_ATTN_S 单段')
        self.stats['attn_op_attn_s'] += 1
        if A.relay_norm is not None:
            r = A.relay_norm
            seg = self._host(seg, r['module'], r['cls'], 'norm',
                             'BERT attention.output.LayerNorm（host）')
            self._attn_meta(A, rq_qk, rqpv, sig_s, A.q.so, A.k.so,
                            [A.v.so], [relays[0].sa if relays else None],
                            extra=dict(exact_temp=True))
        return seg

    def _relay_gemm(self, rec):
        """把吞并的输出投影记录变成 WGemm（重发用）。"""
        wsh = rec['w_shape']
        g = WGemm(f'r{rec["seq"]}', rec['module'],
                  f't{rec["in_ids"][0]}', f'{rec["module"]}#{rec["seq"]}',
                  flat_len(rec['in_shapes'][0]), wsh[1], wsh[0],
                  rec['weight_key'], bool(rec.get('has_bias')),
                  src_seq=rec['seq'])
        if not self.apply_calib(g):
            return None
        return g


    # ---------- RotaryAttention：Q/K 块 host rotary 物化，S 逐头 ----------
    def _attn_rotary(self, seg, A):
        P = self.P
        cols = P['COLS']
        mq, mk, d, H, C = A.mq, A.mk, A.d, A.H, A.C
        mq16, mk16 = ceil16(mq), ceil16(mk)
        kproj = self._relay_gemm(A.out_recs[0])
        seg = self._host(seg, A.module, A.family, 'rotary',
                         'Q/K 块 rotary 物化（逐头 [m16][d]）')
        rq, rqpv, sig_s = self._attn_rq(A, A.q.so, A.k.so, A.v.so, kproj.sa)
        for h in range(H):
            if seg.descs:
                seg = self._close(seg)
            qreg = seg.ctx_alloc(f'rq{h}', mq16 * d)
            kreg = seg.ctx_alloc(f'rk{h}', mk16 * d)
            assert None not in (qreg, kreg)
            self._load_blk(seg, self._blk_in(seg, f'{A.q.y}#h{h}', mq16, d,
                         note=f'Q h{h}'), qreg, f'LOAD Q h{h}')
            self._load_blk(seg, self._blk_in(seg, f'{A.k.y}#h{h}', mk16, d,
                         note=f'K h{h}'), kreg, f'LOAD K h{h}')
            sreg = seg.ctx_alloc(f'rs{h}', mq16 * mk)
            assert sreg is not None
            if mq % 16:
                self._prezero(seg, sreg, mq16 * mk)
            for j0k in range(0, mk, cols):
                nlg = min(cols, mk - j0k)
                self._emit_copy(seg, d, nlg, kreg, d, j0k,
                                f'COPY K^T h{h} keys {j0k}+{nlg}')
                self._emit_gemm(seg, mq, mk, nlg, j0k, d,
                                qreg, 0, sreg, rq, f'QK^T h{h} {j0k}+{nlg}')
            simg = self._blk_out(seg, f'S:{A.module}#{A.seq}', mq16, mk,
                                 note=f'S h{h}')
            self._emit_store(seg, simg, sreg, mq16 * mk * 16, f'STORE S h{h}')
            for nm in (f'rq{h}', f'rk{h}', f'rs{h}'):
                seg.ctx_free_region(nm)
        seg = self._host(seg, A.module, A.family, 'softmax',
                         'S→P（host，含 mask）')
        self._attn_meta(A, rq, rqpv, sig_s, A.q.so, A.k.so,
                        [A.v.so], [kproj.sa])
        o_base = seg.ctx_alloc('ro', mq16 * kproj.k_eff)
        assert o_base is not None
        if mq % 16:
            self._prezero(seg, o_base, mq16 * kproj.k_eff)
        mkp = mk16 * 16                     # VT 列距 = 键数补 16
        for h in range(H):
            preg = seg.ctx_alloc(f'rp{h}', mq16 * mk)
            vtreg = seg.ctx_alloc(f'rv{h}', ceil16(d) * mkp)
            assert None not in (preg, vtreg)
            # P 逐头各一块：_seg_input 按名字去重，名字必须带头号
            self._load_blk(seg, self._blk_in(seg,
                         f'P:{A.module}#{A.seq}#{h}',
                         mq16, mk, kind='p_in', note=f'P h{h}'), preg,
                         f'LOAD P h{h}')
            self._load_blk(seg, self._blk_in(seg, f'{A.v.y}#vt{h}',
                         ceil16(d), mkp, kind='vt_in',
                         note=f'VT h{h}'), vtreg, f'LOAD VT h{h}')
            self._emit_copy(seg, mk, d, vtreg, mkp, 0, f'COPY V h{h}')
            self._emit_gemm(seg, mq, kproj.k_eff, d, h * d, mk, preg, 0,
                            o_base, rqpv, f'PV h{h}')
            seg.ctx_free_region(f'rp{h}')
            seg.ctx_free_region(f'rv{h}')
        self._const_stripe(seg, o_base, kproj.k_eff, mq16,
                           kproj.c_val if kproj.aug else 0, 'rotO')
        seg = self._relay_out(seg, kproj, o_base, mq, mq16)
        self.stats['attn_2ph_rotary'] += 1
        return seg

    def _relay_out(self, seg, g, a_base, m, m16):
        """段内直连输出投影：A=O 区，列组 GEMM + STORE。"""
        cols = self.P['COLS']
        yimg = self._seg_output(seg, g.y, 0, m16 * g.n, kind='act_out', m=m,
                                n=g.n, col_lo=0, col_hi=g.n, pitch=g.n,
                                so=g.so, layout='wm', host_bias=g.host_bias,
                                bias_key=g.wkey, module=g.module,
                                note=f'Y {g.module}')
        yb = seg.ctx_alloc(f'yo:{g.tag}', m16 * g.n)
        assert yb is not None
        if m % 16:                      # pad 行组：GEMM 前预清零
            gi = m // 16
            self._prezero(seg, yb + gi * g.n, g.n)
        for j0 in range(0, g.n, cols):
            nl = min(cols, g.n - j0)
            b_base = self._emit_load_w(seg, g, j0, nl)
            self._emit_gemm(seg, m, g.n, nl, j0, g.k_eff, a_base, b_base, yb,
                            g.rq, f'out {g.module} cols {j0}+{nl}',
                            rq_pc=g.rq_pc)
        self._emit_store(seg, yimg, yb, m16 * g.n * 16, f'STORE Y {g.module}')
        seg.ctx_free_region(f'yo:{g.tag}')
        return seg

    # ---------- TemporalJointGraphAttention：(unit, head) 小块 ----------
    def _attn_temporal(self, seg, A):
        P = self.P
        cols = P['COLS']
        U, mq, mk, d, H = A.units, A.mq, A.mk, A.d, A.H
        mq16, mk16 = ceil16(mq), ceil16(mk)
        kproj = self._relay_gemm(A.out_recs[0])
        seg = self._host(seg, A.module, A.family, 'rotary',
                         'temporal rotary + Q/K 块物化（逐 (j,h)）')
        rq, rqpv, sig_s = self._attn_rq(A, A.q.so, A.k.so, A.v.so, kproj.sa)
        qreg = seg.ctx_alloc('tq', mq16 * d)
        kreg = seg.ctx_alloc('tk', mk16 * d)
        sreg = seg.ctx_alloc('ts', mq16 * mk)
        assert None not in (qreg, kreg, sreg)
        if mq % 16:
            self._prezero(seg, sreg, mq16 * mk)
        for j in range(U):
            for h in range(H):
                self._load_blk(seg, self._blk_in(seg,
                             f'{A.q.y}#h{h}u{j}', mq16, d,
                             note=f'Q u{j}h{h}'), qreg, f'LOAD Q u{j}h{h}')
                self._load_blk(seg, self._blk_in(seg,
                             f'{A.k.y}#h{h}u{j}', mk16, d,
                             note=f'K u{j}h{h}'), kreg, f'LOAD K u{j}h{h}')
                for j0k in range(0, mk, cols):
                    nlg = min(cols, mk - j0k)
                    self._emit_copy(seg, d, nlg, kreg, d, j0k,
                                    f'COPY K^T u{j}h{h} keys {j0k}+{nlg}')
                    self._emit_gemm(seg, mq, mk, nlg, j0k, d, qreg, 0, sreg,
                                    rq, f'QK^T u{j}h{h} {j0k}+{nlg}')
                simg = self._blk_out(seg, f'S:{A.module}#{A.seq}', mq16, mk,
                                     note=f'S u{j}h{h}')
                self._emit_store(seg, simg, sreg, mq16 * mk * 16,
                                 f'STORE S u{j}h{h}')
        for nm in ('tq', 'tk', 'ts'):
            seg.ctx_free_region(nm)
        seg = self._host(seg, A.module, A.family, 'softmax',
                         'S→P（host，joint_distance+temporal mask）')
        self._attn_meta(A, rq, rqpv, sig_s, A.q.so, A.k.so,
                        [A.v.so], [kproj.sa])
        o_base = seg.ctx_alloc('to', U * mq16 * kproj.k_eff)
        assert o_base is not None
        if mq % 16:
            self._prezero(seg, o_base, U * mq16 * kproj.k_eff)
        mkp = mk16 * 16                     # VT 列距 = 键数补 16
        preg = seg.ctx_alloc('tp', mq16 * mk)
        vtreg = seg.ctx_alloc('tv', ceil16(d) * mkp)
        assert None not in (preg, vtreg)
        for j in range(U):
            for h in range(H):
                # P/VT 逐 (u,h)/逐头：输入条目按名字去重，名字必须带索引
                self._load_blk(seg, self._blk_in(seg,
                             f'P:{A.module}#{A.seq}#u{j}h{h}', mq16, mk,
                             kind='p_in', note=f'P u{j}h{h}'), preg,
                               f'LOAD P u{j}h{h}')
                self._load_blk(seg, self._blk_in(seg,
                             f'VT:{A.module}#{A.seq}#h{h}', ceil16(d), mkp,
                             kind='vt_in', note=f'VT u{j}h{h}'), vtreg,
                               f'LOAD VT u{j}h{h}')
                self._emit_copy(seg, mk, d, vtreg, mkp, 0,
                                f'COPY V u{j}h{h}')
                self._emit_gemm(seg, mq, kproj.k_eff, d, h * d, mk, preg,
                                0, o_base + j * mq16 * kproj.k_eff, rqpv,
                                f'PV u{j}h{h}')
        for nm in ('tp', 'tv'):
            seg.ctx_free_region(nm)
        self._const_stripe(seg, o_base, kproj.k_eff, U * mq16,
                           kproj.c_val if kproj.aug else 0, 'tmpO')
        seg = self._relay_out(seg, kproj, o_base, U * mq, U * mq16)
        self.stats['attn_2ph_temporal'] += 1
        return seg

    # ---------- JointGraphAttention：小块单段两相 ----------
    def _attn_jg(self, seg, A):
        mq, mk, d, H = A.mq, A.mk, A.d, A.H
        mq16, mk16 = ceil16(mq), ceil16(mk)
        kproj = self._relay_gemm(A.out_recs[0])
        seg = self._host(seg, A.module, A.family, 'softmax_prep',
                         'Q/K 块物化（query_pos 走 softmax bias）')
        rq, rqpv, sig_s = self._attn_rq(A, A.q.so, A.k.so, A.v.so, kproj.sa)
        qreg = seg.ctx_alloc('jq', mq16 * d)
        kreg = seg.ctx_alloc('jk', mq16 * d)
        sreg = seg.ctx_alloc('js', mq16 * mk)
        assert None not in (qreg, kreg, sreg)
        if mq % 16:
            self._prezero(seg, sreg, mq16 * mk)
        for h in range(H):
            self._load_blk(seg, self._blk_in(seg, f'{A.q.y}#h{h}', mq16, d,
                         note=f'Q h{h}'), qreg, f'LOAD Q h{h}')
            self._load_blk(seg, self._blk_in(seg, f'{A.k.y}#h{h}', mk16, d,
                         note=f'K h{h}'), kreg, f'LOAD K h{h}')
            self._emit_copy(seg, d, mk, kreg, d, 0, f'COPY K^T h{h}')
            self._emit_gemm(seg, mq, mk, mk, 0, d, qreg, 0, sreg, rq,
                            f'QK^T h{h}')
            simg = self._blk_out(seg, f'S:{A.module}#{A.seq}', mq16, mk,
                                 note=f'S h{h}')
            self._emit_store(seg, simg, sreg, mq16 * mk * 16, f'STORE S h{h}')
        for nm in ('jq', 'jk', 'js'):
            seg.ctx_free_region(nm)
        seg = self._host(seg, A.module, A.family, 'softmax',
                         'S→P（host，+query_pos bias）')
        self._attn_meta(A, rq, rqpv, sig_s, A.q.so, A.k.so,
                        [A.v.so], [kproj.sa])
        o_base = seg.ctx_alloc('jo', mq16 * kproj.k_eff)
        assert o_base is not None
        if mq % 16:
            self._prezero(seg, o_base, mq16 * kproj.k_eff)
        mqp = mq16 * 16                     # VT 列距 = 键数补 16
        preg = seg.ctx_alloc('jp', mq16 * mk)
        vtreg = seg.ctx_alloc('jv', ceil16(d) * mqp)
        for h in range(H):
            self._load_blk(seg, self._blk_in(seg,
                         f'P:{A.module}#{A.seq}#{h}',
                         mq16, mk, kind='p_in', note=f'P h{h}'), preg,
                         f'LOAD P h{h}')
            self._load_blk(seg, self._blk_in(seg, f'{A.v.y}#vt{h}',
                         ceil16(d), mqp, kind='vt_in', note=f'VT h{h}'),
                         vtreg, f'LOAD VT h{h}')
            self._emit_copy(seg, mk, d, vtreg, mqp, 0, f'COPY V h{h}')
            self._emit_gemm(seg, mq, kproj.k_eff, d, h * d, mk, preg, 0,
                            o_base, rqpv, f'PV h{h}')
        for nm in ('jp', 'jv'):
            seg.ctx_free_region(nm)
        self._const_stripe(seg, o_base, kproj.k_eff, mq16,
                           kproj.c_val if kproj.aug else 0, 'jgO')
        seg = self._relay_out(seg, kproj, o_base, mq, mq16)
        self.stats['attn_2ph_jg'] += 1
        return seg

    # ---------- MultiheadAttention：合成 in_proj + host softmax ----------
    def _attn_mha(self, seg, A):
        mq, C, H, d = A.mq, A.C, A.H, A.d
        mq16 = ceil16(mq)
        g = WGemm(f'mha{A.seq}', A.module + '.in_proj', None,
                  f'{A.module}#in_proj#{A.seq}', mq, C, 3 * C,
                  A.module + '.in_proj_weight', True, src_seq=None)
        g.host_bias = True            # fp bias 由 host 在 softmax 前补偿
        g.a = f'in:{A.module}'
        if not self.apply_calib(g):
            return seg
        g.aug = False
        seg = self._lower_wgemm(seg, g)
        seg = self._host(seg, A.module, A.family, 'softmax_prep',
                         'qkv Y 拆 Q/K/VT 块 + in_proj_bias/softmax(mask)')
        kproj = self._relay_gemm(A.out_recs[0])
        rq, rqpv, sig_s = self._attn_rq(A, g.so, g.so, g.so, kproj.sa)
        qreg = seg.ctx_alloc('mq', mq16 * d)
        kreg = seg.ctx_alloc('mk', mq16 * d)
        sreg = seg.ctx_alloc('ms', mq16 * mq)
        assert None not in (qreg, kreg, sreg)
        if mq % 16:
            self._prezero(seg, sreg, mq16 * mq)
        for h in range(H):
            self._load_blk(seg, self._blk_in(seg, f'{g.y}#q{h}', mq16, d,
                         note=f'Q h{h}'), qreg, f'LOAD Q h{h}')
            self._load_blk(seg, self._blk_in(seg, f'{g.y}#k{h}', mq16, d,
                         note=f'K h{h}'), kreg, f'LOAD K h{h}')
            self._emit_copy(seg, d, mq, kreg, d, 0, f'COPY K^T h{h}')
            self._emit_gemm(seg, mq, mq, mq, 0, d, qreg, 0, sreg, rq,
                            f'QK^T h{h}')
            simg = self._blk_out(seg, f'S:{A.module}#{A.seq}', mq16, mq,
                                 note=f'S h{h}')
            self._emit_store(seg, simg, sreg, mq16 * mq * 16, f'STORE S h{h}')
        for nm in ('mq', 'mk', 'ms'):
            seg.ctx_free_region(nm)
        seg = self._host(seg, A.module, A.family, 'softmax', 'S→P（host）')
        self._attn_meta(A, rq, rqpv, sig_s, g.so, g.so, [g.so], [kproj.sa])
        o_base = seg.ctx_alloc('mo', mq16 * kproj.k_eff)
        assert o_base is not None
        self._prezero(seg, o_base, mq16 * kproj.k_eff)
        mqp = mq16 * 16                     # VT 列距 = 键数补 16
        preg = seg.ctx_alloc('mp', mq16 * mq)
        vtreg = seg.ctx_alloc('mv', ceil16(d) * mqp)
        for h in range(H):
            self._load_blk(seg, self._blk_in(seg,
                         f'P:{A.module}#{A.seq}#{h}',
                         mq16, mq, kind='p_in', note=f'P h{h}'), preg,
                         f'LOAD P h{h}')
            self._load_blk(seg, self._blk_in(seg, f'{g.y}#vt{h}', ceil16(d),
                         mqp, kind='vt_in', note=f'VT h{h}'), vtreg,
                         f'LOAD VT h{h}')
            self._emit_copy(seg, mq, d, vtreg, mqp, 0, f'COPY V h{h}')
            self._emit_gemm(seg, mq, kproj.k_eff, d, h * d, mq, preg, 0,
                            o_base, rqpv, f'PV h{h}')
        for nm in ('mp', 'mv'):
            seg.ctx_free_region(nm)
        self._const_stripe(seg, o_base, kproj.k_eff, mq16, 0, 'mhaO')
        seg = self._relay_out(seg, kproj, o_base, mq, mq16)
        self.stats['attn_2ph_mha'] += 1
        return seg

    # ---------- BiMultiHeadAttention：双向 4 GEMM/头 ----------
    def _attn_bimha(self, seg, A):
        P = self.P
        cols = P['COLS']
        Nv, Nl, d, H = A.mq, A.mk, A.d, A.H
        Nv16, Nl16 = ceil16(Nv), ceil16(Nl)
        # 生产者（v_proj/l_proj/values_*）已按通用 WGemm 出 Y 图
        seg = self._host(seg, A.module, A.family, 'softmax_prep',
                         'Q/K/VT 块物化（E=1024 拆头 d=256）')
        ac = self.acal.get(A.module)
        if ac and ac.get('s_absmax') and min(A.q.so, A.k.so) > 0:
            rq = self._enc_r(A.q.so * A.k.so / (ac['s_absmax'] / 127.0))
        else:
            rq = self._ph_rq(d)
        qv = seg.ctx_alloc('bqv', Nv16 * d)
        ql = seg.ctx_alloc('bql', Nl16 * d)
        kv = seg.ctx_alloc('bkv', Nv16 * d)
        kl = seg.ctx_alloc('bkl', Nl16 * d)
        for nm, e in (('bqv', qv), ('bql', ql), ('bkv', kv), ('bkl', kl)):
            assert e is not None, nm
        sv = seg.ctx_alloc('bsv', Nv16 * Nl)
        sl = seg.ctx_alloc('bsl', Nl16 * Nv)
        assert None not in (sv, sl)
        if Nv % 16:
            self._prezero(seg, sv, Nv16 * Nl)
        if Nl % 16:
            self._prezero(seg, sl, Nl16 * Nv)
        for h in range(H):
            for nm, src, r16 in (('Qv', A.q, Nv16), ('Kv', A.q, Nv16),
                                 ('Ql', A.k, Nl16), ('Kl', A.k, Nl16)):
                reg = {'Qv': qv, 'Kv': kv, 'Ql': ql, 'Kl': kl}[nm]
                self._load_blk(seg, self._blk_in(seg,
                             f'{src.y}#{nm}{h}', r16, d,
                             note=f'{nm} h{h}'), reg, f'LOAD {nm} h{h}')
            self._emit_copy(seg, d, Nl, kl, d, 0, f'COPY K_l^T h{h}')
            self._emit_gemm(seg, Nv, Nl, Nl, 0, d, qv, 0, sv, rq,
                            f'QK^T S_v h{h}')
            self._emit_copy(seg, d, Nv, kv, d, 0, f'COPY K_v^T h{h}')
            self._emit_gemm(seg, Nl, Nv, Nv, 0, d, ql, 0, sl, rq,
                            f'QK^T S_l h{h}')
            # S 必须逐头落盘：reg 每头被覆盖，循环外存只会剩最后一个头
            for nm, reg, r16, cc in (('S_v', sv, Nv16, Nl),
                                     ('S_l', sl, Nl16, Nv)):
                img = self._blk_out(seg, f'S{nm}:{A.module}#{A.seq}', r16, cc,
                                    note=f'{nm} h{h}')
                self._emit_store(seg, img, reg, r16 * cc * 16,
                                 f'STORE {nm} h{h}')
        for nm in ('bqv', 'bql', 'bkv', 'bkl', 'bsv', 'bsl'):
            seg.ctx_free_region(nm)
        seg = self._host(seg, A.module, A.family, 'softmax',
                         'S_v/S_l→P（host，attention_mask_l）')
        kv1, kl1 = self._relay_gemm(A.out_recs[0]), self._relay_gemm(A.out_recs[1])
        # 交叉 PV：O_v = P_v@V_l（V_l=values_l_proj），O_l = P_l@V_v
        rq, rqpv_v, sig_s = self._attn_rq(A, A.q.so, A.k.so,
                                          A.geo['v_l'].so, kv1.sa)
        rqpv_l = self._enc_r(A.v.so / (127.0 * kl1.sa)) \
            if kl1.sa > 0 else self._ph_rq(Nv16)
        self._attn_meta(A, rq, [rqpv_v, rqpv_l], sig_s, A.q.so, A.k.so,
                        [A.geo['v_l'].so, A.v.so], [kv1.sa, kl1.sa],
                        extra=dict(pv_names=['O_v', 'O_l']))
        ov = seg.ctx_alloc('bov', Nv16 * kv1.k_eff)
        ol = seg.ctx_alloc('bol', Nl16 * kl1.k_eff)
        assert None not in (ov, ol)
        self._prezero(seg, ov, Nv16 * kv1.k_eff)
        self._prezero(seg, ol, Nl16 * kl1.k_eff)
        Nlp, Nvp = Nl16 * 16, Nv16 * 16      # VT 列距 = 键数补 16
        pv = seg.ctx_alloc('bpv', Nv16 * Nl)
        vl = seg.ctx_alloc('bvl', Nl16 * Nv)
        assert None not in (pv, vl)
        for h in range(H):
            self._load_blk(seg, self._blk_in(seg,
                         f'P_v:{A.module}#{A.seq}#{h}', Nv16, Nl, kind='p_in',
                         note=f'P_v h{h}'), pv, f'LOAD P_v h{h}')
            self._load_blk(seg, self._blk_in(seg,
                         f'P_l:{A.module}#{A.seq}#{h}', Nl16, Nv, kind='p_in',
                         note=f'P_l h{h}'), vl, f'LOAD P_l h{h}')
            vtv = seg.ctx_alloc(f'bvtv{h}', ceil16(d) * Nlp)
            vtl = seg.ctx_alloc(f'bvtl{h}', ceil16(d) * Nvp)
            for j0v in range(0, d, cols):
                nl = min(cols, d - j0v)
                # O_v = P_v · V_l（交叉：VT_v 图填 lang 值 V_l，k=Nl）
                self._load_blk(seg, self._blk_in(seg,
                             f'VT_v:{A.module}#{A.seq}#{h}', ceil16(d), Nlp,
                             kind='vt_in', note=f'VT_v h{h}'), vtv,
                             f'LOAD VT_v h{h}')
                self._emit_copy(seg, Nl, nl, vtv, Nlp, j0v,
                                f'COPY V_v h{h} {j0v}+{nl}')
                self._emit_gemm(seg, Nv, kv1.k_eff, nl, h * d + j0v, Nl,
                                pv, 0, ov, rqpv_v, f'PV O_v h{h} {j0v}+{nl}')
                self._load_blk(seg, self._blk_in(seg,
                             f'VT_l:{A.module}#{A.seq}#{h}', ceil16(d), Nvp,
                             kind='vt_in', note=f'VT_l h{h}'), vtl,
                             f'LOAD VT_l h{h}')
                self._emit_copy(seg, Nv, nl, vtl, Nvp, j0v,
                                f'COPY V_l h{h} {j0v}+{nl}')
                self._emit_gemm(seg, Nl, kl1.k_eff, nl, h * d + j0v, Nv,
                                vl, 0, ol, rqpv_l, f'PV O_l h{h} {j0v}+{nl}')
            for nm in (f'bvtv{h}', f'bvtl{h}'):
                seg.ctx_free_region(nm)
        for nm in ('bpv', 'bvl'):
            seg.ctx_free_region(nm)
        self._const_stripe(seg, ov, kv1.k_eff, Nv16,
                           kv1.c_val if kv1.aug else 0, 'biOv')
        self._const_stripe(seg, ol, kl1.k_eff, Nl16,
                           kl1.c_val if kl1.aug else 0, 'biOl')
        seg = self._relay_out(seg, kv1, ov, Nv, Nv16)
        seg = self._relay_out(seg, kl1, ol, Nl, Nl16)
        self.stats['attn_2ph_bimha'] += 1
        return seg

    # ---------- WindowMSA：窗口批两相（Q/K/VT 块 host 物化） ----------
    def _attn_swin(self, seg, A):
        P = self.P
        cols = P['COLS']
        W, H, T, C = A.units, A.H, A.mq, A.C
        d = C // H
        A.d = d                    # pre_scan 只给了 H/C，补 d 供 requant 用
        T16 = ceil16(T)
        Tp = T16 * 16              # VT 列距 = 键数补 16
        kproj = self._relay_gemm(A.out_recs[0])
        Cp = kproj.k_eff
        # qkv 由节点遍历里的独立 WGemm 节点lower（在 WindowMSA 实例之前），
        # 此处不重发，只依赖其 Y 输出图（host swin_split 拆 Q/K/VT 块）。
        # 迭代单位 = 伪窗 pw = w*H + h（窗 w 头 h）：块行序按伪窗排，
        # Q/K 寻址用行号 pw*T16*16、A 基址用字偏移 pw*T16*d。
        seg = self._host(seg, A.module, A.family, 'swin_split',
                         f'qkv Y 拆 Q/K/VT 块（{W} 窗 × {H} 头伪窗，'
                         f'16 行对齐补 0）')
        rq, rqpv, sig_s = self._attn_rq(A, A.qkv.so, A.qkv.so, A.qkv.so,
                                        kproj.sa)
        PW = W * H                                  # 伪窗总数
        est1 = max(1, -(-T // cols)) * 2 + 2        # 每伪窗 COPY+GEMM+STORE
        seq1 = max(1, (P['SEQ_N'] - 2 - 8) // est1)
        bw = max(1, min(PW, (P['DDR_BYTES'] - P['ZERO_SLOT'] * 16 - 2048) //
                        max(1, (T16 * d * 2 + T16 * T) * 16), 96, seq1))
        for pw0 in range(0, PW, bw):
            nb = min(bw, PW - pw0)
            if seg.descs:
                seg = self._close(seg)
            qreg = seg.ctx_alloc('sq', nb * T16 * d)
            kreg = seg.ctx_alloc('sk', nb * T16 * d)
            sreg = seg.ctx_alloc('ss', T16 * T)
            assert None not in (qreg, kreg, sreg)
            if T % 16:
                self._prezero(seg, sreg, T16 * T)
            qimg = self._blk_in(seg, f'{A.qkv.y}#Q', nb * T16, d,
                                kind='blk_in', note=f'Q 块 批{pw0}+{nb}')
            kimg = self._blk_in(seg, f'{A.qkv.y}#K', nb * T16, d,
                                kind='blk_in', note=f'K 块 批{pw0}+{nb}')
            self._load_blk(seg, qimg, qreg, f'LOAD Q 批{pw0}')
            self._load_blk(seg, kimg, kreg, f'LOAD K 批{pw0}')
            for pw in range(nb):
                w, h = divmod(pw0 + pw, H)
                for j0k in range(0, T, cols):
                    nlg = min(cols, T - j0k)
                    self._emit_copy(seg, d, nlg, kreg, d,
                                    pw * T16 * 16 + j0k,
                                    f'COPY K^T w{w}h{h}')
                    self._emit_gemm(seg, T, T, nlg, j0k, d,
                                    qreg + pw * T16 * d, 0, sreg, rq,
                                    f'QK^T w{w}h{h} {j0k}+{nlg}')
                simg = self._blk_out(seg, f'S:{A.module}#{A.seq}',
                                     T16, T, note=f'S w{w}h{h}')
                self._emit_store(seg, simg, sreg, T16 * T * 16,
                                 f'STORE S w{w}h{h}')
            for nm in ('sq', 'sk', 'ss'):
                seg.ctx_free_region(nm)
        seg = self._host(seg, A.module, A.family, 'softmax',
                         'S→P（host，+相对位置 bias 表）')
        self._attn_meta(A, rq, rqpv, sig_s, A.qkv.so, A.qkv.so,
                        [A.qkv.so], [kproj.sa])
        nwg = -(-C // cols)                        # proj 权重图（段内复用）
        wu = ((Cp * cols + 7) // 8 * 8) * nwg + 2048 * nwg
        est2 = 2 + T16 + 2 * nwg + 4     # 每伪窗 COPY/PV+窗尾条纹/投影/存
        seq2 = max(1, (P['SEQ_N'] - 2 - 16) // est2)
        bw2 = max(1, min(PW, (P['DDR_BYTES'] - P['ZERO_SLOT'] * 16 - 2048 - wu)
                         // max(1, (T16 * T + ceil16(d) * Tp + T16 * C) * 16),
                         64, seq2))
        pw0 = 0
        while pw0 < PW:
            nb = min(bw2, PW - pw0)
            if pw0 + nb < PW and nb > H:
                nb = (nb // H) * H   # 批界对齐窗：oreg 的各头列不能跨段
            if seg.descs:
                seg = self._close(seg)
            preg = seg.ctx_alloc('pp', nb * T16 * T)
            oreg = seg.ctx_alloc('po', T16 * Cp)
            yreg = seg.ctx_alloc('py', T16 * C)
            assert None not in (preg, oreg, yreg)
            pimg = self._blk_in(seg, f'P:{A.module}#{A.seq}', nb * T16, T,
                                kind='p_in', note=f'P 批{pw0}+{nb}')
            self._load_blk(seg, pimg, preg, f'LOAD P 批{pw0}')
            if T % 16:
                self._prezero(seg, oreg, T16 * Cp)
                self._prezero(seg, yreg + (T // 16) * C, C)
            vtrow = ceil16(nb * d)                 # VT 行组数
            vreg = seg.ctx_alloc('pv', vtrow * Tp)
            vtimg = self._blk_in(seg, f'{A.qkv.y}#VT', vtrow, Tp,
                                 kind='vt_in', note=f'VT 块 批{pw0}+{nb}')
            self._load_blk(seg, vtimg, vreg, f'LOAD VT 批{pw0}')
            for pw in range(nb):
                w, h = divmod(pw0 + pw, H)
                # PV：COPY 会覆盖 WRAM 基址，proj 权重须在本窗 PV 后现载
                self._emit_copy(seg, T, d, vreg, Tp, pw * d,
                                f'COPY V w{w}h{h}')
                self._emit_gemm(seg, T, Cp, d, h * d, T,
                                preg + pw * T16 * T, 0, oreg, rqpv,
                                f'PV w{w}h{h}')
                if h == H - 1:                     # 本窗全部头完成 → 投影
                    self._const_stripe(seg, oreg, Cp, T16,
                                       kproj.c_val if kproj.aug else 0,
                                       f'swO{w}')
                    for j0 in range(0, C, cols):
                        nl = min(cols, C - j0)
                        wb = self._emit_load_w(seg, kproj, j0, nl)
                        self._emit_gemm(seg, T, C, nl, j0, Cp, oreg, wb, yreg,
                                        kproj.rq, f'proj w{w} {j0}+{nl}',
                                        rq_pc=kproj.rq_pc)
                    yimg = self._blk_out(seg, f'Y:{A.module}#{A.seq}', T16, C,
                                         note=f'w_msa Y w{w}')
                    self._emit_store(seg, yimg, yreg, T16 * C * 16,
                                     f'STORE Y w{w}')
            for nm in ('pp', 'po', 'py', 'pv'):
                seg.ctx_free_region(nm)
            pw0 += nb
        self.stats['attn_2ph_swin'] += 1
        return seg

    # ================= 顶层 lower / emit =================
    FAM_FN = {'WindowMSA': '_attn_swin', 'BertAttention': '_attn_bert',
              'RotaryAttention': '_attn_rotary',
              'TemporalJointGraphAttention': '_attn_temporal',
              'JointGraphAttention': '_attn_jg',
              'MultiheadAttention': '_attn_mha',
              'BiMultiHeadAttention': '_attn_bimha'}

    def lower(self):
        seg = self._fresh_seg()
        self.node_map = []      # 节点 → 段范围 + 驱动元数据（host_driver 用）
        for nd in self.nodes:
            s0 = len(self.segments)
            if isinstance(nd, HostOp):
                seg = self._host(seg, nd.module, nd.cls, nd.kind, nd.note)
                self.node_map.append(dict(kind='host', module=nd.module,
                                          cls=nd.cls, host_kind=nd.kind,
                                          seq=nd.seq, segs=[]))
            elif isinstance(nd, WGemm):
                hm = getattr(nd, 'heads_mode', None)
                if hm:
                    Hh, dd, twin = hm
                    if seg.descs:
                        seg = self._close(seg)
                    seg = self._lower_wgemm_heads(seg, nd, Hh, dd, twin,
                                                  nd.tag)
                else:
                    seg = self._lower_wgemm(seg, nd)
                self.node_map.append(dict(
                    kind='gemm', module=nd.module, tag=nd.tag, seq=nd.src_seq,
                    in_graph=nd.a, out_graph=nd.y, m=nd.m, k=nd.k, n=nd.n,
                    sa=nd.sa, so=nd.so, rq=[nd.rq[0], nd.rq[1]], aug=nd.aug,
                    host_bias=nd.host_bias, heads_mode=(list(hm) if hm
                                                        else None),
                    segs=[self.segments[i].name for i in range(s0,
                                                               len(self.segments))]))
            else:
                fn = self.FAM_FN.get(nd.family)
                assert fn, f'未实现家族 {nd.family}'
                seg = getattr(self, fn)(seg, nd)
                # 注意力边界：注意力段常被 wrapper 的残差加等未 trace 的
                # functional 隔开，紧随其后的 GEMM（如 MSDA value_proj）
                # 的实参要到它自己 forward 时才存在。不关段会把它们拼进
                # 注意力段，驱动在注意力触发时拿不到输入。
                if seg.descs:
                    seg = self._close(seg)
                self.node_map.append(dict(
                    kind='attn', module=nd.module, family=nd.family,
                    seq=nd.seq, out_graphs=[r['module'] + '#' + str(r['seq'])
                                            for r in nd.out_recs],
                    segs=[self.segments[i].name for i in range(s0,
                                                               len(self.segments))]))
        if seg.descs:
            self._finalize(seg)
        else:
            self.segments.remove(seg)
        return self.segments

    def emit(self):
        import struct
        os.makedirs(self.out, exist_ok=True)
        segdir = os.path.join(self.out, 'segments')
        os.makedirs(segdir, exist_ok=True)
        idx = []
        for seg in self.segments:
            d = os.path.join(segdir, seg.name)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, 'seq.mem'), 'w') as f:
                for v, note in seg.descs:
                    f.write(f'{v:064X}\n')
            man = dict(seg=seg.idx, name=seg.name, n_descs=len(seg.descs),
                       weights=seg.weights, inputs=seg.inputs,
                       outputs=seg.outputs,
                       zero_addr=seg.zero_addr,
                       zero_words=self.P['ZERO_SLOT'] if seg.zero_addr
                       is not None else 0,
                       est_cycles=getattr(seg, 'est_cycles', 0),
                       macs=seg.macs, profile=self.P, notes=seg.notes)
            with open(os.path.join(d, 'manifest.json'), 'w') as f:
                json.dump(man, f, indent=1, ensure_ascii=False)
            idx.append(dict(name=seg.name, n_descs=len(seg.descs),
                            est_cycles=man['est_cycles'],
                            w_bytes=seg.w_bytes,
                            n_in=len(seg.inputs), n_out=len(seg.outputs)))
        with open(os.path.join(self.out, 'weights_blob.bin'), 'wb') as f:
            f.write(self.blob)
        with open(os.path.join(self.out, 'host_plan.json'), 'w') as f:
            json.dump(dict(segments=idx, host_steps=self.host_steps,
                           nodes=getattr(self, 'node_map', []),
                           warnings=self.warnings), f, indent=1,
                      ensure_ascii=False)
        st = dict(self.stats)
        st['segments'] = len(self.segments)
        st['blob_bytes'] = len(self.blob)
        st['host_steps'] = len(self.host_steps)
        st['host_bias_total'] = st.get('host_bias_fp_fallback', 0) + \
            st.get('host_bias_k_too_big', 0)
        st['est_ms_at_198p5M'] = st.get('total_cycles', 0) / F_SYN * 1e3
        with open(os.path.join(self.out, 'model_summary.json'), 'w') as f:
            json.dump(st, f, indent=1, ensure_ascii=False)
        return st


def main():
    import argparse
    ap = argparse.ArgumentParser(description='HB-GD 分段指令流编译器 v0')
    ap.add_argument('--trace', default=os.path.join(HERE, 'ops_trace.json'))
    ap.add_argument('--manifest',
                    default=os.path.join(HERE, 'manifest.json'))
    ap.add_argument('--w8', default=os.path.join(HERE, 'w8_export'))
    ap.add_argument('--calib',
                    default=os.path.join(HERE, '..', '02_quant',
                                        'hw_calib_table.json'))
    ap.add_argument('--attn-calib', default=None,
                    help='attn_calib.py 产物（注意力 σ_S），两相 QK^T/PV '
                         'requant 用；缺省退占位')
    ap.add_argument('--rq-max-s', type=int, default=47,
                    help='requant 移位上限。现网 RTL rq_v2 T_MAX=0 只支持 '
                         's=8（对拍现网 RTL 用 8）；桶形版用默认 47')
    ap.add_argument('--pcw-w8', default=None,
                    help='逐通道 INT8 权重目录（pcw_export.py 产物）。给了且 '
                         '校准表条目带 rq_ms_col 时，linear/mha_in_proj 权重'
                         '改从该目录装载（其余仍用 --w8）')
    ap.add_argument('--no-rtn', action='store_true',
                    help='关掉 RTN 就近舍入（rq_m bit15 不置位，回到纯 floor'
                         '——只用于消融对照，pcW 列系数不受影响）')
    ap.add_argument('--exempt-extra', default='',
                    help='杠杆 F：额外豁免前缀（逗号分隔，模块名 startswith '
                         '匹配）。命中的 GEMM 整层 host fp，零 RTL 改动')
    ap.add_argument('--profile', choices=['smoke', 'full'], default='full')
    ap.add_argument('--out', default=None)
    ap.add_argument('--limit', type=int, default=0,
                    help='只编译前 N 个节点（调试）')
    a = ap.parse_args()
    if a.exempt_extra:
        EXEMPT_EXTRA.extend(p.strip() for p in a.exempt_extra.split(',')
                            if p.strip())
    P = dict(PROFILES[a.profile])
    out = a.out or os.path.join(HERE, f'build_{a.profile}')
    trace = json.load(open(a.trace, encoding='utf-8'))
    man = json.load(open(a.manifest, encoding='utf-8'))
    calib = None
    if os.path.exists(a.calib):
        calib = json.load(open(a.calib, encoding='utf-8'))
    acal = None
    if a.attn_calib and os.path.exists(a.attn_calib):
        acal = json.load(open(a.attn_calib, encoding='utf-8'))
    c = Compiler(trace, man, a.w8, calib, P, out, acal=acal,
                 rq_max_s=a.rq_max_s, pcw_w8dir=a.pcw_w8,
                 rtn=not a.no_rtn)
    c.build_ir()
    if a.limit:
        c.nodes = c.nodes[:a.limit]
    c.lower()
    st = c.emit()
    print(f'[compiler] profile={a.profile} 段={st["segments"]} '
          f'描述符={st["d_total"]} 最大SEQ={st["max_seq"]} '
          f'权重blob={st["blob_bytes"] / 1e6:.2f}MB '
          f'预估={st["est_ms_at_198p5M"]:.2f}ms@198.5MHz')
    print(f'[compiler] GEMM={st["d_gemm"]} COPY={st["d_copy"]} '
          f'LOADW={st["d_load_w"]} LOADCTX={st["d_load_ctx"]} '
          f'STORE={st["d_store"]}')
    print(f'[compiler] pcW层={st.get("pcw_layers", 0)} '
          f'host_bias(pcW)={st.get("host_bias_pcw", 0)} '
          f'OP_SF={st.get("d_sf", 0)} '
          f'rq_s抬升={st.get("rq_s_bumped", 0)} '
          f'RTN={"off" if a.no_rtn else "on"}')
    print(f'[compiler] 层统计: aug={st.get("aug_layers")} '
          f'host_bias={st["host_bias_total"]} '
          f'(fp_fallback={st.get("host_bias_fp_fallback")},'
          f'k大={st.get("host_bias_k_too_big")}) '
          f'无bias={st.get("no_bias_layers")} '
          f'豁免={st.get("exempt_host_gemm")} '
          f'host_gemm_k超限={st.get("host_gemm_k_too_big")}')
    print(f'[compiler] 注意力: OP_ATTN_S={st.get("attn_op_attn_s", 0)} '
          f'两相GEMM+hostSM: rotary={st.get("attn_2ph_rotary", 0)} '
          f'temporal={st.get("attn_2ph_temporal", 0)} '
          f'jg={st.get("attn_2ph_jg", 0)} mha={st.get("attn_2ph_mha", 0)} '
          f'bimha={st.get("attn_2ph_bimha", 0)} swin={st.get("attn_2ph_swin", 0)} '
          f'MSDeform(host)={st.get("host_attn_msdeform", 0)} '
          f'requant校准={st.get("attn_rq_calibrated", 0)} '
          f'占位={st.get("attn_rq_placeholder", 0)}')
    if c.warnings:
        print(f'[compiler] 警告 {len(c.warnings)} 条（见 host_plan.json）')


if __name__ == '__main__':
    main()
