# -*- coding: utf-8 -*-
"""est_check.py — 对账用：对 441 个类型代表段重算三档 est（本地跑）

est_v0      : 编译器 v0 口径 = est_desc_cycles(原始描述符)，dma_len 只读 18 位
              窄值（溢出即低估），gemm mt 双重除法（16 倍低估）→ 与 manifest
              est_cycles 对得上，总和 ≈ 313M。
est_lenfix  : 在修复后的描述符流上跑同一模型（每分片长度真实），mt 仍双重除法。
              → 单独看"编码溢出 + 分片开销"对 est 的影响。
est_fixed   : 修正 mt = ceil16(m)（单次除法）后的引擎模型，长度同上。
              → 修完两个 bug 后 est 与 RTL 实测的剩余差 = 模型没建模的
              （LFSR DDR 读停顿、命令间隔、调度气泡等）。

输出 est_by_type.json：[{type_id, rep, n_instances, est_v0, est_lenfix,
est_fixed, est_manifest, macs_v0, macs_padded, macs_useful}]，供 aggregate.py 用。

macs 三口径（对账 RTL mac_total）：
  macs_v0      : compiler 原式 (ceil16(m)//16)*16*COLS*k —— 双重除法，16 倍低
  macs_padded  : 修正后 ceil16(m)*16*COLS*k，与 RTL 语义一致（每 k 切片记
                 16×108，不看实际 n）—— 应与 RTL mac_total 全等
  macs_useful  : m*n_loc*k，去 padding 的有效 MAC（对照算法侧 GMAC 数）
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, '..', '03_compiler', 'build_full')
FIXED = os.path.join(HERE, '..', '03_compiler', 'build_full_fixed')

ROWS, DRAIN, DALIGN, GEMM_CMD_OVH = 16, 64, 2, 2
BURST_B, AR_OVH, CMD_OVH = 2048, 2, 5


def ceil16(x):
    return (x + 15) // 16


def _wb_cycles(n_loc, j0, y_tr):
    if not y_tr:
        return n_loc
    return 16 * (((j0 + n_loc - 1) >> 4) - (j0 >> 4) + 1)


def gemm_cycles(m, k, n_loc, j0, y_tr, cols, mt_fix):
    mt = ceil16(m) if mt_fix else ceil16(m) // 16
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
    return -(-nbytes // 16) * 5 + CMD_OVH


def copy_cycles(k_rows, j_cols, src_j0):
    grps = ((src_j0 + j_cols - 1) >> 4) - (src_j0 >> 4) + 1
    return 2 + 3 * k_rows * grps


def softmax_cycles(m_rows, n_cols, causal):
    tot = 2
    rg = 0
    while rg * 16 < m_rows:
        g_end = min(rg * 16 + 16, m_rows)
        glen = min(n_cols, g_end) if causal else n_cols
        tot += 2 * glen + n_cols + 49
        rg += 1
    return tot


def est_desc_cycles(d, cols, mt_fix):
    """与 compiler.est_desc_cycles 逐字段同口径；mt_fix=True 修双重除法。
    dma_len 恒用 18 位窄值（est_lenfix/est_fixed 喂的是修复后流，窄值=真值）。"""
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
        g = gemm_cycles(m, k, n_loc, j0, y_tr, cols, mt_fix)
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
    return 0


def load_descs(root, seg):
    p = os.path.join(root, 'segments', 'seg_%04d' % seg, 'seq.mem')
    return [int(l, 16) for l in open(p) if l.strip()]


def macs_of(descs, cols=108):
    """返回 (macs_v0, macs_padded, macs_useful)。"""
    v0 = pad = use = 0
    for d in descs:
        op = (d >> 252) & 0xF
        if op not in (0, 1, 2):
            continue
        m = (d >> 228) & 0xFFFF
        k = (d >> 196) & 0xFFFF
        n_loc = (d >> 120) & 0xFFFF
        v0 += (ceil16(m) // 16) * 16 * cols * k
        pad += ceil16(m) * 16 * cols * k
        use += m * n_loc * k
    return v0, pad, use


def main():
    types = json.load(open(os.path.join(HERE, 'types.json')))['types']
    rows = []
    for t in types:
        seg = t['rep']
        d_orig = load_descs(ORIG, seg)
        d_fix = load_descs(FIXED, seg)
        man = json.load(open(os.path.join(
            FIXED, 'segments', 'seg_%04d' % seg, 'manifest.json')))
        m0, mp, mu = macs_of(d_fix)
        rows.append(dict(
            type_id=t['type_id'], rep=seg, n_instances=len(t['instances']),
            est_v0=sum(est_desc_cycles(d, 108, False) for d in d_orig),
            est_lenfix=sum(est_desc_cycles(d, 108, False) for d in d_fix),
            est_fixed=sum(est_desc_cycles(d, 108, True) for d in d_fix),
            est_manifest=man['est_cycles'],
            macs_v0=m0, macs_padded=mp, macs_useful=mu))
    json.dump(rows, open(os.path.join(HERE, 'est_by_type.json'), 'w'), indent=1)
    n = len(rows)
    bad = [r for r in rows if r['est_v0'] != r['est_manifest']]
    s = lambda k: sum(r[k] * r['n_instances'] for r in rows)
    print('类型数 %d；est_v0 与 manifest 不一致 %d' % (n, len(bad)))
    for r in bad[:5]:
        print('  type %d seg %d: v0=%d manifest=%d' %
              (r['type_id'], r['rep'], r['est_v0'], r['est_manifest']))
    print('Σ est_v0      = %d（≈313M v0 口径）' % s('est_v0'))
    print('Σ est_lenfix  = %d（长度修后）' % s('est_lenfix'))
    print('Σ est_fixed   = %d（长度+mt 都修）' % s('est_fixed'))
    print('Σ macs: v0=%d padded=%d useful=%d' %
          (s('macs_v0'), s('macs_padded'), s('macs_useful')))


if __name__ == '__main__':
    main()
