# -*- coding: utf-8 -*-
"""fast_interp_a3.py — fast_interp 的 a3 版：加 op=6 AE_ACTV 三子模式
（2026-08-31）。

在 09_cbound/fast_interp.py（逐位对拍过的向量化段解释器）基础上只加
op=6 分支，其余逐字不变：

  sub=0 ACTV  原地查表：ctx[lane, y_base+(r//16)*n+j] =
              tbl[lane, ctx&0xFF]（表映像 256 字，项 x 复制到字
              b_base+x 的全部 16 lane，见 actv_gold.py）
  sub=1 BIAS  y = sat8((y·rqm + bj[j]) >> rqs)，bj 从 lo/hi 字节表拼
              （hi 有符号×256 + lo 有符号，与 actv_gold 同式）
  sub=2 NORM  逐行走 norm_gold.engine_row（本流 0 站，保守实现）

黄金对照：ACTV/BIAS 与 09_onchip_rtl/sim/actv_gold.py 的微观对拍用例
逐位一致（见 selftest_op6()）；NORM 语义唯一来源是 12_actv/spec/
norm_gold.py，直接 import。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CB = os.path.join(ROOT, '09_cbound')
sys.path.insert(0, CB)
sys.path.insert(0, os.path.join(ROOT, '12_actv', 'spec'))

from fast_interp import run_segment_fast as _run_a2   # noqa: E402,F401
from golden_interp import EXP, decode, sat8, _s8      # noqa: E402

_EXPNP = np.array(EXP, dtype=np.int64)


def _op6_actv(ctx, m, n, yb, tb):
    """sub=0：原地 LUT。行 r → lane r%16、字 yb+(r//16)*n+j；只动 r<m。"""
    tbl = ctx[:, tb:tb + 256]                        # 16×256 有符号
    g = (m + 15) // 16
    rows = np.arange(g * 16)
    lane = (rows % 16)[:, None]
    idx = yb + (rows // 16)[:, None] * n + np.arange(n)[None, :]
    keep = (rows < m)[:, None]
    new = tbl[lane, ctx[lane, idx] & 0xFF]
    ctx[lane, idx] = np.where(keep, new, ctx[lane, idx])


def _op6_bias(ctx, m, n, k, yb, tb, rqm, rqs):
    """sub=1：y = sat8((y·rqm + bj[j]) >> rqs)。"""
    if rqm >= 1 << 15:
        rqm -= 1 << 16
    j = np.arange(k)
    lo = ctx[j % 16, tb + j // 16]
    hi = ctx[j % 16, tb + (k + 15) // 16 + j // 16]
    bj = (hi << 8) + (lo & 0xFF)          # 符号位在 hi，lo 按无符号拼
    g = (m + 15) // 16
    rows = np.arange(g * 16)
    lane = (rows % 16)[:, None]
    idx = yb + (rows // 16)[:, None] * n + np.arange(n)[None, :]
    y = ctx[lane, idx]
    nj = np.minimum(np.arange(n), k - 1)             # 表短于列宽时末项复用
    y2 = sat8((y * rqm + bj[nj][None, :]) >> rqs)
    ctx[lane, idx] = np.where((rows < m)[:, None], y2, y)


def _op6_norm(ctx, m, n, yb, tb):
    """sub=2：逐行 engine_row（norm_spec 契约，语义唯一来源 norm_gold）。"""
    from norm_gold import engine_row
    for r in range(m):
        lane = r % 16
        base = yb + (r // 16) * n
        xs = [int(v) for v in ctx[lane, base:base + n]]
        c0 = base = tb
        invn = int(ctx[0, tb]) & 0xFFFFFF if False else None   # 见下
        # 常数字 word0 的 128b 拼装（lane 字节序）
        w0 = 0
        for L in range(16):
            w0 |= (int(ctx[L, tb]) & 0xFF) << (8 * L)
        invn = w0 & 0xFFFFFF
        eps_q24 = (w0 >> 24) & ((1 << 48) - 1)
        g_shift = (w0 >> 72) & 0x3F
        out_shift = (w0 >> 78) & 0xF
        ln = (w0 >> 82) & 1
        nlo = (n + 15) // 16
        g = np.array([ctx[j % 16, tb + 1 + j // 16] |
                      (ctx[j % 16, tb + 1 + nlo + j // 16] << 8)
                      for j in range(n)], dtype=np.int64)
        b = np.array([ctx[j % 16, tb + 1 + 2 * nlo + j // 16] |
                      (ctx[j % 16, tb + 1 + 3 * nlo + j // 16] << 8)
                      for j in range(n)], dtype=np.int64)
        out = engine_row(xs, invn, eps_q24, g_shift, out_shift, ln,
                         list(g), list(b))
        for j in range(n):
            v = out[j]
            ctx[lane, base + j] = v - 256 if v > 127 else v


def run_segment_fast_a3(seq, ddr_img, P):
    """与 fast_interp.run_segment_fast 相同，仅加 op=6 分支。"""
    cols, ctxw, ww = P['COLS'], P['CTX_WORDS'], P['W_WORDS']
    ddr = ddr_img.copy()
    ctx = np.zeros((16, ctxw), dtype=np.int64)
    wram = np.zeros((cols, ww), dtype=np.int64)
    macs = 0
    for pc, d in enumerate(seq):
        f = decode(d)
        op = f['op']
        if op == 15:
            break
        m, n, k = f['m'], f['n'], f['k']
        if op == 4:                                    # LOAD
            nB = f['dma_len']
            B = _s8(ddr[f['dma_addr']:f['dma_addr'] + nB]).astype(np.int64)
            if f['b_src'] == 0:                        # CTX k-major
                base = f['b_base']
                fw = nB // 16
                if fw:
                    ctx[:, base:base + fw] = B[:fw * 16].reshape(fw, 16).T
                rem = nB - fw * 16
                if rem:
                    ctx[:rem, base + fw] = B[fw * 16:]
            else:                                      # W 每 k 行 COLS 字节
                base = f['b_base']
                nwd = nB // cols
                if nwd:
                    wram[:, base:base + nwd] = B[:nwd * cols].reshape(nwd,
                                                                     cols).T
                rem = nB - nwd * cols
                if rem:
                    wram[:rem, base + nwd] = B[nwd * cols:]
        elif op == 5:                                  # STORE word-major
            W = f['dma_len'] // 16
            base = f['y_base']
            segm = np.ascontiguousarray(ctx[:, base:base + W].T)
            ddr[f['dma_addr']:f['dma_addr'] + W * 16] = \
                (segm & 0xFF).astype(np.uint8).reshape(-1)
        elif op == 3:                                  # COPY CTX→WRAM
            src_j0 = f['rq_m']
            nr = n & 0xFF
            rws = src_j0 + np.arange(nr)
            lanes = rws % 16
            words = f['b_base'] + (rws // 16) * f['b_spad']
            idx = words[:, None] + np.arange(k)[None, :]
            wram[:nr, f['a_base']:f['a_base'] + k] = ctx[lanes[:, None], idx]
        elif op == 6:                                  # AE_ACTV（a3）
            sub = f['b_src']
            if sub == 0:
                _op6_actv(ctx, m, n, f['y_base'], f['b_base'])
            elif sub == 1:
                _op6_bias(ctx, m, n, k, f['y_base'], f['b_base'],
                          f['rq_m'], f['rq_s'])
            elif sub == 2:
                _op6_norm(ctx, m, n, f['y_base'], f['b_base'])
            else:
                raise AssertionError(f'pc={pc}: op6 未知子模式 {sub}')
        elif op in (0, 1, 2):                          # GEMM 族
            macs += ((m + 15) // 16) * 16 * cols * k
            ai = np.arange(m)
            idxA = f['a_base'] + (ai // 16)[:, None] * k + \
                np.arange(k)[None, :]
            A = ctx[(ai % 16)[:, None], idxA]
            B = wram[:, f['b_base']:f['b_base'] + k].T[:, :f['b_spad']]
            Y = sat8(A @ B * np.int64(f['rq_m']) >> np.int64(f['rq_s']))
            m16 = ((m + 15) // 16) * 16
            nl = f['b_spad']
            if f['y_tr']:
                c = f['j0'] + np.arange(nl)
                valid = np.nonzero(c < n)[0]
                cc = c[valid]
                lanes = (cc % 16)[:, None]
                words = (f['y_base'] + (cc // 16) * m16)[:, None] + \
                    np.arange(m)[None, :]
                ctx[lanes, words] = Y[:, valid].T
            else:
                words = (f['y_base'] + (ai // 16) * n + f['j0'])[:, None] + \
                    np.arange(nl)[None, :]
                ctx[(ai % 16)[:, None], words] = Y
            if op == 1:                                # SM16 softmax
                idxS = (f['y_base'] + (ai // 16) * n)[:, None] + \
                    np.arange(n)[None, :]
                S = ctx[(ai % 16)[:, None], idxS]
                from fast_interp import _softmax_rows_fast
                Pm = _softmax_rows_fast(S.astype(np.int8), n, f['sm_causal'])
                ctx[(ai % 16)[:, None], idxS] = Pm
        else:
            raise AssertionError(f'pc={pc}: 未定义 op={op}')
    return ctx, ddr, dict(macs=macs)


# ---------------------------------------------------------------------------
# selftest：与 actv_gold 的微观用例同构（ACTV/BIAS 各几组，含 pad 行/尾数）
# ---------------------------------------------------------------------------
def selftest_op6():
    rng = np.random.default_rng(20260831)
    CTXW = 2048
    # ---- ACTV 黄金（逐行循环，actv_gold 同式）----
    for (m, n, yb, tb) in [(18, 20, 0, 64), (32, 5, 80, 320), (7, 33, 112, 576),
                           (16, 16, 148, 848), (130, 108, 0, 1200)]:
        ctx = np.zeros((16, CTXW), dtype=np.int64)
        gold = np.zeros((16, CTXW), dtype=np.int64)
        init = rng.integers(-128, 128, size=(16, CTXW))
        lut = rng.integers(-128, 128, size=256).astype(np.int64)
        for c in (ctx, gold):
            c[:] = init
            for x in range(256):
                for L in range(16):
                    c[L, tb + x] = lut[x]
        for row in range(m):
            lane, base = row % 16, yb + (row // 16) * n
            for j in range(n):
                x = gold[lane, base + j] & 0xFF
                gold[lane, base + j] = lut[x]
        _op6_actv(ctx, m, n, yb, tb)
        assert (ctx == gold).all(), f'ACTV m={m} n={n} 不一致'
    print('[selftest] op6 ACTV 5 组与黄金逐位一致')
    # ---- BIAS 黄金 ----
    for (m, n, k, yb, tb, rqm, rqs) in [(18, 20, 20, 0, 832, 256, 8),
                                        (33, 17, 17, 48, 836, 257, 8),
                                        (5, 40, 40, 100, 840, -384, 4),
                                        (16, 8, 8, 140, 846, 32767, 0)]:
        ctx = np.zeros((16, CTXW), dtype=np.int64)
        gold = np.zeros((16, CTXW), dtype=np.int64)
        init = rng.integers(-128, 128, size=(16, CTXW))
        bj = rng.integers(-3000, 3001, size=k).astype(np.int64)
        for c in (ctx, gold):
            c[:] = init
            for j in range(k):
                lo, hi = int(bj[j]) & 0xFF, (int(bj[j]) >> 8) & 0xFF
                c[j % 16, tb + j // 16] = lo if lo < 128 else lo - 256
                c[j % 16, tb + (k + 15) // 16 + j // 16] = \
                    hi if hi < 128 else hi - 256
        for row in range(m):
            lane, base = row % 16, yb + (row // 16) * n
            for j in range(n):
                y = int(gold[lane, base + j])
                gold[lane, base + j] = sat8((y * rqm + int(bj[j])) >> rqs)
        _op6_bias(ctx, m, n, k, yb, tb, rqm & 0xFFFF, rqs)
        assert (ctx == gold).all(), f'BIAS m={m} n={n} 不一致'
    print('[selftest] op6 BIAS 4 组与黄金逐位一致')


if __name__ == '__main__':
    selftest_op6()
