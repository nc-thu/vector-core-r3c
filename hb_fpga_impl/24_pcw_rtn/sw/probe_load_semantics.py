# -*- coding: utf-8 -*-
"""probe_load_semantics.py — 本地复现 seg_0000 的 LOAD/GEMM 语义（2026-08-31）。

不用模型：随机 A/W，按 fast_interp 的路由逐条装载 CTX/WRAM，再按 GEMM
读取语义算 Y，与直接矩阵乘对比。定位装载/读取语义差。"""
import sys

import numpy as np

sys.path.insert(0, '.')
from golden_interp import load_seq, decode               # noqa: E402

CTXW = 131072
COLS = 108


def main():
    rng = np.random.default_rng(7)
    m_tot, n, k = 20480, 96, 49
    q49 = rng.integers(-127, 128, (m_tot, k)).astype(np.int8)
    # pack_kact 语义：byte(i,c) = ((i//16)*k + c)*16 + (i%16)
    A_bytes = np.zeros(m_tot * k, dtype=np.uint8)
    for c in range(k):
        col = q49[:, c]
        idx = ((np.arange(m_tot) // 16) * k + c) * 16 + (np.arange(m_tot) % 16)
        A_bytes[idx] = col.view(np.uint8)
    Wimg = rng.integers(-127, 128, (k, COLS)).astype(np.int8)  # [k, cols]
    Wimg[k - 1] = 0                                          # aug 零行
    Wbytes = Wimg.tobytes()

    ddr = np.zeros(1 << 23, dtype=np.uint8)
    ddr[0:m_tot * k] = A_bytes
    W_ddr = 2969600
    ddr[W_ddr:W_ddr + len(Wbytes)] = np.frombuffer(Wbytes, dtype=np.uint8)

    seq = load_seq(r'build_s000_v3/segments/seg_0000')
    ctx = np.zeros((16, CTXW), dtype=np.int64)
    wram = np.zeros((COLS, 4096), dtype=np.int64)

    for pc, d in enumerate(seq):
        f = decode(d)
        op = f['op']
        if op == 15:
            break
        if op == 4:                                   # LOAD
            nB = f['dma_len']
            raw = ddr[f['dma_addr']:f['dma_addr'] + nB].astype(np.int64)
            B = ((raw + 128) & 0xFF) - 128            # _s8 有符号解释
            if f['b_src'] == 0:                       # CTX k-major
                base = f['b_base']
                b = np.arange(nB)
                ctx[b % 16, base + b // 16] = B
            else:                                     # WRAM 每 cols 一行
                base = f['b_base']
                nwd = nB // COLS
                if nwd:
                    wram[:, base:base + nwd] = \
                        B[:nwd * COLS].reshape(nwd, COLS).T
                rem = nB - nwd * COLS
                if rem:
                    wram[:rem, base + nwd] = B[nwd * COLS:]
        elif op == 0:                                 # GEMM
            m_, n_, k_ = f['m'], f['n'], f['k']
            ai = np.arange(m_)
            idxA = f['a_base'] + (ai // 16)[:, None] * k_ + \
                np.arange(k_)[None, :]
            A = ctx[(ai % 16)[:, None], idxA]
            B = wram[:, f['b_base']:f['b_base'] + k_].T[:, :f['b_spad']]
            Y = np.clip((A @ B * f['rq_m']) >> f['rq_s'], -128, 127)
            if pc == 4:
                for cdbg in (0, 2):
                    acc_dbg = A[0].astype(np.int64) @ \
                        B[:, cdbg].astype(np.int64)
                    acc_dir = int(q49[0].astype(np.int64)
                                  @ wram[cdbg, 0:k].astype(np.int64))
                    print('GEMM 行0列%d: 链acc=%d 直acc=%d y=%d B[:3]=%s'
                          ' wram[:3]=%s' % (cdbg, acc_dbg, acc_dir,
                                            (acc_dbg * f['rq_m'])
                                            >> f['rq_s'], B[:3, cdbg],
                                            wram[cdbg, :3]))
                diffab = np.argwhere(A[:13008] != q49[:13008])
                print('  A 与 q49 不同元素:', len(diffab))
                diffB = np.argwhere(B != wram[0:96, 0:k].T)
                print('  B 与 wram.T 不同元素:', len(diffB))
            m16 = ((m_ + 15) // 16) * 16
            words = (f['y_base'] + (ai // 16) * n_)[:, None] + \
                np.arange(n_)[None, :]
            ctx[(ai % 16)[:, None], words] = Y

    # 输出读回（STORE 语义）
    Yout = np.zeros((m_tot, n), dtype=np.int64)
    for d in seq:
        f = decode(d)
        if f['op'] != 5:
            continue
        nwords = f['dma_len'] // 16
        base = f['y_base']
        seg = ctx[:, base:base + nwords].T             # [words, 16]
        # 全局 word 号 u = base + w：行块 u//n，列 u%n
        w = np.arange(nwords)
        u = base + w
        rb = u // n
        col = u % n
        for t in range(16):
            rows = rb * 16 + t
            m2 = rows < m_tot
            Yout[rows[m2], col[m2]] = seg[w[m2], t]
    # 直接矩阵乘（pack 语义下的 A，wram 语义下的 W）
    W_direct = np.zeros((n, k), dtype=np.int64)
    for c in range(COLS):
        if c < n:
            W_direct[c, :] = wram[c, 0:k]
    acc = q49.astype(np.int64) @ W_direct.T
    Ysim = np.clip((acc * 25685) >> 24, -128, 127)
    diff = Yout.astype(int) - Ysim.astype(int)
    nz = diff != 0
    print('不一致: %d / %d (%.1f%%)'
          % (int(nz.sum()), diff.size, 100 * float(nz.mean())))
    if nz.any():
        rows = np.flatnonzero(nz.any(axis=1))
        print('首批行:', rows[:5], '行/16 块:', (rows[:5] // 16).tolist())
        r0 = rows[0]
        c0 = int(np.flatnonzero(nz[r0])[0])
        print('行%d 列%d: 链=%d 直乘=%d' % (r0, c0, Yout[r0, c0], Ysim[r0, c0]))
        # 首行块的 A 对比
        i = r0
        w = (i // 16) * k + np.arange(k)
        print('A 链 vs q49:', ctx[i % 16, 78048 + w][:6],
              q49[i, :6])


if __name__ == '__main__':
    main()
