# -*- coding: utf-8 -*-
"""acct_a3.py — a3 拍数账（2026-08-31）。

在 09_cbound/acct.py（a2 理想拍数账）基础上加两件事：

  1. op=6 AE_ACTV 拍数：本流全部 281 条都是 sub=0 ACTV（原地 LUT），
     引擎拍数 T = 260 + ceil(m/16) * (n + 3)（actv_gold 同式）。
  2. 校准周期模型：各分量理想值 × 校准系数（ZCU104 位精确对拍拟合），
     段数 × 每段常数 1253.5 拍。a2 用同一口径复算作基线。

用法: python acct_a3.py [build_a3] [--loose]
"""
import json
import math
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                '09_cbound'))
from golden_interp import decode, load_seq                      # noqa: E402

ROWS = 16
BURST_B = 2048
AR_OVH = 2
CMD_OVH = 5
GEMM_CMD_OVH = 2
PF_CMD_OVH = 1

# 校准系数（hw_zcu104 位精确对拍拟合，a2 轮给定）
CAL = dict(gemm=1.0514, store=1.1764, load_ctx=2.1388, load_w=2.1307,
           copy=1.0991, softmax=1.0514, ae_actv=1.0)
SEG_CONST = 1253.5


def waitd(cols):
    return ROWS + cols + 3


def wb_cycles(n_loc, j0, y_tr):
    if not y_tr:
        return n_loc
    return 16 * ((((j0 + n_loc - 1) >> 4) - (j0 >> 4)) + 1)


RQ_SH = 4
DRAIN = ROWS * RQ_SH
DALIGN = 2


def gemm_cycles(m, k, n_loc, j0, y_tr, cols=108):
    mt = (m + ROWS - 1) // ROWS
    tile = (1 + (k + 2) + waitd(cols) + DRAIN + DALIGN + 2
            + wb_cycles(n_loc, j0, y_tr) + 1)
    return 2 + GEMM_CMD_OVH + mt * tile


def actv_cycles(m, n):
    """AE_ACTV sub=0：T = 260 + 行组数 × (n+3)。"""
    return 260 + ((m + ROWS - 1) // ROWS) * (n + 3)


def load_ctx_ideal(n):
    return n // 8 + math.ceil(n / BURST_B) * AR_OVH + CMD_OVH


def store_cycles(n):
    return ((n + 15) // 16) * 5 + CMD_OVH


def w_load_ideal(nbytes, cols=108):
    beats = nbytes // 8
    xings = (nbytes // 8) // (cols // 8)
    return beats + xings + math.ceil(nbytes / BURST_B) * AR_OVH + CMD_OVH


def copy_cycles(k_rows, j_cols, j0):
    return 2 + 3 * k_rows * ((((j0 + j_cols - 1) >> 4) - (j0 >> 4)) + 1)


def softmax_cycles(m_rows, n_cols, causal):
    tot = 2
    for i in range(m_rows):
        v = min(i + 1, n_cols) if causal else n_cols
        tot += 2 * v + 2 * n_cols + 42
    return tot


def seg_account(sd, cols=108, pf=True, w_words=4096):
    half = w_words // 2
    acc = Counter()
    byt = Counter()
    cnt = Counter()
    prev_g = None
    prev_half = 0
    prev_k = 0
    for d in load_seq(sd):
        f = decode(d)
        op = f['op']
        if op == 15:
            break
        if op in (0, 1, 2):
            g = gemm_cycles(f['m'], f['k'], f['b_spad'], f['j0'],
                            f['y_tr'], cols)
            acc['gemm'] += g
            if op == 1:
                acc['softmax'] += softmax_cycles(f['m'], f['n'],
                                                 f['sm_causal'])
            prev_g = g
            prev_half = (f['b_base'] // half) & 1
            prev_k = f['k']
        elif op == 4:
            n = f['dma_len']
            if f['b_src'] == 0:
                acc['load_ctx'] += load_ctx_ideal(n)
                byt['load_ctx'] += n
                prev_g = None
            else:
                L = w_load_ideal(n, cols)
                if (pf and prev_g is not None
                        and (f['b_base'] // half) & 1 != prev_half
                        and n // cols <= half and prev_k <= half):
                    acc['load_w'] += max(0, L - prev_g) + PF_CMD_OVH
                    acc['n_pf'] += 1
                else:
                    acc['load_w'] += L
                byt['load_w'] += n
        elif op == 5:
            acc['store'] += store_cycles(f['dma_len'])
            byt['store'] += f['dma_len']
            prev_g = None
        elif op == 3:
            acc['copy'] += copy_cycles(f['k'], f['n'] & 0xFF, f['j0'])
            prev_g = None
        elif op == 6:
            acc['ae_actv'] += actv_cycles(f['m'], f['n'])
            prev_g = None
        cnt[op] += 1
    return acc, byt, cnt


def account(build, cols=108, pf=True):
    segdir = os.path.join(build, 'segments')
    segs = sorted(s for s in os.listdir(segdir) if s.startswith('seg_'))
    T = Counter()
    B = Counter()
    C = Counter()
    per = {}
    for s in segs:
        a, b, c = seg_account(os.path.join(segdir, s), cols, pf)
        per[s] = a
        T += a
        B += b
        C += c
    return T, B, C, per


def report(build, tag):
    T, B, C, per = account(build)
    keys = ['gemm', 'store', 'load_ctx', 'load_w', 'copy', 'softmax',
            'ae_actv']
    op_of = dict(gemm=0, store=5, load_ctx=4, load_w=4, copy=3,
                 softmax=1, ae_actv=6)
    ideal_tot = sum(T[k] for k in keys)
    cal_tot = sum(T[k] * CAL[k] for k in keys) + len(per) * SEG_CONST
    print('== %s（%s）：段=%d' % (tag, build, len(per)))
    for k in keys:
        ci = T[k] * CAL[k]
        print('  %-9s 理想 %8.2fM  校准 %8.2fM  占比 %5.1f%%  字节 %8.2fMB  n=%d'
              % (k, T[k] / 1e6, ci / 1e6, ci / cal_tot * 100,
                 B.get(k, 0) / 1e6, C.get(op_of.get(k, 1), 0)))
    print('  理想合计 %.1fM   校准合计 %.1fM（含每段常数 %.1fM，n_pf=%d）'
          % (ideal_tot / 1e6, cal_tot / 1e6,
             len(per) * SEG_CONST / 1e6, T['n_pf']))
    return T, per, cal_tot


if __name__ == '__main__':
    build = sys.argv[1] if len(sys.argv) > 1 else 'build_a3'
    report(r'..\..\09_cbound\build_a2', 'a2 基线')
    print()
    T, per, tot = report(build, 'a3 本轮')
