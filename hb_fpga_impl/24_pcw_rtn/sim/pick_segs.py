# -*- coding: utf-8 -*-
"""pick_segs.py — 从 build_a3/segments 挑 2-3 个代表段。
三类：①STORE+GEMM 交错（R1 写通道并发）
     ②连续 LOAD_CTX（R2 预取 CTX）
     ③op=6 ACTV/NORM（引擎全链路）
选段准则：描述符数少（≤40，跑得快 ≤10min）、含目标特征。
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
A3 = os.path.normpath(os.path.join(HERE, '..', '..', '12_actv', 'a3'))
sys.path.insert(0, A3)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                '09_cbound'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                '12_actv', 'spec'))
from golden_interp import decode, load_seq
from acct_a3 import seg_account, CAL, SEG_CONST

SEGS_DIR = os.path.join(A3, 'build_a3', 'segments')


def classify(seg_dir):
    """返回 (n_descs, has_store_gemm, has_load_ctx_run, has_op6, est_cyc)。"""
    seq = load_seq(seg_dir)
    ops = []
    for d in seq:
        f = decode(d)
        if f['op'] == 15:
            break
        ops.append(f)
    n = len(ops)
    # ① STORE(op5) 与 GEMM(op0/1/2) 交错：相邻出现且 STORE 后紧跟 GEMM 或反之
    store_gemm = False
    for i in range(len(ops) - 1):
        a, b = ops[i]['op'], ops[i + 1]['op']
        if (a == 5 and b in (0, 1, 2)) or (a in (0, 1, 2) and b == 5):
            store_gemm = True
            break
    # ② 连续 LOAD_CTX（op4 且 b_src==0）≥2 条相邻
    load_ctx_run = False
    run = 0
    for f in ops:
        if f['op'] == 4 and f['b_src'] == 0:
            run += 1
            if run >= 2:
                load_ctx_run = True
                break
        else:
            run = 0
    # ③ op=6
    has_op6 = any(f['op'] == 6 for f in ops)
    # 估算周期（校准口径，单段）
    acc, _, _ = seg_account(seg_dir)
    keys = ('gemm', 'store', 'load_ctx', 'load_w', 'copy', 'softmax',
            'ae_actv')
    est = sum(acc[k] * CAL[k] for k in keys) + SEG_CONST
    return n, store_gemm, load_ctx_run, has_op6, est, \
        [f['op'] for f in ops]


def main():
    segs = sorted(s for s in os.listdir(SEGS_DIR) if s.startswith('seg_'))
    print('总段数:', len(segs))
    # 为每类找最短（最快）的命中段
    cand = {1: None, 2: None, 3: None}   # 类别 -> (est, seg, info)
    for s in segs:
        sd = os.path.join(SEGS_DIR, s)
        n, sg, lc, o6, est, ops = classify(sd)
        if n > 40 or est > 2_000_000:    # 跑得快的
            continue
        for cat, hit in [(1, sg), (2, lc), (3, o6)]:
            if not hit:
                continue
            if cand[cat] is None or est < cand[cat][0]:
                cand[cat] = (est, s, (n, sg, lc, o6, ops))
    for cat, name in [(1, 'STORE+GEMM'), (2, 'LOAD_CTX run'), (3, 'op6 ACTV')]:
        if cand[cat]:
            est, s, (n, sg, lc, o6, ops) = cand[cat]
            print('\n[%s] %s  n_descs=%d est=%d' % (name, s, n, est))
            print('  ops:', ops)
        else:
            print('\n[%s] 无命中' % name)


if __name__ == '__main__':
    main()
