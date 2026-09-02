# -*- coding: utf-8 -*-
"""sm_hypo.py — seg_0221 softmax 差异假设拟合：改 golden 的 softmax 重算整段，
与 RTL dump 全量比，看哪个变体能精确命中。"""
import sys

import numpy as np

sys.path.insert(0, '/tmp/ae_cycles')
import golden_interp as GI
import golden_run as GR
import measure as ME

seg = 221
wd = '/tmp/ae_cycles/w/gold_%04d' % seg
ddr_init = GR.parse_sparse_mem(wd + '/ddr_init.mem', ME.P['DDR_BYTES'])
rd = GI.read_dump(wd + '/dump_ddr_prim2.mem')
seq = GI.load_seq(ME.seg_dir(seg))
man = json.load(open(ME.seg_dir(seg) + '/manifest.json')) if False else None
import json
man = json.load(open(ME.seg_dir(seg) + '/manifest.json'))
EXP = GI.EXP


def softmax_variant(S, n_cols, causal, mode):
    P = np.zeros_like(S)
    for i in range(S.shape[0]):
        vlen = min(i + 1, n_cols) if causal else n_cols
        row = S[i, :vlen].astype(np.int64)
        if vlen == 0:
            continue
        mx = int(row.max())
        am = int(row.argmax())
        e = np.array([EXP[min(mx - int(v), 128)] for v in row], dtype=np.int64)
        se = int(e.sum())
        if mode == 'A':                      # Σexp 去掉 argmax 列
            se2 = se - int(e[am])
        elif mode == 'B':                    # mx 取次大值
            srt = np.sort(row)
            mx2 = int(srt[-2]) if vlen > 1 else mx
            e = np.array([EXP[min(mx2 - int(v), 128)] for v in row],
                         dtype=np.int64)
            se2 = int(e.sum())
        elif mode == 'C':                    # mx+1（LUT 步长一档）
            e = np.array([EXP[min(mx + 1 - int(v), 128)] for v in row],
                         dtype=np.int64)
            se2 = int(e.sum())
        elif mode == 'D':                    # mx-1
            e = np.array([EXP[min(mx - 1 - int(v), 128)] for v in row],
                         dtype=np.int64)
            se2 = int(e.sum())
        elif mode == 'E':                    # 除法少 1
            se2 = se - 1
        quo = (127 << 30) // se2
        p = (e * quo) >> 30
        P[i, :vlen] = np.minimum(p, 127).astype(np.int8)
    return P


for mode in ('A', 'B', 'C', 'D', 'E'):
    GI.softmax_rows = lambda S, n_cols, causal, _m=mode: softmax_variant(
        S, n_cols, causal, _m)
    ctx, gddr, info = GI.run_segment(seq, ddr_init, ME.P)
    bad = 0
    tot = 0
    for e2 in man['outputs']:
        a, n = e2['ddr'], e2['words'] * 16
        bad += int(np.count_nonzero(gddr[a:a + n] != rd[a:a + n]))
        tot += n
    print('mode %s: 输出不一致 %d/%d' % (mode, bad, tot))
