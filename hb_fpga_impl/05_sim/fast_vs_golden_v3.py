# -*- coding: utf-8 -*-
"""fast_vs_golden_v3.py — build_s000_v3 抽段 fast_interp vs golden_interp
逐位对拍（快档引擎正确性证明，2026-08-31，服务器 /tmp/ae_hostdrv）。
随机 DDR（按段号定种子）+ 段内完整描述符流，比对 CTX 终态 + DDR 终态。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/tmp/ae_hostdrv')
import golden_interp as GI                      # noqa: E402
import fast_interp as FI                        # noqa: E402

BUILD = sys.argv[1] if len(sys.argv) > 1 else '/tmp/ae_hostdrv/build_s000_v3'
P = dict(COLS=108, CTX_WORDS=131072, W_WORDS=4096, SEQ_N=2048,
         DDR_BYTES=8388608, ZERO_SLOT=4096)
rng = np.random.default_rng(20260831)
segs = sorted(os.listdir(os.path.join(BUILD, 'segments')))
picks = [segs[i] for i in rng.choice(len(segs), 6, replace=False)]
print('抽段:', picks)
nok = 0
for s in picks:
    sd = os.path.join(BUILD, 'segments', s)
    man = json.load(open(os.path.join(sd, 'manifest.json')))
    blob = np.fromfile(os.path.join(BUILD, 'weights_blob.bin'), dtype=np.uint8)
    ddr = np.zeros(P['DDR_BYTES'], dtype=np.uint8)
    for w in man['weights']:
        ddr[w['ddr']:w['ddr'] + w['blob_len']] = \
            blob[w['blob_off']:w['blob_off'] + w['blob_len']]
    r = np.random.default_rng(0xA5A5_0000 + int(s[-4:]))
    for e in man['inputs']:
        ddr[e['ddr']:e['ddr'] + e['words'] * 16] = \
            r.integers(0, 256, e['words'] * 16, dtype=np.uint8)
    seq = GI.load_seq(sd)
    ctx_g, ddr_g, info_g = GI.run_segment(seq, ddr, P)
    ctx_f, ddr_f, _ = FI.run_segment_fast(seq, ddr, P)
    okc = np.array_equal(ctx_g, ctx_f)
    okd = np.array_equal(ddr_g, ddr_f)
    print('%s ctx=%s ddr=%s' % (s, okc, okd), flush=True)
    nok += (okc and okd)
print('fast vs golden: %d/%d 段逐位一致' % (nok, len(picks)))
