# -*- coding: utf-8 -*-
"""stage_seg.py — v5-R2 段级验证 staging（2026-09-08 20:22）
为指定段生成黄金 DDR 终态 + ddr_init.mem + seq.mem + 模型周期，
输出 r2/sim/stage/<seg>/。口径与 22_r3c_rtl/sim/gen_golden.py 完全一致
（COLS=108 / CTX_WORDS=131072 / W_WORDS=4096 / DDR 512KB，seed=42）。
用法：python stage_seg.py seg_0636 [seg_xxxx ...]
"""
import os, sys, json, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
A3 = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..',
                                   'hb_fpga_impl', '12_actv', 'a3'))
sys.path.insert(0, A3)
sys.path.insert(0, os.path.normpath(os.path.join(A3, '..', '..', '09_cbound')))
sys.path.insert(0, os.path.normpath(os.path.join(A3, '..', 'spec')))

from fast_interp_a3 import run_segment_fast_a3
from golden_interp import load_seq
from acct_a3 import seg_account, CAL, SEG_CONST
from cycle_exact_a3 import seg_exact

BLOB = os.path.join(A3, 'build_a3', 'weights_blob.bin')
SEGS_DIR = os.path.join(A3, 'build_a3', 'segments')
STAGE = os.path.join(HERE, 'stage')

COLS = 108
CTX_WORDS = 131072
W_WORDS = 4096
DDR_BYTES = 524288


def build_ddr_image(seg_dir, rng):
    man = json.load(open(os.path.join(seg_dir, 'manifest.json')))
    blob = np.fromfile(BLOB, dtype=np.uint8)
    ddr = np.zeros(DDR_BYTES, dtype=np.uint8)
    for w in man['weights']:
        ddr[w['ddr']:w['ddr'] + w['blob_len']] = \
            blob[w['blob_off']:w['blob_off'] + w['blob_len']]
    for e in man['inputs']:
        n = e['words']
        if str(e['name']).startswith('const:'):
            v = int(e.get('cval', str(e['name']).split(':')[1])) & 0xFF
            data = np.full(n, v, dtype=np.uint8)
        else:
            data = rng.randint(-128, 128, size=n).astype(np.int8).view(np.uint8)
        ddr[e['ddr']:e['ddr'] + n] = data
    return ddr, man


def write_mem(path, arr):
    arr = np.asarray(arr, dtype=np.uint8).reshape(-1)
    nz = np.flatnonzero(arr)
    if len(nz) == 0:
        open(path, 'w').write('')
        return
    brk = np.flatnonzero(np.diff(nz) > 256)
    starts = np.concatenate(([nz[0]], nz[brk + 1]))
    ends = np.concatenate((nz[brk] + 1, [nz[-1] + 1]))
    parts = []
    for s, e in zip(starts, ends):
        parts.append('@%X\n' % int(s))
        for b in arr[s:e]:
            parts.append('%02X\n' % int(b))
    open(path, 'w').write(''.join(parts))


def main():
    segs = sys.argv[1:]
    os.makedirs(STAGE, exist_ok=True)
    for seg in segs:
        seg_dir = os.path.join(SEGS_DIR, seg)
        out_dir = os.path.join(STAGE, seg)
        os.makedirs(out_dir, exist_ok=True)
        rng = np.random.RandomState(42)
        ddr0, man = build_ddr_image(seg_dir, rng)
        seq = load_seq(seg_dir)
        P = {'COLS': COLS, 'CTX_WORDS': CTX_WORDS, 'W_WORDS': W_WORDS}
        ddr_gold = ddr0.copy()
        ctx_gold, ddr_gold, info = run_segment_fast_a3(seq, ddr_gold, P)
        shutil.copy(os.path.join(seg_dir, 'seq.mem'),
                    os.path.join(out_dir, 'seq.mem'))
        write_mem(os.path.join(out_dir, 'ddr_init.mem'), ddr0)
        ddr_gold.astype(np.uint8).tofile(os.path.join(out_dir, 'golden_ddr.bin'))
        ddr0.astype(np.uint8).tofile(os.path.join(out_dir, 'ddr_init.bin'))
        out_regions = [(o['ddr'], o['words']) for o in man['outputs']]
        with open(os.path.join(out_dir, 'out_regions.json'), 'w') as f:
            json.dump(out_regions, f)
        acc, _, _ = seg_account(seg_dir)
        keys = ('gemm', 'store', 'load_ctx', 'load_w', 'copy', 'softmax',
                'ae_actv')
        model_cal = sum(acc[k] * CAL[k] for k in keys) + SEG_CONST
        exact = seg_exact(seg_dir, cols=COLS, w_words=W_WORDS)
        with open(os.path.join(out_dir, 'model_cyc.json'), 'w') as f:
            json.dump(dict(model_cal=model_cal, exact=exact,
                           acc={k: acc[k] for k in keys}), f)
        print('=== %s: est_cal=%d exact=%d out_regions=%d' %
              (seg, model_cal, exact, len(out_regions)))


if __name__ == '__main__':
    main()
