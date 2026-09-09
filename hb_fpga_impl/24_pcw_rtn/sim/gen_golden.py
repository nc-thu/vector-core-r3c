# -*- coding: utf-8 -*-
"""gen_golden.py — 为 3 个代表段生成黄金 DDR 终态 + ddr_init.mem + seq.mem
输出到 stage/<seg>/ 目录。合成输入数据（seed=42 int8），与 run_segs.py 同口径。
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
A3 = os.path.normpath(os.path.join(HERE, '..', '..', '12_actv', 'a3'))
CB = os.path.normpath(os.path.join(HERE, '..', '..', '09_cbound'))
sys.path.insert(0, A3)
sys.path.insert(0, CB)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..', '12_actv', 'spec')))

from fast_interp_a3 import run_segment_fast_a3
from golden_interp import load_seq, decode
from acct_a3 import seg_account, CAL, SEG_CONST
from cycle_exact_a3 import seg_exact

BLOB = os.path.join(A3, 'build_a3', 'weights_blob.bin')
SEGS_DIR = os.path.join(A3, 'build_a3', 'segments')
STAGE = os.path.join(HERE, 'stage')

TEST_SEGS = ['seg_0600', 'seg_0529', 'seg_0602']
# 仿真档：CTX_WORDS=131072 与 RTL 一致；DDR 取 512KB 够装最大段（250KB）
COLS = 108
CTX_WORDS = 131072
W_WORDS = 4096
DDR_BYTES = 524288   # 512KB


def build_ddr_image(seg_dir, rng):
    """构建 DDR 初值映像：权重从 blob，输入用合成 int8 数据。"""
    man = json.load(open(os.path.join(seg_dir, 'manifest.json')))
    blob = np.fromfile(BLOB, dtype=np.uint8)
    ddr = np.zeros(DDR_BYTES, dtype=np.uint8)
    # 权重
    for w in man['weights']:
        ddr[w['ddr']:w['ddr'] + w['blob_len']] = \
            blob[w['blob_off']:w['blob_off'] + w['blob_len']]
    # 输入（合成）
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
    """numpy uint8 → 一行一字节 hex（$readmemh 兼容）。稀疏：只写非零块。"""
    arr = np.asarray(arr, dtype=np.uint8).reshape(-1)
    nz = np.flatnonzero(arr)
    if len(nz) == 0:
        open(path, 'w').write('')
        return
    # 分段写：@addr + 每行一字节
    brk = np.flatnonzero(np.diff(nz) > 256)
    starts = np.concatenate(([nz[0]], nz[brk + 1]))
    ends = np.concatenate((nz[brk] + 1, [nz[-1] + 1]))
    parts = []
    for s, e in zip(starts, ends):
        parts.append('@%X\n' % int(s))
        for b in arr[s:e]:
            parts.append('%02X\n' % int(b))
    open(path, 'w').write(''.join(parts))


def write_mem_full(path, arr):
    """全量 hex（一行一字节），给 RTL initial $readmemh 用。"""
    arr = np.asarray(arr, dtype=np.uint8).reshape(-1)
    with open(path, 'w') as f:
        for b in arr:
            f.write('%02X\n' % int(b))


def main():
    os.makedirs(STAGE, exist_ok=True)
    for seg in TEST_SEGS:
        seg_dir = os.path.join(SEGS_DIR, seg)
        out_dir = os.path.join(STAGE, seg)
        os.makedirs(out_dir, exist_ok=True)
        rng = np.random.RandomState(42)
        ddr0, man = build_ddr_image(seg_dir, rng)
        # 黄金执行
        seq = load_seq(seg_dir)
        P = {'COLS': COLS, 'CTX_WORDS': CTX_WORDS, 'W_WORDS': W_WORDS}
        ddr_gold = ddr0.copy()
        ctx_gold, ddr_gold, info = run_segment_fast_a3(seq, ddr_gold, P)
        # 写 seq.mem（直接拷贝）
        import shutil
        shutil.copy(os.path.join(seg_dir, 'seq.mem'),
                    os.path.join(out_dir, 'seq.mem'))
        # 写 ddr_init.mem（稀疏，给 +DDRIMG 用）
        write_mem(os.path.join(out_dir, 'ddr_init.mem'), ddr0)
        # 写 golden_ddr.bin（全量，给比对用）
        ddr_gold.astype(np.uint8).tofile(os.path.join(out_dir, 'golden_ddr.bin'))
        ddr0.astype(np.uint8).tofile(os.path.join(out_dir, 'ddr_init.bin'))
        # 输出区清单
        out_regions = [(o['ddr'], o['words']) for o in man['outputs']]
        with open(os.path.join(out_dir, 'out_regions.json'), 'w') as f:
            json.dump(out_regions, f)
        # 模型周期
        acc, _, _ = seg_account(seg_dir)
        keys = ('gemm', 'store', 'load_ctx', 'load_w', 'copy', 'softmax',
                'ae_actv')
        model_cal = sum(acc[k] * CAL[k] for k in keys) + SEG_CONST
        model_ideal = sum(acc[k] for k in keys)
        exact = seg_exact(seg_dir, cols=COLS, w_words=W_WORDS)
        print('=== %s ===' % seg)
        print('  n_descs=%d  est_cyc(cal)=%d  exact(sim)=%d  ideal=%d' %
              (man['n_descs'], model_cal, exact, model_ideal))
        print('  out_regions:', out_regions)
        # 保存周期信息
        with open(os.path.join(out_dir, 'model_cyc.json'), 'w') as f:
            json.dump(dict(model_cal=model_cal, model_ideal=model_ideal,
                           exact=exact), f)
    print('\n黄金数据生成完毕，目录:', STAGE)


if __name__ == '__main__':
    main()
