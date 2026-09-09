# -*- coding: utf-8 -*-
"""fast_selftest.py — fast_interp vs golden_interp 逐位对拍（真实 build 段）。

对每个抽测段：合成随机 int8 输入（数值无关，只验寻址/位精确；const 图填
真实 cval），两个解释器跑同一段，比对 CTX 终态 + 全 DDR 终态逐字节一致。

抽段策略：按段的特征签名（含哪些 op / y_tr / sm_causal / 输入输出 kind）
分桶，每桶抽一个（控制在 12 个以内，golden 纯 Python 很慢）。

用法：
  python fast_selftest.py [build_dir] [seg_0003 ...]   # 缺省自动抽段
"""
import json
import os
import sys
import time

import numpy as np

from golden_interp import build_ddr_image, decode, load_seq, run_segment
from fast_interp import run_segment_fast

HERE = os.path.dirname(os.path.abspath(__file__))


def _features(sd, seq):
    ops = set()
    y_tr = sm = 0
    for d in seq:
        f = decode(d)
        if f['op'] == 15:
            break
        ops.add(f['op'])
        y_tr += f['y_tr']
        sm += f['sm_causal']
    return frozenset(ops), y_tr > 0, sm > 0


def pick_segments(build, cap=12):
    hp = json.load(open(os.path.join(build, 'host_plan.json'),
                        encoding='utf-8'))
    buckets = {}
    for sd in sorted(os.listdir(os.path.join(build, 'segments'))):
        seq = load_seq(os.path.join(build, 'segments', sd))
        ops, y_tr, sm = _features(sd, seq)
        if not ops:
            continue
        kinds = tuple(sorted({i.get('kind', '?') for i in json.load(
            open(os.path.join(build, 'segments', sd, 'manifest.json'),
                 encoding='utf-8')).get('inputs', [])}))
        key = (ops, y_tr, sm, kinds)
        n = len(seq)
        old = buckets.get(key)
        if old is None or n > old[1]:      # 每桶取最大的（覆盖最全路径）
            buckets[key] = (sd, n)
    # golden 慢，按描述符数升序排，大的放后面（失败早停）
    out = sorted(buckets.values(), key=lambda t: t[1])
    return [sd for sd, _ in out[:cap]]


def run_one(build, sd, rng):
    seg_dir = os.path.join(build, 'segments', sd)
    man = json.load(open(os.path.join(seg_dir, 'manifest.json'),
                         encoding='utf-8'))
    act = {}
    for e in man['inputs']:
        if e['name'].startswith('const:'):
            act[e['name']] = np.full(e['words'] * 16,
                                     int(e.get('cval', 0)) & 0xFF,
                                     dtype=np.uint8)
        else:
            act[e['name']] = rng.integers(0, 256, size=e['words'] * 16,
                                          dtype=np.uint8)
    P = run_one.P
    img = build_ddr_image(seg_dir, os.path.join(build, 'weights_blob.bin'),
                          act, P)
    seq = load_seq(seg_dir)
    t0 = time.perf_counter()
    ctx_g, ddr_g, _ = run_segment(seq, img, P)
    t1 = time.perf_counter()
    ctx_f, ddr_f, _ = run_segment_fast(seq, img, P)
    t2 = time.perf_counter()
    ok_c = np.array_equal(ctx_g, ctx_f)
    ok_d = np.array_equal(ddr_g, ddr_f)
    where = ''
    if not ok_d:
        bad = np.nonzero(ddr_g != ddr_f)[0]
        where = (f' first@0x{bad[0]:X} g={ddr_g[bad[0]]:02X} '
                 f'f={ddr_f[bad[0]]:02X} n={len(bad)}')
    if not ok_c:
        bad = np.nonzero(ctx_g != ctx_f)
        where += f' CTX@{bad[0]}'
    print(f'  {"PASS" if ok_c and ok_d else "FAIL"} {sd} '
          f'({len(seq)} desc): ctx={"OK" if ok_c else "DIFF"} '
          f'ddr={"OK" if ok_d else "DIFF"} golden={t1-t0:.1f}s '
          f'fast={t2-t1:.3f}s{where}', flush=True)
    return ok_c and ok_d


if __name__ == '__main__':
    build = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(HERE, 'build_s000')
    import compiler
    run_one.P = compiler.PROFILES['full']
    named = sys.argv[2:]
    targets = named or pick_segments(build)
    print(f'[selftest] {build}: {len(targets)} 段 '
          f'({", ".join(targets[:6])}{"..." if len(targets) > 6 else ""})')
    rng = np.random.default_rng(7)
    allok = True
    for sd in targets:
        allok &= run_one(build, sd, rng)
    print(f'[selftest] {"ALL PASS" if allok else "FAILED"}')
    sys.exit(0 if allok else 1)
