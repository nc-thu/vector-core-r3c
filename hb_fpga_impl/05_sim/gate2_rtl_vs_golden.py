# -*- coding: utf-8 -*-
"""gate2_rtl_vs_golden.py — build_full_v3 抽段 RTL vs 黄金解释器位精确对拍
（2026-08-31，校验门②；服务器 /tmp/ae_v3）

每段：稀疏 ddr_init.mem（权重真值 + 按段号种子的随机激活字节）→
Verilator tb_ae_v 两遍（MODE=0 REF / MODE=1+PF=1 部署口径）→
golden_interp.run_segment → declared_ranges 逐字节比对 + mac_total 对账。
"""
import json
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, '/tmp/ae_hostdrv')
import golden_interp as GI                      # noqa: E402

BUILD = sys.argv[1] if len(sys.argv) > 1 else '/tmp/ae_v3/build_full_v3'
BIN = sys.argv[2] if len(sys.argv) > 2 else '/tmp/ae_v3/sim/obj_dir/tb_ae_v'
SEGS = sys.argv[3].split(',') if len(sys.argv) > 3 else \
    ['seg_0772', 'seg_0928', 'seg_2103']
WD = '/tmp/ae_v3/gate'
P = dict(COLS=108, CTX_WORDS=131072, W_WORDS=4096, SEQ_N=2048,
         DDR_BYTES=8388608, ZERO_SLOT=4096)
CYC_RE = re.compile(r'cycles=(\d+) gemm=(\d+) dma=(\d+) mac_total=(\d+)')


def parse_sparse(path):
    ddr = np.zeros(P['DDR_BYTES'], dtype=np.uint8)
    a = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith('@'):
            a = int(line[1:], 16)
        else:
            ddr[a] = int(line, 16)
            a += 1
    return ddr


def one(seg):
    sd = os.path.join(BUILD, 'segments', seg)
    man = json.load(open(os.path.join(sd, 'manifest.json')))
    os.makedirs(WD, exist_ok=True)
    blob = np.fromfile(os.path.join(BUILD, 'weights_blob.bin'), dtype=np.uint8)
    rng = np.random.default_rng(0xA5A5_0000 + int(seg[-4:]))
    lines = []
    for w in man['weights']:
        lines.append('@%X' % w['ddr'])
        lines += ['%02X' % b for b in blob[w['blob_off']:
                                           w['blob_off'] + w['blob_len']]]
    for e in man['inputs']:
        lines.append('@%X' % e['ddr'])
        lines += ['%02X' % b for b in
                  rng.integers(0, 256, e['words'] * 16, dtype=np.uint8)]
    with open(os.path.join(WD, 'ddr_init.mem'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    if not os.path.exists(os.path.join(WD, 'seq.mem')):
        os.symlink(os.path.join(sd, 'seq.mem'), os.path.join(WD, 'seq.mem'))
    ddr_init = parse_sparse(os.path.join(WD, 'ddr_init.mem'))
    seq = GI.load_seq(sd)
    _, gddr, info = GI.run_segment(seq, ddr_init, P)
    res = dict(seg=seg, golden_macs=info['macs'])
    for tag, mode, pf in (('ref', 0, 0), ('prim_pf1', 1, 1)):
        dump = os.path.join(WD, 'dump_%s.mem' % tag)
        if os.path.exists(dump):
            os.remove(dump)
        r = subprocess.run([BIN, '+MODE=%d' % mode, '+PF=%d' % pf,
                            '+SEQ=' + os.path.join(sd, 'seq.mem'),
                            '+DDRIMG=' + os.path.join(WD, 'ddr_init.mem'),
                            '+DUMP=' + dump],
                           cwd=WD, capture_output=True, text=True,
                           timeout=7200)
        m = CYC_RE.search(r.stdout)
        if m is None:
            res[tag] = dict(ok=False, msg='RTL 无输出: ' + r.stdout[-200:])
            continue
        cyc = dict(zip(('cycles', 'gemm', 'dma', 'mac_total'),
                       (int(x) for x in m.groups())))
        rd = GI.read_dump(dump)
        ranges = GI.declared_ranges(sd, P)
        ok, msg = GI.compare_ranges(gddr, rd, ranges)
        res[tag] = dict(ok=ok, msg=msg, cycles=cyc['cycles'],
                        mac_match=(cyc['mac_total'] == info['macs']))
    return res


if __name__ == '__main__':
    out = []
    for seg in SEGS:
        r = one(seg)
        out.append(r)
        print('[gate2] %s ref: ok=%s mac=%s | prim_pf1: ok=%s mac=%s cyc=%s %s'
              % (seg, r['ref'].get('ok'), r['ref'].get('mac_match'),
                 r['prim_pf1'].get('ok'), r['prim_pf1'].get('mac_match'),
                 r['prim_pf1'].get('cycles'), r['prim_pf1'].get('msg', '')),
              flush=True)
    json.dump(out, open('/tmp/ae_v3/gate2_result.json', 'w'), indent=1)
    allok = all(x['ref']['ok'] and x['prim_pf1']['ok'] for x in out)
    print('[gate2] ALL %s' % ('PASS' if allok else 'FAIL'))
