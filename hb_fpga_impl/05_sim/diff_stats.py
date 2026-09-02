# -*- coding: utf-8 -*-
"""diff_stats.py — seg_0221 失配全量统计（服务器 gold_0221 工作目录还在时用）"""
import json
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
ctx, gddr, info = GI.run_segment(seq, ddr_init, ME.P)
g = np.frombuffer(gddr.astype(np.uint8).tobytes(), dtype=np.uint8)
d = np.frombuffer(rd.astype(np.uint8).tobytes(), dtype=np.uint8)
man = json.load(open(ME.seg_dir(seg) + '/manifest.json'))
for e in man['outputs'] + man['inputs']:
    a, n = e['ddr'], e['words'] * 16
    diff = np.nonzero(g[a:a + n] != d[a:a + n])[0]
    print('%s %s [%X,%X): %d/%d 字节不一致' %
          (e.get('kind'), e['name'][:50], a, a + n, len(diff), n))
    # 差值的形态：golden 值 vs rtl 值
    for off in diff[:24]:
        gv, dv = int(g[a + off]), int(d[a + off])

        def s8(x):
            return x - 256 if x > 127 else x
        print('  +%04d golden=%3d rtl=%3d (signed %d vs %d)' %
              (off, gv, dv, s8(gv), s8(dv)))
    if len(diff) > 24:
        dvals = [(int(g[a + o]), int(d[a + o])) for o in diff]
        ratio = sum(1 for x, y in dvals if y == 0 and x != 0)
        print('  ... 其余 %d 个：rtl=0 且 golden≠0 的 %d 个；样本 %s' %
              (len(diff) - 24, ratio, dvals[24:40]))
