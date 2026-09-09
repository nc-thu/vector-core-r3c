# -*- coding: utf-8 -*-
"""verify_outputs.py — 静态验证每段输出图被 STORE 恰好铺满、不重叠
（2026-08-31，配 _emit_store 的 byte0 修复）。

对每段：解码描述符流，把 op=5 STORE 的 [dma_addr, dma_addr+dma_len)
按所属输出图（ddr 区间包含关系）累计；检查：
  1. 同图写区间互不重叠
  2. Σ写字节 == 图字节（words*16）
  3. 铺满即连续无缝（排序后首尾相接）
用法：python verify_outputs.py <build_dir>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_interp import load_seq, decode          # noqa: E402


def main():
    build = sys.argv[1]
    segs = sorted(os.listdir(os.path.join(build, 'segments')))
    bad_overlap = bad_partial = 0
    nimgs = 0
    for s in segs:
        sd = os.path.join(build, 'segments', s)
        man = json.load(open(os.path.join(sd, 'manifest.json'),
                             encoding='utf-8'))
        seq = load_seq(sd)
        ranges = {}                                  # (name,ddr) → [(start,end)]
        for d in seq:
            f = decode(d)
            if f['op'] != 5:
                continue
            a, n = f['dma_addr'], f['dma_len']
            hit = False
            for o in man['outputs']:
                base, size = o['ddr'], o['words'] * 16
                if base <= a and a + n <= base + size:
                    ranges.setdefault((o['name'], o['ddr']), []).append((a, n))
                    hit = True
                    break
            if not hit:
                print(f'{s}: STORE [{a},{a+n}) 不落在任何输出图内!')
                bad_overlap += 1
        oby = {(o['name'], o['ddr']): o for o in man['outputs']}
        for key, rs in ranges.items():
            nimgs += 1
            name = key[0]
            rs.sort()
            for (a1, n1), (a2, n2) in zip(rs, rs[1:]):
                if a1 + n1 > a2:
                    print(f'{s}: 图 {name}@{key[1]:#x} STORE 重叠 '
                          f'@{a1:#x}/{a2:#x}')
                    bad_overlap += 1
            o = oby[key]
            total = o['words'] * 16
            got = sum(n for _, n in rs)
            if got != total or (rs and rs[0][0] != o['ddr']):
                print(f'{s}: 图 {name}@{key[1]:#x} 字节 {got} != {total} '
                      f'或起点不齐')
                bad_partial += 1
    print(f'[{build}] 段={len(segs)} 图={nimgs} '
          f'重叠/越界={bad_overlap} 未铺满={bad_partial} '
          f'{"PASS" if bad_overlap == 0 and bad_partial == 0 else "FAIL"}')


if __name__ == '__main__':
    main()
