# -*- coding: utf-8 -*-
"""probe_deps.py — 用段 manifest 的输入/输出图名匹配，算 PL→PL 边界真实的
生产者→消费者往返字节（S2 上界的精确口径）。

边界账（11_overlap_plan）把 seg_b 的全部 STORE 和 seg_{b+1} 的全部首装
都记在边界头上；但 attn 内部逐头/逐批切段时，下一段 LOAD 的 Q/K 块是更早
host 步骤写的，跟 seg_b 的 STORE 无关——这部分驻留消不掉。
这里按图名交集算真实依赖，再叠加 CTX 容量过滤。
"""
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
A2 = os.path.join(ROOT, '09_cbound', 'build_a2')
CTX = 131072

hp = json.load(open(os.path.join(A2, 'host_plan.json')))
NSEG = len(hp['segments'])
man = []
for i in range(NSEG):
    man.append(json.load(open(os.path.join(A2, 'segments', 'seg_%04d' % i,
                                           'manifest.json'))))

bounds = defaultdict(list)
for h in hp['host_steps']:
    bounds[h['after_seg']].append(h)

tot_pair_b = 0          # 真实 store→load 对字节（STORE 侧）
tot_pair_l = 0          # LOAD 侧
cap_pair = 0            # 容量可行（≤0.9 CTX）的部分（LOAD 侧）
pair_rows = []
for b in range(NSEG - 1):
    if bounds.get(b):
        continue                      # 只看无 host 边界
    outs = {}
    for e in man[b]['outputs']:
        outs.setdefault(e['name'], e)['words'] = e['words']
    ins = {}
    for e in man[b + 1]['inputs']:
        ins.setdefault(e['name'], e)['words'] = e['words']
    common = set(outs) & set(ins)
    sb = sum(outs[n]['words'] for n in common) * 16
    lb = sum(ins[n]['words'] for n in common) * 16
    tot_pair_b += sb
    tot_pair_l += lb
    if common:
        # 容量：消费者该图 words（A 图直连要求整图驻留）
        ok = all(ins[n]['words'] <= CTX * 0.9 for n in common)
        pair_rows.append((b, sorted(common), sb, lb, ok))
        if ok:
            cap_pair += lb

print('PL→PL 边界（无 host）：%d' % sum(
    1 for b in range(NSEG - 1) if not bounds.get(b)))
print('其中存在真实图依赖（同名输出→输入）的：%d' % len(pair_rows))
print('真实可对消字节：STORE %.1fMB  LOAD %.1fMB' %
      (tot_pair_b / 1e6, tot_pair_l / 1e6))
print('消费者图 ≤0.9*CTX 的 LOAD 侧：%.1fMB' % (cap_pair / 1e6))
print('\n依赖图形态（按 words 分桶）：')
bc = Counter()
for _, names, sb, lb, ok in pair_rows:
    for n in names:
        pass
sizes = Counter()
for b, names, sb, lb, ok in pair_rows:
    sizes['≤4K' if lb <= 4096 * 16 else '4-16K' if lb <= 16384 * 16
          else '16-64K' if lb <= 65536 * 16 else '≤118K' if lb <= 118000 * 16
          else '>CTX'] += 1
for k in ('≤4K', '4-16K', '16-64K', '≤118K', '>CTX'):
    print('    %-8s %4d 边界' % (k, sizes[k]))
print('\n样例（前 12 条有依赖的边界）：')
for b, names, sb, lb, ok in pair_rows[:12]:
    print('    b=%-5d %-40s store=%.2fMB load=%.2fMB cap_ok=%s'
          % (b, ','.join(names)[:40], sb / 1e6, lb / 1e6, ok))
print('\n依赖图的 kind 分布：')
kc = Counter()
for b, names, sb, lb, ok in pair_rows:
    for n in names:
        kc[n.split(':')[0].split('#')[0]] += 1
for k, n in kc.most_common(12):
    print('    %-24s %d' % (k, n))
