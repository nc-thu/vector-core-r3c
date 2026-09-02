# -*- coding: utf-8 -*-
"""probe_bounds.py — 摸清 a2 里 PL→PL 边界与 norm/actv 边界的结构（a3 设计输入）

问题：S2 跨段驻留要求「消费者 A 图整张留在 CTX」，CTX=131072 字。
这里从 build_a2/host_plan.json 的 node_map（含每个 gemm 节点的 m/k/n/段范围）
重建：每条边界的生产者节点 → 消费者节点，算驻留所需字数 m16*(k+1)，
看多少边界容量可行；norm/actv 边界同样算一遍（S1 也要求图直连）。
"""
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # hb_fpga_impl
A2 = os.path.join(ROOT, '09_cbound', 'build_a2')

hp = json.load(open(os.path.join(A2, 'host_plan.json')))
nodes = hp['nodes']
segs = hp['segments']                                  # name 顺序 = 段号
seg_idx = {s['name']: i for i, s in enumerate(segs)}
CTX = 131072

# ---- 每段的归属节点（node_map['segs'] 是该节点发射的段名列表）----
seg_node = {}
for ni, nd in enumerate(nodes):
    for sn in nd.get('segs', []):
        seg_node[seg_idx[sn]] = ni

# ---- 边界 host 步骤 ----
bounds = defaultdict(list)
for h in hp['host_steps']:
    bounds[h['after_seg']].append(h)

# ---- 逐边界：生产者节点（最后一段的归属）→ 消费者节点 ----
rows = []
NSEG = len(segs)
for b in range(NSEG - 1):
    hs = bounds.get(b, [])
    p = seg_node.get(b)
    c = seg_node.get(b + 1)
    pn, cn = nodes[p] if p is not None else None, nodes[c] if c is not None else None
    kinds = sorted({h['kind'] for h in hs})
    rows.append(dict(b=b, kinds=kinds, p=p, c=c,
                     pk=pn['kind'] if pn else None,
                     ck=cn['kind'] if cn else None,
                     pmod=pn['module'] if pn else None,
                     cmod=cn['module'] if cn else None))

# ---- PL→PL：消费者是 gemm 且生产者是 gemm/attn 的，算驻留字数 ----
def c16(x):
    return (x + 15) // 16 * 16

res = []
for r in rows:
    if r['kinds'] or r['ck'] != 'gemm':
        continue
    cn = nodes[r['c']]
    m, k = cn['m'], cn['k']
    words = c16(m) * (k + 1)
    res.append((r['b'], r['pmod'], r['cmod'], m, k, words))

print('PL→PL 且消费者是普通 gemm 的边界：%d' % len(res))
cap = [x for x in res if x[5] <= CTX * 0.9]
print('  驻留字数 ≤ 0.9*CTX 的：%d' % len(cap))
dist = Counter()
for _, _, _, m, k, w in res:
    dist[min(6, int.bit_length(w // 1024) if w >= 1024 else 0)] += 1
print('  驻留字数分布（1024 字为桶）:')
buckets = Counter()
for x in res:
    w = x[5]
    if w <= 4096:
        buckets['≤4K'] += 1
    elif w <= 16384:
        buckets['4-16K'] += 1
    elif w <= 65536:
        buckets['16-64K'] += 1
    elif w <= 118000:
        buckets['64-118K(可驻留)'] += 1
    else:
        buckets['>118K(放不下)'] += 1
for k2 in ('≤4K', '4-16K', '16-64K', '64-118K(可驻留)', '>118K(放不下)'):
    print('    %-18s %5d' % (k2, buckets[k2]))
print('\n消费者 k 分布（前 12）:')
kc = Counter(x[4] for x in res)
for k2, n in kc.most_common(12):
    print('    k=%-6d %5d 边界' % (k2, n))
print('\n消费者 m 分布（前 12）:')
mc = Counter(x[3] for x in res)
for m2, n in mc.most_common(12):
    print('    m=%-6d %5d 边界' % (m2, n))

# 生产者→消费者 kind 组合
pk = Counter((r['pk'], r['ck']) for r in rows if not r['kinds'])
print('\nPL→PL 生产者/消费者 kind 组合:')
for kk, n in pk.most_common(10):
    print('    %s → %s : %d' % (kk[0], kk[1], n))

# ---- norm/actv 边界（S1 范围）----
nk = Counter()
for r in rows:
    if r['kinds'] and set(r['kinds']) <= {'norm', 'actv'}:
        nk[tuple(r['kinds'])] += 1
print('\nhost kind 全在 {norm,actv} 的边界：')
for kk, n in nk.most_common():
    print('    %s : %d' % ('+'.join(kk), n))

# 这些边界的消费者形态
cc = Counter()
for r in rows:
    if r['kinds'] and set(r['kinds']) <= {'norm', 'actv'}:
        cc[r['ck']] += 1
print('  消费者 kind：', dict(cc))
pc = Counter()
for r in rows:
    if r['kinds'] and set(r['kinds']) <= {'norm', 'actv'}:
        pc[r['pk']] += 1
print('  生产者 kind：', dict(pc))
