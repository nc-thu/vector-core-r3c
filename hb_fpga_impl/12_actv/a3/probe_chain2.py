# -*- coding: utf-8 -*-
"""probe_chain2.py — 从 ops_trace 重建张量链，精确评估 a3 两个杠杆的可行规模。

对每个普通 WGemm 消费者 c（node_map kind=gemm、无 heads_mode）：
  沿 trace 向回走：c.rec.in_ids[0] ← 连续 norm/actv host 记录 ← 生产者记录。
  得到链 p → [h1..hk] → c，检查：
    * 生产者是 PL gemm（node_map 里能找到，kind=gemm，plain）
    * 张量单消费者（全 trace 扫 in_ids）
    * 形状直连：p.m == c.m 且 p.n == c.k（host norm/actv 不改形状）
    * 容量：resident = m16*(k+1) ≤ CTX*0.9（再留 tile 余量）
  统计可消字节：STORE 侧 = m16*p.n*16（p 的 Y 图），LOAD 侧 = m16*(k+1)*16。
"""
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
A2 = os.path.join(ROOT, '09_cbound', 'build_a2')
TRACE = os.path.join(ROOT, '09_cbound', 'ops_trace.json')
CTX = 131072

tr = json.load(open(TRACE))
recs = tr['ops']
by_seq = {r['seq']: r for r in recs}
hp = json.load(open(os.path.join(A2, 'host_plan.json')))
nodes = hp['nodes']

node_by = {}
for nd in nodes:
    if nd['kind'] == 'gemm':
        node_by[(nd['module'], nd['seq'])] = nd
heads_mode = {(nd['module'], nd['seq']) for nd in nodes
              if nd['kind'] == 'gemm' and nd.get('heads_mode')}

tid_consumers = defaultdict(list)     # tid → 消费记录 seq
tid_producers = defaultdict(list)     # tid → 生产记录 seq
for r in recs:
    for t in r.get('in_ids', []):
        tid_consumers[t].append(r['seq'])
    for t in r.get('out_ids', []):
        tid_producers[t].append(r['seq'])

NORM_CLS = ('LayerNorm', 'RMSNorm', 'AdaRMSNorm', 'GroupNorm')
ACTV_CLS = ('SiLU', 'GELU', 'GELUActivation', 'ReLU')

cands = []
for nd in nodes:
    if nd['kind'] != 'gemm' or nd.get('heads_mode'):
        continue
    c_rec = by_seq.get(nd['seq'])
    if c_rec is None or not c_rec.get('in_ids'):
        continue
    cur = c_rec['in_ids'][0]
    chain = []
    # 回走：cur 的唯一消费者是一条 norm/actv 记录 → 跨过它
    while True:
        cons = tid_consumers[cur]
        if len(cons) != 1:
            break
        h = by_seq[cons[0]]
        if h['op'] != 'elem_norm' or h['cls'] not in NORM_CLS + ACTV_CLS:
            break
        chain.append(h)
        cur = h['in_ids'][0]
    # 生产者：恰一条记录产出 cur
    prods = tid_producers.get(cur, [])
    if len(prods) != 1:
        continue
    p_rec = by_seq[prods[0]]
    if p_rec['op'] != 'gemm':
        continue
    p_nd = node_by.get((p_rec['module'], p_rec['seq']))
    if p_nd is None or (p_rec['module'], p_rec['seq']) in heads_mode:
        continue
    # p 的输出张量单消费者（除了链上的 host 记录/或直接是 c）
    m, k = nd['m'], nd['k']
    m16 = (m + 15) // 16 * 16
    ok_shape = (p_nd['m'] == m and p_nd['n'] == k)
    res_words = m16 * (k + 1)
    ok_cap = res_words <= CTX * 0.9 - 8192
    kinds = [h['cls'] for h in chain]
    is_norm = any(x in NORM_CLS for x in kinds)
    lev = 'S1-norm' if is_norm else ('S1-actv' if kinds else 'S2')
    cands.append(dict(cmod=nd['module'], pmod=p_nd['module'], m=m, k=k,
                      kinds=kinds, lev=lev, ok_shape=ok_shape,
                      res_words=res_words, ok_cap=ok_cap,
                      store_b=m16 * p_nd['n'] * 16 if ok_shape else 0,
                      load_b=m16 * (k + 1) * 16 if ok_shape else 0))

print('普通 gemm 消费者（能找到 PL 生产者的链）：%d' % len(cands))
both = [c for c in cands if c['ok_shape']]
capok = [c for c in both if c['ok_cap']]
print('形状直连：%d   容量可行：%d' % (len(both), len(capok)))
sb = sum(c['store_b'] for c in capok)
lb = sum(c['load_b'] for c in capok)
print('可消字节：STORE %.1fMB + LOAD %.1fMB = %.1fMB' %
      (sb / 1e6, lb / 1e6, (sb + lb) / 1e6))
print('\n按杠杆分：')
for lev in ('S2', 'S1-actv', 'S1-norm'):
    sub = [c for c in both if c['lev'] == lev]
    okc = [c for c in sub if c['ok_cap']]
    print('  %-8s 形状OK %4d  容量OK %4d  消 STORE %.1f + LOAD %.1f = %.1fMB'
          % (lev, len(sub), len(okc),
             sum(c['store_b'] for c in okc) / 1e6,
             sum(c['load_b'] for c in okc) / 1e6,
             sum(c['store_b'] + c['load_b'] for c in okc) / 1e6))
print('\n形状失败按杠杆：',
      dict(Counter(c['lev'] for c in cands if not c['ok_shape'])))
print('容量失败按杠杆：',
      dict(Counter(c['lev'] for c in both if not c['ok_cap'])))
print('\n容量失败样例（按可消字节排序前 10）：')
for c in sorted([c for c in both if not c['ok_cap']],
                key=lambda x: -(x['store_b'] + x['load_b']))[:10]:
    print('    %-46s m=%-7d k=%-5d res=%dW  想消 %.2fMB  %s' %
          (c['pmod'][:46], c['m'], c['k'], c['res_words'],
           (c['store_b'] + c['load_b']) / 1e6, '+'.join(c['kinds'])))
print('\n容量可行按可消字节排序前 15：')
for c in sorted(capok, key=lambda x: -(x['store_b'] + x['load_b']))[:15]:
    print('    %-46s m=%-7d k=%-5d res=%dW  消 %.2fMB  %s' %
          (c['pmod'][:46], c['m'], c['k'], c['res_words'],
           (c['store_b'] + c['load_b']) / 1e6, '+'.join(c['kinds'])))
json.dump(cands, open(os.path.join(HERE, 'chain_cands.json'), 'w'),
          ensure_ascii=False)
print('\n写出 chain_cands.json')
