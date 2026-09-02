# -*- coding: utf-8 -*-
"""probe_chain3.py — 按执行序（时间序）重建张量链，拿到 a3 可融合站点清单。

修正点：trace 的张量 id 是 Python id()，会被复用；生产者必须取
「seq < 消费者 seq 的记录里 out_ids 含该 tid 的最后一条」。
另外用 02_quant 校准表判定生产者 aug/host_bias（决定边界数值路径）。
"""
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
A2 = os.path.join(ROOT, '09_cbound', 'build_a2')
TRACE = os.path.join(ROOT, '09_cbound', 'ops_trace.json')
CAL = os.path.join(ROOT, '02_quant', 'hw_calib_table.json')
CTX = 131072

tr = json.load(open(TRACE))
recs = sorted(tr['ops'], key=lambda r: r['seq'])
hp = json.load(open(os.path.join(A2, 'host_plan.json')))
nodes = hp['nodes']
cal = json.load(open(CAL))
cal = cal.get('gemms', cal)

node_by = {}
for nd in nodes:
    if nd['kind'] == 'gemm':
        node_by[(nd['module'], nd['seq'])] = nd
heads_mode = {(nd['module'], nd['seq']) for nd in nodes
              if nd['kind'] == 'gemm' and nd.get('heads_mode')}

NORM_CLS = ('LayerNorm', 'RMSNorm', 'AdaRMSNorm', 'GroupNorm')
ACTV_CLS = ('SiLU', 'GELU', 'GELUActivation', 'ReLU')


def producer_of(tid, before_seq):
    """seq < before_seq 中 out_ids 含 tid 的最后一条记录（时间就近）。"""
    best = None
    for r in recs:
        if r['seq'] >= before_seq:
            break
        if tid in r.get('out_ids', []):
            best = r
    return best


def readers_between(tid, lo, hi):
    """seq ∈ (lo, hi) 内把 tid 当输入的记录（多消费者检测）。"""
    out = []
    for r in recs:
        if lo < r['seq'] < hi and tid in r.get('in_ids', []):
            out.append(r)
    return out


sites = []
for nd in nodes:
    if nd['kind'] != 'gemm' or nd.get('heads_mode'):
        continue
    c_rec = next((r for r in recs if r['seq'] == nd['seq']), None)
    if c_rec is None or not c_rec.get('in_ids'):
        continue
    seq_c = c_rec['seq']
    chain = []
    cur, sc = c_rec['in_ids'][0], seq_c
    while True:
        h = producer_of(cur, sc)
        if h is None or h['op'] != 'elem_norm' or \
                h['cls'] not in NORM_CLS + ACTV_CLS:
            break
        if readers_between(cur, h['seq'], sc):
            break                      # 中途有别的读者
        chain.append(h)
        cur, sc = h['in_ids'][0], h['seq']
    p = producer_of(cur, sc)
    if p is None or p['op'] != 'gemm':
        continue
    if readers_between(cur, p['seq'], sc):
        continue
    p_nd = node_by.get((p['module'], p['seq']))
    if p_nd is None or (p['module'], p['seq']) in heads_mode:
        continue
    ce = cal.get(p['module'])
    p_aug = p_nd.get('aug', False)
    p_hb = p_nd.get('host_bias', False)
    m, k = nd['m'], nd['k']
    m16 = (m + 15) // 16 * 16
    kinds = [h['cls'] for h in chain]
    is_norm = any(x in NORM_CLS for x in kinds)
    lev = ('S1-norm' if is_norm else 'S1-actv') if kinds else 'S2'
    ok_shape = (p_nd['m'] == m and p_nd['n'] == k)
    # 交错融合的 CTX 需求（单 tile 最大）：A_p + Y_p + Y_c 各一个 tile
    # 取 tile=4096 行估上界；W 驻留 = Σ组 k_eff 词
    kp = p_nd['k'] + 1
    w_res = kp * -(-p_nd['n'] // 108) + (k + 1) * -(-nd['n'] // 108)
    sites.append(dict(lev=lev, pmod=p['module'], cmod=nd['module'],
                      pseq=p['seq'], cseq=nd['seq'], m=m, k=k, n_c=nd['n'],
                      kinds=kinds, ok_shape=ok_shape, p_aug=p_aug,
                      p_hb=p_hb, w_res=w_res,
                      store_b=m16 * p_nd['n'] * 16 if ok_shape else 0,
                      load_b=m16 * (k + 1) * 16 if ok_shape else 0))

print('链完整的站点（p 与 c 都找到）：', len(sites))
for lev in ('S2', 'S1-actv', 'S1-norm'):
    sub = [s for s in sites if s['lev'] == lev]
    ok = [s for s in sub if s['ok_shape']]
    okh = [s for s in ok if not s['p_hb']]
    print('%-8s 全部 %4d  形状OK %4d  非host_bias %4d  '
          '消 STORE %.1f + LOAD %.1f MB' %
          (lev, len(sub), len(ok), len(okh),
           sum(s['store_b'] for s in okh) / 1e6,
           sum(s['load_b'] for s in okh) / 1e6))
print('\n形状失败站点的类型：',
      dict(Counter(s['lev'] for s in sites if not s['ok_shape'])))
print('host_bias 生产者的可行站点：',
      sum(1 for s in sites if s['ok_shape'] and s['p_hb']),
      '（保留 DDR 往返）')
print('aug 生产者（含进权重的 bias）：',
      sum(1 for s in sites if s['ok_shape'] and s['p_aug']))
print('\nhost 步骤类别分布（形状OK 内）：',
      dict(Counter(','.join(s['kinds']) for s in sites
                   if s['ok_shape'])))
print('\n最大 15 个可行站点（按可消字节）：')
for s in sorted([s for s in sites if s['ok_shape'] and not s['p_hb']],
                key=lambda x: -(x['store_b'] + x['load_b']))[:15]:
    print('  %-8s %-48s m=%-7d k=%-5d 消 %.1fMB %s' %
          (s['lev'], s['pmod'][:48], s['m'], s['k'],
           (s['store_b'] + s['load_b']) / 1e6, ','.join(s['kinds'])))
json.dump(sites, open(os.path.join(HERE, 'sites.json'), 'w'),
          ensure_ascii=False)
print('\n写出 sites.json')
