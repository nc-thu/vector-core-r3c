# -*- coding: utf-8 -*-
"""probe_sites4.py — a3 融合站点普查（修正口径，2026-08-31）

前三版 probe 的教训：
  * ceil16(x)=(x+15)//16 是「行组数」不是 padded 行数——之前的字节/容量
    估算全部大了 16 倍。本版按 A 图字数 = ceil16(m)×k_eff、字节 = 字数×16。
  * 站点必须是 host_plan.nodes 里的「连续节点段」[p, h1..hk, c]——中间夹
    任何别的节点（GEMM/注意力/host 其他算子）都不融合，避免执行序重排。
  * 多消费者用时间序生产者匹配判定（trace 的 id() 会复用）。

产出 sites4.json + 分桶统计，直接作为 compiler_a3 的可行性预演。
"""
import json
import os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
A2 = os.path.join(ROOT, '09_cbound', 'build_a2')
TRACE = os.path.join(ROOT, '09_cbound', 'ops_trace.json')
CTX = 131072
W_WORDS = 4096

NORM_CLS = ('LayerNorm', 'RMSNorm', 'AdaRMSNorm')
ACTV_CLS = ('SiLU', 'GELU', 'GELUActivation', 'ReLU')

tr = json.load(open(TRACE))
recs = sorted(tr['ops'], key=lambda r: r['seq'])
by_seq = {r['seq']: r for r in recs}
hp = json.load(open(os.path.join(A2, 'host_plan.json')))
nodes = hp['nodes']

# norm 权重（引擎代理从公开 checkpoint 抓的 γ/β）
NW_PATH = os.path.join(ROOT, '12_actv', 'data', 'norm_weights.npz')
nw = dict(np.load(NW_PATH)) if os.path.exists(NW_PATH) else {}
nw_mods = sorted({k.rsplit('.', 1)[0] for k in nw})


def producer_of(tid, before_seq):
    """seq < before_seq 中 out_ids 含 tid 的最后一条（时间就近）。"""
    best = None
    for r in recs:
        if r['seq'] >= before_seq:
            break
        if tid in r.get('out_ids', []):
            best = r
    return best


def readers_of(tid, lo, hi):
    """seq ∈ (lo, hi) 内读 tid 的记录。"""
    return [r for r in recs
            if lo < r['seq'] < hi and tid in r.get('in_ids', [])]


sites = []
buckets = Counter()
for i, nd in enumerate(nodes):
    if nd['kind'] != 'gemm' or nd.get('heads_mode'):
        continue
    c = nd
    # ---- 回走连续 host 节点（且必须是 norm/actv 类）----
    chain = []
    j = i - 1
    while j >= 0 and nodes[j]['kind'] == 'host' and \
            nodes[j]['cls'] in NORM_CLS + ACTV_CLS:
        chain.append(nodes[j])
        j -= 1
    p = nodes[j] if j >= 0 else None
    row = dict(cmod=c['module'], cseq=c['seq'], m=c['m'], k=c['k'],
               n_c=c['n'], chain=[h['cls'] for h in chain],
               chain_mods=[h['module'] for h in chain])
    if p is None or p['kind'] != 'gemm' or p.get('heads_mode'):
        buckets['no_pl_producer'] += 1
        continue
    row.update(pmod=p['module'], pseq=p['seq'], pm=p['m'], pn=p['n'],
               pk=p['k'])
    if p.get('host_bias'):
        buckets['p_host_bias'] += 1
        continue
    # 形状直连：host norm/actv 不改形状
    if p['m'] != c['m'] or p['n'] != c['k']:
        buckets['shape'] += 1
        continue
    # ---- trace 张量链核验：p 输出 → h1 → … → hk → c，中途无其他读者 ----
    c_rec = by_seq[c['seq']]
    cur = c_rec['in_ids'][0]
    sc = c['seq']
    ok = True
    for h in reversed(chain):          # 时间正序：离 p 近的先核
        h_rec = by_seq[h['seq']]
        prod = producer_of(cur, sc)
        if prod is None or prod['seq'] != h['seq']:
            ok = False
            break
        if readers_of(cur, prod['seq'], sc):
            ok = False
            break
        cur, sc = h_rec['in_ids'][0], h['seq']
    if ok:
        prod = producer_of(cur, sc)
        ok = prod is not None and prod['seq'] == p['seq'] and \
            not readers_of(cur, prod['seq'], sc)
    if not ok:
        buckets['tensor_link'] += 1
        continue
    # 链上张量在 c 之后还有没有真读者（时间序判别，防 id 复用误伤）
    late = False
    for r in recs:
        if r['seq'] <= c['seq'] or late:
            break
        for t in r.get('in_ids', []):
            pr = producer_of(t, r['seq'])
            if pr is not None and p['seq'] <= pr['seq'] <= c['seq']:
                late = True
                break
    if late:
        buckets['late_reader'] += 1
        continue

    kinds = [h['cls'] for h in chain]
    is_norm = any(x in NORM_CLS for x in kinds)
    # NORM 链：stride 必须 == 宽度 → Y_p 步长 = k（无常数列）→ c 不能 aug
    if is_norm and c.get('aug'):
        buckets['norm_aug_c'] += 1
        continue
    if is_norm:
        miss = [m for m in set(row['chain_mods']) if m not in nw_mods]
        if miss:
            buckets['norm_no_weights'] += 1
            row['miss_w'] = miss
            continue

    m, k, n_c, pk = c['m'], c['k'], c['n'], p['k']
    g = (m + 15) // 16                      # 行组数（ceil16 口径）
    pitch = k if is_norm else k + 1          # Y_p 步长
    # 可消字节（a2 里 p 的 Y 落 DDR + c 的 A 首装）
    row['store_b'] = (p['m'] + 15) // 16 * p['n'] * 16
    row['load_b'] = g * (k + 1) * 16
    # tile 预算：A_p + Y_p + Y_c 同活；表映像只占一次
    tbl = 256 if not is_norm else 1 + 4 * ((k + 15) // 16)
    gt = max(1, (int(CTX * 0.9) - tbl - 256) // (pk + 1 + pitch + n_c))
    row['ntiles'] = max(1, -(-g // gt))
    # W 驻留：Σ组 k_eff ≤ 半区 2048 才能跨 tile 驻留
    wres = (pk + 1) * -(-p['n'] // 108) + (k + 1) * -(-n_c // 108)
    row['w_res_ok'] = wres <= W_WORDS // 2
    row['lev'] = 'S1-norm' if is_norm else ('S1-actv' if chain else 'S2')
    row['ok'] = True
    buckets['ok_' + row['lev']] += 1
    sites.append(row)

print('全部普通 gemm 消费者（含不融合的）分桶：')
for kk, n in buckets.most_common():
    print('  %-18s %4d' % (kk, n))
ok = [s for s in sites]
print('\n可行站点 %d 个' % len(ok))
for lev in ('S2', 'S1-actv', 'S1-norm'):
    sub = [s for s in ok if s['lev'] == lev]
    print('  %-8s %4d 站  消 STORE %.2fMB + LOAD %.2fMB  host步 %d'
          % (lev, len(sub), sum(s['store_b'] for s in sub) / 1e6,
             sum(s['load_b'] for s in sub) / 1e6, sum(len(s['chain']) for s in sub)))
print('host 步类别（链内）:',
      dict(Counter(c for s in ok for c in s['chain'])))
print('\n最大 12 站（按可消字节）：')
for s in sorted(ok, key=lambda x: -(x['store_b'] + x['load_b']))[:12]:
    print('  %-8s %-52s m=%-6d k=%-5d 消 %6.2fMB tiles=%d %s'
          % (s['lev'], s['pmod'][:52], s['m'], s['k'],
             (s['store_b'] + s['load_b']) / 1e6, s['ntiles'],
             ','.join(s['chain'])))
json.dump(dict(buckets=dict(buckets), sites=ok),
          open(os.path.join(HERE, 'sites4.json'), 'w'), ensure_ascii=False)
print('\n写出 sites4.json')
