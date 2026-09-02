# -*- coding: utf-8 -*-
"""taskA_boundary.py — a2 段边界流量分类（11_overlap_plan，2026-08-31）

口径
  数据：09_cbound/build_a2（2762 段，只读）。逐段解码 seq.mem：
    STORE 字节、LOAD_CTX 按 (b_base,dma_len) 段内去重 → 首装/双装。
  边界 b = seg_b 与 seg_{b+1} 之间；host_steps(after_seg=b) 为该边界上的 CPU 算子。
    边界产出账 = seg_b 全部 STORE（消费者=该边界 host 算子，无 host 步则为 PL→PL）
    边界输入账 = seg_{b+1} 的 LOAD_CTX 首装（生产者=同上）
  拍数换算（模型口径，09_cbound 校准系数反推的有效带宽）：
    STORE 2.718 B/cyc、LOAD_CTX 3.701 B/cyc、LOAD_W 5.668 B/cyc
  AE_ACTV 可覆盖 host 算子 kind ∈ {norm, actv, rotary, softmax_prep}
  （softmax 本体是 S→P 归约，不算）；边界可消 = 该边界全部 host kind 可覆盖。
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BUILD = os.path.join(ROOT, '09_cbound', 'build_a2')
NSeg = 2762

# 有效带宽（模型口径）：由 a2 分量拍/字节反推
BW_STORE = 751.35e6 / 276.42e6   # = 2.718 B/cyc
BW_CTX = 837.57e6 / 226.35e6     # = 3.701 B/cyc


def decode(d):
    return dict(op=(d >> 252) & 0xF, b_src=(d >> 246) & 7,
                m=(d >> 228) & 0xFFFF, n=(d >> 212) & 0xFFFF,
                k=(d >> 196) & 0xFFFF, dma_len=(d >> 61) & 0x3FFFF,
                j0=(d >> 62) & 0xFFFF, b_base=(d >> 156) & 0xFFFFF,
                dma_addr=(d >> 29) & 0xFFFFFFFF)   # DDR 源地址（compiler desc）


def seg_stats(path):
    st = 0                # store bytes
    ctx_first = 0         # load_ctx 首装 bytes
    ctx_dbl = 0           # load_ctx 段内双装 bytes
    seen = set()
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = decode(int(line, 16))
        op = d['op']
        if op == 15:
            break
        if op == 4 and d['b_src'] == 0:
            key = (d['dma_addr'], d['dma_len'])   # 同源同长=同内容双装
            if key in seen:
                ctx_dbl += d['dma_len']
            else:
                seen.add(key)
                ctx_first += d['dma_len']
        elif op == 5:
            st += d['dma_len']
    return st, ctx_first, ctx_dbl


# ---- 1. 逐段解码 ----
store = [0] * NSeg
first = [0] * NSeg
dbl = [0] * NSeg
for i in range(NSeg):
    store[i], first[i], dbl[i] = seg_stats(
        os.path.join(BUILD, 'segments', 'seg_%04d' % i, 'seq.mem'))

tot_store = sum(store)
tot_first = sum(first)
tot_dbl = sum(dbl)
print('校验：STORE %.1fMB（已定案 751.5）  首装 %.1fMB（784.1）  双装 %.1fMB（53.5）'
      % (tot_store / 1e6, tot_first / 1e6, tot_dbl / 1e6))

# ---- 2. 边界与 host 算子 ----
hp = json.load(open(os.path.join(BUILD, 'host_plan.json')))
bounds = defaultdict(list)
for h in hp['host_steps']:
    bounds[h['after_seg']].append(h)

COVER = {'norm', 'actv', 'rotary', 'softmax_prep'}


def bucket(h):
    """host 步骤 → 任务要求的 9 桶"""
    k, mod, cls = h['kind'], h['module'], h['cls']
    if mod == 'decoder' or cls in ('UpsampleHead', 'HoloBrainActionDecoder',
                                   'HoloBrainRobotStateEncoder'):
        return 'solver/输出头'
    if mod.startswith('spatial_enhancer'):
        return 'PSE'
    if k == 'deform_host' or mod.startswith('feature_enhancer') and \
            'Deform' in cls:
        return 'deformable'
    if k in ('swin_partition', 'swin_split', 'swin_reverse'):
        return '窗口重排'
    if k == 'norm':
        return 'norm 族'
    if k == 'actv':
        return 'gelu/actv'
    if k in ('softmax', 'softmax_prep'):
        return 'softmax'
    if k == 'rotary':
        return 'rotary'
    return '其他'


# ---- 3. 边界分类账 ----
rows = []   # 每边界一行
for b in range(0, NSeg):          # 边界 b：seg_b 之后（b=NSeg-1 之后即 2761）
    hs = bounds.get(b, [])
    sb = store[b] if b < NSeg else 0
    fb = first[b + 1] if b + 1 < NSeg else 0
    if not hs:
        kind = 'PL→PL'           # 无 host 步：跨段驻留可消
        cover = True
        bk = 'PL→PL(无host)'
    else:
        kinds = {h['kind'] for h in hs}
        cover = kinds <= COVER
        bks = {bucket(h) for h in hs}
        # 边界类别：全部同桶取该桶；混合取并集（排序拼接）
        bk = '+'.join(sorted(bks))
        kind = 'host:' + bk
    rows.append(dict(b=b, n_host=len(hs), kind=kind, bucket=bk, cover=cover,
                     has_host=bool(hs), store_b=sb, first_b=fb))

# ---- 4. 汇总：按桶分 GB ----
agg = defaultdict(lambda: dict(nb=0, store=0, first=0, cover_store=0,
                               cover_first=0))
for r in rows:
    a = agg[r['bucket']]
    a['nb'] += 1
    a['store'] += r['store_b']
    a['first'] += r['first_b']
    if r['has_host'] and r['cover']:      # AE_ACTV 可消（host 边界、全可覆盖）
        a['cover_store'] += r['store_b']
        a['cover_first'] += r['first_b']

print('\n===== 边界分桶（STORE=边界前段写回，FIRST=后段首装；MB / 模型拍）=====')
print('%-22s %6s %10s %10s %10s %10s' %
      ('桶', '边界数', 'STORE MB', 'STORE 拍', '首装 MB', '首装拍'))
order = sorted(agg.items(), key=lambda kv: -(kv[1]['store'] + kv[1]['first']))
for k, a in order:
    print('%-22s %6d %10.1f %10.1f %10.1f %10.1f' %
          (k, a['nb'], a['store'] / 1e6, a['store'] / BW_STORE / 1e6,
           a['first'] / 1e6, a['first'] / BW_CTX / 1e6))
ts = sum(a['store'] for a in agg.values())
tf = sum(a['first'] for a in agg.values())
print('%-22s %6d %10.1f %10.1f %10.1f %10.1f' %
      ('合计', sum(a['nb'] for a in agg.values()), ts / 1e6,
       ts / BW_STORE / 1e6, tf / 1e6, tf / BW_CTX / 1e6))

# ---- 5. 三个消除杠杆 ----
ae_store = sum(r['store_b'] for r in rows if r['has_host'] and r['cover'])
ae_first = sum(r['first_b'] for r in rows if r['has_host'] and r['cover'])
pl_store = sum(r['store_b'] for r in rows if not r['has_host'])
pl_first = sum(r['first_b'] for r in rows if not r['has_host'])
# 跨段驻留只消无 host 边界；AE_ACTV 只消 host 边界 → 两者不重叠，直接相加
res_store = tot_store - ae_store - pl_store
res_first = tot_first - ae_first - pl_first

print('\n===== 消除杠杆（模型口径拍 = 字节/有效带宽）=====')
print('AE_ACTV（norm+actv+rotary+softmax_prep 边界，%d 个）:'
      % sum(1 for r in rows if r['has_host'] and r['cover']))
print('  STORE %.1fMB (%.1fM拍) + 首装 %.1fMB (%.1fM拍) = 合计 %.1fM拍'
      % (ae_store / 1e6, ae_store / BW_STORE / 1e6, ae_first / 1e6,
         ae_first / BW_CTX / 1e6,
         (ae_store / BW_STORE + ae_first / BW_CTX) / 1e6))
print('跨段驻留（无 host 边界，%d 个）:'
      % sum(1 for r in rows if not r['has_host']))
print('  STORE %.1fMB (%.1fM拍) + 首装 %.1fMB (%.1fM拍) = 合计 %.1fM拍'
      % (pl_store / 1e6, pl_store / BW_STORE / 1e6, pl_first / 1e6,
         pl_first / BW_CTX / 1e6, (pl_store / BW_STORE + pl_first / BW_CTX) / 1e6))
print('叠加两者：STORE %.1fMB + 首装 %.1fMB → 残留 STORE %.1fMB / 首装 %.1fMB'
      % ((ae_store + pl_store) / 1e6, (ae_first + pl_first) / 1e6,
         res_store / 1e6, res_first / 1e6))
print('残留边界（host 算子不可覆盖，必须走 DDR）：%d 个'
      % sum(1 for r in rows if r['has_host'] and not r['cover']))

# 残留边界细分
res = defaultdict(lambda: [0, 0, 0])
for r in rows:
    if r['has_host'] and not r['cover']:
        res[r['bucket']][0] += 1
        res[r['bucket']][1] += r['store_b']
        res[r['bucket']][2] += r['first_b']
print('\n残留边界细分：')
for k, (n, s, f) in sorted(res.items(), key=lambda kv: -(kv[1][1] + kv[1][2])):
    print('  %-28s %4d 边界  STORE %8.1fMB  首装 %8.1fMB'
          % (k, n, s / 1e6, f / 1e6))

json.dump(dict(
    totals=dict(store=tot_store, first=tot_first, dbl=tot_dbl),
    by_bucket={k: a for k, a in agg.items()},
    levers=dict(ae_actv=dict(store=ae_store, first=ae_first,
                             n_bounds=sum(1 for r in rows if r['has_host'] and r['cover'])),
                pl_resident=dict(store=pl_store, first=pl_first,
                                 n_bounds=sum(1 for r in rows if not r['has_host'])),
                residual=dict(store=res_store, first=res_first,
                              n_bounds=sum(1 for r in rows if r['has_host'] and not r['cover']))),
    rows=rows,
), open(os.path.join(HERE, 'boundary_account_a2.json'), 'w'), ensure_ascii=False)
print('\n写出 boundary_account_a2.json')
