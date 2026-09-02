# -*- coding: utf-8 -*-
"""analyze_copy.py — COPY 5.4 万条从哪来 + norm 张量形状 + 阶段归因（2026-08-31）"""
import json
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, '03_compiler', 'build_full_fixed')
SIM = os.path.join(ROOT, '05_sim')

stage_map = {}
for line in open(os.path.join(SIM, 'seg_stage.json')) if os.path.exists(
        os.path.join(SIM, 'seg_stage.json')) else []:
    pass
# seg_stage.json 结构未知，先读一下
sj = json.load(open(os.path.join(SIM, 'seg_stage.json')))
if isinstance(sj, dict):
    for k, v in sj.items():
        if k.startswith('seg_'):
            stage_map[int(k.split('_')[1])] = v
elif isinstance(sj, list):
    for i, v in enumerate(sj):
        stage_map[i] = v


def decode(d):
    return dict(op=(d >> 252) & 0xF, b_src=(d >> 246) & 7, m=(d >> 228) & 0xFFFF,
                n=(d >> 212) & 0xFFFF, k=(d >> 196) & 0xFFFF)


copy_by_stage = Counter()
copy_by_mod = Counter()
gemm_by_stage = Counter()
n_copy_total = 0
seg_owner = {}
for i in range(2782):
    mf = os.path.join(BUILD, 'segments', f'seg_{i:04d}', 'manifest.json')
    man = json.load(open(mf))
    owner = ''
    if man.get('outputs'):
        owner = (man['outputs'][0].get('module') or '')
    seg_owner[i] = owner
    words = []
    for line in open(os.path.join(BUILD, 'segments', f'seg_{i:04d}', 'seq.mem')):
        line = line.strip()
        if line:
            words.append(int(line, 16))
    for w in words:
        f = decode(w)
        if f['op'] == 15:
            break
        if f['op'] == 0 or f['op'] == 2:
            gemm_by_stage[stage_map.get(i, '?')] += 1
        elif f['op'] == 3:
            n_copy_total += 1
            copy_by_stage[stage_map.get(i, '?')] += 1
            # 归到顶层模块族
            fam = owner.split('.')[0] if owner else '(no-out)'
            copy_by_mod[(fam, stage_map.get(i, '?'))] += 1

print('COPY 总条数', n_copy_total)
print('\n按阶段：')
for k, v in copy_by_stage.most_common():
    print(f'  {k:24s} {v:>7,}   (GEMM {gemm_by_stage.get(k,0):>6,})')
print('\n按产出模块族×阶段 top15：')
for (fam, st), v in copy_by_mod.most_common(15):
    print(f'  {fam:28s} {st:24s} {v:>7,}')

# host 步骤按阶段
hp = json.load(open(os.path.join(BUILD, 'host_plan.json')))
hs_by_stage = defaultdict(Counter)
for h in hp['host_steps']:
    st = stage_map.get(h['after_seg'], '?')
    hs_by_stage[h['kind']][st] += 1
print('\nhost 步骤按阶段：')
for kind in sorted(hs_by_stage):
    print(f'  {kind:16s}', dict(hs_by_stage[kind].most_common()))

# 边界拍数按阶段（用 boundary_account 的逐边界明细重算不方便，直接按段归）
types = json.load(open(os.path.join(SIM, 'types.json')))
cbt = json.load(open(os.path.join(SIM, 'cycles_by_type.json')))
seg_cyc = {}
for t in types['types']:
    for s in t['instances']:
        seg_cyc[s] = cbt[t['type_id']]['cycles']['pf1']
print('\n各阶段拍数核对：')
st_tot = Counter()
for i, c in seg_cyc.items():
    st_tot[stage_map.get(i, '?')] += c
for k, v in st_tot.most_common():
    print(f'  {k:24s} {v:>13,}')
