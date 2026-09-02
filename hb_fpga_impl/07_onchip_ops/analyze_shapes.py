# -*- coding: utf-8 -*-
"""analyze_shapes.py — 各 host 算子张量形状 / softmax S 阵形状 / 引擎拍数与净收益折算
产出 benefit_table.csv / benefit_table.json（07_onchip_ops，2026-08-31）"""
import csv
import json
import math
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, '03_compiler', 'build_full_fixed')
HERE = os.path.dirname(os.path.abspath(__file__))
F = 198.5e6

# 复用 analyze_boundaries 的逐段统计（直接重算，保持独立可跑）
BURST_B, AR_OVH, CMD_OVH = 2048, 2, 5


def load_ctx_ideal(nbytes):
    return nbytes // 8 + math.ceil(nbytes / BURST_B) * AR_OVH + CMD_OVH


def store_cycles(nbytes):
    return ((nbytes + 15) // 16) * 5 + CMD_OVH


def decode(d):
    return dict(op=(d >> 252) & 0xF, b_src=(d >> 246) & 7, m=(d >> 228) & 0xFFFF,
                n=(d >> 212) & 0xFFFF, k=(d >> 196) & 0xFFFF,
                dma_len=(d >> 61) & 0x3FFFF)


N_SEG = 2782
seg_store_c = [0.0] * N_SEG
seg_store_b = [0] * N_SEG
seg_lctx_c = [0.0] * N_SEG
seg_lctx_b = [0] * N_SEG
for i in range(N_SEG):
    words = []
    for line in open(os.path.join(BUILD, 'segments', f'seg_{i:04d}', 'seq.mem')):
        line = line.strip()
        if line:
            words.append(int(line, 16))
    for w in words:
        f = decode(w)
        if f['op'] == 15:
            break
        if f['op'] == 4 and f['b_src'] == 0:
            seg_lctx_c[i] += load_ctx_ideal(f['dma_len'])
            seg_lctx_b[i] += f['dma_len']
        elif f['op'] == 5:
            seg_store_c[i] += store_cycles(f['dma_len'])
            seg_store_b[i] += f['dma_len']

hp = json.load(open(os.path.join(BUILD, 'host_plan.json')))
bounds = defaultdict(list)
for h in hp['host_steps']:
    bounds[h['after_seg']].append(h)

# ---------------- 张量形状：边界下一段的输入（m,k = 张量 rows×C）----------------
shapes = defaultdict(list)  # kind -> [(m,k)]
attn_shapes = defaultdict(list)  # kind:cls -> [(mq,mk,units,H,d)]
for a, hs in sorted(bounds.items()):
    nxt = a + 1
    mk = None
    if 0 <= nxt < N_SEG:
        man = json.load(open(os.path.join(BUILD, 'segments', f'seg_{nxt:04d}',
                                          'manifest.json')))
        for inp in man.get('inputs', []):
            if inp.get('kind') == 'act_in':
                mk = (inp.get('m', 0), inp.get('k', 0))
                break
    for h in hs:
        shapes[h['kind']].append(mk)
        if 'attn' in h:
            a_ = h['attn']
            attn_shapes[f"{h['kind']}:{h['cls']}"].append(
                (a_.get('mq', 0), a_.get('mk', 0), a_.get('units', 1),
                 a_.get('H', 1), a_.get('d', 1), a_.get('family', '')))

print('=== norm/actv 边界处的张量形状（下一段 A 图 m×k）===')
for kind in ('norm', 'actv', 'rotary'):
    mks = [x for x in shapes[kind] if x]
    if not mks:
        continue
    rows = sum(m * k for m, k in mks)
    print(f'{kind:8s} n={len(mks):4d}  Σm×k={rows:,}  行长 k 分布:',
          dict(Counter(k for _, k in mks).most_common(8)))

print('\n=== softmax 边界的注意力几何 ===')
sm_s_elems = 0
for key, lst in sorted(attn_shapes.items()):
    if not key.startswith('softmax:'):
        continue
    tot = sum(mq * mk * u for mq, mk, u, H, d, fam in lst)
    sm_s_elems += tot
    print(f'{key:44s} n={len(lst):4d}  Σ(mq×mk×units)={tot:,}')
print(f'softmax S 元素总计 {sm_s_elems:,}')

# rotary 张量：q+k = 边界前段 STORE 字节 × 2/3（qkv 三等分）
rot_bytes = 0
for a, hs in bounds.items():
    if any(h['kind'] == 'rotary' for h in hs) and 0 <= a < N_SEG:
        rot_bytes += seg_store_b[a] * (2.0 / 3.0) / len(hs)
print(f'\nrotary q+k 字节估算 {rot_bytes:,.0f}')

# ---------------- 引擎拍数模型 ----------------
# AE_ACTV：16-lane 广播读/写，norm 两遍（2C+45/16行组），eltwise 一遍（C+8/16行组）
# rotary：C/2 对 ×2 遍 ≈ C+8；bias：C+8；sm-bias：C+8
def eng_norm(elems, rowgroups):
    return elems / 16 * 2 + 45 * rowgroups


def eng_elt(elems, rowgroups):
    return elems / 16 + 8 * rowgroups


# 每类算子的元素量（边界前段 STORE 字节 = 生产段输出 = 算子输入）
kind_elem = defaultdict(float)
kind_rg = defaultdict(float)
for a, hs in bounds.items():
    if not (0 <= a < N_SEG):
        continue
    for h in hs:
        k = h['kind']
        w = 1.0 / len(hs)
        if k in ('norm', 'actv', 'softmax', 'swin_partition', 'swin_reverse',
                 'swin_split', 'other', 'im2col', 'softmax_prep', 'deform_host'):
            kind_elem[k] += seg_store_b[a] * w
            mk = None
            nxt = a + 1
            if nxt < N_SEG:
                man = json.load(open(os.path.join(
                    BUILD, 'segments', f'seg_{nxt:04d}', 'manifest.json')))
                for inp in man.get('inputs', []):
                    if inp.get('kind') == 'act_in':
                        mk = (inp.get('m', 0), inp.get('k', 0))
                        break
            kind_rg[k] += (mk[0] / 16 if mk else seg_store_b[a] / 256.0 / 16) * w
        elif k == 'rotary':
            kind_elem[k] += seg_store_b[a] * (2.0 / 3.0) * w
            kind_rg[k] += seg_store_b[a] * (2.0 / 3.0) / 1024.0 * w

# 边界往返拍（同 analyze_boundaries 口径）
kind_rt = defaultdict(float)
for a, hs in bounds.items():
    c = (seg_store_c[a] if 0 <= a < N_SEG else 0) + \
        (seg_lctx_c[a + 1] if a + 1 < N_SEG else 0)
    for h in hs:
        kind_rt[h['kind']] += c / len(hs)

print('\n=== 每类算子：往返拍 / 引擎拍 / 净省拍 ===')
rows = []
for k in sorted(kind_rt, key=lambda x: -kind_rt[x]):
    rt = kind_rt[k]
    el = kind_elem[k]
    rg = kind_rg[k]
    if k == 'norm':
        ec = eng_norm(el, rg)
    else:
        ec = eng_elt(el, rg)
    rows.append((k, rt, el, ec, rt - ec))
    print(f'{k:16s} 往返 {rt:>12,.0f}  元素 {el:>13,.0f}  引擎 {ec:>10,.0f}  净省 {rt-ec:>12,.0f}')

# softmax 特例：S 元素 = attn 几何（比 STORE 字节口径准）
sm_ec = sm_s_elems / 16 * 4 + 8 * 220  # bias 遍 + SM16 三遍
print(f'\nsoftmax 用 attn 几何：S 元素 {sm_s_elems:,}  引擎(含 SM16) {sm_ec:,.0f}')

with open(os.path.join(HERE, 'benefit_table.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['kind', 'roundtrip_cycles', 'tensor_bytes', 'engine_cycles',
                'net_save_cycles', 'net_save_ms_at_198p5'])
    for r in rows:
        w.writerow([r[0], round(r[1]), round(r[2]), round(r[3]), round(r[4]),
                    round((r[4]) / F * 1000, 1)])
json.dump({'rows': [(r[0], round(r[1]), round(r[2]), round(r[3]), round(r[4]))
                    for r in rows],
           'softmax_S_elements': sm_s_elems},
          open(os.path.join(HERE, 'benefit_table.json'), 'w'), indent=1)
print('\n写出 benefit_table.csv / benefit_table.json')
