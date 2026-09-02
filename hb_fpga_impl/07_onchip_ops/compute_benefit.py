# -*- coding: utf-8 -*-
"""compute_benefit.py — 严格口径收益表：只算"整条边界都被子集覆盖而消失"的往返拍，
再减去新引擎自己要跑的拍。产出 benefit_estimation.json / benefit_estimation.csv。
口径见 NOTES.txt 第一节。2026-08-31。"""
import csv
import json
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, '03_compiler', 'build_full_fixed')
HERE = os.path.dirname(os.path.abspath(__file__))
F = 198.5e6
TOTAL_CYC = 1080821046

# ---- RTL 实测常数（gem_cycles.py） ----


def lci(nb):
    return nb // 8 + math.ceil(nb / 2048) * 2 + 5


def scc(nb):
    return ((nb + 15) // 16) * 5 + 5


def decode(d):
    return dict(op=(d >> 252) & 0xF, b_src=(d >> 246) & 7,
                dma_len=(d >> 61) & 0x3FFFF)


N = 2782
seg = []
for i in range(N):
    st, sb, lt, lb = 0, 0, 0, 0
    for line in open(os.path.join(BUILD, 'segments', f'seg_{i:04d}', 'seq.mem')):
        line = line.strip()
        if not line:
            continue
        w = int(line, 16)
        f = decode(w)
        if f['op'] == 15:
            break
        if f['op'] == 5:
            st += scc(f['dma_len'])
            sb += f['dma_len']
        elif f['op'] == 4 and f['b_src'] == 0:
            lt += lci(f['dma_len'])
            lb += f['dma_len']
    seg.append((st, sb, lt, lb))

hp = json.load(open(os.path.join(BUILD, 'host_plan.json')))
bounds = defaultdict(list)
for h in hp['host_steps']:
    bounds[h['after_seg']].append(h)

# softmax 的 S 元素量用 attn 几何（host 步骤自带），比 STORE 字节代理准
S_ELEM = defaultdict(int)
for h in hp['host_steps']:
    if 'attn' in h and h['kind'] == 'softmax':
        a = h['attn']
        S_ELEM[h['after_seg']] += a['mq'] * a['mk'] * a.get('units', 1)

# 每类算子的引擎模型（元素量 → 拍数）
#   norm 两遍(读A口2次/元素组) + 每行组 45 拍开销
#   actv/rotary/bias/scatter 一遍 + 每行组 8 拍
#   softmax = AE_ACTV bias 遍(S/16) + SM16 三遍(3S/16) + 45 拍/行组


def eng(kind, elems):
    if elems <= 0:
        return 0.0
    if kind == 'norm':
        return elems / 16 * 2 + 45 * (elems / 256 / 16)
    if kind == 'softmax':
        return elems / 16 * 4 + 45 * (elems / 256 / 16)
    return elems / 16 + 8 * (elems / 256 / 16)


SETS = {
    'MVP': {'norm', 'actv', 'softmax', 'rotary'},
    'MVP+swin': {'norm', 'actv', 'softmax', 'rotary', 'swin_partition',
                 'swin_split', 'swin_reverse', 'softmax_prep'},
}

res = {}
for name, S in SETS.items():
    rt = eng_c = 0.0
    nb = nsteps = 0
    per_op = defaultdict(lambda: [0.0, 0.0])  # kind -> [往返, 引擎]
    for a, hs in bounds.items():
        if not all(h['kind'] in S for h in hs):
            continue
        st, sb, lt, lb = seg[a] if 0 <= a < N else (0, 0, 0, 0)
        st2, sb2, lt2, lb2 = seg[a + 1] if a + 1 < N else (0, 0, 0, 0)
        rt_b = st + lt2
        rt += rt_b
        nb += 1
        nsteps += len(hs)
        e = 0.0
        for h in hs:
            k = h['kind']
            if k == 'softmax':
                el = S_ELEM.get(a, sb)  # S 元素（无几何信息退回字节代理）
            elif k == 'rotary':
                el = sb * 2.0 / 3.0
            elif k == 'softmax_prep':
                el = lb2  # 物化的是"装载侧"的 Q/K/VT 块体积
            else:
                el = sb
            ee = eng(k, el)
            e += ee
            per_op[k][0] += rt_b / len(hs)
            per_op[k][1] += ee
        eng_c += e
    res[name] = dict(boundaries=nb, host_steps=nsteps, roundtrip=rt,
                     engine=eng_c, net=rt - eng_c)

base_ms = TOTAL_CYC / F * 1000
print(f'基线 {TOTAL_CYC:,} 拍 = {base_ms:.0f} ms')
for name, r in res.items():
    net = r['net']
    print(f"{name:10s} 消边界 {r['boundaries']:4d}/1151  消host步骤 {r['host_steps']:4d}/1424  "
          f"往返 {r['roundtrip']/1e6:6.1f}M  引擎 {r['engine']/1e6:5.1f}M  "
          f"净省 {net/1e6:6.1f}M 拍 = {net/F*1000:5.0f} ms ({net/TOTAL_CYC*100:4.1f}%)"
          f"  → {(TOTAL_CYC-net)/F*1000:.0f} ms, MAC利用率 "
          f"{145824266752/(1728*(TOTAL_CYC-net))*100:.2f}%")

# 逐算子表（分摊口径，用于排序）
rows = []
spec = [
    # kind, 方案, LUT, BRAM36, 独立可建?
    ('actv', 'AE_ACTV eltwise 模式（256×8 直查 ROM ×16 lane）', 1500, 0),
    ('softmax', 'SM16 复用 + 独立 softmax 描述符 + swin/bimha bias 预加遍', 1800, 1),
    ('norm', 'AE_ACTV LN/RMS 模式（整数累加 + rsqrt LUT + 两级缩放）', 9000, 3),
    ('rotary', 'AE_ACTV rotary 模式（cos/sin 表 + 成对双乘，复用乘法器）', 1000, 3),
    ('bias', 'AE_ACTV bias 模式（rq_v2 同款 + 加法）——其他算子融合的前置', 500, 1),
    ('swin', '散射引擎（16×16 字节交叉 + 窗地址 ALU）：partition/reverse/split/prep', 2500, 0),
    ('im2col', '不做硬件（0.94M 拍太小）；保留编译期布局', 0, 0),
    ('deform', '不做（4-tap gather + solve 性价比低，留 host）', 0, 0),
]
# 分摊口径数字来自 analyze_boundaries/analyze_shapes（已跑）
per_op_numbers = {
    'actv': dict(steps=331, bounds=321, rt=15414208, el=19644992, eng=1298905),
    'norm': dict(steps=423, bounds=404, rt=73376375, el=116021772, eng=15525540),
    'softmax': dict(steps=220, bounds=220, rt=22721760, el=28798304, eng=7201336),
    'rotary': dict(steps=180, bounds=180, rt=7648680, el=15016960, eng=1055880),
    'swin': dict(steps=88, bounds=88, rt=17778585, el=61597440, eng=3918831),
    'bias': dict(steps=0, bounds=0, rt=0, el=0, eng=0),
    'im2col': dict(steps=30, bounds=29, rt=935762, el=2003383, eng=140460),
    'deform': dict(steps=6, bounds=6, rt=4585470, el=10444800, eng=673200),
}

out_rows = []
for kind, plan, lut, bram in spec:
    p = per_op_numbers[kind]
    net = p['rt'] - p['eng']
    ratio = net / lut * 1000 if lut else 0
    out_rows.append(dict(
        op=kind, plan=plan, host_steps=p['steps'], boundaries=p['bounds'],
        roundtrip_cycles=round(p['rt']), tensor_elements=round(p['el']),
        engine_cycles=round(p['eng']), net_save_cycles=round(net),
        net_save_ms=round(net / F * 1000, 1),
        lut_est=lut, bram36_est=bram,
        net_cycles_per_klut=round(ratio)))

with open(os.path.join(HERE, 'benefit_estimation.csv'), 'w', newline='',
          encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)

json.dump({
    'meta': {
        'date': '2026-08-31',
        'baseline_cycles_pf1': TOTAL_CYC,
        'baseline_ms_at_198p5': round(base_ms, 1),
        'mac_util_baseline_pct': 7.81,
        'cost_model': 'gem_cycles.py RTL 实测常数（STORE 3.2B/cyc、LOAD_CTX 8B/cyc、'
                      'COPY 2+3*k*组、SM16 3n+45/16 行组）',
        'attribution': '边界往返 = 前段全部 STORE + 后段全部 LOAD_CTX（理想模型拍）；'
                       '共边界多算子均摊',
        'engine_models': 'AE_ACTV 16-lane：norm 2 遍 + 45 拍/行组；eltwise/rotary/scatter '
                         '1 遍 + 8 拍/行组；softmax = bias 遍 + SM16 三遍（S 元素取自 '
                         'host_plan attn 几何）',
    },
    'per_op': out_rows,
    'subsets': {
        name: {k: round(v) if isinstance(v, float) else v for k, v in r.items()}
        for name, r in res.items()
    },
    'totals': {
        'lut_total_est': 16300,
        'lut_with_50pct_margin': 24500,
        'lut_budget': 60000,
        'bram36_total_est': 8,
        'bram36_free': 189,
        'dsp_new': 0,
    },
}, open(os.path.join(HERE, 'benefit_estimation.json'), 'w'), ensure_ascii=False,
    indent=1)

print('\n逐算子（分摊口径）:')
for r in out_rows:
    print(f"  {r['op']:8s} 净省 {r['net_save_cycles']/1e6:6.2f}M 拍 "
          f"({r['net_save_ms']:6.1f} ms)  LUT {r['lut_est']:>5d}  "
          f"性价比 {r['net_cycles_per_klut']/1e6:>5.1f}M 拍/kLUT")
print('\n写出 benefit_estimation.json / benefit_estimation.csv')
