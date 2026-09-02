# -*- coding: utf-8 -*-
"""analyze_boundaries.py — host 算子 → 段边界 → 搬运代价 账本（07_onchip_ops，2026-08-31）

口径
  输入  build_full_fixed（05_sim 测量用的同一套流）：
    - host_plan.json 的 1424 条 host_steps（after_seg = 该算子跑在 seg_N 之后）
    - segments/seg_NNNN/seq.mem 逐条解码描述符（op/b_src/m/n/k/dma_len/j0/y_tr/rq_*）
    - 05_sim/types.json + cycles_by_type.json + results/sweep.json：每段实测拍数
      （MODE=1+PF=1 部署口径）与 gemm/dma 引擎分账
  成本模型常数 = 01_rtl/sim/gem_cycles.py（RTL 实测标定）：
    LOAD_CTX 8 B/cyc、LOAD_W 7.71 B/cyc、STORE 3.2 B/cyc、
    COPY = 2 + 3*k_rows*src组数、GEMM/SM16 见函数
  归因
    一个 host 算子造成的搬运 = 它所在边界的「前段全部 STORE + 后段全部 LOAD_CTX」
    （GEMM 结果必须落 DDR 给 host 读，host 写回的激活必须再装进 CTX）。
    同一边界挂多个 host 算子时费用均摊（norm+actv 常连排）。
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, '03_compiler', 'build_full_fixed')
SIM = os.path.join(ROOT, '05_sim')

# ---- gem_cycles.py 常数（RTL 实测） ----
ROWS = 16
BURST_B = 2048
AR_OVH = 2
CMD_OVH = 5
GEMM_CMD_OVH = 2


def waitd(cols):
    return ROWS + cols + 3


def wb_cycles(n_loc, j0, y_tr):
    if not y_tr:
        return n_loc
    grps = ((j0 + n_loc - 1) >> 4) - (j0 >> 4) + 1
    return 16 * grps


RQ_SH = 4
DRAIN = ROWS * RQ_SH
DALIGN = 2


def gemm_cycles(m, k, n_loc, j0, y_tr, cols=108):
    mt = (m + ROWS - 1) // ROWS
    tile = (1 + (k + 2) + waitd(cols) + DRAIN + DALIGN + 2
            + wb_cycles(n_loc, j0, y_tr) + 1)
    return 2 + GEMM_CMD_OVH + mt * tile


def load_ctx_ideal(nbytes):
    return nbytes // 8 + math.ceil(nbytes / BURST_B) * AR_OVH + CMD_OVH


def store_cycles(nbytes):
    return ((nbytes + 15) // 16) * 5 + CMD_OVH


def w_load_cycles(nbytes, cols=108):
    beats = nbytes // 8
    xings = 0
    wj = 0
    # w_load_profile 的解析式：每 cols 字节跨界一次（8B 粒度推进）
    per_row = cols // 8 * 8  # cols 是 8 的倍数时每行 8B 步数
    # 逐拍模拟太慢，用等价闭式：xings = floor 次数（wj+8 > cols）
    xings = (nbytes // 8) // (cols // 8) if cols % 8 == 0 else 0
    bursts = math.ceil(nbytes / BURST_B)
    return beats + xings + bursts * AR_OVH + CMD_OVH


def copy_cycles(k_rows, j_cols, src_j0):
    grps = ((src_j0 + j_cols - 1) >> 4) - (src_j0 >> 4) + 1
    return 2 + 3 * k_rows * grps


def softmax_cycles(m_rows, n_cols, causal):
    tot = 2
    for i in range(m_rows):
        vlen = min(i + 1, n_cols) if causal else n_cols
        tot += 2 * vlen + 2 * n_cols + 42
    return tot


def decode(d):
    return dict(
        op=(d >> 252) & 0xF, b_src=(d >> 246) & 7, causal=(d >> 245) & 1,
        y_tr=(d >> 244) & 1, m=(d >> 228) & 0xFFFF, n=(d >> 212) & 0xFFFF,
        k=(d >> 196) & 0xFFFF, dma_len=(d >> 61) & 0x3FFFF,
        rq_m=(d >> 104) & 0xFFFF, j0=(d >> 62) & 0xFFFF)


# ---------------- 1. 逐段解码 ----------------
N_SEG = 2782
seg_stat = []
for i in range(N_SEG):
    words = []
    with open(os.path.join(BUILD, 'segments', f'seg_{i:04d}', 'seq.mem')) as f:
        for line in f:
            line = line.strip()
            if line:
                words.append(int(line, 16))
    st = Counter()
    b = Counter()  # bytes
    cyc = Counter()
    for w in words:
        f = decode(w)
        op = f['op']
        if op == 15:
            break
        if op == 4:
            if f['b_src'] == 0:
                st['load_ctx'] += 1
                b['load_ctx'] += f['dma_len']
                cyc['load_ctx'] += load_ctx_ideal(f['dma_len'])
            else:
                st['load_w'] += 1
                b['load_w'] += f['dma_len']
                cyc['load_w'] += w_load_cycles(f['dma_len'])
        elif op == 5:
            st['store'] += 1
            b['store'] += f['dma_len']
            cyc['store'] += store_cycles(f['dma_len'])
        elif op == 3:
            st['copy'] += 1
            cyc['copy'] += copy_cycles(f['k'], f['n'] & 0xFF, f['rq_m'])
        elif op in (0, 2):
            st['gemm'] += 1
            cyc['gemm'] += gemm_cycles(f['m'], f['k'], f['n'], f['j0'], f['y_tr'])
        elif op == 1:
            st['gemm'] += 1
            cyc['gemm'] += gemm_cycles(f['m'], f['k'], f['n'], f['j0'], f['y_tr'])
            cyc['softmax'] += softmax_cycles(f['m'], f['n'], f['causal'])
    cyc['descs'] = len(words)
    seg_stat.append({'n': st, 'b': b, 'c': cyc})

# ---------------- 2. 实测拍数（types → cycles_by_type） ----------------
types = json.load(open(os.path.join(SIM, 'types.json')))
cbt = json.load(open(os.path.join(SIM, 'cycles_by_type.json')))
sweep = {r['type_id']: r for r in json.load(open(os.path.join(SIM, 'results', 'sweep.json')))}
seg_cycles = {}
for t in types['types']:
    tid = t['type_id']
    c = cbt[tid]['cycles']['pf1']
    g = sweep[tid]['runs'][1]['gemm']
    d = sweep[tid]['runs'][1]['dma']
    for s in t['instances']:
        seg_cycles[s] = {'cyc': c, 'gemm': g, 'dma': d}

tot_meas = sum(v['cyc'] for v in seg_cycles.values())
assert len(seg_cycles) == N_SEG

# ---------------- 3. host 步骤 → 边界归因 ----------------
hp = json.load(open(os.path.join(BUILD, 'host_plan.json')))
host_steps = hp['host_steps']

# 每个边界的 host 步骤集合
bounds = defaultdict(list)
for h in host_steps:
    bounds[h['after_seg']].append(h)

# 每段所属边界归属（用于把段内 COPY/段级开销归到"接下来这个 host 算子"）
seg2bound = {}
cur = None
for i in range(N_SEG - 1, -1, -1):
    if i in bounds:
        cur = i
    seg2bound[i] = cur  # 段 i 的产出流向边界 after_seg=i 的 host 算子


def bound_cost(pre, post):
    """边界搬运代价 = 前段 STORE + 后段 LOAD_CTX（理想模型拍数）"""
    c = Counter()
    n = Counter()
    byt = Counter()
    if pre is not None and 0 <= pre < N_SEG:
        c['store'] = seg_stat[pre]['c']['store']
        n['store'] = seg_stat[pre]['n']['store']
        byt['store'] = seg_stat[pre]['b']['store']
    if post is not None and post < N_SEG:
        c['load_ctx'] = seg_stat[post]['c']['load_ctx']
        n['load_ctx'] = seg_stat[post]['n']['load_ctx']
        byt['load_ctx'] = seg_stat[post]['b']['load_ctx']
    return c, n, byt


KIND_ORDER = ['norm', 'actv', 'softmax', 'softmax_prep', 'rotary', 'swin_partition',
              'swin_split', 'swin_reverse', 'im2col', 'deform_host', 'host_gemm',
              'exempt', 'other']

agg = {k: {'steps': 0, 'bounds': set(), 'cyc': Counter(), 'n': Counter(),
           'b': Counter(), 'cls': Counter(), 'mods': Counter()}
       for k in KIND_ORDER}
agg_by_cls = defaultdict(lambda: {'steps': 0, 'cyc': Counter(), 'n': Counter(),
                                  'b': Counter(), 'bounds': set()})

for aftar, hs in sorted(bounds.items()):
    c, n, byt = bound_cost(aftar, aftar + 1)
    for h in hs:
        k = h['kind'] if h['kind'] in agg else 'other'
        w = 1.0 / len(hs)
        a = agg[k]
        a['steps'] += 1
        a['bounds'].add(aftar)
        for kk in c:
            a['cyc'][kk] += c[kk] * w
            a['n'][kk] += n[kk] * w
            a['b'][kk] += byt[kk] * w
        a['cls'][h['cls']] += 1
        cl = f"{h['kind']}:{h['cls']}"
        ac = agg_by_cls[cl]
        ac['steps'] += 1
        ac['bounds'].add(aftar)
        for kk in c:
            ac['cyc'][kk] += c[kk] * w
            ac['n'][kk] += n[kk] * w
            ac['b'][kk] += byt[kk] * w

# 段级 COPY 归因：段 i 的 COPY 拍数记到 seg2bound[i] 的 host 算子
for i in range(N_SEG):
    bd = seg2bound[i]
    if bd is None:
        continue
    hs = bounds[bd]
    w = 1.0 / len(hs)
    cp = seg_stat[i]['c']['copy']
    for h in hs:
        k = h['kind'] if h['kind'] in agg else 'other'
        agg[k]['cyc']['copy_assoc'] += cp * w
        cl = f"{h['kind']}:{h['cls']}"
        agg_by_cls[cl]['cyc']['copy_assoc'] += cp * w

# ---------------- 4. 汇总输出 ----------------
print('=' * 100)
print(f"总段数 {N_SEG}  host 步骤 {len(host_steps)}  边界数(有 host 步骤的 after_seg) {len(bounds)}")
print(f"实测总拍(pf1) {tot_meas:,}  = {tot_meas/198.5e6*1000:.1f} ms @198.5MHz")

tot_n = Counter()
tot_b = Counter()
tot_c = Counter()
for s in seg_stat:
    tot_n += s['n']
    tot_b += s['b']
    tot_c += s['c']
print('\n全模型描述符统计（核对任务口径）')
for k in ('load_ctx', 'load_w', 'store', 'copy', 'gemm'):
    print(f"  {k:9s} 条数 {tot_n[k]:>7,}  字节 {tot_b[k]:>13,}  理想拍 {tot_c[k]:>13,}")
print(f"  描述符总数 {tot_c['descs']:,}")

ideal_sum = sum(v for k, v in tot_c.items() if k != 'descs')
print(f"\n理想模型合计 {ideal_sum:,} 拍 vs 实测 {tot_meas:,}（比值 {tot_meas/ideal_sum:.3f}，"
      f"差额=LFSR 停顿+仲裁+调度器拍）")

# 各组件占实测比例
print('\n组件理想拍 / 占实测总拍比例')
for k in ('gemm', 'load_ctx', 'load_w', 'store', 'copy', 'softmax'):
    print(f"  {k:9s} {tot_c[k]:>13,}  {tot_c[k]/tot_meas*100:5.1f}%")

print('\n===== 按 host 算子类别的边界搬运账（理想拍，多算子共边界均摊）=====')
hdr = f"{'kind':16s}{'steps':>7s}{'bounds':>8s}{'STORE拍':>12s}{'LDCTX拍':>12s}{'合计拍':>12s}{'COPY关联拍':>12s}{'STORE字节':>14s}{'LDCTX字节':>14s}"
print(hdr)
rows = []
for k in KIND_ORDER:
    a = agg[k]
    sc = a['cyc'].get('store', 0)
    lc = a['cyc'].get('load_ctx', 0)
    ca = a['cyc'].get('copy_assoc', 0)
    rows.append((k, a['steps'], len(a['bounds']), sc, lc, sc + lc, ca,
                 a['b'].get('store', 0), a['b'].get('load_ctx', 0)))
for r in sorted(rows, key=lambda r: -(r[3] + r[4])):
    print(f"{r[0]:16s}{r[1]:>7d}{r[2]:>8d}{r[3]:>12,.0f}{r[4]:>12,.0f}{r[5]:>12,.0f}"
          f"{r[6]:>12,.0f}{r[7]:>14,.0f}{r[8]:>14,.0f}")

print('\n===== 细分到 class（拍数降序 top 25）=====')
crows = []
for cl, a in agg_by_cls.items():
    sc = a['cyc'].get('store', 0)
    lc = a['cyc'].get('load_ctx', 0)
    ca = a['cyc'].get('copy_assoc', 0)
    crows.append((cl, a['steps'], len(a['bounds']), sc, lc, sc + lc, ca))
for r in sorted(crows, key=lambda r: -(r[3] + r[4]))[:25]:
    print(f"{r[0]:40s}{r[1]:>5d}{r[2]:>5d}{r[3]:>11,.0f}{r[4]:>11,.0f}{r[5]:>11,.0f}{r[6]:>11,.0f}")

# 边界处 host 算子连排统计
multi = Counter(len(v) for v in bounds.values())
print('\n每边界挂的 host 算子数分布:', dict(sorted(multi.items())))

out = {
    'total': {
        'segments': N_SEG,
        'host_steps': len(host_steps),
        'boundaries': len(bounds),
        'cycles_pf1': tot_meas,
        'desc_counts': {k: tot_n[k] for k in ('load_ctx', 'load_w', 'store', 'copy', 'gemm')},
        'bytes': {k: tot_b[k] for k in ('load_ctx', 'load_w', 'store')},
        'ideal_cycles': {k: tot_c[k] for k in ('gemm', 'load_ctx', 'load_w', 'store', 'copy', 'softmax')},
    },
    'by_kind': {
        r[0]: {'steps': r[1], 'boundaries': r[2], 'store_cyc': r[3], 'loadctx_cyc': r[4],
               'roundtrip_cyc': r[5], 'copy_assoc_cyc': r[6],
               'store_bytes': r[7], 'loadctx_bytes': r[8]}
        for r in rows
    },
    'by_class': {
        r[0]: {'steps': r[1], 'boundaries': r[2], 'store_cyc': r[3], 'loadctx_cyc': r[4],
               'roundtrip_cyc': r[5], 'copy_assoc_cyc': r[6]}
        for r in crows
    },
    'per_host_op_detail': [
        {'after_seg': a, 'kinds': [f"{h['kind']}:{h['cls']}" for h in hs]}
        for a, hs in sorted(bounds.items())
    ],
}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'boundary_account.json'), 'w') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print('\n写出 boundary_account.json')
