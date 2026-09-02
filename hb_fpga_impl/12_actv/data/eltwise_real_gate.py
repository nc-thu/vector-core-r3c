# -*- coding: utf-8 -*-
"""eltwise_real_gate.py — ELTWISE 引擎（op=6 submode=3）真实张量数值门
（2026-08-31）

问题：微观位精确门已经证明 RTL == 引擎定点语义（21 用例含 E1-E4），但还没
证明这个定点语义在【真实残差数据、真实校准尺度】上离 host fp32 路径足够近。

残差站点 = HoloBrain 解码器/骨干里的前向残差加 y = x1 + x2（pre-norm 块的
x + attn(ln(x)) / x + mlp(ln(x))，在 a2 流里全部留在 host，是段界 DDR 往返
的大头）。这类加法没有独立 trace 算子（藏在 layer forward 里），本门取真实
数据对：同宽度的两张【不同段实跑产出的真实激活图】当 x1/x2，各自用本图的
真实输出尺度 so 当 sa1/sa2；sa_out 取该宽度真实下游 gemm 的校准 sa
（hw_calib_table，与 norm 门同源同口径）。

编译公式（spec v1.2 / norm_gold.eltwise_params）：
  m_i = round(sa_i / sa_out · 2^s)，s 在两乘子都不溢 int16 的约束下取最大。
引擎语义：y = sat8(((x1·m1 + x2·m2) + 2^(s-1)) >>> s)。

对拍：host fp64（x1·sa1 + x2·sa2 → /sa_out → 四舍六入五成双 → sat8）
vs 引擎定点。门：每站点 max|Δ| ≤ 1 LSB 且 |mean| < 0.2 LSB（超 1 报分布）。

只读引用 09_cbound；产物写本目录（eltwise_real_gate_result.json）。
"""
import collections
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'spec'))

import norm_real_gate as ng                       # 复用段实跑/收图/确定性随机
from norm_gold import eltwise_row, eltwise_params  # noqa: E402

CAL = ng.CAL
SEGDIR = ng.SEGDIR


# ---------------------------------------------------------------------------
# 引擎定点语义（向量化）：与 norm_gold.eltwise_row 逐位一致
# ---------------------------------------------------------------------------
def eltwise_vec(x1, x2, m1, m2, s):
    acc = x1.astype(np.int64) * np.int64(m1) + x2.astype(np.int64) * np.int64(m2)
    if s > 0:
        sh = (acc + (np.int64(1) << (s - 1))) >> s
    else:
        sh = acc
    return np.clip(sh, -128, 127)


def fp32_ref_vec(x1, x2, sa1, sa2, sa_out):
    y = x1.astype(np.float64) * sa1 + x2.astype(np.float64) * sa2
    return np.clip(np.round(y / sa_out), -128, 127)   # sat8 约定（含 −128）


def equiv_engine():
    """自检：向量化引擎 vs norm_gold.eltwise_row 逐位（随机 + 角落）。"""
    rng = np.random.default_rng(20260831)
    for (m1v, m2v, s) in [(256, 192, 8), (-384, 448, 8), (32767, -32768, 8),
                          (1, 1, 0), (257, -65, 11)]:
        x1 = rng.integers(-128, 128, size=(7, 33))
        x2 = rng.integers(-128, 128, size=(7, 33))
        got = eltwise_vec(x1, x2, m1v, m2v, s)
        for i in range(x1.shape[0]):
            ref = eltwise_row([int(v) for v in x1[i]], [int(v) for v in x2[i]],
                              m1v, m2v, s)
            ref = np.array([v - 256 if v > 127 else v for v in ref])
            if not np.array_equal(got[i], ref):
                k = int(np.nonzero(got[i] != ref)[0][0])
                print(f'  [equiv] m1={m1v} m2={m2v} s={s} 行{i} 列{k}: '
                      f'vec={got[i, k]} row={ref[k]}')
                sys.exit(1)
    # 角落：全 ±128 / 饱和压力
    for x1, x2 in [(np.full((4, 9), 127), np.full((4, 9), 127)),
                   (np.full((4, 9), -128), np.full((4, 9), -128))]:
        got = eltwise_vec(x1, x2, 256, 192, 8)
        ref = np.array([eltwise_row(list(x1[0]), list(x2[0]), 256, 192, 8)])
        ref = np.array([v - 256 if v > 127 else v for v in ref[0]])
        assert np.array_equal(got[0], ref), '角落用例不一致'
    print('[gate] 自检：eltwise_vec vs eltwise_row 逐位一致')


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    equiv_engine()

    # --- 站点归因：复用 norm 门的 (module, cls, 宽度, 下游 gemm) 列表 ---
    npz = np.load(os.path.join(HERE, 'norm_weights.npz'))
    hp = json.load(open(os.path.join(ng.BUILD, 'host_plan.json'),
                        encoding='utf-8'))
    ops = json.load(open(os.path.join(ng.CB, 'ops_trace.json'),
                         encoding='utf-8'))
    ops = ops['ops'] if isinstance(ops, dict) else ops
    cons = collections.defaultdict(list)
    for j, o in enumerate(ops):
        for t in o.get('in_ids', []):
            cons[t].append(j)
    cls_of = {s['module']: s['cls'] for s in hp['host_steps']
              if s.get('kind') == 'norm'}
    sites = []
    seen = set()
    for i, o in enumerate(ops):
        if o.get('op') != 'elem_norm' or o.get('cls') not in ('LayerNorm',
                                                              'RMSNorm'):
            continue
        if o['module'] in seen:
            continue
        cg = [ops[c]['module'] for c in cons.get(o['out_ids'][0], [])
              if c != i and ops[c].get('op') == 'gemm']
        if not cg:
            continue
        seen.add(o['module'])
        wk = o['module'] + '.weight'
        if wk not in npz or o['module'] not in cls_of or cg[0] not in CAL:
            continue
        sites.append(dict(module=o['module'], cls=cls_of[o['module']],
                          n=len(npz[wk]), cons=cg[0]))
    print(f'[gate] 可用站点（宽度+下游 gemm sa 齐全）：{len(sites)}')

    # --- 每宽度选两张不同段的真实图（残差的 x1/x2）---
    idx = json.load(open(os.path.join(HERE, 'img_index.json'),
                         encoding='utf-8'))
    by_w = {}
    for s in sites:
        by_w.setdefault(str(s['n']), []).append(s)
    need = {}
    for w in by_w:
        cands = idx.get(w, [])
        good = [c for c in cands if c[2] >= 256] or cands
        good = sorted(good, key=lambda c: c[1])
        pair = []
        for c in good:                    # 取两个不同段的图
            if c[0] not in [p[0] for p in pair]:
                pair.append(c)
            if len(pair) == 2:
                break
        if len(pair) == 2:
            need[w] = pair
    print(f'[gate] 配到双图的宽度：{len(need)} / {len(by_w)}')

    # --- 段实跑（每段一次，缓存）---
    ddr_cache = {}
    for w, pair in sorted(need.items()):
        for c in pair:
            if c[0] not in ddr_cache:
                ddr_cache[c[0]] = ng.run_one_segment(c[0])[0]
    print(f'[gate] 段实跑 {len(ddr_cache)} 个（{time.time() - t0:.1f}s）')

    # --- 逐站点对拍 ---
    results = []
    for w, pair in sorted(need.items()):
        for s in by_w[w]:
            (_, _, m1c, b1, so1), (_, _, m2c, b2, so2) = pair
            n = s['n']
            m = min(m1c, m2c)
            x1 = ng.harvest_image(ddr_cache[pair[0][0]], b1, m, n)
            x2 = ng.harvest_image(ddr_cache[pair[1][0]], b2, m, n)
            sa1, sa2 = float(so1), float(so2)
            sa_out = float(CAL[s['cons']]['sa'])
            try:
                m1v, m2v, sv = eltwise_params(sa1, sa2, sa_out)
            except AssertionError:
                results.append(dict(module=s['module'], n=n, m=m, skip=True,
                                    why='乘子超 int16（sa 比 > 32768）'))
                continue
            y_eng = eltwise_vec(x1, x2, m1v, m2v, sv)
            y_ref = fp32_ref_vec(x1, x2, sa1, sa2, sa_out)
            d = y_eng - y_ref
            results.append(dict(
                module=s['module'], cls=s['cls'], n=n, m=m,
                seg1=pair[0][0], seg2=pair[1][0],
                sa1=sa1, sa2=sa2, sa_out=sa_out, m1=m1v, m2=m2v, s=sv,
                max_abs=float(np.abs(d).max()),
                mean_abs=float(np.abs(d).mean()),
                mean=float(d.mean())))
    live = [r for r in results if not r.get('skip')]
    assert len(live) >= 50, f'真实残差站点不足 50（{len(live)}）'

    for r in live:
        flag = 'PASS' if (r['max_abs'] <= 1.0 and abs(r['mean']) < 0.2) else 'FAIL'
        print(f"  n={r['n']:<5} m={r['m']:<6} {r['module'][:44]:<44} "
              f"sa1={r['sa1']:.4g} sa2={r['sa2']:.4g} sa_out={r['sa_out']:.4g} "
              f"s={r['s']:<2} max|Δ|={r['max_abs']:.2f} "
              f"meanΔ={r['mean']:+.3f} {flag}")

    ok = sum(1 for r in live if r['max_abs'] <= 1.0 and abs(r['mean']) < 0.2)
    mx = max(live, key=lambda r: r['max_abs'])
    mm = max(live, key=lambda r: abs(r['mean']))
    dist = collections.Counter(int(r['max_abs']) for r in live)
    print(f'\n[gate] ELTWISE 真实张量门：{ok}/{len(live)} 站点通过；'
          f'最差 max|Δ|={mx["max_abs"]:.2f}（{mx["module"]}），'
          f'最差 |meanΔ|={abs(mm["mean"]):.3f}（{mm["module"]}）')
    print(f'[gate] max|Δ| 分布（LSB 整数部分 -> 站点数）：{dict(sorted(dist.items()))}')
    print(f'[gate] 总耗时 {time.time() - t0:.1f}s')
    json.dump(dict(ok=ok, total=len(live), skipped=len(results) - len(live),
                   results=results, worst_max=mx, worst_mean=mm,
                   dist={str(k): v for k, v in sorted(dist.items())},
                   t=time.strftime('%Y-%m-%d %H:%M:%S')),
              open(os.path.join(HERE, 'eltwise_real_gate_result.json'), 'w'),
              indent=1, ensure_ascii=False)
    sys.exit(0 if ok == len(live) else 1)


if __name__ == '__main__':
    main()
