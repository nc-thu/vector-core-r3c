# -*- coding: utf-8 -*-
"""norm_real_gate.py — NORM 引擎真实张量数值门（2026-08-31）

问题：微观位精确门已经证明 RTL == 引擎定点语义，但还没证明这个定点语义
在【真实 γ/β、真实校准尺度、真实激活分布】上离 host fp32 路径足够近。

口径（与 09_cbound/gate_rtl.py 的既有验证口径一致）：
  1. 站点 = 模型里一个真实 LayerNorm/RMSNorm 模块（γ/β 取自 HF checkpoint）
     × a2 流上一张真实的 act_out 图（宽度 = γ 长度，字节 = 段实跑产出，
     sa_in = 该图的 so，即生产段的真实输出尺度）。
  2. sa_out = max(下游第一个 gemm 消费者的 sa, z_max·γ_max/512)——
     下限是引擎 w 域可表示性（w 粒度 Δy ≈ G·2^-(gs+1)，G = γ/sa_out），
     γ 动态范围大的站点靠抬 sa_out 保精度，下游 gemm 的 m_requant 吸收
     尺度差；z_max = 1.05×站点实测 max|z|（真实校准的取法）。
  3. 段实跑：权重 = 真实 blob，段输入 = 确定性随机 int8（gate_rtl.py 同款
     rng_for），解释器 = fast_interp 的逐位拷贝，仅把 GEMM 累加换成 fp64
     ——int8 乘加的每个部分积与求和都 < 2^53，fp64 整数算术精确，先在
     真实段上与原版逐字节对拍证明等价。
  4. 对拍：host fp32 路径（反量化→fp32 LN/RMS→按 sa_out 重量化）vs
     norm_gold.build_image + 引擎定点语义（norm_gold.engine_row 的向量化
     拷贝，先逐行对拍证明等价）。门：每站点 max|Δ| ≤ 2 LSB 且 |mean| < 0.2 LSB。

只读引用 09_cbound / 02_quant；产物写到本目录（norm_real_gate_result.json）。
"""
import collections
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CB = os.path.abspath(os.path.join(HERE, '..', '..', '09_cbound'))
sys.path.insert(0, CB)
sys.path.insert(0, os.path.join(HERE, '..', 'spec'))

from fast_interp import run_segment_fast, _softmax_rows_fast   # noqa: E402
from golden_interp import decode, sat8, _s8, load_seq          # noqa: E402
from norm_gold import build_image, engine_row, RSQRT_LUT       # noqa: E402

BUILD = os.path.join(CB, 'build_a2')
SEGDIR = os.path.join(BUILD, 'segments')
CAL = json.load(open(os.path.join(CB, '..', '02_quant',
                                  'hw_calib_table.json'), encoding='utf-8'))['gemms']
EPS_LN = 1e-5
EPS_RMS = float(np.finfo(np.float32).eps)


# ---------------------------------------------------------------------------
# fp64 段解释器：fast_interp.run_segment_fast 的逐字拷贝，只改 GEMM 累加一行
# （A@B 换 fp64 精确乘加；|acc| ≤ 127·127·5120 < 2^27 ≪ 2^53，无舍入）
# ---------------------------------------------------------------------------
def run_segment_fast64(seq, ddr_img, P):
    cols, ctxw, ww = P['COLS'], P['CTX_WORDS'], P['W_WORDS']
    ddr = ddr_img.copy()
    ctx = np.zeros((16, ctxw), dtype=np.int64)
    wram = np.zeros((cols, ww), dtype=np.int64)
    for d in seq:
        f = decode(d)
        op = f['op']
        if op == 15:
            break
        m, n, k = f['m'], f['n'], f['k']
        if op == 4:                                    # LOAD
            nB = f['dma_len']
            B = _s8(ddr[f['dma_addr']:f['dma_addr'] + nB]).astype(np.int64)
            if f['b_src'] == 0:                        # CTX k-major
                base = f['b_base']
                fw = nB // 16
                if fw:
                    ctx[:, base:base + fw] = B[:fw * 16].reshape(fw, 16).T
                rem = nB - fw * 16
                if rem:
                    ctx[:rem, base + fw] = B[fw * 16:]
            else:                                      # W 每 k 行 COLS 字节
                base = f['b_base']
                nwd = nB // cols
                if nwd:
                    wram[:, base:base + nwd] = B[:nwd * cols].reshape(nwd, cols).T
                rem = nB - nwd * cols
                if rem:
                    wram[:rem, base + nwd] = B[nwd * cols:]
        elif op == 5:                                  # STORE word-major
            W = f['dma_len'] // 16
            base = f['y_base']
            seg = np.ascontiguousarray(ctx[:, base:base + W].T)
            ddr[f['dma_addr']:f['dma_addr'] + W * 16] = \
                (seg & 0xFF).astype(np.uint8).reshape(-1)
        elif op == 3:                                  # COPY CTX→WRAM
            src_j0 = f['rq_m']
            nr = n & 0xFF
            rows = src_j0 + np.arange(nr)
            lanes = rows % 16
            words = f['b_base'] + (rows // 16) * f['b_spad']
            idx = words[:, None] + np.arange(k)[None, :]
            wram[:nr, f['a_base']:f['a_base'] + k] = ctx[lanes[:, None], idx]
        elif op in (0, 1, 2):                          # GEMM 族
            ai = np.arange(m)
            idxA = f['a_base'] + (ai // 16)[:, None] * k + np.arange(k)[None, :]
            A = ctx[(ai % 16)[:, None], idxA]
            B = wram[:, f['b_base']:f['b_base'] + k].T[:, :f['b_spad']]
            acc = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.int64)
            Y = sat8(acc * np.int64(f['rq_m']) >> np.int64(f['rq_s']))
            m16 = ((m + 15) // 16) * 16
            nl = f['b_spad']
            if f['y_tr']:
                c = f['j0'] + np.arange(nl)
                valid = np.nonzero(c < n)[0]
                cc = c[valid]
                lanes = (cc % 16)[:, None]
                words = (f['y_base'] + (cc // 16) * m16)[:, None] + \
                    np.arange(m)[None, :]
                ctx[lanes, words] = Y[:, valid].T
            else:
                words = (f['y_base'] + (ai // 16) * n + f['j0'])[:, None] + \
                    np.arange(nl)[None, :]
                ctx[(ai % 16)[:, None], words] = Y
            if op == 1:                                # SM16 softmax
                idxS = (f['y_base'] + (ai // 16) * n)[:, None] + \
                    np.arange(n)[None, :]
                S = ctx[(ai % 16)[:, None], idxS]
                Pm = _softmax_rows_fast(S.astype(np.int8), n, f['sm_causal'])
                ctx[(ai % 16)[:, None], idxS] = Pm
        else:
            raise AssertionError(f'未定义 op={op}')
    return ctx, ddr, dict(macs=0)


def rng_for(key):
    """确定性随机源（'09cb' 命名空间，同 key 永远同数据）。gate_rtl.py 的
    原实现把 tuple 直接喂 sha256 会抛 TypeError，这里用 repr 字符串哈希——
    确定性语义相同，只是字节不必与 gate_rtl 历史产物一致。"""
    h = hashlib.sha256(repr(('09cb', *key) if isinstance(key, tuple)
                            else ('09cb', str(key))).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:4], 'little'))


def run_one_segment(seg):
    """段独立实跑（gate_rtl 口径：权重真值 + 输入确定性随机），返回终态 ddr。"""
    sd = os.path.join(SEGDIR, seg)
    man = json.load(open(os.path.join(sd, 'manifest.json'), encoding='utf-8'))
    P = man['profile']
    ddr = np.zeros(P['DDR_BYTES'], dtype=np.uint8)
    blob = np.fromfile(os.path.join(BUILD, 'weights_blob.bin'), dtype=np.uint8)
    for w in man['weights']:
        ddr[w['ddr']:w['ddr'] + w['blob_len']] = \
            blob[w['blob_off']:w['blob_off'] + w['blob_len']]
    by_name = {}
    for e in man['inputs']:
        n = e['words'] * 16
        by_name.setdefault(e['name'],
                           rng_for(('in', seg, e['name'])
                                   ).integers(0, 256, n, dtype=np.uint8))
        a = e['ddr']
        ddr[a:a + n] = by_name[e['name']]
    for e in man['outputs']:
        a, n = e['ddr'], e['words'] * 16
        ddr[a:a + n] = rng_for(('out', seg, a)).integers(0, 256, n, dtype=np.uint8)
    # 零槽保持 0（ddr 初值即 0）
    seq = load_seq(sd)
    _, gddr, _ = run_segment_fast64(seq, ddr, P)
    return gddr, man


def harvest_image(ddr, base, m, n):
    """STORE 'wm' 布局取整张图：x[r,c] = ddr[base + ((r//16)*n + c)*16 + r%16]。"""
    r = np.arange(m)
    idx = (base + ((r // 16) * n)[:, None] * 16) + \
        (np.arange(n)[None, :] * 16) + (r % 16)[:, None]
    return ddr[idx].astype(np.int8).astype(np.int64)


# ---------------------------------------------------------------------------
# 引擎定点语义（向量化）：norm_gold.engine_row 的 int64 拷贝，逐行位一致
# ---------------------------------------------------------------------------
def _rsqrt_q20_vec(v):
    """v int64 [m] -> invstat_q20 int64 [m]（LOD 偶化 → LUT → 一次牛顿）。"""
    v = np.maximum(v, 1 << 13)
    e0 = np.floor(np.log2(v.astype(np.float64))).astype(np.int64)
    # log2 浮点误差校正：保证 2^E ≤ v < 2^(E+1)（整数精确）
    fix = (np.int64(1) << e0) > v
    e0 = np.where(fix, e0 - 1, e0)
    fix2 = (np.int64(1) << (e0 + 1)) <= v
    e0 = np.where(fix2, e0 + 1, e0)
    E = e0 & np.int64(~1)
    f = E >> 1
    m_q11 = v >> np.maximum(E - 11, 0)
    r0 = np.array(RSQRT_LUT, dtype=np.int64)[(m_q11 - 2048) >> 4]
    r1 = (r0 * (np.int64(3) * (1 << 39) - m_q11 * r0 * r0)) >> 40
    r1 = np.clip(r1, -16384, 16383)                    # sat15
    shl = np.maximum(18 - f, 0)
    shr = np.maximum(f - 18, 0)
    sh = np.where(f <= 18, r1 << shl, r1 >> shr)
    inv = np.clip(sh, -(1 << 26), (1 << 26) - 1)       # sat27
    return np.maximum(inv, 1)


def engine_vec(xs, consts, g, b):
    """xs int64 [m,n]（有符号）；与 engine_row 逐位一致。返回 int64 [m,n]。"""
    invn = consts['invn']
    gs, os_ = consts['g_shift'], consts['out_shift']
    ln = consts['ln']
    g = np.asarray(g, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    s1 = xs.sum(axis=1)
    s2 = (xs * xs).sum(axis=1)
    mu = s1 * invn
    ms = s2 * invn
    if ln:
        var = np.maximum(0, ms - ((mu * mu) >> 24))
    else:
        var = ms
        mu = np.zeros_like(mu)
    v = np.maximum(var + consts['eps_q24'], 1 << 13)
    inv = _rsqrt_q20_vec(v)
    S = 44 - gs
    u = (xs << 24) - mu[:, None]
    prod = u * inv[:, None]
    w = (prod + (1 << (S - 1))) >> S
    w = np.clip(w, -256, 255)                          # sat9
    t = (w * g[None, :] + 128) >> 8
    tb = t + b[None, :]
    if os_ > 0:
        y = (tb + (1 << (os_ - 1))) >> os_
    else:
        y = tb
    return np.clip(y, -128, 127)


def fp32_ref_vec(xs, gamma, beta, eps, sa_in, sa_out, ln):
    a = xs.astype(np.float64) * sa_in
    if ln:
        mu = a.mean(axis=1, keepdims=True)
        v = ((a - mu) ** 2).mean(axis=1, keepdims=True)
    else:
        mu, v = 0.0, (a ** 2).mean(axis=1, keepdims=True)
    y = (a - mu) / np.sqrt(v + eps) * gamma[None, :] + beta[None, :]
    return np.clip(np.round(y / sa_out), -128, 127)          # sat8 约定（含 −128）


def site_z_stats(x, eps, ln):
    """站点级 z 统计（fp64）：z = 归一化后的每元素值，返回 max|z|。"""
    a = x.astype(np.float64)
    if ln:
        mu = a.mean(axis=1, keepdims=True)
        v = ((a - mu) ** 2).mean(axis=1, keepdims=True)
        z = np.abs((a - mu) / np.sqrt(v + eps))
    else:
        z = np.abs(a / np.sqrt((a ** 2).mean(axis=1, keepdims=True) + eps))
    return float(z.max())


SA_W_FLOOR = 512.0   # sa_out 下限系数：z_max·γ_max/sa_out ≤ 512 保证 w 半 LSB ≤ 2（留 2x 余量后 ≤1）


# ---------------------------------------------------------------------------
# 自检 ①：fp64 解释器 vs 原版 fast_interp 逐字节（两个真实段）
# ---------------------------------------------------------------------------
def equiv_interp(segs):
    for seg in segs:
        sd = os.path.join(SEGDIR, seg)
        man = json.load(open(os.path.join(sd, 'manifest.json'), encoding='utf-8'))
        P = man['profile']
        seq = load_seq(sd)
        ddr0 = np.zeros(P['DDR_BYTES'], dtype=np.uint8)
        blob = np.fromfile(os.path.join(BUILD, 'weights_blob.bin'), dtype=np.uint8)
        for w in man['weights']:
            ddr0[w['ddr']:w['ddr'] + w['blob_len']] = \
                blob[w['blob_off']:w['blob_off'] + w['blob_len']]
        for e in man['inputs']:
            n = e['words'] * 16
            ddr0[e['ddr']:e['ddr'] + n] = rng_for(('in', seg, e['name'])
                                                  ).integers(0, 256, n, dtype=np.uint8)
        for e in man['outputs']:
            a, n = e['ddr'], e['words'] * 16
            ddr0[a:a + n] = rng_for(('out', seg, a)).integers(0, 256, n, dtype=np.uint8)
        _, d1, _ = run_segment_fast(seq, ddr0, P)
        _, d2, _ = run_segment_fast64(seq, ddr0, P)
        same = np.array_equal(d1, d2)
        print(f'  [equiv-interp] {seg}: fast_interp vs fp64 版 '
              f'{"逐字节一致" if same else "不一致!!"}（{len(d1)} B）')
        if not same:
            bad = np.nonzero(d1 != d2)[0][:5]
            print('    首几个差异偏移:', bad, d1[bad], d2[bad])
            sys.exit(1)


# ---------------------------------------------------------------------------
# 自检 ②：engine_vec vs norm_gold.engine_row 逐位（随机 + 真实 γ/β 角落）
# ---------------------------------------------------------------------------
def equiv_engine(cases):
    for name, xs, consts, g, b in cases:
        got = engine_vec(xs, consts, g, b)
        for i in range(xs.shape[0]):
            ref = engine_row([int(v) for v in xs[i]], consts['invn'],
                             consts['eps_q24'], consts['g_shift'],
                             consts['out_shift'], consts['ln'], g, b)
            ref = np.array([v - 256 if v > 127 else v for v in ref])
            if not np.array_equal(got[i], ref):
                k = int(np.nonzero(got[i] != ref)[0][0])
                print(f'  [equiv-eng] {name} 行{i} 列{k}: vec={got[i, k]} '
                      f'row={ref[k]}')
                sys.exit(1)
        print(f'  [equiv-eng] {name}: {xs.shape[0]} 行逐位一致')


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    npz = np.load(os.path.join(HERE, 'norm_weights.npz'))
    hp = json.load(open(os.path.join(BUILD, 'host_plan.json'), encoding='utf-8'))
    ops = json.load(open(os.path.join(CB, 'ops_trace.json'),
                         encoding='utf-8'))
    ops = ops['ops'] if isinstance(ops, dict) else ops

    # --- 站点归因：norm 模块 -> (cls, 宽度, 第一个 gemm 消费者) ---
    cons = collections.defaultdict(list)
    for j, o in enumerate(ops):
        for t in o.get('in_ids', []):
            cons[t].append(j)
    cls_of = {s['module']: s['cls'] for s in hp['host_steps']
              if s.get('kind') == 'norm'}
    modinfo = {}
    for i, o in enumerate(ops):
        if o.get('op') != 'elem_norm' or o.get('cls') not in ('LayerNorm', 'RMSNorm'):
            continue
        if o['module'] in modinfo:
            continue
        cg = [ops[c]['module'] for c in cons.get(o['out_ids'][0], [])
              if c != i and ops[c].get('op') == 'gemm']
        if not cg:
            continue
        modinfo[o['module']] = cg[0]
    sites = []
    for mod, cmod in modinfo.items():
        wk = mod + '.weight'
        if wk not in npz or mod not in cls_of:
            continue
        if cls_of[mod] not in ('LayerNorm', 'RMSNorm') or cmod not in CAL:
            continue
        sites.append(dict(module=mod, cls=cls_of[mod], n=len(npz[wk]),
                          cons=cmod))
    print(f'[gate] 可用 norm 站点（γ+消费者+cal 齐全）：{len(sites)}')

    # --- 段图像索引（缓存 img_index.json，避免重复扫 2762 个 manifest）---
    cache = os.path.join(HERE, 'img_index.json')
    if os.path.exists(cache):
        idx = json.load(open(cache, encoding='utf-8'))
    else:
        idx = {}
        for seg in sorted(os.listdir(SEGDIR)):
            man = json.load(open(os.path.join(SEGDIR, seg, 'manifest.json'),
                                 encoding='utf-8'))
            for e in man.get('outputs', []):
                if e.get('kind') != 'act_out':
                    continue
                idx.setdefault(str(e['n']), []).append(
                    [seg, man.get('est_cycles', 0), e['m'], e['ddr'], e['so']])
        json.dump(idx, open(cache, 'w'))
    print(f'[gate] 图像索引：{len(idx)} 种宽度（缓存 {os.path.basename(cache)}）')

    # --- 每站点选图：同宽度里优先 m≥256 且 est_cycles 最小的段 ---
    chosen = {}
    seg_of = {}
    for s in sites:
        cands = idx.get(str(s['n']), [])
        if not cands:
            s['img'] = None
            continue
        good = [c for c in cands if c[2] >= 256] or cands
        c = min(good, key=lambda c: c[1])
        s['img'] = c                       # [seg, est, m, ddr, so]
        seg_of.setdefault(c[0], []).append(s)
    live = [s for s in sites if s.get('img')]
    print(f'[gate] 配到图的站点：{len(live)}，唯一段 {len(seg_of)} 个')
    assert len(live) >= 50, '站点数不足 50'

    # --- 自检：fp64 解释器 & 向量化引擎 ---
    print('[gate] 自检① fp64 段解释器 vs fast_interp 原版（真实段逐字节）')
    equiv_interp(sorted(seg_of)[:2])
    print('[gate] 自检② engine_vec vs engine_row（逐位）')
    rng = np.random.default_rng(7)
    ecases = []
    for n, ln in [(96, True), (256, False), (3072, True), (1, True)]:
        xs = rng.integers(-64, 64, size=(6, n))
        gamma = rng.normal(1.0, 0.3, size=n)
        beta = rng.normal(0.0, 0.2, size=n)
        im = build_image(n, gamma, beta, 1e-5 if ln else 1.19e-7,
                         0.0146, 0.030, ln=ln)
        ecases.append((f'n={n} {"LN" if ln else "RMS"}', xs, im['consts'],
                       im['g'], im['b']))
    xs = np.full((6, 96), 37, dtype=np.int64)
    im = build_image(96, np.ones(96), np.zeros(96), 1e-5, 0.0146, 0.030, ln=True)
    ecases.append(('const 行', xs, im['consts'], im['g'], im['b']))
    equiv_engine(ecases)
    print(f'[gate] 自检完成（{time.time() - t0:.1f}s）')

    # --- 段实跑 + 逐站点对拍 ---
    results = []
    for k, (seg, ss) in enumerate(sorted(seg_of.items(), key=lambda kv: kv[0])):
        t1 = time.time()
        ddr, man = run_one_segment(seg)
        print(f'[gate] 段 {seg} 实跑完成（{time.time() - t1:.1f}s，'
              f'{len(ss)} 站点用图）')
        for s in ss:
            _, _, m, ddr_base, so = s['img']
            n = s['n']
            x = harvest_image(ddr, ddr_base, m, n)
            gamma = npz[s['module'] + '.weight'].astype(np.float64)
            bk = s['module'] + '.bias'
            beta = (npz[bk].astype(np.float64) if bk in npz
                    else np.zeros(n))
            eps = EPS_LN if s['cls'] == 'LayerNorm' else EPS_RMS
            sa_in = float(so)
            ln = (s['cls'] == 'LayerNorm')
            # 站点校准量（真实编译器按校准数据取，这里同一来源）：
            #   z_max = 1.05 × 站点实测 max|z|（spec：z_max 按校准分位取）
            #   sa_out = max(下游 gemm sa, z_max·γ_max/512)
            #     —— 下限是 w 域可表示性：w 粒度 Δy ≈ G·2^-(gs+1)，
            #        G_max = γ_max/sa_out ≤ 2^(gs+1) 需 z_max·G_max ≤ 512
            #        （gs=floor(log2(256/z_max)) 时 2^(gs+1) ≈ 512/z_max）。
            #        γ 动态范围过大的站点靠抬 sa_out 保精度，
            #        下游 gemm 的 m_requant 吸收尺度差。
            z_max = 1.05 * site_z_stats(x, eps, ln)
            sa_cons = float(CAL[s['cons']]['sa'])
            sa_out = max(sa_cons, z_max * float(np.abs(gamma).max()) / SA_W_FLOOR)
            im = build_image(n, gamma, beta, eps, sa_in, sa_out, ln=ln,
                             z_max=z_max)
            y_eng = engine_vec(x, im['consts'], im['g'], im['b'])
            y_ref = fp32_ref_vec(x, gamma, beta, eps, sa_in, sa_out, ln)
            d = y_eng - y_ref
            r = dict(module=s['module'], cls=s['cls'], n=n, m=m, seg=seg,
                     sa_in=sa_in, sa_out=sa_out, sa_out_cons=sa_cons,
                     z_max=z_max, g_shift=im['g_shift'],
                     out_shift=im['out_shift'],
                     sa_floor_engaged=(sa_out > sa_cons * 1.0001),
                     max_abs=float(np.abs(d).max()),
                     mean_abs=float(np.abs(d).mean()),
                     mean=float(d.mean()))
            results.append(r)
            flag = 'PASS' if (r['max_abs'] <= 2.0 and abs(r['mean']) < 0.2) else 'FAIL'
            print(f"  {s['cls']:<9} n={n:<5} m={m:<6} {s['module'][:44]:<44} "
                  f"max|Δ|={r['max_abs']:.2f} mean|Δ|={r['mean_abs']:.3f} "
                  f"meanΔ={r['mean']:+.3f} {flag}")

    ok = sum(1 for r in results if r['max_abs'] <= 2.0 and abs(r['mean']) < 0.2)
    mx = max(results, key=lambda r: r['max_abs'])
    mm = max(results, key=lambda r: abs(r['mean']))
    print(f'\n[gate] 真实张量数值门：{ok}/{len(results)} 站点通过；'
          f'最差 max|Δ|={mx["max_abs"]:.2f}（{mx["module"]}），'
          f'最差 |meanΔ|={abs(mm["mean"]):.3f}（{mm["module"]}）')
    print(f'[gate] cls 覆盖：{dict(collections.Counter(r["cls"] for r in results))}')
    print(f'[gate] 总耗时 {time.time() - t0:.1f}s')
    json.dump(dict(ok=ok, total=len(results), results=results,
                   worst_max=mx, worst_mean=mm,
                   t=time.strftime('%Y-%m-%d %H:%M:%S')),
              open(os.path.join(HERE, 'norm_real_gate_result.json'), 'w'),
              indent=1, ensure_ascii=False)
    sys.exit(0 if ok == len(results) else 1)


if __name__ == '__main__':
    main()
