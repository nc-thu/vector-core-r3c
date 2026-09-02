# -*- coding: utf-8 -*-
"""route_study.py — 下一代架构路线研究（A 现行 / B packed-INT8 / softmax 并行化）
在 gem_cycles.py 的 RTL 实测常数上做情景推演（不改 RTL，纯账本外推）：
  A   = 现行 16×108（1 DSP = 1 INT8 MAC，峰值 343 GMAC/s @198.5 MHz）
  B   = packed INT8：同一批 1728 DSP，j 方向两路打包（偏置无符号 + 行程和校正，
        共享 A 广播，WRAM TDP 双读）→ 有效 16×216，峰值 686 GMAC/s
  SMx = softmax 引擎行并行 ×x（行间天然独立；v1 为逐元素串行）
情景可叠加（B+SM8 等）。DMA/COPY 常数不变（字节量相同，接口未动）。
生成：python route_study.py（控制台表，供 NEXT_ARCH.html 引用）
"""
import os
import sys

import gem_cycles as G

SIM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(SIM)),
                                'research_evo1_hif8'))
import evo1_spec as S   # noqa: E402

ROWS = 16
F_SYN = G.F_SYN


def gemm_cycles_w(m, k, n_loc, j0, y_tr, cols):
    """gem_cycles.gemm_cycles 的宽度参数化版（cols≠108 情景用）"""
    mt = (m + ROWS - 1) // ROWS
    tile = 1 + (k + 2) + (ROWS + cols + 3) + 16 + 2 \
        + G.wb_cycles(n_loc, j0, y_tr) + 1
    return 2 + G.GEMM_CMD_OVH + mt * tile, mt


def hw_gemm_w(m, n, k, cols, act=False, store=False):
    """宽度参数化的逐列组 GEMM 账（W 装载字节数只随 (k,总 n) 变，与 cols 无关）"""
    gemm = w_load = compute = 0
    mt = (m + ROWS - 1) // ROWS
    n_rem = n
    while n_rem > 0:
        n_loc = min(cols, n_rem)
        w_load += G.load_w_ideal(k, cols)
        g, _ = gemm_cycles_w(m, k, n_loc, 0, 0, cols)
        gemm += g
        compute += 4 + mt * (k + 2)
        n_rem -= n_loc
    dma = w_load
    if act:
        dma += G.load_ctx_ideal(m * k)
    if store:
        dma += G.store_cycles(m * n)
    return gemm, dma, compute


def hw_attn_w(lq, lk, dhead, heads, causal, cols, sm_par):
    """每头：COPY(Kᵀ) + QKᵀ + softmax(÷sm_par) + COPY(Vᵀ) + PV；COPY 与阵列宽无关"""
    cp_qk = gemm_qk = compute = 0
    n_rem, j0 = lk, 0
    while n_rem > 0:
        n_loc = min(cols, n_rem)
        cp_qk += G.copy_cycles(dhead, n_loc, j0)
        g, _ = gemm_cycles_w(lq, dhead, n_loc, j0, 0, cols)
        gemm_qk += g
        compute += 4 + ((lq + ROWS - 1) // ROWS) * (dhead + 2)
        n_rem -= n_loc
        j0 += n_loc
    sm = G.softmax_cycles(lq, lk, causal) / sm_par
    cp_pv = G.copy_cycles(lk, dhead, 0)
    g_pv, _ = gemm_cycles_w(lq, lk, dhead, 0, 0, cols)
    compute += 4 + ((lq + ROWS - 1) // ROWS) * (lk + 2)
    per = dict(copy=cp_qk + cp_pv, gemm=gemm_qk + g_pv, sm=sm, compute=compute)
    return {kk: v * heads for kk, v in per.items()}


BOUNDARY_ACT = {'patch_embed(14x14x3 conv)', 'state_encoder'}
BOUNDARY_STORE = {'mlp_head(896->1024->1200)'}


def account(spec, cols, sm_par, dead=0):
    tot = dict(gemm=0, dma=0, copy=0, sm=0, macs=0, compute=0)
    for it in spec['gemm_items']:
        c = spec['config']
        mul = c['denoise_steps'] if it['stage'] == 'ActionHead/step' else 1
        if it['kind'] == 'gemm':
            g, d, comp = hw_gemm_w(it['m'], it['n'], it['k'], cols,
                                   act=it['name'] in BOUNDARY_ACT,
                                   store=it['name'] in BOUNDARY_STORE)
            tot['gemm'] += g * it['count'] * mul
            tot['dma'] += d * it['count'] * mul
            tot['compute'] += comp * it['count'] * mul
            tot['macs'] += it['m'] * it['n'] * it['k'] * it['count'] * mul
        else:
            if 'ViT' in it['stage']:
                lq = lk = S.VIT_SEQ
                dh, hd, ca = S.VIT_H // S.VIT_HEADS, S.VIT_HEADS, False
            elif 'LLM' in it['stage']:
                lq = lk = S.LLM_SEQ
                dh, hd, ca = S.LLM_HEAD_DIM, S.LLM_HEADS, S.LLM_CAUSAL
            else:
                lq, hd = S.HEAD_HORIZON, S.HEAD_HEADS
                lk, dh, ca = S.CTX_TOKENS, S.HEAD_E // S.HEAD_HEADS, False
            h = hw_attn_w(lq, lk, dh, hd, ca, cols, sm_par)
            for key in ('gemm', 'copy', 'sm', 'compute'):
                tot[key] += h[key] * it['count'] * mul
            tot['macs'] += it['macs'] * mul
    tot['dma'] += dead
    return tot


def main():
    ref = S.build_spec(vit_tiles=2, hoist_kv=False)
    opt = S.build_spec(vit_tiles=2, hoist_kv=True)
    dead = 8 * hw_gemm_w(1, 2 * S.HEAD_E, S.HEAD_E, 108)[1] * (S.DENOISE_STEPS - 1)

    scen = [
        ('A  现行 16×108（343 峰值）',          108, 1),
        ('B  packed INT8 16×216（686 峰值）',   216, 1),
        ('SM8 softmax ×8（只动 softmax）',      108, 8),
        ('B+SM8',                              216, 8),
        ('B+SM8+LW 装载完美隐藏（上界）',       216, 8),
    ]
    print(f"{'情景':<34}{'REF M cyc':>11}{'PRIM M cyc':>11}{'s/inf':>8}"
          f"{'有效GMAC/s':>11}{'softmax占':>10}")
    base = None
    for name, cols, smp in scen:
        tr = account(ref, cols, smp)
        tp = account(opt, cols, smp, dead)
        tot_r = tr['gemm'] + tr['dma'] + tr['copy'] + tr['sm']
        tot_p = tp['gemm'] + tp['dma'] + tp['copy'] + tp['sm']
        if name.startswith('B+SM8+LW'):
            # W 装载完美隐藏情景：GEMM 段只计 compute（喂料下限），DMA 仅剩边界流量
            tot_r = tr['sm'] + tr['copy'] + tr['compute']
            tot_p = tp['sm'] + tp['copy'] + tp['compute']
        if base is None:
            base = tot_r
        eff = tp['macs'] / tot_p * F_SYN / 1e9
        print(f"{name:<34}{tot_r/1e6:>11.0f}{tot_p/1e6:>11.0f}"
              f"{tot_p/F_SYN:>8.2f}{eff:>11.1f}{tp['sm']/tot_p*100:>9.0f}%")

    # A 情景 GEMM 引擎内部拆分（REF 口径，供瓶颈堆叠图）
    tr = account(ref, 108, 1)
    feed = tr['compute']
    gemm_eng = tr['gemm']
    print(f"\n[A·REF 引擎拆分] softmax {tr['sm']/1e6:.0f}M | GEMM 引擎 "
          f"{gemm_eng/1e6:.0f}M（喂料 {feed/1e6:.0f}M + 每tile开销 "
          f"{(gemm_eng-feed)/1e6:.0f}M）| DMA(权重+边界) {tr['dma']/1e6:.0f}M | "
          f"COPY {tr['copy']/1e6:.0f}M")
    tb = account(ref, 216, 1)
    print(f"[B·REF 引擎拆分] softmax {tb['sm']/1e6:.0f}M | GEMM 引擎 "
          f"{tb['gemm']/1e6:.0f}M | DMA {tb['dma']/1e6:.0f}M | "
          f"COPY {tb['copy']/1e6:.0f}M")
    t8 = account(ref, 108, 8)
    print(f"[SM8·REF 拆分]  softmax {t8['sm']/1e6:.0f}M | GEMM 引擎 "
          f"{t8['gemm']/1e6:.0f}M | DMA {t8['dma']/1e6:.0f}M | "
          f"COPY {t8['copy']/1e6:.0f}M")
    tp8 = account(opt, 216, 8, dead)
    print(f"\n[PRIM B+SM8 相对 A·PRIM 增益] "
          f"{(6126 - (tp8['gemm']+tp8['dma']+tp8['copy']+tp8['sm'])/1e6):.0f}M 周期"
          f"（A·PRIM 基准 6126M 取自 CYCLE_ACCOUNT.md）")


if __name__ == '__main__':
    main()
