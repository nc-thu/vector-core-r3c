# -*- coding: utf-8 -*-
"""v2_perf.py — v2 选定路线的性能推演（先建模，不动 RTL）
路线拍板（2026-08-27）：softmax 行并行 ×16/×32 + HiF8-native datapath + DMA 延迟隐藏。

三个旋钮的建模口径：
  SMx   softmax ÷x（×16 = CTX 单读口上限：S 布局 16 lane 同地址、广播读全回；
        ×32 需双读口——本脚本照除，端口代价在页面里标注）
  H     HiF8-native：DSP→0，阵列 MAC 数 = LUT 预算 ÷ LUT/MAC（敏感性 35/45/60/80）。
        周期账沿用 gem_cycles 的宽度参数化 tile；requant 被 normalize/encode 替换
        （流水 1/op，无周期差）。面积模型：
          可用 LUT = 230400×0.88 − 固定 8528（dma/sched/SM16 softmax/ctx/top/仲裁）
          每列成本 = copy 176 + normalize 80 + gemm 控制 119 = 375 LUT/col
          阵列 = 16×cols×LUT_per_MAC
  LW    DMA 隐藏两档：
          ideal = sm+copy+compute（装载全消失的上界，与 route_study 口径一致）
          real  = max(串行段, W 装载流)——7.71 B/cyc 是 v1 引擎 64-bit AXI 主口
                  实测口径（非 DDR 能力：64-bit DDR4-2400 峰值 19.2 GB/s ≈
                  96.7 B/cyc@198.5MHz，墙在引擎通路宽度），
                  边界 act/store DMA 串行叠加；死装载按 v2 调度器可跳（=0）
对照：B packed INT8（DSP 打包 ×2，cols=216）同口径并跑。
自检：A 情景必须复现 CYCLE_ACCOUNT 的 6126M；LW-ideal 必须复现 route_study 的 1059M。
生成：python v2_perf.py
"""
import os
import sys

import gem_cycles as G

SIM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM)
import route_study as R          # noqa: E402
import evo1_spec as S            # noqa: E402

F = G.F_SYN

# ---------------- v2 面积模型：LUT 预算 → HiF8 阵列宽度 ----------------
LUT_DEV = 230400                 # xczu7ev
UTIL    = 0.88                   # LUT 占用上限（现行 70.7%，留布线余量）
FIX_LUT = 2395 + 1033 + 3500 + 260 + 119 + 1221   # dma+sched+SM16softmax+ctx+top+仲裁
PER_COL = 176 + 80 + 119         # copy/col + requant→normalize/col + gemm 控制/col
BRAM_CAP = 297                   # 312 − SEQ 14.5（WRAM cols 个 bank 的 BRAM 上限）


def hif8_cols(lut_mac):
    """给定 LUT/MAC，器件能容纳的 HiF8 阵列宽度（16 行 × cols）"""
    return int((LUT_DEV * UTIL - FIX_LUT) // (PER_COL + 16 * lut_mac))


# ---------------- 账本封装 ----------------
def account_v2(spec, cols, sm_par, dead, lw=None):
    """返回 (总周期, 明细)。lw=None 串行；'ideal' 完美隐藏；'real' 带宽地板口径"""
    t = R.account(spec, cols, sm_par, 0 if lw else dead)
    t['total'] = t['gemm'] + t['dma'] + t['copy'] + t['sm']
    if lw == 'ideal':
        t['total'] = t['sm'] + t['copy'] + t['compute']
    elif lw == 'real':
        w_tot = w_total_of(spec, cols)
        act_store = t['dma'] - w_tot          # 边界激活/回写（藏不掉，串行）
        ser = t['sm'] + t['copy'] + t['compute'] + act_store
        t['w_total'] = w_tot
        t['ser'] = ser
        t['total'] = max(ser, w_tot + act_store)
    return t


def w_total_of(spec, cols):
    """全部权重装载流（仅 gemm 类条目；attention 的 B 来自 COPY，不走 WRAM 装载）"""
    tot = 0
    for it in spec['gemm_items']:
        mul = spec['config']['denoise_steps'] if it['stage'] == 'ActionHead/step' else 1
        if it['kind'] == 'gemm':
            tot += R.hw_gemm_w(it['m'], it['n'], it['k'], cols)[1] * it['count'] * mul
    return tot


def row(name, t, freq=F):
    tot = t['total']
    return (name, tot / 1e6, tot / freq, t['macs'] / tot * freq / 1e9,
            t['sm'] / tot * 100)


def main():
    ref = S.build_spec(vit_tiles=2, hoist_kv=False)
    opt = S.build_spec(vit_tiles=2, hoist_kv=True)
    dead = 8 * R.hw_gemm_w(1, 2 * S.HEAD_E, S.HEAD_E, 108)[1] * (S.DENOISE_STEPS - 1)

    # ---- 自检 1：A 情景复现 CYCLE_ACCOUNT ----
    a = account_v2(opt, 108, 1, dead)
    print(f"[自检1] A·PRIM = {a['total']/1e6:.1f}M（CYCLE_ACCOUNT 基准 6126M，"
          f"偏差 {abs(a['total']/1e6 - 6126)/6126*100:.2f}%）")
    # ---- 自检 2：LW-ideal 复现 route_study B+SM8+LW ----
    b = account_v2(opt, 216, 8, dead, lw='ideal')
    print(f"[自检2] B+SM8+LW-ideal = {b['total']/1e6:.1f}M（route_study 基准 1059M，"
          f"偏差 {abs(b['total']/1e6 - 1059)/1059*100:.2f}%）")

    # ---- 面积敏感性：LUT/MAC → cols ----
    print("\n== HiF8 阵列宽度敏感性（LUT 预算模型）==")
    print(f"{'LUT/MAC':>8}{'cols(16×W)':>12}{'MAC/拍':>8}{'阵列LUT':>9}"
          f"{'非阵列LUT':>10}{'BRAM(含SEQ)':>12}")
    lut_tab = {}
    for lm in (35, 45, 60, 80):
        c = min(hif8_cols(lm), BRAM_CAP)
        arr = 16 * c * lm
        nonarr = FIX_LUT + PER_COL * c
        lut_tab[lm] = c
        print(f"{lm:>8}{c:>12}{16*c:>8}{arr:>9,}{nonarr:>10,}"
              f"{c + 14.5:>12.1f}")
    print(f"（现行 16×108 的阵列胶水 47.0k + requant 78.7k 作为对照；"
          f"packed INT8 cols=216 不受 LUT 预算约束，走 DSP）")

    # ---- 主表：路线阶梯（PRIM 口径）----
    scen = [
        ('A   现行 16×108 串行（基线）',        108, 1,  None),
        ('+SM16 softmax×16',                    108, 16, None),
        ('+SM32 softmax×32（需双读口）',        108, 32, None),
        ('+SM16 +LWreal DMA隐藏(带宽地板)',     108, 16, 'real'),
        ('+SM16 +LWideal 装载全消(上界)',       108, 16, 'ideal'),
        (f'H60 HiF8 16×{lut_tab[60]} +SM16 +LWreal', lut_tab[60], 16, 'real'),
        (f'H45 HiF8 16×{lut_tab[45]} +SM16 +LWreal', lut_tab[45], 16, 'real'),
        (f'H35 HiF8 16×{lut_tab[35]} +SM16 +LWreal', lut_tab[35], 16, 'real'),
        ('B   packed INT8 16×216 +SM16 +LWreal（对照）', 216, 16, 'real'),
        (f'H35 +SM32 +LWreal（激进）',          lut_tab[35], 32, 'real'),
    ]
    print(f"\n== 路线阶梯（PRIM，2 相机，@198.5 MHz）==")
    print(f"{'情景':<42}{'M cyc':>8}{'s/inf':>8}{'GMAC/s':>8}{'sm占':>6}")
    for name, cols, smp, lw in scen:
        t = account_v2(opt, cols, smp, dead, lw)
        n, mc, sec, eff, smpct = row(name, t)
        print(f"{n:<42}{mc:>8.0f}{sec:>8.2f}{eff:>8.1f}{smpct:>5.0f}%")

    # ---- SM 敏感性 × HiF8 档位（全 LWreal）----
    print(f"\n== SM 并行度 × HiF8 档位（LWreal，PRIM）==")
    hdr = 'cols\\sm'
    print(f"{hdr:>10}{'×8':>9}{'×16':>9}{'×32':>9}")
    for lm in (80, 60, 45, 35):
        c = lut_tab[lm]
        cells = []
        for smp in (8, 16, 32):
            t = account_v2(opt, c, smp, dead, 'real')
            cells.append(t['total'] / 1e6)
        print(f"16×{c}{'':<3}{cells[0]:>9.0f}{cells[1]:>9.0f}{cells[2]:>9.0f}")

    # ---- 优胜情景拆解 ----
    best = account_v2(opt, lut_tab[45], 16, dead, 'real')
    print(f"\n[H45+SM16+LWreal 拆解] softmax {best['sm']/1e6:.0f}M | GEMM喂料 "
          f"{best['compute']/1e6:.0f}M | COPY {best['copy']/1e6:.0f}M | "
          f"W装载流 {best['w_total']/1e6:.0f}M | 串行段 {best['ser']/1e6:.0f}M | "
          f"总 {best['total']/1e6:.0f}M（{'W 流是地板' if best['w_total'] > best['ser'] else '串行段是地板'}）")
    h35 = account_v2(opt, lut_tab[35], 16, dead, 'real')
    print(f"[H35+SM16+LWreal 拆解] softmax {h35['sm']/1e6:.0f}M | GEMM喂料 "
          f"{h35['compute']/1e6:.0f}M | COPY {h35['copy']/1e6:.0f}M | "
          f"W装载流 {h35['w_total']/1e6:.0f}M | 串行段 {h35['ser']/1e6:.0f}M | "
          f"总 {h35['total']/1e6:.0f}M")
    # 180 MHz 档（HiF8 micro 综合门槛 Fmax≥180）
    for lm in (45, 35):
        c = lut_tab[lm]
        t = account_v2(opt, c, 16, dead, 'real')
        print(f"[H{lm} @180MHz] {t['total']/180e6:.2f} s/inf"
              f"（@198.5 = {t['total']/F:.2f}）")


if __name__ == '__main__':
    main()
