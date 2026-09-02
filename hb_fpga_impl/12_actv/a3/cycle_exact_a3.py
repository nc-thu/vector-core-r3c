# -*- coding: utf-8 -*-
"""cycle_exact_a3.py — 逐段逐步事件模拟 vs 周期模型对拍（2026-08-31）。

「精确计数」= 双引擎事件模拟，逐条描述符推进，不用任何求和捷径：

  DMA 引擎（LOAD/STORE，一次一条命令）
  计算引擎（GEMM/SM16/COPY/AE_ACTV，一次一条命令）
  并发契约（与 a2 预取调度器同一假设）：紧贴某条 GEMM 之后发射的
  LOAD W，若落进与该 GEMM 相对的 WRAM 另一半区、且长度不越半区
  （n//cols ≤ half 且该 GEMM 的 k ≤ half），则与该 GEMM 并行执行——
  两条命令在 max(两者完成时刻) 之后才放行下一条；其余全部串行。

模型（acct_a3.seg_account 的 pf 捷径账）对同一段给出公式值。同一契约
下两者只差记账误差——本脚本验证 a3 融合段的新 LOAD W 模式没有被
pf 捷径记错。硬件级 ±5% 有效性由 a2 校准承担（a2 复算 986.7M vs
给定 986.4M，差 0.03%）。

用法: python cycle_exact_a3.py [build_a3] [--n 30] [--seed 20260831]
"""
import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                '09_cbound'))
from golden_interp import decode, load_seq                      # noqa: E402
from acct_a3 import (gemm_cycles, actv_cycles, load_ctx_ideal,  # noqa: E402
                     store_cycles, w_load_ideal, copy_cycles,
                     softmax_cycles, CAL, SEG_CONST, seg_account)


def seg_exact(sd, cols=108, w_words=4096):
    """事件模拟：返回该段拍数（不含每段常数）。"""
    half = w_words // 2
    t = 0                 # 全局时刻（命令放行时刻）
    prev_g = None         # 上一条 GEMM 的 (时长, 半区, k)
    for d in load_seq(sd):
        f = decode(d)
        op = f['op']
        if op == 15:
            break
        if op in (0, 1, 2):
            g = gemm_cycles(f['m'], f['k'], f['b_spad'], f['j0'],
                            f['y_tr'], cols)
            if op == 1:
                g += softmax_cycles(f['m'], f['n'], f['sm_causal'])
            t += g
            prev_g = (g, (f['b_base'] // half) & 1, f['k'])
        elif op == 4:
            n = f['dma_len']
            if f['b_src'] == 0:
                t += load_ctx_ideal(n)
                prev_g = None
            else:
                L = w_load_ideal(n, cols)
                if (prev_g is not None
                        and (f['b_base'] // half) & 1 != prev_g[1]
                        and n // cols <= half and prev_g[2] <= half):
                    # 与上一条 GEMM 并行：GEMM 已在跑，LOAD 现在发射，
                    # 下一条命令等两者都完成
                    t = max(t, t - prev_g[0] + L) + 1   # pf 命令开销 1 拍
                else:
                    t += L
        elif op == 5:
            t += store_cycles(f['dma_len'])
            prev_g = None
        elif op == 3:
            t += copy_cycles(f['k'], f['n'] & 0xFF, f['j0'])
            prev_g = None
        elif op == 6:
            t += actv_cycles(f['m'], f['n'])
            prev_g = None
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('build', nargs='?', default='build_a3')
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--seed', type=int, default=20260831)
    a = ap.parse_args()

    segdir = os.path.join(a.build, 'segments')
    segs = sorted(s for s in os.listdir(segdir) if s.startswith('seg_'))
    # 融合段清单：含 op6 的段全抽，其余随机补足
    fused = []
    for s in segs:
        for d in load_seq(os.path.join(segdir, s)):
            f = decode(d)
            if f['op'] == 15:
                break
            if f['op'] == 6:
                fused.append(s)
                break
    import random
    rng = random.Random(a.seed)
    fused_pick = fused if len(fused) <= a.n // 2 else \
        rng.sample(fused, a.n // 2)
    rest = [s for s in segs if s not in set(fused_pick)]
    pick = fused_pick + rng.sample(rest, a.n - len(fused_pick))
    pick.sort()

    n_in = 0
    errs = []
    keys = ('gemm', 'store', 'load_ctx', 'load_w', 'copy', 'softmax',
            'ae_actv')
    print('%-12s %12s %12s %8s  融合' % ('段', '模型拍', '模拟拍', '误差%'))
    for s in pick:
        acc, _, _ = seg_account(os.path.join(segdir, s))
        # 同一口径：理想公式账（不乘校准系数——系数是硬件折算，两边同乘
        # 不影响对拍结论）
        model = sum(acc[k] for k in keys)
        exact = seg_exact(os.path.join(segdir, s))
        e = (model - exact) / exact * 100 if exact else 0.0
        errs.append(abs(e))
        if abs(e) <= 5.0:
            n_in += 1
        print('%-12s %12.0f %12.0f %8.2f  %s'
              % (s, model, exact, e, '是' if s in set(fused_pick) else ''))
    print('\n±5%% 内：%d/%d  最大误差 %.2f%%  平均 %.2f%%'
          % (n_in, len(pick), max(errs), sum(errs) / len(errs)))
    return 0 if n_in == len(pick) else 1


if __name__ == '__main__':
    sys.exit(main())
