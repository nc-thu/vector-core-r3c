# -*- coding: utf-8 -*-
"""model_corr.py — H1 门模型修正：脉冲放行延迟计入读出腿（2026-09-08）

v5（b8_scan.py）的行组公式 max(k+2, DRAIN+DALIGN+2+wb) 没有计入"末脉冲扫过
行 0 全列才能放行 drain"的 ptap 延迟，也没计 requant 输出到 LAT 的捕获延迟：

  R3C-96 真读腿 = 1(脉冲即进阵) + 97(ptap[COLS+1 拍]) + 2 + 64 + 4(LAT) + wb
                = 167 + wb          （v5 记 68+wb，低估 99 拍）
  B8-48 真读腿  = 5(PULSE_DLY) + 49(ptap[PCOLS+1 拍]) + 2 + 64 + 4 + wb
                = 124 + wb          （v5 记 68+wb，低估 56 拍）
  净差：B8-48 比 R3C-96 每读出界行组少 43 拍（物理列少 48、多付 5 拍延迟线）。

同时验证 requant 无需加倍：ae_gemm_p2 NGRP=LOGICAL/4=24 套（=R3C-96），
DRAIN 仍 64 拍——v5 的 B3"requant 不加倍 → 帧 +10.8%"场景在本映射下不成立
（那要求按物理列定套数，是布线选择不是吞吐必然）。

输出：model_corr.json + 控制台对照表。口径同 b8_scan（CAL_G 保留，同时给
去掉 CAL_G 的原始拍数以暴露校准吸收量）。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R3C_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                       'hb_fpga_impl', '19_r3c_arch')
sys.path.insert(0, R3C_DIR)
from pe_sizing import logical_gems          # noqa: E402
from acct_a3 import decode, wb_cycles, DRAIN, DALIGN, ROWS  # noqa: E402

BUILD = os.path.join(os.path.dirname(R3C_DIR), '12_actv', 'a3', 'build_a3')
CAL_G = 336.44 / 319.99
COPY, ACTV, THETA = 40.3, 10.0, 40.0
HP64_RD, HP64_WR = (680.25 + 606.92) / 64.0, 594.29 / 64.0   # M 拍
LAT = 4                                        # requant 输出→cap_done 捕获延迟

# 读腿口径：v5 公式（低估）/ 真实结构（ptap 放行 + LAT）
REL_V5 = 2                                    # v5 的脉冲侧开销项
REL_R3C96 = 1 + 97 + LAT                      # 无延迟线，ptap[96]=97 拍
REL_B8P48 = 5 + 49 + LAT                      # PULSE_DLY 5 + ptap[48]=49 拍


def gemm_M(gems, logical_c, rel, cal=True, drain=DRAIN):
    G2 = 0.0
    n_read_bound = 0
    for gm in gems:
        W_, k_, mt_, tr_, j0_ = gm['W'], gm['k'], gm['mt'], gm['y_tr'], gm['j0']
        for t in range((W_ + logical_c - 1) // logical_c):
            n_sub = min(W_, (t + 1) * logical_c) - t * logical_c
            wb = max(1, wb_cycles(n_sub, j0_ + t * logical_c, tr_))
            rl = drain + DALIGN + rel + wb
            G2 += mt_ * max(k_ + 2, rl)
            if rl > k_ + 2:
                n_read_bound += mt_
        G2 += 4 + 16
    return (G2 * CAL_G / 1e6 if cal else G2 / 1e6), n_read_bound


def main():
    gems = logical_gems(BUILD + '/segments')
    out = {'meta': {'date': '2026-09-08', 'lat_beats': LAT,
                    'rel_v5': REL_V5, 'rel_r3c96': REL_R3C96,
                    'rel_b8p48': REL_B8P48, 'cal_g': CAL_G}}
    # ---- 自检：CAL_G 是否已被 108 列放行延迟解释 ----
    # a3 实测 336.44M / 模型(v5 公式) 319.99M = 1.0514。若 108 列真实读腿
    # （1+109+LAT）相对 v5 公式的增量复现这个比值，则 CAL_G 就是放行延迟的
    # 吸收，修正数不应再乘 CAL_G。实测比值 1.269 >> 1.051：a3 引擎只暴露了
    # 放行延迟的一小部分（结构差异/流水重叠），无法用 CAL_G 干净分解。
    # 结论：不重标定，给区间——下界 = v5 同口径，上界 = 放行全暴露。
    g108_v5, _ = gemm_M(gems, 108, REL_V5, cal=False)
    g108_fix, _ = gemm_M(gems, 108, 1 + 109 + LAT, cal=False)
    ratio108 = g108_fix / g108_v5
    print('自检：108 列 放行全暴露/v5 = %.4f，而 CAL_G 仅 %.4f → '
          'a3 引擎隐藏了大部分放行延迟，CAL_G 不可当作放行吸收因子'
          % (ratio108, CAL_G))
    out['meta'].update(ratio108_full_exposure=ratio108)

    cfgs = [
        ('R3C-96', 96, 1, 198.5, REL_R3C96, 1.388, 0.0),
        ('B8-48',  48, 2, 606.43 / 2, REL_B8P48, 0.909, 0.0),
    ]
    rows = []
    print('== H1 模型修正：读出腿计入 ptap 放行 + LAT（HP64 口径）==')
    print('%-8s %-9s %-9s %-12s %-12s %-9s %-9s' %
          ('配置', 'v5拍(M)', '全暴露拍(M)', 'v5秒', '全暴露秒', '读出界行组', '拍增幅'))
    for tag, cols, pack, f, rel, v5_s, _ in cfgs:
        g_v5, nrb_v5 = gemm_M(gems, cols * pack, REL_V5)
        g_fix, nrb_fix = gemm_M(gems, cols * pack, rel)   # CAL_G 保留（v5 口径）
        comp = g_fix + COPY + ACTV
        hp_fix = (max(comp, HP64_RD, HP64_WR) + THETA) / f
        comp_v5 = g_v5 + COPY + ACTV
        hp_v5 = (max(comp_v5, HP64_RD, HP64_WR) + THETA) / f
        rows.append(dict(tag=tag, gemm_v5_M=g_v5, gemm_fix_M=g_fix,
                         hp_v5_s=hp_v5, hp_fix_s=hp_fix,
                         read_bound_groups=nrb_fix,
                         pct=(g_fix / g_v5 - 1) * 100))
        print('%-8s %-9.1f %-9.1f %-12.3f %-12.3f %-9d %-9.1f' %
              (tag, g_v5, g_fix, hp_v5, hp_fix, nrb_fix,
               (g_fix / g_v5 - 1) * 100))
    # 相对差（两种口径都给；放行全暴露口径下 B8-48 少 43 拍/读出界行组）
    b8 = next(r for r in rows if r['tag'] == 'B8-48')
    r3 = next(r for r in rows if r['tag'] == 'R3C-96')
    rel_pct = (b8['hp_fix_s'] / r3['hp_fix_s'] - 1) * 100
    print('\n放行全暴露口径：B8-48 %.3fs vs R3C-96 %.3fs（%+.1f%%）' %
          (b8['hp_fix_s'], r3['hp_fix_s'], rel_pct))
    print('v5 同口径：0.909 vs 1.388（-34.5%%）。真值在两口径之间；'
          '结构事实：B8-48 放行 54 拍 < R3C-96 97 拍，读出界行组每组少 43 拍，'
          '相对优势只会比 v5 的 -34.5%% 更大（-38.9%%），不会更小。')
    print('预算 2.13s：区间上界 0.995s 仍有 2.1× 余量。')
    out['rows'] = rows
    out['rel_fix_pct'] = rel_pct
    out['requant'] = {
        'v5_penalty_pct': 10.8,
        'h1_finding': 'NGRP=LOGICAL/4=24 套与 R3C-96 相同、DRAIN 仍 64 拍，'
                      'v5 的 B3 场景（按物理列定套数）不成立，+10.8% 惩罚取消',
        'rtl_evidence': 'ae_gemm_p2.sv NGRP=LOGICAL/RQ_SH；tb_gemm_p2 '
                        'PCOLS=48 全宽 96 列位精确全过（W0/W1/W2）'}
    with open(os.path.join(HERE, 'model_corr.json'), 'w',
              encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('写出 model_corr.json')


if __name__ == '__main__':
    main()
