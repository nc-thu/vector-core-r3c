# -*- coding: utf-8 -*-
"""b8_final.py — B8 定版模型（架构线 v6，2026-09-08）

三件事（对应 hw v5 同轮采的点）：
  1. LUT 模型用 H1 实测锚点修正：v5 的引擎外推 = 物理列×3140 LUT（436 列基
     础设施 + 16×169 PE），B8-48 引擎估 150.7k；H1 布线实测 eng48=118,148，
     模型高 27.6%。修正 = 按实测反推每列有效 LUT（118,148/48=2461.4），全档
     重算；eng64 综合数落地后（measured.json）优先用双点实测。
     芯片级 = 引擎实测/外推 + 25k 非引擎基础设施（R3C-96=120,616 实测锚定）。
  2. H2 相位平移的帧级代价：读出 walk 每次 +1 拍上界（DRAIN 64→65 敏感性），
     对照 hw v5 TB 实测（9 描述符 6 个拍数不变、3 个 +4 拍）。
  3. R2 双读引擎后的写侧占比：R2 把 TB 读腿 301.0→184.0M 拍后，B8-48 回到
     计算地板、B8-64 变写瓶颈（W=218.4M）——R5 写侧压缩进 H3 的量化依据。

口径与 v5（b8_scan.py）完全一致：TB=验证平台 / HP64=真机 64B/拍，单位 M 拍，
GEMM 校准 336.44/319.99，θ=40。纯模型 + 实测锚点，不碰 RTL。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V5 = os.path.join(os.path.dirname(HERE), 'v5_2026-09-08_1414_b8_sizing_read_compress')
sys.path.insert(0, V5)
import b8_scan as v5          # noqa: E402  (collect/gemm_recomp_M/V_M/PES 全复用)

# H1/v5 实测锚点（hw/v4 H1 门 + 本轮 hw/v5 综合；来源见 CHANGELOG）
ENG48_LUT = 118148            # eng48 布线后 OOC 实测（WNS −0.169ns 那一版）
R3C96_CHIP_LUT = 120616       # R3C-96 芯片实测（25k 基础设施锚定的来源）
INFRA_LUT = 25000             # 非引擎基础设施（R3C-96 实测 − 引擎外推校准）
DEV_LUT = 230400

# 本轮实测（有则用；measured.json 由综合收口时写出）
MFILE = os.path.join(HERE, 'measured.json')
MEAS = {}
if os.path.exists(MFILE):
    MEAS = json.load(open(MFILE, encoding='utf-8'))


def engine_lut(c):
    """引擎 LUT：eng48 实测反推每列 2461.4 外推；eng64 实测优先直接用。"""
    if str(c) in MEAS.get('eng_lut', {}):
        return MEAS['eng_lut'][str(c)], '实测'
    slope = ENG48_LUT / 48.0
    return int(slope * c), '外推'


def main():
    data = v5.collect()
    print('段 %d　逻辑 GEMM %d　MAC %.1f G/帧' %
          (data['segs'], len(data['gems']), data['macs'] / 1e9))

    # ---- 1. H1 锚点 LUT 表 ----
    print('\n== LUT：v5 外推 vs H1 锚点修正（引擎 OOC / 芯片=引擎+25k 基础设施）==')
    print('%-8s %-12s %-10s %-10s %-12s %-10s %s' %
          ('配置', 'v5引擎外推', 'H1实测', '锚点外推', '芯片(v5估→修)', '占用%', '口径'))
    lut_tbl = []
    eng_model_col = v5.COL_INFRA + v5.ROWS * v5.PES['B8']['pe_lut']   # 3140
    for c in [24, 32, 48, 64, 96]:
        v5_eng = int(eng_model_col * c)
        eng, src = engine_lut(c)
        chip = eng + INFRA_LUT
        v5_chip = v5_eng + INFRA_LUT
        ok = 'OK' if chip / DEV_LUT <= 0.85 else ('紧张' if chip / DEV_LUT <= 1.0 else '放不下')
        print('B8-%-4d %-12d %-10s %-10d %-12d %-10.1f %s（%s）' %
              (c, v5_eng, '118,148' if c == 48 else '—', eng,
               int(v5_chip), chip / DEV_LUT * 100,
               '%d/%d/%s' % (chip, DEV_LUT, ok), src))
        lut_tbl.append(dict(cols=c, eng_lut=eng, eng_src=src, chip_lut=chip,
                            chip_pct=chip / DEV_LUT * 100, v5_eng_est=v5_eng,
                            fits_85pct=chip / DEV_LUT <= 0.85))
    m48 = 48 * eng_model_col
    print('   模型偏差：eng48 外推 %d vs 实测 %d → 模型高 %.1f%%（锚点斜率 %.1f LUT/列）'
          % (m48, ENG48_LUT, (m48 / ENG48_LUT - 1) * 100, ENG48_LUT / 48.0))

    # ---- 2. H2 相位平移帧级代价（DRAIN+1 敏感性上界）----
    print('\n== H2 走读 +1 拍的帧级代价（DRAIN 64→65，上界：所有 walk 都吃满）==')
    h2_tbl = []
    for c in [24, 48, 64, 96]:
        g0 = v5.gemm_recomp_M(data['gems'], c * 2)
        g1 = v5.gemm_recomp_M(data['gems'], c * 2, drain=v5.DRAIN + 1)
        comp0, comp1 = g0 + v5.COPY + v5.ACTV, g1 + v5.COPY + v5.ACTV
        tb0 = max(comp0, v5.X_ANCHOR + v5.V_M(data['w_loads'], data['v108'], c),
                  v5.W_ANCHOR) + v5.THETA
        tb1 = max(comp1, v5.X_ANCHOR + v5.V_M(data['w_loads'], data['v108'], c),
                  v5.W_ANCHOR) + v5.THETA
        hp1 = (comp1 + v5.THETA) / v5.PES['B8']['f_iface']
        h2_tbl.append(dict(cols=c, gemm_d65_M=g1, gemm_delta_pct=(g1 / g0 - 1) * 100,
                           tb_d65_M=tb1, tb_delta_pct=(tb1 / tb0 - 1) * 100,
                           hp_d65_s=hp1))
        print('B8-%-3d GEMM %3.0f→%3.0fM（+%4.2f%%）　TB %3.0f→%3.0fM（+%4.2f%%）'
              '　HP64 %.3fs' % (c, g0, g1, (g1 / g0 - 1) * 100,
                               tb0, tb1, (tb1 / tb0 - 1) * 100, hp1))
    print('   （TB 实测口径：9 描述符 6 个拍数不变、3 个 +4 拍，+0.9~1.5%）')

    # ---- 3. R2 后写侧占比（R5 进 H3 的量化依据）----
    print('\n== R2 双读引擎后各腿占比（TB 口径）==')
    r5_tbl = []
    for c in [48, 64]:
        cfg = v5.cell('B8', c, data)
        V = cfg['V_M']
        rd2 = max(v5.X_ANCHOR, V)          # R2 后读腿
        tb2 = max(cfg['comp_M'], rd2, v5.W_ANCHOR) + v5.THETA
        bind = max([('计算', cfg['comp_M']), ('读', rd2), ('写', v5.W_ANCHOR)],
                   key=lambda t: t[1])[0]
        w_share = v5.W_ANCHOR / tb2 * 100
        # R5 假想上界：写腿压到 comp/读腿水平（拍数收益）
        tb_r5 = max(cfg['comp_M'], rd2) + v5.THETA
        r5_tbl.append(dict(cols=c, tb_R2_M=tb2, tb_R2_s=tb2 / 303.215,
                           bind=bind, w_share_pct=w_share,
                           tb_R5_s=tb_r5 / 303.215,
                           r5_gain_pct=(1 - tb_r5 / tb2) * 100))
        print('B8-%-3d R2 后 TB %.1fM 拍 = %.3fs（瓶颈 %s）　写腿占 %.1f%%'
              '　R5 写压缩到地板 → %.3fs（上限 −%.1f%%）' %
              (c, tb2, tb2 / 303.215, bind, w_share,
               tb_r5 / 303.215, (1 - tb_r5 / tb2) * 100))

    # ---- 定版判据（实测数到齐后回填）----
    print('\n== 定版判据（B8-64 抢 48 的三个条件，eng64 综合数回填）==')
    for k, cond in [('eng64_wns_ns', '≥ 0（303.215MHz 布线收敛）'),
                    ('eng64_lut', '芯片 ≤ 85%（≈196k）'),
                    ('eng64_pwr_mw', '≤ 9000 mW（vectorless 档）')]:
        val = MEAS.get(k, '待综合')
        print('   %-14s = %-10s（判据 %s）' % (k, val, cond))

    out = dict(meta=dict(date='2026-09-08', build='build_a3',
                         base='arch/v5 b8_scan.py 口径复用',
                         eng48_lut=ENG48_LUT, r3c96_chip_lut=R3C96_CHIP_LUT),
               lut=lut_tbl, h2_drain65=h2_tbl, r2_writeside=r5_tbl,
               measured=MEAS)
    with open(os.path.join(HERE, 'b8_final.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n写出 b8_final.json')


if __name__ == '__main__':
    main()
