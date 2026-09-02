# -*- coding: utf-8 -*-
"""taskB_matrix.py — 重叠 roofline 决策矩阵（11_overlap_plan，2026-08-31）

模型（全部模型口径，非实测）：
  总拍 = lane 公式 + 依赖停顿 θ
  C0 单引擎串行（现状）   : G+COPY+X+V_exp+W + θ0        （θ0=3.3M 校准复现 986.4M）
  C1 独立写通道 R1        : max(G+COPY+X+V_exp, W) + θ   （STORE 全异步，A 侧仍阻塞）
  C2 R1+R2 全重叠         : TB 口径（读写共口分时）: max(G+COPY, R+W) + θ
                            DUP/HP 口径（读写全双工）: max(G+COPY, R, W) + θ
  R = 读通道占用 = X(全量 ctx 字节) + V_full(全量权重字节)
  W = 写通道占用 = STORE 字节 / 写带宽

轴
  带宽口径 TB : ctx 3.701 / w_full 3.458 / store 2.718 B/cyc（a2 模型分量反推）
            DUP: 同 TB 速率但读写并行（全双工假设）
            HP16/32/64: 真机全双工，读写同速 16/32/64 B/cyc
  流量场景 S0 a2 现状 / S1 +AE_ACTV / S2 +跨段驻留 / S3 叠加 / S4 = S3+WRAM4 组消重装
  字节数据来自 taskA_boundary.py（boundary_account_a2.json）

θ 默认 40M（对齐 10_cbound_report 的 646M 下限口径），敏感度 3.3M（串行模型常数项）
/ 19.0M（a2 自拟合常数项 6870×2762）。
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MHZ = 198.5e6

# ---- 分量（模型口径，09_cbound repro 拟合系数 × a2 理想量，已验证合计 986.44M）----
G = 336.44          # GEMM（含行组固定开销；softmax 项 −0.14M 忽略）
COPY = 40.30        # 片上重排（与 GEMM 同用 CTX 端口，保守放计算通道）
THETA0 = 3.32       # 串行口径常数项（1253.5 cyc/段 × 2762 − softmax 项）
THETA = 40.0        # 重叠口径依赖停顿（起步值，见敏感度）

# ---- 有效带宽（B/cyc，模型口径）----
BW = {
    'TB':  dict(ctx=837.57 / 226.35, wf=587.06 / 169.78, st=751.35 / 276.42),
    'DUP': dict(ctx=837.57 / 226.35, wf=587.06 / 169.78, st=751.35 / 276.42),
    'HP16': dict(uniform=16.0), 'HP32': dict(uniform=32.0), 'HP64': dict(uniform=64.0),
}
# TB 口径全量权重服务 = pf=False 理想 79.68M × 2.1307 = 169.78M（587.06MB 全物理搬运）

MB = 1e6


def svc(caliber, kind, nbytes):
    if 'uniform' in BW[caliber]:
        return nbytes / BW[caliber]['uniform']
    return nbytes / BW[caliber][kind]


# ---- 场景字节（来自任务 A 分类账；MB）----
SCEN = {
    'S0 a2现状':      dict(ctx=837.6, wf=587.1, w_exp=103.61, st=751.4),
    'S1 +AE_ACTV':    dict(ctx=604.7, wf=587.1, w_exp=103.61, st=568.2),
    'S2 +跨段驻留':   dict(ctx=443.7, wf=587.1, w_exp=103.61, st=342.5),
    'S3 两者叠加':    dict(ctx=210.8, wf=587.1, w_exp=103.61, st=159.4),
    'S4 S3+WRAM4组':  dict(ctx=210.8, wf=438.6, w_exp=55.0,   st=159.4),
}
# S4 的 w_exp=55.0M：首装 438.6MB 暴露 55.0M + 重装 148.5MB 暴露 48.6M（10_cbound 报告）

COMP = G + COPY     # 计算通道 376.7M


def cell(caliber, scen, cfg, theta=THETA, comp=None):
    s = SCEN[scen]
    comp = COMP if comp is None else comp
    X = svc(caliber, 'ctx', s['ctx'] * MB) / 1e6          # ctx 全量读服务
    Vfull = svc(caliber, 'wf', s['wf'] * MB) / 1e6        # 权重全量读服务
    W = svc(caliber, 'st', s['st'] * MB) / 1e6            # 写服务
    # 暴露权重拍 = 该口径全量服务 × 暴露占比（TB 口径下标定：103.61/169.78）
    f_exp = s['w_exp'] / (s['wf'] / BW['TB']['wf'])
    Vexp = Vfull * f_exp
    if cfg == 'C0 单引擎串行':
        tot = G + COPY + X + Vexp + W + THETA0
        bind = '串行和'
    elif cfg == 'C1 R1独立写':
        base = G + COPY + X + Vexp
        tot = max(base, W) + theta
        bind = '计算+读串行' if base >= W else '写'
    else:  # C2 R1+R2
        R = X + Vfull
        if caliber == 'TB':          # 读写共口分时
            lanes = [('计算', comp), ('读+写共口', R + W)]
        else:                        # 全双工
            lanes = [('计算', comp), ('读', R), ('写', W)]
        bind = max(lanes, key=lambda t: t[1])
        tot = bind[1] + theta
    # ms 沿用项目口径：M拍 / 198.5 = ms（与 986.4M→4.97ms 锚点同一算法）
    return dict(total=tot, ms=tot / 198.5, bind=bind[0],
                X=X, Vfull=Vfull, W=W, Vexp=Vexp,
                hidden=(bind[0] == '计算'))


# ---- 验证：C0/TB/S0 必须复现 a2 986.4M ----
v = cell('TB', 'S0 a2现状', 'C0 单引擎串行')
dev = (v['total'] - 986.4) / 986.4 * 100
print('== 模型验证：C0 单引擎串行 / TB / S0 ==')
print('   模型 %.1fM vs 已定案 986.4M，偏差 %+.2f%%（门 2%%）%s'
      % (v['total'], dev, 'PASS' if abs(dev) <= 2 else 'FAIL — 查明再继续'))
print('   分量：G %.1f + COPY %.1f + X %.1f + Vexp %.1f + W %.1f + θ0 %.1f'
      % (G, COPY, v['X'], v['Vexp'], v['W'], THETA0))

# ---- 全矩阵 ----
CALI = ['TB', 'DUP', 'HP16', 'HP32', 'HP64']
CFGS = ['C0 单引擎串行', 'C1 R1独立写', 'C2 R1+R2']
out = {}
print('\n===== 决策矩阵：总拍 M｜ms@198.5MHz｜◎=瓶颈=计算通道（LOAD/STORE 全藏）=====')
for cfg in CFGS:
    print('\n--- %s（拍 M / ms）---' % cfg)
    print('%-16s' % '场景\\带宽' + ''.join('%13s' % c for c in CALI))
    for scen in SCEN:
        line = '%-16s' % scen
        for cal in CALI:
            c = cell(cal, scen, cfg)
            out['%s|%s|%s' % (cfg, scen, cal)] = c
            mark = '*' if c['hidden'] else ' '
            line += '%11.1f%2s' % (c['total'], mark)
        print(line)
        line = '%-16s' % '  ms→'
        for cal in CALI:
            c = out['%s|%s|%s' % (cfg, scen, cal)]
            line += '%13.2f' % c['ms']
        print(line)

# ---- 646M 锚点对照（报告口径：TB、单引擎、读加写串行、θ=40）----
print('\n== 646M 下限对照（TB 口径 C2）==')
c = cell('TB', 'S0 a2现状', 'C2 R1+R2')
print('   本模型 C2/TB/S0 = %.1fM（瓶颈=%s：读 %.1f + 写 %.1f = %.1f > 计算 %.1f）'
      % (c['total'], c['bind'], c['X'] + c['Vfull'], c['W'],
         c['X'] + c['Vfull'] + c['W'], COMP))
print('   报告 646M 的算式用了 pf 暴露权重 103.6M 而非全量物理服务 %.1fM；'
      '按暴露算：max(%.1f, %.1f+%.1f+%.1f)+40 = %.1fM（复现 646M）'
      % (c['Vfull'], COMP, c['X'], c['Vexp'], c['W'],
         max(COMP, c['X'] + c['Vexp'] + c['W']) + THETA))

# ---- θ 敏感度（关键格）----
print('\n== θ 敏感度（关键格总拍 M）==')
for key in ['C2 R1+R2|S3 两者叠加|DUP', 'C2 R1+R2|S4 S3+WRAM4组|TB',
            'C2 R1+R2|S4 S3+WRAM4组|DUP', 'C1 R1独立写|S0 a2现状|TB',
            'C2 R1+R2|S1 +AE_ACTV|DUP']:
    cfg, scen, cal = key.split('|')
    row = '  %-30s' % (scen.split(' ')[0] + '/' + cal + '/' + cfg.split(' ')[0])
    for th in (3.3, 19.0, 40.0):
        cc = cell(cal, scen, cfg, theta=th)
        row += '  θ=%4.1f: %6.1f' % (th, cc['total'])
    print(row)

# ---- 达成条件扫描：C2 全双工"全藏"所需最低带宽（θ=40M）----
print('\n== C2 全双工口径"全藏"所需最低带宽（B/cyc；读含权重，θ=40M，计算通道 %.1fM）==' % COMP)
for scen in SCEN:
    s = SCEN[scen]
    need_r = (s['ctx'] + s['wf']) / (COMP - THETA)
    need_w = s['st'] / (COMP - THETA)
    print('  %-16s 读(含W) ≥ %5.2f   写 ≥ %5.2f   （TB 现口径读 3.46-3.70 / 写 2.72）'
          % (scen, need_r, need_w))

# ---- R2 单独（无 R1）对照：A 预取但写仍占引擎 → 口径同共口 ----
print('\n== R2 单独（无 R1，写回仍占同一引擎/口）==')
for scen in SCEN:
    cc = cell('TB', scen, 'C2 R1+R2')   # 共口 max(计算, R+W) 即 R2-only 的上界口径
    print('  %-16s 总拍 %.1fM (%.2fms)' % (scen, cc['total'], cc['ms']))

# ---- 646M 何时被破（纯编译器 / +R1 / +R2，TB 口径）----
print('\n== 646M 下限何时被破（TB 口径，θ=40M）==')
for cfg in CFGS:
    for scen in SCEN:
        cc = cell('TB', scen, cfg)
        if cc['total'] < 646.0:
            print('  %-16s %-16s %.1fM（比 646 低 %.0fM）' % (cfg, scen, cc['total'], 646 - cc['total']))

# ---- R3 敏感度：行组流水把 GEMM 压向稳态乘加 ----
G_R3 = 133.9 * 1.0514          # 稳态乘加 133.9M 理想 × 系数
COMP_R3 = G_R3 + COPY
print('\n== R3（行组流水）敏感度：G %.1f→%.1fM，计算通道 %.1f→%.1fM（θ=40）=='
      % (G, G_R3, COMP, COMP_R3))
for cal in ['TB', 'DUP', 'HP16']:
    for scen in ['S3 两者叠加', 'S4 S3+WRAM4组']:
        cc = cell(cal, scen, 'C2 R1+R2', comp=COMP_R3)
        print('  C2/%-5s/%-14s 总拍 %.1fM (%.2fms) 瓶颈=%s'
              % (cal, scen, cc['total'], cc['ms'], cc['bind']))

json.dump(dict(components=dict(G=G, COPY=COPY, COMP=COMP, theta0=THETA0,
                               theta=THETA, BW=BW),
               scenarios=SCEN, cells=out),
          open(os.path.join(HERE, 'overlap_matrix.json'), 'w'),
          ensure_ascii=False, indent=1)
print('\n写出 overlap_matrix.json')
