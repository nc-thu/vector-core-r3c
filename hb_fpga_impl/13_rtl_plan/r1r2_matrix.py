# -*- coding: utf-8 -*-
"""r1r2_matrix.py — R1(异步写通道)+R2(CTX A 预取双缓冲) 决策矩阵
基于 a3 真实流分量，重算在 TB 从机 / 真机 HP 双口径下能否让 GEMM 成为主要计算时间。

模型（沿用 11_overlap_plan/taskB_matrix.py 的交叠语义，分量换成 a3 真实值）：
  总拍 = lane 公式 + 依赖停顿 θ
  C0 单引擎串行（现状）: comp + X + V + W + θ0       （θ0 校准复现 901.1M）
  C1 +R1 独立写通道     : max(comp + X + V, W) + θ   （STORE 异步并发，读仍串行）
  C2 +R1+R2 全重叠      : max(comp, X+V, W) + θ      （全双工：读/写/计算三路并发）
       —— TB 从机物理上是 AXI 全双工（AR/R 与 AW/W/B 独立 always_ff，见
          01_rtl/sim/tb_ae.sv:65-111），所以 R1 的独立写通道在 TB 下真能并发；
          taskB 的"共口分时"是保守下界，本脚本主算全双工，共口作对照。

a3 真实分量（校准口径，外样 99.9%，2580 段，@198.5MHz=4.54ms）：
  GEMM 336.9 / STORE 218.4 / LOAD_CTX 184.0 / LOAD_W 108.1 / COPY 40.3 /
  AE_ACTV 10.0 / 每段常数 3.2 → 总 901.1M
字节：LOAD_CTX 680MB / LOAD_W 608MB / STORE 593MB
TB 有效带宽（由 a3 字节/分量反推）：ctx 3.70 / w 5.62 / st 2.72 B/cyc
真机 HP 口径：16/32/64 B/cyc 全双工

θ0 = 3.4M（a3 单引擎串行反推：901.1 − 387.2 − 184.0 − 108.1 − 218.4 = 3.4）
θ  = 40M（重叠口径依赖停顿，沿用 taskB 保守值；敏感度见文末）

注意：a3 的 LOAD_W=108.1M 是 pf 命中后的"暴露"值。a3 pf 命中变差
（9702 次，比 a2 少），暴露值已接近全量物理搬运 608/5.62=108.2M。
所以本模型对 LOAD_W 直接用 108.1M 当 V（≈Vfull），不再区分 Vexp/Vfull
——在 a3 pf 几乎没遮蔽的前提下两者差 <1%，不影响结论。
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MHZ = 198.5e6

# ============ a3 真实分量（M 拍，校准口径）============
G       = 336.9    # GEMM（含行组固定开销 185.8M + 稳态 MAC 133.9M，理想 ×1.0514）
COPY    = 40.3     # 片上重排
AE_ACTV = 10.0     # op=6 引擎（192 站 281 条描述符）
COMP    = G + COPY + AE_ACTV   # 计算通道 = 387.2M（AE_ACTV 与 GEMM 串行发射，计入计算关键路径）

X       = 184.0    # LOAD_CTX 校准服务（680MB / 3.70 B/cyc）
V       = 108.1    # LOAD_W 校准服务（pf 后暴露 ≈ 全量物理 608MB / 5.62 B/cyc）
W       = 218.4    # STORE 校准服务（593MB / 2.72 B/cyc）

THETA0  = 3.4      # 串行口径常数（a3 反推：901.1 − 387.2 − 184.0 − 108.1 − 218.4）
THETA   = 40.0     # 重叠口径依赖停顿（taskB 保守值，敏感度见末段）

# 字节（MB）—— HP 口径重算服务时间用
BYTES = dict(ctx=680.0, w=608.0, st=593.0)
MB = 1e6

# ============ 带宽口径 ============
# TB：从 a3 字节/分量反推（已含校准开销 1beat/2cyc + 1/8 LFSR）
BW_TB = dict(ctx=BYTES['ctx']/X,   w=BYTES['w']/V,   st=BYTES['st']/W)
# HP：真机全双工，读写同速
HP_CALIBERS = {'HP16': 16.0, 'HP32': 32.0, 'HP64': 64.0}


def svc_tb(kind):
    """TB 口径服务时间 = a3 分量本身（已是 TB 校准值）"""
    return {'ctx': X, 'w': V, 'st': W}[kind]


def svc_hp(bw, kind):
    """HP 口径服务时间 = 字节 / 带宽"""
    return BYTES[kind] * MB / bw / 1e6


def cell(caliber, cfg, theta=THETA):
    """算一格：返回 total(M)/ms/GEMM 占比/瓶颈/是否 GEMM 主要(>50%)"""
    if caliber == 'TB':
        x, v, w = X, V, W
    else:
        bw = HP_CALIBERS[caliber]
        x, v, w = svc_hp(bw, 'ctx'), svc_hp(bw, 'w'), svc_hp(bw, 'st')

    if cfg == 'C0 单引擎串行':
        tot = COMP + x + v + w + THETA0
        bind = '串行和'
    elif cfg == 'C1 R1独立写':
        base = COMP + x + v        # 计算+读串行
        tot = max(base, w) + theta
        bind = '计算+读串行' if base >= w else '写'
    else:  # C2 R1+R2（全双工，三路并发）
        lanes = [('计算', COMP), ('读(X+V)', x + v), ('写(W)', w)]
        bind_lane = max(lanes, key=lambda t: t[1])
        bind = bind_lane[0]
        tot = bind_lane[1] + theta

    gemm_share = G / tot * 100
    return dict(total=tot, ms=tot / 198.5, bind=bind,
                x=x, v=v, w=w, gemm_share=gemm_share,
                gemm_major=gemm_share > 50.0)


# ============ 1. 校准验证：C0/TB 必须复现 a3 901.1M ============
print('=' * 72)
print('1. 模型校准验证：C0 单引擎串行 / TB 口径')
print('=' * 72)
v0 = cell('TB', 'C0 单引擎串行')
dev = (v0['total'] - 901.1) / 901.1 * 100
print('   模型 %.1fM vs a3 定案 901.1M，偏差 %+.2f%%（门 5%%）%s'
      % (v0['total'], dev, 'PASS' if abs(dev) <= 5 else 'FAIL'))
print('   分量：COMP %.1f(G %.1f+COPY %.1f+AE_ACTV %.1f) + X %.1f + V %.1f + W %.1f + θ0 %.1f'
      % (COMP, G, COPY, AE_ACTV, v0['x'], v0['v'], v0['w'], THETA0))
print('   TB 有效带宽：ctx %.2f / w %.2f / st %.2f B/cyc（由 a3 字节/分量反推）'
      % (BW_TB['ctx'], BW_TB['w'], BW_TB['st']))

# ============ 2. 6 格决策矩阵 ============
CALI = ['TB', 'HP64']
CFGS = ['C0 单引擎串行', 'C1 R1独立写', 'C2 R1+R2']
out = {}
print()
print('=' * 72)
print('2. 决策矩阵：3 配置 × 2 口径（6 格）')
print('=' * 72)
print('   总拍 M | ms@198.5MHz | GEMM 占比 | 瓶颈 | GEMM>50%?')
print('-' * 72)
for cfg in CFGS:
    for cal in CALI:
        c = cell(cal, cfg)
        key = '%s | %s' % (cfg, cal)
        out[key] = c
        flag = 'YES' if c['gemm_major'] else 'no'
        print('  %-18s | %7.1f M | %5.2f ms | %5.1f%% | %-12s | %s'
              % (cfg + '/' + cal, c['total'], c['ms'], c['gemm_share'],
                 c['bind'], flag))
    print('-' * 72)

# ============ 3. 关键判断：TB 口径下 R1+R2 能否让 GEMM>50% ============
print()
print('=' * 72)
print('3. 关键判断：TB 口径下 R1+R2 能否让 GEMM 占比 >50%')
print('=' * 72)
c2_tb = cell('TB', 'C2 R1+R2')
c2_hp = cell('HP64', 'C2 R1+R2')
print('   TB 口径 C2 R1+R2: 总拍 %.1fM (%.2fms), GEMM 占比 %.1f%%, 瓶颈=%s'
      % (c2_tb['total'], c2_tb['ms'], c2_tb['gemm_share'], c2_tb['bind']))
print('   HP64 口径 C2 R1+R2: 总拍 %.1fM (%.2fms), GEMM 占比 %.1f%%, 瓶颈=%s'
      % (c2_hp['total'], c2_hp['ms'], c2_hp['gemm_share'], c2_hp['bind']))
print()
print('   写墙分析（任务问的核心）:')
print('     TB 写服务 W = 218.4M（593MB / 2.72 B/cyc，2.72 是 1beat/2cyc + 1/8 LFSR 写最差）')
print('     TB 读服务 X+V = %.1fM（184.0 + 108.1）' % (X + V))
print('     计算通道 COMP = %.1fM（GEMM %.1f + COPY %.1f + AE_ACTV %.1f）'
      % (COMP, G, COPY, AE_ACTV))
print('     → 三路并发 max(%.1f, %.1f, %.1f) = %.1fM = 计算通道'
      % (COMP, X + V, W, max(COMP, X + V, W)))
print('     → W=%.1fM < COMP=%.1fM，写墙被计算完全藏住，不是硬墙' % (W, COMP))
print('     → X+V=%.1fM < COMP=%.1fM，读也被藏住' % (X + V, COMP))
print()
print('   TB 从机物理双工验证（01_rtl/sim/tb_ae.sv:65-111）:')
print('     AR/R 通道与 AW/W/B 通道是两个独立 always_ff，arready/awready 独立')
print('     → TB 从机本身就是 AXI 全双工，R1 的独立写通道在 TB 下真能并发')
print('     → taskB 的"TB 共口分时"是保守下界，物理上不成立')
print()
print('   结论: %s — TB 口径下 R1+R2 让 GEMM 占比 %.1f%% %s 50%%'
      % ('达成' if c2_tb['gemm_major'] else '未达',
         c2_tb['gemm_share'],
         '>' if c2_tb['gemm_major'] else '<'))

# ============ 4. 对照：taskB 保守"共口分时"口径 ============
print()
print('=' * 72)
print('4. 对照：taskB 保守"共口分时"口径（TB 读写共口，非物理现实）')
print('=' * 72)
def cell_tb_shared(cfg, theta=THETA):
    """TB 共口分时：读写共享物理口，R+W 串行"""
    if cfg == 'C0 单引擎串行':
        return cell('TB', cfg)
    elif cfg == 'C1 R1独立写':
        # 共口下 R1 无意义（写仍占读的口）= 串行
        return cell('TB', 'C0 单引擎串行')
    else:  # C2: max(comp, X+V+W) + θ
        tot = max(COMP, X + V + W) + theta
        bind = '计算' if COMP >= X + V + W else '读+写共口'
        return dict(total=tot, ms=tot/198.5, bind=bind,
                    gemm_share=G/tot*100, gemm_major=G/tot*100 > 50)
for cfg in CFGS:
    c = cell_tb_shared(cfg)
    flag = 'YES' if c['gemm_major'] else 'no'
    print('  %-18s | %7.1f M | %5.2f ms | %5.1f%% | %-12s | %s'
          % (cfg + '/TB共口', c['total'], c['ms'], c['gemm_share'],
             c['bind'], flag))

# ============ 5. HP16/HP32 敏感度 ============
print()
print('=' * 72)
print('5. HP 敏感度（16/32/64 B/cyc）')
print('=' * 72)
for cal in ['HP16', 'HP32', 'HP64']:
    for cfg in CFGS:
        c = cell(cal, cfg)
        flag = 'YES' if c['gemm_major'] else 'no'
        print('  %-18s | %7.1f M | %5.2f ms | %5.1f%% | %-12s | %s'
              % (cfg + '/' + cal, c['total'], c['ms'], c['gemm_share'],
                 c['bind'], flag))
    print('-' * 72)

# ============ 6. θ 敏感度 ============
print()
print('=' * 72)
print('6. θ 敏感度（C2 R1+R2 关键格，总拍 M / GEMM 占比）')
print('=' * 72)
for cal in ['TB', 'HP64']:
    row = '  %-8s' % cal
    for th in (3.4, 10.0, 20.0, 40.0, 80.0):
        c = cell(cal, 'C2 R1+R2', theta=th)
        row += '  θ=%4.1f: %6.1fM/%4.1f%%' % (th, c['total'], c['gemm_share'])
    print(row)

# ============ 7. 若 TB 不够：缺口清单 ============
print()
print('=' * 72)
print('7. 缺口清单（若 TB 口径下 R1+R2 不够，还需要的额外条件）')
print('=' * 72)
c2_tb = cell('TB', 'C2 R1+R2')
if c2_tb['gemm_major']:
    print('   TB 口径下 R1+R2 已达 GEMM %.1f%% > 50%%，无需额外条件'
          % c2_tb['gemm_share'])
    print('   但若要进一步压到 672M（GEMM 占比 50% 的等价总拍 = G/0.5）:')
    target = G / 0.5
    print('     目标总拍 = GEMM/0.5 = %.1fM, 当前 C2/TB = %.1fM, 缺口 %.1fM'
          % (target, c2_tb['total'], c2_tb['total'] - target))
else:
    target = G / 0.5
    gap = c2_tb['total'] - target
    print('   TB 口径下 C2 = %.1fM, 目标 %.1fM, 缺口 %.1fM' %
          (c2_tb['total'], target, gap))
    print('   补缺口需要的额外条件（任一即可）:')
    # 条件1：降 W
    w_max = COMP - THETA
    print('   ① 写服务需降到 %.1fM 以下（当前 %.1fM，需省 %.1fM = %.0fMB 字节）'
          % (w_max, W, W - w_max, (W - w_max) * BW_TB['st'] / 1e6 * 1e6 / 1e6))
    # 条件2：第二 DMA 引擎
    print('   ② 第二 DMA 引擎（读/写各一，物理双口）—— 但 TB 从机已双工，'
          '不增收益')
    # 条件3：降 θ
    print('   ③ θ 从 40M 降到 %.1fM（见敏感度表）' % (c2_tb['total'] - target))

# ============ 8. R1/R2 RTL 改造范围预估 ============
print()
print('=' * 72)
print('8. R1/R2 RTL 改造范围预估')
print('=' * 72)
print('''
现状硬件（01_rtl/rtl）:
  ae_dma.sv  220 行，单 FSM（D_IDLE→D_AR/D_R/D_R2 读 / D_AW/D_RD/D_W/D_B 写）
             串行执行 LOAD/STORE，AR/R 与 AW/W/B 通道物理独立但 FSM 互斥
  ae_ctx_ram.sv 30 行，单体 CTX（131072×128b），A 口只读/B 口只写，ram_style=ultra
  ae_sched.sv 287 行，T_RUN_* 互斥，pf 仅认 TAG_W（前瞻深度 1）
  ae_dma 现规模 1469 LUT；WNS 瓶颈在 u_actv（−1.359@250MHz），不在 ae_dma

R1 改造（独立写通道）:
  改 ae_dma.sv: 把单 FSM 拆成两个并发 FSM（读引擎 D_AR/D_R/D_R2 +
                写引擎 D_AW/D_RD/D_W/D_B），共享地址/命令锁存、CTX/WRAM
                端口已物理分离（STORE 用 ctx_raddr/A 口，LOAD 用 ctx_addr/B 口）
  改 ae_sched.sv: 加 STORE 与 GEMM/LOAD 并发状态（不再 T_RUN_* 互斥）
  LUT 增量: +700~1300（FSM 拆分 + 调度并发，数据通路复用）
            其中 ae_dma +500~900, ae_sched +200~400
  BRAM 增量: 0
  风险: 中低。CTX A/B 口物理分离，AXI AR/R 与 AW/W/B 独立，端口层无冲突；
        WNS 瓶颈在 u_actv 不在 ae_dma，改 ae_dma 不恶化关键路径。
        主要风险: 两引擎对 AXI 互连的并发请求（背靠背时），需验 B 通道响应时序。

R2 改造（CTX A 预取双缓冲）:
  改 ae_ctx_ram.sv: CTX 分区方案（推荐，不增 BRAM）——把 CTX 地址空间划出
                "当前 A 区"和"预取 A 区"，GEMM 读当前 A 时 DMA 把下一段 A 写进
                预取区，段切换时翻指针。或乒乓双体方案（CTX 容量翻倍，
                BRAM/URAM 增量大，ZCU104 URAM 预算紧）
  改 ae_sched.sv 的 pf FSM: 扩 TAG_C（CTX A 预取标签），pf 前瞻深度从 1
                增到 2（同时跟踪 W 和 CTX A 两个目标）
  LUT 增量: +500~900（分区方案）
            其中 ae_ctx_ram 选通 +100~200, ae_sched pf FSM 扩展 +300~500,
            地址生成 +100~200
  BRAM 增量: 0（分区方案） / +4~8 BRAM tiles（乒乓双体方案，不推荐）
  风险: 中。CTX 分区调度增加复杂度，pf FSM 扩展有交错风险
        （pf_bg_start 与 GEMM 抢 CTX B 口写）；WNS 在 u_actv，CTX 路径有余量。
        乒乓双体方案 BRAM 预算风险高（CTX 2MB 已近 URAM 上限，翻倍可能放不下）。

R1+R2 合计: +1200~2200 LUT, +0 BRAM（分区方案）
            对照全片余量 95,666 LUT（41.5%），资源完全放得下。
''')

# ============ 9. 写出 JSON ============
result = dict(
    components=dict(G=G, COPY=COPY, AE_ACTV=AE_ACTV, COMP=COMP,
                    X=X, V=V, W=W, THETA0=THETA0, THETA=THETA),
    bytes_MB=BYTES, BW_TB=BW_TB,
    cells=out,
    c2_tb=dict(total=c2_tb['total'], ms=c2_tb['ms'],
               gemm_share=c2_tb['gemm_share'], gemm_major=c2_tb['gemm_major'],
               bind=c2_tb['bind']),
)
with open(os.path.join(HERE, 'r1r2_matrix.json'), 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=1)
print('写出 r1r2_matrix.json')
