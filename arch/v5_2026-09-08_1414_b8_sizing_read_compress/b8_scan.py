# -*- coding: utf-8 -*-
"""b8_scan.py — B8 PE 旋钮进周期模型 + 读压缩三路线重算（架构线 v5，2026-09-08）

问题（A1）：B8 PE（Pack2×Pump2，64 集群 606.4 MHz，1.213 GMAC/s/DSP = R3C PE 6.11 倍）
出来之后，到底该用多少列 B8？窄阵列换面积（DSP/LUT/BRAM/功耗）划不划算？

方法：完全沿用 19_r3c_arch/pe_sizing.py 的重编译口径（逻辑 GEMM 重建 + 按新列宽
均衡重切，A 重喂次数 ceil(W/逻辑列数) 是固有代价），只加一个 PE 类型旋钮：

  R3C：1 MAC/接口拍/DSP，接口时钟 198.5 MHz（部署基线口径）
  B8 ：4 MAC/接口拍/DSP（Pack2 每物理列驻 2 条逻辑列 × Pump2 核心时钟 2 倍），
       接口时钟 303.2 MHz（64 PE 集群布线后 606.43 MHz / 2）
  逻辑列数 = 物理列数 × pack（R3C pack=1，B8 pack=2）
  行组 tile 公式不变 max(k+2, DRAIN+DALIGN+2+wb)：喂数/读出两条腿都在接口时钟域，
  wb 随逻辑宽度变宽、子组数变少，总读出工作量不变——由重切机制自动反映。
  B8 PE 面积锚点（pe_w8a8_sota/rounds/2026-09-04_150930 实测）：单核布线后
  171 LUT / 215 FF；cl64 = 10,806 LUT ≈ 169 LUT/PE。

  lane 模型（13_rtl_plan/r1r2_matrix.py 口径，TB=验证平台 / HP64=真机 64B/拍）：
  TB   = max(comp, X+V(C), W) + θ        θ=40（保守停顿）
  HP64 = max(comp, (ctx+w 字节)/64, st 字节/64) + θ
  X/W 字节驱动与列数无关（锚定 a3 实测 184.0 / 218.4M 拍）；V 的 xing 项随物理
  列数变，用 acct_a3.w_load_ideal 逐描述符重算（108 列复现实测 108.1M 后按比例外推）。

A2（读压缩三路线在 B8 语境重算，20_read_compress 页的 a3 字节口径）：
  路线1 CTX 双区驻留 + host 折链：ctx 字节 680.25 − 191.6（容量内驻留候选）
       − 20（host 折链）→ 468.65 MB，X 按字节等比缩
  路线2 双读引擎：读通道 X+V → max(X, V)（ctx/w 各一台读引擎）
  路线3 WRAM 4 组：V → V×(55/108.1)（页口径：暴露装载 108.1→约 55M）
  路线可叠加；HP64 口径读侧只有 20.1M 拍，路线只影响 TB 口径和 DDR 字节/功耗。

资源/功耗为粗档外推（非综合数）：LUT = 固定 25k + 物理列×(列基础设施 436 +
16×PE_lut)，R3C PE_lut≈35、B8=169，锚定 R3C-96 = 120,619；BRAM = 18.5 + 每列
1 颗（R3C）/ 2 颗（B8 每物理列驻 2 逻辑列权重，深度翻倍）；功耗 = v1 实测
4,465 mW @1728DSP@200MHz（85% 在 GEMM 路径）按 DSP 数 × 核心时钟 × B8 翻转
1.4 倍外推，vectorless 误差大，只分档。H1/H2/H4 综合实测定案。

输出：b8_scan.json + 控制台 A1 选型表 / A2 路线表。纯模型，不碰 RTL。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R3C_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                       'hb_fpga_impl', '19_r3c_arch')
sys.path.insert(0, R3C_DIR)
from pe_sizing import logical_gems          # noqa: E402  (自带 a3 sys.path)
from acct_a3 import (decode, wb_cycles, DRAIN, DALIGN, ROWS,   # noqa: E402
                     load_ctx_ideal, w_load_ideal, store_cycles)

BUILD = os.path.join(os.path.dirname(R3C_DIR), '12_actv', 'a3', 'build_a3')
CAL_G = 336.44 / 319.99        # GEMM 校准（a3 口径）
COPY, ACTV, THETA = 40.3, 10.0, 40.0
X_ANCHOR, V_ANCHOR, W_ANCHOR = 184.0, 108.1, 218.4   # a3 实测（M 拍）
BYTES = dict(ctx=680.25, w=606.92, st=594.29)        # MB
HP64_RD = (BYTES['ctx'] + BYTES['w']) / 64.0         # 20.1M 拍
HP64_WR = BYTES['st'] / 64.0                         # 9.3M 拍

# PE 旋钮：接口/核心时钟（MHz）、每接口拍 MAC、pack、PE LUT/FF、每列 BRAM
PES = dict(
    R3C=dict(f_iface=198.5, f_core=198.5, mac=1, pack=1, pe_lut=35,
             bram_col=1.0, lut_note='锚定 R3C-96=120,619'),
    B8=dict(f_iface=606.43 / 2, f_core=606.43, mac=4, pack=2, pe_lut=169,
            bram_col=2.0, lut_note='cl64 实测 169 LUT/PE'),
)
COLS = [24, 32, 48, 64, 96]     # 用户点名的 {24,48,96} + 补 32/64 看可行域边界
LUT_FIXED, COL_INFRA = 25000, 436
DEV_LUT, DEV_DSP = 230400, 1728
P_BASE = 4465.0                 # mW，v1 vectorless @1728DSP@200MHz，85% GEMM
B8_SWITCH = 1.4                 # 每核心拍翻转倍数（2 乘 + 4 上下文）

# A2 路线参数（20_read_compress 页 a3 口径）
R1_CTX_MB = BYTES['ctx'] - 191.6 - 20.0    # 468.65
R3_V_FRAC = 55.0 / 108.1                   # WRAM4 组暴露装载占比


def collect():
    """一遍扫全部段：逻辑 GEMM + 三类搬运用量（供 V(C) 重算与 X/W 核对）。"""
    segs = sorted(os.listdir(os.path.join(BUILD, 'segments')))
    w_loads = []                 # op=4 b_src!=0 的 dma_len 列表
    x_ideal = w_ideal_cycles = st_ideal = 0
    macs = 0
    for s in segs:
        for line in open(os.path.join(BUILD, 'segments', s, 'seq.mem')):
            line = line.strip()
            if not line:
                continue
            d = decode(int(line, 16))
            op = d['op']
            if op in (0, 1, 2):
                mt = (d['m'] + ROWS - 1) // ROWS
                macs += mt * ROWS * d['k'] * d['b_spad']
            elif op == 4:
                n = d['dma_len']
                if d['b_src'] == 0:
                    x_ideal += load_ctx_ideal(n)
                else:
                    w_loads.append(n)
            elif op == 5:
                st_ideal += store_cycles(d['dma_len'])
    gems = logical_gems(BUILD + '/segments')
    v108 = sum(w_load_ideal(n, 108) for n in w_loads)
    return dict(segs=len(segs), gems=gems, macs=macs, w_loads=w_loads,
                x_ideal=x_ideal, v108=v108, st_ideal=st_ideal)


def gemm_recomp_M(gems, logical_c, wb_div=1, drain=DRAIN):
    """重编译口径 GEMM 拍数（pe_sizing 口径 B 原式）。

    wb_div=2 是 B8 的读出链加宽敏感性：每物理列每拍出 2 个结果（CTX B 口
    128b→256b 或等效双缓冲），wb 减半。
    drain=128 是 requant 不加倍的惩罚下界：B8 每物理列结果数翻倍，若 rq_ms
    套数/带宽不随逻辑列数加倍，RQ_SH 4→8、DRAIN 64→128。
    """
    G2 = 0.0
    for gm in gems:
        W_, k_, mt_, tr_, j0_ = gm['W'], gm['k'], gm['mt'], gm['y_tr'], gm['j0']
        for t in range((W_ + logical_c - 1) // logical_c):
            n_sub = min(W_, (t + 1) * logical_c) - t * logical_c
            wb = max(1, wb_cycles(n_sub, j0_ + t * logical_c, tr_) // wb_div)
            G2 += mt_ * max(k_ + 2, drain + DALIGN + 2 + wb)
        G2 += 4 + 16
    return G2 * CAL_G / 1e6


def V_M(w_loads, v108, c_phys):
    """LOAD_W 服务拍随物理列数重算（xing 项变宽），锚定 108 列=108.1M。"""
    if c_phys == 108:
        return V_ANCHOR
    return V_ANCHOR * sum(w_load_ideal(n, c_phys) for n in w_loads) / v108


def cell(pe_name, c, data):
    p = PES[pe_name]
    lc = c * p['pack']
    G2 = gemm_recomp_M(data['gems'], lc)
    comp = G2 + COPY + ACTV
    V = V_M(data['w_loads'], data['v108'], c)
    tb = max(comp, X_ANCHOR + V, W_ANCHOR) + THETA
    hp = max(comp, HP64_RD, HP64_WR) + THETA
    dsp = ROWS * c
    lut = LUT_FIXED + c * (COL_INFRA + ROWS * p['pe_lut'])
    pe_lut_floor = ROWS * c * p['pe_lut']        # 光 PE 核（下界）
    bram = 18.5 + c * p['bram_col']
    # 功耗粗档：动态 ∝ DSP×核心时钟×翻转，静态 ~0.65W
    p_dyn = P_BASE * 0.85 * (dsp / DEV_DSP) * (p['f_core'] / 200.0) * \
        (B8_SWITCH if pe_name == 'B8' else 1.0)
    pwr = p_dyn + P_BASE * 0.15
    util = data['macs'] / (ROWS * c * p['mac'] * G2 * 1e6) * 100
    return dict(pe=pe_name, cols_phys=c, cols_logic=lc, dsp=dsp,
                dsp_pct=dsp / DEV_DSP * 100, gemm_M=G2, comp_M=comp,
                util_pct=util, read_M=X_ANCHOR + V, V_M=V, W_M=W_ANCHOR,
                tb_M=tb, tb_s=tb / p['f_iface'],
                hp_M=hp, hp_s=hp / p['f_iface'],
                f_iface=p['f_iface'], peak_gmacs=ROWS * c * p['mac'] *
                p['f_iface'] / 1e3,
                lut_est=int(lut), lut_pct=lut / DEV_LUT * 100,
                pe_lut_floor=int(pe_lut_floor),
                bram_est=bram, pwr_est_mw=round(pwr),
                pwr_band=('低' if pwr < 5500 else '中' if pwr < 9500 else '高'))


def route_table(cfg, data):
    """A2：读压缩三路线（含叠加）在给定配置下的 TB 口径重算。"""
    p = PES[cfg['pe']]
    V = cfg['V_M']
    X1 = X_ANCHOR * R1_CTX_MB / BYTES['ctx']     # 路线1 后的 X
    V3 = V * R3_V_FRAC                            # 路线3 后的 V
    routes = [
        ('现状', X_ANCHOR + V),
        ('R1 驻留+折链', X1 + V),
        ('R2 双读引擎', max(X_ANCHOR, V)),
        ('R3 WRAM4组', X_ANCHOR + V3),
        ('R2+R3', max(X_ANCHOR, V3)),
        ('R1+R2+R3', max(X1, V3)),
    ]
    out = []
    for name, rd in routes:
        tb = max(cfg['comp_M'], rd, W_ANCHOR) + THETA
        bind = max([('计算', cfg['comp_M']), ('读', rd), ('写', W_ANCHOR)],
                   key=lambda t: t[1])[0]
        out.append(dict(route=name, read_M=rd, tb_M=tb,
                        tb_s=tb / p['f_iface'], bind=bind))
    return out


def main():
    data = collect()
    print('段 %d　逻辑 GEMM %d　MAC %.1f G/帧' %
          (data['segs'], len(data['gems']), data['macs'] / 1e9))
    # 口径自检：X/W 用全量理想×校准复现实测锚点；V 锚定实测 108.1M（pf/驻留
    # 折抵后的暴露口径，acct 暴露理想 50.05M vs 全量字节理想 82.3M），列数
    # 敏感只按 xing 项在全量字节上缩放（上界），不逐列重算 pf。
    print('口径自检：X 理想×2.1388 = %.1fM（实测 184.0）　'
          'W 理想×1.1764 = %.1fM（实测 218.4）　'
          'V 全量理想×2.1307 = %.1fM（锚点 108.1 = 暴露口径，pf/驻留折 %.0f%%）'
          % (data['x_ideal'] * 2.1388 / 1e6, data['st_ideal'] * 1.1764 / 1e6,
             data['v108'] * 2.1307 / 1e6,
             100 - V_ANCHOR / (data['v108'] * 2.1307 / 1e6) * 100))

    cells = [cell(pe, c, data) for pe in ['R3C', 'B8'] for c in COLS]
    base = next(c for c in cells if c['pe'] == 'R3C' and c['cols_phys'] == 96)
    for c in cells:
        c['hp_vs_base_pct'] = (c['hp_s'] / base['hp_s'] - 1) * 100
        c['tb_vs_base_pct'] = (c['tb_s'] / base['tb_s'] - 1) * 100

    print('\n== A1 选型表（帧拍/秒 @各自接口时钟；基线=R3C-96 HP64 %.2fs）=='
          % base['hp_s'])
    hdr = ('%-4s %-4s %-5s %-5s %-6s %-7s %-7s %-9s %-8s %-9s %-8s '
           '%-8s %-8s %-8s %-6s %-6s %s')
    print(hdr % ('PE', '列', '逻辑', 'DSP%', 'GEMM拍', '利用率', '读拍',
                 'TB秒', 'HP64秒', 'vs基线', '峰值GM/s', 'LUT估%', 'PE核LUT',
                 'BRAM', '功耗mW', '档', 'TB瓶颈'))
    for c in cells:
        bind = max([('comp', c['comp_M']), ('read', c['read_M']),
                    ('W', c['W_M'])], key=lambda t: t[1])[0]
        print('%-4s %-4d %-5d %-5.1f %-6d %-7.1f %-7.1f %-9.3f %-8.3f '
              '%-9.1f%% %-8.0f %-8.1f %-8d %-6.1f %-6d %-5s %s' % (
                  c['pe'], c['cols_phys'], c['cols_logic'], c['dsp_pct'],
                  round(c['gemm_M']), c['util_pct'], c['read_M'],
                  c['tb_s'], c['hp_s'], c['hp_vs_base_pct'],
                  c['peak_gmacs'], c['lut_pct'], c['pe_lut_floor'],
                  c['bram_est'], c['pwr_est_mw'], c['pwr_band'], bind))

    print('\n== A2 读压缩三路线（TB 口径，秒 @接口时钟）==')
    a2 = {}
    for tag, pe, c in [('R3C-96', 'R3C', 96), ('B8-24', 'B8', 24),
                       ('B8-48', 'B8', 48), ('B8-64', 'B8', 64),
                       ('B8-96', 'B8', 96)]:
        cfg = next(x for x in cells if x['pe'] == pe and x['cols_phys'] == c)
        a2[tag] = dict(cfg=cfg, routes=route_table(cfg, data))
        print('-- %s（comp %.0fM拍 / V %.1fM拍）' % (tag, cfg['comp_M'],
                                                    cfg['V_M']))
        for r in a2[tag]['routes']:
            print('   %-14s 读 %.1fM拍 → TB %.1fM拍 = %.3fs（瓶颈 %s）'
                  % (r['route'], r['read_M'], r['tb_M'], r['tb_s'],
                     r['bind']))

    # B8 读出链 2× 宽敏感性（H1/H3 的关键 RTL 自由度）
    print('\n== B8 敏感性：读出链 2× 宽（wb 减半）与 requant 不加倍（DRAIN 128）==')
    b8_wb2 = []
    for c in COLS:
        g0 = gemm_recomp_M(data['gems'], c * 2)
        G2 = gemm_recomp_M(data['gems'], c * 2, wb_div=2)
        G2nq = gemm_recomp_M(data['gems'], c * 2, drain=2 * DRAIN)
        comp = G2 + COPY + ACTV
        hp_s = (max(comp, HP64_RD, HP64_WR) + THETA) / PES['B8']['f_iface']
        util = data['macs'] / (ROWS * c * 4 * G2 * 1e6) * 100
        b8_wb2.append(dict(cols_phys=c, gemm_M=G2, comp_M=comp,
                           util_pct=util, hp_s=hp_s,
                           gemm_drain128_M=G2nq))
        print('   B8-%-3d 读出1×: %3.0fM拍　读出2×: %3.0fM拍（%.3fs）　'
              'requant 不加倍: %3.0fM拍（+%4.1f%%）'
              % (c, g0, G2, hp_s, G2nq, (G2nq / g0 - 1) * 100))

    out = dict(meta=dict(date='2026-09-08', build='build_a3',
                         segs=data['segs'], gems=len(data['gems']),
                         macs_G=data['macs'] / 1e9,
                         budget_s=2.13,
                         note='B8 旋钮 + 读压缩三路线，纯模型'),
               pe_params=PES, cols=COLS, cells=cells, b8_wb2=b8_wb2, a2=a2)
    with open(os.path.join(HERE, 'b8_scan.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n写出 b8_scan.json')


if __name__ == '__main__':
    main()
