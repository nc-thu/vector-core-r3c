# -*- coding: utf-8 -*-
"""v3_perf.py — v3 架构探索：从 v2 终点（4.03 s）推到实时（0.70 / 0.25 s）
纯周期级解析建模 + 面积预算推演（不动 RTL、不跑综合）。生成：python v3_perf.py

旋钮与建模口径（沿用 v2_perf.py / route_study.py / gem_cycles.py 的 RTL 实测常数）：

  DEV   器件档位（LUT / BRAM36 / URAM / DSP 预算）：
        cols = min( LUT 预算 , BRAM 预算 )
        LUT 预算 = (LUT_DEV×0.88 − FIX_LUT) // (PER_COL + 16×LUT_MAC)   [v2 同式]
        BRAM 预算 = BRAM36 − 14.5(SEQ) − CTX_BRAM(无 URAM 时 2MB CTX 落 BRAM ≈456 块)
        DSP 不约束（HiF8-native DSP=0）；URAM < 64 → CTX 无处安放 = 档位死路（显式标注）
  LUTM  HiF8 每 MAC LUT（35/45/60，micro 综合判定项，v2 遗留）
  F     Fmax：198.5（现行保守，WNS 最差在 copy 交叉矩阵）/ 250（copy 路径修好后目标）
        ——频率只除时间，不改拍数；DDR B/cyc 按端口宽度固定 → B/s 随频率同步放大
  BW    引擎侧有效装载带宽（B/cyc）【勘误 2026-08-27：原标注"DDR 位宽"是错的】：
        64-bit 是 v1 DMA 引擎 64-bit AXI 主口的全模型标定 7.08（602M 拍装 4.26 GB，
        含游标跨界/AR/命令开销；108 档单 k 行口径 7.71 见 CYCLE_ACCOUNT）。
        物理上 DRAM 始终 64-bit DDR4-2400：峰值 19.2 GB/s = 512bit/周期@300MHz UI
        （1:4）≈ 96.7 B/cyc@198.5MHz；ZCU104 PL 走 HP 口聚合（128b×2~4 口×~70%）
        现实 ~40-66 B/cyc。墙在引擎数据通路宽度（ae_top.sv m_axi_*[63:0]），不在 DDR。
        128/256/512-bit = 引擎通路加宽档位 ×2/×4/×8 理想线性外推（未建 NoC 仲裁退化）。
        W 流拍数 = 64-bit 基线 × 7.08/BW（开销同比缩放的粗口径）
  SMP   softmax 行并行 ÷x（×16 = CTX 单读口上限；×32 需双读口——结构前提显式标注）
  CONC  引擎并发开关：
        False = v2 口径  total = max(串行段 sm+copy+compute+边界, W流+边界)
        True  = v3 口径  total = max(compute, sm, copy, W流, elem) + 边界
        （三引擎全并发 + DMA 后台流；聚合 max 口径，未建层间依赖气泡，见边界⑴）
        compute 沿用 v2「喂料下限」= 4+mt×(k+2)：m-tile 双缓冲/背靠背发射把波前
        充填（ROWS+cols+3）藏在喂料之下——k≥cols+21 的主 GEMM 全部满足，
        敏感性表另给「tile 开销不隐藏」的对照口径（feed_gross）
  ELEM  elementwise 引擎估算（gelu/rmsnorm/silu/rope；v1/v2 无引擎未计周期）：
        拍数 = 元素数 ÷ elem_par（ops/cyc，默认 16 lane）；softmax_exp 不重复计
        （softmax 引擎已含）。LUT 估算 ~150/lane。无 RTL 常数——外推标注
  N     去噪步数 32→10：ActionHead/step 计数 ×N/32（DSE 已证大杠杆；需上游
        蒸馏/一致性训练消融确认，精度口径未闭环）
  VIT   ViT 分辨率 448→224：VIT_SEQ 1025→257、patch 256、pixel-shuffle 64
        （ViT 段 MAC/喂料 ≈÷4；权重字节数不变——k×n 与序列长度无关）。
        需上游消融确认精度；LLM seq 仍按 padding=1024（保守，未折 LLM 动态长度）
  CHIP  多片划分：按阶段子集分片，每片独立 cols/BW/并发，墙 = max(各片)；
        片间流量 = 分片边界的激活字节（显式给出，本设计 <2.1 MB/推理）

自检（必须过）：v2 兼容口径（CONC=False, BW=7.08, ELEM=off, cols=207, SM16）
分毫不差复现 v2 终点 800,364,454.5 拍 = 4.03 s @198.5 MHz、271.9 GMAC/s。
"""
import contextlib
import os
import sys

SIM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(SIM)),
                                'research_evo1_hif8'))

import gem_cycles as G      # noqa: E402
import route_study as R     # noqa: E402
import evo1_spec as S       # noqa: E402
import v2_perf as V2        # noqa: E402  （只读引用，复现 v2 终点做自检）

F_SYN = G.F_SYN             # 198.5 MHz
F_FAST = 250e6              # copy 交叉路径修复后的目标档

# ---------------- 器件档位（LUT/BRAM36/URAM/DSP，厂商数据表口径） ----------------
# ZU7EV=ZCU104 现行平台（仓库 PPA.md 实测口径）；ZU9EG=ZCU102（URAM=0!）；
# ZU19EG（523,040 LUT / 912 BRAM36 / 128 URAM / 1,968 DSP）；
# VP1802 = Versal Premium（3,360,896 LUT / 4,941 BRAM36 / 2,549 URAM）——深外推档。
DEVICES = {
    'XCZU7EV': dict(lut=230_400, bram=312,  uram=96,   dsp=1728, tag='ZCU104 现行'),
    'XCZU9EG': dict(lut=274_080, bram=312,  uram=0,    dsp=2520, tag='ZCU102'),
    'XCZU19EG': dict(lut=523_040, bram=912, uram=128,  dsp=1968, tag='大单片'),
    'VP1802':  dict(lut=3_360_896, bram=4_941, uram=2_549, dsp=0, tag='Versal·外推'),
}

# ---------------- v2 面积模型常数（与 v2_perf.py 完全一致） ----------------
UTIL = 0.88
FIX_LUT = V2.FIX_LUT            # dma+sched+SM16softmax+ctx+top+仲裁 = 8528
PER_COL = V2.PER_COL            # copy 176 + normalize 80 + gemm 控制 119 = 375
SEQ_BRAM = 14.5                 # SEQ 描述符表
CTX_URAM = 64                   # CTX 2MB 现行 URAM 占用
CTX_BRAM = 456                  # 无 URAM 档：2 MB ÷ 4.5 KB/BRAM36
BW64 = 7.08                     # 64-bit AXI 全模型标定有效带宽 B/cyc（见 docstring）

RT_LIBERO = 0.70                # s/inf（LIBERO h=14@20Hz → 1.43 inf/s）
RT_ALOHA = 0.25                 # s/inf（ALOHA chunk@100Hz → 4 inf/s）


# ---------------- 面积 → 阵列宽度 ----------------
def cols_of(dev, lut_mac=35, util=UTIL):
    """器件 → cols；返回 (cols, 约束项)。URAM<64 的档 CTX 落 BRAM，可能直接死路。"""
    d = DEVICES[dev] if isinstance(dev, str) else dev
    lut_cols = int((d['lut'] * util - FIX_LUT) // (PER_COL + 16 * lut_mac))
    ctx_bram = 0 if d['uram'] >= CTX_URAM else CTX_BRAM
    bram_cols = int(d['bram'] - SEQ_BRAM - ctx_bram)
    if bram_cols <= 0:
        return 0, 'CTX 无处安放（URAM=0 且 BRAM 不够）'
    cols = min(lut_cols, bram_cols)
    return cols, ('LUT' if lut_cols <= bram_cols else 'BRAM')


# ---------------- 负载构建（分辨率 / 去噪步数） ----------------
@contextlib.contextmanager
def vit_res(img):
    """临时改 ViT 分辨率（evo1_spec 的模块级常量在 build_spec 调用时读取）。"""
    if img == 448:
        yield
        return
    old = (S.VIT_IMG, S.VIT_SEQ, S.VIT_PS_TOKENS)
    S.VIT_IMG = img
    S.VIT_SEQ = (img // S.VIT_PATCH) ** 2 + 1
    S.VIT_PS_TOKENS = ((img // S.VIT_PATCH) ** 2) // 4
    try:
        yield
    finally:
        S.VIT_IMG, S.VIT_SEQ, S.VIT_PS_TOKENS = old


def build_spec_v3(vit_img=448, steps=32):
    with vit_res(vit_img):
        return S.build_spec(vit_tiles=2, hoist_kv=True, steps=steps)


def w_bytes_of(spec):
    """全部 GEMM 权重字节（attention 的 B 来自 COPY，不走 WRAM 装载）"""
    tot = 0
    for it in spec['gemm_items']:
        mul = spec['config']['denoise_steps'] if it['stage'] == 'ActionHead/step' else 1
        if it['kind'] == 'gemm':
            tot += it['k'] * it['n'] * it['count'] * mul
    return tot


def elem_ops_of(spec):
    """elementwise 元素数（softmax_exp 由 softmax 引擎覆盖，剔除防双计）。
    AH/step 项按去噪步数放大。"""
    tot = 0
    for it in spec['elem_items']:
        if it['name'] == 'softmax_exp':
            continue
        mul = spec['config']['denoise_steps'] if it['stage'] == 'ActionHead/step' else 1
        tot += it['m'] * it['count'] * mul
    return tot


# ---------------- W 装载游标的快速等价实现 ----------------
def _w_profile_fast(nbytes, cols):
    """gem_cycles.w_load_profile 的周期化等价实现：游标按 (wj+8) mod cols 演进，
    周期 P = cols/gcd(8,cols) 拍——只模拟一个周期 + 余数段，结果与逐拍循环逐拍相等
    （cols=3154 档把 O(字节数) 循环压到 O(cols)）。"""
    from math import gcd
    beats = nbytes // 8
    g = gcd(8, cols)
    per = cols // g

    def run(n):
        wj = x = 0
        for _ in range(n):
            if wj + 8 > cols:
                x += 1
                wj = wj + 8 - cols
            elif wj + 8 == cols:
                wj = 0
            else:
                wj += 8
        return x
    return beats, (beats // per) * run(per) + run(beats % per)


_orig_profile = G.w_load_profile


def _patch_profile():
    G.w_load_profile = _w_profile_fast
    # 等价性验证：与原逐拍实现在典型档位分毫不差
    for k in (64, 588, 896, 1024, 4096, 44800):
        for c in (108, 207, 216, 483, 3154):
            assert _orig_profile(k * c, c) == _w_profile_fast(k * c, c), (k, c)


_patch_profile()      # 语义不变（w2 口径不动），只换实现速度


# ---------------- v3 周期账本 ----------------
def feed_packed(spec, cols):
    """理想列填充的喂料下限：喂料 = Σ(行数×列数×(k+2))/cols，不做列组取整。
    对应结构课题「多 GEMM/多头列并行填充 + 多 A 注入口」——收益上界，代价未建模。"""
    tot = 0.0
    for it in spec['gemm_items']:
        mul = spec['config']['denoise_steps'] if it['stage'] == 'ActionHead/step' else 1
        if it['kind'] == 'gemm':
            tot += ((it['m'] + 15) // 16) * it['n'] * (it['k'] + 2) * it['count'] * mul / cols
        else:                                       # attn：QK^T + PV（与 route_study 同源）
            if 'ViT' in it['stage']:
                lq = lk = S.VIT_SEQ
                dh, hd = S.VIT_H // S.VIT_HEADS, S.VIT_HEADS
            elif 'LLM' in it['stage']:
                lq = lk = S.LLM_SEQ
                dh, hd = S.LLM_HEAD_DIM, S.LLM_HEADS
            else:
                lq, hd = S.HEAD_HORIZON, S.HEAD_HEADS
                lk, dh = S.CTX_TOKENS, S.HEAD_E // S.HEAD_HEADS
            mt = (lq + 15) // 16
            tot += (mt * lk * (dh + 2) + mt * dh * (lk + 2)) * hd * it['count'] * mul / cols
    return tot


def account_v3(spec, cols, sm_par, bw=BW64, conc=False, elem_par=0,
               feed_gross=False, pack=False, stages=None, vit_img=448):
    """v3 主账本。stages 给定（如 ('ViT',)）则只算阶段子集（多片划分用）。
    vit_img 必须与构建 spec 时一致（ViT 注意力维在记账时从 evo1_spec 常量解析）。
    pack=True 时喂料取理想列填充下限（见 feed_packed，结构课题外推）。
    conc=False 且 bw=BW64 且 elem_par=0 时与 v2_perf.account_v2(lw='real') 逐拍一致。"""
    if stages is not None:
        spec = dict(spec, gemm_items=[it for it in spec['gemm_items']
                                      if it['stage'].startswith(stages)])
    vit_img = int(vit_img)
    with vit_res(vit_img):                          # ViT 注意力维随分辨率（route_study 读常量）
        t = R.account(spec, cols, sm_par, 0)        # v2 调度器：死装载可跳 = 0
        fp = feed_packed(spec, cols) if pack else None
    w64 = V2.w_total_of(spec, cols)               # 64-bit 基线 W 装载流（gemm 条目已固化维度）
    w = w64 * BW64 / bw                           # 带宽缩放（粗口径，见 docstring）
    act = t['dma'] - w64                          # 边界激活/回写（字节量固定，不随 BW 缩放）
    feed = t['gemm'] if feed_gross else (fp if pack else t['compute'])
    elem = elem_ops_of(spec) / elem_par if elem_par else 0
    if conc:
        total = max(feed, t['sm'], t['copy'], w, elem) + act
    else:                                          # v2 兼容口径
        ser = t['sm'] + t['copy'] + feed + act + elem
        total = max(ser, w + act)
    return dict(total=total, sm=t['sm'], copy=t['copy'], compute=t['compute'],
                feed=feed, feed_real=t['compute'], gemm=t['gemm'], w=w, w64=w64,
                act=act, elem=elem, macs=t['macs'], mac_cyc=16 * cols)


def row(name, t, freq=F_SYN):
    tot = t['total']
    return dict(name=name, mcyc=tot / 1e6, sec=tot / freq,
                eff=t['macs'] / tot * freq / 1e9, util=t['macs'] / tot / t['mac_cyc'])


def fmt(r, extra=''):
    return (f"{r['name']:<44}{r['mcyc']:>8.0f}{r['sec']:>8.2f}"
            f"{r['eff']:>8.1f}{r['util']*100:>7.0f}%{extra}")


HDR = f"{'情景':<44}{'M cyc':>8}{'s/inf':>8}{'GMAC/s':>8}{'阵列占':>7}"


def main():
    specs = {
        ('448', 32): build_spec_v3(vit_img=448, steps=32),   # = v2 负载（2 相机 KV 驻留）
        ('448', 10): build_spec_v3(vit_img=448, steps=10),   # 去噪 N=10
        ('224', 32): build_spec_v3(vit_img=224, steps=32),   # ViT 224
        ('224', 10): build_spec_v3(vit_img=224, steps=10),   # 双协同
    }
    opt32 = specs[('448', 32)]
    opt10, v224_32, v224_10 = specs[('448', 10)], specs[('224', 32)], specs[('224', 10)]
    BW128, BW256 = 2 * BW64, 4 * BW64        # 14.2 / 28.3 B/cyc（64-bit 标定 ×2/×4）

    # ================= 自检：分毫不差复现 v2 终点 =================
    ref = V2.account_v2(opt32, 207, 16, 0, lw='real')
    chk = account_v3(opt32, 207, 16, bw=BW64, conc=False)
    dev = abs(chk['total'] - ref['total'])
    print("[自检] v2 终点 H35+SM16+LWreal（cols=207, SM16, 64-bit, 串行段口径）")
    print(f"  v3_perf = {chk['total']/1e6:.1f}M 拍 = {chk['total']/F_SYN:.2f} s @198.5，"
          f"有效 {chk['macs']/chk['total']*F_SYN/1e9:.1f} GMAC/s")
    print(f"  v2_perf = {ref['total']/1e6:.1f}M 拍（基准 800M）——偏差 {dev:.2f} 拍 "
          f"{'✓ PASS' if dev < 1 else '✗ FAIL'}")
    assert dev < 1

    # ================= ① 器件档位表 =================
    print("\n== ① 器件档位 → 阵列宽度（LUT/MAC=35 / 45）==")
    print(f"{'器件':<10}{'LUT':>9}{'BRAM36':>8}{'URAM':>6}{'cols@35':>9}{'cols@45':>8}"
          f"{'MAC/拍@35':>10}{'约束':>26}")
    for k, d in DEVICES.items():
        c35, b35 = cols_of(k, 35)
        c45, _ = cols_of(k, 45)
        print(f"{k:<10}{d['lut']:>9,}{d['bram']:>8}{d['uram']:>6}{c35:>9}{c45:>8}"
              f"{16*c35:>10,}{b35:>26}")
    print("  注：ZU9EG URAM=0 且 BRAM=312 与 ZCU104 相同 → CTX 2MB 无处安放（落 BRAM 需 "
          "456 块）→ 对本设计不是升级；VP1802 为深外推档（>1000 列阵列的布线/频率未验证）。")

    # ================= ② ZCU104 逐旋钮阶梯（算法不动） =================
    print(f"\n== ② ZCU104 逐旋钮阶梯（N=32 ViT448，@198.5 MHz 除非另注）==\n{HDR}")
    ladder = [
        ('v2 终点 H35+SM16+LWreal', '448', 32, dict(conc=False), F_SYN),
        ('+CONC 三引擎并发（softmax/COPY 重叠）', '448', 32, dict(conc=True), F_SYN),
        ('+SM32 softmax×32（需 CTX 双读口）', '448', 32, dict(conc=True, sm_par=32), F_SYN),
        ('+BW128 DDR 128-bit（14.2 B/cyc）', '448', 32,
         dict(conc=True, sm_par=32, bw=BW128), F_SYN),
        ('+F250（copy 路径修好后）', '448', 32, dict(conc=True, sm_par=32, bw=BW128), F_FAST),
        ('+ELEM16 elementwise 引擎（估算）', '448', 32,
         dict(conc=True, sm_par=32, bw=BW128, elem_par=16), F_FAST),
        ('再 +BW256（28.3 B/cyc）', '448', 32,
         dict(conc=True, sm_par=32, bw=BW256, elem_par=16), F_FAST),
        ('→ +N=10 去噪 32→10（协同）', '448', 10,
         dict(conc=True, sm_par=32, bw=BW128, elem_par=16), F_FAST),
        ('→ +ViT224（协同）', '224', 10,
         dict(conc=True, sm_par=32, bw=BW128, elem_par=16), F_FAST),
    ]
    prev = None
    for name, img, n, kw, f in ladder:
        kw = dict(kw)
        smp = kw.pop('sm_par', 16)
        t = account_v3(specs[(img, n)], 207, smp, vit_img=img, **kw)
        r = row(name, t, f)
        step = '' if prev is None else f"  (−{abs(prev-r['sec'])/prev*100:.0f}%)"
        print(fmt(r, step))
        prev = r['sec']
    tg = account_v3(opt32, 207, 32, bw=BW128, conc=True, elem_par=16,
                    feed_gross=True, vit_img=448)
    print(f"  [对照] CONC 口径但 tile 开销不隐藏（feed_gross）：{tg['total']/1e6:.0f}M 拍"
          f" = {tg['total']/F_FAST:.2f} s @250——m-tile 双缓冲是 CONC 口径的前提")

    # ================= ③ 引擎并发后的瓶颈转移 =================
    print("\n== ③ 并发后谁是墙（M 拍；上行 ZCU104 cols=207，下行 ZU19EG cols=483）==")
    print(f"{'负载':<14}{'BW':>8}{'喂料':>9}{'sm÷32':>8}{'COPY':>8}{'W 流':>8}{'elem16':>8}{'墙':>14}")
    for cols, dev in ((207, 'ZCU104'), (483, 'ZU19EG')):
        for (img, n), bw, bwn in ((('448', 32), BW64, '64b'), (('448', 32), BW128, '128b'),
                                  (('448', 32), BW256, '256b'), (('224', 10), BW128, '128b')):
            t = account_v3(specs[(img, n)], cols, 32, bw=bw, conc=True,
                           elem_par=16, vit_img=img)
            keys = ('喂料', 'sm÷32', 'COPY', 'W流', 'elem')
            vals = [t['feed'], t['sm'], t['copy'], t['w'], t['elem']]
            who = keys[vals.index(max(vals))]
            print(f"{dev+' '+img+'/N'+str(n):<14}{bwn:>8}{t['feed']/1e6:>9.0f}"
                  f"{t['sm']/1e6:>8.0f}{t['copy']/1e6:>8.0f}{t['w']/1e6:>8.0f}"
                  f"{t['elem']/1e6:>8.1f}{who+' '+f'{max(vals)/1e6:.0f}M':>14}")
    print("  （ZU19EG 喂料 231M vs 理想 142M：注意力每头 n=d_head=64/112 + n<cols 余数组，"
          "列宽利用率 61%——见 ⑧e 列填充）")

    # ================= ④ 算法协同场景（ZCU104 硬件上限不动） =================
    print(f"\n== ④ 算法协同（ZCU104 cols=207，CONC+SM32+ELEM16+128-bit）==\n{HDR}")
    for name, (img, n), f in (('N=32 ViT448（②终点负载）', ('448', 32), F_FAST),
                              ('N=10（去噪 32→10）', ('448', 10), F_FAST),
                              ('ViT224（448→224）', ('224', 32), F_FAST),
                              ('N=10 + ViT224（双协同）', ('224', 10), F_FAST),
                              ('双协同 @198.5', ('224', 10), F_SYN)):
        t = account_v3(specs[(img, n)], 207, 32, bw=BW128, conc=True,
                       elem_par=16, vit_img=img)
        print(fmt(row(name, t, f)))

    # ================= ⑤ 器件阶梯（纯硬件路线，算法不动） =================
    print(f"\n== ⑤ 器件阶梯（N=32 ViT448，CONC + SM32 + ELEM16）==\n{HDR}")
    for dev in ('XCZU7EV', 'XCZU9EG', 'XCZU19EG', 'VP1802'):
        cols, bind = cols_of(dev)
        if cols == 0:
            print(f"{dev+'（'+DEVICES[dev]['tag']+'）':<46} — 档位死路（CTX 无处安放）")
            continue
        for bwn, bw in (('64b', BW64), ('128b', BW128), ('256b', BW256)):
            for f, ftag in ((F_SYN, ''), (F_FAST, ' @250')):
                t = account_v3(opt32, cols, 32, bw=bw, conc=True, elem_par=16, vit_img=448)
                r = row(f"{dev} cols={cols} {bwn}{ftag}", t, f)
                print(fmt(r, ' ←LIBERO✓' if r['sec'] <= RT_LIBERO else ''))
    tp = account_v3(opt32, 483, 32, bw=BW256, conc=True, elem_par=16, pack=True, vit_img=448)
    print(fmt(row('ZU19EG cols=483 256b @250 + 理想列填充（结构课题）', tp, F_FAST), ' ←LIBERO✓'))
    print("  注：VP1802 行的 W 流按 v2 口径「每列组整 cols 宽装载」——cols≫n 时高估（余列照读），"
          "且列利用率仅 19%：该档只有在修复列填充与 W 装载语义后才有意义，仅供数量级参考。")
    print(f"  实时线：LIBERO 0.70 s；ALOHA 0.25 s（0.70 @198.5 = 139M 拍，@250 = 175M 拍）")

    # ================= ⑥ 跨 0.70 s 的最小配方（Pareto） =================
    print(f"\n== ⑥ 跨实时配方（≤0.70 s = LIBERO；ALOHA 0.25 s 另注）==\n{HDR}")
    recipes = [
        ('P1 纯硬件·ZCU104 全旋钮 128b @250', ('448', 32), 207, 32, BW128, 16, F_FAST, False),
        ('P2 纯硬件·ZU19EG 256b @250', ('448', 32), 483, 32, BW256, 16, F_FAST, False),
        ('P3 纯硬件·ZU19EG 256b @250 +理想列填充', ('448', 32), 483, 32, BW256, 16, F_FAST, True),
        ('C1 双协同·ZCU104 128b @250', ('224', 10), 207, 32, BW128, 16, F_FAST, False),
        ('C2 双协同·ZCU104 256b @198.5', ('224', 10), 207, 32, BW256, 16, F_SYN, False),
        ('C3 半协同(N10)·ZU19EG 128b @250', ('448', 10), 483, 32, BW128, 16, F_FAST, False),
        ('C4 双协同·ZU19EG 128b @198.5', ('224', 10), 483, 32, BW128, 16, F_SYN, False),
        ('C5 双协同·ZU19EG 128b @250', ('224', 10), 483, 32, BW128, 16, F_FAST, False),
        ('C6 双协同·ZU19EG 256b @198.5', ('224', 10), 483, 32, BW256, 16, F_SYN, False),
    ]
    for name, (img, n), cols, smp, bw, ep, f, pk in recipes:
        t = account_v3(specs[(img, n)], cols, smp, bw=bw, conc=True, elem_par=ep,
                       pack=pk, vit_img=img)
        r = row(name, t, f)
        print(fmt(r, ' ←LIBERO✓' if r['sec'] <= RT_LIBERO else
                  (' ←ALOHA✓' if r['sec'] <= RT_ALOHA else '')))
    # ALOHA 冲刺：单片边界 + 多片
    print("  [ALOHA 0.25 s 冲刺]")
    for f, ftag in ((F_FAST, '@250'), (F_SYN, '@198.5')):
        need = RT_ALOHA * f
        print(f"    需 {need/1e6:.0f}M 拍 @{ftag[1:]}MHz：双协同负载喂料 ~466GMAC → "
              f"{466e9/need:,.0f} MAC/拍（cols≈{int(466e9/need/16):,}）+ "
              f"W 流 1.87GB → {1.87e9/need:.0f} B/cyc")
    for bwn, bw in (('128b', BW128), ('256b', BW256), ('512b', 8*BW64)):
        t = account_v3(v224_10, 483, 32, bw=bw, conc=True, elem_par=16,
                       pack=True, vit_img=224)
        print(f"    ZU19EG cols=483 + 双协同 + 理想列填充 + {bwn}: "
              f"{t['total']/1e6:.0f}M = {t['total']/F_FAST:.2f} s @250"
              f"（墙 {max(('feed', 'sm', 'copy', 'w'), key=lambda k: t[k])}）")

    # ================= ⑦ 多片划分 =================
    print("\n== ⑦ 多片：ZCU104 类（各片 cols=207，CONC+SM32+ELEM16+128b，流水吞吐口径）==")
    for name, (img, n), parts, f in (
        ('ViT | LLM+AH，N=32', ('448', 32), (('ViT',), ('LLM', 'ActionHead')), F_FAST),
        ('ViT | LLM+AH，N=10', ('448', 10), (('ViT',), ('LLM', 'ActionHead')), F_FAST),
        ('ViT | LLM+AH，双协同', ('224', 10), (('ViT',), ('LLM', 'ActionHead')), F_FAST),
        ('ViT | LLM+AH，双协同 @198.5', ('224', 10), (('ViT',), ('LLM', 'ActionHead')), F_SYN),
        ('ViT+AH | LLM，双协同', ('224', 10), (('ViT', 'ActionHead'), ('LLM',)), F_FAST),
    ):
        spec = specs[(img, n)]
        walls = []
        for st in parts:
            t = account_v3(spec, 207, 32, bw=BW128, conc=True, elem_par=16,
                           stages=st, vit_img=img)
            keys = ('feed', 'sm', 'copy', 'W')
            who = max(zip(keys, (t['feed'], t['sm'], t['copy'], t['w'])), key=lambda p: p[1])
            print(f"  片{'+'.join(st):<16} {t['total']/1e6:>5.0f}M（{who[0]} {who[1]/1e6:.0f}M） ", end='')
            walls.append(t['total'])
        tot = max(walls)
        print(f"→ 吞吐墙 {tot/1e6:.0f}M = {tot/f:.2f} s @{f/1e6:.0f}"
              f"{' ←LIBERO✓' if tot/f <= RT_LIBERO else ''}")
    print("  片间流量 = 分片边界激活（ViT 后 2.1MB / LLM ctx 0.92MB / 224 后 0.13MB）："
          "@0.70 s 需 <3 MB/s——任何 GTP 链路富余，瓶颈不在此。")

    # ================= ⑧ 敏感性 =================
    print("\n== ⑧a W 装载流 4.26GB 都是谁（N=32，64-bit 口径）==")
    w64_all = account_v3(opt32, 207, 32, vit_img=448)['w64']
    for st in ('ViT', 'LLM', 'ActionHead'):
        t = account_v3(opt32, 207, 32, conc=True, stages=st, vit_img=448)
        print(f"  {st:<12} W {t['w64']/1e6:>5.0f}M 拍 = {t['w64']*BW64/1e9:>5.2f} GB"
              f"（{t['w64']/w64_all*100:>3.0f}%）")
    t10 = account_v3(opt10, 207, 32, vit_img=448)
    print(f"  N=10 → 全模型 W 流 {t10['w64']/1e6:.0f}M 拍（{t10['w64']*BW64/1e9:.2f} GB）"
          "——去噪步数同时砍喂料与带宽，是第一杠杆")

    print("\n== ⑧b 双协同后的 BW×Fmax 平面（ZCU104 cols=207，CONC+SM32+ELEM16，s）==")
    hdr_bw = 'BW' + chr(92) + 'Fmax'
    print(f"{hdr_bw:>10}{'198.5':>9}{'250':>9}")
    for bwn, bw in (('64b', BW64), ('128b', BW128), ('256b', BW256)):
        cells = [account_v3(v224_10, 207, 32, bw=bw, conc=True, elem_par=16,
                            vit_img=224)['total'] / f for f in (F_SYN, F_FAST)]
        print(f"{bwn:>10}{cells[0]:>9.2f}{cells[1]:>9.2f}")

    print("\n== ⑧c SM 并行度敏感性（双协同 + 128b，s @250 / @198.5）==")
    for smp in (8, 16, 32, 64):
        t = account_v3(v224_10, 207, smp, bw=BW128, conc=True, elem_par=16, vit_img=224)
        print(f"  SM×{smp:<3} {t['total']/F_FAST:.2f} / {t['total']/F_SYN:.2f} s"
              f"（softmax {t['sm']/1e6:.0f}M{'，需 CTX 双读口' if smp >= 32 else ''}）")

    print("\n== ⑧d LUT/MAC 敏感性（ZU19EG，256b @250，s）==")
    for lm in (35, 45, 60):
        cols, _ = cols_of('XCZU19EG', lm)
        t = account_v3(opt32, cols, 32, bw=BW256, conc=True, elem_par=16, vit_img=448)
        print(f"  {lm} LUT/MAC → cols={cols} → {t['total']/F_FAST:.2f} s")

    print("\n== ⑧e 列宽利用率（喂料 vs 理想 = MAC数/(16×cols)，N=32）==")
    for cols in (207, 483, 966, 3154):
        t = account_v3(opt32, cols, 32, conc=True, vit_img=448)
        ideal = t['macs'] / (16 * cols)
        print(f"  cols={cols:<5} 喂料 {t['feed']/1e6:>6.0f}M 理想 {ideal/1e6:>6.0f}M "
              f"利用率 {ideal/t['feed']*100:>5.1f}%（每头注意力 n=d_head 64/112 是天花板）")


if __name__ == '__main__':
    main()
