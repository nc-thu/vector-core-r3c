# -*- coding: utf-8 -*-
"""holobrain_breakdown.py — HB-GD 0.2B 16×108 阵列周期账本的逐条目/分阶段拆解 + 架构优化杠杆账。

现有 holobrain_hw.py 只给总量和墙排序；本脚本把 54.31M 拍的 CONC 账拆到
阶段/条目/m-bucket，并量化每个架构杠杆能省多少，供 FPGA 硬件与架构决策。

口径出处（全部沿用现有账本，不另起炉灶）：
  - spec 加载：holobrain_hw.py:117-118 同款 —— HB.build_spec()（reference 忠实上游）
    与 HB.build_spec(hoist_kv=True)（opt 档：K/V+dist hoist），模块导入不读磁盘 JSON；
    阶段归属 matcher 同 holobrain_hw.py:132（Vision2D=Swin2D+Neck2D、
    Vision3D+PSE=Swin3D+Neck3D+PSE、Text、Fusion、ActionHead）。
  - account()/models()/hw_gemm()/hw_attn()：swiftvla_hw.py:109/155/46/51。
    compute = 喂料下限（逐 m-tile 串行，swiftvla_hw.py:46-48 → route_study.py:33
    hw_gemm_w 每列组 4 + mt*(k+2)）；ActionHead/step 条目乘 denoise_steps=10。
  - W 装载拍：route_study.py:33 hw_gemm_w 的 dma（不带 act/store 即纯 W 装载，
    = swiftvla_hw.py:135 w_total 口径）；W 端口字节 = w_cyc × BW64(7.08 B/cyc)，
    与 holobrain_hw.json hw_main 的 w_gb=m['w64']*BW64/1e9 同源（含 n 不满 108
    列的补齐与 burst/AR/CMD 开销）；spec 级 n×k×count×mul 字节另列 w_spec_MB。
  - attn 维度：holobrain_hw.py:45 hb_attn_dims_of（lq/lk/dhead/heads/causal）。
  - sm = softmax_cycles÷sm_par（SM16 现行 RTL）；copy = hw_attn 的 COPY(Kᵀ/Vᵀ)；
    elem = elem_ops÷16；act = 边界激活 DMA（t['dma'] − w64）；F_SYN=198.5 MHz。
  - n_exec：gemm 条目 = ceil(n/108)×count×mul（列组执行次数，对齐 account 的
    n_gemm）；attn 条目 = heads×(列组数+1)×count×mul（对齐 account 的 n_exec）。

验收门（脚本内 assert，全过才写 JSON）：
  ① account(108) 与 holobrain_hw.json hw_main 一致：CONC 54.309M 拍 / LWreal
     89.403M 拍 / W 0.349 GB（容差 0.5%）
  ② per_stage gmacs 加总 = 73.661 GMAC；五阶段 17.259/5.202/1.367/26.423/23.410
     GMAC（容差 0.5%，ActionHead = step×10 + once 合计）
  ③ 各阶段 feed/sm/copy/n_exec 加总 = account() 总量（容差 0.1%，本拆解与
     account 逐条同式调用，应为精确相等）
  ④ +BW128 杠杆的 W 墙 = 基线 W 墙/2（容差 2%）

生成：python -X utf8 holobrain_breakdown.py  →  holobrain_breakdown.json + 控制台七段表
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import holobrain_hw as HWH       # noqa: E402  复用其 spec 加载/维度工厂/BOUNDARY 集合
import swiftvla_hw as HW         # noqa: E402  account/models/hw_gemm/hw_attn 通用账本
import holobrain_spec as HB      # noqa: E402

F_SYN, BW64, ROWS = HWH.F_SYN, HWH.BW64, HW.ROWS    # 198.5 MHz / 7.08 B/cyc / 16 行
COLS = 108                                          # 主口径 = 现行硅片 16×108

STAGE_PREFIXES = [                                  # holobrain_hw.py:132 同款 matcher
    ("Vision2D", ("Swin2D", "Neck2D")),
    ("Vision3D+PSE", ("Swin3D", "Neck3D", "PSE")),
    ("Text", ("Text",)),
    ("Fusion", ("Fusion",)),
]
STAGE_ORDER = ["Vision2D", "Vision3D+PSE", "Text", "Fusion",
               "ActionHead/step", "ActionHead/once"]


def stage_of(it_or_stage) -> str:
    st = it_or_stage["stage"] if isinstance(it_or_stage, dict) else it_or_stage
    for key, pref in STAGE_PREFIXES:
        if st.startswith(pref):
            return key
    if st in ("ActionHead/step", "ActionHead/once"):
        return st
    raise ValueError(f"unmapped stage: {st}")


def drop_stage(spec, prefixes):
    """从 spec 里整体去掉一个阶段（gemm_items + elem_items 同步过滤）。"""
    sp = dict(spec)
    sp["gemm_items"] = [i for i in spec["gemm_items"]
                        if not i["stage"].startswith(prefixes)]
    sp["elem_items"] = [i for i in spec["elem_items"]
                        if not i["stage"].startswith(prefixes)]
    return sp


def item_rows(spec, dims, cols=COLS, sm_par=16):
    """逐条目周期账：与 swiftvla_hw.account（swiftvla_hw.py:109）完全同式调用，
    每条额外保留 feed/sm/copy/W 装载拍/n_exec/gmacs，供分阶段与排序。"""
    rows = []
    for it in spec["gemm_items"]:
        mul = spec["config"]["denoise_steps"] if it["stage"] == "ActionHead/step" else 1
        r = dict(stage=it["stage"], name=it["name"], kind=it["kind"],
                 count=it["count"], mul=mul, n=it["n"], k=it["k"],
                 feed=0, sm=0, copy=0, w_cyc=0, gemm_cyc=0, n_exec=0, gmacs=0)
        if it["kind"] == "gemm":
            # 不带 act/store → dma 即纯 W 装载拍（w_total 口径）；compute 不受 flag 影响
            g0, d_w, comp = HW.hw_gemm(it["m"], it["n"], it["k"], cols)
            r.update(m=it["m"], feed=comp * it["count"] * mul,
                     w_cyc=d_w * it["count"] * mul, gemm_cyc=g0 * it["count"] * mul,
                     n_exec=-(-it["n"] // cols) * it["count"] * mul,
                     gmacs=it["m"] * it["n"] * it["k"] * it["count"] * mul,
                     w_spec_bytes=it["k"] * it["n"] * it["count"] * mul)
        else:
            lq, lk, dh, hd, ca = dims(it)
            h = HW.hw_attn(lq, lk, dh, hd, ca, cols, sm_par)
            r.update(m=f"{lq}x{lk}", lq=lq, lk=lk, heads=hd,
                     feed=h["compute"] * it["count"] * mul,
                     sm=h["sm"] * it["count"] * mul,
                     copy=h["copy"] * it["count"] * mul,
                     gemm_cyc=h["gemm"] * it["count"] * mul,
                     n_exec=h["n_exec"] * it["count"] * mul,
                     gmacs=it["macs"] * mul, w_spec_bytes=0)
        r["stage_key"] = stage_of(r)
        rows.append(r)
    return rows


def m_bucket(m):
    """feed 按 m 分桶：m-tile=ceil(m/16)，小 m 的喂料被 pad 行和逐列组固定开销吃掉。"""
    if m <= 16:
        return "微m≤16 (t_embed1/NJ8/BERT16)"
    if m == 20:
        return "m=20 (lvl4/bi-attn)"
    if m == 49:
        return "m=49 (Swin窗口attn)"
    if m < 128:
        return "小m17-127 (bi40/dist64)"
    if m == 128:
        return "m=128 (动作token)"
    if m <= 512:
        return "中m129-512 (kv136/s4qkv196/head512)"
    return "大m≥800 (img_kv/Swin s1 s2/PSE/Fusion)"


BUCKET_ORDER = ["微m≤16 (t_embed1/NJ8/BERT16)", "m=20 (lvl4/bi-attn)",
                "m=49 (Swin窗口attn)", "小m17-127 (bi40/dist64)",
                "m=128 (动作token)", "中m129-512 (kv136/s4qkv196/head512)",
                "大m≥800 (img_kv/Swin s1 s2/PSE/Fusion)"]


# ============================================================================
def main():
    gates = []

    def chk(name, got, want, rtol):
        ok = abs(got - want) <= rtol * abs(want)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got:,.6g}"
              f" vs want {want:,.6g}（rtol {rtol:.2%}）")
        assert ok, f"验收门未过: {name} got={got} want={want}"
        gates.append(dict(name=name, got=got, want=want, rtol=rtol, pass_=ok))

    hb = HB.build_spec()                    # reference 主档（holobrain_hw.py:117 同款）
    hb_hoist = HB.build_spec(hoist_kv=True)  # opt 档（K/V+dist hoist）
    dims = HWH.hb_attn_dims_of(hb)
    dims_hoist = HWH.hb_attn_dims_of(hb_hoist)
    steps = hb["config"]["denoise_steps"]

    # ---- 108 列主口径 total（与 holobrain_hw.py ④段第一 CONC 行同配置）----
    t108 = HW.account(hb, COLS, 16, dims, HWH.HB_BOUNDARY_ACT, HWH.HB_BOUNDARY_STORE)
    m108 = HW.models(t108, hb, COLS, 16, BW64, elem_par=16)

    print("=" * 96)
    print("验收门（对 holobrain_hw.json hw_main 108 行，容差见括号）")
    chk("CONC 108 SM16 BW64 (M拍)", m108["conc"] / 1e6, 54.309228, 0.005)
    chk("LWreal 108 (M拍)", m108["lwreal"] / 1e6, 89.402504, 0.005)
    chk("W 流 (GB, 端口口径 w64×BW64)", m108["w64"] * BW64 / 1e9, 0.34921047204, 0.005)

    # ---- ① per_stage：逐阶段聚合 ----
    rows = item_rows(hb, dims)
    agg = {k: dict(gmacs=0, feed=0, sm=0, copy=0, w_cyc=0, n_exec=0, w_spec=0)
           for k in STAGE_ORDER}
    for r in rows:
        a = agg[r["stage_key"]]
        for f in ("gmacs", "feed", "sm", "copy", "w_cyc", "n_exec"):
            a[f] += r[f]
        a["w_spec"] += r["w_spec_bytes"]

    # 验收门③：分阶段加总必须精确回到 account()（同式调用，应为 0 漂移）
    chk("per_stage feed 加总 = account.compute (M拍)",
        sum(a["feed"] for a in agg.values()) / 1e6, t108["compute"] / 1e6, 0.001)
    chk("per_stage sm 加总 = account.sm (M拍)",
        sum(a["sm"] for a in agg.values()) / 1e6, t108["sm"] / 1e6, 0.001)
    chk("per_stage copy 加总 = account.copy (M拍)",
        sum(a["copy"] for a in agg.values()) / 1e6, t108["copy"] / 1e6, 0.001)
    chk("per_stage n_exec 加总 = account n_gemm+n_exec",
        sum(a["n_exec"] for a in agg.values()),
        t108["n_gemm"] + t108["n_exec"], 0.001)

    per_stage = {}
    for k in STAGE_ORDER:
        a = agg[k]
        per_stage[k] = dict(
            gmacs=a["gmacs"], gmacs_g=a["gmacs"] / 1e9,
            feed_mcyc=a["feed"] / 1e6, w_cyc_mcyc=a["w_cyc"] / 1e6,
            w_gb_port=a["w_cyc"] * BW64 / 1e9, w_mb_spec=a["w_spec"] / 1e6,
            sm_mcyc=a["sm"] / 1e6, copy_mcyc=a["copy"] / 1e6, n_exec=a["n_exec"])
    ah = dict(
        gmacs=agg["ActionHead/step"]["gmacs"] + agg["ActionHead/once"]["gmacs"],
        feed=agg["ActionHead/step"]["feed"] + agg["ActionHead/once"]["feed"],
        w_cyc=agg["ActionHead/step"]["w_cyc"] + agg["ActionHead/once"]["w_cyc"],
        sm=agg["ActionHead/step"]["sm"] + agg["ActionHead/once"]["sm"],
        copy=agg["ActionHead/step"]["copy"] + agg["ActionHead/once"]["copy"],
        n_exec=agg["ActionHead/step"]["n_exec"] + agg["ActionHead/once"]["n_exec"],
        w_spec=agg["ActionHead/step"]["w_spec"] + agg["ActionHead/once"]["w_spec"])
    per_stage["ActionHead合计"] = dict(
        gmacs_g=ah["gmacs"] / 1e9, feed_mcyc=ah["feed"] / 1e6,
        w_cyc_mcyc=ah["w_cyc"] / 1e6, w_gb_port=ah["w_cyc"] * BW64 / 1e9,
        w_mb_spec=ah["w_spec"] / 1e6, sm_mcyc=ah["sm"] / 1e6,
        copy_mcyc=ah["copy"] / 1e6, n_exec=ah["n_exec"])

    # 验收门②：各阶段 GMAC
    for k, want in (("Vision2D", 17.259282432), ("Vision3D+PSE", 5.202419712),
                    ("Text", 1.366818816), ("Fusion", 26.422542336)):
        chk(f"per_stage gmacs {k} (GMAC)", per_stage[k]["gmacs_g"], want, 0.005)
    chk("per_stage gmacs ActionHead(step+once) (GMAC)",
        per_stage["ActionHead合计"]["gmacs_g"], 23.410393088, 0.005)
    chk("per_stage gmacs 总计 (GMAC)",
        sum(per_stage[k]["gmacs_g"] for k in STAGE_ORDER), 73.661456384, 0.005)

    print("\n" + "=" * 96)
    print(f"① per_stage 分阶段账（cols={COLS} 主口径；feed=喂料下限；sm=SM16；"
          f"W=装载拍@BW64；step 条目已 ×{steps}）")
    print("=" * 96)
    print(f"{'阶段':<20}{'GMAC':>8}{'feed(M拍)':>11}{'W流(M拍)':>10}{'W(GB)':>8}"
          f"{'sm(M拍)':>9}{'copy(M拍)':>10}{'n_exec':>9}")
    for k in STAGE_ORDER + ["ActionHead合计"]:
        p = per_stage[k]
        print(f"{k:<20}{p['gmacs_g']:>8.2f}{p['feed_mcyc']:>11.2f}"
              f"{p['w_cyc_mcyc']:>10.2f}{p['w_gb_port']:>8.3f}{p['sm_mcyc']:>9.2f}"
              f"{p['copy_mcyc']:>10.2f}{p['n_exec']:>9,}")
    print(f"{'—— 合计(=account)':<20}{t108['macs']/1e9:>8.2f}{t108['compute']/1e6:>11.2f}"
          f"{m108['w64']/1e6:>10.2f}{m108['w64']*BW64/1e9:>8.3f}{t108['sm']/1e6:>9.2f}"
          f"{t108['copy']/1e6:>10.2f}{t108['n_gemm']+t108['n_exec']:>9,}")
    stp = per_stage["ActionHead/step"]
    print(f"  每去噪步（step/{steps}）：GMAC {stp['gmacs_g']/steps:.3f}"
          f"  feed {stp['feed_mcyc']/steps:.2f}M拍"
          f"（10 步合计占 feed 墙 {stp['feed_mcyc']/(t108['compute']/1e6)*100:.0f}%）"
          f"  W {stp['w_cyc_mcyc']/steps:.2f}M拍")

    # ---- ② top_items：按 feed 排序前 20 ----
    top = sorted(rows, key=lambda r: r["feed"], reverse=True)[:20]
    top_items = []
    print("\n== ② top_items（按 feed 排序前 20；attn 的 m 列 = lq×lk）==")
    print(f"{'#':>2} {'阶段/条目':<34}{'kind':<6}{'m':>8}{'n':>6}{'k':>6}{'次数':>6}"
          f"{'x':>3}{'GMAC':>8}{'feed(M拍)':>10}{'W(M拍)':>9}{'sm(M拍)':>9}{'cp(M拍)':>9}")
    for i, r in enumerate(top, 1):
        top_items.append(dict(
            stage=r["stage"], name=r["name"], kind=r["kind"], m=r["m"],
            n=r["n"], k=r["k"], count=r["count"], mul=r["mul"],
            gmacs_g=r["gmacs"] / 1e9, feed_mcyc=r["feed"] / 1e6,
            w_mcyc=r["w_cyc"] / 1e6, w_gb_port=r["w_cyc"] * BW64 / 1e9,
            sm_mcyc=r["sm"] / 1e6, copy_mcyc=r["copy"] / 1e6,
            feed_share=r["feed"] / t108["compute"]))
        print(f"{i:>2} {r['stage'] + '/' + r['name']:<34}{r['kind']:<6}{str(r['m']):>8}"
              f"{r['n']:>6}{r['k']:>6}{r['count']:>6}{r['mul']:>3}{r['gmacs']/1e9:>8.3f}"
              f"{r['feed']/1e6:>10.2f}{r['w_cyc']/1e6:>9.2f}{r['sm']/1e6:>9.2f}"
              f"{r['copy']/1e6:>9.2f}")

    # ---- ③ feed_composition：喂料墙按阶段 + 按 m 分桶 ----
    feed_tot = t108["compute"]
    by_stage = {k: dict(feed_mcyc=agg[k]["feed"] / 1e6,
                        share=agg[k]["feed"] / feed_tot) for k in STAGE_ORDER}
    buckets = {}
    for r in rows:
        mm = r["lq"] if r["kind"] == "attn" else r["m"]
        b = buckets.setdefault(m_bucket(mm), dict(feed=0, gmacs=0, n_exec=0, ms=set()))
        b["feed"] += r["feed"]
        b["gmacs"] += r["gmacs"]
        b["n_exec"] += r["n_exec"]
        b["ms"].add(mm)
    print("\n== ③ feed_composition：喂料墙 "
          f"{feed_tot/1e6:.2f}M 拍怎么来的 ==")
    print(f"{'按阶段':<22}{'feed(M拍)':>10}{'占feed':>8}      {'按 m 分桶':<36}"
          f"{'feed(M拍)':>10}{'占feed':>8}{'占MAC':>8}  m 值")
    bl = [((k), buckets[k]) for k in BUCKET_ORDER if k in buckets]
    keys6 = STAGE_ORDER + [None] * (len(bl) - len(STAGE_ORDER))
    for k1, (k2, b) in zip(keys6, bl):
        line_l = (f"{k1:<22}{by_stage[k1]['feed_mcyc']:>10.2f}"
                  f"{by_stage[k1]['share']*100:>7.0f}%      ") if k1 else " " * 47
        print(line_l + f"{k2:<36}{b['feed']/1e6:>10.2f}{b['feed']/feed_tot*100:>7.0f}%"
              f"{b['gmacs']/t108['macs']*100:>7.0f}%  {sorted(b['ms'])}")
    print(f"  每去噪步 feed = {agg['ActionHead/step']['feed']/steps/1e6:.2f}M拍"
          f" × {steps} 步 = {agg['ActionHead/step']['feed']/1e6:.2f}M拍，"
          f"占 feed 墙 {agg['ActionHead/step']['feed']/feed_tot*100:.0f}%")
    small = sum(b["feed"] for k, b in buckets.items() if k.startswith(("微m", "m=20", "m=49", "小m")))
    print(f"  小 m 合计（微m≤16 + 20 + 49 + 17-127）= {small/1e6:.2f}M拍"
          f" 占 feed 墙 {small/feed_tot*100:.0f}%，"
          f"但其 GMAC 只占 {sum(b['gmacs'] for k, b in buckets.items() if k.startswith(('微m', 'm=20', 'm=49', '小m')))/t108['macs']*100:.0f}%"
          f"（m-tile pad + 逐列组固定开销）")
    feed_composition = dict(
        feed_wall_mcyc=feed_tot / 1e6,
        by_stage=by_stage,
        by_m_bucket={k: dict(feed_mcyc=buckets[k]["feed"] / 1e6,
                             feed_share=buckets[k]["feed"] / feed_tot,
                             gmacs_share=buckets[k]["gmacs"] / t108["macs"],
                             n_exec=buckets[k]["n_exec"],
                             m_values=sorted(buckets[k]["ms"])) for k in BUCKET_ORDER
                    if k in buckets},
        per_denoise_step=dict(feed_mcyc=agg["ActionHead/step"]["feed"] / steps / 1e6,
                              share=agg["ActionHead/step"]["feed"] / feed_tot,
                              gmacs_g=agg["ActionHead/step"]["gmacs"] / steps / 1e9))

    # ---- ④ w_composition：W 流按阶段 + 可缓存性 ----
    w_tot = m108["w64"]
    print("\n== ④ w_composition：W 装载流 "
          f"{w_tot/1e6:.2f}M 拍@BW64（端口 {w_tot*BW64/1e6:.0f}MB）按阶段 ==")
    print(f"{'阶段':<22}{'W流(M拍)':>10}{'占W':>7}{'端口MB':>9}{'spec MB':>9}  可缓存性")
    wcache = {
        "Vision2D": "每 chunk 重装（视觉必须重跑）；59MB >> WRAM 442KB，跨 chunk 驻留不现实",
        "Vision3D+PSE": "同上（depth 分支每 chunk 一次）",
        "Text": "指令跨 chunk 不变 → BERT 输出可整条缓存，W/feed/sm 全免（杠杆+BERT缓存）",
        "Fusion": "每 chunk 一次；权重 29.7MB 无法驻留",
        "ActionHead/step": f"同一套权重每步重装 ×{steps}（忠实上游）→ 驻留可省 9/10，"
                           f"但每步 GEMM 权重 ~15.5MB >> WRAM 442KB → 只能挑大项（=K/V hoist）",
        "ActionHead/once": "每 chunk 一次，量小",
    }
    w_rows = []
    for k in STAGE_ORDER:
        a = agg[k]
        w_rows.append(dict(stage=k, w_cyc_mcyc=a["w_cyc"] / 1e6,
                           share=a["w_cyc"] / w_tot, w_mb_port=a["w_cyc"] * BW64 / 1e6,
                           w_mb_spec=a["w_spec"] / 1e6, cache_note=wcache[k]))
        print(f"{k:<22}{a['w_cyc']/1e6:>10.2f}{a['w_cyc']/w_tot*100:>6.0f}%"
              f"{a['w_cyc']*BW64/1e6:>9.0f}{a['w_spec']/1e6:>9.0f}  {wcache[k]}")
    print(f"{'—— 合计':<22}{w_tot/1e6:>10.2f}{'100%':>7}"
          f"{w_tot*BW64/1e6:>9.0f}{HB.w_bytes_of(hb)/1e6:>9.0f}"
          f"  （端口 MB 含 n<108 列组补齐+burst/AR/CMD 开销 → 略大于 spec MB）")
    w_composition = dict(
        w_wall_mcyc=w_tot / 1e6, w_gb_port=w_tot * BW64 / 1e9,
        by_stage=w_rows,
        cacheables=dict(
            bert_text=dict(w_mcyc=agg["Text"]["w_cyc"] / 1e6,
                           note="指令不变场景整阶段免跑：W/feed/sm 全免"),
            action_step_reload=dict(w_mcyc=agg["ActionHead/step"]["w_cyc"] / 1e6,
                                    saveable_9of10_mcyc=agg["ActionHead/step"]["w_cyc"]
                                    * (steps - 1) / steps / 1e6,
                                    note="每步同一套权重重装 ×10；全驻留需 ~15.5MB/步"
                                         " >> WRAM 442KB，hoist 挑 img_kv/txt_kv/dist 大项")))

    # ---- ⑤ engine_occupancy：CONC 是 max+act 结构，各引擎占用率 ----
    act = t108["dma"] - m108["w64"]
    conc = m108["conc"]
    eng = [("compute/feed(阵列喂数)", t108["compute"]),
           ("W 装载流@BW64", m108["w"]),
           ("sm÷16", t108["sm"]),
           ("copy(Kᵀ/Vᵀ重排)", t108["copy"]),
           ("elem÷16", m108["elem"]),
           ("act(边界激活DMA)", act)]
    print("\n== ⑤ engine_occupancy（108 CONC 总 "
          f"{conc/1e6:.2f}M 拍；max+act 并发结构，占用率之和 >100%）==")
    print(f"{'引擎':<24}{'拍数(M)':>10}{'占CONC':>9}")
    engines = {}
    for nm, v in eng:
        engines[nm] = dict(mcyc=v / 1e6, pct=v / conc * 100)
        print(f"{nm:<24}{v/1e6:>10.2f}{v/conc*100:>8.1f}%")
    print(f"{'合计(占用率非分解)':<24}{sum(v for _, v in eng)/1e6:>10.2f}"
          f"{sum(v for _, v in eng)/conc*100:>8.1f}%  ← 不是时延分解，是各引擎忙时占比")
    engine_occupancy = dict(
        conc_mcyc=conc / 1e6, engines=engines,
        note="CONC = max(compute, sm, copy, W, elem) + act（swiftvla_hw.py:164）；"
             "各引擎 % 之和 >100% 属预期——报的是占用率，不是加和分解。"
             "基线不含 drain52（hw_main 108 CONC 行同口径）。")

    # ---- ⑥ levers：架构优化杠杆账（都报 108 列 CONC，与基线对比）----
    def lever(name, spec, d, sm_par=16, bw=BW64, note=""):
        t = HW.account(spec, COLS, sm_par, d, HWH.HB_BOUNDARY_ACT, HWH.HB_BOUNDARY_STORE)
        m = HW.models(t, spec, COLS, 16, bw, elem_par=16)
        walls = dict(feed=t["compute"] / 1e6, sm=t["sm"] / 1e6,
                     copy=t["copy"] / 1e6, W=m["w"] / 1e6, elem=m["elem"] / 1e6)
        who = max(walls, key=walls.get)
        return dict(name=name, mcyc=m["conc"] / 1e6, sec_1985=m["conc"] / F_SYN,
                    util=t["macs"] / m["conc"] / (16 * COLS),
                    delta_mcyc=(conc - m["conc"]) / 1e6,
                    delta_pct=(conc - m["conc"]) / conc * 100,
                    wall=who, walls=walls, note=note)

    hb_notext = drop_stage(hb, ("Text",))
    hb_hoist_notext = drop_stage(hb_hoist, ("Text",))
    levers = [
        lever("baseline SM16 BW64（现行硅片，忠实上游）", hb, dims,
              note="K/V 每步重算 + 动作专家权重每步重装 ×10（hw 账本 reference 档）"),
        lever("+BERT缓存（Text 阶段整条免跑）", hb_notext, dims,
              note="指令跨 chunk 不变 → 缓存 BERT 文本特征；1.37 GMAC / 12.4M拍W / Text sm 全免"),
        lever("+K/V hoist（spec_opt 档）", hb_hoist, dims_hoist,
              note="img_kv/txt_kv/temp_joint_dist 提出去噪循环，×10→×1（数学等价，纯软件）"),
        lever("+BW128（W 引擎口加宽）", hb, dims, bw=2 * BW64,
              note="W 墙 49.3→24.7M，但 feed 54.2M 仍是墙 → CONC 不动"),
        lever("+SM32（softmax 并行×2）", hb, dims, sm_par=32,
              note="sm 18.6→9.3M，远低于 feed → CONC 不动"),
        lever("组合A: BERT缓存 + hoist（零硬件改动）", hb_hoist_notext, dims_hoist,
              note="两个软件杠杆全上：feed 与 W 同步缩"),
        lever("组合B: A + BW128（DMA 加宽）", hb_hoist_notext, dims_hoist, bw=2 * BW64,
              note="W 墙再减半；feed 是否仍为墙见 walls"),
        lever("组合C: B + SM32", hb_hoist_notext, dims_hoist, sm_par=32, bw=2 * BW64,
              note="硬件三件套全上（108 列内）"),
    ]
    chk("+BW128 的 W 墙 = 基线 W 墙/2 (M拍)", levers[3]["walls"]["W"],
        levers[0]["walls"]["W"] / 2, 0.02)

    print("\n== ⑥ levers 架构优化杠杆（都报 108 列 CONC；drain52 未含，与 hw_main 108 行同口径）==")
    print(f"{'杠杆':<40}{'M拍':>8}{'s@198.5':>9}{'util':>7}{'ΔM拍':>8}{'Δ%':>7}"
          f"{'新墙':>16}")
    for lv in levers:
        wall_txt = "%s %.1fM" % (lv["wall"], lv["walls"][lv["wall"]])
        print(f"{lv['name']:<40}{lv['mcyc']:>8.2f}{lv['sec_1985']:>9.3f}"
              f"{lv['util']*100:>6.1f}%{lv['delta_mcyc']:>8.2f}{lv['delta_pct']:>6.1f}%"
              f"{wall_txt:>16}")
    print(f"{'配置':<40}{'feed':>8}{'sm':>8}{'copy':>8}{'W':>8}{'elem':>8}   新墙")
    for lv in levers:
        w = lv["walls"]
        print(f"{lv['name']:<40}{w['feed']:>8.1f}{w['sm']:>8.1f}{w['copy']:>8.1f}"
              f"{w['W']:>8.1f}{w['elem']:>8.1f}   {lv['wall']}")

    # ---- ⑦ configs_ref：216/BW128 参照行（抄 holobrain_hw.json hw_main）----
    configs_ref = []
    hwjson = HERE / "holobrain_hw.json"
    if hwjson.exists():
        for r in json.loads(hwjson.read_text(encoding="utf-8"))["hw_main"]:
            if "216" in r["name"]:
                configs_ref.append(dict(name=r["name"], mcyc=r["mcyc"],
                                        sec_1985=r["sec_1985"], util=r["util"],
                                        w_gb=r["w_gb"]))
    print("\n== ⑦ configs_ref（216 列参照，抄 holobrain_hw.json hw_main）==")
    print(f"{'配置':<46}{'M拍':>9}{'s@198.5':>9}{'util':>7}{'W(GB)':>8}")
    for r in configs_ref:
        print(f"{r['name']:<46}{r['mcyc']:>9.1f}{r['sec_1985']:>9.3f}"
              f"{r['util']*100:>6.1f}%{r['w_gb']:>8.3f}")

    out = dict(per_stage=per_stage, top_items=top_items,
               feed_composition=feed_composition, w_composition=w_composition,
               engine_occupancy=engine_occupancy, levers=levers,
               configs_ref=configs_ref, gates=gates,
               meta=dict(cols=COLS, sm_par=16, bw=HW.BW64, f_syn=F_SYN,
                         drain52=False,
                         n_items=len(rows), denoise_steps=steps))
    (HERE / "holobrain_breakdown.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n全部门验收 PASS（{len(gates)} 条）→ holobrain_breakdown.json")


if __name__ == "__main__":
    main()
