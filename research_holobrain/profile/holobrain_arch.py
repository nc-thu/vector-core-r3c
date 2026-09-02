# -*- coding: utf-8 -*-
"""holobrain_arch.py — 架构问题账本：全流水线(II=1)能省多少？浪费在哪里能拿回？

问题：把"喂数据 + 搬权重"做成完美流水线，108 列芯片能省多少？
答案的方法：四个机器模型 + 浪费逐条目归因 + 三种回收机制 + 7×4×2 矩阵。

口径（全部与 swiftvla_hw.py / holobrain_hw.py 同源，不改 RTL）：
  feed（阵列喂数） per 列组 = 4 + ceil(m/16)*(k+2)，条目合计 ×ceil(n/108)×count×mul
  attn 喂料 per 头   = ceil(lk/108)*(4+ceil(lq/16)*(dhead+2))   [QK^T, 逐列组]
                       + (4+ceil(lq/16)*(lk+2))                 [PV, n=dhead 1 列组]
  CONC 墙 = max(feed, sm, copy, W, elem) + act ← 这已经是"各引擎完美并行"的假设

机器模型（单位 M拍，cols=108, SM16, @198.5MHz）：
  M0   现行：Σ 每条目 feed（锚点 54.2138）
  M1   屋顶线：总 GMAC ÷ (16×108) —— 任何调度/流水线魔法的下限
  M2d  完美预取全流水（= CONC 口径本身）：各引擎完全重叠、条目间零气泡
  M2a  小 m 行拼批（合法=同 W 跨视角；上界=再含异 W 的独立 bmm/窗口）
  M2b  列多租户：列组数 ceil(n/108)→n/108（分数），m 取整保留（上界）
  M2c  跨视角拼批：视角共享 W 的条目 m×views、W 装载减半
  best M2b(feed) + M2a(上界) + 软件杠杆

生成：python holobrain_arch.py → holobrain_arch.json + 控制台表
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # e:/GPU ARCH/vector_core_sim
sys.path.insert(0, str(ROOT / "hw_zcu104" / "sim"))
sys.path.insert(0, str(ROOT / "research_swiftvla" / "profile"))
sys.path.insert(0, str(HERE))

import gem_cycles as G            # noqa: E402  (RTL 实测常数)
import swiftvla_hw as HW          # noqa: E402  (account/w_total/elem_ops 通用账本)
import holobrain_spec as HB       # noqa: E402
import holobrain_hw as HHW        # noqa: E402  (hb_attn_dims_of / 边界集合)

ROWS, COLS = 16, 108
MAC_CYC = ROWS * COLS             # 1728 MAC/拍
F_SYN = G.F_SYN                   # 198.5 MHz
BW64 = HW.BW64                    # 7.08 B/cyc
BWS = {"BW64": BW64, "BW128": 2 * BW64}

# 锚点（holobrain_hw.json walls / hw_main，2026-08 版）
ANCHOR = dict(M0=54.2138, W64=49.323513, sm=18.59718, copy=8.56272,
              elem16=7.933376, conc108=54.309228, gmacs=73.661456384e9,
              ref216_128=34.829874)

# ----------------------------------------------------------------------------
# 拼批因子表：count 里哪些是"彼此独立"的重复
#   tier A = 同一份 W 被独立激活复用（跨视角）——现行权重驻留阵列下完全合法
#   tier B = 独立但异 W（头间 bmm、窗口各自 K）——需 W 随 m-tile 流动，按上界报
#   串行依赖（BERT 12 层、去噪 10 步、6 层 decoder、MLP 链）一律不拼
# ----------------------------------------------------------------------------
NW = [HB.ceil7(h) * HB.ceil7(w) // HB.WIN // HB.WIN for (h, w) in HB.grids()]
VIEW_STAGES = ("Swin2D", "Swin3D", "Neck2D", "Neck3D", "PSE", "Fusion/img")


def factor_of(it, spec):
    """返回 (拼批因子 f, tier)。f 作用于 m（attn 为 lq），count 除以 f。"""
    views = spec["config"]["views"]
    st, nm = it["stage"], it["name"]
    if it["kind"] == "gemm":
        if st.startswith(VIEW_STAGES):
            return views, "A"                      # 同权重跨视角，count 含 ×views
        if st == "ActionHead/step" and nm == "temp_av":
            return HB.DEC_HEADS, "B"               # 8 头独立 bmm（异 W）
        return 1, "-"
    # attn 条目：win_attn 的 count = depth×views×NW，层串行 → 只拼 views×NW
    if st.startswith(("Swin2D", "Swin3D")):
        return views * NW[int(st[-1]) - 1], "B"    # 窗口独立但 K 各异（异 W）
    return 1, "-"


def cdiv(a, b):
    return -(-a // b)


def rceil(x):
    """抗浮点噪声的 ceil（m×f 后可能出现 735.0000001）。"""
    return math.ceil(x - 1e-9)


# ----------------------------------------------------------------------------
# 逐条目 feed / W（与 route_study.hw_gemm_w、swiftvla_hw.hw_attn 逐拍一致）
# ----------------------------------------------------------------------------
def feed_items(spec, frac=False, fmode=None):
    """Σ 阵列喂数拍数。frac=列多租户（列组数取分数）；fmode='A'/'AB' 启用拼批。"""
    dims = HHW.hb_attn_dims_of(spec)
    steps = spec["config"]["denoise_steps"]
    total = 0.0
    for it in spec["gemm_items"]:
        mul = steps if it["stage"] == "ActionHead/step" else 1
        cnt = it["count"] * mul
        f, tier = factor_of(it, spec) if fmode else (1, "-")
        if fmode == "A" and tier != "A":
            f = 1
        assert cnt % f == 0, f"拼批因子不整除: {it['stage']}/{it['name']} cnt={cnt} f={f}"
        if it["kind"] == "gemm":
            m, n, k = it["m"] * f, it["n"], it["k"]
            mt = rceil(m / ROWS)
            cg = (n / COLS) if frac else cdiv(n, COLS)
            total += (4 + mt * (k + 2)) * cg * (cnt // f)
        else:
            lq, lk, dh, hd, _ = dims(it)
            lq *= f
            mt = rceil(lq / ROWS)
            cg_qk = (lk / COLS) if frac else cdiv(lk, COLS)
            cg_pv = (dh / COLS) if frac else 1
            per_head = cg_qk * (4 + mt * (dh + 2)) + cg_pv * (4 + mt * (lk + 2))
            total += per_head * hd * (cnt // f)
    return total


def w_items(spec, view_halve=False):
    """W 装载拍数（64-bit 口径）。view_halve：tier-A 视角共享条目装载减半。"""
    total = 0
    for it in spec["gemm_items"]:
        if it["kind"] != "gemm":
            continue
        mul = spec["config"]["denoise_steps"] if it["stage"] == "ActionHead/step" else 1
        f = factor_of(it, spec)[0] if (view_halve and factor_of(it, spec)[1] == "A") else 1
        total += (G.load_w_ideal(it["k"], COLS) * cdiv(it["n"], COLS)
                  * (it["count"] * mul // f))
    return total


def engines(spec):
    """(sm, copy, elem16, act, macs, feed_base) —— HW.account 同源对账用。"""
    steps = spec["config"]["denoise_steps"]
    sm = copy = 0.0
    dims = HHW.hb_attn_dims_of(spec)
    for it in spec["gemm_items"]:
        mul = steps if it["stage"] == "ActionHead/step" else 1
        if it["kind"] != "attn":
            continue
        lq, lk, dh, hd, ca = dims(it)
        h = HW.hw_attn(lq, lk, dh, hd, ca, COLS, 16)
        sm += h["sm"] * it["count"] * mul
        copy += h["copy"] * it["count"] * mul
    elem = sum(i["m"] * i["count"] *
               (steps if i["stage"] == "ActionHead/step" else 1)
               for i in spec["elem_items"] if i["name"] != "softmax_exp") / 16
    macs = sum(i["macs"] * (steps if i["stage"] == "ActionHead/step" else 1)
               for i in spec["gemm_items"])
    return dict(sm=sm, copy=copy, elem=elem, macs=macs,
                feed=feed_items(spec))


def act_dma(spec):
    """边界激活 DMA（图像入/动作出）：account.dma − w_total。"""
    t = HW.account(spec, COLS, 16, HHW.hb_attn_dims_of(spec),
                   HHW.HB_BOUNDARY_ACT, HHW.HB_BOUNDARY_STORE)
    return t["dma"] - HW.w_total(spec, COLS)


# ----------------------------------------------------------------------------
# 浪费分解：gap = M0 − M1 按成因逐条目归因（gemm 与 attn 子 GEMM 同一套代数）
#   actual = (4 + ā(k+2))·b̄·cnt          ā=ceil(m/16), b̄=ceil(n/108)
#   ideal  = (m/16)·k·(n/108)·cnt          = MACs/1728
#   actual − ideal = n_round + m_round + cross + fixed   （恒等拆分，见 assert）
# ----------------------------------------------------------------------------
def waste_rows(spec):
    dims = HHW.hb_attn_dims_of(spec)
    steps = spec["config"]["denoise_steps"]
    rows = []

    def one(label, is_attn, m, k, n, cnt, bb=None):
        a, ba = m / ROWS, cdiv(m, ROWS)
        b = n / COLS
        if bb is None:
            bb = cdiv(n, COLS)
        actual = (4 + ba * (k + 2)) * bb * cnt
        ideal = a * k * b * cnt
        d = dict(
            item=label, attn=is_attn,
            n_round=(k + 2) * a * (bb - b) * cnt,
            m_round=(k + 2) * (ba - a) * b * cnt,
            cross=(k + 2) * (ba - a) * (bb - b) * cnt,
            fixed=(4 * bb + 2 * a * b) * cnt,
            actual=actual, ideal=ideal)
        assert abs(actual - (ideal + d["n_round"] + d["m_round"] + d["cross"]
                             + d["fixed"])) < 1e-6 * max(1.0, actual)
        rows.append(d)

    for it in spec["gemm_items"]:
        mul = steps if it["stage"] == "ActionHead/step" else 1
        cnt = it["count"] * mul
        label = f"{it['stage']}/{it['name']}"
        if it["kind"] == "gemm":
            one(label, False, it["m"], it["k"], it["n"], cnt)
        else:
            lq, lk, dh, hd, _ = dims(it)
            c2 = cnt * hd
            one(f"{label}[QK^T]", True, lq, dh, lk, c2)
            # 账本口径：PV 整条 1 次列组（hw_attn 不按 dhead 切列组，即使 dh>108）
            one(f"{label}[PV]", True, lq, lk, dh, c2, bb=1)
    return rows


def agg(rows, key):
    by = {}
    for r in rows:
        by[r["item"]] = by.get(r["item"], 0.0) + r[key]
    return sorted(by.items(), key=lambda kv: -kv[1])


# ----------------------------------------------------------------------------
# 软件杠杆档
# ----------------------------------------------------------------------------
def spec_of(lever):
    if lever == "none":
        return HB.build_spec()
    if lever == "hoist":
        return HB.build_spec(hoist_kv=True)
    sp = HB.build_spec(hoist_kv=(lever == "both"))
    sp["gemm_items"] = [i for i in sp["gemm_items"] if not i["stage"].startswith("Text")]
    sp["elem_items"] = [i for i in sp["elem_items"] if not i["stage"].startswith("Text")]
    return sp


LEVERS = [("none", "无"), ("bert", "+BERT缓存(删Text段)"),
          ("hoist", "+K/V hoist"), ("both", "两者都上")]


def main():
    out = {"meta": dict(
        cols=COLS, f_syn=F_SYN, bw64=BW64, mac_cyc=MAC_CYC,
        question="喂数据+搬权重全流水(II=1)能省多少？",
        formula_feed_gemm="compute += 4 + ceil(m/16)*(k+2)  # 每列组一次"
                          "；合计 ×ceil(n/108)×count×mul  (route_study.hw_gemm_w L43)",
        formula_feed_attn="QK^T: ceil(lk/108)*(4+ceil(lq/16)*(dhead+2)); "
                          "PV: (4+ceil(lq/16)*(lk+2)); ×heads×count "
                          "(swiftvla_hw.hw_attn L61-68)",
        formula_conc="conc = max(compute+drain, sm, copy, w, elem) + act (+drain)"
                     "  (swiftvla_hw.models L164) ← 已是各引擎完美并行的全流水口径",
        anchors=ANCHOR)}

    asserts = []

    def check(name, ok, detail=""):
        asserts.append(dict(name=name, passed=bool(ok), detail=detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
        return ok

    hb = HB.build_spec()
    eng = engines(hb)

    # ---------- ① 公式核实（3 条独立条目 + 用户手算那条） ----------
    print("=" * 78)
    print("① feed 公式核实（route_study.hw_gemm_w：每列组 4+ceil(m/16)*(k+2)）")
    print("=" * 78)
    spots = [
        ("Fusion/img/img_ffn_fc2", 1700, 256, 2048, 12),   # 用户手算条目
        ("Swin2D/s4/qkv", 196, 2304, 768, 4),
        ("ActionHead/step/temp_kv", 136, 512, 256, 60),
        ("PSE/pts_fc(3->128)", 217600, 128, 3, 2),
    ]
    verif = []
    for label, m, n, k, cnt in spots:
        mine = (4 + cdiv(m, ROWS) * (k + 2)) * cdiv(n, COLS) * cnt
        hw = HW.hw_gemm(m, n, k, COLS)[2] * cnt
        ok = mine == hw
        verif.append(dict(item=label, m=m, n=n, k=k, count=cnt,
                          my_cycles=mine, hw_cycles=hw))
        print(f"  {label:<30} m={m:>6} n={n:>5} k={k:>5} cnt={cnt:>3}"
              f" → {mine:>9,} 拍  (账本 {hw:>9,})  {'✓' if ok else '✗'}")
        check(f"公式核实 {label}", ok, f"{mine} == {hw}")
    print(f"  手算锚点 img_ffn_fc2: 107 批×3 列组×(2048+2)+4×3 → 658,062/次 ×12"
          f" = 7,896,744 ≈ 7.90M ✓（与账本逐拍一致）")
    out["formula_verification"] = verif

    # ---------- ② 机器模型 ----------
    print("\n" + "=" * 78)
    print("② 四个机器模型（cols=108，M拍）")
    print("=" * 78)
    M0 = eng["feed"]
    M1 = eng["macs"] / MAC_CYC
    w64 = w_items(hb)
    act = act_dma(hb)
    check("M0 = 54.2138 ±0.01", abs(M0 / 1e6 - ANCHOR["M0"]) < 0.01,
          f"{M0/1e6:.6f}")
    check("Σ feed 与 HW.account 逐拍一致", M0 == HW.account(
        hb, COLS, 16, HHW.hb_attn_dims_of(hb), HHW.HB_BOUNDARY_ACT,
        HHW.HB_BOUNDARY_STORE)["compute"])
    check("总 GMAC = 73.66G", abs(eng["macs"] - ANCHOR["gmacs"]) < 1e6,
          f"{eng['macs']/1e9:.4f} GMAC")
    check("W@BW64 = 49.3235 ±0.01", abs(w64 / 1e6 - ANCHOR["W64"]) < 0.01,
          f"{w64/1e6:.6f}")
    check("sm = 18.5972 ±0.01", abs(eng["sm"] / 1e6 - ANCHOR["sm"]) < 0.01,
          f"{eng['sm']/1e6:.6f}")
    check("copy = 8.5627 ±0.01", abs(eng["copy"] / 1e6 - ANCHOR["copy"]) < 0.01,
          f"{eng['copy']/1e6:.6f}")
    check("elem16 = 7.9334 ±0.01", abs(eng["elem"] / 1e6 - ANCHOR["elem16"]) < 0.01,
          f"{eng['elem']/1e6:.6f}")

    M2d = {bwn: max(M0, w64 * BW64 / bw, eng["sm"], eng["copy"], eng["elem"]) + act
           for bwn, bw in BWS.items()}
    check("M2d(BW64) == CONC 54.3092 ±0.01",
          abs(M2d["BW64"] / 1e6 - ANCHOR["conc108"]) < 0.01, f"{M2d['BW64']/1e6:.6f}")
    gap = M0 - M1
    print(f"  M0  现行喂数墙           = {M0/1e6:8.4f} M拍")
    print(f"  M1  屋顶线 73.66G/1728   = {M1/1e6:8.4f} M拍  (100% 阵列利用)")
    print(f"  M2d 完美全流水墙  BW64   = {M2d['BW64']/1e6:8.4f} M拍"
          f"（feed 仍最大 → 墙=feed+act）")
    print(f"  M2d 完美全流水墙  BW128  = {M2d['BW128']/1e6:8.4f} M拍")
    print(f"  gap = M0 − M1            = {gap/1e6:8.4f} M拍"
          f"（全流水 II=1 可省 = M0−M2d = {(M0-M2d['BW64'])/1e6:+.4f} M拍"
          f" —— 即省 0，墙就是喂数本身）")
    print(f"  act（边界激活 DMA）      = {act/1e6:.4f} M拍")
    out["machines"] = dict(
        M0_mcyc=M0 / 1e6, M1_mcyc=M1 / 1e6, gap_mcyc=gap / 1e6,
        M2d_mcyc={k: v / 1e6 for k, v in M2d.items()},
        full_pipeline_saving_mcyc=(M0 - M2d["BW64"]) / 1e6,
        w64_mcyc=w64 / 1e6, act_mcyc=act / 1e6, gmacs=eng["macs"],
        sm_mcyc=eng["sm"] / 1e6, copy_mcyc=eng["copy"] / 1e6,
        elem_mcyc=eng["elem"] / 1e6)

    # ---------- ③ 浪费分解 ----------
    print("\n" + "=" * 78)
    print("③ gap 逐条目归因（M拍；恒等拆分 actual = ideal + n/m/cross/fixed）")
    print("=" * 78)
    wr = waste_rows(hb)
    tot = {k: sum(r[k] for r in wr) for k in ("n_round", "m_round", "cross",
                                              "fixed", "actual", "ideal")}
    check("分解合计 == M0", abs(tot["actual"] - M0) < 1e-6, f"{tot['actual']/1e6:.6f}")
    check("ideal 合计 == M1", abs(tot["ideal"] - M1) < 1e-6, f"{tot['ideal']/1e6:.6f}")
    decomp = []
    for key, cname in (("n_round", "n 列组取整 ceil(n/108)−n/108"),
                       ("m_round", "m 批取整 ceil(m/16)−m/16"),
                       ("cross", "两者交叉"),
                       ("fixed", "固定开销(每列组+4、每批+2 拍流水边)")):
        v = tot[key]
        top = agg(wr, key)[:3]
        attn_share = sum(r[key] for r in wr if r["attn"]) / v if v else 0
        print(f"  {cname:<42} {v/1e6:7.4f} M  ({v/gap*100:5.1f}% of gap,"
              f" attn 条目占 {attn_share*100:4.1f}%)")
        for it, iv in top:
            print(f"      top: {it:<44} {iv/1e6:7.4f} M")
        decomp.append(dict(cause=cname, mcyc=round(v / 1e6, 4),
                           pct_of_gap=round(v / gap * 100, 2),
                           attn_share_pct=round(attn_share * 100, 2),
                           top_items=[dict(item=i, mcyc=round(vv / 1e6, 4))
                                      for i, vv in top]))
    out["gap_decomposition"] = decomp

    # ---------- ④ M2 机制 ----------
    print("\n" + "=" * 78)
    print("④ 能拿回浪费的机制（feed，M拍；cols=108 主档）")
    print("=" * 78)
    m2a_strict = feed_items(hb, fmode="A")
    m2a_upper = feed_items(hb, fmode="AB")
    m2b = feed_items(hb, frac=True)
    m2c = feed_items(hb, fmode="A")
    m2c_w = w_items(hb, view_halve=True)
    check("M2a_strict == M2c feed（合法拼批只剩视角）", m2a_strict == m2c,
          f"{m2a_strict/1e6:.4f}")
    check("feed ≥ 屋顶线（各档）", all(f >= M1 - 1e-6 for f in
          (m2a_upper, m2b, m2c)))
    print(f"  M0                          {M0/1e6:8.4f}")
    print(f"  M2a 行拼批(上界, A+B 档)    {m2a_upper/1e6:8.4f}"
          f"  省 {(M0-m2a_upper)/1e6:.4f} —— 合法部分(仅跨视角)只省"
          f" {(M0-m2a_strict)/1e6:.4f}")
    print(f"  M2b 列多租户(上界)          {m2b/1e6:8.4f}  省 {(M0-m2b)/1e6:.4f}"
          f" ← 大头，正好消掉 n_round+cross"
          f" = {(tot['n_round']+tot['cross'])/1e6:.4f}")
    print(f"  M2c 跨视角拼批 feed         {m2c/1e6:8.4f}  省 {(M0-m2c)/1e6:.4f}"
          f"（feed 几乎不动；W {w64/1e6:.2f}→{m2c_w/1e6:.2f} 才是收益）")
    best_feed = feed_items(hb, frac=True, fmode="AB")
    print(f"  best feed = M2b+M2a(上界)   {best_feed/1e6:8.4f}"
          f"  （距屋顶线还差 {(best_feed-M1)/1e6:.4f} = 残余 m_round+fixed）")

    # ---------- ⑤ 矩阵：模型 × 软件杠杆 × 带宽 ----------
    print("\n" + "=" * 78)
    print("⑤ 矩阵 wall = max(feed,W,sm,copy,elem)+act（M拍 / s@198.5MHz）")
    print("=" * 78)
    hoist_eng = engines(spec_of("hoist"))
    check("hoist 只删 GEMM：sm/copy 不变",
          hoist_eng["sm"] == eng["sm"] and hoist_eng["copy"] == eng["copy"],
          f"sm {hoist_eng['sm']/1e6:.4f} copy {hoist_eng['copy']/1e6:.4f}")
    check("elem 不随 hoist 变", hoist_eng["elem"] == eng["elem"])

    matrix = []
    cache = {}

    def lever_ctx(lever):
        if lever not in cache:
            sp = spec_of(lever)
            e = engines(sp)
            cache[lever] = dict(sp=sp, eng=e, w=w_items(sp),
                                act=act_dma(sp))
        return cache[lever]

    def model_feed(model, ctx):
        sp, e = ctx["sp"], ctx["eng"]
        return {"M0": e["feed"], "M1": e["macs"] / MAC_CYC,
                "M2a": feed_items(sp, fmode="AB"),
                "M2b": feed_items(sp, frac=True),
                "M2c": feed_items(sp, fmode="A"),
                "M2d": e["feed"],
                "best": feed_items(sp, frac=True, fmode="AB")}[model]

    hdr = f"{'模型':<6}{'杠杆':<10}{'BW':<7}{'feed':>8}{'W':>8}{'墙':>9}" \
          f"{'s@198.5':>9}{'墙者':>8}"
    model_desc = {"M0": "现行", "M1": "屋顶线", "M2a": "拼批(上界)",
                  "M2b": "列多租户", "M2c": "视角拼批", "M2d": "全流水",
                  "best": "b+a+杠杆"}
    for model in ("M0", "M1", "M2a", "M2b", "M2c", "M2d", "best"):
        print(f"-- {model} ({model_desc[model]}) --")
        print(hdr)
        for lever, lname in LEVERS:
            ctx = lever_ctx(lever)
            e = ctx["eng"]
            fd = model_feed(model, ctx)
            w_base = ctx["w"]
            w = w_items(ctx["sp"], view_halve=True) if model in ("M2c", "best") else w_base
            for bwn, bw in BWS.items():
                parts = dict(feed=fd, W=w * BW64 / bw, sm=e["sm"],
                             copy=e["copy"], elem=e["elem"])
                who = max(parts, key=parts.get)
                wall = parts[who] + ctx["act"]
                matrix.append(dict(
                    model=model, model_desc=model_desc[model], lever=lever,
                    lever_desc=lname, bw=bwn,
                    feed_mcyc=round(fd / 1e6, 4),
                    w_mcyc=round(parts["W"] / 1e6, 4),
                    wall_mcyc=round(wall / 1e6, 4),
                    wall_sec=round(wall / F_SYN, 4),
                    wall_engine=who if who != "feed" else "feed(阵列喂数)"))
                print(f"{model:<6}{lname:<10}{bwn:<7}{fd/1e6:>8.2f}"
                      f"{parts['W']/1e6:>8.2f}{wall/1e6:>9.2f}"
                      f"{wall/F_SYN:>9.3f}{matrix[-1]['wall_engine']:>10}")
    out["matrix"] = matrix

    # ---------- ⑥ best(108) vs 216 列 BW128 ----------
    print("\n" + "=" * 78)
    print("⑥ 关键对比：同一颗 108 列芯片靠调度/并发 vs 加宽到 216 列")
    print("=" * 78)
    ref = ANCHOR["ref216_128"]
    b_cell = next(c for c in matrix if c["model"] == "best" and c["lever"] == "both"
                  and c["bw"] == "BW128")
    b64 = next(c for c in matrix if c["model"] == "best" and c["lever"] == "both"
               and c["bw"] == "BW64")
    m0_cell = next(c for c in matrix if c["model"] == "M0" and c["lever"] == "none"
                   and c["bw"] == "BW64")
    sp_both = lever_ctx("both")["sp"]
    t216 = HW.account(sp_both, 216, 16, HHW.hb_attn_dims_of(sp_both),
                      HHW.HB_BOUNDARY_ACT, HHW.HB_BOUNDARY_STORE)
    ref_fair = HW.models(t216, sp_both, 216, 16, 2 * BW64, elem_par=16)["conc"] / 1e6
    m0w = m0_cell["wall_mcyc"]
    recover = (m0w - b_cell["wall_mcyc"]) / (m0w - ref) * 100
    print(f"  现行 M0(无杠杆,BW64)            {m0w:.2f} M = {m0_cell['wall_sec']:.3f} s")
    print(f"  best(108列, 双杠杆, BW64)        {b64['wall_mcyc']:.2f} M"
          f" = {b64['wall_sec']:.3f} s  墙={b64['wall_engine']}")
    print(f"  best(108列, 双杠杆, BW128)       {b_cell['wall_mcyc']:.2f} M"
          f" = {b_cell['wall_sec']:.3f} s  墙={b_cell['wall_engine']}")
    print(f"  216 列 BW128 CONC（无杠杆）      {ref:.2f} M = {ref/F_SYN*1e6:.3f} s")
    print(f"  216 列 BW128 CONC（同双杠杆）    {ref_fair:.2f} M"
          f" = {ref_fair/F_SYN*1e6:.3f} s ← 公平对比基准")
    print(f"  → best@108 是 216(无杠杆) 的 {b_cell['wall_mcyc']/ref*100:.0f}%"
          f"（慢 {(b_cell['wall_mcyc']/ref-1)*100:.0f}%）；对同杠杆 216 列是"
          f" {b_cell['wall_mcyc']/ref_fair*100:.0f}%")
    print(f"  → 以「M0@108 → 216列BW128」的 {m0w-ref:.2f}M 差距为分母，"
          f"108 列靠调度/并发吃回 {recover:.0f}%")
    out["vs_216"] = dict(best108_both_bw64_mcyc=b64["wall_mcyc"],
                         best108_both_bw128_mcyc=b_cell["wall_mcyc"],
                         cols216_bw128_mcyc=ref,
                         cols216_bw128_both_levers_mcyc=round(ref_fair, 4),
                         ratio_pct=round(b_cell["wall_mcyc"] / ref * 100, 1),
                         ratio_vs216_fair_pct=round(
                             b_cell["wall_mcyc"] / ref_fair * 100, 1),
                         widening_gain_recovered_pct=round(recover, 1),
                         best_engine_bw64=b64["wall_engine"],
                         best_engine_bw128=b_cell["wall_engine"])

    out["m2"] = dict(
        M2a_upper_mcyc=m2a_upper / 1e6, M2a_strict_mcyc=m2a_strict / 1e6,
        M2a_note=f"上界含异 W 独立实例(temp_av 8 头 bmm、win_attn 窗口×视角)；"
                 f"合法档(同 W 跨视角) feed 只省 {(M0-m2a_strict)/1e6:.3f} M",
        M2b_mcyc=m2b / 1e6,
        M2b_note="列组取整改分数(列多租户硬件上界)；恰好回收 n_round+cross",
        M2c_feed_mcyc=m2c / 1e6, M2c_w64_mcyc=m2c_w / 1e6,
        M2c_note="视角拼批对 feed 收益极小，W 装载减半才是收益",
        best_feed_mcyc=best_feed / 1e6,
        residual_to_roofline_mcyc=(best_feed - M1) / 1e6)

    # 权重端口 vs 真实字节（M2b 下 W 还能再省的余量）
    true_b = HB.w_bytes_of(hb)
    port_b = w64 * BW64
    out["w_bytes"] = dict(port_mb=port_b / 1e6, true_mb=true_b / 1e6,
                          padding_pct=round((1 - true_b / port_b) * 100, 1),
                          note="列多租户下小 n 条目可按真实字节装载，W 再省 ~4.7%")

    # ---------- 结论 ----------
    nr, mr, cr, fx = (tot["n_round"] / 1e6, tot["m_round"] / 1e6,
                      tot["cross"] / 1e6, tot["fixed"] / 1e6)
    out["conclusions"] = [
        f"账本 CONC 口径本来就是全流水（max(各引擎)+act），M2d={M2d['BW64']/1e6:.2f}M"
        f" ≥ M0={M0/1e6:.2f}M：把喂数据/搬权重做成 II=1 一拍也省不下来，"
        f"墙是阵列喂数本身（feed 比 W@BW64 {w64/1e6:.1f}M、sm 18.6M 都高）。",
        f"M0 与屋顶线 M1={M1/1e6:.2f}M 之间 gap={gap/1e6:.2f}M，成因："
        f"n 列组取整 {nr:.2f}M（占 {tot['n_round']/gap*100:.0f}%，主导）、"
        f"m 批取整 {mr:.2f}M、交叉 {cr:.2f}M、固定开销 {fx:.2f}M。",
        f"回收机制排序：M2b 列多租户 {m2b/1e6:.2f}M（省 {(M0-m2b)/1e6:.2f}M，"
        f"正好吃掉 n_round+cross）≫ M2c/M2a feed {(M0-m2c)/1e6:.3f}M"
        f"（合法拼批只有跨视角，m 都是 16 倍数或大 m，行取整本来就小）。",
        f"M2c 真正的价值在 W：视角共享条目装载减半 {w64/1e6:.1f}→{m2c_w/1e6:.1f}M，"
        f"只在 W 成墙的档位（M2b 后 BW64）起作用。",
        f"best = M2b+M2a+双软件杠杆：108 列 BW128 {b_cell['wall_mcyc']:.1f}M"
        f"（{b_cell['wall_sec']:.3f}s），是 216 列 BW128 无杠杆 {ref:.1f}M 的"
        f" {b_cell['wall_mcyc']/ref*100:.0f}%、同杠杆 {ref_fair:.1f}M 的"
        f" {b_cell['wall_mcyc']/ref_fair*100:.0f}%；吃回加宽收益的"
        f" {recover:.0f}%。BW64 档 {b64['wall_mcyc']:.1f}M 墙是"
        f" {b64['wall_engine']}。",
    ]
    out["asserts"] = asserts
    ok = all(a["passed"] for a in asserts)
    out["asserts_all_passed"] = ok
    print("\n" + "=" * 78)
    print(f"asserts: {sum(a['passed'] for a in asserts)}/{len(asserts)} passed"
          f" → {'ALL PASS' if ok else 'FAIL'}")
    for c in out["conclusions"]:
        print(f"  · {c}")

    (HERE / "holobrain_arch.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n-> holobrain_arch.json")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
