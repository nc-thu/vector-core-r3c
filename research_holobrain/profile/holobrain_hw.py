# -*- coding: utf-8 -*-
"""holobrain_hw.py — HB-GD 0.2B 映射到 hw_zcu104 脉动阵列架构的周期账。

与 swiftvla_hw.py 完全同源（不改 RTL，纯账本）：
  GEMM/COPY/softmax/DMA 常数 = gem_cycles.py 的 RTL 实测；account()/models()/hw_attn()
  直接复用 swiftvla_hw 的通用实现（Evo-1/SwiftVLA/HB-GD 共用账本）；
  W 装载流、LWreal/CONC 两档并发、BW64=7.08 B/cyc、drain+52、cols=108/216。

口径（任务约定）：
  主档 = LIBERO 2 视图(agentview+腕部) + N_j=8 + lang16 + 320x256 + depth on +
  DPMSolver++ 10 步（忠实上游：K/V 每步重算无缓存，动作专家权重每步重装 x10）；
  每 chunk = pred_steps 64 → 预算线 64/30=2.13s(30Hz) / 64/50=1.28s(50Hz)。

生成：python holobrain_hw.py  →  holobrain_hw.json + 控制台七段表
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # e:/GPU ARCH/vector_core_sim
sys.path.insert(0, str(ROOT / "hw_zcu104" / "sim"))
sys.path.insert(0, str(ROOT / "research_swiftvla" / "profile"))
sys.path.insert(0, str(HERE))

import gem_cycles as G            # noqa: E402  (RTL 实测常数)
import swiftvla_hw as HW          # noqa: E402  (account/models/row 通用账本)
import swiftvla_spec as SV        # noqa: E402
import evo1_spec as S             # noqa: E402
import holobrain_spec as HB       # noqa: E402

F_SYN = G.F_SYN                   # 198.5 MHz
F_FAST = 250e6
BW64 = HW.BW64                    # 7.08 B/cyc
CTX_B, WRAM_K = HW.CTX_B, HW.WRAM_K

HB_BOUNDARY_ACT = {"patch_embedSwin2D(4x4x3 conv)",     # RGB 245.8KB/视角
                   "patch_embedSwin3D(4x4x1 conv)"}    # depth 81.9KB/视角
HB_BOUNDARY_STORE = {"head_out(64->8)"}                 # 64 步动作 4KB


def hb_attn_dims_of(spec):
    """HB-GD 注意力维度工厂：按 stage/name 解析 (lq, lk, dhead, heads, causal)。"""
    cfg = spec["config"]
    L, V, nj = cfg["lang_tokens"], cfg["views"], cfg["num_joints"]
    n_img, q, kv = cfg["n_img_tokens"], cfg["action_tokens"], cfg["temp_kv"]
    lvl4v = cfg["lvl4_tokens_per_view"]

    def dims(it):
        st, nm = it["stage"], it["name"]
        if st.startswith("Swin2D/s"):
            i = int(st[-1]) - 1
            return 49, 49, HB.SW2_C[i] // HB.SW2_H[i], HB.SW2_H[i], False
        if st.startswith("Swin3D/s"):
            i = int(st[-1]) - 1
            return 49, 49, HB.SW3_C[i] // HB.SW3_H[i], HB.SW3_H[i], False
        if st.startswith("Text/BERT"):
            return L, L, HB.BERT_H // HB.BERT_HEADS, HB.BERT_HEADS, False
        if st.startswith("Fusion/bi"):
            return V * lvl4v, L, HB.BI_PROJ // HB.BI_HEADS, HB.BI_HEADS, False
        if st.startswith("Fusion/txt"):
            return L, L, HB.ENH_D // HB.TXT_HEADS, HB.TXT_HEADS, False
        if nm.startswith("img_attn"):
            return q, n_img, HB.DEC_D // HB.DEC_HEADS, HB.DEC_HEADS, False
        if nm.startswith("txt_attn"):
            return q, L, HB.DEC_D // HB.DEC_HEADS, HB.DEC_HEADS, False
        if nm.startswith("enc_jga"):
            return nj, nj, HB.DEC_D // HB.DEC_HEADS, HB.DEC_HEADS, False
        raise ValueError(f"unmapped attn item: {st}/{nm}")

    return dims


NOT_MODELED = [
    {"name": "MSDeformableAttention 采样+加权和",
     "ops": "6.96 MMAC/层/视角(加权和) + 双线性插值~4x(每点4邻居 gather+MAC)；x6层x2视角 ≈ 84 MMAC + 335 M 元素操作/chunk",
     "where": "向量核/CPU（gather+MAC 混合算子，非脉动阵列负载）；投影 4 条(value/output/sampling/weights)已进 gemm_items"},
    {"name": "PSE get_pts 4x4 反投影",
     "ops": "217,600 次 4x4 linalg.solve/视角 ≈ 14 MMAC 等价",
     "where": "CPU 预求逆+批量 matvec；相机固定时 projection_mat 为常量，pts 与 pts_fc(83.6 MMAC) 可整体离线缓存"},
    {"name": "PSE depth 概率加权聚合",
     "ops": "1700 个 [1x128]·[128x128] = 27.9 MMAC/视角",
     "where": "向量核（M=1 病态，16 行阵列利用率 1/16）；softmax(128bin) 另有 21.8 万 exp/视角"},
    {"name": "temp_joint 6D einsum（QK^T 等价）",
     "ops": "N_j=8: 4.46 MMAC/层/步（N_j=14: 13.6M），8x64 个 [16x32]@[32x17] 微 GEMM/层",
     "where": "向量核或按层重排 bmm；K/N 远小于 108 列，脉动阵列利用率极差"},
    {"name": "temp_joint tril 因果 mask",
     "ops": "16x17 时间注意力有效项 56%，稠密 AV 已全额计入（浪费 44% 可稀疏化省）",
     "where": "mask 本身是标量比较，软逻辑"},
    {"name": "UpsampleHead 双线性插值",
     "ops": "~0.17M lerp/步 x10",
     "where": "软逻辑/向量核"},
    {"name": "FK recompute + DPMSolver++ step + 采样噪声",
     "ops": "[1,64,N_j] 矩阵链 x10 步，串行依赖",
     "where": "CPU/软逻辑（每步一次，不可并入阵列）"},
    {"name": "RoPE/RMSNorm/LayerNorm/SiLU/AdaRMS 调制/gate",
     "ops": "elem 粗账 1.24 G-ops/chunk（softmax 部分由 SM 引擎另行计入）",
     "where": "向量核；memory-bound，小算子链"},
    {"name": "Swin 窗口切分/roll/rel-pos-bias/PatchMerging unfold",
     "ops": "每层 ~T x C 元素 gather-scatter（2D+3D 共 24 块 x 视角）",
     "where": "DMA/软逻辑"},
    {"name": "BERT tokenizer + word embedding 查表",
     "ops": "L 次 23.4M 参数表 gather（L=16）",
     "where": "embedding 表驻 DDR 仅 gather；指令跨 chunk 不变可整条缓存（BERT 1.37 GMAC 一并省掉）"},
    {"name": "depth 输入预处理",
     "ops": "uint16mm PNG 解码 /1000 + cv2.resize 双线性（2x 320x256x16bit 输入）",
     "where": "CPU；输入带宽已按 81.9KB/视角 int8 计入 boundary DMA"},
]


# ============================================================================
def main():
    out = {}
    hb = HB.build_spec()                                  # LIBERO 主档（忠实上游）
    hb_hoist = HB.build_spec(hoist_kv=True)
    hb_dims = hb_attn_dims_of(hb)

    # ---------- ① 负载概览 ----------
    print("=" * 78)
    print("① HB-GD 0.2B 负载概览（LIBERO 主档：每 action chunk = 64 步动作）")
    print("=" * 78)
    t = hb["totals"]
    cfg = hb["config"]
    print(f"  {cfg['views']} 视图 {cfg['img_wh'][0]}x{cfg['img_wh'][1]} depth=on,"
          f" N_j={cfg['num_joints']}, lang={cfg['lang_tokens']},"
          f" 去噪 {cfg['denoise_steps']} 步（K/V 每步重算，忠实上游）")
    print(f"{'阶段':<26}{'GMAC':>9}{'占 MAC':>8}{'W 流(MB)':>10}")
    stage_w = {}
    matchers = [("Vision2D", ("Swin2D", "Neck2D")), ("Vision3D+PSE", ("Swin3D", "Neck3D", "PSE")),
                ("Text", ("Text",)), ("Fusion", ("Fusion",)), ("ActionHead", ("ActionHead",))]
    for st_key, prefixes in matchers:
        macs = sum(i["macs"] * (cfg["denoise_steps"] if i["stage"] == "ActionHead/step" else 1)
                   for i in hb["gemm_items"] if i["stage"].startswith(prefixes))
        wb = sum(i["k"] * i["n"] * i["count"] *
                 (cfg["denoise_steps"] if i["stage"] == "ActionHead/step" else 1)
                 for i in hb["gemm_items"]
                 if i["stage"].startswith(prefixes) and i["kind"] == "gemm")
        stage_w[st_key] = dict(macs=macs, w_bytes=wb)
        print(f"{st_key:<26}{macs/1e9:>9.2f}{macs/t['gemm_macs_total']*100:>7.0f}%{wb/1e6:>10.0f}")
    print(f"{'合计':<26}{t['gemm_macs_total']/1e9:>9.2f}{'100%':>8}"
          f"{HB.w_bytes_of(hb)/1e6:>10.0f}")
    out["workload"] = dict(
        total_gmacs=t["gemm_macs_total"], w_bytes=HB.w_bytes_of(hb),
        params_total=sum(v["params"] for v in hb["params"].values()),
        per_step_gmacs=t["gemm_macs_per_step"],
        stages={k: dict(gmacs=v["macs"], w_mb=v["w_bytes"] / 1e6) for k, v in stage_w.items()})

    # ---------- ② 逐 GEMM 清单 ----------
    print("\n== ② 逐 GEMM 清单（M/N/K x 次数；denoise 10 步已折算 ActionHead/step）==")
    print(f"{'stage':<18}{'项':<32}{'M':>6}{'N':>6}{'K':>7}{'次数':>7}{'GMAC':>8}"
          f"{'k≤4096':>8}{'出buf':>10}")
    gemm_rows = []
    for it in hb["gemm_items"]:
        mul = cfg["denoise_steps"] if it["stage"] == "ActionHead/step" else 1
        if it["kind"] != "gemm":
            continue
        macs = it["m"] * it["n"] * it["k"] * it["count"] * mul
        buf = it["m"] * it["n"]
        flag = "OK" if it["k"] <= WRAM_K else f"x{it['k']//WRAM_K} 切"
        bufs = f"{buf/1024:.0f}KB" + ("" if buf <= CTX_B else " !>2MB")
        print(f"{it['stage']:<18}{it['name']:<32}{it['m']:>6}{it['n']:>6}{it['k']:>7}"
              f"{it['count']*mul:>7}{macs/1e9:>8.3f}{flag:>8}{bufs:>10}")
        gemm_rows.append(dict(stage=it["stage"], name=it["name"], m=it["m"], n=it["n"],
                              k=it["k"], count=it["count"] * mul, gmacs=macs / 1e9,
                              k_ok=it["k"] <= WRAM_K, out_bytes=buf))
    for it in hb["gemm_items"]:
        mul = cfg["denoise_steps"] if it["stage"] == "ActionHead/step" else 1
        if it["kind"] != "attn":
            continue
        lq, lk, dh, hd, _ = hb_dims(it)
        print(f"{it['stage']:<18}{it['name'] + f'({hd}h x {lk})':<32}{lq:>6}{lk:>6}"
              f"{dh:>7}{it['count']*mul:>7}{it['macs']*mul/1e9:>8.3f}{'COPY':>8}"
              f"{lq*lk//1024:.0f}KB")
    out["gemm_items"] = gemm_rows

    # ---------- ③ 容量核查 ----------
    print("\n== ③ 容量核查（CTX 2MB / WRAM k<=4096 / 激活驻留）==")
    big = [g for g in gemm_rows if g["out_bytes"] > CTX_B]
    kbig = [g for g in gemm_rows if not g["k_ok"]]
    kbig_txt = ", ".join(g["name"] + f" k={g['k']}" for g in kbig)
    big_txt = ", ".join(g["name"] + f" {g['out_bytes']/1e6:.2f}MB" for g in big)
    print(f"  WRAM 违例：{kbig_txt}"
          f" → k 两段切（+1 组波前/排空，多 1 次 requant 舍入）")
    print(f"  CTX 违例（单输出>2MB）：{big_txt}")
    print("    img_ffn_fc1 (1700x2048=3.48MB) → n 拆 2 列组组对（周期不变，描述符 +4 拍）")
    feat2d = cfg["lvl_tokens_per_view"] * 256 * cfg["views"]
    print(f"  4 级特征驻留: 2D {feat2d/1e6:.2f}MB + 3D {cfg['lvl_tokens_per_view']*128*cfg['views']/1e6:.2f}MB"
          f" + img_feature({cfg['n_img_tokens']}tok) {cfg['n_img_tokens']*256/1e3:.0f}KB"
          f" → CTX 2MB 放得下（无 SwiftVLA 式 KV cache 溢出问题）")
    print(f"  BERT word embedding 23.4M 参数驻 DDR（仅 gather，不进 W 流）")
    b_dma = cfg["views"] * (320 * 256 * 3 + 320 * 256 * 1) + cfg["lang_tokens"] * 2
    print(f"  边界 DMA/chunk：RGB {cfg['views']}x245.8KB + depth {cfg['views']}x81.9KB"
          f" + 指令/动作 ≈ {b_dma/1e3:.0f} KB")
    out["capacity"] = dict(
        wram_k_violations=[(g["name"], g["k"]) for g in kbig],
        ctx_over_2mb=[(g["name"], g["out_bytes"]) for g in big],
        boundary_dma_bytes=b_dma, feat_pool_bytes=feat2d)

    # ---------- ④ 硬件档位主表 ----------
    print("\n== ④ 硬件映射主表（s/chunk；SM16=现行 RTL，BW64=现行 64-bit 引擎口）==")
    print(f"{'配置':<46}{'M cyc':>9}{'s@198.5':>9}{'s@250':>8}{'阵列占':>7}{'GMAC/s':>8}")
    configs = [
        ("cols=108 SM16 BW64（现行硅片）LWreal", dict(cols=108, sm_par=16, bw=BW64)),
        ("cols=108 SM16 BW64（现行硅片）CONC", dict(cols=108, sm_par=16, bw=BW64)),
        ("cols=216 SM16 BW64（R3 packed）CONC", dict(cols=216, sm_par=16, bw=BW64)),
        ("cols=216 SM16 BW128 CONC", dict(cols=216, sm_par=16, bw=2 * BW64)),
        ("cols=216 SM32 BW128 CONC", dict(cols=216, sm_par=32, bw=2 * BW64)),
        ("cols=216 SM32 BW128 CONC +drain52", dict(cols=216, sm_par=32, bw=2 * BW64,
                                                   drain52=True)),
        ("cols=216 SM32 BW256 CONC", dict(cols=216, sm_par=32, bw=4 * BW64)),
        ("cols=108 SM16 BW64 CONC +drain52(仅参照)", dict(cols=108, sm_par=16, bw=BW64,
                                                          drain52=True)),
    ]
    main_rows = []
    for name, kw in configs:
        cols = kw.pop("cols")
        t0 = HW.account(hb, cols, kw.pop("sm_par"), hb_dims,
                        HB_BOUNDARY_ACT, HB_BOUNDARY_STORE)
        m = HW.models(t0, hb, cols, 16, kw.pop("bw"), elem_par=16, **kw)
        r = HW.row(name, m, F_SYN)
        s250 = r["sec"] * F_SYN / F_FAST
        print(f"{name:<46}{r['mcyc']:>9.0f}{r['sec']:>9.2f}"
              f"{s250:>8.2f}{r['util']*100:>6.0f}%{r['gmacs']:>8.1f}")
        main_rows.append(dict(name=name, mcyc=r["mcyc"], sec_1985=r["sec"],
                              sec_250=s250, util=r["util"],
                              lwreal_sec=m["lwreal"] / F_SYN,
                              w_gb=m["w64"] * BW64 / 1e9))
    out["hw_main"] = main_rows

    # ---------- ⑤ 墙分解 ----------
    print("\n== ⑤ 谁是墙（M 拍；CONC 口径，含 elem16）==")
    print(f"{'配置':<34}{'喂料':>8}{'sm÷16':>8}{'COPY':>8}{'W流':>8}{'elem16':>8}"
          f"{'drain52':>8}{'墙':>18}")
    walls = []
    for cols, bwn, bw in ((108, "64b", BW64), (216, "64b", BW64),
                          (216, "128b", 2 * BW64), (216, "256b", 4 * BW64)):
        t0 = HW.account(hb, cols, 16, hb_dims, HB_BOUNDARY_ACT, HB_BOUNDARY_STORE)
        m = HW.models(t0, hb, cols, 16, bw, elem_par=16, drain52=True)
        keys = ("feed", "sm", "copy", "W", "elem", "drain")
        vals = [t0["compute"], t0["sm"], t0["copy"], m["w"], m["elem"], m["drain"]]
        who = keys[vals.index(max(vals))]
        print(f"{f'cols={cols} {bwn}':<34}{vals[0]/1e6:>8.0f}{vals[1]/1e6:>8.0f}"
              f"{vals[2]/1e6:>8.0f}{vals[3]/1e6:>8.0f}{vals[4]/1e6:>8.0f}"
              f"{vals[5]/1e6:>8.0f}{who + f' {max(vals)/1e6:.0f}M':>18}")
        walls.append(dict(cfg=f"cols={cols} {bwn}",
                          **{k: v / 1e6 for k, v in zip(keys, vals)}, wall=who))
    out["walls"] = walls

    # ---------- ⑥ 敏感度 ----------
    print("\n== ⑥ 敏感度（cols=216 SM16 BW128 CONC，s/chunk @198.5；预算 2.13s/30Hz）==")
    sens = []
    for label, kw in [
        ("基线 LIBERO 2视图 Nj=8 lang16 N=10", dict()),
        ("3 视图（ur5_wsg/真机相机数）", dict(views=3)),
        ("4 视图（RoboTwin 相机数）", dict(views=4)),
        ("N_j=14（RoboTwin 双臂/已发布权重口径）", dict(nj=14)),
        ("RGB-only（depth 分支整体移除）", dict(depth=False)),
        ("去噪 4 步（外推，需精度验证）", dict(steps=4)),
        ("语言 32 token", dict(lang=32)),
        ("分辨率 256x256（LIBERO 原生）", dict(img_w=256, img_h=256)),
        ("分辨率 384x256（真机 processor）", dict(img_w=384)),
    ]:
        sp = HB.build_spec(**kw)
        t0 = HW.account(sp, 216, 16, hb_attn_dims_of(sp), HB_BOUNDARY_ACT,
                        HB_BOUNDARY_STORE)
        m = HW.models(t0, sp, 216, 16, 2 * BW64, elem_par=16)
        print(f"  {label:<40} {sp['totals']['gemm_macs_total']/1e9:>6.1f} GMAC"
              f"  W {m['w64']*BW64/1e9:>5.2f} GB  {m['conc']/F_SYN:>6.2f} s"
              f"  ({m['conc']/F_FAST:>5.2f} @250)")
        sens.append(dict(label=label, gmacs=sp["totals"]["gemm_macs_total"],
                         w_gb=m["w64"] * BW64 / 1e9, sec=m["conc"] / F_SYN))
    out["sensitivity"] = sens

    # ---------- ⑦ HB-GD vs SwiftVLA vs Evo-1（同口径） ----------
    print("\n== ⑦ HB-GD vs SwiftVLA vs Evo-1（同账本、同硬件档）==")
    print(f"{'负载':<38}{'GMAC':>8}{'W(GB)':>8}{'108/64b':>9}{'216/128b':>9}"
          f"{'216/256b':>9}{'util216':>8}")

    @contextlib.contextmanager
    def vit224():
        old = (S.VIT_IMG, S.VIT_SEQ, S.VIT_PS_TOKENS)
        S.VIT_IMG, S.VIT_SEQ, S.VIT_PS_TOKENS = 224, 257, 64
        try:
            yield
        finally:
            S.VIT_IMG, S.VIT_SEQ, S.VIT_PS_TOKENS = old

    with vit224():                      # spec 构建须在 224 上下文内（读模块全局）
        evo_c1_224 = S.build_spec(vit_tiles=2, hoist_kv=True, steps=10)
    rows = [  # (label, spec, dims_factory, boundary_act, boundary_store, ctx)
        ("HB-GD LIBERO 2视图 Nj8 N=10（主档）", hb, lambda sp: hb_dims,
         HB_BOUNDARY_ACT, HB_BOUNDARY_STORE, contextlib.nullcontext),
        ("HB-GD LIBERO 2视图 +K/V hoist", hb_hoist, lambda sp: hb_attn_dims_of(sp),
         HB_BOUNDARY_ACT, HB_BOUNDARY_STORE, contextlib.nullcontext),
        ("HB-GD RoboTwin 4视图 Nj14 lang32", HB.build_spec(views=4, nj=14, lang=32),
         lambda sp: hb_attn_dims_of(sp), HB_BOUNDARY_ACT, HB_BOUNDARY_STORE,
         contextlib.nullcontext),
        ("HB-GD RGB-only 2视图（无depth）", HB.build_spec(depth=False),
         lambda sp: hb_attn_dims_of(sp), HB_BOUNDARY_ACT, HB_BOUNDARY_STORE,
         contextlib.nullcontext),
        ("SwiftVLA 部署 3 视图 N=10（hoist）",
         SV.build_spec(hoist_kv=True), lambda sp: HW.sv_attn_dims_of(sp),
         HW.SV_BOUNDARY_ACT, HW.SV_BOUNDARY_STORE, contextlib.nullcontext),
        ("Evo-1 448/N32/2cam（v2 终点）", S.build_spec(vit_tiles=2, hoist_kv=True),
         lambda sp: HW.evo_attn_dims, HW.EVO_BOUNDARY_ACT, HW.EVO_BOUNDARY_STORE,
         contextlib.nullcontext),
        ("Evo-1 224/N10/2cam（C1 负载）",
         evo_c1_224, lambda sp: HW.evo_attn_dims,
         HW.EVO_BOUNDARY_ACT, HW.EVO_BOUNDARY_STORE, vit224),
    ]
    comp_rows = []
    for label, sp, adims_of, ba, bs, ctx in rows:
        adims = adims_of(sp)
        with ctx():
            cells = []
            for cols, bw in ((108, BW64), (216, 2 * BW64), (216, 4 * BW64)):
                t0 = HW.account(sp, cols, 16, adims, ba, bs)
                m = HW.models(t0, sp, cols, 16, bw, elem_par=16)
                cells.append(m["conc"] / F_SYN)
            t216 = HW.account(sp, 216, 16, adims, ba, bs)
        util = t216["macs"] / (cells[1] * F_SYN) / (16 * 216)
        w_bytes = sum(i["k"] * i["n"] * i["count"] *
                      (sp["config"]["denoise_steps"] if i["stage"] == "ActionHead/step" else 1)
                      for i in sp["gemm_items"] if i["kind"] == "gemm")
        print(f"{label:<38}{t216['macs']/1e9:>8.1f}{w_bytes/1e9:>8.2f}{cells[0]:>9.2f}"
              f"{cells[1]:>9.2f}{cells[2]:>9.2f}{util*100:>7.0f}%")
        comp_rows.append(dict(label=label, gmacs=t216["macs"], w_gb=w_bytes / 1e9,
                              sec_108_64b=cells[0], sec_216_128b=cells[1],
                              sec_216_256b=cells[2], util_216=util))
    out["comparison"] = comp_rows
    t_ref = HW.account(hb, 216, 16, hb_dims, HB_BOUNDARY_ACT, HB_BOUNDARY_STORE)
    m_ref = HW.models(t_ref, hb, 216, 16, 2 * BW64, elem_par=16)
    t_ho = HW.account(hb_hoist, 216, 16, hb_attn_dims_of(hb_hoist),
                      HB_BOUNDARY_ACT, HB_BOUNDARY_STORE)
    m_ho = HW.models(t_ho, hb_hoist, 216, 16, 2 * BW64, elem_par=16)
    print(f"  [K/V hoist] 每步重算(忠实上游) vs hoist: {m_ref['conc']/F_SYN:.2f} vs "
          f"{m_ho['conc']/F_SYN:.2f} s (差 {100*(1-m_ho['conc']/m_ref['conc']):.0f}%)")

    out["not_modeled"] = NOT_MODELED
    out["realtime_lines"] = dict(
        libero_30hz_64act=64 / 30, libero_50hz_64act=64 / 50,
        swiftvla_ref_30hz=50 / 30, evo1_libero_line=0.70)
    print(f"\n  实时线：HB-GD chunk=64 → 30Hz 预算 {64/30:.2f}s / 50Hz {64/50:.2f}s；"
          f"Evo-1 旧线 0.70s")
    lw108 = main_rows[0]["sec_1985"]
    cc108 = main_rows[1]["sec_1985"]
    print(f"  108 列现行硅片：LWreal {lw108:.2f}s / CONC {cc108:.2f}s →"
          f" 30Hz {'达标' if cc108 <= 64/30 else '不达标'}，"
          f"50Hz {'达标' if cc108 <= 64/50 else '不达标'}（CONC 口径）")

    (HERE / "holobrain_hw.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n-> holobrain_hw.json")


if __name__ == "__main__":
    main()
