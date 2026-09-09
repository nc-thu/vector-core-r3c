# -*- coding: utf-8 -*-
"""actv_gold.py — AE_ACTV 微观对拍：随机向量生成 + numpy 黄金 + RTL dump 比对

用法：
  python actv_gold.py gen     # 生成 actv_ctx0.mem / actv_cases.mem / actv_ctx_exp.mem
  python actv_gold.py check   # 读 actv_ctx_out.mem（tb_ae_actv dump），逐字节比对

描述符编码与 01_rtl/sim/gen_vectors.py desc() 同一切片（op=6）：
  b_src[2:0]=子模式(0=ACTV 1=BIAS 2=NORM 3=ELTWISE) m=行 n=列(=stride) k=表长(NORM 时=n)
  y_base=原地张量 CTX 基址  b_base=表映像 CTX 基址  rq_m/rq_s=BIAS 的 m(Q8.8)/s
  ELTWISE：x1=y_base 原地输出、x2=b_base 只读，m1=rq_m、m2=desc[135:120]、s=rq_s
表映像布局（与 DMA TAG_CTX 字节路由 lane=b%16, addr=base+b/16 一致）：
  ACTV：表项 x 复制在字 b_base+x 的全部 16 lane 槽（映像 256 字 = 512B）
  BIAS：项 j 的 lo 在 lane j%16 @ b_base + j//16，hi 同相位 @ b_base+NLO + j//16
        （NLO = ceil(k/16)，两区各 NLO 字）
  NORM：字 b_base+0 为常数区（invn/eps_q24/g_shift/out_shift/flags），随后
        g 表 lo/hi、b 表 lo/hi 四区各 ceil(n/16) 字——语义唯一来源是
        ../spec/norm_spec.json + ../spec/norm_gold.py（编译器代理同源）。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "spec"))
from norm_gold import build_image, engine_row          # noqa: E402

CTX_WORDS = 13824
N_CASES = 21
rng = np.random.default_rng(20260831)


def desc6(submode, m, n, k, y_base, b_base, rq_m=0, rq_s=0, rq_m2=0):
    v = (6 << 252) | (submode << 246) | (m << 228) | (n << 212) | (k << 196)
    v |= (y_base << 136) | (b_base << 156)
    v |= ((rq_m & 0xFFFF) << 104) | (rq_s << 96)
    v |= ((rq_m2 & 0xFFFF) << 120)      # desc[135:120]：ELTWISE 第二乘子 m2
    v |= 0xF << 92                      # inv 恒 0xF（op=6 不参与 skip）
    return v


def sat8(x):
    return int(np.clip(x, -128, 127))


def build():
    ctx = rng.integers(-128, 128, size=(16, CTX_WORDS)).astype(np.int64)
    cases = []                          # (dict 供黄金执行)

    def add_actv(name, y_base, m, n, tbl_base):
        lut = rng.integers(-128, 128, size=256).astype(np.int64)
        for x in range(256):            # 表映像：项 x 复制到字 tbl_base+x 的全部 16 槽
            for L in range(16):
                ctx[L, tbl_base + x] = lut[x]
        cases.append(dict(name=name, sub=0, y=y_base, m=m, n=n,
                          tbl=tbl_base, k=0, lut=lut))

    def add_bias(name, y_base, m, n, tbl_base, k, rq_m, rq_s, bmin=-3000, bmax=3000):
        bj = rng.integers(bmin, bmax + 1, size=k).astype(np.int64)
        nlo = (k + 15) // 16
        for j in range(k):
            lo = int(bj[j]) & 0xFF
            hi = (int(bj[j]) >> 8) & 0xFF
            ctx[j % 16, tbl_base + j // 16] = lo if lo < 128 else lo - 256
            ctx[j % 16, tbl_base + nlo + j // 16] = hi if hi < 128 else hi - 256
        cases.append(dict(name=name, sub=1, y=y_base, m=m, n=n,
                          tbl=tbl_base, k=k, rqm=rq_m, rqs=rq_s, bj=bj))

    def add_norm(name, y_base, m, n, tbl_base, ln=True, gamma=None, beta=None,
                 eps=1e-5, sa_in=0.0146, sa_out=0.030, fill=None, t=None,
                 stripe=None):
        """NORM 用例：γ/β 缺省随机（γ 带负值、β 带饱和压力），表映像按 spec 布局。
        t 给定时为 AdaRMS 用例（逐行缩放进 t 区，word0 bit83=1）；
        stripe=(off, val, cnt) 在每组数据块后放 cnt 个常数条纹字（原地边界用例）。"""
        if gamma is None:
            gamma = rng.normal(1.0, 0.45, size=n)
            gamma[::7] *= -0.5                      # 一部分负 γ
        if beta is None:
            beta = rng.normal(0.0, 0.3, size=n)
        ada = t is not None
        im = build_image(n, gamma, beta, eps, sa_in, sa_out, ln=ln,
                         ada=ada, t=t, m_rows=m)
        nw = im["image"].shape[1]
        ctx[:, tbl_base:tbl_base + nw] = im["image"]
        if fill is not None:                        # 常数行/全 127/全 -128 角落
            for row in range(m):
                lane, base = row % 16, y_base + (row // 16) * n
                ctx[lane, base:base + n] = fill[row % len(fill)] if len(fill) else 0
        if stripe is not None:                      # 组块后的常数条纹（引擎不得触碰）
            off, val, cnt = stripe
            ngrp = (m + 15) // 16
            for g in range(ngrp):
                for L in range(16):
                    ctx[L, y_base + g * n + off: y_base + g * n + off + cnt] = val
        cases.append(dict(name=name, sub=2, y=y_base, m=m, n=n,
                          tbl=tbl_base, k=n, consts=im["consts"], g=im["g"],
                          b=im["b"], t6=im["t6"]))

    def add_eltwise(name, y_base, x2_base, m, n, m1, m2, s):
        """ELTWISE 用例：x1（原地输出）走 y_base，x2 走 x2_base，乘子 m1/m2 与右移 s
        直接来自描述符（rq_m / desc[135:120] / rq_s），数据用 ctx 随机初值。"""
        cases.append(dict(name=name, sub=3, y=y_base, m=m, n=n, tbl=x2_base,
                          k=n, rqm=m1, rqm2=m2, rqs=s))

    # 8 个旧用例（ACTV/BIAS，回归不回退的锚点）+ 6 个 NORM 用例：
    # n=1（var=0 只剩 eps）/ 尾组 / 长行 / RMS / 常数行与全 ±128 / γ 负与 β 饱和
    add_actv("A1 pad-row  m=18 n=20", 0, 18, 20, 64)
    add_actv("A2 2 full  m=32 n=5 ", 80, 32, 5, 320)
    add_actv("A3 1grp    m=7  n=33", 112, 7, 33, 576)
    add_bias("B1 k=n     m=18 n=20", 0, 18, 20, 832, 20, 256, 8)
    add_bias("B2 ktail   m=33 n=17", 48, 33, 17, 836, 17, 257, 8)
    add_bias("B3 neg-m   m=5  n=40", 100, 5, 40, 840, 40, -384, 4)
    add_bias("B4 sat     m=16 n=8 ", 140, 16, 8, 846, 8, 32767, 0, -30000, 30000)
    add_actv("A4 square  m=16 n=16", 148, 16, 16, 848)
    # ---- NORM：数据区避开旧表（64..1104）：N5 1200 / N6 1400 / 长行 3000+；
    #      表映像集中在 12000 起（N4 数据 5700..11843，不冲突）----
    add_norm("N1 n=1 LN  m=5     ", 3010, 5, 1, 12000)
    add_norm("N2 tail    m=18 n=257", 3020, 18, 257, 12010, ln=True)
    add_norm("N3 long LN m=16 n=2049", 3600, 16, 2049, 12100, ln=True)
    add_norm("N4 RMS pad m=17 n=3072", 5700, 17, 3072, 12700, ln=False,
             eps=1.1920929e-7)
    add_norm("N5 corner  m=48 n=64 ", 1200, 48, 64, 13500,
             fill=[37, 127, -128, 0, -1])
    add_norm("N6 γneg βsat m=33 n=48", 1400, 33, 48, 13530, ln=True)
    # ---- AdaRMS（t 区逐行缩放）：常规 t∈[0.5,1.9] / 极值 t∈[0.05,2.0] ----
    add_norm("N7 adaRMS  m=20 n=96 ", 1650, 20, 96, 13560, ln=False,
             t=rng.uniform(0.5, 1.9, size=20))
    add_norm("N8 adaLN-tx m=17 n=48", 1860, 17, 48, 13600, ln=True,
             t=rng.uniform(0.05, 2.0, size=17))
    # ---- 原地边界：每组数据块（n 列）后跟 8 个常数条纹字，引擎不得越界写 ----
    add_norm("N9 stripe  m=8  n=48 ", 1550, 8, 48, 13620, ln=True,
             stripe=(48, 90, 8))
    # ---- ELTWISE：x1=y_base 原地、x2=tbl_base，乘子/右移组合含负值/极值/移 0 ----
    add_eltwise("E1 0.75add m=18 n=40", 2100, 2200, 18, 40, 256, 192, 8)
    add_eltwise("E2 neg-m1 m=33 n=17 ", 2300, 2400, 33, 17, -384, 448, 8)
    add_eltwise("E3 extremes m=16 n=64", 2500, 2600, 16, 64, 32767, -32768, 8)
    add_eltwise("E4 s=0    m=5  n=1  ", 2700, 2750, 5, 1, 1, 1, 0)
    assert len(cases) == N_CASES

    # ---- 初态快照：映像装载完、黄金执行前（actv_ctx0.mem 必须是执行前的状态！）----
    ctx0 = ctx.copy()

    # ---- 黄金执行（按用例顺序原地改 ctx，与 RTL 逐字节同语义）----
    def dump_ctx(tag):
        with open(f"actv_ctx_exp_{tag}.mem", "w") as f:
            for L in range(16):
                for a in range(CTX_WORDS):
                    f.write(f"{int(ctx[L, a]) & 0xFF:02X}\n")

    for ci, c in enumerate(cases):
        m, n, yb = c["m"], c["n"], c["y"]
        if c["sub"] == 0:
            lut = c["lut"]
            for row in range(m):
                lane, base = row % 16, yb + (row // 16) * n
                for j in range(n):
                    x = ctx[lane, base + j] & 0xFF
                    ctx[lane, base + j] = lut[x]
        elif c["sub"] == 2:
            # NORM：黄金逐位镜像引擎语义（norm_gold.engine_row 是唯一来源）
            cs = c["consts"]
            for row in range(m):
                lane, base = row % 16, yb + (row // 16) * n
                xs = [int(v) for v in ctx[lane, base:base + n]]
                out = engine_row(xs, cs["invn"], cs["eps_q24"], cs["g_shift"],
                                 cs["out_shift"], cs["ln"], c["g"], c["b"],
                                 t6=c["t6"][row])
                ctx[lane, base:base + n] = [v - 256 if v > 127 else v
                                            for v in out]
        elif c["sub"] == 3:
            # ELTWISE：y = sat8(((x1·m1 + x2·m2) + 2^(s-1))>>>s)，x1 原地、x2 只读
            from norm_gold import eltwise_row
            m1, m2, s = c["rqm"], c["rqm2"], c["rqs"]
            for row in range(m):
                lane, base = row % 16, yb + (row // 16) * n
                x2b = c["tbl"] + (row // 16) * n
                x1 = [int(v) for v in ctx[lane, base:base + n]]
                x2 = [int(v) for v in ctx[lane, x2b:x2b + n]]
                out = eltwise_row(x1, x2, m1, m2, s)
                ctx[lane, base:base + n] = [v - 256 if v > 127 else v
                                            for v in out]
        else:
            rqm, rqs, bj = c["rqm"], c["rqs"], c["bj"]
            for row in range(m):
                lane, base = row % 16, yb + (row // 16) * n
                for j in range(n):
                    y = int(ctx[lane, base + j])
                    ctx[lane, base + j] = sat8((y * rqm + int(bj[j])) >> rqs)
        dump_ctx(f"c{ci}")

    # ---- 写文件 ----
    with open("actv_ctx0.mem", "w") as f:            # 128b/行：word = {lane15..lane0}
        for a in range(CTX_WORDS):
            w = 0
            for L in range(16):
                w |= (int(ctx0[L, a]) & 0xFF) << (8 * L)
            f.write(f"{w:032X}\n")
    with open("actv_cases.mem", "w") as f:
        for c in cases:
            v = desc6(c["sub"], c["m"], c["n"], c["k"], c["y"], c["tbl"],
                      c.get("rqm", 0), c.get("rqs", 0), c.get("rqm2", 0))
            f.write(f"{v:064X}\n")
    with open("actv_ctx_exp.mem", "w") as f:         # 扁平字节：idx = lane*WORDS+addr
        for L in range(16):
            for a in range(CTX_WORDS):
                f.write(f"{int(ctx[L, a]) & 0xFF:02X}\n")
    print(f"[gold] {N_CASES} 用例生成完毕（CTX {CTX_WORDS} 字）")
    for i, c in enumerate(cases):
        print(f"  case{i}: {c['name']}  tbl_base={c['tbl']} k={c['k']}")


def check():
    exp = np.loadtxt("actv_ctx_exp.mem", dtype=str)
    out = np.loadtxt("actv_ctx_out.mem", dtype=str)
    if len(exp) != len(out):
        print(f"[check] FATAL 行数不等 exp={len(exp)} out={len(out)}")
        sys.exit(1)
    e = np.array([int(s, 16) for s in exp], dtype=np.int64)
    o = np.array([int(s, 16) for s in out], dtype=np.int64)
    diff = e != o
    nbad = int(diff.sum())
    if nbad == 0:
        print(f"[check] PASS — {len(e)} 字节全部位精确（{N_CASES} 用例）")
    else:
        idx = np.nonzero(diff)[0][:10]
        print(f"[check] FAIL — {nbad}/{len(e)} 字节不符；前 10 处：")
        for i in idx:
            L, a = i // CTX_WORDS, i % CTX_WORDS
            print(f"  lane={L} addr={a}: exp={e[i]:02X} out={o[i]:02X}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("gen", "check"):
        print("用法: python actv_gold.py gen|check")
        sys.exit(2)
    (build if sys.argv[1] == "gen" else check)()
