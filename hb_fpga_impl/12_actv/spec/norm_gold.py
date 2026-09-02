# -*- coding: utf-8 -*-
"""norm_gold.py — AE_ACTV submode=2 (NORM) 的黄金参考实现（语义唯一来源）

对应 spec/norm_spec.json v1.0。两件事：

1. build_image()：编译器参考——从模型 γ/β/eps 和校准尺度生成表映像
   （常数区 + g 表 + b 表的 CTX 字节序列）。
2. engine_ref()：引擎定点语义的逐位仿真——Python 整数运算，与 RTL
   逐位一致（同一个 rsqrt LUT、同一次牛顿、同样的移位与饱和）。

独立可跑：python norm_gold.py 自测（随机张量 + 角落，对 fp32 参考
报 max|Δ|/mean|Δ|，单位 LSB）。

约定：>>> 一律按 Python 大整数算术右移（向负无穷），sat{N} 对称饱和。
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 位工具（RTL 同款语义）
# ---------------------------------------------------------------------------
def sat(x, n):
    """对称饱和到 n 位有符号 [-2^(n-1), 2^(n-1)-1]。"""
    lo, hi = -(1 << (n - 1)), (1 << (n - 1)) - 1
    return lo if x < lo else (hi if x > hi else x)


def asr(x, s):
    """算术右移（Python 的 >> 对 int 本来就是算术右移）。"""
    return x >> s if s > 0 else x << (-s)


def msb(v):
    """最高有效位位置（v>0）；v=0 返回 0。"""
    return v.bit_length() - 1


# ---------------------------------------------------------------------------
# rsqrt LUT（512 项 × 15b）——golden 与 RTL 用同一张表
# ---------------------------------------------------------------------------
def build_rsqrt_lut():
    """idx -> round(2^14 / sqrt((2048 + 16*idx + 8)/2^11))

    m_q11 ∈ [2048, 8192) 即 m ∈ [1,4)；索引 idx = (m_q11-2048)>>4，
    bin 宽 16 个 q11 单位，共 384 项（深度留 512，RTL 同款）。
    """
    lut = []
    for idx in range(512):
        mc = (2048 + 16 * idx + 8) / 2048.0      # bin 中心
        lut.append(int(round((1 << 14) / np.sqrt(mc))))
    return lut


RSQRT_LUT = build_rsqrt_lut()

# 给 RTL $readmemh 用（sim 目录展开；含表是契约一部分）
def dump_rsqrt_lut(path):
    with open(path, "w") as f:
        for v in RSQRT_LUT:
            f.write(f"{v:04X}\n")


# ---------------------------------------------------------------------------
# 引擎定点语义：rsqrt 共享单元（逐位）
# ---------------------------------------------------------------------------
def rsqrt_q20(v):
    """v (>0, Q24) -> invstat_q20 (27b 有符号容器内的正数)。

    与 RTL 逐位一致：LOD 归一化到 m∈[1,4)（指数取偶）→ 512 项 LUT →
    一次牛顿 → 左移 (18-f) 后 sat27。
    """
    v_eff = max(v, 1 << 13)
    e = msb(v_eff) & ~1                          # 偶化 MSB，E ∈ {12..38}
    f = e >> 1
    m_q11 = v_eff >> (e - 11)                    # 13b, [2048, 8192)
    r0 = RSQRT_LUT[(m_q11 - 2048) >> 4]         # 384 项内
    # 牛顿：r1 = (r0 * (3*2^39 - m_q11*r0*r0)) >>> 40
    r1 = asr(r0 * (3 * (1 << 39) - m_q11 * r0 * r0), 40)
    r1 = sat(r1, 15)
    inv = sat(asr(r1, f - 18) if f != 18 else r1, 27)   # r1 <<< (18-f)
    return max(inv, 1)


# ---------------------------------------------------------------------------
# 引擎定点语义：整行 NORM（一行 = 一个 lane 的 n 列）
# ---------------------------------------------------------------------------
def engine_row(xs, invn, eps_q24, g_shift, out_shift, ln, g_j, b_j, t6=64):
    """xs: 长 n 的 int8 列表（一行）；g_j/b_j: 每列 16b 定点表；t6: 本行
    AdaRMS 逐 token 缩放（= round(t·64)，int8；非 ada 恒 64 即 t=1.0）。

    返回：长 n 的 int8 输出列表。与 RTL 数据通路逐位一致。
    ada 链：ta = (t17·t6 + 32)>>>6——t6=64 时 (64x+32)>>>6 == x 对一切整数
    x 成立（0≤32<64，floor 除法恒等），所以非 ada 路径与 v1.1 逐位相同。
    """
    n = len(xs)
    s1 = sum(xs)                                 # 精确 24b
    s2 = sum(x * x for x in xs)                  # 精确 30b
    mu = s1 * invn                               # Q24，≤ 2^31
    ms = s2 * invn
    if ln:
        var = max(0, ms - asr(mu * mu, 24))
    else:
        var = ms
        mu = 0                                   # RMS 不减均值：u = x<<<24
    v = max(var + eps_q24, 1 << 13)
    inv = rsqrt_q20(v)                           # Q20
    S = 44 - g_shift
    out = []
    for j in range(n):
        x = xs[j]
        u = (x << 24) - mu                       # 34b
        prod = u * inv                           # = u_norm * 2^44，精确
        w = sat(asr(prod + (1 << (S - 1)), S), 9)      # round-half-up
        t17 = asr(w * g_j[j] + 128, 8)           # 17b，round-half-up
        ta = asr(t17 * t6 + 32, 6)               # AdaRMS 逐行缩放（t6=64 恒等）
        tb = ta + b_j[j]
        if out_shift > 0:
            out.append(sat(asr(tb + (1 << (out_shift - 1)), out_shift), 8) & 0xFF)
        else:
            out.append(sat(tb, 8) & 0xFF)
    return out


# ---------------------------------------------------------------------------
# 编译器参考：表映像生成
# ---------------------------------------------------------------------------
def build_image(n, gamma, beta, eps, sa_in, sa_out, ln=True, z_max=8.0,
                ada=False, t=None, m_rows=16):
    """γ/β/eps + 尺度 -> 常数与 g_j/b_j 表（+ AdaRMS 的 t 区）。

    ada=True 时 t 为逐行长 m_rows 的缩放向量（fp32，t=γ 缩放前的逐 token
    乘子，量纲与 γ 同层：y = z·γ·t + β）；t 区 = ceil(m/16) 个 CTX 字，
    字 g 的 lane L 字节 = t6[row=16g+L]，t6 = round(t·64) 钳到 int8。
    返回 dict(invn, eps_q24, g_shift, out_shift, ln, g, b, t6, image_bytes)。
    """
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta if beta is not None else np.zeros(n), dtype=np.float64)
    assert len(gamma) == n and len(beta) == n
    eps_q24 = int(round(eps / (sa_in * sa_in) * (1 << 24)))
    eps_q24 = max(eps_q24, 1 << 13)
    # 量纲：LN 输出 = z（无量纲 z 分数）*γ + β，量化只除 sa_out——
    # sa_in 只进 eps 换算（方差按 int8 单位算）。
    G = gamma / sa_out
    B = beta / sa_out
    import math
    gmax = float(np.max(np.abs(G))) if n else 0.0
    bmax = float(np.max(np.abs(B))) if n else 0.0
    # ① g_shift 只由 w 动态范围定：w = sat9(z*2^gs)，z_max 按校准分位取
    #   （缺省 8；w 预算 220，留饱和余量）。
    # ② out_shift 在 g_j = G*2^(8+os-gs) ≤ 32767 与 b_j = B*2^os ≤ 32767
    #   两个约束下取最大——os 越大 g/b 分辨率越高。
    # ③ 若 G_min < 0.5 需警惕 w 饱和丢线性范围（契约要求 256*G_min > 127）。
    # z_max 是每站点校准量（默认 8）：g_shift 只由 w 动态范围定，
    # w = sat9(z*2^gs)。取 floor 保证 z_max·2^gs ≤ 256（w 域不饱和优先，
    # ceil 对小 z_max 会过冲到 2 倍、w 大面积饱和）。
    # 粒度不足（G_max·2^-(gs+1) > 1）不靠抬 gs，由 sa_out 下限兜住
    # （见 compile_formulas：sa_out ≥ z_max·γ_max/512）。
    # 引擎约束 g_shift ≤ 7（S=44-gs ∈ [37,44]，桶移 3 级）；z_max>16 的站点不适用
    g_shift = max(0, min(7, int(math.floor(math.log2(256.0 / z_max)))))
    os_g = math.floor(15 - 8 + g_shift - math.log2(max(gmax, 1e-12)))
    os_b = math.floor(15 - math.log2(max(bmax, 1.0)))
    out_shift = int(max(0, min(os_g, os_b, 7)))
    g = [int(round(Gj * (1 << (8 + out_shift - g_shift)))) for Gj in G]
    while max(abs(gi) for gi in g) > 32767 and g_shift > 0:
        g_shift -= 1
        g = [int(round(Gj * (1 << (8 + out_shift - g_shift)))) for Gj in G]
    b = [int(round(Bj * (1 << out_shift))) for Bj in B]
    assert max(abs(bi) for bi in b) <= 32767, "b_j 超 16b，需加大 out_shift"
    invn = min((1 << 24) // n, (1 << 24) - 1)   # n=1 时恰 25b，钳进 u24 域
    #（n≥2 时 floor ≤ 2^23，钳位只在 n=1 生效；consts 与映像字节同值，RTL 才能对上）

    # 表映像字节（word0 常数区 + g lo/hi + b lo/hi + 可选 t 区），字内 lane j%16
    nlo = (n + 15) // 16
    ntw = (m_rows + 15) // 16 if ada else 0     # t 区字数 = ceil(m/16)
    words = [0] * (1 + 4 * nlo + ntw)           # Python 大整数，128b 字
    c = (invn & 0xFFFFFF) | ((eps_q24 & ((1 << 48) - 1)) << 24) \
        | ((g_shift & 0x3F) << 72) | ((out_shift & 0xF) << 78) \
        | ((1 if ln else 0) << 82) | ((1 if ada else 0) << 83)
    words[0] = c
    for j in range(n):
        lo_g, hi_g = g[j] & 0xFF, (g[j] >> 8) & 0xFF
        lo_b, hi_b = b[j] & 0xFF, (b[j] >> 8) & 0xFF
        lane, wd = j % 16, j // 16
        words[1 + wd] |= lo_g << (8 * lane)
        words[1 + nlo + wd] |= hi_g << (8 * lane)
        words[1 + 2 * nlo + wd] |= lo_b << (8 * lane)
        words[1 + 3 * nlo + wd] |= hi_b << (8 * lane)
    t6 = [64] * m_rows
    if ada:
        assert t is not None and len(t) == m_rows
        t6 = [int(min(127, max(-128, round(float(tv) * 64.0)))) for tv in t]
        for r in range(m_rows):                 # 字 r//16 的 lane r%16 = t6[r]
            words[1 + 4 * nlo + r // 16] |= (t6[r] & 0xFF) << (8 * (r % 16))
    # 转成 (16, nwords) 的字节图（与 actv_gold.py 的 ctx 同构：lane 行 × 字列）
    img = np.zeros((16, len(words)), dtype=np.int64)
    for wi, wv in enumerate(words):
        wv &= (1 << 128) - 1
        for lane in range(16):
            img[lane, wi] = (wv >> (8 * lane)) & 0xFF
    return dict(invn=invn, eps_q24=eps_q24, g_shift=g_shift, out_shift=out_shift,
                ln=ln, g=g, b=b, n=n, t6=t6, ada=ada, image=img,
                consts=dict(invn=invn, eps_q24=eps_q24, g_shift=g_shift,
                            out_shift=out_shift, ln=ln, ada=ada))


# ---------------------------------------------------------------------------
# host fp32 参考（数值门的对照路径）
# ---------------------------------------------------------------------------
def fp32_ref(xs, gamma, beta, eps, sa_in, sa_out, ln=True, t=1.0):
    """反量化 → fp32 LN/RMS → 按 sa_out 重量化（round-half-even）。

    量化约定与部署一致：sat8 = clip[-128, 127]（同 golden_interp.sat8 /
    GEMM 的 y 量化；负轨允许 -128，不是对称的 ±127）。
    t 为本行 AdaRMS 逐 token 缩放（缺省 1.0 = 非 ada）。"""
    a = np.asarray(xs, dtype=np.float64) * sa_in
    if ln:
        mu = a.mean()
        v = ((a - mu) ** 2).mean()
    else:
        mu, v = 0.0, (a ** 2).mean()
    y = (a - mu) / np.sqrt(v + eps) * gamma * t + beta
    return np.clip(np.round(y / sa_out), -128, 127).astype(np.int64)


def engine_full(x_int8, consts, g, b, t6=None):
    """整批行过引擎语义。x_int8: int8 数组 [m, n]（行主序）；返回有符号 int。"""
    m, n = x_int8.shape
    out = np.zeros((m, n), dtype=np.int64)
    for i in range(m):
        row = engine_row([int(v) for v in x_int8[i]], consts["invn"],
                         consts["eps_q24"], consts["g_shift"],
                         consts["out_shift"], consts["ln"], g, b,
                         t6=64 if t6 is None else t6[i])
        out[i] = [v - 256 if v > 127 else v for v in row]
    return out


# ---------------------------------------------------------------------------
# submode=3 ELTWISE：双输入残差加 y = sat8(((x1·m1 + x2·m2) + 半)>>>s)
# ---------------------------------------------------------------------------
def eltwise_row(x1, x2, m1, m2, s):
    """x1/x2: 同形 int8 列表；m1/m2: 16b 有符号乘子；s: 右移 0..15。
    round-half-up，sat8 = clip[-128,127]。与 RTL 数据通路逐位一致。"""
    assert -(1 << 15) <= m1 < (1 << 15) and -(1 << 15) <= m2 < (1 << 15)
    out = []
    for a, b in zip(x1, x2):
        acc = a * m1 + b * m2
        if s > 0:
            acc = asr(acc + (1 << (s - 1)), s)
        out.append(sat(acc, 8) & 0xFF)
    return out


def eltwise_params(sa1, sa2, sa_out, s=8):
    """编译器参考：从两个输入尺度与输出尺度算 m1/m2/s。
    y_fp32 = x1·sa1 + x2·sa2 → 量化到 sa_out：
      m_i = round(sa_i / sa_out · 2^s)，必须落在 int16；
    s 在两乘子都不溢出的约束下取最大（缺省 8 = Q8.8，分辨率 1/256）。"""
    while s > 0 and (round(sa1 / sa_out * (1 << s)) >= (1 << 15)
                     or round(sa2 / sa_out * (1 << s)) >= (1 << 15)):
        s -= 1
    m1 = int(round(sa1 / sa_out * (1 << s)))
    m2 = int(round(sa2 / sa_out * (1 << s)))
    assert abs(m1) < (1 << 15) and abs(m2) < (1 << 15), "乘子超 int16，站点不适合 ELTWISE"
    return m1, m2, s


# ---------------------------------------------------------------------------
# 自测：随机 + 角落，报定点 vs fp32 的 LSB 差
# ---------------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(20260831)
    worst = []
    cases = []
    for n, ln in [(256, True), (256, False), (3072, True), (1, True),
                  (257, False), (2048, True)]:
        m = 64
        if n == 1:
            xs = rng.integers(-128, 128, size=(m, n))
        else:
            xs = rng.integers(-64, 64, size=(m, n))
        gamma = rng.normal(1.0, 0.35, size=n)
        beta = rng.normal(0.0, 0.15, size=n)
        cases.append((f"n={n} {'LN' if ln else 'RMS'}", xs, gamma, beta, 1e-5 if ln else 1.19e-7, ln))
    # 角落：常数行 / 全 127 / 全 -128 / γ 负 / β 大
    cases.append(("const rows", np.full((16, 256), 37, dtype=np.int64),
                  np.ones(256), np.zeros(256), 1e-5, True))
    cases.append(("all 127", np.full((16, 256), 127, dtype=np.int64),
                  np.ones(256), np.zeros(256), 1e-5, True))
    cases.append(("all -128", np.full((16, 256), -128, dtype=np.int64),
                  np.ones(256), np.zeros(256), 1e-5, True))
    gneg = np.ones(256); gneg[::2] = -0.8
    cases.append(("neg gamma", rng.integers(-64, 65, size=(64, 256)),
                  gneg, np.zeros(256), 1e-5, True))
    cases.append(("big beta", rng.integers(-64, 65, size=(64, 256)),
                  np.ones(256), np.full(256, 2.5), 1e-5, True))
    # AdaRMS：RMS + 逐行缩放 t（与 γ 同层乘子），t 覆盖 0.5..1.9 与极值
    for name, ln, tlo, thi in [("ada RMS", False, 0.5, 1.9),
                                ("ada LN t-ext", True, 0.05, 2.0)]:
        n = 96 if not ln else 48
        m = 20 if not ln else 17
        xs = rng.integers(-64, 65, size=(m, n))
        gamma = rng.normal(1.0, 0.35, size=n)
        beta = rng.normal(0.0, 0.15, size=n)
        t = rng.uniform(tlo, thi, size=m)
        sa_in, sa_out = 0.0146, 0.030
        im = build_image(n, gamma, beta, 1e-5 if ln else 1.19e-7, sa_in,
                         sa_out, ln=ln, ada=True, t=t, m_rows=m)
        got = engine_full(xs, im["consts"], im["g"], im["b"], t6=im["t6"])
        ref = np.stack([fp32_ref(xs[i], gamma, beta, 1e-5 if ln else 1.19e-7,
                                 sa_in, sa_out, ln=ln, t=t[i])
                        for i in range(m)])
        d = (got - ref).astype(np.float64)
        print(f"{name:<16}{np.abs(d).max():>8.2f}{np.abs(d).mean():>9.3f}"
              f"{d.mean():>8.3f}  {'-':>5}  -")
        if np.abs(d).max() > 2.0 or abs(d.mean()) > 0.2:
            ok = False

    print(f"{'用例':<16}{'max|Δ|':>8}{'mean|Δ|':>9}{'meanΔ':>8}  w饱和  g_j量程")
    ok = True
    for name, xs, gamma, beta, eps, ln in cases:
        sa_in, sa_out = 0.0146, 0.030            # sa_out 按 LN 输出（z 分数）量级
        im = build_image(xs.shape[1], gamma, beta, eps, sa_in, sa_out, ln=ln)
        got = engine_full(xs.astype(np.int64), im["consts"], im["g"], im["b"])
        ref = np.stack([fp32_ref(xs[i], gamma, beta, eps, sa_in, sa_out, ln)
                        for i in range(xs.shape[0])])
        d = (got - ref).astype(np.float64)
        # w 饱和统计（复用 engine_row 的中间量，口径一致）
        nsat = 0
        for i in range(min(16, xs.shape[0])):
            row = [int(v) for v in xs[i]]
            s1, s2 = sum(row), sum(v * v for v in row)
            mu = s1 * im["invn"]
            ms = s2 * im["invn"]
            var = max(0, ms - ((mu * mu) >> 24)) if ln else ms
            inv = rsqrt_q20(max(var + im["eps_q24"], 1 << 13))
            mu_eff = mu if ln else 0
            S = 44 - im["g_shift"]
            for x in row:
                w = sat((((x << 24) - mu_eff) * inv + (1 << (S - 1))) >> S, 9)
                if abs(w) >= 256:
                    nsat += 1
        gmax = max(abs(v) for v in im["g"])
        print(f"{name:<16}{np.abs(d).max():>8.2f}{np.abs(d).mean():>9.3f}"
              f"{d.mean():>8.3f}  {nsat:>5}  {gmax}")
        if np.abs(d).max() > 2.0 or abs(d.mean()) > 0.2:
            ok = False
    print("自测:", "PASS（max≤2 LSB, |mean|<0.2 LSB）" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    if "--dump-lut" in sys.argv:
        dump_rsqrt_lut(os.path.join(HERE, "..", "sim", "rsqrt_lut.mem"))
        print("[gold] rsqrt_lut.mem 已导出（512×15b，RTL $readmemh 用）")
        sys.exit(0)
    sys.exit(0 if _selftest() else 1)
