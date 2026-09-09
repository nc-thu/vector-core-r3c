# -*- coding: utf-8 -*-
"""
gen_micro_vec.py — ae_hif8_dot16 微架构实验的位精确黄金模型 + 测试向量生成

模型逐条镜像 RTL（E:/ae_syn/micro/rtl/ae_hif8_dot16.sv）：
  1. hif8 解码（dot 前缀码 / 符号-数值指数 / 锥化尾数；0x00、0x80 → 0）
  2. 每 lane 独立：sig8 = sa*sb，e7 = ea+eb（双重偏置 +44），4 binade/桶共 19 桶，
     桶内左移 r=e7&3 后符号累加 —— 全程整数，无中途舍入
  3. 末端 W = Σ acc_b<<4b（刻度 2^-50，sig8 含 3 位小数尺度）→ 绝对值左移规格化 → TA(半向上)舍入
     → HiF8 编码；溢出饱和到最大正常数 2^15（0x6E/0xEE，白皮书可选行为），
     下溢阈值 2^-23（之上取 2^-22，之下取 0）
自检：与 Fraction 精确"最近可表示值(平手向上)"神谕在随机向量上逐位一致。
输出：ctrl.mem / a.mem / b.mem / expect.mem（tb_hif8_dot16.sv 读）
"""
import random
from fractions import Fraction as Fr

LANES = 16
NB = 19

# ---------------------------------------------------------------- 解码（镜像 hif8_dec）
def hif8_dec(c):
    d4  = (c >> 5) & 3 == 0b11
    d3  = (c >> 5) & 3 == 0b10
    d2  = (c >> 5) & 3 == 0b01
    d1  = (c >> 4) & 7 == 0b001
    d0  = (c >> 3) & 15 == 0b0001
    dml = (c >> 3) & 15 == 0b0000
    se, mag, sig = 0, 0, 8 | (c & 7)
    if d4:
        mag, se = 8 | ((c >> 1) & 7), (c >> 4) & 1
        sig = 8 | ((c & 1) << 2)
    elif d3:
        mag, se = 4 | ((c >> 2) & 3), (c >> 4) & 1
        sig = 8 | ((c & 3) << 1)
    elif d2:
        mag, se = 2 | ((c >> 3) & 1), (c >> 4) & 1
    elif d1:
        mag, se = 1, (c >> 3) & 1
    elif d0:
        mag, se = 0, 0
    else:  # dml
        se, mag, sig = 1, 23 - (c & 7), 8
        if (c & 7) == 0:
            sig = 0                       # 0x00 零 / 0x80 NaN→0
    eidx = 22 - mag if se else 22 + mag
    if sig == 0:
        eidx = 0                          # 零/NaN：桶索引不越界（加的是 ±0，任意桶皆可）
    return (c >> 7) & 1, sig, eidx        # (s, sig4, eidx=E+22)

def dec_val(c):
    """解码的精确值（Fraction），供神谕用。"""
    s, sig, eidx = hif8_dec(c)
    if sig == 0:
        return Fr(0)
    return Fr(sig, 8) * Fr(2) ** (eidx - 22) * (-1 if s else 1)

# ---------------------------------------------------------------- 编码（镜像 hif8_enc）
def hif8_enc(s, E, frac):
    if E <= -16:                                     # DML: M = E+23 ∈[1,7]
        return (s << 7) | ((E + 23) & 7)
    mag, se = abs(E), (1 if E < 0 else 0)
    if mag == 0:                                     # D=0
        return (s << 7) | (0b0001 << 3) | (frac & 7)
    if mag == 1:                                     # D=1
        return (s << 7) | (0b001 << 4) | (se << 3) | (frac & 7)
    if mag <= 3:                                     # D=2
        return (s << 7) | (0b01 << 5) | (se << 4) | ((mag & 1) << 3) | (frac & 7)
    if mag <= 7:                                     # D=3
        return (s << 7) | (0b10 << 5) | (se << 4) | ((mag & 3) << 2) | (frac & 3)
    return (s << 7) | (0b11 << 5) | (se << 4) | ((mag & 7) << 1) | (frac & 1)  # D=4

def k_taper(E):
    if E >= 16 or E <= -23: return 7
    if E <= -16: return 0
    if -3 <= E <= 3: return 3
    if -7 <= E <= 7: return 2
    return 1

def normalize_enc(W):
    """镜像 ST_CA/ST_N/ST_E：W 为刻度 2^-44 的整数和。
    注意：E=15 只有 1.0×2^15 一个可表示正常数（1.5×2^15 是 Inf 模式），
    故 |v| ≥ 2^15 一律饱和到 0x6E/0xEE（白皮书"溢出饱和到边界"选项）。"""
    if W == 0:
        return 0x00
    s = 1 if W < 0 else 0
    A = abs(W)
    p = A.bit_length() - 1
    E = p - 50
    if E >= 15:
        return (s << 7) | 0b01101110                # 饱和 2^15
    if E <= -23:
        return ((s << 7) | 0b00000001) if p >= 27 else 0x00
    k = k_taper(E)
    if k == 0:
        frac, half = 0, (A >> (p - 1)) & 1
    else:
        frac, half = (A >> (p - k)) & ((1 << k) - 1), (A >> (p - k - 1)) & 1
    if half:
        frac += 1
    if frac == (1 << k):                            # 进位 → 1.0×2^(E+1)
        E, frac = E + 1, 0
        if E >= 15:
            return (s << 7) | 0b01101110
    return hif8_enc(s, E, frac)

# ---------------------------------------------------------------- 逐位模型
def model_dot(a_cycles, b_cycles, share_b=False):
    """a_cycles/b_cycles: 每 cycle 一组 16 lane 字节；返回 16 lane 的 HiF8 字节。"""
    acc = [[0] * NB for _ in range(LANES)]
    for av, bv in zip(a_cycles, b_cycles):
        for ln in range(LANES):
            sA, sgA, eA = hif8_dec(av[ln])
            sB, sgB, eB = hif8_dec(bv[0] if share_b else bv[ln])
            sig8 = sgA * sgB
            e7 = eA + eB
            val = (sig8 << (e7 & 3)) * (-1 if (sA ^ sB) else 1)
            acc[ln][e7 >> 2] += val
    out = []
    for ln in range(LANES):
        W = 0
        for b in range(NB):
            W += acc[ln][b] << (4 * b)
        out.append(normalize_enc(W))
    return out

# ---------------------------------------------------------------- Fraction 神谕自检
# 候选集：全部数值可表示模式；排除 0x80(NaN) 与 0x6F/0xEF(Inf 模式，饱和口径下
# 不作为输出候选 —— 超过最大正常数 2^15 一律饱和到边界)
REP = [(b, dec_val(b)) for b in range(256)
       if b != 0x80 and b not in (0x6F, 0xEF)]

def oracle_enc(W):
    """精确最近可表示值（平手向上），Inf 模式不作候选 → 溢出自动饱和 2^15。"""
    v = Fr(W, 2 ** 50)
    if v == 0:
        return 0x00
    best, bd = None, None
    for byt, rv in REP:
        d = abs(v - rv)                    # 有符号距离（v/rv 都带符号）
        if bd is None or d < bd or (d == bd and abs(rv) > abs(best[1])):
            best, bd = (byt, rv), d
    return best[0]

def lane_W(av, bv):
    """单 lane 的精确 W（整数为 2^-44 刻度）。"""
    W = 0
    for x, y in zip(av, bv):
        sA, sgA, eA = hif8_dec(x)
        sB, sgB, eB = hif8_dec(y)
        e7 = eA + eB
        W += ((sgA * sgB) << (e7 & 3) << (4 * (e7 >> 2))) * (-1 if (sA ^ sB) else 1)
    return W

def self_check(n=3000, seed=7):
    rng = random.Random(seed)
    n_ok = 0
    for t in range(n):
        K = rng.randint(1, 64)
        av = [rng.randrange(256) for _ in range(K)]
        bv = [rng.randrange(256) for _ in range(K)]
        a_c = [[av[i]] * LANES for i in range(K)]     # 16 lane 同值 → 各 lane 同果
        b_c = [[bv[i]] * LANES for i in range(K)]
        got = model_dot(a_c, b_c)
        W = lane_W(av, bv)
        oc = oracle_enc(W)
        for ln in range(LANES):
            assert got[ln] == oc, \
                f"MISMATCH t={t} lane={ln} got={got[ln]:02x} oracle={oc:02x} W={W} v={Fr(W,2**44)}"
        n_ok += 1
    print(f"[self-check] {n} 随机 dot（K≤64）×16 lane 全部与 Fraction 神谕逐位一致 → PASS")

# ---------------------------------------------------------------- 向量生成
def gen_pattern(rng, kind):
    if kind == 0:      # 全随机字节（覆盖全部 dot/Em/M 组合 + 特殊值）
        return rng.randrange(256)
    if kind == 1:      # 小幅值（DML/低指数密集）
        return rng.choice([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                           0x81, 0x82, 0x87, 0x00, 0x80])
    if kind == 2:      # 大幅值（D=4 区 + Inf 模式数值 0x6F/0xEF）
        return rng.choice([0x6C, 0x6D, 0x6E, 0x6F, 0xEC, 0xED, 0xEE, 0xEF,
                           0x68, 0x69, 0x6A, 0x70])
    if kind == 3:      # 同 binade 对撞（桶冲突压力）
        return rng.choice([0x17, 0x1F, 0x97, 0x23, 0x2B])
    return rng.choice([0x00, 0x80, 0x6F, 0xEF])       # 零/NaN/Inf 混入

def main():
    self_check()
    rng = random.Random(20260827)
    dots = []
    for d in range(12):
        K = rng.choice([33, 64, 65, 128, 200, 257, 512])
        a_l = [bytearray() for _ in range(LANES)]
        b_l = [bytearray() for _ in range(LANES)]
        for cyc in range(K):
            ka = rng.random()
            kind_a = 0 if ka < 0.5 else 1 if ka < 0.7 else 2 if ka < 0.85 else 3
            kind_b = rng.randrange(5)
            for ln in range(LANES):
                a_l[ln].append(gen_pattern(rng, kind_a))
                b_l[ln].append(gen_pattern(rng, kind_b))
        dots.append((K, a_l, b_l))

    expect = []
    expect_sb = []                       # SHARE_B=1 口径：B 取 lane0 广播
    for K, a_l, b_l in dots:
        a_c = [[a_l[ln][c] for ln in range(LANES)] for c in range(K)]
        b_c = [[b_l[ln][c] for ln in range(LANES)] for c in range(K)]
        b_s = [[b_l[0][c]] * LANES for c in range(K)]
        expect.extend(model_dot(a_c, b_c))
        expect_sb.extend(model_dot(a_c, b_s, share_b=True))

    with open("ctrl.mem", "w") as f:
        f.write(f"{len(dots):08x}\n")
        for K, _, _ in dots:
            f.write(f"{K:08x}\n")
    with open("a.mem", "w") as f:
        for K, a_l, b_l in dots:
            for c in range(K):
                for ln in range(LANES):
                    f.write(f"{a_l[ln][c]:02x}\n")
    with open("b.mem", "w") as f:
        for K, a_l, b_l in dots:
            for c in range(K):
                for ln in range(LANES):
                    f.write(f"{b_l[ln][c]:02x}\n")
    with open("expect.mem", "w") as f:
        for e in expect:
            f.write(f"{e:02x}\n")
    with open("expect_sb.mem", "w") as f:
        for e in expect_sb:
            f.write(f"{e:02x}\n")
    tot = sum(K for K, _, _ in dots) * LANES
    print(f"[gen] {len(dots)} dots, {tot} 乘积 → ctrl/a/b/expect(+_sb).mem 写出完成")
    print(f"[gen] 样例期望(dot0): {' '.join(f'{e:02x}' for e in expect[:16])}")

if __name__ == "__main__":
    main()
