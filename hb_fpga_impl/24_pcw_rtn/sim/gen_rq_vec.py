# -*- coding: utf-8 -*-
"""
gen_rq_vec.py — requant 二代（门 1）对拍向量生成
相位 A: s=8，混合 x 分布（全域随机 27b / 仿真 GEMM 累加器 / 边界值），m 混合
        （Q8.8 全域随机 + 正小尺度 + 极值）→ 全部精确变体 vs rq_v1 逐位对拍。
相位 B: s∈[8,47] 随机 → 仅 rq_v2(T_MAX=39) vs rq_v1（其余变体 s 域外）。
相位 C: s=8，m 全域 → rq_m6（m 量化 6b）偏差数据：分解 (m6,t6) 由本脚本算好，
        RTL 与 v1 输出差异落盘供统计（数值决策归用户）。
相位 D(★24_pcw_rtn): RTN 就近舍入（rn_en=1）—— 期望 er.mem 由本脚本按
        research_w8a8_error/REPORT.md §4 口径计算（=软件线公式）：
          s≥9: y = sat8((x·m + 2^(s-1)) >>> s)   [与 RTL (sum+2^(s-9))>>>(s-8) 逐位等价]
          s=8: 退化为截断（rn=0，REPORT §4「s=8 退化为截断」）
        s 覆盖：HB 标定域 [21,27] 加权 / 全域 [9,47] / s=8 退化行 / 小 s [9,11]
        （小 s + 大 |x·m| 撞 sat8 上下界与舍入同拍）。
输出: ctrl.mem(4 行) / xa,ma,sa / xb,mb,sb / xc,mc,m6c,t6c / xr,mr,sr,er (.mem)
"""
import numpy as np

rng = np.random.RandomState(20260827)

def to_hex(v, bits):
    return f"{int(v) & ((1 << bits) - 1):0{(bits + 3) // 4}x}"

def sat8(v):
    return np.clip(v, -128, 127).astype(np.int64)

# ---------------- x 分布 ----------------
def gen_x(n):
    xs = []
    # 1/3 全域随机 27b
    r = rng.randint(-(1 << 26), (1 << 26), size=n // 3)
    xs.append(r.astype(np.int64))
    # 1/3 仿真 GEMM 累加器（K=4096 INT8·INT8 求和）
    k = n // 3
    A = rng.randint(-128, 128, size=(k, 4096)).astype(np.int64)
    B = rng.randint(-128, 128, size=4096).astype(np.int64)
    xs.append(A @ B)
    # 1/3 边界 + 小幅值
    edges = [0, 1, -1, 127, -128, 128, -129, 255, 256,
             (1 << 25), -(1 << 25), (1 << 26) - 1, -(1 << 26), (1 << 26),
             127 * 4096, -128 * 4096, 16256 * 4096, 128 * 128 * 4096,
             96 * 8, 56 * 8, 64 * 8, 40 * 8]          # RQ 表常用 m 的 x*m 临界点
    k2 = n - 2 * (n // 3)
    pick = rng.randint(0, len(edges) + 1, size=k2)
    small = rng.randint(-(1 << 20), (1 << 20), size=k2)
    xs.append(np.array([edges[p] if p < len(edges) else small[i]
                        for i, p in enumerate(pick)], dtype=np.int64))
    x = np.concatenate(xs)
    # ★ 27b 有符号域是 [−2^26, 2^26−1]：上界取 2^26−1（R3C 版写 2^26，+2^26 在
    #   RTL 里被解释成 −2^26；相位 A/B 对硬件神谕 rq_v1 双侧同错抵消，相位 D 对
    #   python 神谕才暴露 —— 2026-09-04 修）
    return np.clip(x, -(1 << 26), (1 << 26) - 1)

def gen_m(n):
    ms = []
    ms.append(rng.randint(-32768, 32768, size=n // 2).astype(np.int64))   # 全域
    ms.append(np.concatenate([                                        # 常用+极值
        rng.choice([16, 40, 48, 56, 64, 96, 8, 4, 2, 1, 255, 256, 257,
                    -1, -2, -255, -256, 32767, -32768, 100, 300],
                   size=n - n // 2)]).astype(np.int64))
    return np.concatenate(ms)

# ---------------- 相位 A ----------------
nA = 60000
xa = gen_x(nA); ma = gen_m(nA); sa = np.full(nA, 8, dtype=np.int64)

# ---------------- 相位 B ----------------
nB = 30000
xb = gen_x(nB); mb = gen_m(nB); sb = rng.randint(8, 48, size=nB)

# ---------------- 相位 C：m → (m6, t6) 分解 ----------------
nC = 30000
xc = gen_x(nC); mc = gen_m(nC)
m6c = np.zeros(nC, dtype=np.int64); t6c = np.zeros(nC, dtype=np.int64)
for i, m in enumerate(mc):
    am = abs(int(m))
    e_m = max(0, am.bit_length() - 5)                 # m6 恰入 [-32,31]
    mag = int(np.floor(am / (1 << e_m) + 0.5))        # 半向上
    if am == 0:
        m6, e_m = 0, 0
    else:
        m6 = int(np.sign(m) * min(mag, 31))          # 6b 有符号上界 +31（-32 可表示但对称取 31）
    m6c[i] = m6
    t6c[i] = 8 - e_m
assert np.all(np.abs(m6c) <= 32) and np.all(t6c >= -8) and np.all(t6c <= 8)

# ---------------- 相位 D：RTN 就近舍入（rn_en=1）----------------
nD = 60000
xd = gen_x(nD); md = gen_m(nD)
half, quar, eig = nD // 2, nD // 4, nD // 8
sd = np.concatenate([
    rng.randint(21, 28, size=half),                   # HB 真实标定域 s∈[21,27]
    rng.randint(9, 48, size=quar),                    # 全域 [9,47]
    np.full(eig, 8, dtype=np.int64),                  # s=8 → rn 退化（截断）
    rng.randint(9, 12, size=nD - half - quar - eig),  # 小 s：饱和边界撞舍入
]).astype(np.int64)
prod = xd.astype(np.int64) * md
rnd = np.where(sd >= 9, np.left_shift(np.int64(1), np.clip(sd - 1, 0, 62)),
               np.int64(0))
er = sat8((prod + rnd) >> sd)                         # >>>= np 算术右移
# 退化自检：s=8 行必须与 floor 公式一致
m8 = sd == 8
assert np.array_equal(er[m8], sat8(prod[m8] >> 8))

with open("rq_ctrl.mem", "w") as f:   # ★24_pcw_rtn：与 tb_rq.sv 读名对齐（R3C 历史坑：
    #   生成器写 ctrl.mem、TB 读 rq_ctrl.mem，靠手工改名凑合——现在生成侧对齐）
    f.write(f"{to_hex(nA,32)}\n{to_hex(nB,32)}\n{to_hex(nC,32)}\n"
            f"{to_hex(nD,32)}\n")
for tag, arr, bits in [("xa", xa, 27), ("ma", ma, 16), ("sa", sa, 8),
                       ("xb", xb, 27), ("mb", mb, 16), ("sb", sb, 8),
                       ("xc", xc, 27), ("mc", mc, 16),
                       ("m6c", m6c, 6), ("t6c", t6c, 5),
                       ("xr", xd, 27), ("mr", md, 16), ("sr", sd, 8),
                       ("er", er, 8)]:
    with open(f"{tag}.mem", "w") as f:
        for v in arr:
            f.write(to_hex(v, bits) + "\n")

# 相位 C 预估（python 侧参考，交叉核对 RTL）：m6 量化相对误差
relq = np.abs(mc - m6c * (2 ** (8 - t6c))) / np.maximum(np.abs(mc), 1)
print(f"[gen] nA={nA} nB={nB} nC={nC} nD={nD}")
print(f"[gen] 相位 D 覆盖：s=8 行 {int(m8.sum())}（截断退化）、s∈[9,47] {int((~m8).sum())}；"
      f"饱和命中 {int(np.sum((er == 127) | (er == -128)))} 行")
print(f"[gen] m6 量化相对误差: mean={relq.mean():.5f} max={relq.max():.5f} "
      f"p99={np.percentile(relq,99):.5f}")
