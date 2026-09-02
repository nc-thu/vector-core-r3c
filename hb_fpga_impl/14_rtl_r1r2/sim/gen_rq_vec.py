# -*- coding: utf-8 -*-
"""
gen_rq_vec.py — requant 二代（门 1）对拍向量生成
相位 A: s=8，混合 x 分布（全域随机 27b / 仿真 GEMM 累加器 / 边界值），m 混合
        （Q8.8 全域随机 + 正小尺度 + 极值）→ 全部精确变体 vs rq_v1 逐位对拍。
相位 B: s∈[8,47] 随机 → 仅 rq_v2(T_MAX=39) vs rq_v1（其余变体 s 域外）。
相位 C: s=8，m 全域 → rq_m6（m 量化 6b）偏差数据：分解 (m6,t6) 由本脚本算好，
        RTL 与 v1 输出差异落盘供统计（数值决策归用户）。
输出: ctrl.mem / xa,ma,sa / xb,mb,sb / xc,mc,m6c,t6c (.mem)
"""
import numpy as np

rng = np.random.RandomState(20260827)

def to_hex(v, bits):
    return f"{int(v) & ((1 << bits) - 1):0{(bits + 3) // 4}x}"

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
    return np.clip(x, -(1 << 26), (1 << 26))          # 27b 域

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

with open("ctrl.mem", "w") as f:
    f.write(f"{to_hex(nA,32)}\n{to_hex(nB,32)}\n{to_hex(nC,32)}\n")
for tag, arr, bits in [("xa", xa, 27), ("ma", ma, 16), ("sa", sa, 8),
                       ("xb", xb, 27), ("mb", mb, 16), ("sb", sb, 8),
                       ("xc", xc, 27), ("mc", mc, 16),
                       ("m6c", m6c, 6), ("t6c", t6c, 5)]:
    with open(f"{tag}.mem", "w") as f:
        for v in arr:
            f.write(to_hex(v, bits) + "\n")

# 相位 C 预估（python 侧参考，交叉核对 RTL）：m6 量化相对误差
relq = np.abs(mc - m6c * (2 ** (8 - t6c))) / np.maximum(np.abs(mc), 1)
print(f"[gen] nA={nA} nB={nB} nC={nC}")
print(f"[gen] m6 量化相对误差: mean={relq.mean():.5f} max={relq.max():.5f} "
      f"p99={np.percentile(relq,99):.5f}")
