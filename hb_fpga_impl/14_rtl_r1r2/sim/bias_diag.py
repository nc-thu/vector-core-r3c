# 最小 BIAS 诊断：单用例 BIAS，检查 RTL vs 黄金
import numpy as np
import subprocess, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "spec"))
from norm_gold import build_image, engine_row

CTX_WORDS = 13824
rng = np.random.default_rng(42)

# 构建一个简单 BIAS 用例
ctx = np.zeros((16, CTX_WORDS), dtype=np.int64)
# 填随机数据
ctx[:, :] = rng.integers(-128, 128, size=(16, CTX_WORDS))

# BIAS 参数
m, n, k = 16, 8, 8
y_base, tbl_base = 100, 200
rqm, rqs = -384, 4

# 构造 bias 表
bj = np.array([1764, 2448, -1306, 2737, 675, -2649, -2780, 331], dtype=np.int64)
nlo = (k + 15) // 16
for j in range(k):
    lo = int(bj[j]) & 0xFF
    hi = (int(bj[j]) >> 8) & 0xFF
    ctx[j % 16, tbl_base + j // 16] = lo if lo < 128 else lo - 256
    ctx[j % 16, tbl_base + nlo + j // 16] = hi if hi < 128 else hi - 256

# 写 ctx0
with open("diag_ctx0.mem", "w") as f:
    for a in range(CTX_WORDS):
        w = 0
        for L in range(16):
            w |= (int(ctx[L, a]) & 0xFF) << (8 * L)
        f.write(f"{w:032X}\n")

# 描述符
def desc6(submode, m, n, k, y_base, b_base, rq_m=0, rq_s=0, rq_m2=0):
    v = (6 << 252) | (submode << 246) | (m << 228) | (n << 212) | (k << 196)
    v |= (y_base << 136) | (b_base << 156)
    v |= ((rq_m & 0xFFFF) << 104) | (rq_s << 96)
    v |= ((rq_m2 & 0xFFFF) << 120)
    v |= 0xF << 92
    return v

v = desc6(1, m, n, k, y_base, tbl_base, rqm, rqs)
with open("diag_cases.mem", "w") as f:
    f.write(f"{v:064X}\n")

# 黄金
def sat8(x):
    return int(np.clip(x, -128, 127))

for row in range(m):
    lane, base = row % 16, y_base + (row // 16) * n
    for j in range(n):
        y = int(ctx[lane, base + j])
        ctx[lane, base + j] = sat8((y * rqm + int(bj[j])) >> rqs)

with open("diag_exp.mem", "w") as f:
    for L in range(16):
        for a in range(CTX_WORDS):
            f.write(f"{int(ctx[L, a]) & 0xFF:02X}\n")

print("诊断文件已生成: diag_ctx0.mem / diag_cases.mem / diag_exp.mem")
print(f"BIAS: m={m} n={n} k={k} y={y_base} tbl={tbl_base} rqm={rqm} rqs={rqs}")
print(f"bj = {bj}")

# 前 8 个输入和黄金输出
ctx2 = np.zeros((16, CTX_WORDS), dtype=np.int64)
for a2, line in enumerate(open("diag_ctx0.mem").readlines()):
    w = int(line.strip(), 16)
    for L in range(16):
        ctx2[L, a2] = (w >> (8*L)) & 0xFF
        if ctx2[L, a2] >= 128: ctx2[L, a2] -= 256
print("\nrow 0 (lane 0, addr 100..107):")
for j in range(8):
    y = int(ctx2[0, 100 + j])
    gold = sat8((y * rqm + int(bj[j])) >> rqs)
    print(f"  j={j} addr={100+j}: input={y:4d} bj={int(bj[j]):6d} raw={(y*rqm+int(bj[j])):8d} shifted={(y*rqm+int(bj[j]))>>rqs:6d} gold={gold:4d}")
