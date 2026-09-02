# -*- coding: utf-8 -*-
"""actv_gold.py — AE_ACTV 微观对拍：随机向量生成 + numpy 黄金 + RTL dump 比对

用法：
  python actv_gold.py gen     # 生成 actv_ctx0.mem / actv_cases.mem / actv_ctx_exp.mem
  python actv_gold.py check   # 读 actv_ctx_out.mem（tb_ae_actv dump），逐字节比对

描述符编码与 01_rtl/sim/gen_vectors.py desc() 同一切片（op=6）：
  b_src[2:0]=子模式(0=ACTV 1=BIAS) m=行 n=列(=stride) k=BIAS 表长
  y_base=原地张量 CTX 基址  b_base=表映像 CTX 基址  rq_m/rq_s=BIAS 的 m(Q8.8)/s
表映像布局（与 DMA TAG_CTX 字节路由 lane=b%16, addr=base+b/16 一致）：
  ACTV：表项 x 复制在字 b_base+x 的全部 16 lane 槽（映像 256 字 = 512B）
  BIAS：项 j 的 lo 在 lane j%16 @ b_base + j//16，hi 同相位 @ b_base+NLO + j//16
        （NLO = ceil(k/16)，两区各 NLO 字）
"""
import sys
import numpy as np

CTX_WORDS = 1280
N_CASES = 8
rng = np.random.default_rng(20260831)


def desc6(submode, m, n, k, y_base, b_base, rq_m=0, rq_s=0):
    v = (6 << 252) | (submode << 246) | (m << 228) | (n << 212) | (k << 196)
    v |= (y_base << 136) | (b_base << 156)
    v |= ((rq_m & 0xFFFF) << 104) | (rq_s << 96)
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

    # 8 个用例：行组尾数/列宽/表长尾数/饱和角落全覆盖
    add_actv("A1 pad-row  m=18 n=20", 0, 18, 20, 64)
    add_actv("A2 2 full  m=32 n=5 ", 80, 32, 5, 320)
    add_actv("A3 1grp    m=7  n=33", 112, 7, 33, 576)
    add_bias("B1 k=n     m=18 n=20", 0, 18, 20, 832, 20, 256, 8)
    add_bias("B2 ktail   m=33 n=17", 48, 33, 17, 836, 17, 257, 8)
    add_bias("B3 neg-m   m=5  n=40", 100, 5, 40, 840, 40, -384, 4)
    add_bias("B4 sat     m=16 n=8 ", 140, 16, 8, 846, 8, 32767, 0, -30000, 30000)
    add_actv("A4 square  m=16 n=16", 148, 16, 16, 848)
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
                      c.get("rqm", 0), c.get("rqs", 0))
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
