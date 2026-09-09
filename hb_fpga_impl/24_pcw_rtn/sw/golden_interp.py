# -*- coding: utf-8 -*-
"""golden_interp.py — 段黄金解释器 + DDR 镜像构建器（与 RTL 位精确）

用法（模块）：
  from golden_interp import build_ddr_image, run_segment, load_seq
  img = build_ddr_image(seg_dir, blob_path, act_data)   # 全 DDR 初值
  ddr = run_segment(load_seq(seg_dir), img, P)          # 黄金执行
  # 与 segment_runner 的 dump 逐字节比对

执行语义照抄 hw_zcu104/sim/gen_vectors.py 的 run()（唯一权威）：
  requant y = sat8((acc·m) >>> s)   纯 floor
  softmax  e=EXP[min(mx-v,128)] Q12，quo=floor(127·2^30/Σe)，P=min((e·quo)>>>30,127)
  CTX      lane = 行 mod 16，addr = (行 div16)*pitch + k
  WRAM     lane = 列 j，addr = k（[COLS, W_WORDS]）
  LOAD CTX 字节流 k-major（byte b → lane=b%16, addr=b//16）
  LOAD W   字节流按「每 k 恰好 COLS 字节」路由
  STORE    字节流 word-major（word w 的 16 lane 连续 16 字节）
黄金 CTX 零初始化——RTL CTX 上电为 X，两者一致当且仅当段内所有被读格子
先写后读（编译器零槽预清零契约，本解释器就是来验证这件事的）。

pcW+RTN 扩展（2026-09-04，24_pcw_rtn；与 fast_interp 逐位一致）：
  requant 就近舍入（RTN）：rq_m bit15 = RTN 使能，置位时
      y = sat8((acc·m + 2^(s-1)) >>> s)      （s>=1，编译器强制 s>=9）
  逐列系数：rq_s bit7 = 逐列模式，系数来自 sf 寄存器组（OP_SF 装载）：
      y[:, j] = sat8((acc[:, j]·m_j + 2^(s_j-1)) >>> s_j)
  OP_SF（op=14）：256b 描述符字装 10 个 24b 槽（槽 i = bits[24i+23:24i]，
  值 = m<<8 | s），bits[251:240] = 本字填充的起始槽号 slot0。
  RTL（rq_v2 内部 t=s-8 口径）等价式：y = sat8((x·mh + ((x·ml)>>>8)
  + 2^(s-9)) >>> (s-8))，s>=9 —— 与上式逐位相等（整数恒等式，见
  24_pcw_rtn/REPORT_SW.md 的证明）。
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

EXP = [int(np.floor((2.0 ** (-d / 16.0)) * 4096 + 0.5)) for d in range(129)]


def sat8(x):
    return np.clip(x, -128, 127).astype(np.int64)


def requant(acc, m, s, rn=0):
    """floor（rn=0）/ RTN（rn=2^(s-1)）requant。"""
    return sat8(acc.astype(np.int64) * np.int64(m) + np.int64(rn)
                >> np.int64(s))


def softmax_rows(S, n_cols, causal):
    P = np.zeros_like(S)
    for i in range(S.shape[0]):
        vlen = min(i + 1, n_cols) if causal else n_cols
        row = S[i, :vlen].astype(np.int64)
        if vlen == 0:
            continue
        mx = int(row.max())
        e = np.array([EXP[min(mx - int(v), 128)] for v in row], dtype=np.int64)
        se = int(e.sum())
        quo = (127 << 30) // se
        p = (e * quo) >> 30
        P[i, :vlen] = np.minimum(p, 127).astype(np.int8)
    return P


# ---------------- 描述符解码 ----------------
def decode(d):
    f = {}
    f['op'] = (d >> 252) & 0xF
    f['m'] = (d >> 228) & 0xFFFF
    f['n'] = (d >> 212) & 0xFFFF
    f['k'] = (d >> 196) & 0xFFFF
    f['a_base'] = (d >> 176) & 0xFFFFF
    f['b_base'] = (d >> 156) & 0xFFFFF
    f['y_base'] = (d >> 136) & 0xFFFFF
    f['b_spad'] = (d >> 120) & 0xFFFF
    raw_m = (d >> 104) & 0xFFFF
    f['rq_rn'] = bool(raw_m & 0x8000)     # bit15：RTN 就近舍入使能
    f['rq_m'] = raw_m & 0x7FFF
    raw_s = (d >> 96) & 0xFF
    f['rq_pc'] = bool(raw_s & 0x80)       # bit7：逐列系数（sf 寄存器组）
    f['rq_s'] = raw_s & 0x7F
    f['dma_len'] = (d >> 61) & 0x3FFFF
    f['dma_addr'] = (d >> 29) & 0xFFFFFFFF
    f['j0'] = (d >> 62) & 0xFFFF
    f['sm_causal'] = (d >> 245) & 1
    f['y_tr'] = (d >> 244) & 1
    f['b_src'] = (d >> 246) & 7
    return f


def load_seq(seg_dir):
    seq = []
    with open(os.path.join(seg_dir, 'seq.mem')) as f:
        for line in f:
            line = line.strip()
            if line:
                seq.append(int(line, 16))
    return seq


# ---------------- 黄金执行 ----------------
def _s8(u):
    """DDR 字节 → 有符号 int8 值（CTX/WRAM 存有符号数）"""
    u = np.asarray(u, dtype=np.int64)
    return np.where(u >= 128, u - 256, u)


def run_segment(seq, ddr_img, P):
    cols, ctxw, ww = P['COLS'], P['CTX_WORDS'], P['W_WORDS']
    ddr = ddr_img.copy()
    ctx = np.zeros((16, ctxw), dtype=np.int64)
    wram = np.zeros((cols, ww), dtype=np.int64)
    sf_m = [0] * cols                  # 逐列 requant 系数（OP_SF 装载）
    sf_s = [0] * cols
    macs = 0
    for pc, d in enumerate(seq):
        f = decode(d)
        op = f['op']
        if op == 15:
            break
        m, n, k = f['m'], f['n'], f['k']
        if op == 14:                                   # SF 系数装载
            slot0 = (d >> 240) & 0xFFF
            for i in range(10):
                v = (d >> (24 * i)) & 0xFFFFFF
                jj = slot0 + i
                if jj < cols:
                    sf_m[jj] = (v >> 8) & 0xFFFF
                    sf_s[jj] = v & 0xFF
        elif op == 4:                                  # LOAD
            if f['b_src'] == 0:                        # CTX k-major
                for b in range(f['dma_len']):
                    ctx[b % 16, f['b_base'] + b // 16] = \
                        _s8(ddr[f['dma_addr'] + b])
            else:                                      # W 每 k 行 COLS 字节
                wj, wk = 0, 0
                for b in range(f['dma_len']):
                    wram[wj, f['b_base'] + wk] = \
                        _s8(ddr[f['dma_addr'] + b])
                    wj += 1
                    if wj == cols:
                        wj = 0
                        wk += 1
        elif op == 5:                                  # STORE word-major
            for w in range(f['dma_len'] // 16):
                for half in range(2):
                    for q in range(8):
                        lane = half * 8 + q
                        ddr[f['dma_addr'] + w * 16 + half * 8 + q] = \
                            ctx[lane, f['y_base'] + w]
        elif op == 3:                                  # COPY CTX→WRAM
            src_j0 = f['rq_m']
            for j in range(n & 0xFF):
                gcol = src_j0 + j
                for kk in range(k):
                    wram[j, f['a_base'] + kk] = \
                        ctx[gcol % 16, f['b_base'] + (gcol // 16)
                            * f['b_spad'] + kk]
        elif op in (0, 1, 2):                          # GEMM 族
            macs += ((m + 15) // 16) * 16 * cols * k
            A = np.zeros((m, k), dtype=np.int64)
            for i in range(m):
                for kk in range(k):
                    A[i, kk] = ctx[i % 16, f['a_base'] + (i // 16) * k + kk]
            B = wram[:, f['b_base']:f['b_base'] + k].T[:, :f['b_spad']]
            acc = A @ B
            if f['rq_pc']:                             # 逐列系数 + RTN
                nl = f['b_spad']
                assert all(sf_m[j] >= 1 for j in range(nl)), \
                    f'pc={pc}: sf 槽未装载（m=0）'
                Y = np.zeros((m, nl), dtype=np.int64)
                for j in range(nl):
                    mj, sj = sf_m[j], sf_s[j]
                    rj = (1 << (sj - 1)) if f['rq_rn'] else 0
                    Y[:, j] = requant(acc[:, j], mj, sj, rj)
            else:
                rn = (1 << (f['rq_s'] - 1)) if f['rq_rn'] else 0
                Y = requant(acc, f['rq_m'], f['rq_s'], rn)
            m16 = ((m + 15) // 16) * 16
            Yp = np.zeros((m16, f['b_spad']), dtype=np.int64)
            Yp[:m, :] = Y
            if f['y_tr']:
                for i in range(m16):
                    if i >= m:
                        continue
                    for lc in range(f['b_spad']):
                        c = f['j0'] + lc
                        if c >= n:
                            continue
                        ctx[c % 16, f['y_base'] + (c // 16) * m16 + i] = \
                            Yp[i, lc]
            else:
                for i in range(m16):
                    if i >= m:
                        continue
                    for c in range(f['b_spad']):
                        ctx[i % 16, f['y_base'] + (i // 16) * n
                            + f['j0'] + c] = Yp[i, c]
            if op == 1:                                # SM16 softmax
                S = np.zeros((m, n), dtype=np.int64)
                for i in range(m):
                    for c in range(n):
                        S[i, c] = ctx[i % 16, f['y_base'] + (i // 16) * n + c]
                Pm = softmax_rows(S.astype(np.int8), n, f['sm_causal'])
                for i in range(m):
                    for c in range(n):
                        ctx[i % 16, f['y_base'] + (i // 16) * n + c] = Pm[i, c]
        else:
            raise AssertionError(f'pc={pc}: 未定义 op={op}')
    return ctx, ddr, dict(macs=macs)


# ---------------- DDR 镜像构建 ----------------
def build_ddr_image(seg_dir, blob_path, act_data, P):
    """act_data: {输入名: int8 ndarray}（host 提供，按声明布局填充）。
    返回全 DDR 初值（长度 DDR_BYTES，未声明区为 0）。"""
    man = json.load(open(os.path.join(seg_dir, 'manifest.json'),
                         encoding='utf-8'))
    blob = np.fromfile(blob_path, dtype=np.uint8)
    ddr = np.zeros(P['DDR_BYTES'], dtype=np.uint8)
    for w in man['weights']:
        ddr[w['ddr']:w['ddr'] + w['blob_len']] = \
            blob[w['blob_off']:w['blob_off'] + w['blob_len']]
    if man.get('zero_addr') is not None:
        pass        # 零槽保持 0
    for e in man['inputs']:
        data = act_data.get(e['name'])
        if data is None:
            raise KeyError(f'缺输入数据 {e["name"]}')
        a = np.ascontiguousarray(data, dtype=np.uint8).reshape(-1)
        ddr[e['ddr']:e['ddr'] + len(a)] = a
    return ddr


def write_mem(path, arr):
    """一行一字节 hex（$readmemh 兼容）"""
    with open(path, 'w') as f:
        for v in np.asarray(arr, dtype=np.uint8).reshape(-1):
            f.write(f'{v:02X}\n')


def read_dump(path):
    # $writememh 每 16 字节插一行 "// 0x…" 地址注释，跳过
    return np.array([int(l, 16) for l in open(path)
                     if l.strip() and not l.startswith('//')],
                    dtype=np.uint8)


def read_dump_lenient(path):
    """'xx' 记为 256，保持逐字节对位（配合稀疏初始化用）。"""
    out = []
    for l in open(path):
        l = l.strip()
        if not l or l.startswith('//'):
            continue
        out.append(int(l, 16) if l != 'xx' else 256)
    return np.array(out, dtype=np.int32)


def declared_ranges(seg_dir, P):
    """manifest 里声明的 DDR 区间：权重 ∪ 输入 ∪ 输出 ∪ 零槽。"""
    man = json.load(open(os.path.join(seg_dir, 'manifest.json'),
                         encoding='utf-8'))
    rs = []
    for w in man['weights']:
        rs.append((w['ddr'], w['ddr'] + w['blob_len']))
    for e in man['inputs']:
        rs.append((e['ddr'], e['ddr'] + e['words'] * 16))
    for e in man['outputs']:
        rs.append((e['ddr'], e['ddr'] + e['words'] * 16))
    z = man.get('zero_addr')
    if z is not None:
        rs.append((z, z + P['ZERO_SLOT'] * 16))
    return rs


def write_mem_sparse(path, ddr, ranges):
    """只写给定区间的 @ 分区 mem（$readmemh 原生支持，其余 DDR 保持 X）。
    8MB 全量文本在 iverilog 里加载要分钟级，稀疏写降到秒级。"""
    with open(path, 'w') as f:
        for a, b in sorted(ranges):
            f.write(f'@{a:X}\n')
            for v in np.asarray(ddr[a:b], dtype=np.uint8):
                f.write(f'{v:02X}\n')


def compare_ranges(golden_ddr, dump_arr, ranges):
    """只比对声明区间：区间外 RTL 保持 X 属正常（稀疏初始化）。
    区间内任何字节不一致都算失败（含 xx=256，可抓 CTX 未初始化泄漏）。"""
    g = np.asarray(golden_ddr)
    d = np.asarray(dump_arr)
    if len(g) != len(d):
        return False, f'长度不一致 golden={len(g)} dump={len(d)}'
    bad = []
    for a, b in ranges:
        m = np.nonzero(g[a:b] != d[a:b])[0]
        bad.extend((a + int(x)) for x in m[:8])
        if len(m) > 8:
            break
    if not bad:
        return True, ''
    x = bad[0]
    return False, (f'{len(bad)}+ 字节不一致，首个 @0x{x:X}: '
                   f'golden=0x{int(g[x]):02X} rtl=0x{int(d[x]):02X}；'
                   '前几个偏移：' + ','.join(f'0x{y:X}' for y in bad[:8]))


def compare(golden_ddr, dump_arr):
    g = np.asarray(golden_ddr, dtype=np.uint8)
    d = np.asarray(dump_arr, dtype=np.uint8)
    if len(g) != len(d):
        return False, f'长度不一致 golden={len(g)} dump={len(d)}'
    bad = np.nonzero(g != d)[0]
    if len(bad) == 0:
        return True, ''
    return False, (f'{len(bad)} 字节不一致，首个 @0x{bad[0]:X}: '
                   f'golden=0x{g[bad[0]]:02X} rtl=0x{d[bad[0]]:02X}；'
                   f'前 16 个偏移：' + ','.join(f'0x{x:X}' for x in bad[:16]))
