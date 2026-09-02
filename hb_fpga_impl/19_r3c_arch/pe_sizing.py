# -*- coding: utf-8 -*-
"""pe_sizing.py — PE 阵列规模数据驱动选型（2026-09-01）

问题：现阵列 16 行 × 108 列 = 1728 PE，DSP48E2 在 XCZU7EV 上占满 100%。
要减 100~300 个 DSP 提高可移植性，问 COLS 减到多少最合适。

方法：对 a3 真实流（2580 段）的每条 GEMM 描述符，在候选 COLS 下把
n_loc > COLS 的描述符切成 ceil(n_loc/COLS) 个子组（A 矩阵每组要重喂数，
这是窄阵列的固有代价），子组周期沿用 R3C 模型 tile = max(k+2, 68+wb)。
校准系数、lane 口径与 r3c_model.py 完全一致。

注意：本脚本是「按 108 编译的固定流 + 硬件重切」口径（上界，偏悲观）。
按新 COLS 重跑编译器只能消除窄尾组的取整浪费，但 A 重喂次数
ceil(n/COLS) 是任何输出驻留式脉动阵列的下界，结论不变。

输出：pe_sizing.json + 控制台对比表。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
A3 = os.path.join(os.path.dirname(HERE), '12_actv', 'a3')
sys.path.insert(0, A3)
from acct_a3 import decode, wb_cycles, DRAIN, DALIGN, ROWS  # noqa: E402

BUILD = os.path.join(A3, 'build_a3')
CAL = 336.44 / 319.99          # a3 校准系数
CANDS = [108, 104, 100, 96, 92]  # 92/96/100/104 均为 4 的倍数（RQ_SH=4）

# lane 模型分量（与 r3c_model.py 一致）
X, V, W = 184.0, 108.1, 218.4
COPY, ACTV, THETA = 40.3, 10.0, 40.0
BYTES = dict(ctx=680.25, w=606.92, st=594.29)


def logical_gems(segs_dir):
    """重建逻辑 GEMM：同段内连续 op∈{0,1,2} 描述符，若 (m,k,a_base,y_base,y_tr)
    相同、j0 链式递进（j0[i+1]=j0[i]+n_loc[i]）、中间只夹 op∈{3,4}（B 重驻留/
    LOAD_W，属同族正常结构），合并为一个逻辑宽度 W。a_base 不同的（im2col 每列组
    自带独立 A）不合并——每条描述符自己就是独立逻辑 GEMM。"""
    gems = []                                   # (W, k, mt, y_tr, j0)
    for s in sorted(os.listdir(segs_dir)):
        cur = None                              # 上一条 GEMM 描述符
        pend = None                             # 待闭合的逻辑组
        for line in open(os.path.join(segs_dir, s, 'seq.mem')):
            line = line.strip()
            if not line:
                continue
            d = decode(int(line, 16))
            if d['op'] in (0, 1, 2):
                if pend is not None:
                    key = (d['m'], d['k'], d['a_base'], d['y_base'], d['y_tr'])
                    pkey = (pend['m'], pend['k'], pend['a_base'],
                            pend['y_base'], pend['y_tr'])
                    chained = d['j0'] == pend['j0'] + pend['b_spad']
                    if key == pkey and chained and not pend['sep']:
                        pend['W'] += d['b_spad']     # 同一逻辑 GEMM 续宽
                    else:
                        gems.append(pend)
                        pend = dict(W=d['b_spad'], k=d['k'],
                                    mt=(d['m'] + ROWS - 1) // ROWS,
                                    y_tr=d['y_tr'], j0=d['j0'],
                                    m=d['m'], a_base=d['a_base'], sep=False)
                else:
                    pend = dict(W=d['b_spad'], k=d['k'],
                                mt=(d['m'] + ROWS - 1) // ROWS,
                                y_tr=d['y_tr'], j0=d['j0'],
                                m=d['m'], a_base=d['a_base'], sep=False)
                pend.update(m=d['m'], k=d['k'], a_base=d['a_base'],
                            y_base=d['y_base'], y_tr=d['y_tr'],
                            j0=d['j0'], b_spad=d['b_spad'], sep=False)
                cur = d
            elif d['op'] in (3, 4):
                if pend is not None:
                    pend['sep'] = False         # 允许夹 B 重驻留
            else:
                if pend is not None:
                    pend['sep'] = True          # 其他算子：截断逻辑组
                cur = None
        if pend is not None:
            gems.append(pend)
    return gems


def main():
    segs = sorted(os.listdir(os.path.join(BUILD, 'segments')))
    rows = []                                   # (m, k, n_loc, j0, y_tr, mt)
    for s in segs:
        for line in open(os.path.join(BUILD, 'segments', s, 'seq.mem')):
            line = line.strip()
            if not line:
                continue
            d = decode(int(line, 16))
            if d['op'] in (0, 1, 2):
                mt = (d['m'] + ROWS - 1) // ROWS
                rows.append((d['m'], d['k'], d['b_spad'], d['j0'],
                             d['y_tr'], mt))

    macs = sum(mt * ROWS * k * n for m, k, n, j0, tr, mt in rows)
    gems = logical_gems(BUILD + '/segments')
    out = {'meta': dict(date='2026-09-01', build='build_a3',
                        segs=len(segs), descs=len(rows), macs_G=macs / 1e9),
           'logical_gems': len(gems), 'cands': []}
    print('逻辑 GEMM 数：%d（描述符 %d 条）' % (len(gems), len(rows)))

    for C in CANDS:
        # ---- 口径 A：固定流（按 108 编译）+ 硬件重切（上界，偏悲观）----
        G = 0.0
        split_descs = 0
        for m, k, n, j0, tr, mt in rows:
            g = (n + C - 1) // C                # 子组数 = A 重喂次数
            if g > 1:
                split_descs += 1
            cyc = 0                             # 每行组：Σ 子组 tile
            for t in range(g):
                n_sub = min(n, (t + 1) * C) - t * C
                wb = wb_cycles(n_sub, j0 + t * C, tr)
                cyc += max(k + 2, DRAIN + DALIGN + 2 + wb)
            # 固定开销 4 拍每描述符一次（与 r3c_model 口径一致），末组读出链 +16
            G += mt * cyc + 4 + 16
        G *= CAL / 1e6                          # M 拍

        # ---- 口径 B：按新 COLS 重编译（逻辑 GEMM 整宽重切，均衡分组）----
        G2 = 0.0
        for gm in gems:
            W_, k_, mt_, tr_, j0_ = (gm['W'], gm['k'], gm['mt'],
                                     gm['y_tr'], gm['j0'])
            g = (W_ + C - 1) // C
            for t in range(g):
                n_sub = min(W_, (t + 1) * C) - t * C
                wb = wb_cycles(n_sub, j0_ + t * C, tr_)
                G2 += mt_ * max(k_ + 2, DRAIN + DALIGN + 2 + wb)
            G2 += 4 + 16
        G2 *= CAL / 1e6

        comp = G2 + COPY + ACTV
        tb = max(comp, X + V, W) + THETA
        hp_rd = (BYTES['ctx'] + BYTES['w']) / 64.0
        hp = max(comp, hp_rd, BYTES['st'] / 64.0) + THETA
        # 阵列口径利用率：MAC 总量 / (PE 数 × GEMM 拍数)——含读出/切组浪费
        util = macs / (16 * C * G2 * 1e6) * 100
        out['cands'].append(dict(
            cols=C, pe=16 * C, dsp_pct=16 * C / 1728 * 100,
            split_descs=split_descs, gemm_M=G, gemm_recomp_M=G2,
            gemm_delta_pct=(G2 / out['cands'][0]['gemm_recomp_M'] - 1) * 100
            if out['cands'] else 0.0,
            tb_M=tb, tb_ms=tb / 198.5, hp64_M=hp, hp64_ms=hp / 198.5,
            util_pct=util))

    with open(os.path.join(HERE, 'pe_sizing.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print('描述符 %d 条，MAC 总量 %.1f G/帧' % (len(rows), macs / 1e9))
    print('%-6s %-6s %-8s %-9s %-10s %-10s %-8s %-9s %-9s %s' % (
        'COLS', 'PE', 'DSP%', '切组描述符', '重切(M)', '重编译(M)', 'Δ重编译',
        'TB(ms)', 'HP64(ms)', '利用率'))
    for c in out['cands']:
        print('%-6d %-6d %-8.1f %-9d %-10.1f %-10.1f %-8.1f %-9.2f %-9.2f %.1f%%' % (
            c['cols'], c['pe'], c['dsp_pct'], c['split_descs'], c['gemm_M'],
            c['gemm_recomp_M'], c['gemm_delta_pct'], c['tb_ms'],
            c['hp64_ms'], c['util_pct']))


if __name__ == '__main__':
    main()
