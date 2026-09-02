# -*- coding: utf-8 -*-
"""r3c_model.py — R3 方案 C（PE 快照 + 行组两段流水）架构模型（2026-09-01）

把方案 C 写进周期模型：对 a3 真实流（2580 段）的每条 GEMM 描述符，
把行组周期从「串行全流程」改成 max(喂数, 读出链)：
  现状 tile = 1 + (k+2) + waitd(127) + DRAIN(64) + DALIGN(2) + 2 + wb + 1
  方案C tile = max(k+2, DRAIN+DALIGN+2+wb)        （两条流水并行，谁长谁定周期）
  每条描述符另有固定开销 2+2，末行组的读出链一次性暴露（+16 拍保守）。

口径：
  校准系数 1.0528 = 336.44M / 319.99M（a3 acct 校准，外样 99.9%）
  lane 模型沿用 13_rtl_plan/r1r2_matrix.py：总拍 = max(comp, X+V, W) + θ
  X=184.0 / V=108.1 / W=218.4 / COPY=40.3 / AE_ACTV=10.0 / θ=40.0（M 拍）
  HP64：读服务 = (680.25+606.92)MB / 64B/cyc，写服务 = 594.29MB / 64B/cyc

输出：r3c_model.json + 控制台汇总。纯模型，不碰 RTL。
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
A3 = os.path.join(os.path.dirname(HERE), '12_actv', 'a3')
sys.path.insert(0, A3)
from acct_a3 import decode, wb_cycles, DRAIN, DALIGN, ROWS  # noqa: E402

BUILD = os.path.join(A3, 'build_a3')
COLS = 108
CAL = 336.44 / 319.99          # 校准系数（a3 GEMM 校准/理想）
GEMM_CMD_OVH = 2

# lane 模型分量（a3 校准口径，M 拍）——与 r1r2_matrix.py 完全一致
X, V, W = 184.0, 108.1, 218.4
COPY, ACTV, THETA = 40.3, 10.0, 40.0
BYTES = dict(ctx=680.25, w=606.92, st=594.29)   # MB


def lane_total(comp, x, v, w, theta=THETA):
    tot = max(comp, x + v, w) + theta
    return tot


def main():
    segs = sorted(os.listdir(os.path.join(BUILD, 'segments')))
    n_grp = 0
    G0 = G1 = 0.0                    # 现状 / 方案C（理想拍）
    kgrp = Counter()                 # k -> 行组数
    ksave = Counter()                # k -> 省的拍
    for s in segs:
        for line in open(os.path.join(BUILD, 'segments', s, 'seq.mem')):
            line = line.strip()
            if not line:
                continue
            d = decode(int(line, 16))
            if d['op'] in (0, 1, 2):
                m, k = d['m'], d['k']
                n_loc, j0, y_tr = d['b_spad'], d['j0'], d['y_tr']
                mt = (m + ROWS - 1) // ROWS
                wb = wb_cycles(n_loc, j0, y_tr)
                tile0 = 1 + (k + 2) + 127 + DRAIN + DALIGN + 2 + wb + 1
                tile1 = max(k + 2, DRAIN + DALIGN + 2 + wb)
                G0 += 2 + GEMM_CMD_OVH + mt * tile0
                G1 += 2 + GEMM_CMD_OVH + mt * tile1 + 16   # 末组读出链整段暴露
                n_grp += mt
                kgrp[k] += mt
                ksave[k] += mt * (tile0 - tile1)

    g0 = G0 * CAL / 1e6
    g1 = G1 * CAL / 1e6
    out = {
        'meta': dict(date='2026-09-01', build='build_a3', segs=len(segs),
                     row_groups=n_grp, calib=CAL, note='方案C模型，未实施RTL'),
        'gemm': dict(base_M=g0, r3c_M=g1, saved_M=g0 - g1,
                     saved_pct=(G0 - G1) / G0 * 100,
                     feed_M=133.9, base_overhead_M=185.8),
        'lanes': {},
    }
    for tag, Gm in [('R1+R2 现状', g0), ('R3C', g1)]:
        comp = Gm + COPY + ACTV
        tb = lane_total(comp, X, V, W)
        hp_rd = (BYTES['ctx'] + BYTES['w']) / 64.0
        hp_wr = BYTES['st'] / 64.0
        hp = max(comp, hp_rd, hp_wr) + THETA
        out['lanes'][tag] = dict(
            comp_M=comp, tb_M=tb, tb_ms=tb / 198.5, tb_gemm_pct=Gm / tb * 100,
            hp64_M=hp, hp64_ms=hp / 198.5, hp64_gemm_pct=Gm / hp * 100)
    out['top_k'] = [
        dict(k=k, groups=kgrp[k], saved_M=ksave[k] / 1e6)
        for k in sorted(kgrp, key=lambda x: -ksave[x])[:8]]

    with open(os.path.join(HERE, 'r3c_model.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print('段 %d  行组 %d  校准 %.4f' % (len(segs), n_grp, CAL))
    print('GEMM: %.1fM -> %.1fM（省 %.1fM，%.1f%%）' % (
        g0, g1, g0 - g1, (G0 - G1) / G0 * 100))
    for tag, v in out['lanes'].items():
        print('%s: comp %.1f | TB %.1fM %.2fms GEMM %.1f%% | HP64 %.1fM %.2fms GEMM %.1f%%'
              % (tag, v['comp_M'], v['tb_M'], v['tb_ms'], v['tb_gemm_pct'],
                 v['hp64_M'], v['hp64_ms'], v['hp64_gemm_pct']))
    print('top-k 省拍:', [(r['k'], round(r['saved_M'], 1)) for r in out['top_k']])


if __name__ == '__main__':
    main()
