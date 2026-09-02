# -*- coding: utf-8 -*-
"""check_streams_a3.py — a3 段描述符流静态 hazard 检查（2026-08-31）。

在 09_cbound/check_streams.py（a2 逐位沿用）基础上只加 op=6 AE_ACTV 分支：

  sub=0 ACTV  读 Y 区（行 r<m：lane r%16，字 y_base+(r//16)*n+j）+ 表区
              （全 16 lane × 256 字，地址由数据值决定，按全表保守建模），
              写回同一 Y 区
  sub=1 BIAS  读 Y 区 + lo/hi 表（2×ceil16(k) 字，j 列走 lane j%16），
              写回 Y 区（本流 0 站，保守实现）
  sub=2 NORM  同 BIAS 口径（本流 0 站）

其余检查（X 读、单写者、越界）与 a2 逐字相同。用法：
    python check_streams_a3.py [build_a3] [--max-report 20]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                '09_cbound'))
from golden_interp import decode, load_seq                      # noqa: E402


def check_build(build, max_report=20):
    man0 = json.load(open(os.path.join(
        build, 'segments', 'seg_0000', 'manifest.json')))
    P = man0['profile']
    ctxw, ww, cols, ddr_bytes = (P['CTX_WORDS'], P['W_WORDS'],
                                 P['COLS'], P['DDR_BYTES'])
    n_err = n_warn = n_info = 0
    reports = []
    seg_dirs = sorted(x[0] for x in os.walk(
        os.path.join(build, 'segments')))
    nseg = n_op6 = 0
    for sd in seg_dirs:
        seg = os.path.basename(sd)
        if not seg.startswith('seg_'):
            continue
        nseg += 1
        seq = load_seq(sd)
        ctx_w = np.full((16, ctxw), -1, np.int64)
        wram_w = np.full((cols, ww), -1, np.int64)
        op_of = [(d >> 252) & 0xF for d in seq]

        for pc, d in enumerate(seq):
            f = decode(d)
            op = f['op']
            if op == 15:
                break
            m, n, k = f['m'], f['n'], f['k']
            if op == 4:
                ln = f['dma_len']
                if f['dma_addr'] + ln > ddr_bytes:
                    reports.append((seg, pc, 'ERR',
                                    'DDR 越界 %x+%d' % (f['dma_addr'], ln)))
                    n_err += 1
                if f['b_src'] == 0:
                    b = np.arange(ln)
                    la, wo = b % 16, f['b_base'] + b // 16
                    if wo.max() >= ctxw:
                        reports.append((seg, pc, 'ERR', 'CTX 越界 LOAD'))
                        n_err += 1
                        continue
                    ctx_w[la, wo] = pc
                else:
                    b = np.arange(ln)
                    la, wo = b % cols, f['b_base'] + b // cols
                    if wo.max() >= ww:
                        reports.append((seg, pc, 'ERR', 'WRAM 越界 LOAD W'))
                        n_err += 1
                        continue
                    wram_w[la, wo] = pc
            elif op == 5:
                nw = f['dma_len'] // 16
                if f['dma_addr'] + f['dma_len'] > ddr_bytes:
                    reports.append((seg, pc, 'ERR', 'DDR 越界 STORE'))
                    n_err += 1
                wo = f['y_base'] + np.arange(nw)
                if wo.max() >= ctxw:
                    reports.append((seg, pc, 'ERR', 'CTX 越界 STORE'))
                    n_err += 1
                    continue
                sub = ctx_w[:, wo]
                if (sub < 0).any():
                    reports.append((seg, pc, 'INFO',
                                    'STORE 读未写 CTX(pad lane) %d 格'
                                    % int((sub < 0).sum())))
                    n_info += 1
            elif op == 3:
                jj = np.arange(n & 0xFF)
                kk = np.arange(k)
                gcol = f['rq_m'] + jj
                src_w = f['b_base'] + (gcol // 16)[:, None] * f['b_spad'] + kk
                src_l = gcol % 16
                if src_w.max() >= ctxw:
                    reports.append((seg, pc, 'ERR', 'CTX 越界 COPY 读'))
                    n_err += 1
                    continue
                vals = ctx_w[src_l[:, None], src_w]
                if (vals < 0).any():
                    reports.append((seg, pc, 'ERR',
                                    'COPY 读到未写 CTX %d 格'
                                    % int((vals < 0).sum())))
                    n_err += 1
                dst_w = f['a_base'] + kk
                if dst_w.max() >= ww:
                    reports.append((seg, pc, 'ERR', 'WRAM 越界 COPY 写'))
                    n_err += 1
                    continue
                wram_w[jj[:, None], dst_w] = pc
            elif op == 6:
                # ---- AE_ACTV：Y 区读写（行 r<m）+ 表区读 ----
                n_op6 += 1
                sub = f['b_src']
                ii = np.arange(m)
                jj = np.arange(n)
                y_w = f['y_base'] + (ii // 16)[:, None] * n + jj
                y_l = (ii % 16)[:, None]
                if y_w.max() >= ctxw:
                    reports.append((seg, pc, 'ERR', 'CTX 越界 op6 Y'))
                    n_err += 1
                    continue
                vals = ctx_w[y_l, y_w]
                if (vals < 0).any():
                    # 常数列（第 n-1 列）在 ACTV 前故意不写：增广时 ACTV 后
                    # 由 _const_stripe 覆盖、非增广时乘 W 零行，读到的值不
                    # 进任何结果——只有这一形态算 INFO，其余列是真 hazard。
                    unw = np.argwhere(vals < 0)
                    cols_unw = np.unique(unw[:, 1] % n)
                    benign = (len(cols_unw) == 1 and cols_unw[0] == n - 1)
                    reports.append((seg, pc, 'INFO' if benign else 'ERR',
                                    ('op6 读未写 CTX(常数列) %d 格'
                                     if benign else
                                     'op6 读到未写 CTX %d 格')
                                    % int((vals < 0).sum())))
                    n_info += benign
                    n_err += (not benign)
                tw = f['b_base'] + np.arange(
                    256 if sub == 0 else 2 * ((k + 15) // 16))
                if tw.max() >= ctxw:
                    reports.append((seg, pc, 'ERR', 'CTX 越界 op6 表'))
                    n_err += 1
                else:
                    tv = ctx_w[:, tw]          # 表映像 16 lane 复制，全读
                    if (tv < 0).any():
                        reports.append((seg, pc, 'ERR',
                                        'op6 表读到未写 CTX %d 格'
                                        % int((tv < 0).sum())))
                        n_err += 1
                ctx_w[y_l, y_w] = pc
            elif op in (0, 1, 2):
                ii = np.arange(m)
                kk = np.arange(k)
                a_w = f['a_base'] + (ii // 16)[:, None] * k + kk
                a_l = (ii % 16)[:, None]
                if a_w.max() >= ctxw:
                    reports.append((seg, pc, 'ERR', 'CTX 越界 GEMM A'))
                    n_err += 1
                    continue
                aw = ctx_w[a_l, a_w]
                if (aw < 0).any():
                    reports.append((seg, pc, 'ERR',
                                    'GEMM A 读到未写 CTX %d 格'
                                    % int((aw < 0).sum())))
                    n_err += 1
                jj = np.arange(f['b_spad'])
                b_w = f['b_base'] + kk
                if b_w.max() >= ww:
                    reports.append((seg, pc, 'ERR', 'WRAM 越界 GEMM B'))
                    n_err += 1
                    continue
                bw = wram_w[jj[:, None], b_w]
                if (bw < 0).any():
                    reports.append((seg, pc, 'ERR',
                                    'GEMM B 读到未写 WRAM %d 格'
                                    % int((bw < 0).sum())))
                    n_err += 1
                else:
                    uw = np.unique(bw)
                    if len(uw) > 1 and any(op_of[x] != 4 for x in uw):
                        reports.append((seg, pc, 'WARN',
                                        'GEMM B 混写者含非 LOAD（%s）'
                                        % [int(x) for x in uw[:6]]))
                        n_warn += 1
                m16 = ((m + 15) // 16) * 16
                cc = np.arange(f['b_spad'])
                if f['y_tr']:
                    c = f['j0'] + cc
                    c = c[c < n]
                    y_w = f['y_base'] + (c // 16)[:, None] * m16 + ii
                    if y_w.max() >= ctxw:
                        reports.append((seg, pc, 'ERR',
                                        'CTX 越界 GEMM Y(tr)'))
                        n_err += 1
                        continue
                    ctx_w[(c % 16)[:, None], y_w] = pc
                else:
                    y_w = f['y_base'] + (ii // 16)[:, None] * n \
                        + f['j0'] + cc
                    if y_w.max() >= ctxw:
                        reports.append((seg, pc, 'ERR', 'CTX 越界 GEMM Y'))
                        n_err += 1
                        continue
                    ctx_w[(ii % 16)[:, None], y_w] = pc
                if op == 1:
                    cn = np.arange(n)
                    s_w = f['y_base'] + (ii // 16)[:, None] * n + cn
                    if s_w.max() >= ctxw:
                        reports.append((seg, pc, 'ERR', 'CTX 越界 SM'))
                        n_err += 1
                        continue
                    sw = ctx_w[(ii % 16)[:, None], s_w]
                    if (sw < 0).any():
                        reports.append((seg, pc, 'ERR', 'SM 读到未写 CTX'))
                        n_err += 1
                    ctx_w[(ii % 16)[:, None], s_w] = pc
        if n_err > 8 and len(reports) > max_report * 30:
            print('  ……错误过多，提前停止')
            break
    print('检查 %s：%d 段，op6=%d，ERR=%d WARN=%d INFO(pad)=%d'
          % (build, nseg, n_op6, n_err, n_warn, n_info))
    seen = set()
    shown = 0
    for seg, pc, lvl, msg in reports:
        key = (lvl, msg.split('（')[0].split(' %')[0])
        if key in seen and shown > max_report:
            continue
        seen.add(key)
        print('  [%s] %s pc=%d %s' % (lvl, seg, pc, msg))
        shown += 1
        if shown > max_report * 4:
            print('  ……（截断）')
            break
    return n_err, n_warn


if __name__ == '__main__':
    build = sys.argv[1] if len(sys.argv) > 1 else 'build_a3'
    e, w = check_build(build)
    sys.exit(1 if e else 0)
