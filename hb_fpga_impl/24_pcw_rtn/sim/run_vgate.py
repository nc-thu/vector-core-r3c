#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_vgate.py — 在服务器上跑 3 个代表段，比对位精确 + 周期。
用法：python run_vgate.py  （在 /tmp/ae_vgate/ 跑）
每段：+MODE=1 +PF=1 +SEQ +DDRIMG +DUMP → 读 dump → 比对 golden_ddr.bin
"""
import json, os, re, subprocess, sys
import numpy as np

ROOT = '/tmp/ae_vgate'
BIN = os.path.join(ROOT, 'obj_dir', 'Vtb_ae_v')
STAGE = os.path.join(ROOT, 'stage')
SEGS = ['seg_0600', 'seg_0529', 'seg_0602']
WD = '/tmp/ae_vgate/runwd'
CYC_RE = re.compile(r'cycles=(\d+) gemm=(\d+) dma=(\d+) mac_total=(\d+)')


def read_dump_v(path, n):
    """全量 dump.mem（每行一字节 hex，可能有 @addr 段头）→ uint8 数组。"""
    arr = np.zeros(n, dtype=np.uint8)
    base = 0
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith('@'):
                base = int(s[1:], 16)
                continue
            if base < n:
                arr[base] = int(s, 16)
                base += 1
    return arr


def run_seg(seg):
    seg_dir = os.path.join(STAGE, seg)
    seq_mem = os.path.join(seg_dir, 'seq.mem')
    ddr_init = os.path.join(seg_dir, 'ddr_init.mem')
    dump = os.path.join(WD, 'dump_%s.mem' % seg)
    # TB 的 $readmemh(ddr_f, ddr) 在独立 initial 块里，与 +DDRIMG 赋值有竞争。
    # 把 ddr_init.mem 拷到 cwd 作默认名兜底（与 rtl_seg.py 同口径）。
    # RTL initial $readmemh("seq.mem") 也需 cwd 有 seq.mem（+SEQ 端口装载会覆盖）。
    import shutil
    cwd_init = os.path.join(WD, 'ddr_init.mem')
    cwd_seq = os.path.join(WD, 'seq.mem')
    shutil.copy(ddr_init, cwd_init)
    shutil.copy(seq_mem, cwd_seq)
    if os.path.exists(dump):
        os.remove(dump)
    # 跑 RTL：MODE=1 (PRIM) + PF=1 (预取开，验证 R2)
    r = subprocess.run([BIN, '+MODE=1', '+PF=1',
                        '+SEQ=' + seq_mem, '+DDRIMG=' + cwd_init,
                        '+DUMP=' + dump],
                       cwd=WD, capture_output=True, text=True, timeout=900)
    out = r.stdout + r.stderr
    m = CYC_RE.search(out)
    if m is None:
        print('  [%s] RTL 无输出: %s' % (seg, out[-400:]))
        return None
    cyc = int(m.group(1)); gemm = int(m.group(2)); dma = int(m.group(3))
    mac = int(m.group(4))
    # 读 dump
    ddr_rtl = read_dump_v(dump, 524288)
    # 比对 golden
    golden = np.fromfile(os.path.join(seg_dir, 'golden_ddr.bin'),
                         dtype=np.uint8)
    regions = json.load(open(os.path.join(seg_dir, 'out_regions.json')))
    mc = json.load(open(os.path.join(seg_dir, 'model_cyc.json')))
    # 全 DDR 比对（只比对非零区，因为 dump 是全量但 init 是稀疏）
    # 但 golden 是全量 512KB，dump 也是全量 512KB → 直接全比
    diff_total = int(np.sum(golden != ddr_rtl))
    total = len(golden)
    # 输出区逐区
    pass_bytes = 0
    total_bytes = 0
    for (addr, n_words) in regions:
        g = golden[addr:addr + n_words]
        r2 = ddr_rtl[addr:addr + n_words]
        d = int(np.sum(g != r2))
        total_bytes += n_words
        if d == 0:
            pass_bytes += n_words
        else:
            print('  [%s] 输出区 ddr=%d 不一致 %d/%d' % (seg, addr, d, n_words))
    # 周期比对
    model_cal = mc['model_cal']
    model_exact = mc['exact']
    dev_cal = 100.0 * (cyc - model_cal) / model_cal if model_cal else 0
    dev_exact = 100.0 * (cyc - model_exact) / model_exact if model_exact else 0
    print('  [%s] cycles=%d  gemm=%d  dma=%d  mac=%d' %
          (seg, cyc, gemm, dma, mac))
    print('  [%s] model_cal=%d(±15%%门=[%d,%d])  偏差=%.1f%%' %
          (seg, model_cal, int(model_cal * 0.85), int(model_cal * 1.15),
           dev_cal))
    print('  [%s] model_exact=%d  偏差=%.1f%%' % (seg, model_exact, dev_exact))
    print('  [%s] 位精确: 输出区 %d/%d 字节 PASS  全DDR diff=%d/%d' %
          (seg, pass_bytes, total_bytes, diff_total, total))
    return dict(seg=seg, cyc=cyc, gemm=gemm, dma=dma, mac=mac,
                model_cal=model_cal, model_exact=model_exact,
                dev_cal=dev_cal, dev_exact=dev_exact,
                pass_bytes=pass_bytes, total_bytes=total_bytes,
                diff_total=diff_total)


def main():
    os.makedirs(WD, exist_ok=True)
    results = []
    for seg in SEGS:
        print('\n=== %s ===' % seg)
        r = run_seg(seg)
        if r:
            results.append(r)
    print('\n=== 汇总 ===')
    for r in results:
        ok = 'PASS' if r['pass_bytes'] == r['total_bytes'] and \
                       r['diff_total'] == 0 else 'FAIL'
        cyc_ok = 'IN_15%' if abs(r['dev_cal']) <= 15 else 'OUT_15%'
        print('%s %s  cyc=%d  model_cal=%d(%.1f%%)  exact=%d(%.1f%%)  %s  out=%d/%d  ddr_diff=%d' %
              (r['seg'], ok, r['cyc'], r['model_cal'], r['dev_cal'],
               r['model_exact'], r['dev_exact'], cyc_ok,
               r['pass_bytes'], r['total_bytes'], r['diff_total']))


if __name__ == '__main__':
    main()
