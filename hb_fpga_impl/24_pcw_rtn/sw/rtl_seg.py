# -*- coding: utf-8 -*-
"""rtl_seg.py — RTL 段执行引擎（慢档，2026-08-31）。

每段起一个 Verilator Vtb_ae_v 进程（部署口径 MODE=1 + PF=1）：
  host DDR 图 → 稀疏 ddr_init.mem（只写非零块，Verilator 2 态未写区=0）
  → Vtb_ae_v +SEQ +DDRIMG +DUMP → dump.mem 全量 hex → numpy 向量化读回。
与 fast_interp.run_segment_fast 同签名（None, ddr, info），host_driver 直接换。
"""
import os
import re
import subprocess

import numpy as np

CYC_RE = re.compile(r'cycles=(\d+) gemm=(\d+) dma=(\d+) mac_total=(\d+)')

HEXV = np.full(256, 255, np.uint8)
for _i, _c in enumerate(b'0123456789abcdef'):
    HEXV[_c] = _i
for _i, _c in enumerate(b'0123456789ABCDEF'):
    HEXV[_c] = _i

LUT = np.zeros((256, 3), np.uint8)
for _i in range(256):
    LUT[_i] = np.frombuffer(('%02X\n' % _i).encode(), np.uint8)


def write_sparse_mem(path, img, gap=256):
    """只写非零块：@addr + 每行一字节 hex。返回写入字节数。"""
    nz = np.flatnonzero(img)
    if len(nz) == 0:
        open(path, 'w').write('')
        return 0
    brk = np.flatnonzero(np.diff(nz) > gap)
    starts = np.concatenate(([nz[0]], nz[brk + 1]))
    ends = np.concatenate((nz[brk] + 1, [nz[-1] + 1]))
    parts = []
    for s, e in zip(starts, ends):
        parts.append('@%X\n' % s)
        parts.append(LUT[img[s:e]].tobytes().decode())
    open(path, 'w').write(''.join(parts))
    return int((ends - starts).sum())


def read_dump_v(path):
    """全量 dump.mem（每行一字节 hex）→ uint8 数组，numpy 向量化。"""
    raw = open(path, 'rb').read()
    arr = np.frombuffer(raw, np.uint8)
    arr = arr[arr != 10]
    v = HEXV[arr]
    if (v == 255).any():
        i = int(np.flatnonzero(v == 255)[0])
        raise ValueError('dump 非 hex 字符 @%d: %r'
                         % (i, raw[max(0, i - 8):i + 8]))
    d = v.reshape(-1, 2)
    return ((d[:, 0] << 4) | d[:, 1]).astype(np.uint8)


def run_segment_rtl(seg_dir, img, P,
                    bin_path='/tmp/ae_v3/sim/obj_dir/Vtb_ae_v',
                    wd='/tmp/ae_hostdrv/rtlwd', seq_mem=None, keep=False):
    os.makedirs(wd, exist_ok=True)
    dmem = os.path.join(wd, 'ddr_init.mem')
    dump = os.path.join(wd, 'dump.mem')
    write_sparse_mem(dmem, img)
    if os.path.exists(dump):
        os.remove(dump)
    r = subprocess.run([bin_path, '+MODE=1', '+PF=1',
                        '+SEQ=' + (seq_mem or os.path.join(seg_dir, 'seq.mem')),
                        '+DDRIMG=' + dmem, '+DUMP=' + dump],
                       cwd=wd, capture_output=True, text=True, timeout=7200)
    m = CYC_RE.search(r.stdout)
    if m is None:
        raise RuntimeError('RTL 无输出: ' + r.stdout[-300:] + r.stderr[-300:])
    ddr = read_dump_v(dump)
    if not keep:
        os.remove(dump)
        os.remove(dmem)
    return None, ddr, dict(cycles=int(m.group(1)), macs=int(m.group(4)))
