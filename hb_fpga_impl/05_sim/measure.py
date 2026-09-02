# -*- coding: utf-8 -*-
"""measure.py — 441 个拍数类型的分层测量驱动（服务器 /tmp/ae_cycles，2026-08-31）

每个类型取代表段（types.json 的 rep），现场生成稀疏 ddr_init.mem
（权重区 = weights_blob 真值切片，激活区 = 按段号定种子的随机 int8），
跑 Verilator tb_ae_v（+SEQ/+DDRIMG/+DUMP），记录 cycles/gemm/dma/mac_total。
两遍：MODE=0（REF）与 MODE=1+PF=1（PRIM+权重预取，部署口径）。

附加实验：
  --aux   数据无关性（5 类型：激活全零 vs 随机，cycles 必须全等）
          + 确定性（3 段同配置重跑两遍）
用法：
  python measure.py sweep  --bin <Vtb_ae_v> [--jobs 16] [--types types.json]
  python measure.py aux    --bin <Vtb_ae_v>
  python measure.py one    --bin <Vtb_ae_v> --seg 730 --mode 1 --pf 1 [--zeros] [--dump out.mem]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, 'build_full')
TYPES = os.path.join(HERE, 'types.json')
RES = os.path.join(HERE, 'results')
WROOT = os.path.join(HERE, 'w')
P = dict(COLS=108, CTX_WORDS=131072, W_WORDS=4096, SEQ_N=2048,
         DDR_BYTES=8388608, ZERO_SLOT=4096)

import numpy as np

CYC_RE = re.compile(r'cycles=(\d+) gemm=(\d+) dma=(\d+) mac_total=(\d+)')


def seg_dir(seg):
    return os.path.join(BUILD, 'segments', 'seg_%04d' % seg)


def blob():
    return np.fromfile(os.path.join(BUILD, 'weights_blob.bin'), dtype=np.uint8)


def gen_ddr(wd, seg, zeros=False):
    """生成稀疏 ddr_init.mem（@ 分区）+ 返回 (路径, 濂活区描述)。
    权重区写真值，激活区写定种子随机 int8（zeros=True 时全 0）。"""
    man = json.load(open(os.path.join(seg_dir(seg), 'manifest.json')))
    bl = blob()
    rng = np.random.default_rng(0xA5A5_0000 + seg)
    path = os.path.join(wd, 'ddr_init%s.mem' % ('_zero' if zeros else ''))
    regions = []          # (start, np.uint8 array)
    for w in man['weights']:
        data = bl[w['blob_off']:w['blob_off'] + w['blob_len']]
        regions.append((w['ddr'], data))
    for e in man['inputs']:
        n = e['words'] * 16
        data = (np.zeros(n, np.uint8) if zeros else
                rng.integers(0, 256, n, dtype=np.uint8))
        regions.append((e['ddr'], data))
    regions.sort(key=lambda r: r[0])
    for i in range(1, len(regions)):
        a0, d0 = regions[i - 1]
        a1, _ = regions[i]
        assert a0 + len(d0) <= a1, 'seg %d 区间重叠 @%X' % (seg, a1)
    with open(path, 'w') as f:
        for a, d in regions:
            f.write('@%X\n' % a)
            d.tofile(f, sep='\n', format='%02X')
            f.write('\n')
    return path


def prep(wd, seg):
    """工作目录：seq.mem 占位（+SEQ 装载会覆盖，同时喂 RTL initial）+ LUT 链接。"""
    os.makedirs(wd, exist_ok=True)
    if not os.path.exists(os.path.join(wd, 'exp2_lut.mem')):
        os.symlink(os.path.join(HERE, 'exp2_lut.mem'),
                   os.path.join(wd, 'exp2_lut.mem'))
    if not os.path.exists(os.path.join(wd, 'seq.mem')):
        os.symlink(os.path.join(seg_dir(seg), 'seq.mem'),
                   os.path.join(wd, 'seq.mem'))


def ensure_ddr(wd, seg, zeros=False):
    name = 'ddr_init%s.mem' % ('_zero' if zeros else '')
    path = os.path.join(wd, name)
    if not os.path.exists(path):
        gen_ddr(wd, seg, zeros=zeros)
    return path


def run_sim(binpath, wd, seg, mode, pf, zeros=False, dump=None, timeout=7200):
    binpath = os.path.abspath(binpath)
    img = ensure_ddr(wd, seg, zeros=zeros)
    cmd = [binpath, '+MODE=%d' % mode, '+PF=%d' % pf,
           '+SEQ=%s' % os.path.join(seg_dir(seg), 'seq.mem'),
           '+DDRIMG=%s' % img,
           '+DUMP=%s' % (dump if dump else '/dev/null')]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=wd, capture_output=True, text=True,
                       timeout=timeout)
    dt = time.time() - t0
    m = CYC_RE.search(r.stdout)
    if not m:
        return dict(seg=seg, mode=mode, pf=pf, zeros=zeros, rc=r.returncode,
                    error=r.stdout[-2000:] + r.stderr[-2000:], wall_s=dt)
    return dict(seg=seg, mode=mode, pf=pf, zeros=zeros, rc=r.returncode,
                cycles=int(m.group(1)), gemm=int(m.group(2)),
                dma=int(m.group(3)), mac_total=int(m.group(4)), wall_s=dt)


def task_type(args):
    binpath, tid, seg, modes = args
    wd = os.path.join(WROOT, 't%03d' % tid)
    prep(wd, seg)
    out = []
    for mode, pf in modes:
        out.append(run_sim(binpath, wd, seg, mode, pf))
    return dict(type_id=tid, seg=seg, runs=out)


def sweep(binpath, jobs, types_path, modes):
    types = json.load(open(types_path))
    os.makedirs(RES, exist_ok=True)
    tl = [(binpath, t['type_id'], t['rep'], modes) for t in types['types']]
    done = 0
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for r in ex.map(task_type, tl):
            results.append(r)
            done += 1
            if done % 25 == 0 or done == len(tl):
                print('[sweep] %d/%d 类型完成，累计 %.1f min' %
                      (done, len(tl), (time.time() - t0) / 60), flush=True)
    json.dump(results, open(os.path.join(RES, 'sweep.json'), 'w'), indent=1)
    print('[sweep] 写 results/sweep.json')
    # 快速健康检查
    bad = [r for r in results if any('error' in u for u in r['runs'])]
    print('[sweep] 失败类型数：', len(bad))
    for r in bad[:5]:
        print('  type', r['type_id'], 'seg', r['seg'],
              [u.get('error', '')[-200:] for u in r['runs'] if 'error' in u])


# 数据无关性 5 类型：按实例数排序取 spread（大中小都有）
def aux(binpath, types_path):
    types = json.load(open(types_path))
    tl = types['types']
    n = len(tl)
    pick = [tl[0], tl[n // 4], tl[n // 2], tl[3 * n // 4], tl[-1]]
    res_ind = []
    for t in pick:
        wd = os.path.join(WROOT, 'aux_z%03d' % t['type_id'])
        prep(wd, t['rep'])
        for mode, pf in ((0, 0), (1, 1)):
            r0 = run_sim(binpath, wd, t['rep'], mode, pf, zeros=False)
            r1 = run_sim(binpath, wd, t['rep'], mode, pf, zeros=True)
            res_ind.append(dict(type_id=t['type_id'], seg=t['rep'],
                                mode=mode, random=r0.get('cycles'),
                                zeros=r1.get('cycles'),
                                equal=(r0.get('cycles') == r1.get('cycles'))))
            print('[aux-ind] type %d seg %d mode %d: random=%s zeros=%s equal=%s'
                  % (t['type_id'], t['rep'], mode, r0.get('cycles'),
                     r1.get('cycles'), r0.get('cycles') == r1.get('cycles')),
                  flush=True)
    # 确定性：3 段 × 两遍（同配置）
    det_pick = [tl[1], tl[n // 3], tl[-2]]
    res_det = []
    for t in det_pick:
        wd = os.path.join(WROOT, 'aux_d%03d' % t['type_id'])
        prep(wd, t['rep'])
        for mode, pf in ((0, 0), (1, 1)):
            rs = [run_sim(binpath, wd, t['rep'], mode, pf) for _ in range(2)]
            res_det.append(dict(type_id=t['type_id'], seg=t['rep'], mode=mode,
                                runs=[r.get('cycles') for r in rs],
                                equal=all(r.get('cycles') == rs[0].get('cycles')
                                          for r in rs)))
            print('[aux-det] type %d seg %d mode %d: %s' %
                  (t['type_id'], t['rep'], mode, res_det[-1]['runs']),
                  flush=True)
    json.dump(dict(data_independence=res_ind, determinism=res_det),
              open(os.path.join(RES, 'aux.json'), 'w'), indent=1)


def one(binpath, seg, mode, pf, zeros, dump):
    wd = os.path.join(WROOT, 'one_%04d' % seg)
    prep(wd, seg)
    r = run_sim(binpath, wd, seg, mode, pf, zeros=zeros, dump=dump)
    print(json.dumps(r, indent=1))
    return r


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['sweep', 'aux', 'one'])
    ap.add_argument('--bin', required=True)
    ap.add_argument('--jobs', type=int, default=16)
    ap.add_argument('--types', default=TYPES)
    ap.add_argument('--seg', type=int, default=730)
    ap.add_argument('--mode', type=int, default=1)
    ap.add_argument('--pf', type=int, default=1)
    ap.add_argument('--zeros', action='store_true')
    ap.add_argument('--dump', default=None)
    a = ap.parse_args()
    if a.cmd == 'sweep':
        sweep(a.bin, a.jobs, a.types, [(0, 0), (1, 1)])
    elif a.cmd == 'aux':
        aux(a.bin, a.types)
    else:
        one(a.bin, a.seg, a.mode, a.pf, a.zeros, a.dump)
