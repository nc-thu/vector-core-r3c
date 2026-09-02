# -*- coding: utf-8 -*-
"""golden_run.py — 10 个类型代表段的 RTL vs 黄金解释器位精确对拍（全参数档）

RTL 侧：Verilator tb_ae_v +SEQ/+DDRIMG/+DUMP（MODE=1+PF=1，部署口径），
        ddr_init.mem 由 measure.ensure_ddr 生成（权重真值 + 按段号定种子的
        随机 int8 激活）——RTL 与黄金读的是同一份文件，激活字节天然一致。
黄金侧：golden_interp.run_segment 逐描述符解释（与 hw_zcu104/gen_vectors.run
        同一语义，39/39 位精确验收过）。
比对：declared_ranges（权重 ∪ 输入 ∪ 输出 ∪ 零槽）逐字节，外加 mac_total 对账。
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import golden_interp as GI
import measure as ME

P = ME.P


def parse_sparse_mem(path, size):
    """把 measure 生成的 @ 分区 mem 读回全 DDR 数组（未声明区 = 0，与
    Verilator 2 态语义一致）。"""
    ddr = np.zeros(size, dtype=np.uint8)
    addr = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('@'):
                addr = int(line[1:], 16)
            else:
                ddr[addr] = int(line, 16)
                addr += 1
    return ddr


def one(binpath, seg, mode=1, pf=1):
    wd = os.path.join(HERE, 'w', 'gold_%04d' % seg)
    ME.prep(wd, seg)
    img = ME.ensure_ddr(wd, seg)
    # 注意：sim_fast 二进制由旧版 tb_ae_v.sv（161 行，无 %s plusarg）编译，
    # +DUMP 不生效，dump 落在 cwd 默认名 dump_ddr_prim2.mem（mode=1,pf=1）。
    dump = os.path.join(wd, 'dump_ddr_prim2.mem' if (mode == 1 and pf)
                        else ('dump_ddr_prim.mem' if mode else
                              'dump_ddr_ref.mem'))
    if os.path.exists(dump):
        os.remove(dump)
    r = ME.run_sim(binpath, wd, seg, mode, pf, dump=dump)
    if 'error' in r:
        return dict(seg=seg, rtl=r, ok=False, msg='RTL 运行失败')
    if not os.path.exists(dump):
        return dict(seg=seg, rtl=r, ok=False, msg='dump 文件未生成')
    ddr_init = parse_sparse_mem(img, P['DDR_BYTES'])
    seq = GI.load_seq(ME.seg_dir(seg))
    ctx, gddr, info = GI.run_segment(seq, ddr_init, P)
    rd = GI.read_dump(dump)
    ranges = GI.declared_ranges(ME.seg_dir(seg), P)
    ok, msg = GI.compare_ranges(gddr, rd, ranges)
    return dict(seg=seg, rtl=r, golden_macs=info['macs'],
                mac_match=(info['macs'] == r['mac_total']), ok=ok, msg=msg)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True)
    ap.add_argument('--segs', required=True, help='逗号分隔段号')
    a = ap.parse_args()
    out = []
    for s in a.segs.split(','):
        seg = int(s)
        res = one(a.bin, seg)
        out.append(res)
        print('[gold] seg %04d ok=%s mac_match=%s cycles=%s %s' %
              (seg, res.get('ok'), res.get('mac_match'),
               res.get('rtl', {}).get('cycles'), res.get('msg', '')),
              flush=True)
    json.dump(out, open(os.path.join(HERE, 'results', 'golden.json'), 'w'),
              indent=1)
