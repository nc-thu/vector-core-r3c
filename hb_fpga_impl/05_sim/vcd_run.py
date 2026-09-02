# -*- coding: utf-8 -*-
"""vcd_run.py — Top-N 类型 VCD 采集 + SAIF 转换（服务器 /tmp/ae_cycles）

从 sweep.json 取 cycles_pf1×instances 贡献最大的 N 个类型，用 trace 版
（V_HALF=2 → 4ns/周期，250MHz 功耗口径）跑完整段并 dump VCD，随后
vcd2saif.py 转 SAIF（--re-root tb_ae_v.dut --wrap tb_ae_v.dut.u_core，
窗口从 start 脉冲前开，t-start 0）。

用法：python vcd_run.py --bin sim_trace/obj_dir/Vtb_ae_v --top 3
"""
import argparse
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
import measure as ME

CYC_RE = ME.CYC_RE


def cap_one(binpath, out_dir, contrib, tid, seg):
    wd = os.path.join(out_dir, 't%03d' % tid)
    ME.prep(wd, seg)
    img = ME.ensure_ddr(wd, seg)
    vcd = os.path.join(wd, 'wave_t%03d.vcd' % tid)
    t0 = time.time()
    r = subprocess.run(
        [os.path.abspath(binpath), '+MODE=1', '+PF=1', '+VCD',
         '+VCDFILE=%s' % vcd,
         '+SEQ=%s' % os.path.join(ME.seg_dir(seg), 'seq.mem'),
         '+DDRIMG=%s' % img, '+DUMP=/dev/null'],
        cwd=wd, capture_output=True, text=True, timeout=7200)
    m = CYC_RE.search(r.stdout)
    dt = time.time() - t0
    sz = os.path.getsize(vcd) / 1e9 if os.path.exists(vcd) else -1
    saif = os.path.join(wd, 't%03d.saif' % tid)
    # 窗口起点 = VCD 首个时间戳（dumpvars 在 start 脉冲旁开窗；t-start 0 会把
    # VCD 之前的 SEQ 装载阶段补成 X，稀释翻转率）
    t_start = 0
    with open(vcd) as f:
        for line in f:
            if line.startswith('#'):
                t_start = int(line[1:])
                break
    t1 = time.time()
    cr = subprocess.run(
        ['~/.conda/envs/vsim/bin/python',
         os.path.join(HERE, 'vcd2saif.py'), vcd, saif,
         '--re-root', 'tb_ae_v.dut', '--wrap', 'tb_ae_v.dut.u_core',
         '--t-start', str(t_start)], capture_output=True, text=True)
    note = dict(type_id=tid, seg=seg, cycles=m.group(1) if m else None,
                cycles_int=int(m.group(1)) if m else -1,
                contrib=contrib, vcd_gb=round(sz, 2),
                vcd_wall_s=round(dt, 1), saif=saif,
                t_start_ps=t_start, saif_rc=cr.returncode,
                saif_bytes=os.path.getsize(saif) if os.path.exists(saif) else -1,
                saif_wall_s=round(time.time() - t1, 1))
    print('[vcd] type %d seg %04d cycles=%s vcd=%.2fGB rc=%d saif=%dB' %
          (tid, seg, m.group(1) if m else '?', sz, cr.returncode,
           note['saif_bytes']), flush=True)
    if cr.returncode:
        print(cr.stdout[-500:], cr.stderr[-500:])
    if os.path.exists(vcd):
        os.remove(vcd)          # SAIF 落地后删 VCD，省盘
    return note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True)
    ap.add_argument('--top', type=int, default=3)
    ap.add_argument('--par', type=int, default=3)
    ap.add_argument('--out', default=os.path.join(HERE, 'vcd'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    types = {t['type_id']: t for t in
             json.load(open(os.path.join(HERE, 'types.json')))['types']}
    sweep = json.load(open(os.path.join(HERE, 'results', 'sweep.json')))
    cand = []
    for r in sweep:
        pf1 = next((u['cycles'] for u in r['runs']
                    if u['mode'] == 1 and 'cycles' in u), None)
        if pf1 is None:
            continue
        cand.append((pf1 * len(types[r['type_id']]['instances']),
                     r['type_id'], r['seg'], pf1))
    cand.sort(reverse=True)
    print('[vcd] 候选贡献 top5：')
    for c in cand[:5]:
        print('  type %d seg %04d cycles=%d contrib=%d' % (c[1], c[2], c[3], c[0]))

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=a.par) as ex:
        notes = list(ex.map(lambda c: cap_one(a.bin, a.out, c[0], c[1], c[2]),
                            cand[:a.top]))
    notes.sort(key=lambda n: -n['contrib'])
    json.dump(notes, open(os.path.join(a.out, 'vcd_notes.json'), 'w'), indent=1)
    print('[vcd] 写 vcd/vcd_notes.json')


if __name__ == '__main__':
    main()
