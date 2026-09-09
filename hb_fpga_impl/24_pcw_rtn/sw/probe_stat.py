# -*- coding: utf-8 -*-
"""probe_stat.py — seg_0000 段输出 vs int64 精确模拟的统计对拍（2026-08-31）。"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODNAME = 'backbone.patch_embed.projection'


def main():
    HB = '~/workspace/holobrain'
    if HB not in sys.path:
        sys.path.insert(0, HB)
    import bringup                                   # noqa: F401
    from robo_orchard_lab.models.holobrain.processor import HoloBrainProcessor
    from robo_orchard_lab.models.mixin import ModelMixin
    import compiler
    import host_driver as HD
    from fast_interp import run_segment_fast
    from golden_interp import load_seq

    torch.manual_seed(20260830)
    processor = HoloBrainProcessor.load(bringup.CKPT, bringup.PROCESSOR_JSON)
    model = ModelMixin.load_model(bringup.MODEL_DIR, load_impl='native')
    model = model.float().eval()
    mods = dict(model.named_modules())
    mod = mods[MODNAME]
    batch = torch.load('/tmp/ae_hostdrv/batch_s000.pt', map_location='cpu',
                       weights_only=False)
    HD._fix_kinematics_device(batch)
    drv = HD.HostDriver('/tmp/ae_hostdrv/build_s000_v3',
                        '/tmp/ae_hostdrv/trace_s000.json',
                        '/tmp/ae_hostdrv/hw_calib_table_v2.json', model)
    drv.P = compiler.PROFILES['full']
    e = json.load(open('/tmp/ae_hostdrv/hw_calib_table_v2.json',
                       encoding='utf-8'))['gemms'][MODNAME]

    cap = {}
    h = mod.register_forward_hook(
        lambda m, i, o: cap.update(x=i[0].detach().clone()))
    with torch.no_grad():
        model(batch)
    h.remove()
    x = cap['x']
    x2d = drv.im2col(mod, x)
    q48 = HD.q_round(x2d, e['sa'])
    q49 = torch.cat([q48, torch.ones(q48.shape[0], 1, dtype=torch.int8)], 1)
    W = mod.weight.detach().reshape(96, -1)
    Wq = HD.q_round(W, e['sw'])
    Waug = torch.cat([Wq, torch.zeros(Wq.shape[0], 1, dtype=torch.int8)], 1)

    # int64 精确
    acc = q49.long() @ Waug.long().t()                # torch int64 mm
    m_rq, s = e['m_requant'], e['s_shift']
    y64 = (acc * m_rq) >> s                           # int64 算术右移
    y_sim = torch.clamp(y64, -128, 127).to(torch.int8).numpy()

    seg_dir = '/tmp/ae_hostdrv/build_s000_v3/segments/seg_0000'
    man = json.load(open(os.path.join(seg_dir, 'manifest.json'),
                         encoding='utf-8'))
    P = drv.P
    blob = np.fromfile('/tmp/ae_hostdrv/build_s000_v3/weights_blob.bin',
                       dtype=np.uint8)
    img = np.zeros(P['DDR_BYTES'], dtype=np.uint8)
    for w in man['weights']:
        img[w['ddr']:w['ddr'] + w['blob_len']] = \
            blob[w['blob_off']:w['blob_off'] + w['blob_len']]
    en = man['inputs'][0]
    img[en['ddr']:en['ddr'] + en['words'] * 16] = \
        np.frombuffer(HD.pack_kact(q49), dtype=np.uint8)
    _, ddr, _ = run_segment_fast(load_seq(seg_dir), img, P)
    o = man['outputs'][0]
    a = np.frombuffer(ddr[o['ddr']:o['ddr'] + o['words'] * 16].tobytes(),
                      dtype=np.int8).reshape(-1, 16)
    g = a.reshape(-1, o['pitch'], 16).transpose(0, 2, 1).reshape(-1,
                                                                 o['pitch'])
    g = g[:o['m']]
    diff = (g.astype(int) - y_sim.astype(int))
    nz = diff != 0
    print('不一致元素: %d / %d (%.2f%%)'
          % (int(nz.sum()), diff.size, 100 * float(nz.mean())))
    if nz.any():
        ad = np.abs(diff[nz])
        print('|diff| 分布: p50=%.0f p95=%.0f max=%.0f'
              % (float(np.median(ad)), float(np.quantile(ad, .95)),
                 float(ad.max())))
        rows = np.flatnonzero(nz.any(axis=1))
        blk = rows // 16
        ub = np.unique(blk)
        print('涉及 %d 个 16 行块（共 %d）: %s'
              % (len(ub), diff.shape[0] // 16, ub[:10]))
        cols = np.flatnonzero(nz.any(axis=0))
        print('涉及列数 %d / %d，列 %s' % (len(cols), diff.shape[1],
                                         cols[:10]))
        r0 = rows[0]
        c0 = int(np.flatnonzero(nz[r0])[0])
        print('首例 行%d 列%d: 段=%d 模拟=%d acc=%d y64=%d'
              % (r0, c0, g[r0, c0], y_sim[r0, c0], int(acc[r0, c0]),
                 int(y64[r0, c0])))
        # 段输出是否 = acc 截断的其他公式
        acc0 = int(acc[r0, c0])
        print('  候选: floor(acc*m/2^s)=%.2f  (acc*m)>>s=%d  floor(acc*r*)=%d'
              % (acc0 * m_rq / 2 ** s, (acc0 * m_rq) >> s,
                 int(np.floor(acc0 * e['r_star']))))


if __name__ == '__main__':
    main()
