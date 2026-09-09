# -*- coding: utf-8 -*-
"""probe_seg0.py — seg_0000 段级对拍 + 饱和统计（2026-08-31）。

假设：样本 00x 图像大量接近纯白（|归一化|>2.2），sa 按 absmax=1.856 校准 →
A 侧 80% clamp 后，纯白 patch 的 q 全为 ±127，acc 超过校准预估 →
requant 输出大面积饱和在 ±127×so，第一层 rel 0.48。
验证：数饱和比例 + 段输出与 torch 模拟（同 A 同 W 同 requant）逐位对拍。
"""
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

    seed = 20260830
    torch.manual_seed(seed)
    processor = HoloBrainProcessor.load(bringup.CKPT, bringup.PROCESSOR_JSON)
    model = ModelMixin.load_model(bringup.MODEL_DIR, load_impl='native')
    model = model.float().eval()
    batch = torch.load('/tmp/ae_hostdrv/batch_s000.pt', map_location='cpu',
                       weights_only=False)
    HD._fix_kinematics_device(batch)
    drv = HD.HostDriver('/tmp/ae_hostdrv/build_s000_v3',
                        '/tmp/ae_hostdrv/trace_s000.json',
                        '/tmp/ae_hostdrv/hw_calib_table_v2.json', model)
    drv.P = compiler.PROFILES['full']

    mods = dict(model.named_modules())
    mod = mods[MODNAME]
    cap = {}
    h = mod.register_forward_hook(
        lambda m, i, o: cap.update(x=i[0].detach().clone()))
    with torch.no_grad():
        model(batch)
    h.remove()
    x = cap['x']

    tbl = json.load(open('/tmp/ae_hostdrv/hw_calib_table_v2.json',
                         encoding='utf-8'))['gemms']
    e = tbl[MODNAME]
    x2d = drv.im2col(mod, x)
    q48 = HD.q_round(x2d, e['sa'])
    q49 = torch.cat([q48, torch.ones(q48.shape[0], 1, dtype=torch.int8)], 1)
    W = mod.weight.detach().reshape(mod.out_channels, -1)
    Wq = HD.q_round(W, e['sw'])
    Waug = torch.cat([Wq, torch.zeros(Wq.shape[0], 1, dtype=torch.int8)], 1)
    acc = q49.float() @ Waug.t().float()
    ypre = (acc * e['m_requant'] / (2.0 ** e['s_shift']))
    yfloor = torch.floor(ypre)
    sat = (yfloor.abs() > 127)
    print('输出 requant 饱和比例: %.2f%%  （acc_absmax=%.0f，校准预估 %.0f）'
          % (100 * float(sat.float().mean()), float(acc.abs().max()),
             e['acc_absmax_est']))
    y_sim = torch.clamp(yfloor, -128, 127).to(torch.int8)

    # 段级对拍：A 用同一 q49，权重走 blob，跑 fast_interp
    seg_dir = '/tmp/ae_hostdrv/build_s000_v3/segments/seg_0000'
    man = json.load(open(os.path.join(seg_dir, 'manifest.json'),
                         encoding='utf-8'))
    P = drv.P
    img = np.zeros(P['DDR_BYTES'], dtype=np.uint8)
    blob = np.fromfile('/tmp/ae_hostdrv/build_s000_v3/weights_blob.bin',
                       dtype=np.uint8)
    for w in man['weights']:
        img[w['ddr']:w['ddr'] + w['blob_len']] = \
            blob[w['blob_off']:w['blob_off'] + w['blob_len']]
    act = {}
    for en in man['inputs']:
        act[en['name']] = np.frombuffer(HD.pack_kact(q49), dtype=np.uint8)
    img[man['inputs'][0]['ddr']:
        man['inputs'][0]['ddr'] + man['inputs'][0]['words'] * 16] = \
        act[man['inputs'][0]['name']]
    _, ddr, _ = run_segment_fast(load_seq(seg_dir), img, P)
    o = man['outputs'][0]
    buf = ddr[o['ddr']:o['ddr'] + o['words'] * 16]
    a = np.frombuffer(buf.tobytes(), dtype=np.int8).reshape(-1, 16)
    g = a.reshape(-1, o['pitch'], 16).transpose(0, 2, 1).reshape(-1, o['pitch'])
    g = g[:o['m']]
    eq = np.array_equal(g, y_sim.numpy())
    print('段输出 vs torch 模拟（同 A/W/requant）: 逐位一致 =', eq)
    if not eq:
        bad = np.argwhere(g != y_sim.numpy())
        i, j = bad[0]
        print(' 首个不一致 [%d,%d]: 段=%d 模拟=%d acc=%.0f ypre=%.1f'
              % (i, j, g[i, j], y_sim.numpy()[i, j], acc[i, j], ypre[i, j]))

    # 如果不饱和（理想 requant 上限），rel 是多少
    bias = mod.bias.detach()
    Y_ns = yfloor * e['so'] + bias           # 无饱和 fp
    B, Cout, Ho, Wo = 4, 96, 64, 80
    pl_ns = Y_ns.reshape(B, Ho, Wo, Cout).permute(0, 3, 1, 2).contiguous()
    fp = mod(x)
    d = pl_ns - fp
    Ys = torch.clamp(yfloor, -128, 127) * e['so'] + bias
    pl_s = Ys.reshape(B, Ho, Wo, Cout).permute(0, 3, 1, 2).contiguous()
    print('只去饱和 rel=%.4f（有饱和=%.4f）'
          % (float(d.norm() / fp.norm()),
             float((pl_s - fp).norm() / fp.norm())))


if __name__ == '__main__':
    main()
