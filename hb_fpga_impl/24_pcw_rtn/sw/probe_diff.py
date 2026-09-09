# -*- coding: utf-8 -*-
"""probe_diff.py — gemm_call 的 pl 与手工量化链 sim 的差值分解（2026-08-31）。

sim（A clamp+requant+so+bias+reshape）rel=0.1755，pl 实测 0.4772，
但段输出 int8 与各环节解析均已对上。本脚本把 d = pl − sim 分解成
通道常数 / 空间常数 / 随机残差 三部分，看差异形态。"""
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
    import json
    e = json.load(open('/tmp/ae_hostdrv/hw_calib_table_v2.json',
                       encoding='utf-8'))['gemms'][MODNAME]

    cap = {}
    h = mod.register_forward_hook(
        lambda m, i, o: cap.update(x=i[0].detach().clone()))
    with torch.no_grad():
        model(batch)
    h.remove()
    x = cap['x']

    rec = drv.recs_by_mod[MODNAME][0]
    node = drv.node_by[(rec['module'], rec['seq'])]
    pl = drv.gemm_call(mod, node, rec, x)

    # 手工量化链
    x2d = drv.im2col(mod, x)
    q48 = HD.q_round(x2d, e['sa'])
    q49 = torch.cat([q48, torch.ones(q48.shape[0], 1, dtype=torch.int8)], 1)
    W = mod.weight.detach().reshape(96, -1)
    Wq = HD.q_round(W, e['sw'])
    Waug = torch.cat([Wq, torch.zeros(Wq.shape[0], 1, dtype=torch.int8)], 1)
    acc = q49.float() @ Waug.t().float()
    yfloor = torch.floor(acc * e['m_requant'] / (2.0 ** e['s_shift']))
    ysim = torch.clamp(yfloor, -128, 127) * e['so'] + mod.bias.detach()
    sim = ysim.reshape(4, 64, 80, 96).permute(0, 3, 1, 2).contiguous()

    fp = mod(x)
    print('pl rel=%.4f  sim rel=%.4f  pl vs sim rel=%.4f'
          % (float((pl - fp).norm() / fp.norm()),
             float((sim - fp).norm() / fp.norm()),
             float((pl - sim).norm() / sim.norm())))
    d = (pl - sim)                                    # [4,96,64,80]
    dch = d.mean(dim=(0, 2, 3))                       # 通道常数
    drest = d - dch.view(1, -1, 1, 1)
    print('通道常数部分 norm=%.4f（占总差 %.1f%%）'
          % (float(dch.norm()), 100 * float(dch.norm() / d.norm())))
    print('残差 norm=%.4f' % float(drest.norm()))
    print('dch 前 12:', [round(float(v), 4) for v in dch[:12]])
    print('bias 前 12:', [round(float(v), 4) for v in mod.bias[:12]])
    so = e['so']
    print('dch/so 前 12:', [round(float(v) / so, 2) for v in dch[:12]])
    # 每通道差的直方（LSB 单位）
    dn = (d / so)
    print('d/so 分位数: p5=%.1f p50=%.1f p95=%.1f max=%.1f'
          % tuple(float(dn.quantile(q)) for q in (.05, .5, .95))
          if False else
          'd/so: p50=%.2f p95=%.2f max=%.2f min=%.2f'
          % (float(dn.median()), float(dn.quantile(.95)),
             float(dn.max()), float(dn.min())))
    # 每通道 d 的 std（若某通道整块偏移 → 常数；随机 → std 大）
    stds = d.std(dim=(0, 2, 3))
    print('通道 std 前 12:', [round(float(v), 4) for v in stds[:12]])


if __name__ == '__main__':
    main()
