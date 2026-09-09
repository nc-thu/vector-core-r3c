# -*- coding: utf-8 -*-
"""probe_first_layer.py — backbone.patch_embed.projection 单模块深挖（2026-08-31）。

diag_pl 显示全网第一个 PL 模块 rel=0.477。本脚本判定误差形态：
  1. 干净 fp 遍 hook 抓该模块输入 x / fp 输出。
  2. 手动调 drv.gemm_call 得 PL 输出。
  3. 逐通道均值差 vs 模型 bias：相关且相等 → host bias 没补上；
     随机分布 → 量化/布局问题。
  4. 纯 torch 量化仿真（sa/sw/so/aug 语义）对照，分辨「量化本底」与「链路 bug」。
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

    def h(m, inp, out):
        cap['x'] = inp[0].detach().clone()
        cap['fp'] = out.detach().clone()
    hh = mod.register_forward_hook(h)
    with torch.no_grad():
        model(batch)
    hh.remove()
    x, fp = cap['x'], cap['fp']
    print('输入', tuple(x.shape), '范围 [%.4f, %.4f]' % (x.min(), x.max()))
    print('fp 输出', tuple(fp.shape), '范围 [%.3f, %.3f]' % (fp.min(), fp.max()))

    # PL 一次
    rec = drv.recs_by_mod[MODNAME][0]
    node = drv.node_by[(rec['module'], rec['seq'])]
    pl = drv.gemm_call(mod, node, rec, x)
    d = (pl - fp)
    rel = float(d.norm() / fp.norm())
    print('PL vs fp rel=%.4f  MAE=%.4f' % (rel, float(d.abs().mean())))

    bias = mod.bias.detach()
    per_ch = d.mean(dim=(0, 2, 3))                   # [96]
    err_after_bias = (d - bias.view(1, -1, 1, 1))
    rel2 = float(err_after_bias.norm() / fp.norm())
    cc = float(np.corrcoef(per_ch.numpy(), bias.numpy())[0, 1])
    print('逐通道均值差 vs bias: corr=%.4f' % cc)
    print('差值减 bias 后 rel=%.4f（若远小于 rel → 差值就是 bias 没补）' % rel2)

    # 纯 torch 量化仿真
    tbl = json.load(open('/tmp/ae_hostdrv/hw_calib_table_v2.json',
                         encoding='utf-8'))['gemms']
    e = tbl[MODNAME]
    x2d = drv.im2col(mod, x)
    W = mod.weight.detach().reshape(mod.out_channels, -1)   # [96, 48]
    q = HD.q_round(x2d, e['sa']).float()
    Wq = HD.q_round(W, e['sw']).float()
    ysim = (q @ Wq.t()) * (e['sa'] * e['sw']) + bias
    Y = drv.assemble(node['out_graph'])
    sim_out = drv._shape_out(ysim, mod, rec['in_shapes'][0],
                             rec['out_shapes'][0])
    dsim = sim_out - fp
    print('纯 torch 量化仿真 vs fp rel=%.4f（量化本底，与 PL rel 相当=链路忠实）'
          % float(dsim.norm() / fp.norm()))
    print('PL vs 量化仿真 rel=%.4f（链路额外误差）'
          % float((pl - sim_out).norm() / sim_out.norm()))
    # 理想 sa：本样本 absmax/127（量化上限对照）
    sa2 = float(x2d.abs().max()) / 127.0
    q2 = HD.q_round(x2d, sa2).float()
    ysim2 = (q2 @ Wq.t()) * (sa2 * e['sw']) + bias
    sim2 = drv._shape_out(ysim2, mod, rec['in_shapes'][0],
                          rec['out_shapes'][0])
    print('理想 sa=%.5f 时量化 rel=%.4f（clamp=%.2f%%）'
          % (sa2, float((sim2 - fp).norm() / fp.norm()),
             100 * float((HD.q_round(x2d, sa2).abs() >= 127).float().mean())))
    print('calib: sa=%.3e sw=%.3e so=%.3e s_shift=%d m=%d fp_fb=%s'
          % (e['sa'], e['sw'], e['so'], e['s_shift'], e['m_requant'],
             e['bias_fp_fallback']))
    # clamp 情况
    qa = HD.q_round(x2d, e['sa'])
    print('A clamp%%: %.2f%%  |A|max=%.3f sa*127=%.3f'
          % (float((qa.abs() >= 127).float().mean()) * 100,
             float(x2d.abs().max()), e['sa'] * 127))
    # 段 manifest 的输出条目（host_bias 字段）
    for seg in drv.wseg.get(node['out_graph'], []):
        man = drv.mans[seg]
        for o in man['outputs']:
            if o['name'].split('@')[0] == node['out_graph']:
                print('STORE 条目:', seg, {k: o.get(k) for k in
                      ('host_bias', 'bias_key', 'so', 'm', 'n')})


if __name__ == '__main__':
    main()
