# -*- coding: utf-8 -*-
"""probe_so.py — 第一层 GEMM 量化误差分解 + so 重校定界（2026-08-31）。

段链输出语义：out = sat8((acc*m)>>s) * so + host_bias(fp)
分解各环节对 rel的贡献：
  1. A int8（q_round(x, sa)，含 clamp）
  2. W int8
  3. requant m/s 截断（int64 acc -> int8 的乘移舍入）
  4. 输出 int8（so = out_max/127，分布失配时 int8 只用低位）
并测"so 理想化"（本样本输出 absmax/127，重算 m/s）后的 rel 下界。
"""
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

M_LIM = 32767
MODNAME = 'backbone.patch_embed.projection'


def v1_encode(r_star):
    r = max(float(r_star), 1e-30)
    s = max(0, int(math.floor(math.log2(M_LIM / r))))
    m = int(round(r * (1 << s)))
    while m > M_LIM and s > 0:
        s -= 1
        m = int(round(r * (1 << s)))
    if m < 1:
        m, s = 1, min(s, 63)
    return m, s


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
    batch = torch.load('/tmp/ae_hostdrv/batch_s000.pt', map_location='cpu',
                       weights_only=False)
    HD._fix_kinematics_device(batch)
    drv = HD.HostDriver('/tmp/ae_hostdrv/build_s000_v3',
                        '/tmp/ae_hostdrv/trace_s000.json',
                        '/tmp/ae_hostdrv/hw_calib_table_v2.json', model)
    drv.P = compiler.PROFILES['full']
    e = json.load(open('/tmp/ae_hostdrv/hw_calib_table_v2.json',
                       encoding='utf-8'))['gemms'][MODNAME]

    mods = dict(model.named_modules())
    mod = mods[MODNAME]
    rec = drv.recs_by_mod[MODNAME][0]
    node = drv.node_by[(rec['module'], rec['seq'])]
    cap = {}

    def h(m, i, o):
        cap.update(x=i[0].detach().clone(), fp=o.detach().clone())
    hh = mod.register_forward_hook(h)
    with torch.no_grad():
        model(batch)
    hh.remove()
    x, fp = cap['x'], cap['fp']

    def shape_out(y2d):
        return drv._shape_out(y2d, mod, rec['in_shapes'][0],
                              rec['out_shapes'][0])
    x2d = drv.im2col(mod, x)
    W = mod.weight.detach().reshape(96, -1)
    bias = mod.bias.detach().reshape(1, -1)

    sa, sw, so = e['sa'], e['sw'], e['so']
    q = HD.q_round(x2d, sa)
    q48 = q[:, :48]
    qaug = torch.cat([q48, torch.ones(q48.shape[0], 1, dtype=torch.int8)], 1)
    Wq = HD.q_round(W, sw)
    Waug = torch.cat([Wq, torch.zeros(Wq.shape[0], 1, dtype=torch.int8)], 1)
    acc = qaug.long() @ Waug.long().t()

    n = float(fp.norm())

    def rel(o2d):
        d = shape_out(o2d) - fp
        return float(d.norm() / n)

    # 1. A+W int8，requant 无截断、输出无 int8（fp 域重建）
    y1 = (qaug[:, :48].float() @ Wq.float().t()) * (sa * sw) + bias
    print('仅 A+W int8（无requant/无输出int8） rel=%.4f' % rel(y1))

    # 2. + requant m/s（当前表）
    m0, s0 = e['m_requant'], e['s_shift']
    y2i = torch.clamp((acc * m0) >> s0, -128, 127).to(torch.int8)
    y2 = y2i.float() * so + bias
    print('+requant+输出int8（=段链语义, so=%.5f） rel=%.4f' % (so, rel(y2)))

    # 3. requant 无截断（m/s 换 float 精确 r）+ 输出 int8（理想 so）
    out_abs = float(y1.abs().max())
    so_i = max(out_abs, 1e-12) / 127.0
    r_i = (sa * sw) / so_i
    yi = torch.clamp(torch.round(acc.double() * r_i), -127, 127)
    y3 = yi.float() * so_i + bias
    print('requant精确+理想so=%.5f             rel=%.4f' % (so_i, rel(y3)))

    # 4. 只换 so（requant 用重编码 m/s，保持硬件语义）
    m_i, s_i = v1_encode(r_i)
    y4i = torch.clamp((acc * m_i) >> s_i, -128, 127).to(torch.int8)
    y4 = y4i.float() * so_i + bias
    print('硬件m/s=%d/2^%d+理想so               rel=%.4f' % (m_i, s_i, rel(y4)))

    # 5. 只把 so 缩小一半/放大一倍看敏感性
    for k in (0.5, 2.0):
        so_k = so * k
        r_k = (sa * sw) / so_k
        m_k, s_k = v1_encode(r_k)
        yki = torch.clamp((acc * m_k) >> s_k, -128, 127).to(torch.int8)
        print('  so×%.1f rel=%.4f' % (k, rel(yki.float() * so_k + bias)))

    print('本样本 |out|max=%.3f  so*127=%.3f  理想 so=%.5f（现 %.5f, x%.2f）'
          % (out_abs, so * 127, so_i, so, so / so_i))


if __name__ == '__main__':
    main()
