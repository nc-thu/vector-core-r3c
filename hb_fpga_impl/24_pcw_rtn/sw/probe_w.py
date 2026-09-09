# -*- coding: utf-8 -*-
"""probe_w.py — blob 权重 vs q_round(W_fp, sw_v2) 对比（2026-08-31）。"""
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

    torch.manual_seed(20260830)
    processor = HoloBrainProcessor.load(bringup.CKPT, bringup.PROCESSOR_JSON)
    model = ModelMixin.load_model(bringup.MODEL_DIR, load_impl='native')
    model = model.float().eval()
    mods = dict(model.named_modules())
    mod = mods[MODNAME]

    blob = np.fromfile('/tmp/ae_v3/build_s000_v3/weights_blob.bin',
                       dtype=np.uint8)
    man = json.load(open('/tmp/ae_v3/build_s000_v3/segments/seg_0000/'
                         'manifest.json', encoding='utf-8'))
    we = man['weights'][0]
    raw = blob[we['blob_off']:we['blob_off'] + we['blob_len']]
    img = np.frombuffer(raw.tobytes(), dtype=np.int8)[:49 * 108].reshape(
        49, 108)
    print('W 图形状', img.shape, '（49 行 × 108 列，末行应全零）')
    print('末行（aug 零行）非零个数:', int((img[48] != 0).sum()))
    Wblob = img[:48, :96].T                              # [96, 48]

    e = json.load(open('/tmp/ae_hostdrv/hw_calib_table_v2.json',
                       encoding='utf-8'))['gemms'][MODNAME]
    W = mod.weight.detach().reshape(96, -1)
    q = torch.clamp(torch.round(W / e['sw']), -127, 127).to(torch.int8)
    qn = q.numpy()
    print('blob W vs q_round(W_fp, sw): 逐位一致 =',
          np.array_equal(Wblob, qn),
          ' 不同=%d/%d' % (int((Wblob != qn).sum()), Wblob.size))
    if not np.array_equal(Wblob, qn):
        idx = np.argwhere(Wblob != qn)[:6]
        for i, j in idx:
            print('  [%d,%d] blob=%d qr=%d fp/sw=%.3f'
                  % (i, j, Wblob[i, j], qn[i, j], float(W[i, j]) / e['sw']))
        print('  blob absmax=%d  q_round absmax=%d'
              % (np.abs(Wblob).max(), np.abs(qn).max()))

    # 用 blob W 重算完整量化链 rel
    batch = torch.load('/tmp/ae_hostdrv/batch_s000.pt', map_location='cpu',
                       weights_only=False)
    HD._fix_kinematics_device(batch)
    drv = HD.HostDriver('/tmp/ae_hostdrv/build_s000_v3',
                        '/tmp/ae_hostdrv/trace_s000.json',
                        '/tmp/ae_hostdrv/hw_calib_table_v2.json', model)
    drv.P = compiler.PROFILES['full']
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
    Waug = torch.from_numpy(
        np.concatenate([Wblob, np.zeros((96, 1), np.int8)], 1).astype(
            np.int64))
    acc = q49.float() @ Waug.t().float()
    yfloor = torch.floor(acc * e['m_requant'] / (2.0 ** e['s_shift']))
    ysim = torch.clamp(yfloor, -128, 127) * e['so'] + mod.bias.detach()
    sim = ysim.reshape(4, 64, 80, 96).permute(0, 3, 1, 2).contiguous()
    fp = mod(x)
    print('blob W 量化链 rel=%.4f（host 链实测 0.4772）'
          % float((sim - fp).norm() / fp.norm()))


if __name__ == '__main__':
    main()
