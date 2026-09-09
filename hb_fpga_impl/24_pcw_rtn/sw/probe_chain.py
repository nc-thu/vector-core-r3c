# -*- coding: utf-8 -*-
"""probe_chain.py — patch_embed 一层链内逐环节对齐（2026-08-31）。

量化数学（同 A 同 W）rel=0.1755，但 host 全链 gemm_call 输出 rel=0.477。
本脚本在同一进程里对比：
  1. gemm_call 内 act_image 量化的 A 字节 vs 手工 q_round+pack 的 A
  2. gemm_call 跑出的段 STORE int8 块 vs 手工 A 跑段的 int8 块
  3. assemble 后的 Y / 最终 pl vs 手工链路
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
    a_manual = np.frombuffer(HD.pack_kact(q49), dtype=np.uint8)

    rec = drv.recs_by_mod[MODNAME][0]
    node = drv.node_by[(rec['module'], rec['seq'])]
    segs = drv.wseg[node['out_graph']]
    print('段:', segs)

    # 手工链：A=a_manual 跑段
    seg_dir = os.path.join('/tmp/ae_hostdrv/build_s000_v3/segments', segs[0])
    man = json.load(open(os.path.join(seg_dir, 'manifest.json'),
                         encoding='utf-8'))
    P = drv.P
    blob = np.fromfile('/tmp/ae_hostdrv/build_s000_v3/weights_blob.bin',
                       dtype=np.uint8)

    def run_with_a(a_bytes):
        img = np.zeros(P['DDR_BYTES'], dtype=np.uint8)
        for w in man['weights']:
            img[w['ddr']:w['ddr'] + w['blob_len']] = \
                blob[w['blob_off']:w['blob_off'] + w['blob_len']]
        for en in man['inputs']:
            if en.get('kind') == 'act_in':
                img[en['ddr']:en['ddr'] + en['words'] * 16] = a_bytes
        _, ddr, _ = run_segment_fast(load_seq(seg_dir), img, P)
        return ddr

    ddr_manual = run_with_a(a_manual)

    # host 链：gemm_call（内部 act_image 量化）
    pl = drv.gemm_call(mod, node, rec, x)

    # 抓 host 链跑段时用的 A：monkeypatch act_image
    a_host_cap = []
    orig_ai = drv.act_image

    def spy_ai(en):
        b = orig_ai(en)
        a_host_cap.append((en['name'], en.get('sa'), b))
        return b
    drv.act_image = spy_ai
    # 再跑一次（blk/act store 会多一份，无碍对比）
    pl2 = drv.gemm_call(mod, node, rec, x)
    drv.act_image = orig_ai
    for nm, sa, b in a_host_cap:
        print('act_image 输入条目 %s sa=%.6e 字节=%d' % (nm, sa, len(b)))
        if b is not None:
            bh = np.frombuffer(b, dtype=np.uint8)
            print('  与手工 A 一致:', np.array_equal(bh, a_manual),
                  ' 不同的字节: %d / %d'
                  % (int((bh != a_manual).sum()), len(a_manual)))
            if not np.array_equal(bh, a_manual):
                idx = np.flatnonzero(bh != a_manual)[:5]
                for i in idx:
                    j = int(i)
                    r, c = divmod(j // 16, 49) if False else (None, None)
                    print('   [%d] host=%02X manual=%02X' % (j, bh[j],
                                                             a_manual[j]))
    o = man['outputs'][0]
    gm = np.frombuffer(ddr_manual[o['ddr']:o['ddr'] + o['words'] * 16]
                       .tobytes(), dtype=np.int8).reshape(-1, 16)
    gm = gm.reshape(-1, o['pitch'], 16).transpose(0, 2, 1).reshape(-1,
                                                                   o['pitch'])
    # host 链的段输出（act store 里最新一份）
    occ = drv.act.get(node['out_graph']) or []
    sname, so_ent, buf = occ[-1]
    gh = np.frombuffer(buf, dtype=np.int8).reshape(-1, 16)
    gh = gh.reshape(-1, so_ent['pitch'], 16).transpose(0, 2, 1).reshape(
        -1, so_ent['pitch'])
    m = min(o['m'], gh.shape[0])
    print('host 段输出 vs 手工 A 段输出: 逐位一致 =',
          np.array_equal(gh[:m], gm[:m]))
    fp = mod(x)
    print('pl vs pl2 一致:', torch.equal(pl, pl2))
    print('pl rel=%.4f  pl2 rel=%.4f'
          % (float((pl - fp).norm() / fp.norm()),
             float((pl2 - fp).norm() / fp.norm())))


if __name__ == '__main__':
    main()
