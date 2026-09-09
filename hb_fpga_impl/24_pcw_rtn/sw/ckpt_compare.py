# -*- coding: utf-8 -*-
"""ckpt_compare.py — 同一 batch 上干净 fp32 与驱动链的逐环节对拍（v2）。

  A. 干净 fp32（不 patch）
  B. 驱动链（host_driver：PL 段 + host 算子）

按「A 跑里首次调用的顺序」打印每个检查点的 mean/max 绝对误差、
相对幅度、最大误差位置，定位第一个发散环节。
"""
import json
import os
import re
import sys
import time

import numpy as np
import torch

HB = '~/workspace/holobrain'
for p in (os.path.join(HB, 'shims'), os.path.join(HB, 'robo_orchard_lab'), HB):
    if p not in sys.path:
        sys.path.insert(0, p)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bringup  # noqa: F401
from robo_orchard_lab.models.mixin import ModelMixin  # noqa
import host_driver as hd
import compiler

PATTERNS = [
    r'^backbone$', r'^backbone\.patch_embed$', r'^backbone\.stages\.\d+$',
    r'^backbone\.stages\.\d+\.\d+$', r'^backbone\.norm\d$',
    r'^backbone_3d$', r'^backbone_3d\.patch_embed$',
    r'^backbone_3d\.stages\.\d+$', r'^backbone_3d\.norm\d$',
    r'^neck$', r'^neck_3d$', r'^text_feat_map$',
    r'^text_encoder\.language_backbone\.body\.model\.encoder\.layer\.\d+$',
    r'^feature_enhancer$', r'^feature_enhancer\.positional_encoding$',
    r'^feature_enhancer\.(text|img|text_img)_attn_blocks\.\d+$',
    r'^spatial_enhancer$', r'^spatial_enhancer\.(pts_prob_pre_fc|pts_prob_fc'
    r'|pts_fc|fusion_fc|fusion_norm)$',
    r'^decoder\.t_embed$', r'^decoder\.input_layers\.\d+$',
    r'^decoder\.robot_encoder$',
    r'^decoder\.layers\.(0|1|2|3|4|5|10|20|30|40|50|60|63|64)$',
    r'^decoder\.head$',
]


def pick_names(model):
    pats = [re.compile(p) for p in PATTERNS]
    return [n for n, _ in model.named_modules()
            if any(p.match(n) for p in pats)]


def install(model, names):
    store = {}
    order = []

    def mk(nm):
        def hook(m, i, o):
            t = o[0] if isinstance(o, (tuple, list)) else o
            if torch.is_tensor(t):
                if nm not in store:
                    order.append(nm)
                store.setdefault(nm, []).append(t.detach().float().cpu())
        return hook
    mods = dict(model.named_modules())
    hs = [mods[nm].register_forward_hook(mk(nm)) for nm in names]
    return store, hs, order


def build_model():
    m = ModelMixin.load_model(bringup.MODEL_DIR, load_impl='native')
    return m.float().eval()


def run_clean(seed, batch_fp, names):
    torch.manual_seed(seed)
    model = build_model()
    store, hs, order = install(model, names)
    batch = torch.load(batch_fp, map_location='cpu', weights_only=False)
    hd._fix_kinematics_device(batch)
    t0 = time.perf_counter()
    with torch.no_grad():
        outs = model(batch)
    print(f'[ckpt] fp32 前向 {time.perf_counter() - t0:.1f}s')
    for h in hs:
        h.remove()
    act = outs[0]['pred_actions'].detach().cpu().numpy()[0, :, :, 0]
    del model
    return store, order, act


def run_driver(seed, batch_fp, names, build, trace, calib):
    torch.manual_seed(seed)
    model = build_model()
    store, hs, order = install(model, names)
    batch = torch.load(batch_fp, map_location='cpu', weights_only=False)
    hd._fix_kinematics_device(batch)
    drv = hd.HostDriver(build, trace, calib, model)
    drv.P = compiler.PROFILES['full']
    drv.patch()
    t0 = time.perf_counter()
    with torch.no_grad():
        outs = model(batch)
    print(f'[ckpt] 驱动前向 {time.perf_counter() - t0:.1f}s '
          f'segments={drv.stats["segments"]}/{len(drv.seg_names)}')
    for h in hs:
        h.remove()
    act = outs[0]['pred_actions'].detach().cpu().numpy()[0, :, :, 0]
    return store, act


def main():
    build = sys.argv[1] if len(sys.argv) > 1 else '/tmp/ae_hostdrv/build_s000'
    trace = sys.argv[2] if len(sys.argv) > 2 else '/tmp/ae_hostdrv/trace_s000.json'
    batch_fp = sys.argv[3] if len(sys.argv) > 3 else '/tmp/ae_hostdrv/batch_s000.pt'
    ref_fp = sys.argv[4] if len(sys.argv) > 4 else '/tmp/ae_hostdrv/fp32_ref_000.npz'
    calib = sys.argv[5] if len(sys.argv) > 5 else '/tmp/ae_hostdrv/hw_calib_table_v2.json'

    ref = np.load(ref_fp)
    seed = int(ref['seed']) if 'seed' in ref.files else 20260830
    ra = ref['action']

    store_a, order, act_a = run_clean(seed, batch_fp,
                                      pick_names(build_model()))
    print(f'[ckpt] A vs ref MAE={np.abs(act_a - ra).mean():.4f}')
    store_b, act_b = run_driver(seed, batch_fp, list(store_a.keys()),
                                build, trace, calib)
    db = np.abs(act_b - ra)
    print(f'[ckpt] B vs ref MAE={db.mean():.4f} max={db.max():.4f} '
          f'arm12={db[:, :12].mean():.4f} gripper={db[:, [6, 13]].mean():.4f}')

    print(f'\n{"checkpoint":52s} {"calls":>4s} {"mean_abs":>9s} '
          f'{"max_abs":>9s} {"amp":>8s} {"rel_max":>8s}  max_at')
    report = []
    for nm in order:
        la, lb = store_a.get(nm, []), store_b.get(nm, [])
        if not la:
            continue
        if len(la) != len(lb):
            print(f'{nm:52s} {len(la)}/{len(lb):<3d}  CALLS-MISMATCH')
            continue
        w_mean = w_max = 0.0
        amp = 0.0
        loc = ''
        for ta, tb in zip(la, lb):
            d = (ta - tb).abs()
            amp = max(amp, float(ta.abs().max()))
            w_mean = max(w_mean, float(d.mean()))
            if float(d.max()) > w_max:
                w_max = float(d.max())
                f = d.flatten()
                i = int(torch.argmax(f))
                loc = (f'{tuple(ta.shape)}#{i} '
                       f'a={float(ta.flatten()[i]):.3f} '
                       f'b={float(tb.flatten()[i]):.3f}')
        rel = w_max / max(amp, 1e-9)
        report.append(dict(name=nm, calls=len(la), mean_abs=w_mean,
                           max_abs=w_max, amp=amp, rel_max=rel, max_at=loc))
        print(f'{nm:52s} {len(la):>4d} {w_mean:9.4f} {w_max:9.4f} '
              f'{amp:8.3f} {rel:8.4f}  {loc}')
    with open(os.path.join(HERE, 'ckpt_report.json'), 'w',
              encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print('[ckpt] 写出 ckpt_report.json')


if __name__ == '__main__':
    main()
