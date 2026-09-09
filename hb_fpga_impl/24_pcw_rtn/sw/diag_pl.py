# -*- coding: utf-8 -*-
"""diag_pl.py — PL 段链 vs fp 原生逐模块对比，定位端到端超差根因（2026-08-31）。

方法（v2，两遍式）：
  遍 1（干净模型，未 patch）：forward hook 抓每个 PL 目标模块的 fp 输出。
  遍 2（drv.patch() 后正常 PL 链）：同批 hook 抓 PL 输出。
  两遍前都 torch.manual_seed(seed)，随机轨迹对齐；遍 2 输入由 PL 链演变决定，
  对比反映误差在链上的真实传播。按全局调用序记录
  rel_l2 = ||pl-fp||/||fp||，找拓扑最早的大偏差模块。
（v1 的同调用双跑方案会经 patched 子模块二次推进 cursor，废弃。）
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def relerr(pl, fp):
    if isinstance(pl, (tuple, list)) or isinstance(fp, (tuple, list)):
        pls = pl if isinstance(pl, (tuple, list)) else (pl,)
        fps = fp if isinstance(fp, (tuple, list)) else (fp,)
        rs = [relerr(p, f) for p, f in zip(pls, fps)]
        rs = [r for r in rs if r is not None]
        return max(rs) if rs else None
    if not torch.is_tensor(pl) or not torch.is_tensor(fp):
        return None
    if pl.shape != fp.shape:
        return None
    d = pl.float() - fp.float()
    n = float(fp.float().norm())
    return float(d.norm() / (n + 1e-12))


def grab(model, targets, seed, batch):
    store = defaultdict(list)
    ctr = [0]

    def mk(name):
        def h(mod, inp, out):
            store[name].append((ctr[0], out))
            ctr[0] += 1
        return h
    hs = [dict(model.named_modules())[n].register_forward_hook(mk(n))
          for n in targets]
    torch.manual_seed(seed)
    with torch.no_grad():
        outs = model(batch)
    for h in hs:
        h.remove()
    return store, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', required=True)
    ap.add_argument('--trace', required=True)
    ap.add_argument('--batch', required=True)
    ap.add_argument('--ref', default=None)
    ap.add_argument('--calib', default=None)
    ap.add_argument('--out', default='/tmp/ae_hostdrv/diag_000.json')
    ap.add_argument('--only', default=None,
                    help="None=全 PL；gemm=只 PL 化 GEMM；attn=只 PL 化注意力")
    args = ap.parse_args()

    HB = '~/workspace/holobrain'
    if HB not in sys.path:
        sys.path.insert(0, HB)
    import bringup                                   # noqa: F401
    from robo_orchard_lab.models.holobrain.processor import HoloBrainProcessor
    from robo_orchard_lab.models.mixin import ModelMixin
    import compiler
    import host_driver as HD

    P = compiler.PROFILES['full']
    seed = 20260830
    if args.ref and os.path.exists(args.ref):
        try:
            seed = int(np.load(args.ref)['seed'])
        except Exception:
            pass
    torch.manual_seed(seed)
    processor = HoloBrainProcessor.load(bringup.CKPT, bringup.PROCESSOR_JSON)
    model = ModelMixin.load_model(bringup.MODEL_DIR, load_impl='native')
    model = model.float().eval()
    batch = torch.load(args.batch, map_location='cpu', weights_only=False)
    HD._fix_kinematics_device(batch)
    calib = args.calib or '/tmp/ae_hostdrv/hw_calib_table_v2.json'
    drv = HD.HostDriver(args.build, args.trace, calib, model)
    drv.P = P

    mods = dict(model.named_modules())
    # patch 前先抓 fp 参考（被 gemm/attn patch 的模块才有对比意义）
    kinds_by_mod = defaultdict(set)
    for name, recs in drv.recs_by_mod.items():
        ks = {drv.node_by.get((r['module'], r['seq']), {}).get('kind')
              for r in recs}
        ks.discard(None)
        if ks & {'gemm', 'attn'} and mods.get(name) is not None:
            kinds_by_mod[name] = ks
    targets = sorted(kinds_by_mod)
    print(f'[diag] 目标模块 {len(targets)}', flush=True)

    t0 = time.perf_counter()
    fp_store, _ = grab(model, targets, seed, batch)
    t1 = time.perf_counter()
    print(f'[diag] fp 遍 {t1 - t0:.1f}s', flush=True)

    only = set(x.strip() for x in args.only.split(',')) if args.only else None
    drv.patch(only=only)
    if only:
        print(f'[diag] only={sorted(only)}')
    pl_store, outs = grab(model, targets, seed, batch)
    t2 = time.perf_counter()
    print(f'[diag] PL 遍 {t2 - t1:.1f}s segments={drv.stats["segments"]}',
          flush=True)

    pa = outs[0]['pred_actions'].detach().cpu().numpy()
    act = pa[0, :, :, 0]

    recs_log = []
    for name in targets:
        fps = fp_store.get(name, [])
        pls = pl_store.get(name, [])
        for (of, f), (op, p) in zip(fps, pls):
            r = relerr(p, f)
            if r is not None:
                recs_log.append(dict(i=of, name=name, rel=r,
                                     shape=list(f.shape)
                                     if torch.is_tensor(f) else None))
    recs_log.sort(key=lambda r: r['i'])
    rep = dict(n_calls=len(recs_log), fp_s=round(t1 - t0, 1),
               pl_s=round(t2 - t1, 1), targets=len(targets),
               per_call=recs_log)
    if args.ref and os.path.exists(args.ref):
        ref = np.load(args.ref)
        d = np.abs(act - ref['action'])
        rep['jpos_mae'] = float(d.mean())
        rep['per_joint_mae'] = [float(d[:, j].mean()) for j in range(14)]
        print('[diag] PL 遍复核 jpos MAE=%.4f（正式跑 0.2975）' % d.mean())
    big = [r for r in recs_log if r['rel'] > 0.05]
    rep['n_big'] = len(big)
    rep['first_big'] = big[:60]
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f'[diag] rel>0.05 的调用 {len(big)}/{len(recs_log)}')
    for r in big[:60]:
        print('  #%d %-72s rel=%.3f %s' % (r['i'], r['name'], r['rel'],
                                           r['shape']))
    print('[done]', args.out)


if __name__ == '__main__':
    main()
