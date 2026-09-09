# -*- coding: utf-8 -*-
"""clean_check.py — 干净 fp32 前向 vs fp32_ref：判定 ref 可复现性。"""
import os
import sys

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

batch_fp = sys.argv[1] if len(sys.argv) > 1 else '/tmp/ae_hostdrv/batch_s000.pt'
ref_fp = sys.argv[2] if len(sys.argv) > 2 else '/tmp/ae_hostdrv/fp32_ref_000.npz'

ref = np.load(ref_fp)
seed = int(ref['seed']) if 'seed' in ref.files else 20260830
print('ref seed:', seed)

torch.manual_seed(seed)
m = ModelMixin.load_model(bringup.MODEL_DIR, load_impl='native').float().eval()
batch = torch.load(batch_fp, map_location='cpu', weights_only=False)
hd._fix_kinematics_device(batch)
with torch.no_grad():
    outs = m(batch)
pa = outs[0]['pred_actions'].detach().cpu().numpy()
act = pa[0, :, :, 0]
ra = ref['action']
d = np.abs(act - ra)
print(f'[clean-vs-ref] jpos MAE={d.mean():.4f} max={d.max():.4f} '
      f'arm12={d[:, :12].mean():.4f} gripper={d[:, [6, 13]].mean():.4f}')
print('ref action range:', ra.min(), ra.max())
# 全部 8 个输出维都比一遍（driver 只取了第 0 维）
for k in range(pa.shape[-1]):
    dk = np.abs(pa[0, :, :, k] - (ref['pred_actions_raw'][..., k]
                                  if 'pred_actions_raw' in ref.files
                                  else ra))
    print(f'  dim{k}: mae={dk.mean():.4f}')
