# -*- coding: utf-8 -*-
"""conv_repro.py — 隔离复现 patch_embed（Conv2d im2col + PL GEMM）误差。

1) 干净模型 hook 住 backbone.patch_embed 的输入/输出，前向一次留存。
2) 驱动模型（patch 后）用同一输入单独调 backbone.patch_embed，
   与干净输出逐元素比对，打印误差分布。
"""
import json
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
import compiler

batch_fp = sys.argv[1] if len(sys.argv) > 1 else '/tmp/ae_hostdrv/batch_s000.pt'
ref_fp = sys.argv[2] if len(sys.argv) > 2 else '/tmp/ae_hostdrv/fp32_ref_000.npz'
target = sys.argv[3] if len(sys.argv) > 3 else 'backbone.patch_embed'

ref = np.load(ref_fp)
seed = int(ref['seed']) if 'seed' in ref.files else 20260830

# ---- A：干净前向，捕获目标模块输入/输出 ----
torch.manual_seed(seed)
model_a = ModelMixin.load_model(bringup.MODEL_DIR, load_impl='native').float().eval()
cap = {}
mod_a = dict(model_a.named_modules())[target]


def pre_hook(m, inp):
    cap['in'] = [t.detach().float().cpu() if torch.is_tensor(t) else t
                 for t in inp]


def post_hook(m, inp, out):
    if torch.is_tensor(out):
        cap['out'] = out.detach().float().cpu()
    else:
        cap['out'] = [t.detach().float().cpu() for t in out
                      if torch.is_tensor(t)]


mod_a.register_forward_pre_hook(pre_hook)
mod_a.register_forward_hook(post_hook)
batch = torch.load(batch_fp, map_location='cpu', weights_only=False)
hd._fix_kinematics_device(batch)
with torch.no_grad():
    model_a(batch)
print('input :', type(cap['in']),
      [tuple(t.shape) for t in cap['in'] if torch.is_tensor(t)])
if isinstance(cap['out'], list):
    cap['out'] = cap['out'][0]
print('output:', tuple(cap['out'].shape), 'amp',
      float(cap['out'].abs().max()))
del model_a

# ---- B：驱动链，同一输入单调目标模块 ----
torch.manual_seed(seed)
model_b = ModelMixin.load_model(bringup.MODEL_DIR, load_impl='native').float().eval()
drv = hd.HostDriver('/tmp/ae_hostdrv/build_s000',
                    '/tmp/ae_hostdrv/trace_s000.json',
                    '/tmp/ae_hostdrv/hw_calib_table_v2.json', model_b)
drv.P = compiler.PROFILES['full']
drv.patch()
mod_b = dict(model_b.named_modules())[target]
x = cap['in'][0]
with torch.no_grad():
    y = mod_b(*cap['in'])
if isinstance(y, (tuple, list)):
    ts = [t for t in y if torch.is_tensor(t)]
    y = ts[0]
y = y.detach().float().cpu()

a = cap['out']
if isinstance(a, list):
    a = a[0]
d = (a - y).abs()
print(f'[repro] mean={d.mean():.4f} max={d.max():.4f} '
      f'amp={a.abs().max():.4f}')
fm = d.flatten()
i = int(torch.argmax(fm))
print('worst @', i, 'a=', float(a.flatten()[i]), 'b=', float(y.flatten()[i]))

# ---- C：手算 int8 语义（区分量化误差 vs 搬运/布局错误） ----
import torch.nn.functional as Fn
node = drv.gnodes[target][0]
cal = drv.calib.get(target, {})
sa, so, rq = node['sa'], node['so'], node['rq']
sw = cal.get('sw', 1.0)
augc = cal.get('bias_aug_c')
W = mod_b.weight.detach().float().reshape(mod_b.out_channels, -1)  # [n, k]
Wq = torch.clamp(torch.round(W / sw), -127, 127)
x2d = drv.im2col(mod_b, x)                                        # [m, k]
Aq = torch.clamp(torch.round(x2d / sa), -127, 127)
if node.get('aug'):
    c = float(drv.augc.get(target, 1.0))
    Aq = torch.cat([Aq, torch.full((Aq.shape[0], 1), c)], 1)
    Wq = torch.cat([Wq, torch.tensor(cal['w_bias_int8'],
                                     dtype=torch.float32).unsqueeze(0)], 0)
acc = Aq @ Wq.T
yq8 = torch.clamp(torch.floor(acc * rq[0] / (1 << rq[1])), -128, 127)
yq = yq8 * so
if node.get('host_bias'):
    yq = yq + mod_b.bias.detach().float().unsqueeze(0)
# 行序 (n, h, w) → (N, C, H, W)
osh = a.shape
yq4 = yq.reshape(osh[0], osh[2], osh[3], osh[1]).permute(0, 3, 1, 2)
print(f'[quant] node aug={node.get("aug")} host_bias={node.get("host_bias")} '
      f'sa={sa:.5f} sw={sw:.5f} so={so:.5f} rq={rq} m={node["m"]} '
      f'k={node["k"]}')
print(f'[quant] x2d {tuple(x2d.shape)} amp={float(x2d.abs().max()):.2f} '
      f'sa*127={sa * 127:.2f}  饱和比例='
      f'{float((torch.round(x2d / sa).abs() > 127).float().mean()):.4f}')
dq = (a - yq4).abs()
print(f'[manual-vs-clean] mean={dq.mean():.4f} max={dq.max():.4f}')
Y = drv.must_assemble(node['out_graph'])
if Y is not None:
    Y4 = Y.reshape(osh[0], osh[2], osh[3], osh[1]).permute(0, 3, 1, 2)
    dd = (Y4 - yq4).abs()
    print(f'[manual-vs-driver] mean={dd.mean():.4f} max={dd.max():.4f}')
    dd2 = (a - Y4).abs()
    print(f'[driver-vs-clean] mean={dd2.mean():.4f} max={dd2.max():.4f}')

# ---- D：逐字节对拍 A 图（驱动喂的 vs 手算的） ----
segs = drv.wseg.get(node['out_graph'], [])
print('[A图] 段数', len(segs))
e0 = None
for s in segs:
    for e in drv.mans[s]['inputs']:
        if e.get('kind') == 'act_in':
            e0 = e
            break
    if e0:
        break
if e0 is not None:
    img = drv.act_image(e0)
    rl = e0.get('row_lo', 0)
    rh = e0.get('row_hi', e0['m'])
    k = e0['k']
    Am = torch.cat([Aq[rl:rh, :k]], 0)
    # aug 列对齐：act_image 在 k==cols+1 时补常数列
    if e0['k'] == Am.shape[1] + 1:
        Am = torch.cat([Am, Am.new_full((Am.shape[0], 1), 1)], 1)
    ref_bytes = hd.pack_kact(Am.to(torch.int8))
    ncmp = min(len(ref_bytes), len(img))
    ba = np.frombuffer(ref_bytes[:ncmp], dtype=np.int8)
    bb = np.frombuffer(img[:ncmp], dtype=np.int8)
    nd = int((ba != bb).sum())
    print(f'[A图] rows[{rl}:{rh}) k={k} 字节 {ncmp} 不同 {nd} '
          f'({nd / max(ncmp, 1):.4f})')
    if nd:
        idx = np.nonzero(ba != bb)[0][:8]
        for i in idx:
            print(f'   byte#{i}: manual={ba[i]} driver={bb[i]}')

# ---- E：W 图对拍（blob 字节 vs 手算 Wq，按 LOAD W 语义展开） ----
from golden_interp import load_seq, decode
seg0 = segs[0]
seqw = load_seq(os.path.join(drv.dir, 'segments', seg0))
w_addr = w_len = None
for w_ in seqw:
    f = decode(int(w_))
    if f['op'] == 4 and f.get('b_src') == 1:
        w_addr, w_len = f['dma_addr'], f['dma_len']
if w_addr is not None:
    man0 = json.load(open(os.path.join(drv.dir, 'segments', seg0,
                                       'manifest.json'), encoding='utf-8'))
    we = next(w for w in man0['weights'] if w['ddr'] == w_addr)
    blob = np.fromfile(os.path.join(drv.dir, 'weights_blob.bin'),
                       dtype=np.uint8)
    wb = blob[we['blob_off']:we['blob_off'] + we['blob_len']].view(np.int8)
    C = drv.P['COLS']
    nwd = len(wb) // C
    wram = wb[:nwd * C].reshape(nwd, C)      # 行 = k 字，列 = COLS
    Wb = wram[:node['k'] + 1, :node['n']]    # [k_eff, n]
    Wm = Wq.T.to(torch.int8).numpy()         # [k, n] 手算
    ndw = int((Wb[:node['k']] != Wm).sum())
    print(f'[W图] bytes={len(wb)} nwd={nwd} k={node["k"]} n={node["n"]} '
          f'不同 {ndw}/{Wm.size}')
    biasrow = Wb[node['k']]
    print(f'[W图] bias 行 (k={node["k"]}): max|.|={np.abs(biasrow).max()}')
    if ndw:
        ij = np.nonzero(Wb[:node['k']] != Wm)
        for t in range(min(6, len(ij[0]))):
            kk, j = int(ij[0][t]), int(ij[1][t])
            print(f'   W[{kk},{j}]: manual={Wm[kk, j]} blob={Wb[kk, j]} '
                  f'(fp={float(W[j, kk]):.5f})')

# ---- F：直接跑解释器，取原始 int8 Y 与手算 yq8 比 ----
from golden_interp import build_ddr_image
from fast_interp import run_segment_fast
segd = os.path.join(drv.dir, 'segments', seg0)
img = np.frombuffer(drv.act_image(e0), dtype=np.uint8)
ddr_img = build_ddr_image(segd, os.path.join(drv.dir, 'weights_blob.bin'),
                          {e0['name']: img}, drv.P)
_, ddr_out, _ = run_segment_fast(seqw, ddr_img, drv.P)
o0 = man0['outputs'][0]
raw = np.frombuffer(ddr_out[o0['ddr']:o0['ddr'] + o0['words'] * 16]
                    .tobytes(), dtype=np.int8)
m_, n_ = o0['m'], o0['n']
Y8 = raw.reshape(-1, 16).reshape(-1, n_, 16).transpose(0, 2, 1) \
    .reshape(-1, n_)
ym = yq8.detach().numpy()
ndy = int((Y8[:m_] != ym).sum())
print(f'[Y8] store {Y8.shape} vs manual {ym.shape} 不同 {ndy}/{ym.size}')
if ndy:
    ij = np.nonzero(Y8[:m_] != ym)
    for t in range(min(6, len(ij[0]))):
        r, c = int(ij[0][t]), int(ij[1][t])
        print(f'   Y[{r},{c}]: manual={ym[r, c]} interp={Y8[r, c]}')
    r, c = int(ij[0][0]), int(ij[1][0])
    print(f'   acc_manual={float(acc[r, c]):.1f} '
          f'量化A行前8={Aq[r, :8].tolist()}')

# ---- G：复刻 LOAD 路由，把 interp 眼中的 A 行/B 列抠出来对 ----
ctxw = drv.P['CTX_WORDS']
ctx2 = np.zeros((16, ctxw), dtype=np.int64)
COLS = drv.P['COLS']
wram2 = np.zeros((COLS, drv.P['W_WORDS']), dtype=np.int64)
for w_ in seqw:
    f = decode(int(w_))
    if f['op'] == 15:
        break
    if f['op'] == 4:
        nB = f['dma_len']
        Bx = np.frombuffer(ddr_img[f['dma_addr']:f['dma_addr'] + nB]
                           .astype(np.int64).tobytes(), dtype=np.int64) \
            if False else ddr_img[f['dma_addr']:f['dma_addr'] + nB] \
            .astype(np.int64)
        if f['b_src'] == 0:
            base, fw = f['b_base'], nB // 16
            ctx2[:, base:base + fw] = Bx[:fw * 16].reshape(fw, 16).T
        else:
            base, nwd = f['b_base'], nB // COLS
            wram2[:, base:base + nwd] = Bx[:nwd * COLS].reshape(nwd,
                                                                COLS).T
    elif f['op'] in (0, 1, 2):
        g = f
        Arow = np.stack([ctx2[i % 16,
                              g['a_base'] + (i // 16) * g['k']:g['a_base']
                              + (i // 16) * g['k'] + g['k']]
                         for i in (r, r + 1)])
        Bcol = wram2[:, g['b_base']:g['b_base'] + g['k']].T[:, :g['b_spad']]
        acc_i = int(Arow[0].astype(np.int64) @ Bcol[:, c].astype(np.int64))
        y_i = int(np.clip((acc_i * g['rq_m']) >> g['rq_s'], -128, 127))
        print(f'[G] GEMM m={g["m"]} 行{r} 列{c}: acc={acc_i} '
              f'y={y_i} manual_acc={float(acc[r, c]):.0f} '
              f'manual_y={float(yq8[r, c]):.0f} interp_y={int(Y8[r, c])}')
        print('    interp A 行:', Arow[0].astype(int).tolist()[:12])
        print('    manual A 行:', [int(v) for v in Aq[r, :12].tolist()])
        if r < g['m']:
            break
