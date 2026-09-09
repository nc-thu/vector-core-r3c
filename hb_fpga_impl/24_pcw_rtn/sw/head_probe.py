# -*- coding: utf-8 -*-
"""head_probe.py — 输出头 5 层量化成分解剖（S8/S9 佐证实验）。

在不动 /tmp/pcw_rtn 的前提下，子类化 HostDriver，只对 5 个输出头模块
（decoder.head.convs.0/1、output_layers.1/3/4）替换 gemm_call：

  stats 模式：部署基线照跑（segment 真执行），宿主侧用表值复算同一条
              量化链做逐位对拍 + 输入/输出饱和统计（T4）。
  A2   模式：权重 fp（W/swc 不取整），激活量化/requant/clamp 照部署
              ——回答"头权重量化是否有害"。
  A3   模式：权重照部署 int8，累加器直接出（不 requant、不 clamp、
              不进 int8 网格），out = acc*(sa*swc_j)
              ——回答"输出 requant/clamp 是否是回拉主体"。

数值口径与部署一致：q_round(torch.round 五成双)、K+1 增广偏置列
（w_bias_int8 原样）、RTN=floor((acc*m+2^(s-1))/2^s)、sat8=clamp(-128,127)。
A2/A3 走 fp_after 同款旁路（segment 不执行，输出照常 reg），与
run000bisect809 的 5 个未跑段行为一致。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


def _im2col_true(mod, x):
    """Conv1d 的真值 im2col（H 维不 pad）。

    HostDriver.im2col 把 Conv1d 垫成 (N,C,1,L) 后用标量 padding，H=1
    也被垫成 3，窗口数 3×（真值只在 h=pad 那片，部署段靠 STORE 寻址
    只回写真行）。旁路计算不能带这些伪行，这里 H 维 padding 置 0。
    Conv2d 沿用原版语义。"""
    if isinstance(mod, torch.nn.Conv1d):
        g = lambda v: v[0] if isinstance(v, tuple) else v
        ks, dil = g(mod.kernel_size), g(mod.dilation)
        pad, st = g(mod.padding), g(mod.stride)
        x4 = x[:, :, None, :]
        cols = F.unfold(x4, (1, ks), dilation=dil, padding=(0, pad),
                        stride=st)
        return cols.transpose(1, 2).reshape(-1, cols.shape[1])
    from host_driver import HostDriver
    return HostDriver.im2col(mod, x)

sys.path.insert(0, '/tmp/pcw_rtn/sw')
import host_driver as hd
from host_driver import HostDriver, q_round

HEAD5 = [
    'decoder.head.convs.0', 'decoder.head.convs.1',
    'decoder.head.output_layers.1', 'decoder.head.output_layers.3',
    'decoder.head.output_layers.4',
]


class HeadProbeDriver(HostDriver):
    mode = 'stats'
    probe_stats = None      # {module: {...}}

    # ---------- 宿主侧复算部署量化链 ----------
    def _head_replay(self, nm, mod, node, rec, x):
        e = self.calib.get(nm)
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Conv1d)):
            x2d = _im2col_true(mod, x)
        else:
            x2d = x.contiguous().reshape(-1, x.shape[-1])
        sa = float(e['sa'])
        q = q_round(x2d, sa)                                   # int8 [m,k]
        W = mod.weight.detach().float()
        W2 = W.reshape(W.shape[0], -1)                          # [n,k]
        wb = e.get('w_bias_int8')
        c_aug = float(e.get('bias_aug_c') or 1.0)
        if e.get('pcw'):
            sc = torch.tensor(e['swc'], dtype=torch.float64)
            if self.mode == 'A2':
                Wq = (W2.double() / sc[:, None])
            else:
                Wq = torch.clamp(torch.round(W2.double() / sc[:, None]),
                                 -127.0, 127.0)
            ms = e['rq_ms_col']
            mj = torch.tensor([m for m, s in ms], dtype=torch.float64)
            sj = torch.tensor([s for m, s in ms], dtype=torch.float64)
        else:
            sw = float(e['sw'])
            sc = torch.full((W2.shape[0],), sw, dtype=torch.float64)
            if self.mode == 'A2':
                Wq = W2.double() / sw
            else:
                Wq = torch.clamp(torch.round(W2.double() / sw),
                                 -127.0, 127.0)
            mj = torch.full((W2.shape[0],), float(e['m_requant']),
                            dtype=torch.float64)
            sj = torch.full((W2.shape[0],), float(e['s_shift']),
                            dtype=torch.float64)
        qa = q.double()
        if wb:
            col = torch.full((q.shape[0], 1), c_aug, dtype=torch.float64)
            qa = torch.cat([qa, col], 1)
            wcol = torch.tensor(wb, dtype=torch.float64).reshape(-1, 1)
            Wq = torch.cat([Wq, wcol], 1)
        acc = qa @ Wq.T                                         # fp64 [m,n]
        if self.mode == 'A3':
            out2d = acc * (sa * sc)[None, :]
            extra = dict(pre_max_ratio=float('nan'))
        else:
            rn = torch.where(sj >= 1.0, 2.0 ** (sj - 1.0),
                             torch.zeros_like(sj))
            pre = acc * mj[None, :] + rn[None, :]
            yq = torch.clamp(torch.floor(pre / (2.0 ** sj)[None, :]),
                             -128.0, 127.0)
            out2d = yq * float(e['so'])
            # pre-clamp 超限比值（回拉强度）：|pre/2^s| 相对 127 的最大倍数
            pmax = (pre / (2.0 ** sj)[None, :]).abs().max().item()
            extra = dict(pre_max_ratio=float(pmax) / 127.0)
        # 输入/输出饱和统计（对 q 与 pre-clamp 值）
        st = self.probe_stats.setdefault(nm, dict(
            calls=0, n_in=0, in_sat=0, in_max_ratio=0.0,
            n_out=0, out_sat=0, out_max_ratio=0.0, xmatch=0, xn=0))
        v = (x2d.double() / sa).abs()
        st['calls'] += 1
        st['n_in'] += int(v.numel())
        st['in_sat'] += int((v > 127.5).sum().item())
        st['in_max_ratio'] = max(st['in_max_ratio'],
                                 float(v.max().item()) / 127.0)
        if self.mode != 'A3':
            pv = ((acc * mj[None, :]) / (2.0 ** sj)[None, :]).abs()
            st['n_out'] += int(pv.numel())
            st['out_sat'] += int((pv > 127.5).sum().item())
            st['out_max_ratio'] = max(st['out_max_ratio'],
                                      float(pv.max().item()) / 127.0)
        else:
            st['n_out'] += int(acc.numel())
        st['pre_max_ratio_last'] = extra['pre_max_ratio']
        return out2d, yq if self.mode != 'A3' else None

    def gemm_call(self, mod, node, rec, x):
        nm = node['module']
        if nm not in HEAD5:
            return super().gemm_call(mod, node, rec, x)
        self.stats['gemm_calls'] += 1
        out2d, yq = self._head_replay(nm, mod, node, rec, x)
        out = self._shape_out(out2d.float(), mod, rec['in_shapes'][0],
                              rec['out_shapes'][0])
        if self.mode == 'stats':
            # 对拍：真跑部署段，逐位比对宿主复算（Conv1d 输出 (N,C,L)
            # 需 permute 到 (N,L,C) 才与 out2d 行序一致）
            dep = super().gemm_call(mod, node, rec, x)
            if isinstance(mod, torch.nn.Conv1d) and dep.dim() == 3:
                dep2d = dep.reshape(dep.shape[0], dep.shape[1], -1) \
                    .permute(0, 2, 1).reshape(-1, dep.shape[1])
            else:
                dep2d = dep.reshape(-1, dep.shape[-1])
            dep2d = dep2d / float(self.calib[nm]['so'])
            mine = (out2d / float(self.calib[nm]['so']))
            eq = (dep2d.round().long() == mine.round().long())
            st = self.probe_stats[nm]
            st['xmatch'] += int((~eq).sum().item())
            st['xn'] += int(eq.numel())
            d = (dep2d.round() - mine.round()).abs()
            st['dm_max'] = max(st.get('dm_max', 0.0), float(d.max().item()))
            st['dm_sum'] = st.get('dm_sum', 0.0) + float(d.sum().item())
            # rail 占比 + 逐通道时间唯一值（常数通道数）
            st['dep_rail'] = st.get('dep_rail', 0) + int(
                (dep2d.round().abs() >= 126.5).sum().item())
            st['my_rail'] = st.get('my_rail', 0) + int(
                (mine.round().abs() >= 126.5).sum().item())
            dep_u = dep2d.round()
            my_u = mine.round()
            st['dep_const_ch'] = st.get('dep_const_ch', 0) + int(
                sum(1 for c in range(dep_u.shape[1])
                    if len(dep_u[:, c].unique()) == 1))
            st['my_const_ch'] = st.get('my_const_ch', 0) + int(
                sum(1 for c in range(my_u.shape[1])
                    if len(my_u[:, c].unique()) == 1))
            st['ch_n'] = st.get('ch_n', 0) + int(dep_u.shape[1])
            return dep
        # A2/A3：fp_after 同款旁路（不跑段，reg 输出）
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Conv1d)):
            self.reg(rec['in_ids'][0], self.im2col(mod, x))
        else:
            self.reg(rec['in_ids'][0], x)
        self.reg(rec['out_ids'][0], out)
        return out


def run_probe(args, sample_id):
    HB = '~/workspace/holobrain'
    if HB not in sys.path:
        sys.path.insert(0, HB)
    import bringup  # noqa: F401
    from robo_orchard_lab.models.holobrain.processor import HoloBrainProcessor
    from robo_orchard_lab.models.mixin import ModelMixin
    import compiler

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
    hd._fix_kinematics_device(batch)
    drv = HeadProbeDriver(args.build, args.trace, args.calib, model,
                          engine='fast', fp_after=None)
    drv.P = P
    drv.mode = args.mode
    drv.probe_stats = {}
    t0 = time.perf_counter()
    drv.patch()
    t1 = time.perf_counter()
    with torch.no_grad():
        outs = model(batch)
    pa = outs[0]['pred_actions'].detach().cpu().numpy()
    t2 = time.perf_counter()
    act = pa[0, :, :, 0]
    np.savez(args.out, action=act, pred_actions_raw=pa[0])
    print(f'[probe] mode={args.mode} patch={t1 - t0:.1f}s '
          f'forward={t2 - t1:.1f}s segments={drv.stats["segments"]}/'
          f'{len(drv.seg_names)} gemm_calls={drv.stats["gemm_calls"]}')
    report = dict(sample=sample_id, mode=args.mode,
                  action_shape=list(act.shape),
                  segments_run=drv.stats['segments'],
                  segments_total=len(drv.seg_names),
                  forward_s=round(t2 - t1, 2), wall_s=round(t2 - t0, 2))
    if args.ref and os.path.exists(args.ref):
        ref = np.load(args.ref)
        d = np.abs(act - ref['action'])
        report.update(jpos_mae=float(d.mean()), jpos_max=float(d.max()),
                      jpos_mae_arm12=float(d[:, :12].mean()),
                      jpos_mae_gripper=float(d[:, [6, 13]].mean()),
                      per_joint_mae=[float(d[:, j].mean())
                                     for j in range(d.shape[1])],
                      ref_seed=int(ref['seed']) if 'seed' in ref.files else None)
        print(f'[vs-ref] mode={args.mode} jpos MAE={d.mean():.4f} '
              f'max={d.max():.4f}')
    # probe stats 汇总
    for nm, st in drv.probe_stats.items():
        st['in_sat_frac'] = st['in_sat'] / max(st['n_in'], 1)
        st['out_sat_frac'] = (st['out_sat'] / max(st['n_out'], 1)
                              if st['n_out'] else None)
        print(f'[stat] {nm}: in_sat={st["in_sat_frac"]:.4f} '
              f'in_max/127={st["in_max_ratio"]:.2f} '
              f'out_sat={st["out_sat_frac"] if st["out_sat_frac"] is None else round(st["out_sat_frac"], 4)} '
              f'out_max/127={st["out_max_ratio"]:.2f} '
              f'xcheck_mismatch={st["xmatch"]}/{st["xn"]} '
              f'dm_max={st.get("dm_max", 0.0):.0f} '
              f'dm_mean={st.get("dm_sum", 0.0) / max(st["xn"], 1):.2f} '
              f'dep_rail={st.get("dep_rail", 0)} my_rail={st.get("my_rail", 0)} '
              f'dep_const_ch={st.get("dep_const_ch", 0)}/{st.get("ch_n", 0)} '
              f'my_const_ch={st.get("my_const_ch", 0)}/{st.get("ch_n", 0)}')
    report['probe_stats'] = drv.probe_stats
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', required=True)
    ap.add_argument('--trace', required=True)
    ap.add_argument('--batch', required=True)
    ap.add_argument('--ref', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--mode', choices=['stats', 'A2', 'A3'], default='stats')
    ap.add_argument('--sample-id', default='000')
    args = ap.parse_args()
    rep = run_probe(args, args.sample_id)
    with open(args.out.replace('.npz', '_vs_fp32.json'), 'w',
              encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f'[done] {args.out}')


if __name__ == '__main__':
    main()
