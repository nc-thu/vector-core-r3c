# -*- coding: utf-8 -*-
"""mk_ideal_calib.py — 用本样本各 GEMM 输入的实际 absmax 生成"无 clamp"
校准表（2026-08-31，定论实验：区分 sa 校准失配 vs 链路 bug）。

规则（最小干预）：
  sa_new = max(sa_old, absmax_sample / 127)
  —— 只对发生 clamp 的层放大 sa 到刚好消除截断；没 clamp 的层保持
     原 sa（分辨率不降）。r_star/m_requant/s_shift 用 02_quant 同一
     v1_encode 规则重算，其余字段（sw/so/bias 等）原样保留。
输出 ideal 表后配合重编译，对比端到端 jpos MAE：
  若 MAE 显著回落 → 剩余误差主体是校准集分布失配（02_quant 问题）；
  若不动 → 链上还有实现问题，继续查。
"""
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

M_LIM = 32767


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
    import host_driver as HD

    calib_fp = sys.argv[1] if len(sys.argv) > 1 else \
        '/tmp/ae_hostdrv/hw_calib_table_v2.json'
    batch_fp = sys.argv[2] if len(sys.argv) > 2 else \
        '/tmp/ae_hostdrv/batch_s000.pt'
    out_fp = sys.argv[3] if len(sys.argv) > 3 else \
        '/tmp/ae_hostdrv/hw_calib_table_ideal.json'
    seed = 20260830

    torch.manual_seed(seed)
    processor = HoloBrainProcessor.load(bringup.CKPT, bringup.PROCESSOR_JSON)
    model = ModelMixin.load_model(bringup.MODEL_DIR, load_impl='native')
    model = model.float().eval()
    batch = torch.load(batch_fp, map_location='cpu', weights_only=False)
    HD._fix_kinematics_device(batch)
    drv = HD.HostDriver('/tmp/ae_hostdrv/build_s000_v3',
                        '/tmp/ae_hostdrv/trace_s000.json', calib_fp, model)
    import compiler
    drv.P = compiler.PROFILES['full']

    table = json.load(open(calib_fp, encoding='utf-8'))
    gemms = table['gemms']
    mods = dict(model.named_modules())

    # hook 每个 gemm 模块抓输入（im2col 后的 A 分布）
    absmax = {}

    def mk(name):
        def h(mod, inp, out):
            x = inp[0]
            if not torch.is_tensor(x):
                return
            try:
                a = drv.im2col(mod, x)
            except Exception:
                a = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
            v = float(a.detach().abs().max())
            if v > absmax.get(name, 0.0):
                absmax[name] = v
        return h

    C_CANDIDATES = (1, 2, 4, 8, 16, 32, 64)

    hs = []
    hooked = 0
    for name in gemms:
        key = name.split('#')[0]
        if key in mods and mods[key] is not None:
            hs.append(mods[key].register_forward_hook(mk(key)))
            hooked += 1
    print(f'[ideal] 表内 {len(gemms)} 条，hook 到 {hooked} 个模块', flush=True)
    with torch.no_grad():
        model(batch)
    for h in hs:
        h.remove()

    n_up = 0
    n_keep = 0
    n_nodata = 0
    ups = []
    for name, e in gemms.items():
        key = name.split('#')[0]
        v = absmax.get(key)
        if v is None:
            n_nodata += 1
            continue
        sa_old = e.get('sa')
        if sa_old is None:               # exempt_fp 等特殊条目，原样保留
            n_nodata += 1
            continue
        sa_new = max(sa_old, v / 127.0)
        e['a_absmax_sample'] = v
        if sa_new > sa_old * 1.0001:
            n_up += 1
            ups.append((sa_new / sa_old, name, sa_old, sa_new))
            e['sa'] = sa_new
            # bias 三元组随 sa 重算（b_acc = bias/(sa*sw)，C 候选取
            # round 后 |w|<=127 的最小 c；fit 不进则转 fp fallback）
            mod = mods[key]
            if mod.bias is not None:
                b_acc = mod.bias.detach().double() / (sa_new * e['sw'])
                b_abs = float(b_acc.abs().amax())
                c_sel, wb = None, None
                for c in C_CANDIDATES:
                    w_try = torch.round(b_acc / c)
                    if float(w_try.abs().amax()) <= 127.0:
                        c_sel, wb = c, w_try
                        break
                if c_sel is None:
                    e['bias_fp_fallback'] = True
                    e['b_acc_absmax'] = b_abs
                else:
                    e['bias_fp_fallback'] = False
                    e['bias_aug_c'] = c_sel
                    e['w_bias_int8'] = [int(t) for t in wb.cpu().tolist()]
                    e['b_acc_absmax'] = b_abs
        else:
            n_keep += 1
        r_star = (e['sa'] * e['sw']) / e['so']
        e['r_star'] = r_star
        m, s = v1_encode(r_star)
        e['m_requant'], e['s_shift'] = m, s
        e['acc_absmax_est'] = 127.0 / r_star
    print(f'[ideal] 放大 sa: {n_up} 层，保持: {n_keep}，无数据: {n_nodata}',
          flush=True)
    ups.sort(reverse=True)
    for r_, name, o, nw in ups[:10]:
        print('  x%-6.2f %-60s sa %.5f -> %.5f' % (r_, name[:60], o, nw))
    with open(out_fp, 'w', encoding='utf-8') as f:
        json.dump(table, f, ensure_ascii=False)
    print('[done]', out_fp)


if __name__ == '__main__':
    main()
