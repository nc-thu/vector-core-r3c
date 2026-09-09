# smooth_probe_collect.py -- SmoothQuant LN 折叠轮：邻接探测 + 统计 + 平滑计划
#（服务器 CPU，25_smooth_quant / alg_smooth）
#
# 做三件事（对原始 fp 模型，一次真实样本前向）：
#   1) 探测：leaf 模块 hook，按 data_ptr 跟踪张量流。Linear/MHA 的输入若由
#      某个普通 LayerNorm/RMSNorm（非 Ada 调制式）直接产出，且该 norm 的输出
#      只被这一组 Linear/MHA 消费，则可平滑（probe 判定）。
#      robot_encoder 的 joint_self_attn 前有 permute+flatten（通道不变的
#      拷贝），data_ptr 追不上 → 用源码核实的 allowlist（op 顺序
#      [norm, joint_self_attn, None, None, norm, ffn]*4+[norm]）。
#   2) 统计：组成员 Linear 输入的逐通道 absmax（跨调用取 max）、通道比值
#      max/median、对 v2 表 sa 的饱和率；W 逐输入通道 absmax。
#   3) 计划：s_i = Xmax_i^a / Wmax_i^(1-a)，除以几何均值归一，clamp [2^-6,2^6]。
#      分组 = norm 名；成员 = 消费该 norm 的 Linear（q/k/v 一组）。
#
# Run (server):
#   cd ~/workspace/holobrain && CUDA_VISIBLE_DEVICES= $PY /tmp/alg_smooth/smooth_probe_collect.py \
#       --batch /tmp/ae_hostdrv/batch_s000.pt --v2 /tmp/ae_hostdrv/hw_calib_table_v2.json \
#       --out /tmp/alg_smooth/probe_out.json --alpha 0.5
import argparse
import json
import math
import os
import re
import sys
import time

HB = os.path.expanduser("~/workspace/holobrain")
for p in (HB, os.path.join(HB, "quant"), "/tmp/pcw_rtn"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import bringup  # noqa: E402
from gate import force_mha_slow_path  # noqa: E402
from gate_real import _fix_kinematics_device  # noqa: E402
from robo_orchard_lab.models.holobrain.processor import (  # noqa: E402
    HoloBrainProcessor,
)
from robo_orchard_lab.models.mixin import ModelMixin  # noqa: E402

NORM_TYPES = (nn.LayerNorm, nn.RMSNorm)

# 源码核实的 allowlist：robot_encoder.layers 的 op 顺序（config gd_common：
# [norm, joint_self_attn, None, None, norm, ffn]*4+[norm]）。attn 在 i%6==1，
# ffn 在 i%6==5，生产者 norm = i-1。norm 输出只进该 op（identity 用 norm 前
# 的 x，源码 robot_state_encoder.forward核实）。
RE_ENC_ATTN = re.compile(
    r"^decoder\.robot_encoder\.layers\.(\d+)\.(q_proj|k_proj|v_proj)$")
RE_ENC_FFN = re.compile(
    r"^decoder\.robot_encoder\.layers\.(\d+)\.ffn\.layers\.0\.0$")


def leaf_tag(mod):
    if isinstance(mod, nn.MultiheadAttention):
        return "mha"
    if isinstance(mod, nn.Linear):
        return "linear"
    if isinstance(mod, NORM_TYPES) and "Ada" not in type(mod).__name__:
        return "norm"
    return None


class FlowTracer:
    """leaf 模块事件流：norm 输出 data_ptr -> norm 名（持有引用防复用）；
    linear/mha 每次调用的输入 ptr -> 生产者 norm（时间序增量匹配）。"""

    def __init__(self, model):
        self.norm_out = {}     # ptr -> norm name
        self.hold = []         # 防 gc 复用地址
        self.lin_calls = []    # (linear_name, tag, ptr_or_None, norm_or_None)
        self.norm_consumers = {}  # norm name -> set(consumer name)
        self.norm_nout = {}    # norm name -> 输出 ptr 总次数
        self.handles = []
        for name, mod in model.named_modules():
            tag = leaf_tag(mod)
            if tag is None or name == "":
                continue

            def pre(m, args, kwargs, name=name, tag=tag):
                m._slot = {"name": name, "tag": tag, "ins": []}
                src_args = args if args else [kwargs.get(k) for k in
                                               ("query", "x", "value")
                                               if kwargs.get(k) is not None]
                for a in src_args:
                    if torch.is_tensor(a) and torch.is_floating_point(a):
                        m._slot["ins"].append(a.data_ptr())

            def post(m, args, out, name=name, tag=tag):
                slot = getattr(m, "_slot", None)
                if slot is None:
                    return
                outs = []

                def walk(o):
                    if torch.is_tensor(o) and torch.is_floating_point(o):
                        outs.append(o)
                    elif isinstance(o, (list, tuple)):
                        for x in o:
                            walk(x)
                    elif isinstance(o, dict):
                        for x in o.values():
                            walk(x)

                walk(out)
                if tag == "norm":
                    for t in outs:
                        self.hold.append(t)
                        self.norm_out[t.data_ptr()] = name
                        self.norm_nout[name] = self.norm_nout.get(name, 0) + 1
                else:  # linear / mha
                    ptr = slot["ins"][0] if slot["ins"] else None
                    src = self.norm_out.get(ptr)
                    self.lin_calls.append(
                        (name, tag, ptr, src))
                    if src is not None:
                        self.norm_consumers.setdefault(src, set()).add(
                            (name, tag))
                    # 持有该输入，防其地址被回收后误配
                    for a in args:
                        if torch.is_tensor(a) and torch.is_floating_point(a):
                            self.hold.append(a)
                m._slot = None

            self.handles.append(
                mod.register_forward_pre_hook(pre, with_kwargs=True))
            self.handles.append(mod.register_forward_hook(post))

    def remove(self):
        for h in self.handles:
            h.remove()


def per_channel_absmax(x):
    """[..., C] -> [C]，逐通道 absmax（浮点输入）。"""
    if not torch.is_tensor(x) or not torch.is_floating_point(x):
        return None
    if x.dim() == 0 or x.shape[-1] < 2:
        return None
    c = x.shape[-1]
    return x.detach().abs().reshape(-1, c).amax(0).double().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="/tmp/ae_hostdrv/batch_s000.pt")
    ap.add_argument("--v2", default="/tmp/ae_hostdrv/hw_calib_table_v2.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260830)
    a = ap.parse_args()

    t0 = time.perf_counter()
    processor = HoloBrainProcessor.load(bringup.CKPT, bringup.PROCESSOR_JSON)
    model = ModelMixin.load_model(bringup.MODEL_DIR, load_impl="native")
    model = model.float().eval()
    n_mha = force_mha_slow_path(model)
    print(f"[load] model up (eager MHA x{n_mha}) {time.perf_counter()-t0:.0f}s",
          flush=True)

    batch = torch.load(a.batch, map_location="cpu", weights_only=False)
    _fix_kinematics_device(batch)
    v2 = json.load(open(a.v2, encoding="utf-8"))["gemms"]

    # ---- pass 1: 邻接探测（ tracer + 统计 hook 一起挂，跑一次前向） ----
    tracer = FlowTracer(model)
    stat_absmax = {}   # linear name -> [C] running max
    stat_sat = {}      # linear name -> [n_elem_over, n_elem_total] vs v2 sa
    stat_handles = []

    def mk_stat_hook(name):
        sa = v2.get(name, {}).get("sa")

        def pre(m, args):
            if not args:
                return
            x = args[0]
            if not torch.is_tensor(x) or not torch.is_floating_point(x):
                return
            am = per_channel_absmax(x)
            if am is None:
                return
            if name in stat_absmax:
                torch.maximum(stat_absmax[name], am, out=stat_absmax[name])
            else:
                stat_absmax[name] = am.clone()
            if sa is not None:
                e = stat_sat.setdefault(name, [0, 0, 0])
                xf = x.detach().abs().reshape(-1)
                e[0] += int((xf > sa * 127.0).sum())
                e[1] += int(xf.numel())
                e[2] += int((xf > sa * 127.0 * 0.9).sum())
        return pre

    # 统计 hook 先挂到全部 Linear/MHA（探测完再筛也行，先全收）
    for name, mod in model.named_modules():
        if name == "":
            continue
        if isinstance(mod, (nn.Linear, nn.MultiheadAttention)):
            stat_handles.append(
                mod.register_forward_pre_hook(mk_stat_hook(name)))

    torch.manual_seed(a.seed)
    with torch.no_grad():
        outs = model(batch)
    print(f"[fwd ] 1 real forward {time.perf_counter()-t0:.0f}s", flush=True)
    for h in stat_handles:
        h.remove()
    tracer.remove()

    # ---- 探测分析：strict 组（norm 输出只被一组 linear/mha 消费） ----
    lin_src = {}   # linear name -> set(producer norms over calls)
    for (name, tag, ptr, src) in tracer.lin_calls:
        if tag in ("linear", "mha"):
            lin_src.setdefault(name, set()).add(src)  # src 可能 None
    groups = {}    # norm name -> set(member linear names)   [strict]
    for lname, srcs in lin_src.items():
        srcs = {s for s in srcs if s is not None}
        if len(srcs) == 1:
            n = next(iter(srcs))
            groups.setdefault(n, set()).add(lname)
    # norm 的全部消费者必须都在组内且都是 linear/mha
    strict_ok = {}
    for n, mem in list(groups.items()):
        cons = tracer.norm_consumers.get(n, set())
        ok = (all(ct in ("linear", "mha") for (_, ct) in cons)
              and {cn for (cn, _) in cons} <= set(mem))
        strict_ok[n] = bool(ok)
        if not ok:
            print(f"[skip] norm {n} 消费者不纯: {sorted(cons)}", flush=True)

    # ---- allowlist 组（源码核实） ----
    mods = dict(model.named_modules())
    allow_groups = {}
    for lname in lin_src:
        m1 = RE_ENC_ATTN.match(lname)
        m2 = RE_ENC_FFN.match(lname)
        if m1:
            nidx = int(m1.group(1)) - 1
        elif m2:
            nidx = int(m2.group(1)) - 1
        else:
            continue
        nname = f"decoder.robot_encoder.layers.{nidx}"
        nmod = mods.get(nname)
        if nmod is None or leaf_tag(nmod) != "norm":
            print(f"[skip] allowlist {lname}: {nname} 不是普通 norm", flush=True)
            continue
        allow_groups.setdefault(nname, set()).add(lname)

    # 合并：strict 通过的 + allowlist 的（重名以 allowlist 覆盖也一致）
    all_groups = {}
    for n, mem in groups.items():
        if strict_ok.get(n):
            all_groups[n] = {"members": sorted(mem), "source": "probe"}
    for n, mem in allow_groups.items():
        all_groups[n] = {"members": sorted(mem), "source": "allowlist"}

    # ---- 逐组建计划 ----
    sd = model.state_dict()
    plan = []
    skipped_nostat = []
    for n, g in sorted(all_groups.items()):
        mem = g["members"]
        # 通道数 & 权重逐输入通道 absmax（组内取 max）
        wmax = None
        C = None
        for mname in mem:
            wkeys = [mname + ".weight"]
            if mname.endswith("in_proj_weight"):
                wkeys = [mname]
            wk = None
            for k in wkeys:
                if k in sd:
                    wk = k
                    break
            if wk is None:
                wmax = None
                break
            W = sd[wk].detach().float()
            if W.dim() != 2:
                wmax = None
                break
            wm = W.abs().amax(dim=0).double().cpu()
            C = W.shape[1]
            wmax = wm if wmax is None else torch.maximum(wmax, wm)
        xmax = None
        for mname in mem:
            am = stat_absmax.get(mname)
            if am is None:
                continue
            if am.numel() != C:
                print(f"[skip] {n}: {mname} 统计通道 {am.numel()} != 权重 {C}",
                      flush=True)
                xmax = None
                break
            xmax = am if xmax is None else torch.maximum(xmax, am)
        if wmax is None or xmax is None or C is None:
            skipped_nostat.append(n)
            continue
        alpha = a.alpha
        s = torch.clamp(xmax.clamp(min=1e-12), min=1e-12).pow(alpha) / \
            wmax.clamp(min=1e-12).pow(1.0 - alpha)
        s = s / s.log().mean().exp()                # 除以几何均值
        s = torch.clamp(s, 2.0 ** -6, 2.0 ** 6)
        ratio_before = float(xmax.max() / xmax.median().clamp(min=1e-30))
        ratio_after = float((xmax / s).max() / (xmax / s).median().clamp(min=1e-30))
        entry = {
            "norm": n,
            "norm_kind": "rms" if isinstance(mods[n], nn.RMSNorm)
            and "Ada" not in type(mods[n]).__name__ else "layer",
            "members": mem,
            "source": g["source"],
            "C": C,
            "s": [float(v) for v in s],
            "s_min": float(s.min()), "s_max": float(s.max()),
            "ratio_before": ratio_before,
            "ratio_after_pred": ratio_after,
            "xmax_ratio": [float(v) for v in (xmax / s)],
        }
        plan.append(entry)

    # 统计信息（含未入组层，供诊断）
    stats = {}
    for lname, am in stat_absmax.items():
        sa = v2.get(lname, {}).get("sa")
        sat = stat_sat.get(lname)
        stats[lname] = {
            "ratio_max_median": float(am.max() / am.median().clamp(min=1e-30)),
            "v2_sa": sa,
            "sat_frac": (sat[0] / sat[1]) if sat and sat[1] else None,
            "sat_frac_09": (sat[2] / sat[1]) if sat and sat[1] else None,
        }

    out = {
        "meta": {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "batch": a.batch,
            "alpha": a.alpha,
            "seed_fwd": a.seed,
            "n_lin_hooked": len(stat_absmax),
            "n_strict_groups": sum(1 for v in strict_ok.values() if v),
            "n_skip_impure": sum(1 for v in strict_ok.values() if not v),
            "n_allow_groups": len(allow_groups),
            "skipped_nostat": skipped_nostat,
        },
        "plan": plan,
        "stats": stats,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    ns5 = sum(1 for e in plan if e["norm"].startswith("decoder.robot_encoder."))
    print(f"[done] groups={len(plan)} (S5 robot_encoder={ns5}) -> {a.out} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)
    for e in plan[:40]:
        print(f"  [grp] {e['norm']} ({e['source']},{e['norm_kind']},C={e['C']})"
              f" mem={len(e['members'])} s=[{e['s_min']:.3f},{e['s_max']:.3f}] "
              f"ratio {e['ratio_before']:.1f}->{e['ratio_after_pred']:.1f}",
              flush=True)


if __name__ == "__main__":
    main()
