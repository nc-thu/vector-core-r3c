# diag_real.py -- 真实样本激活诊断仪表盘（25_alg_calib，T1）
#
# 目的：对 fp 模型在真实 RoboTwin 样本上前向，hook v3 表覆盖的全部量化
# 模块（438 键，含 6 个 MHA in_proj 别名），收集输入/输出统计：
#   - 精确量：absmax（running max）、|x|>127*sa_v3 的饱和计数（精确）
#   - 采样量：reservoir 均匀采样 -> p99/p99.9/p99.99
#   - per-channel absmax（Linear 取末维，Conv 取通道维），算 max/median 比
# 段标签用 bisect 日志的调用序（#0-816）按首次出现位置分段。
#
# hook 口径与 02_quant hw_calib.calibrate 完全一致（同样的模块、同样取
# args[0]），只是数据从合成扰动批换成真实 batch —— 差异只在数据，不在语义。
# MHA in_proj 的"输出"用 F.linear(q, in_proj_weight) 现算（真输出，不是
# 注意力输出）。force_mha_slow_path 必须开，否则 eval 快路径绕过 out_proj。
#
# Run (server):
#   cd ~/workspace/holobrain && CUDA_VISIBLE_DEVICES= $PY /tmp/alg_calib/diag_real.py \
#       --batch /tmp/ae_hostdrv/batch_s000.pt --out /tmp/alg_calib/diag_real_s000.json
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

PCW = "/tmp/pcw_rtn"
HB = os.path.expanduser("~/workspace/holobrain")
for p in (PCW, HB, os.path.join(HB, "quant")):
    if p not in sys.path:
        sys.path.insert(0, p)

import bringup  # noqa: E402
from gate import force_mha_slow_path  # noqa: E402
from robo_orchard_lab.models.holobrain.processor import (  # noqa: E402
    HoloBrainProcessor,
)
from robo_orchard_lab.models.mixin import ModelMixin  # noqa: E402

V3_TABLE = "/tmp/pcw_rtn/hw_calib_table_pcw_v3.json"
BISECT_LOG = "/tmp/pcw_rtn/run000bisect815.log"
RES_CAP = 160000        # 每模块 reservoir 上限（输入/输出各一份）
PER_CALL_CAP = 4000     # 每次调用最多采多少值（均匀 stride）


def load_segments():
    """bisect 日志 -> 每个量化调用的 (idx, kind, module)。"""
    calls = []
    with open(BISECT_LOG, encoding="utf-8") as f:
        for line in f:
            if "[bisect] #" not in line:
                continue
            body = line.split("[bisect] #", 1)[1].strip()
            idx, kind, mod = body.split(" ", 2)
            calls.append((int(idx), kind, mod.strip()))
    return calls


def seg_of_module(calls):
    """模块 -> (首次调用 idx, 段名)。段名按调用区间硬编码（日志实测）。
    bisect 日志只记 attn 容器和直接 gemm 模块；表键（如 w_msa.qkv、
    cross_attn.q_proj 等成员）用最长前缀匹配归属到容器所在段。"""
    first = {}
    for idx, _k, mod in calls:
        if mod not in first:
            first[mod] = idx

    def seg(idx, name):
        if "patch_embed" in name:
            return "patch_embed"
        if name.startswith(("backbone.", "backbone_3d.", "neck", "neck_3d")):
            return "visual_backbone"
        if name.startswith("text_encoder"):
            return "bert_text"
        if name.startswith(("feature_enhancer", "spatial_enhancer")) \
                or name == "text_feat_map":
            return "feature_enhance"
        if name.startswith("decoder.robot_encoder"):
            return "robot_encoder_entry"
        if name.startswith("decoder.head") or "t_embed" in name \
                or name.startswith("decoder.input_layers"):
            return "output_head"
        if name.startswith("decoder."):
            return "decoder_body"
        return "other"

    segmap = {m: (i, seg(i, m)) for m, i in first.items()}
    logged = sorted(first, key=len, reverse=True)   # 长名优先

    def lookup(key):
        if key in segmap:
            return segmap[key]
        for m in logged:
            if key.startswith(m + "."):
                return segmap[m]
        return (None, "other")

    return lookup, first


def thin(res_list):
    arr = np.concatenate(res_list)
    return [arr[::2]]


def first_arg(args, kwargs):
    """args[0]，空则退 kwargs['query']（MHA 常用关键字调用）。"""
    if args:
        return args[0]
    return (kwargs or {}).get("query")


def upd(st, which, x):
    """更新 absmax（精确）+ 饱和计数（精确）+ reservoir（采样）。"""
    flat = x.detach().abs().reshape(-1)
    n = flat.numel()
    if n == 0:
        return
    am = float(flat.max())
    if am > st[which + "_absmax"]:
        st[which + "_absmax"] = am
    thr = 127.0 * st[which + "_scale_ref"]
    st[which + "_n"] += n
    st[which + "_over"] += int((flat > thr).sum())
    if n > PER_CALL_CAP:
        sub = flat[:: max(1, n // PER_CALL_CAP)]
    else:
        sub = flat
    res = st[which + "_res"]
    res.append(sub.numpy().astype(np.float32))
    if sum(len(r) for r in res) > RES_CAP:
        st[which + "_res"] = thin(res)


def upd_chan(st, x, is_conv):
    """per-channel running absmax。Linear/注意力输入取末维；Conv 取 dim1。"""
    a = x.detach().abs()
    if is_conv and a.dim() > 2:
        cm = a.amax(dim=tuple(range(1, a.dim())))
    else:
        cm = a.reshape(-1, a.shape[-1]).amax(0)
    cm = cm.float().numpy()
    if st["chan"] is None:
        st["chan"] = cm
    else:
        np.maximum(st["chan"], cm, out=st["chan"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260830)
    a = ap.parse_args()

    t0 = time.perf_counter()
    tab = json.load(open(V3_TABLE, encoding="utf-8"))["gemms"]
    q = {k: e for k, e in tab.items()
         if "sa" in e and not e.get("exempt_fp")}
    calls = load_segments()
    seg_lookup, first_call = seg_of_module(calls)
    print(f"[tab ] {len(q)} quantized keys, {len(calls)} bisect calls",
          flush=True)

    model = ModelMixin.load_model(bringup.MODEL_DIR, load_impl="native")
    model = model.float().eval()
    n_mha = force_mha_slow_path(model)
    print(f"[load] model up, eager MHA x{n_mha} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)
    mods = dict(model.named_modules())

    stats = {}
    unmatched = []
    handles = []
    for key, e in q.items():
        if key == "text_feat_map":
            unmatched.append((key, "非模块（常量映射），保持 v2"))
            continue
        tgt = mods.get(key)
        alias = None
        if tgt is None and key.endswith(".in_proj_weight"):
            alias = key[: -len(".in_proj_weight")]
            tgt = mods.get(alias)
        if tgt is None:
            unmatched.append((key, "找不到模块"))
            continue
        is_conv = e.get("type") == "nn.Conv2d" or e.get("type") == "nn.Conv1d"
        seg_i, seg_name = seg_lookup(key)
        st = dict(
            sa_v3=float(e["sa"]), so_v3=float(e["so"]),
            in_absmax=0.0, out_absmax=0.0, calls=0,
            in_res=[], out_res=[], chan=None,
            in_n=0, in_over=0, out_n=0, out_over=0,
            in_scale_ref=float(e["sa"]), out_scale_ref=float(e["so"]),
            seg=seg_name,
            first_idx=seg_i,
            is_alias_in_proj=alias is not None,
        )
        stats[key] = st

        if alias is not None:
            # MHA：输入=query（部署链 in_proj 的真实输入）；
            # 输出 = F.linear(q, in_proj_weight) 的真输出（不是注意力输出）
            W = tgt.in_proj_weight.detach().float()

            def pre(m, args, kwargs=None, st=st):
                x = first_arg(args, kwargs)
                if torch.is_tensor(x) and torch.is_floating_point(x):
                    st["calls"] += 1
                    upd(st, "in", x)
                    upd_chan(st, x, False)

            def post(m, args, kwargs, out, st=st, W=W):
                x = first_arg(args, kwargs)
                if not (torch.is_tensor(x) and torch.is_floating_point(x)):
                    return
                y = torch.nn.functional.linear(
                    x.detach().reshape(-1, x.shape[-1]).float(), W)
                upd(st, "out", y)
            kind = "mha-alias"
        else:
            def pre(m, args, kwargs=None, st=st, is_conv=is_conv):
                x = first_arg(args, kwargs)
                if torch.is_tensor(x) and torch.is_floating_point(x):
                    st["calls"] += 1
                    upd(st, "in", x)
                    upd_chan(st, x, is_conv)

            def post(m, args, out, st=st):
                o = out[0] if isinstance(out, (tuple, list)) else out
                if torch.is_tensor(o) and torch.is_floating_point(o):
                    upd(st, "out", o)
            kind = "module"
        handles.append(
            tgt.register_forward_pre_hook(pre, with_kwargs=True))
        if alias is not None:
            handles.append(tgt.register_forward_hook(post, with_kwargs=True))
        else:
            handles.append(tgt.register_forward_hook(post))
    print(f"[hook] {len(stats)} 键挂上（{kind} 混合），"
          f"unmatched={unmatched}", flush=True)

    batch = torch.load(a.batch, map_location="cpu", weights_only=False)
    # kinematics device 修复（host_driver.py:94 同款）
    def _fix(obj, seen=None):
        import torch as _t
        if seen is None:
            seen = set()
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, dict):
            for v in obj.values():
                _fix(v, seen)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _fix(v, seen)
        else:
            ch = getattr(obj, "chain", None)
            if ch is not None and hasattr(ch, "_root"):
                stack = [ch._root]
                while stack:
                    fr = stack.pop()
                    for side in ("joint", "link"):
                        jt = getattr(fr, side, None)
                        t = getattr(jt, "offset", None) if jt is not None \
                            else None
                        if t is not None and hasattr(t, "device"):
                            t.device = _t.device("cpu")
                            t.dtype = _t.float32
                    stack.extend(getattr(fr, "children", ()) or ())
                try:
                    ch.device = _t.device("cpu")
                    ch.dtype = _t.float32
                except AttributeError:
                    pass
    _fix(batch)
    print(f"[data] {a.batch} ({time.perf_counter()-t0:.0f}s)", flush=True)

    torch.manual_seed(a.seed)
    with torch.no_grad():
        outs = model(batch)
    pa = outs[0]["pred_actions"].detach().cpu().numpy()
    print(f"[fwd ] done ({time.perf_counter()-t0:.0f}s) "
          f"pred_actions {pa.shape}", flush=True)
    for h in handles:
        h.remove()

    # ---------------- 汇总 ----------------
    out_mod = {}
    chan_store = {}     # per-module 通道 absmax -> npz
    all_chan = []       # (ratio, key, chan_idx, chan_absmax, sa_v3) 全通道
    for key, st in stats.items():
        in_res = np.concatenate(st["in_res"]) if st["in_res"] \
            else np.zeros(0, np.float32)
        out_res = np.concatenate(st["out_res"]) if st["out_res"] \
            else np.zeros(0, np.float32)
        sa, so = st["sa_v3"], st["so_v3"]
        ent = dict(
            seg=st["seg"], first_idx=st["first_idx"], calls=st["calls"],
            sa_v3=sa, so_v3=so,
            in_absmax=st["in_absmax"], out_absmax=st["out_absmax"],
            sat_in_frac=(st["in_over"] / st["in_n"]) if st["in_n"] else 0.0,
            sat_out_frac=(st["out_over"] / st["out_n"]) if st["out_n"]
            else 0.0,
            util_in=st["in_absmax"] / (127.0 * sa) if sa > 0 else 0.0,
            util_out=st["out_absmax"] / (127.0 * so) if so > 0 else 0.0,
            alias_in_proj=st["is_alias_in_proj"],
        )
        if in_res.size:
            ent["in_p99"], ent["in_p999"], ent["in_p9999"] = [
                float(v) for v in np.percentile(
                    in_res, [99.0, 99.9, 99.99])]
        if out_res.size:
            ent["out_p99"], ent["out_p999"], ent["out_p9999"] = [
                float(v) for v in np.percentile(
                    out_res, [99.0, 99.9, 99.99])]
        if st["chan"] is not None and st["chan"].size:
            ch = st["chan"]
            chan_store[key.replace(".", "__")] = ch.astype(np.float32)
            med = float(np.median(ch))
            ent["n_chan"] = int(ch.size)
            ent["chan_med"] = med
            ent["chan_ratio"] = float(ch.max() / med) if med > 0 else None
            if med > 0:
                ratio = ch / med
                for ci in np.nonzero(ratio)[0]:
                    all_chan.append((float(ratio[ci]), key, int(ci),
                                     float(ch[ci]), sa))
        out_mod[key] = ent

    all_chan.sort(reverse=True)
    n_tot = len(all_chan)
    top01 = all_chan[: max(1, n_tot // 1000)]
    top1 = all_chan[: min(500, max(1, n_tot // 100))]

    # 段汇总
    seg_sum = {}
    for key, e in out_mod.items():
        s = seg_sum.setdefault(e["seg"], dict(
            n=0, util_in=[], sat_in=[], chan_r=[], firsts=[]))
        s["n"] += 1
        s["util_in"].append(e["util_in"])
        s["sat_in"].append(e["sat_in_frac"])
        if e.get("chan_ratio"):
            s["chan_r"].append(e["chan_ratio"])
        if e["first_idx"] is not None:
            s["firsts"].append(e["first_idx"])
    for sname, s in seg_sum.items():
        u = sorted(s["util_in"])
        seg_sum[sname] = dict(
            n=s["n"],
            call_range=(min(s["firsts"]), max(s["firsts"]))
            if s["firsts"] else None,
            util_in_median=float(np.median(u)),
            util_in_max=float(max(u)),
            n_util_gt105=int(sum(1 for v in u if v > 1.05)),
            n_util_gt125=int(sum(1 for v in u if v > 1.25)),
            sat_in_median=float(np.median(s["sat_in"])),
            sat_in_max=float(max(s["sat_in"])),
            chan_ratio_median=float(np.median(s["chan_r"]))
            if s["chan_r"] else None,
            chan_ratio_max=float(max(s["chan_r"])) if s["chan_r"] else None,
        )

    res = dict(
        meta=dict(
            generated=time.strftime("%Y-%m-%d %H:%M:%S"),
            batch=a.batch, seed=a.seed,
            res_cap=RES_CAP, per_call_cap=PER_CALL_CAP,
            unmatched=unmatched,
            n_modules=len(out_mod),
            n_chan_total=n_tot,
            top1_list_cap=len(top1),
        ),
        segments=seg_sum,
        top_chan_01pct=[dict(ratio=r, module=k, chan=c, absmax=v, sa_v3=s)
                        for r, k, c, v, s in top01],
        top_chan_1pct=[dict(ratio=r, module=k, chan=c, absmax=v, sa_v3=s)
                       for r, k, c, v, s in top1],
        modules=out_mod,
    )
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    npz = a.out.replace(".json", "_chan.npz")
    np.savez_compressed(npz, **chan_store)
    print(f"[save] {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB) + {npz} "
          f"({os.path.getsize(npz)/1e6:.1f} MB, {len(chan_store)} 模块)",
          flush=True)
    print(f"[done] {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
