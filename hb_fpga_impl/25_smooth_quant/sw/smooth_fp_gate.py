# smooth_fp_gate.py -- T1 fp 等价性门 + 平滑后统计（25_smooth_quant）
#
# 门：修改版 fp 模型 forward_actions（seed 20260830，与 fp32_ref 存档同款）
# 对 fp32_ref_000.npz 的 action 算 jpos MAE，必须 < 0.01（理想 ≈0，受
# DPM 噪声固定种子限制应为机器精度级）。
# 同时收集：组成员 Linear 输入的逐通道 absmax（平滑后）——T5 机制诊断用，
# 与 probe_out.json 的 stats_before 对比。
#
# Run:
#   cd ~/workspace/holobrain && CUDA_VISIBLE_DEVICES= $PY /tmp/alg_smooth/smooth_fp_gate.py \
#       --sd /tmp/alg_smooth/smoothed_sd_s5.pt --plan /tmp/alg_smooth/probe_out.json \
#       --scope s5 --ref /tmp/ae_hostdrv/fp32_ref_000.npz \
#       --gate-json /tmp/alg_smooth/fpgate_s5.json --stats /tmp/alg_smooth/stats_after_s5.json
import argparse
import json
import os
import sys
import time

HB = os.path.expanduser("~/workspace/holobrain")
for p in (HB, os.path.join(HB, "quant"), "/tmp/pcw_rtn"):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import bringup  # noqa: E402
from gate import force_mha_slow_path, forward_actions  # noqa: E402
from gate_real import _fix_kinematics_device  # noqa: E402
from robo_orchard_lab.models.holobrain.processor import (  # noqa: E402
    HoloBrainProcessor,
)
from robo_orchard_lab.models.mixin import ModelMixin  # noqa: E402

SCOPES = {
    "s5": lambda n: n.startswith("decoder.robot_encoder."),
    "s6": lambda n: True,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sd", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--scope", default="s5")
    ap.add_argument("--batch", default="/tmp/ae_hostdrv/batch_s000.pt")
    ap.add_argument("--ref", default="/tmp/ae_hostdrv/fp32_ref_000.npz")
    ap.add_argument("--v2", default="/tmp/ae_hostdrv/hw_calib_table_v2.json")
    ap.add_argument("--gate-json", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--seed", type=int, default=20260830)
    a = ap.parse_args()

    t0 = time.perf_counter()
    processor = HoloBrainProcessor.load(bringup.CKPT, bringup.PROCESSOR_JSON)
    model = ModelMixin.load_model(bringup.MODEL_DIR, load_impl="native")
    model = model.float().eval()
    n_mha = force_mha_slow_path(model)
    missing, unexpected = model.load_state_dict(
        torch.load(a.sd, map_location="cpu"), strict=True)
    print(f"[load] model up (MHA x{n_mha}), sd applied "
          f"{time.perf_counter()-t0:.0f}s", flush=True)

    plan = json.load(open(a.plan, encoding="utf-8"))["plan"]
    keep = [e for e in plan if SCOPES[a.scope](e["norm"])]
    members = sorted({m for e in keep for m in e["members"]})
    s_map = {e["norm"]: e["s"] for e in keep}

    batch = torch.load(a.batch, map_location="cpu", weights_only=False)
    _fix_kinematics_device(batch)
    ref = np.load(a.ref)
    v2 = json.load(open(a.v2, encoding="utf-8"))["gemms"]

    # 统计 hook（平滑后成员输入）
    stat_absmax, stat_sat = {}, {}
    handles = []

    def mk_hook(name):
        sa = v2.get(name, {}).get("sa")

        def pre(m, args):
            if not args:
                return
            x = args[0]
            if not torch.is_tensor(x) or not torch.is_floating_point(x):
                return
            if x.dim() == 0 or x.shape[-1] < 2:
                return
            am = x.detach().abs().reshape(-1, x.shape[-1]).amax(0).double().cpu()
            if name in stat_absmax:
                torch.maximum(stat_absmax[name], am, out=stat_absmax[name])
            else:
                stat_absmax[name] = am.clone()
            if sa is not None:
                e = stat_sat.setdefault(name, [0, 0])
                xf = x.detach().abs().reshape(-1)
                e[0] += int((xf > sa * 127.0).sum())
                e[1] += int(xf.numel())
        return pre

    for name, mod in model.named_modules():
        if name in set(members) and isinstance(
                mod, (nn.Linear, nn.MultiheadAttention)):
            handles.append(mod.register_forward_pre_hook(mk_hook(name)))

    pa = forward_actions(model, batch, a.seed)
    for h in handles:
        h.remove()
    ref_a = torch.from_numpy(ref["action"]).float()          # [64, 14]
    ours = pa[..., 0].reshape(-1, 14)                        # [64, 14]
    jpos = float((ours - ref_a).abs().mean())
    print(f"[gate] jpos vs fp32_ref = {jpos:.6f} "
          f"({'PASS <0.01' if jpos < 0.01 else 'FAIL >=0.01'}) "
          f"max={float((ours - ref_a).abs().max()):.6f} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)

    with open(a.gate_json, "w", encoding="utf-8") as f:
        json.dump({"jpos_mae": jpos,
                   "jpos_max": float((ours - ref_a).abs().max()),
                   "pass": bool(jpos < 0.01),
                   "seed": a.seed, "sd": a.sd, "scope": a.scope,
                   "n_groups": len(keep)}, f, indent=1)

    stats = {}
    for e in keep:
        for mname in e["members"]:
            am = stat_absmax.get(mname)
            if am is None:
                continue
            stats[mname] = {
                "norm": e["norm"],
                "ratio_max_median": float(
                    am.max() / am.median().clamp(min=1e-30)),
                "sat_frac": (stat_sat[mname][0] / stat_sat[mname][1]
                             if mname in stat_sat and stat_sat[mname][1]
                             else None),
                "absmax": [float(v) for v in am],
            }
    # 组级汇总（成员逐通道 absmax 取 max 后按 s 还原比较）
    grp_summary = []
    for e in keep:
        am = None
        for mname in e["members"]:
            v = stat_absmax.get(mname)
            if v is None:
                continue
            am = v if am is None else torch.maximum(am, v)
        if am is None:
            continue
        grp_summary.append({
            "norm": e["norm"],
            "ratio_before": e["ratio_before"],
            "ratio_after_measured": float(
                am.max() / am.median().clamp(min=1e-30)),
            "ratio_after_pred": e["ratio_after_pred"],
            "sat_frac_after": stats.get(e["members"][0], {}).get("sat_frac"),
        })
    with open(a.stats, "w", encoding="utf-8") as f:
        json.dump({"groups": grp_summary, "members": stats}, f,
                  ensure_ascii=False, indent=1)
    print(f"[done] {a.gate_json} / {a.stats} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
