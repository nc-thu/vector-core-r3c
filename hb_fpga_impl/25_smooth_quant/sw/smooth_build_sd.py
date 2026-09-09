# smooth_build_sd.py -- 按平滑计划改 state_dict（25_smooth_quant）
#   LN:  gamma/=s, beta/=s（LayerNorm 有 bias，RMSNorm 无）
#   Linear: W'[:, i] *= s_i（即 W·diag(s)，按输入通道）
#   MHA in_proj_weight: 同 Linear（输入通道 = 最后一维）
# 数学上 fp 域逐位等价（乘除同 s），T1 fp 门独立验证。
#
# Run:
#   $PY smooth_build_sd.py --plan probe_out.json --scope s5 \
#       --out /tmp/alg_smooth/smoothed_sd_s5.pt --edits /tmp/alg_smooth/edits_s5.json
import argparse
import json
import os
import sys

HB = os.path.expanduser("~/workspace/holobrain")
for p in (HB, os.path.join(HB, "quant"), "/tmp/pcw_rtn"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

import bringup  # noqa: E402
from robo_orchard_lab.models.holobrain.processor import (  # noqa: E402
    HoloBrainProcessor,
)
from robo_orchard_lab.models.mixin import ModelMixin  # noqa: E402

SCOPES = {
    # S5：robot_encoder 入口段（调用 #190~#214 的全部 LN 喂给 Linear 组）
    "s5": lambda n: n.startswith("decoder.robot_encoder."),
    # S6：+ 解码器 DiT 层的 img/text cross-attn q/k/v + 前面全部编码器/增强器
    "s6": lambda n: True,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--scope", choices=sorted(SCOPES), default="s5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--edits", required=True)
    a = ap.parse_args()

    plan = json.load(open(a.plan, encoding="utf-8"))["plan"]
    keep = [e for e in plan if SCOPES[a.scope](e["norm"])]
    print(f"[plan] {len(keep)}/{len(plan)} groups in scope '{a.scope}'",
          flush=True)

    processor = HoloBrainProcessor.load(bringup.CKPT, bringup.PROCESSOR_JSON)
    model = ModelMixin.load_model(bringup.MODEL_DIR, load_impl="native")
    model = model.float().eval()
    sd = model.state_dict()
    edits = []
    for e in keep:
        s = torch.tensor(e["s"], dtype=torch.float32)
        # norm 侧
        for k in (e["norm"] + ".weight", e["norm"] + ".bias"):
            if k in sd:
                before = float(sd[k].abs().max())
                sd[k] = (sd[k].float() / s).to(sd[k].dtype)
                edits.append({"key": k, "op": "div_s",
                              "absmax_before": before,
                              "absmax_after": float(sd[k].abs().max())})
        # 权重侧
        for m in e["members"]:
            k = m + ".weight" if not m.endswith("in_proj_weight") else m
            assert k in sd, k
            W = sd[k].detach().float()
            assert W.shape[1] == s.numel(), (k, W.shape, s.numel())
            sd[k] = (W * s.unsqueeze(0)).to(sd[k].dtype)
            edits.append({"key": k, "op": "mul_s_right",
                          "absmax_before": float(W.abs().max()),
                          "absmax_after": float(sd[k].abs().max())})
    torch.save(sd, a.out)
    with open(a.edits, "w", encoding="utf-8") as f:
        json.dump({"scope": a.scope, "n_groups": len(keep),
                   "n_edits": len(edits), "edits": edits}, f,
                  ensure_ascii=False, indent=1)
    dg = sum(e["absmax_after"] / max(e["absmax_before"], 1e-30)
             for e in edits if e["op"] == "mul_s_right")
    ng = max(1, sum(1 for e in edits if e["op"] == "mul_s_right"))
    print(f"[done] {len(keep)} groups, {len(edits)} edits -> {a.out}; "
          f"权重 absmax 平均变化 x{dg/ng:.3f}", flush=True)


if __name__ == "__main__":
    main()
