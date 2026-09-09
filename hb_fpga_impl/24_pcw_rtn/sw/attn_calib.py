"""attn_calib.py -- 每个注意力模块的 plain q@k^T 积 absmax（服务器侧）。

为什么需要它：编译器两相注意力的 QK^T 合成 GEMM 现在用占位 requant
(_ph_rq，r=2048/(k*127^2))，与真实需要的 r_qk = so_q*so_k/sigma_S 差好几个
数量级，S_int 会被 floor 成 0 或直接饱和。这个脚本在标定分布上量出每个模块
的 sigma_S = absmax(plain q@k)/127（plain = 恰好 host 要量化进 PL 的那个
张量：rotary/temporal 在 rotary 之后、无 scale；JG 不含 query_pos 调制；
MHA 含 in_proj bias；BIMHA/Swin 不含 scale）。

标定样本 = hw_calib 的 8 个 bringup 扰动批 + 命令行给的 npz 真样本
（驱动要跑的就是真样本，必须盖住它们的分布）。

Run (server):
  cd ~/workspace/holobrain
  CUDA_VISIBLE_DEVICES=1 ~/.conda/envs/holobrain/bin/python \
      /tmp/ae_hostdrv/attn_calib.py \
      --samples /tmp/ae_hostdrv/sample_000.npz /tmp/ae_hostdrv/sample_001.npz \
      --out /tmp/ae_hostdrv/attn_calib.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HB = "~/workspace/holobrain"
sys.path.insert(0, os.path.join(HB, "shims"))
sys.path.insert(0, os.path.join(HB, "robo_orchard_lab"))
sys.path.insert(0, HB)
sys.path.insert(0, os.path.join(HB, "quant"))
sys.path.insert(0, os.path.join(HB, "hw_calib"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import bringup  # noqa: E402
from gate import load_everything, force_mha_slow_path  # noqa: E402
from hw_calib import perturbed_batch  # noqa: E402  (hw_calib/hw_calib.py)
from robo_orchard_lab.models.holobrain.processor import (  # noqa: E402
    HoloBrainProcessor,
    MultiArmManipulationInput,
)

FAMILY = {
    "RotaryAttention": "rotary",
    "JointGraphAttention": "jg",
    "TemporalJointGraphAttention": "temporal",
    "MultiheadAttention": "mha",
    "BiMultiHeadAttention": "bimha",
    "WindowMSA": "swin",
    "ShiftWindowMSA": "swin",
    "BertAttention": "bert",
}


def A(i, name, args, kwargs, default=None):
    if len(args) > i:
        return args[i]
    return kwargs.get(name, default)


def plain_qk(mod, fam, args, kwargs):
    """返回 host 会量化给 QK^T GEMM 的 fp 张量积（未乘 scale）。"""
    with torch.no_grad():
        if fam == "rotary":
            query = A(0, "query", args, kwargs)
            key = A(1, "key", args, kwargs)
            qpos = A(3, "query_pos", args, kwargs)
            kpos = A(4, "key_pos", args, kwargs)
            B, N, _ = query.shape
            H = mod.num_heads
            q = mod.q_proj(query).reshape(B, N, H, -1).permute(0, 2, 1, 3)
            k = mod.k_proj(key).reshape(B, key.shape[1], H, -1).permute(
                0, 2, 1, 3)
            q, k = mod.apply_position_encode(q, qpos, k, kpos)
            return q @ k.transpose(-1, -2)
        if fam == "jg":
            query = A(0, "query", args, kwargs)
            key = A(1, "key", args, kwargs)
            B, N, _ = query.shape
            H = mod.num_heads
            q = mod.q_proj(query).reshape(B, N, H, -1).permute(0, 2, 1, 3)
            k = mod.k_proj(key).reshape(B, key.shape[1], H, -1).permute(
                0, 2, 1, 3)
            return q @ k.transpose(-1, -2)          # 不含 query_pos 调制
        if fam == "temporal":
            query = A(0, "query", args, kwargs)
            key = A(1, "key", args, kwargs)
            tpq = kwargs.get("temporal_pos_q")
            tpk = kwargs.get("temporal_pos_k")
            B, N, Tq, C = query.shape
            M, Tk = key.shape[1:3]
            H = mod.num_heads
            q = mod.q_proj(query).reshape(B, N, Tq, H, -1).permute(
                0, 3, 1, 2, 4)
            k = mod.k_proj(key).reshape(B, M, Tk, H, -1).permute(
                0, 3, 1, 2, 4)
            q = mod.temporal_position_encoder(q, tpq)
            k = mod.temporal_position_encoder(k, tpk)
            return torch.einsum("bhnqc,bhmkc->bhnqmk", q, k)  # 无 jdist
        if fam == "mha":
            query = A(0, "query", args, kwargs)
            key = A(1, "key", args, kwargs)
            E, H = mod.embed_dim, mod.num_heads
            w, b = mod.in_proj_weight, mod.in_proj_bias
            q = F.linear(query, w[:E], b[:E] if b is not None else None)
            k = F.linear(key, w[E:2 * E],
                         b[E:2 * E] if b is not None else None)
            N, B_, _ = q.shape
            hd = E // H
            q = q.reshape(N, B_ * H, hd).transpose(0, 1)
            k = k.reshape(N, B_ * H, hd).transpose(0, 1)
            return q @ k.transpose(-1, -2)
        if fam == "bimha":
            vision = A(0, "vision", args, kwargs)
            lang = A(1, "lang", args, kwargs)
            bsz = vision.shape[0]
            q = mod._shape(mod.v_proj(vision), -1, bsz)
            k = mod._shape(mod.l_proj(lang), -1, bsz)
            return q @ k.transpose(-1, -2)
        if fam == "swin":
            x = A(0, "x", args, kwargs)
            B, N, C = x.shape
            H = mod.num_heads
            qkv = mod.qkv(x).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
            return qkv[0] @ qkv[1].transpose(-1, -2)
        if fam == "bert":
            h = A(0, "hidden_states", args, kwargs)
            B, T, C = h.shape
            H = mod.self.num_attention_heads
            q = mod.self.query(h).view(B, T, H, -1).permute(0, 2, 1, 3)
            k = mod.self.key(h).view(B, T, H, -1).permute(0, 2, 1, 3)
            return q @ k.transpose(-1, -2)
    raise KeyError(fam)


def build_inp(s, flip_rgb):
    imgs, deps = s["imgs"], s["depths"]
    camn = [str(c) for c in s["cam_names"]]
    if flip_rgb:
        images = {c: [np.ascontiguousarray(imgs[i][..., ::-1])]
                  for i, c in enumerate(camn)}
    else:
        images = {c: [imgs[i]] for i, c in enumerate(camn)}
    depth = {c: [deps[i].astype(np.float32)] for i, c in enumerate(camn)}
    return MultiArmManipulationInput(
        image=images, depth=depth,
        intrinsic={c: s["intrinsic_4x4"][i].astype(np.float64)
                   for i, c in enumerate(camn)},
        t_world2cam={c: s["t_world2cam_4x4"][i].astype(np.float64)
                     for i, c in enumerate(camn)},
        t_robot2world=s["t_base2world"].astype(np.float64), t_robot2ego=None,
        history_joint_state=[s["joint_state_14"].astype(np.float64)],
        history_ee_pose=None, instruction=str(s["instruction"]), urdf=None,
        remaining_actions=None, delay_horizon=None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="*", default=[])
    ap.add_argument("--n-bringup", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    processor, model, _ = load_everything()   # (processor, model, batch)
    model = model.cuda().float().eval()
    n_mha = force_mha_slow_path(model)
    print(f"[load] model ready, mha_eager={n_mha}", flush=True)

    stats = {}

    def mk_hook(name, fam):
        def hook(mod, args, kwargs):
            try:
                S = plain_qk(mod, fam, args, kwargs)
                mx = float(S.detach().abs().max())
            except Exception as e:  # noqa: BLE001
                mx = None
                err = repr(e)
                st = stats.setdefault(name, {"calls": 0, "errs": []})
                st["errs"].append(err)
                st["calls"] += 1
                return
            st = stats.setdefault(
                name, {"family": fam, "absmax": 0.0, "calls": 0, "errs": []})
            st["family"] = fam
            st["absmax"] = max(st["absmax"], mx)
            st["calls"] += 1
        return hook

    n_hook = 0
    for name, mod in model.named_modules():
        fam = FAMILY.get(type(mod).__name__)
        if fam is None:
            continue
        mod.register_forward_pre_hook(mk_hook(name, fam), with_kwargs=True)
        n_hook += 1
    print(f"[hook] {n_hook} attention modules", flush=True)

    def run_batch(batch, tag):
        batch = {k: (v.cuda() if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        torch.manual_seed(4000)      # 噪声与量程无关，seed 不影响 absmax
        with torch.no_grad():
            model(batch)
        n = sum(1 for st in stats.values() if st.get("calls"))
        print(f"[run ] {tag}: hooks fired on {n} modules", flush=True)

    for i in range(a.n_bringup):
        run_batch(perturbed_batch(processor, i), f"bringup{i}")
    for path in a.samples:
        s = dict(np.load(path))
        batch = None
        for flip in (True, False):
            with torch.no_grad():
                b = processor.pre_process(build_inp(s, flip))
            ib = b["imgs"][0].numpy()
            if np.array_equal(np.clip(np.round(ib), 0, 255).astype(np.uint8),
                              s["imgs"]):
                batch = b
                break
        assert batch is not None, f"{path}: flip 两种都对不上 npz imgs"
        run_batch(batch, os.path.basename(path))

    out = {"_meta": {
        "n_bringup": a.n_bringup, "samples": a.samples,
        "note": "absmax of the fp tensor exactly as host quantizes it into "
                "the PL QK^T GEMM (post-rotary, no attn scale, JG without "
                "query_pos, temporal without joint_distance, MHA with "
                "in_proj bias). sigma_S = absmax/127.",
    }}
    n_err = 0
    for name in sorted(stats):
        st = stats[name]
        if st.get("errs"):
            n_err += 1
            print(f"[warn] {name}: {st['errs'][0]}")
        out[name] = {
            "family": st.get("family"),
            "s_absmax": st.get("absmax"),
            "calls": st["calls"],
        }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    fams = {}
    for name, st in out.items():
        if name.startswith("_"):
            continue
        fams[st["family"]] = fams.get(st["family"], 0) + 1
    print(f"[save] {a.out}: {len(fams)} families {fams}, "
          f"{sum(st.get('calls', 0) for st in stats.values())} calls, "
          f"{n_err} modules with errors", flush=True)


if __name__ == "__main__":
    main()
