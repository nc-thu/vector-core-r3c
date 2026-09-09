"""trace_sample.py -- 在真实 RoboTwin 样本上重抓 HB-GD 算子 trace（服务器侧）。

背景：build_full 用的 ops_trace.json 是 bringup 合成 fixture 抓的（文本 8
token）；真实 sample_000 文本 21 token，形状对不上，必须在真实样本上重抓。

batch 重建走 04_dataset/extract_sample.py 的原路（同一 h5、同一 jpeg 解码、
同一 GL2CV 外参换算、同一平面深度合成、同一 pre_process），保证与 fp32_ref
是同一次前向。npz 只用来对拍（imgs 逐位、projection_mat 逐位——曾试过用
npz 里的 imgs/内参回喂，但 npz 图是处理器 resize 到 256 高之后的，回喂会
少一次 240→256 的内参缩放，projection 行 1 差 6.7%，所以必须回 h5）。

同一次运行把 model_input_batch 原样存盘（batch_s00K.pt），host 驱动直接加
载，避免在驱动里复刻 pre_process。

Run (server):
  cd ~/workspace/holobrain
  CUDA_VISIBLE_DEVICES=1 ~/.conda/envs/holobrain/bin/python \
      /tmp/ae_hostdrv/trace_sample.py --ep episode_0000000.hdf5 --t 40 \
      --instr scalar \
      --sample /tmp/ae_hostdrv/sample_000.npz \
      --ref /tmp/ae_hostdrv/fp32_ref_000.npz \
      --trace /tmp/ae_hostdrv/trace_s000.json \
      --batch /tmp/ae_hostdrv/batch_s000.pt
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import torch

HB = "~/workspace/holobrain"
DATA_DIR = os.path.join(HB, "robotwin_subset/place_empty_cup/aloha_agilex")
sys.path.insert(0, os.path.join(HB, "shims"))
sys.path.insert(0, os.path.join(HB, "robo_orchard_lab"))
sys.path.insert(0, HB)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import cv2  # noqa: E402
import h5py  # noqa: E402
import bringup  # noqa: E402
import trace_model as TM  # noqa: E402  (classify/make_hook 钩子机器)
from robo_orchard_lab.models.mixin import ModelMixin  # noqa: E402
from robo_orchard_lab.models.holobrain.processor import (  # noqa: E402
    HoloBrainProcessor,
    MultiArmManipulationInput,
)

SEED = 20260830  # extract_sample.py 同款（fp32_ref 的去噪初始噪声）

CAM_MAP = [
    ("cam_third_view", "front_camera"),
    ("cam_left_wrist", "left_camera"),
    ("cam_right_wrist", "right_camera"),
    ("cam_head", "head_camera"),
]
T_BASE2WORLD = np.array(
    [[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, -0.65],
     [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])


def jpeg_bytes(v):
    return v if isinstance(v, bytes) else bytes(v)


def make_plane_depth(K, T_w2c, z_plane, w=320, h=240):
    Kinv = np.linalg.inv(K[:3, :3])
    yy, xx = np.mgrid[0:h, 0:w]
    pix = np.stack([xx, yy, np.ones_like(xx)], -1).astype(np.float64)
    dirs_cam = pix @ Kinv.T
    dirs_w = dirs_cam @ T_w2c[:3, :3].T
    o_w = T_w2c[:3, 3]
    s = (z_plane - o_w[2]) / dirs_w[..., 2]
    bad = ~np.isfinite(s) | (s <= 0.2) | (s >= 2.5)
    s = np.where(bad, 1.0, s)
    s = np.clip(s, 0.2, 2.5)
    return np.round(s * 1000).astype(np.uint16).astype(np.float32) / 1000.0


def episode_table_height(h):
    lz = h["state/left_ee_poses"][:, 2]
    rz = h["state/right_ee_poses"][:, 2]
    z = min(lz.min(), rz.min()) - 0.03
    return float(np.clip(z, 0.60, 0.90))


def build_sample(h, t, instr_text):
    images, depths, intr, w2c = {}, {}, {}, {}
    z_plane = episode_table_height(h)
    for src, dst in CAM_MAP:
        g = h[f"vision/{src}"]
        img = cv2.imdecode(
            np.frombuffer(jpeg_bytes(g["colors"][t]), np.uint8),
            cv2.IMREAD_COLOR)
        K = np.array(g["intrinsic_matrix"][t], np.float64)
        gl = np.array(g["extrinsics_matrix"][t], np.float64)
        T_w2c = np.linalg.inv(gl @ GL2CV)
        images[dst] = [img]
        depths[dst] = [make_plane_depth(K, T_w2c, z_plane)]
        K44 = np.eye(4)
        K44[:3, :3] = K
        intr[dst] = K44
        w2c[dst] = T_w2c
    joint = np.array(h["state/joint_states"][t], np.float64)
    return images, depths, intr, w2c, joint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", required=True)
    ap.add_argument("--t", type=int, required=True)
    ap.add_argument("--instr", choices=["scalar", "variant"], default="scalar")
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--sample", default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--batch", required=True)
    a = ap.parse_args()

    h = h5py.File(os.path.join(DATA_DIR, a.ep), "r")
    if a.instr == "scalar":
        text = h["instruction"][()]
        text = text.decode() if isinstance(text, bytes) else str(text)
    else:
        v = h["instructions"][a.variant]
        text = v.decode() if isinstance(v, bytes) else str(v)

    processor = HoloBrainProcessor.load(bringup.CKPT, bringup.PROCESSOR_JSON)
    model = ModelMixin.load_model(bringup.MODEL_DIR, load_impl="native")
    model = model.cuda().float().eval()
    n_mha = TM.force_mha_slow_path(model)

    images, depths, intr, w2c, joint = build_sample(h, a.t, text)
    inp = MultiArmManipulationInput(
        image=images, depth=depths, intrinsic=intr, t_world2cam=w2c,
        t_robot2world=T_BASE2WORLD.copy(), t_robot2ego=None,
        history_joint_state=[joint.copy()], history_ee_pose=None,
        instruction=text, urdf=None, remaining_actions=None, delay_horizon=None,
    )
    with torch.no_grad():
        batch = processor.pre_process(inp)
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}

    # ---- 与 extract 存的 npz 对拍（证明就是 fp32_ref 那次的输入）----
    if a.sample:
        s = dict(np.load(a.sample))
        with torch.no_grad():
            imgs_b = np.clip(np.round(batch["imgs"][0].cpu().numpy()), 0, 255
                             ).astype(np.uint8)
        ok = np.array_equal(imgs_b, s["imgs"])
        print(f"[chk ] imgs bitwise vs npz: {ok}")
        assert ok, "imgs 对不上（jpeg 解码/pre_process 路径有出入）"
        for key, idx in (("depths", (0,)), ("hist_robot_state", (0, 0)),
                         ("projection_mat", (0,)), ("embodiedment_mat", (0,)),
                         ("joint_scale_shift", (0,))):
            v = batch[key]
            for i in idx:
                v = v[i]
            v = v.detach().cpu().numpy()
            sv = s[key].astype(np.float32)
            assert v.shape == sv.shape, (key, v.shape, sv.shape)
            d = float(np.abs(v.astype(np.float64)
                             - sv.astype(np.float64)).max())
            print(f"[chk ] {key}: max|d|={d:.3e}")
            assert d <= 1e-5, f"{key} 与 npz 不一致"

    # ---- 钩子 ----
    TM.RECORDS.clear()
    sd_keys = set(model.state_dict().keys())
    n_hook = 0
    for name, mod in model.named_modules():
        op = TM.classify(mod)
        if op is None:
            continue
        mod.register_forward_hook(TM.make_hook(name, type(mod).__name__, op,
                                               sd_keys), with_kwargs=True)
        n_hook += 1
    for name, child in model.named_children():
        if TM.classify(child) is None:
            child.register_forward_hook(
                TM.make_hook(name, type(child).__name__, "custom", sd_keys),
                with_kwargs=True)
            n_hook += 1
    print(f"[hook] {n_hook} modules hooked, mha_eager={n_mha}", flush=True)

    # ---- 前向（fp32_ref 同款 seed）----
    t0 = time.perf_counter()
    with torch.no_grad():
        torch.manual_seed(SEED)
        model_outs = model(batch)
        torch.cuda.synchronize()
    print(f"[fwd] {time.perf_counter()-t0:.1f}s, {len(TM.RECORDS)} records",
          flush=True)

    pa = model_outs[0]["pred_actions"].detach().cpu().numpy()
    action = processor.post_process(model_outs, batch).action.cpu().numpy()
    d1 = d2 = None
    if a.ref:
        ref = np.load(a.ref)
        d1 = float(np.abs(pa - ref["pred_actions_raw"]).max())
        d2 = float(np.abs(action - ref["action"]).max())
        print(f"[chk ] pred_actions_raw vs fp32_ref: max|d|={d1:.3e} "
              f"({'OK' if d1 < 1e-5 else 'MISMATCH'})")
        print(f"[chk ] action         vs fp32_ref: max|d|={d2:.3e} "
              f"({'OK' if d2 < 1e-5 else 'MISMATCH'})")
        assert d1 < 1e-5 and d2 < 1e-5, "与 fp32_ref 不一致，不能当 trace 基准"

    trace = {
        "meta": {
            "model": "HorizonRobotics/HoloBrain_v0.0_GD@post_training_robotwin",
            "input": f"{a.ep} t={a.t} instr={a.instr}"
                     f"{'' if a.instr == 'scalar' else f'#{a.variant}'} "
                     f"({text[:60]!r})",
            "seed": SEED, "dtype": "float32",
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "hook_note": "records are forward-hook firing order = post-order "
                         "(children before parents); id() links tensors for "
                         "dataflow edges",
            "pred_check": {"raw_max_diff": d1, "action_max_diff": d2},
        },
        "summary": {
            "total_records": len(TM.RECORDS),
            "op_counts": dict(collections.Counter(r["op"] for r in TM.RECORDS)),
        },
        "ops": TM.RECORDS,
    }
    with open(a.trace, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False)
    cpu_batch = {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
    torch.save(cpu_batch, a.batch)
    print(f"[save] {a.trace} ({len(TM.RECORDS)} recs), {a.batch} "
          f"({len(cpu_batch)} entries)", flush=True)


if __name__ == "__main__":
    main()
