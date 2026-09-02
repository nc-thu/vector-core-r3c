"""HoloBrain (HB-GD, ~0.2B VLA) V100 inference bringup.

Runs the full inference path of HorizonRobotics/HoloBrain_v0.0_GD
(post_training_robotwin checkpoint, RoboTwin 2.0 embodiment):

  1. deterministic synthetic scene (4 cams RGB + depth + dual-arm state)
  2. official HoloBrainProcessor pre-process (tokenizer + transforms + FK)
  3. full model forward incl. 10-step DPM-Solver denoising (fp32, V100)
  4. save fixture (inputs + outputs) and module inventory (Linear/Conv2d)

Run on server:
  cd /home/nc23/workspace/holobrain
  CUDA_VISIBLE_DEVICES=2 /home/nc23/.conda/envs/holobrain/bin/python bringup.py
"""

import json
import os
import sys
import time

SHIMS = "/home/nc23/workspace/holobrain/shims"
REPO = "/home/nc23/workspace/holobrain/robo_orchard_lab"
CKPT = "/home/nc23/workspace/holobrain/ckpt/HoloBrain_v0.0_GD"
OUT = "/home/nc23/workspace/holobrain/out"
MODEL_DIR = os.path.join(CKPT, "post_training_robotwin")
PROCESSOR_JSON = "robotwin2_0_processor.json"
SEED = 20260830
INSTRUCTION = "put the bowl on the plate"

sys.path.insert(0, SHIMS)  # pytorch3d shim (pure-torch rotation_conversions)
sys.path.insert(0, REPO)  # robo_orchard_lab package

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch


# --------------------------------------------------------------------------
# 1. deterministic synthetic scene (RoboTwin-style: 4 cams, dual arm, 14 dof)
# --------------------------------------------------------------------------
W0, H0 = 640, 480  # native camera resolution; processor resizes to 320x256
FX = FY = 617.0
CX, CY = 320.0, 240.0
CAM_NAMES = ["front_camera", "left_camera", "right_camera", "head_camera"]

# robot base -> world (matches AddItems T_base2world of robotwin2_0_processor)
T_BASE2WORLD = np.array(
    [[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, -0.65],
     [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)

K44 = np.eye(4, dtype=np.float64)
K44[0, 0], K44[1, 1], K44[0, 2], K44[1, 2] = FX, FY, CX, CY


def look_at_cv(eye, target):
    """T_world2cam, OpenCV convention (x right, y down, z forward)."""
    eye = np.asarray(eye, np.float64)
    target = np.asarray(target, np.float64)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 0)
    T[:3, 3] = -T[:3, :3] @ eye
    return T


TARGET3D = np.array([0.60, 0.0, 0.18])  # workspace point in front of robot
CAM_POSES = {
    "front_camera": look_at_cv([0.00, 0.00, 0.55], TARGET3D),
    "left_camera": look_at_cv([0.25, -0.45, 0.50], TARGET3D),
    "right_camera": look_at_cv([0.25, 0.45, 0.50], TARGET3D),
    "head_camera": look_at_cv([0.40, 0.10, 0.80], TARGET3D),
}


def make_rgb(cam_idx):
    """Structured deterministic RGB (uint8 HxWx3): gradient table + bowl/plate."""
    yy, xx = np.mgrid[0:H0, 0:W0].astype(np.float32)
    img = np.zeros((H0, W0, 3), np.float32)
    img[..., 0] = 40 + 0.10 * xx
    img[..., 1] = 60 + 0.08 * yy
    img[..., 2] = 90 + 0.05 * 0.5 * (xx + yy)
    img[yy > H0 / 2] = np.array([122, 96, 64], np.float32)  # table
    img[np.hypot(xx - 260, yy - 300) < 60] = np.array(
        [40, 60, 180], np.float32
    )  # bowl (blue)
    img[np.hypot(xx - 380, yy - 310) < 75] = np.array(
        [212, 205, 198], np.float32
    )  # plate (grey)
    img += 4 * cam_idx  # per-camera tint offset
    return np.clip(img, 0, 255).astype(np.uint8)


def make_depth(T_w2c):
    """Depth in meters (float32) from ray-plane intersection with world
    table plane z=0.18; bumps under bowl/plate. Quantized through uint16 mm
    to mimic a real RGB-D camera."""
    Kinv = np.linalg.inv(K44[:3, :3])
    yy, xx = np.mgrid[0:H0, 0:W0]
    pix = np.stack([xx, yy, np.ones_like(xx)], -1).astype(np.float64)
    dirs_cam = pix @ Kinv.T  # z component == 1
    dirs_w = dirs_cam @ T_w2c[:3, :3].T
    origin_w = T_w2c[:3, 3]
    s = (0.18 - origin_w[2]) / dirs_w[..., 2]  # plane z_world=0.18
    s = np.where(np.isfinite(s) & (s > 0.2) & (s < 1.15), s, 1.0)
    s = s.astype(np.float32)
    s[np.hypot(xx - 260, yy - 300) < 60] -= 0.08  # bowl stands taller
    s[np.hypot(xx - 380, yy - 310) < 75] -= 0.03  # plate
    s = np.clip(s, 0.2, 1.15)
    mm = np.round(s * 1000).astype(np.uint16)
    return (mm.astype(np.float32)) / 1000.0, mm


# 14-dof dual arm: [left arm 6, left gripper 1, right arm 6, right gripper 1]
LEFT_ARM = [0.10, -0.55, 0.95, 0.00, 0.75, 0.00]
RIGHT_ARM = [-0.10, -0.55, 0.95, 0.00, 0.75, 0.00]
JOINT_STATE = np.array(LEFT_ARM + [0.52] + RIGHT_ARM + [0.52], np.float64)


def build_raw_inputs():
    images, depths, depths_mm = {}, {}, {}
    for i, cam in enumerate(CAM_NAMES):
        images[cam] = [make_rgb(i)]
        d_m, d_mm = make_depth(CAM_POSES[cam])
        depths[cam] = [d_m]
        depths_mm[cam] = [d_mm]
    return images, depths, depths_mm


# --------------------------------------------------------------------------
# 2. load processor + model, run forward
# --------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    from robo_orchard_lab.models.holobrain.processor import (
        HoloBrainProcessor,
        MultiArmManipulationInput,
    )
    from robo_orchard_lab.models.mixin import ModelMixin

    images, depths, depths_mm = build_raw_inputs()

    processor = HoloBrainProcessor.load(CKPT, PROCESSOR_JSON)
    print(f"[load] processor ok ({PROCESSOR_JSON})")

    model = ModelMixin.load_model(MODEL_DIR, load_impl="native")
    model = model.cuda().float().eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[load] model ok on {next(model.parameters()).device}, "
          f"params={n_params/1e6:.1f}M")

    inp = MultiArmManipulationInput(
        image=images,
        depth=depths,
        intrinsic={c: K44.copy() for c in CAM_NAMES},
        t_world2cam={c: CAM_POSES[c].copy() for c in CAM_NAMES},
        t_robot2world=T_BASE2WORLD.copy(),
        t_robot2ego=None,
        history_joint_state=[JOINT_STATE.copy()],
        history_ee_pose=None,
        instruction=INSTRUCTION,
        urdf=None,
        remaining_actions=None,
        delay_horizon=None,
    )

    t0 = time.perf_counter()
    batch = processor.pre_process(inp)
    t_prep = time.perf_counter() - t0
    print(f"[prep] processor pre_process {t_prep*1000:.1f} ms")
    for k, v in batch.items():
        shape = tuple(v.shape) if torch.is_tensor(v) else type(v).__name__
        print(f"       {k:20s} {shape}")

    # ---- warmup + timed forwards ----
    with torch.no_grad():
        model(batch)  # warmup (compilations, cudnn autotune, allocator)
        torch.cuda.synchronize()

        times = []
        for _ in range(3):
            torch.manual_seed(SEED)  # denoise init noise uses global RNG
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model_outs = model(batch)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        torch.manual_seed(SEED)  # fixture run, reproducible
        t0 = time.perf_counter()
        model_outs = model(batch)
        torch.cuda.synchronize()
        t_fixture = time.perf_counter() - t0

    t_fwd_ms = 1000 * sum(times) / len(times)
    vram_alloc = torch.cuda.max_memory_allocated() / 2**30
    vram_resv = torch.cuda.max_memory_reserved() / 2**30
    print(f"[fwd] forward(10 denoise steps) avg {t_fwd_ms:.1f} ms "
          f"(min {1000*min(times):.1f}, fixture run {1000*t_fixture:.1f})")
    print(f"[fwd] VRAM peak: alloc {vram_alloc:.2f} GiB, "
          f"reserved {vram_resv:.2f} GiB")

    out = processor.post_process(model_outs, inp)
    action = out.action  # [num_pred_steps, num_joint]
    pa = model_outs[0]["pred_actions"]
    print(f"[out] action {tuple(action.shape)}, pred_actions "
          f"{tuple(pa.shape)}, dtype {action.dtype}")
    print(f"[out] first-step joints: {np.round(action[0].cpu().numpy(), 4)}")

    # ---- fixture ----
    batch_tensors = {
        k: v.detach().cpu() for k, v in batch.items() if torch.is_tensor(v)
    }
    fixture = {
        "meta": {
            "model_id": "HorizonRobotrics_placeholder",
            "checkpoint": "HorizonRobotics/HoloBrain_v0.0_GD@post_training_robotwin",
            "processor": PROCESSOR_JSON,
            "instruction": INSTRUCTION,
            "seed": SEED,
            "dtype": "float32",
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "num_inference_timesteps": 10,
            "forward_ms_avg": t_fwd_ms,
            "forward_ms_fixture": 1000 * t_fixture,
            "prep_ms": 1000 * t_prep,
            "vram_alloc_gib": vram_alloc,
            "vram_reserved_gib": vram_resv,
            "note": (
                "raw depth stored as uint16 mm (RGB-D convention); "
                "model input depth is meters float32. "
                "raw RGB is uint8 BGR-order=RGB-order neutral (gradients+shapes); "
                "processor ImageChannelFlip and model channel_flip cancel out."
            ),
        },
        "raw_input": {
            "cam_names": CAM_NAMES,
            "image_rgb_u8": {
                c: images[c][0] for c in CAM_NAMES
            },  # HxWx3 uint8
            "depth_u16_mm": {c: depths_mm[c][0] for c in CAM_NAMES},
            "depth_meters_f32": {c: depths[c][0] for c in CAM_NAMES},
            "intrinsic_4x4": np.stack([K44] * 4),
            "T_world2cam_4x4": np.stack([CAM_POSES[c] for c in CAM_NAMES]),
            "T_base2world_4x4": T_BASE2WORLD,
            "joint_state_14": JOINT_STATE,
            "scale_shift_14x2": batch_tensors["joint_scale_shift"].numpy(),
        },
        "model_input_batch": batch_tensors,  # exact tensors fed to model()
        "model_output": {
            "pred_actions_denorm": pa.detach().cpu(),  # incl. scale-shift inverse
            "action_final": action.cpu()
            if torch.is_tensor(action)
            else torch.as_tensor(action),
        },
    }
    # replace placeholder with real repo id
    fixture["meta"]["model_id"] = "HorizonRobotics/HoloBrain_v0.0_GD"
    fx_path = os.path.join(OUT, "fixture.pt")
    torch.save(fixture, fx_path)
    print(f"[save] fixture -> {fx_path} ({os.path.getsize(fx_path)/1e6:.1f} MB)")

    # ---- module inventory ----
    import torch.nn as tnn

    inv = {"nn.Linear": [], "nn.Conv2d": [], "summary": {}}
    n_lin = n_conv = 0
    for name, mod in model.named_modules():
        if isinstance(mod, tnn.Linear):
            kind, n_lin = "nn.Linear", n_lin + 1
        elif isinstance(mod, tnn.Conv2d):
            kind, n_conv = "nn.Conv2d", n_conv + 1
        else:
            continue
        inv[kind].append(
            {
                "name": name,
                "type": kind,
                "params": sum(p.numel() for p in mod.parameters()),
                "requires_grad": bool(
                    all(p.requires_grad for p in mod.parameters())
                ),
                "in_features": getattr(mod, "in_features", None)
                if kind == "nn.Linear"
                else None,
                "out_features": getattr(mod, "out_features", None)
                if kind == "nn.Linear"
                else None,
                "kernel_size": list(mod.kernel_size)
                if kind == "nn.Conv2d"
                else None,
                "in_channels": mod.in_channels
                if kind == "nn.Conv2d"
                else None,
                "out_channels": mod.out_channels
                if kind == "nn.Conv2d"
                else None,
            }
        )

    # per-subtree param totals + freeze status (frozen = no trainable params)
    subtree = {}
    frozen_roots = []
    for child_name, child in model.named_children():
        total = sum(p.numel() for p in child.parameters())
        trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
        subtree[child_name] = {
            "params": total,
            "trainable_params": trainable,
            "frozen": total > 0 and trainable == 0,
        }
        if total > 0 and trainable == 0:
            frozen_roots.append(child_name)

    inv["summary"] = {
        "checkpoint": "HorizonRobotics/HoloBrain_v0.0_GD@post_training_robotwin",
        "total_params": n_params,
        "trainable_params": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "n_linear": n_lin,
        "n_conv2d": n_conv,
        "linear_params": sum(m["params"] for m in inv["nn.Linear"]),
        "conv2d_params": sum(m["params"] for m in inv["nn.Conv2d"]),
        "subtree_params": subtree,
        "frozen_subtrees": frozen_roots,
        "text_encoder_is_BERT": "text_encoder" in subtree,
    }
    inv_path = os.path.join(OUT, "module_inventory.json")
    with open(inv_path, "w") as f:
        json.dump(inv, f, indent=1)
    print(f"[save] inventory -> {inv_path} "
          f"({n_lin} Linear, {n_conv} Conv2d, frozen={frozen_roots})")
    print("[done]")


if __name__ == "__main__":
    main()
