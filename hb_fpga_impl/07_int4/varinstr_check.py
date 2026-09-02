# varinstr_check.py -- robustness of the chosen W4 mix on instruction
# variants. BERT (half the weight bytes) goes W4 in the final config, but the
# eval set has only 3 instructions; this feeds sample_000's OBSERVATION with
# several of the episode's 99 instruction variants and re-measures jpos.
import json
import os
import sys

sys.path.insert(0, "/tmp/int4_hb")
import int4_scan as IS  # reuses TAB/CFG/patches/loaders (no side effects)

import h5py
import numpy as np
import torch
from robo_orchard_lab.models.holobrain.processor import (
    MultiArmManipulationInput,
)
import extract_sample as XS
from gate import act_metrics, forward_actions


def variant_batch(processor, k, vidx):
    fname, t, _imode, _ = XS.SAMPLES[k]
    h = h5py.File(os.path.join(XS.DATA_DIR, fname), "r")
    v = h["instructions"][vidx]
    text = v.decode() if isinstance(v, bytes) else str(v)
    images, depths, intr, w2c, joint, z = XS.build_sample(h, t, text)
    inp = MultiArmManipulationInput(
        image=images, depth=depths, intrinsic=intr, t_world2cam=w2c,
        t_robot2world=XS.T_BASE2WORLD.copy(), t_robot2ego=None,
        history_joint_state=[joint.copy()], history_ee_pose=None,
        instruction=text, urdf=None, remaining_actions=None,
        delay_horizon=None)
    batch = processor.pre_process(inp)
    h.close()
    return batch, text


def main(cfg_path, tag):
    torch.backends.cudnn.deterministic = True
    processor, model, _ = IS.load_model()
    B, R = IS.batches_and_refs(processor, model)  # refs BEFORE patch
    IS.patch_all(model)
    cfg = json.load(open(cfg_path))

    VARIANTS = [(0, 0), (0, 10), (0, 25), (0, 50), (1, 5), (1, 40), (2, 15)]
    rows = []
    for k, vidx in VARIANTS:
        batch, text = variant_batch(processor, k, vidx)
        # TRUE fp reference: every quantizable module set to "fp"
        # (patched forward with mode 'fp' == the original module forward)
        for key in IS.TAB:
            IS.CFG[key] = "fp"
        torch.manual_seed(20260830)
        with torch.no_grad():
            fp = model(batch)[0]["pred_actions"].detach().cpu()
        # W8 baseline (empty CFG -> defaults)
        IS.CFG.clear()
        w8 = forward_actions(model, batch, 20260830)
        # config under test
        for kk_, vv_ in cfg.items():
            IS.CFG[kk_] = vv_
        q = forward_actions(model, batch, 20260830)
        IS.CFG.clear()
        m_w8 = act_metrics(fp, w8)
        m_q = act_metrics(fp, q)
        rows.append({"sample": k, "vidx": vidx, "instr": text[:60],
                     "w8_jpos": m_w8["mae_jointpos"],
                     "w8_mae_all": m_w8["mae_all"],
                     "cfg_jpos": m_q["mae_jointpos"],
                     "cfg_mae_all": m_q["mae_all"]})
        print(f"[{tag}] s{k} v{vidx}: W8 {m_w8['mae_jointpos']:.5f} | "
              f"cfg {m_q['mae_jointpos']:.5f} (mae_all "
              f"{m_q['mae_all']:.5f}) | {text[:48]!r}", flush=True)
    with open(os.path.join(IS.WORK, f"varinstr_{tag}.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"[done] varinstr_{tag}.json", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
