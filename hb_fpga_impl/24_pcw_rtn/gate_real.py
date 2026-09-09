# gate_real.py -- 真实样本版硬件语义门禁（pcW+RTN 轮，阶段 4）。
#
# 背景：合成 bringup 批门禁绿（0.02881）但真实样本部署全深度 0.2993 rad 红——
# 合成批的动作输出恰好容忍了权重量化误差，门禁对真实部署没有预测力
# （research_w8a8_error/REPORT.md §2.2）。本脚本把评估集换成真实 RoboTwin
# 样本 s000/s001，其余协议（标定、量化语义、指标）与 hw_calib.py mode B_v1
# 完全一致：直接 import hw_calib 复用 calibrate/build_hw_params/patch_hw，
# 不复制粘贴数值代码，保证语义不会漂移。
#
# 与 hw_calib 的差异：
#   eval 集：batch_s000.pt / batch_s001.pt（真实样本，处理器已 pre_process）
#   fp 参考：同批同种子重算（forward_actions，与 hw_calib 同款确定性协议）
#   门限：主判据用部署判据 0.045 rad（红/绿）；旧门限 0.030/0.060 一并列出
#   双变体归因：acc_bias（bias 进累加器，理想下界）vs host_bias（compiler
#   部署约定，判据口径）——差距即 bias 放置的代价
#   附加：真实批上的逐层误差排名（diff_pass）与标定量程覆盖（coverage_pass）
#
# Run (server): cd ~/workspace/holobrain && CUDA_VISIBLE_DEVICES=<gpu> \
#   ~/.conda/envs/holobrain/bin/python /tmp/pcw_rtn/gate_real.py [--smoke]
# 结果：/tmp/pcw_rtn/gate_real_results.json
import argparse
import json
import os
import sys
import time

HB = os.path.expanduser("~/workspace/holobrain")
sys.path.insert(0, HB)
sys.path.insert(0, os.path.join(HB, "quant"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gate import (  # noqa: E402
    act_metrics,
    force_mha_slow_path,
    forward_actions,
    mean_of_dicts,
)
import bringup  # noqa: E402
from robo_orchard_lab.models.holobrain.processor import (  # noqa: E402
    HoloBrainProcessor,
)
from robo_orchard_lab.models.mixin import ModelMixin  # noqa: E402
import hw_calib  # noqa: E402  (calibrate / build_hw_params / patch_hw / diff_pass / coverage_pass)

# hw_calib.HB 是未展开的 "~/workspace/holobrain" 字面量，build_hw_params 里
# open(os.path.join(HB, ...)) 会炸；换成展开后的绝对路径。
hw_calib.HB = HB

BATCH_DIR = "/tmp/ae_hostdrv"
OUT_DIR = "/tmp/pcw_rtn"
SAMPLES = ["s000", "s001"]
GATE_DEPLOY = 0.045          # 部署判据（REPORT §1：jpos MAE，rad）
GATE_GREEN_OLD = 0.030       # 合成批门禁旧门限（对照用）
GATE_YELLOW_OLD = 0.060
V2_TABLE = os.path.join(BATCH_DIR, "hw_calib_table_v2.json")
FIXTURE_SEED = 20260830      # fixture.meta.seed（fp32_ref 存档时的种子）


def params_from_table(model, table_path):
    """从部署版 v2 标定表重建 patch_hw 需要的 params（Wq64/w_acc 从模型权重
    + 表内 sw/bias 重建，确定性）。表里按参数名登记的条目（如 MHA 的
    in_proj_weight）不是 nn.Module，patch_hw 打不上——跳过并计数（诚实口径：
    这些层在本门里保持 fp，与 ab 实验存在覆盖差）。"""
    t = json.load(open(table_path))
    gemms = t.get("gemms") or next(v for k, v in t.items() if k != "_meta")
    mods = dict(model.named_modules())
    import torch.nn as nn
    params, skipped = {}, []
    for name, ent in gemms.items():
        mod = mods.get(name)
        # patch_hw 只支持 Linear/Conv2d：ConvTranspose（decoder.head 动作头）
        # 打不上，保持 fp 并计数；exempt_fp 条目（pts_prob_fc.layers.1）与
        # 非 module 条目（in_proj_weight/text_feat_map）同样保持 fp。
        if not isinstance(mod, (nn.Linear, nn.Conv2d)) or "sw" not in ent:
            skipped.append(name)
            continue
        W = mod.weight.detach().float()
        sw = ent["sw"]
        Wq = torch.clamp(torch.round(W / sw), -127.0, 127.0).double()
        w_acc = None
        if ent.get("bias_fp_fallback") and getattr(mod, "bias", None) is not None:
            # fp 偏置回退：精确 bias 进累加器（与 build_hw_params 的 w_acc=b_acc
            # 同款；此前版本误置 None = 整层丢 bias，0.24 的假红来自这里）
            w_acc = mod.bias.detach().float().double()
        elif ent.get("w_bias_int8") is not None \
                and ent.get("bias_aug_c") is not None:
            w_acc = torch.tensor(ent["w_bias_int8"], dtype=torch.float64) \
                * ent["bias_aug_c"]
        params[name] = {
            "sa": ent["sa"], "so": ent["so"], "sw": sw,
            "m_s8": ent.get("m_s8_q8_8", 0),
            "m_v1": ent.get("m_requant", 1), "s_v1": ent.get("s_shift", 8),
            "Wq64": Wq, "w_acc": w_acc,
        }
    return params, skipped


def _hb_linear_fwd(p, m_rq, s_sh):
    """host-bias 约定版 _hw_linear_fwd（与 hw_calib._hw_linear_fwd 同款，仅
    bias 放置不同）：p 带 "hb_bias" 的层 bias 不进累加器，改为输出侧 fp
    相加——即 03_compiler/compiler.py:31 的部署约定"PL 出 requant int8，
    host 反量化加 fp bias"。aug 层仍走 w_acc（K+1 增广，可进整数累加器）。"""
    sa, so = p["sa"], p["so"]
    Wq64 = p["Wq64"]
    hb = p.get("hb_bias")
    w_acc = None if hb is not None else p["w_acc"]
    denom = float(1 << s_sh)

    def forward(self, x):
        out_dtype = x.dtype
        with torch.autocast(device_type="cpu", enabled=False):
            xq = torch.clamp(torch.round(x.float() / sa),
                             -127.0, 127.0).double()
            acc = F.linear(xq, Wq64)
            if w_acc is not None:
                acc = acc + w_acc
            yq = torch.clamp(torch.floor(acc * m_rq / denom), -128.0, 127.0)
            y = yq.float() * so
            if hb is not None:
                y = y + hb.float()
        return y.to(out_dtype)

    return forward


def _hb_conv_fwd(p, m_rq, s_sh):
    sa, so = p["sa"], p["so"]
    Wq64 = p["Wq64"]
    hb = p.get("hb_bias")
    w_acc = None if hb is not None else p["w_acc"]
    denom = float(1 << s_sh)

    def forward(self, x):
        out_dtype = x.dtype
        with torch.autocast(device_type="cpu", enabled=False):
            xq = torch.clamp(torch.round(x.float() / sa),
                             -127.0, 127.0).double()
            acc = F.conv2d(xq, Wq64, None, self.stride, self.padding,
                           self.dilation, self.groups)
            if w_acc is not None:
                acc = acc + w_acc.reshape(-1, 1, 1)
            yq = torch.clamp(torch.floor(acc * m_rq / denom), -128.0, 127.0)
            y = yq.float() * so
            if hb is not None:
                y = y + hb.reshape(-1, 1, 1).float()
        return y.to(out_dtype)

    return forward


def _fix_kinematics_device(obj, seen=None):
    """batch 在 GPU 环境生成：map_location 只搬张量，Transform3d 的
    .device 属性仍是 'cuda'，chain 会误触发跨设备 clone（无 GPU 时崩）。
    递归找出带 chain 的对象，把整条链的 device/dtype 属性改回 cpu。
    （移植自 host_driver.py:94，逐字同款。）"""
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, dict):
        for v in obj.values():
            _fix_kinematics_device(v, seen)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _fix_kinematics_device(v, seen)
    else:
        ch = getattr(obj, 'chain', None)
        if ch is not None and hasattr(ch, '_root'):
            stack = [ch._root]
            n = 0
            while stack:
                fr = stack.pop()
                jt = getattr(fr, 'joint', None)
                t = getattr(jt, 'offset', None) if jt is not None else None
                if t is not None and hasattr(t, 'device'):
                    t.device = torch.device('cpu')
                    t.dtype = torch.float32
                    n += 1
                lk = getattr(fr, 'link', None)
                lt = getattr(lk, 'offset', None) if lk is not None else None
                if lt is not None and hasattr(lt, 'device'):
                    lt.device = torch.device('cpu')
                    lt.dtype = torch.float32
                    n += 1
                stack.extend(getattr(fr, 'children', ()) or ())
            try:
                ch.device = torch.device('cpu')
                ch.dtype = torch.float32
                n += 1
            except AttributeError:
                pass
            if n:
                print(f'[fix] kinematics chain {n} 处 device→cpu', flush=True)


def main(smoke):
    samples = SAMPLES[:1] if smoke else SAMPLES
    seeds = [1000, FIXTURE_SEED] if smoke else [1000, 1100, 1200, FIXTURE_SEED]
    n_cal = 2 if smoke else 8
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.perf_counter()

    # 1. 模型：CPU 口径。不用 gate.load_everything（它 .cuda()）——action
    #    decoder 的 FK 链持有不随 .cuda() 移动的普通张量，真实样本会触发
    #    recompute 分支，GPU 上必炸 device mismatch。CPU 与 ab_quant/
    #    host_driver（0.18937 / 0.2993 基线）同口径。
    processor = HoloBrainProcessor.load(bringup.CKPT, bringup.PROCESSOR_JSON)
    model = ModelMixin.load_model(bringup.MODEL_DIR, load_impl="native")
    model = model.float().eval()
    n_mha = force_mha_slow_path(model)
    print(f"[load] model up on CPU (eager MHA x{n_mha})", flush=True)

    # 2. 真实样本批（host_driver 同款加载方式）
    batches = {}
    for s in samples:
        p = os.path.join(BATCH_DIR, f"batch_{s}.pt")
        batches[s] = torch.load(p, map_location="cpu", weights_only=False)
        _fix_kinematics_device(batches[s])
        print(f"[data] {s} <- {p}", flush=True)

    # 3. 标定：与 hw_calib 完全同款（合成扰动批，8 样本）——本轮刻意不动标定，
    #    只换评估集（REPORT §5 第 4 步的范围）
    calib = hw_calib.calibrate(model, processor, n_cal)
    params, table, st = hw_calib.build_hw_params(model, calib)
    print(f"[cal] n_linear={st['n_linear_quant']}/{st['n_linear']} "
          f"n_conv={st['n_conv_quant']}/{st['n_conv']} "
          f"bias_fp={len(st['bias_fp_layers'])} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)

    # 4. fp 参考（未打补丁，真实批，固定种子）
    fp = {s: {seed: forward_actions(model, batches[s], seed) for seed in seeds}
          for s in samples}
    print(f"[fp  ] {len(samples)}x{len(seeds)} references "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)

    # 5. 两个变体各评一遍（标定、层覆盖完全相同，只差 bias 放置）：
    #    acc_bias  = hw_calib 原语义：88 个 fallback 层的精确 bias 塞进
    #                累加器（fp64）。PL 整数累加器放不下这些层——这是
    #                "bias 零代价"的理想口径 = per-tensor W8A8 的误差下界。
    #    host_bias = 部署可实现约定（compiler.py:31）：aug 层 K+1 增广进
    #                累加器，fallback 层 PL 出 int8、host 反量化后加 fp
    #                bias。与 ab V0_deployed / fast_interp 部署同款 → 判据口径。
    #    归因：acc_bias 与 host_bias 的差距 = bias 放置的代价；
    #    host_bias 与 ab V0（0.189，另多量化 in_proj 等 9 条目）的差距
    #    = 门禁未覆盖条目的贡献。
    mods = dict(model.named_modules())
    fp_names = {x["name"] if isinstance(x, dict) else x
                for x in st["bias_fp_layers"]}
    params_hb = {}
    for n, p in params.items():
        q = dict(p)
        if n in fp_names and getattr(mods.get(n), "bias", None) is not None:
            q["w_acc"] = None
            q["hb_bias"] = mods[n].bias.detach().float().double()
        params_hb[n] = q
    variants = [("acc_bias", params, False), ("host_bias", params_hb, True)]
    # 对照诊断：部署 v2 表 vs 今日 fresh 标定的数值一致性（[tbl] 全 1.0 =
    # 表值本来就同源；此前"表失配是独立误差源"的判断系重建 bug 假象，撤回）
    params_v2, skipped_v2 = params_from_table(model, V2_TABLE)
    print(f"[v2  ] table params {len(params_v2)} layers, "
          f"{len(skipped_v2)} 条目非 module（保持 fp，如 in_proj_weight）",
          flush=True)
    per_sample = {}
    for vname, vparams, hb in variants:
        if hb:
            _orig = (hw_calib._hw_linear_fwd, hw_calib._hw_conv_fwd)
            hw_calib._hw_linear_fwd = _hb_linear_fwd
            hw_calib._hw_conv_fwd = _hb_conv_fwd
        n_patch, restore = hw_calib.patch_hw(model, vparams, enc="v1")
        if hb:
            hw_calib._hw_linear_fwd, hw_calib._hw_conv_fwd = _orig
        for s in samples:
            per_seed = []
            for seed in seeds:
                q = forward_actions(model, batches[s], seed)
                m = act_metrics(fp[s][seed], q)
                per_seed.append({"seed": seed, **m})
                print(f"[{vname}] {s} seed {seed}: "
                      f"jpos={m['mae_jointpos']:.5f} "
                      f"max_jpos={m['max_jointpos']:.4f} "
                      f"mae_all={m['mae_all']:.5f}", flush=True)
            mean_s = mean_of_dicts(per_seed)
            jm = mean_s["mae_jointpos"]
            per_sample.setdefault(s, {})[vname] = {
                "per_seed": per_seed,
                "mean": mean_s,
                "gate_deploy": "green" if jm <= GATE_DEPLOY else "red",
            }
            print(f"[{vname}] {s} MEAN jpos={jm:.5f} -> deploy "
                  f"{'GREEN' if jm <= GATE_DEPLOY else 'RED'} "
                  f"(判据 {GATE_DEPLOY}; 旧门限绿 {GATE_GREEN_OLD}/黄 "
                  f"{GATE_YELLOW_OLD})", flush=True)
        restore()

    # 6. 两表失配统计：v2 vs fresh 的 sa/so 比值分布（解释红绿翻转的机制）
    common = sorted(set(params) & set(params_v2))
    import statistics as _st
    ratios = {"sa": [], "so": [], "sw": [], "m_v1": []}
    per_layer_mismatch = []
    for n in common:
        for k in ("sa", "so", "sw", "m_v1"):
            a, b = params_v2[n][k], params[n][k]
            if a > 0 and b > 0:
                ratios[k].append(a / b)
    for k in ("sa", "so", "sw", "m_v1"):
        v = sorted(ratios[k])
        if v:
            print(f"[tbl ] v2/fresh {k}: n={len(v)} p10={v[len(v)//10]:.3f} "
                  f"p50={v[len(v)//2]:.3f} p90={v[3*len(v)//4]:.3f} "
                  f"max={v[-1]:.2f}", flush=True)
    per_layer_mismatch = sorted(
        ((abs(params_v2[n]["so"] / params[n]["so"]
              if params[n]["so"] else 1.0) if params[n]["so"] else 1.0, n)
         for n in common), reverse=True)[:10]

    # 7. 逐层误差排名（真实批；fp 输出缓存必须在未打补丁时跑——重打一遍）
    diff = {}
    for s in samples:
        cache = hw_calib.cache_fp_outputs(model, batches[s])
        rows = hw_calib.diff_pass(model, batches[s], params_v2, cache)
        diff[s] = rows[:15]
        del cache
        print(f"[diff] {s} top: " + "; ".join(
            f"{r['name'][-45:]}({r['rel_mae']:.2f})" for r in rows[:5]),
            flush=True)

    # 8. 标定量程覆盖：真实输入 vs 两张表的 sa（关键证据：v2 的超量程比例）
    cov = {}
    for s in samples:
        cov[s] = {
            "fresh": hw_calib.coverage_pass(model, batches[s], params, table),
            "v2": hw_calib.coverage_pass(
                model, batches[s], params_v2,
                {n: {"so": p["so"], "sa": p["sa"]} for n, p in params_v2.items()}),
        }
        for tag, c in cov[s].items():
            print(f"[cov ] {s} [{tag}] input>calib*1.05 on "
                  f"{c['n_input_exceeds_calib_gt5pct']}/"
                  f"{c['n_modules_checked']} modules; sat_frac>1% on "
                  f"{c['n_sat_frac_gt1pct']}", flush=True)

    results = {
        "meta": {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "protocol": "hw_calib mode B_v1, eval set swapped to real samples; "
                        "variants: acc_bias (in-accumulator fp bias, ideal "
                        "lower bound) vs host_bias (compiler.py deployed "
                        "convention = criterion semantics)",
            "calibration": f"unchanged synthetic perturbed flow (n_cal={n_cal})",
            "eval_samples": samples,
            "seeds": seeds,
            "gate_deploy_criterion": GATE_DEPLOY,
            "old_gate_thresholds": {"green": GATE_GREEN_OLD,
                                    "yellow": GATE_YELLOW_OLD},
            "baseline_refs": {
                "synthetic_gate_green_last_round": 0.02881,
                "pl_fulldepth_v0": {"s000": 0.2993, "s001": 0.2108},
                "fp_resample_floor": 0.0457,
            },
        },
        "per_sample": per_sample,
        "table_mismatch": {
            "v2_over_fresh_quartiles": {k: (sorted(v)[len(v)//10],
                                            sorted(v)[len(v)//2],
                                            sorted(v)[3*len(v)//4],
                                            sorted(v)[-1] if v else None)
                                        for k, v in ratios.items()},
            "worst_so_layers": per_layer_mismatch[:10],
        },
        "top_error_layers": diff,
        "coverage": cov,
        "total_seconds": time.perf_counter() - t0,
    }
    out = os.path.join(OUT_DIR, "gate_real_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"[save] {out} ({os.path.getsize(out)/1e3:.0f} kB)", flush=True)
    print(f"[done] total {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="s000 单样本单种子，n_cal=2，快速验证脚本本身")
    main(ap.parse_args().smoke)
