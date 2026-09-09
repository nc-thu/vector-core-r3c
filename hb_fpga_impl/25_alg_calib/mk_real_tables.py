# mk_real_tables.py -- 用真实样本统计重生成 v2' 校准表（25_alg_calib，T2/T3/T4）
#
# 输入：diag_real.py 的 stats（真实样本逐模块 in/out absmax + 分位数）
#       /tmp/ae_hostdrv/hw_calib_table_v2.json（部署 v2 表，sa/so 的合成口径）
# 输出：v2' 表 —— 选定 scope 内的模块 sa/so 换成真实值，并且**所有派生字段
#       用 hw_calib.build_hw_params 同款公式整体重算**：
#         r_star = sa*sw/so, m_requant/s_shift = v1_encode(r),
#         m_s8_q8_8 = s8_encode(r), acc_absmax_est,
#         偏置增广 w_bias = round(b/(sa*sw)/c)，c 从 (1,2,...,64) 重选，
#         放不下 -> bias_fp_fallback=True
# 然后 v2' 过 mk_pcw_calib.py（v3 流程）重生成 pcW 表：rq_ms_col 和逐通道
# 增广偏置都会从新 sa 整体重算 —— 不手改任何 bias 字段（bias 纪律）。
#
# scope：
#   all           全部有统计的模块（S1）
#   patch_embed   键名含 patch_embed（S2）
#   seg140_250    bisect 日志调用 #140~#250 覆盖的模块（S3；attn 条目按
#                 前缀展开到成员：in_proj_weight/qkv/proj/...）
# --mult k       段内 sa 乘 k（S4 扫描；1.0 = S3 本身）
# --mode absmax|p999   sa = absmax/127 或 p99.9/127（so 同口径取输出统计）
import argparse
import json
import math
import os
import sys
import time

C_CANDIDATES = (1, 2, 4, 8, 16, 32, 64)
M_LIM = 32767
BISECT_LOG = "/tmp/pcw_rtn/run000bisect815.log"


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


def s8_encode(r_star):
    m_raw = int(round(float(r_star) * 256))
    return max(0, min(M_LIM, m_raw))


def bias_keys(key):
    ks = [key, key + '.weight', key + '_weight']
    for suf in ('.weight', '_weight'):
        if key.endswith(suf):
            ks.append(key[:-len(suf)])
    return ks


def load_seg_scope(lo, hi):
    """bisect 日志 #lo..#hi 的模块名 -> 表键集合（含成员展开）。"""
    names = set()
    with open(BISECT_LOG, encoding="utf-8") as f:
        for line in f:
            if "[bisect] #" not in line:
                continue
            body = line.split("[bisect] #", 1)[1].strip()
            idx, _kind, mod = body.split(" ", 2)
            if lo <= int(idx) <= hi:
                names.add(mod.strip())
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True, help="diag_real_*.json")
    ap.add_argument("--v2", default="/tmp/ae_hostdrv/hw_calib_table_v2.json")
    ap.add_argument("--biases", default="/tmp/pcw_rtn/fp_biases.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["absmax", "p999"], default="absmax")
    ap.add_argument("--scope", choices=["all", "patch_embed", "seg140_250"],
                    default="all")
    ap.add_argument("--seg-lo", type=int, default=140)
    ap.add_argument("--seg-hi", type=int, default=250)
    ap.add_argument("--mult", type=float, default=1.0)
    a = ap.parse_args()

    diag = json.load(open(a.stats, encoding="utf-8"))["modules"]
    v2 = json.load(open(a.v2, encoding="utf-8"))
    bias_all = json.load(open(a.biases, encoding="utf-8")) \
        if os.path.exists(a.biases) else {}
    gemms = v2["gemms"]

    if a.scope == "all":
        scope = set(diag)
    elif a.scope == "patch_embed":
        scope = {k for k in diag if "patch_embed" in k}
    else:
        names = load_seg_scope(a.seg_lo, a.seg_hi)
        scope = {k for k in diag
                 if k in names or k.rsplit(".in_proj_weight", 1)[0] in names
                 or any(k.startswith(n + ".") for n in names)}

    out = {"_meta": dict(v2.get("_meta", {})), "gemms": {}}
    st = dict(n_changed=0, n_kept=0, n_no_stats=0, sa_ratio=[],
              so_ratio=[], n_bias_fp=0, n_c_hist={})
    for key, e in gemms.items():
        e2 = dict(e)
        if e.get("exempt_fp") or "sa" not in e or key not in scope \
                or key not in diag:
            if "sa" in e and not e.get("exempt_fp") and key not in scope:
                st["n_kept"] += 1
            elif "sa" in e and not e.get("exempt_fp"):
                st["n_no_stats"] += 1
            out["gemms"][key] = e2
            continue
        d = diag[key]
        if a.mode == "absmax":
            sa_new = d["in_absmax"] / 127.0
            so_new = d["out_absmax"] / 127.0
        else:
            sa_new = d.get("in_p999", d["in_absmax"]) / 127.0
            so_new = d.get("out_p999", d["out_absmax"]) / 127.0
        sa_new *= a.mult
        sa_new = max(sa_new, 1e-12)
        so_new = max(so_new, 1e-12)
        st["sa_ratio"].append(sa_new / float(e["sa"]))
        st["so_ratio"].append(so_new / float(e["so"]))
        e2["sa"], e2["so"] = sa_new, so_new

        sw = e.get("sw")
        if sw:
            r = sa_new * float(sw) / so_new
            e2["r_star"] = r
            m8 = s8_encode(r)
            e2["m_s8_q8_8"] = m8
            e2["m_s8_dead"] = bool(m8 == 0)
            mv, sv = v1_encode(r)
            e2["m_requant"], e2["s_shift"] = mv, sv
            e2["acc_absmax_est"] = 127.0 / r
        # 偏置增广：per-tensor 公式整体重算（pcW 层随后被 mk_pcw_calib 的
        # 逐通道版覆盖；conv 层保留这里的值 —— 与 v2 生成流程同款）
        b = None
        for bk in bias_keys(key):
            if bias_all.get(bk):
                b = bias_all[bk]
                break
        if b and sw:
            b_acc = [float(bj) / (sa_new * float(sw)) for bj in b]
            mx = max(abs(v) for v in b_acc)
            c_sel = None
            for c in C_CANDIDATES:
                if max(abs(round(v / c)) for v in b_acc) <= 127.0:
                    c_sel = c
                    break
            if c_sel is not None:
                e2["bias_aug_c"] = c_sel
                e2["w_bias_int8"] = [int(round(v / c_sel)) for v in b_acc]
                e2["bias_fp_fallback"] = False
                st["n_c_hist"][str(c_sel)] = \
                    st["n_c_hist"].get(str(c_sel), 0) + 1
            else:
                e2["bias_aug_c"] = None
                e2["w_bias_int8"] = None
                e2["bias_fp_fallback"] = True
                st["n_bias_fp"] += 1
            e2["b_acc_absmax"] = mx
        st["n_changed"] += 1
        out["gemms"][key] = e2

    out["_meta"]["real_calib_note"] = (
        f"25_alg_calib real-sample recalibration: stats={a.stats} "
        f"mode={a.mode} scope={a.scope}"
        f"{'(#%d-#%d)' % (a.seg_lo, a.seg_hi) if a.scope == 'seg140_250' else ''} "
        f"mult={a.mult}; sa/so from real batch, derived fields (r_star/"
        "m_requant/s_shift/m_s8/bias aug) fully recomputed per hw_calib "
        "build_hw_params formulas; weights sw/swc untouched")
    out["_meta"]["real_calib_stats"] = st
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    def q(v, p):
        v = sorted(v)
        return v[min(len(v) - 1, int(len(v) * p))] if v else None
    print(f"[mk  ] changed={st['n_changed']} kept={st['n_kept']} "
          f"no_stats={st['n_no_stats']} bias_fp={st['n_bias_fp']} "
          f"c_hist={st['n_c_hist']}")
    print(f"[mk  ] sa real/v2: p10={q(st['sa_ratio'],0.1):.3f} "
          f"p50={q(st['sa_ratio'],0.5):.3f} p90={q(st['sa_ratio'],0.9):.3f} "
          f"min={min(st['sa_ratio']) if st['sa_ratio'] else None:.3f} "
          f"max={max(st['sa_ratio']) if st['sa_ratio'] else None:.3f}")
    print(f"[mk  ] so real/v2: p10={q(st['so_ratio'],0.1):.3f} "
          f"p50={q(st['so_ratio'],0.5):.3f} p90={q(st['so_ratio'],0.9):.3f}")


if __name__ == "__main__":
    main()
