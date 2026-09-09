# mk_pcw_calib.py -- 合成 pcW+RTN 校准表（本地/服务器均可，numpy+json）
#
# 输入：
#   hw_calib_table_v2.json   02_quant 部署校准表（sa/so/m_requant/s_shift/
#                            bias_aug_c/w_bias_int8 ...，逐 GEMM 一条）
#   pcw_scales.json          pcw_export.py 的逐通道 swc（只有 linear/
#                            mha_in_proj 权重）
#   fp_biases.json           dump_bias.py 的逐层 fp 偏置（--biases，可选：
#                            给了才做逐通道 K+1 增广，否则 bias 走 host fp）
# 输出：
#   hw_calib_table_pcw.json  在 v2 条目上做三件事：
#     1) linear/mha_in_proj 且非豁免：加 "pcw": true, "swc": [...],
#        "rq_ms_col": [[m,s]]*n —— 每输出列一组 requant 系数，
#        r_j = (sa * swc_j) / so，编码 r = m*2^-s 与 v1_encode 同款
#        （s=floor(log2(32767/r))，m=round(r*2^s) 截到 [1,32767]），
#        但 s 下限抬到 9：RTN 舍入常数 rn=2^(s-9)（RTL 内部 t=s-8 口径）
#        要求 s>=9。
#     2) 偏置：注意力内部的 qkv/投影层，其 int8 输出在 PL 段内被 QK^T/PV
#        直接消费，fp 偏置没法插在中间——必须走 K+1 增广（bias 进累加器，
#        部署数据流，零 RTL）。pcW 下增广单位逐通道化：
#            w_bias_j = round(b_j / (sa * swc_j * c))，c = 每层统一的
#            2 的幂（<=64，取到 max_j|...| <= 127 的最小者）
#        （v2 的逐 tensor 版是 round(b/(sa*sw*c))）。conv 与无偏置层不改。
#        例外：c=64 仍溢出的层保留 host fp 偏置（bias_fp_fallback，与
#        v2 同判据口径）。
#        【2026-09-04 修正】首版把所有层 bias 改 host fp 是错的：编译链里
#        qkv 的 int8 输出不回 host，fp 偏置加不进去，全部注意力层的
#        q/k/v 偏置被静默丢掉（jpos 0.30->0.33 的根因）。
#     3) conv（nn.Conv2d/nn.Conv1d）：保持 v2 逐 tensor 权重与 (m,s)
#        （权威 V2_rn_pcW 的 conv 就是逐 tensor + rn）与 v2 的增广偏置。
#    sa/so 完全不动（权威 V2c_pcW：激活静态逐 tensor）。
#
# 用法：
#   python mk_pcw_calib.py --v2 ../02_quant/hw_calib_table_v2.json \
#       --scales pcw_scales.json --biases fp_biases.json \
#       --out hw_calib_table_pcw.json
import argparse
import json
import math
import re

M_LIM = 32767
S_MIN = 9        # RTN 在 rq_v2 内部口径 rn=2^(s-9) 需要 s>=9
S_MAX = 47       # 与编译器 rq_max_s 默认一致
C_MAX = 64       # K+1 增广常数列上限（部署规格，2 的幂）

# 注意力内部生产层（qkv 类）：int8 输出被 QK^T/PV 在 PL 段内直接消费，
# host fp 偏置插不进去——这批 key 溢出也必须饱和增广（不能退 fp fallback）
PROD = re.compile(
    r'\.(qkv|query|key|value|q_proj|k_proj|v_proj|l_proj|values_v_proj|'
    r'values_l_proj)(\.weight)?$|in_proj_weight')


def bias_keys(key):
    """v2 表键 -> dump_bias 归一键的候选列表（去 .weight/_weight 后缀）。"""
    ks = [key, key + '.weight', key + '_weight']
    for suf in ('.weight', '_weight'):
        if key.endswith(suf):
            ks.append(key[:-len(suf)])
    return ks


def enc_col(r):
    """r -> (m, s)，s 夹到 [S_MIN, S_MAX]。返回 (m, s, clamped)。"""
    r = max(float(r), 1e-30)
    s = int(math.floor(math.log2(M_LIM / r)))
    s_nat = s
    s = max(S_MIN, min(S_MAX, s))
    m = int(round(r * (1 << s)))
    clamped = False
    if m > M_LIM:
        m, clamped = M_LIM, True
    if m < 1:
        m, clamped = 1, True
    return m, s, clamped or (s != s_nat)


def pcw_aug(sa, swc, b, saturate=False):
    """逐通道 K+1 增广：选 c（2 的幂，<=64）使 max_j |b_j/(sa·swc_j·c)|<=127。
    返回 (c, w_bias 列表) 或 None（c=64 仍溢出）。
    saturate=True（注意力内部生产层用）：溢出时 c=64 并对 w_bias 逐通道
    饱和到 ±127——近似偏置好过零偏置（这些层的 int8 输出被 QK^T/PV 在
    PL 段内直接消费，fp 偏置插不进去；v2 部署表对其中 32 层就是静默丢
    偏置的）。"""
    need = [abs(float(bj)) / (sa * float(sj))
            for bj, sj in zip(b, swc) if sa * float(sj) > 0]
    if not need:
        return None
    mx = max(need)
    if mx <= 127.0:
        c = 1.0
    else:
        c = 2.0 ** math.ceil(math.log2(mx / 127.0))
        if c > C_MAX:
            if not saturate:
                return None
            c = C_MAX
    wb = [max(-127, min(127, int(round(float(bj) / (sa * float(sj) * c)))))
          for bj, sj in zip(b, swc)]
    return c, wb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True)
    ap.add_argument("--scales", required=True)
    ap.add_argument("--biases", default=None,
                    help="dump_bias.py 产物；给了才做逐通道 K+1 增广，"
                         "否则 pcW 层 bias 走 host fp（错误口径，仅对照）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    v2 = json.load(open(a.v2, encoding="utf-8"))
    sc = json.load(open(a.scales, encoding="utf-8"))
    swc_all = sc["swc"]
    bias_all = json.load(open(a.biases, encoding="utf-8")) if a.biases else {}
    gemms = v2["gemms"]
    out = {"_meta": dict(v2.get("_meta", {})), "gemms": {}}
    st = {"n": 0, "n_pcw": 0, "n_conv_kept": 0, "n_aug_dropped": 0,
          "n_aug_pcw": 0, "n_fp_fallback_pcw": 0, "n_aug_sat": 0,
          "n_s_clamped": 0, "no_scale": [], "s_range": [999, 0],
          "m_range": [1 << 30, 0], "c_hist": {}}

    for key, e in gemms.items():
        e2 = dict(e)
        if e.get("exempt_fp") or "sa" not in e:
            out["gemms"][key] = e2
            continue
        st["n"] += 1
        # 1) 逐通道 requant（只有 linear/mha_in_proj）。pcw_scales.json 按
        #    manifest 键（带 .weight 后缀）存，v2 表键无后缀，两个都试
        swc = swc_all.get(key) or swc_all.get(key + '.weight')
        is_lin = e.get("type") == "nn.Linear" or key.endswith("in_proj_weight")
        if swc and is_lin:
            sa, so = float(e["sa"]), float(e["so"])
            ms = []
            for sj in swc:
                m, s, cl = enc_col(sa * float(sj) / so)
                ms.append([m, s])
                st["n_s_clamped"] += int(cl)
                st["s_range"][0] = min(st["s_range"][0], s)
                st["s_range"][1] = max(st["s_range"][1], s)
                st["m_range"][0] = min(st["m_range"][0], m)
                st["m_range"][1] = max(st["m_range"][1], m)
            e2["pcw"] = True
            e2["swc"] = swc
            e2["rq_ms_col"] = ms
            st["n_pcw"] += 1
            # 2) 偏置：优先逐通道 K+1 增广（部署数据流，qkv 等 PL 内消费
            #    层必须）；c=64 溢出才退 host fp。bias_all 的键是
            #    dump_bias 归一过的（去 .weight / _weight 后缀），这里
            #    同样把后缀变体都试一遍
            b = None
            for bk in bias_keys(key):
                if bias_all.get(bk):
                    b = bias_all[bk]
                    break
            if b and len(b) == len(swc):
                sat = bool(PROD.search(key))
                aug = pcw_aug(sa, swc, b, saturate=sat)
                if aug is not None:
                    c, wb = aug
                    e2["bias_aug_c"] = c
                    e2["w_bias_int8"] = wb
                    e2["bias_fp_fallback"] = False
                    st["n_aug_pcw"] += 1
                    if sat:
                        st["n_aug_sat"] += 1
                    ck = str(int(c)) if c == int(c) else str(c)
                    st["c_hist"][ck] = st["c_hist"].get(ck, 0) + 1
                else:
                    e2["bias_aug_c"] = None
                    e2["w_bias_int8"] = None
                    e2["bias_fp_fallback"] = True
                    st["n_fp_fallback_pcw"] += 1
            else:
                if e2.get("bias_aug_c") is not None:
                    st["n_aug_dropped"] += 1
                e2["bias_aug_c"] = None
                e2["w_bias_int8"] = None
                e2["bias_fp_fallback"] = True
        else:
            if is_lin and not swc:
                st["no_scale"].append(key)
            st["n_conv_kept"] += 1
        out["gemms"][key] = e2

    out["_meta"]["pcw_note"] = (
        "pcW+RTN table derived from v2: linear/mha_in_proj weights "
        "per-output-channel (rq_ms_col per column, r_j=(sa*swc_j)/so, "
        "s in [9,47]); bias via per-channel K+1 augmentation "
        "(w_bias_j=round(b_j/(sa*swc_j*c)), c in {1..64} power of two; "
        "fp_host fallback when c=64 overflows) — deployed dataflow, PL-"
        "internal consumers (attention qkv) require in-accumulator bias; "
        "conv weights stay per-tensor (authority V2_rn_pcW conv treatment); "
        "sa/so unchanged from v2")
    out["_meta"]["pcw_stats"] = st
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[mk ] quantized={st['n']} pcw={st['n_pcw']} "
          f"conv_kept={st['n_conv_kept']} aug_pcw={st['n_aug_pcw']} "
          f"fp_fb_pcw={st['n_fp_fallback_pcw']} aug_sat={st['n_aug_sat']} "
          f"aug_dropped={st['n_aug_dropped']} c_hist={st['c_hist']} "
          f"s_clamped={st['n_s_clamped']} s_range={st['s_range']} "
          f"m_range={st['m_range']} no_scale={st['no_scale']}")


if __name__ == "__main__":
    main()
