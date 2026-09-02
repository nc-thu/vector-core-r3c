# -*- coding: utf-8 -*-
"""fetch_norm_weights.py — 从 HF 公开 checkpoint 按 HTTP Range 只拉 norm 层 γ/β

模型在服务器上（host_driver 要 torch 环境），但 checkpoint 是公开的
（HorizonRobotics/HoloBrain_v0.0_GD，apache-2.0）。safetensors 头部有每个
张量的字节区间，norm 的 γ/β 一共才 ~350KB，散在 724MB 文件里——按区间拉
103 个请求共 ~416KB，不必下整个模型。

产物：norm_weights.npz（module → weight/bias 的 fp32 数组），给真实张量
数值门用（host fp32 路径需要真 γ/β）。
"""
import io
import json
import os
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HEAD = os.path.join(HERE, "st_header.json")
OUT = os.path.join(HERE, "norm_weights.npz")
MIRROR = "https://hf-mirror.com/HorizonRobotics/HoloBrain_v0.0_GD/resolve/main/post_training_robotwin/model.safetensors"
MERGE = 16384


def cdn_url():
    """hf-mirror 的 resolve 会 302 到带签名的 CDN 地址（Range 只对 CDN 有效）。"""
    req = urllib.request.Request(MIRROR, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.url


def main():
    host = json.load(open(os.path.join(HERE, "..", "..", "09_cbound",
                                       "build_a2", "host_plan.json"),
                          encoding="utf-8"))
    mods = sorted(set(s["module"] for s in host["host_steps"]
                      if s.get("kind") == "norm" and s["cls"] != "GroupNorm"))
    raw = open(HEAD, encoding="utf-8").read()
    h = json.JSONDecoder().raw_decode(raw)[0]

    # safetensors 的 data_offsets 相对数据区起点（8B 长度 + 头部 JSON 之后），
    # 绝对文件偏移要加 BASE。第一版漏了 BASE，拉到的全是错位垃圾。
    BASE = 8 + len(raw.encode("utf-8"))
    rs = []
    for m in mods:
        for suf in (".weight", ".bias"):
            t = h.get(m + suf)
            if t and t["dtype"] == "F32":
                rs.append((BASE + t["data_offsets"][0],
                           BASE + t["data_offsets"][1], m + suf))
    rs.sort()
    print(f"[fetch] {len(mods)} 模块 / {len(rs)} 张量")

    url = cdn_url()
    merged = []
    for lo, hi, name in rs:
        if merged and lo - merged[-1][1] < MERGE:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    print(f"[fetch] 合并成 {len(merged)} 个区间，共 {sum(hi-lo for lo,hi in merged)} B")

    blob = {}
    for i, (lo, hi) in enumerate(merged):
        req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi-1}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            blob[lo] = r.read()
        if (i + 1) % 20 == 0:
            print(f"[fetch] {i+1}/{len(merged)}")

    out = {}
    for lo, hi, name in rs:
        blk = None
        for base in blob:
            if base <= lo and hi <= base + len(blob[base]):
                blk = blob[base]
                break
        assert blk is not None, f"{name} 区间没被任何下载覆盖"
        arr = np.frombuffer(blk[lo-base:hi-base], dtype="<f4")
        out[name] = arr.copy()
    np.savez(OUT, **out)
    n = sum(v.size for v in out.values())
    print(f"[fetch] 存 {OUT}：{len(out)} 张量 / {n} 个 fp32")


if __name__ == "__main__":
    sys.exit(main())
