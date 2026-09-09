# -*- coding: utf-8 -*-
"""compare.py — RTL dump 与 golden 期望位精确比对"""
import sys, os

SIM = os.path.dirname(os.path.abspath(__file__))

def load(name):
    with open(os.path.join(SIM, name)) as f:
        return [int(l.strip(), 16) for l in f if l.strip()]

def cmp(dump, exp, label):
    a, b = load(dump), load(exp)
    if len(a) != len(b):
        print(f"[cmp] {label}: 长度不一致 dump={len(a)} exp={len(b)}")
        return False
    bad = [i for i in range(len(a)) if a[i] != b[i]]
    if bad:
        print(f"[cmp] {label}: FAIL — {len(bad)} 处不一致，前 8 处:")
        for i in bad[:8]:
            print(f"    idx={i} (bank={i//len(a)//1 if False else ''}line {i}): dump={a[i]:02X} exp={b[i]:02X}")
        return False
    print(f"[cmp] {label}: PASS ({len(a)} 字节一致)")
    return True

ok = True
ok &= cmp("dump_ctx_ref.mem",  "expected_ctx_ref.mem",  "CTX REF")
ok &= cmp("dump_ddr_ref.mem",  "expected_ddr_ref.mem",  "DDR REF")
ok &= cmp("dump_ctx_prim.mem", "expected_ctx_prim.mem", "CTX PRIM")
ok &= cmp("dump_ddr_prim.mem", "expected_ddr_prim.mem", "DDR PRIM")
print("[cmp] ==> " + ("全部位精确一致 ✔" if ok else "存在不一致 ✘"))
sys.exit(0 if ok else 1)
