# -*- coding: utf-8 -*-
"""cmp_vcd.py — 对比 iverilog(iv.vcd) 与 Verilator(vl.vcd) 的 dut 内部信号时间线。

用法：python cmp_vcd.py iv.vcd vl.vcd [关键字 ...]
不传关键字则用默认关注集（sched/dma 状态机 + 启动握手）。
输出：两边归一化时间线的 unified diff（t=<ns> <信号> = <值>），
分歧行会直接暴露 X vs 0 或时序错位。
"""
import sys

DEFAULT_KEYS = [
    'rst_n', 'start', 'dma_start',
    'u_sched.st', 'u_sched.pc', 'u_sched.running', 'u_sched.desc_r',
    'u_dma.st', 'u_dma.arvalid', 'u_dma.busy', 'u_dma.start',
]

def parse_vcd(path, keys):
    ids = {}          # id -> name
    names = set()
    changes = []      # (t, name, value)
    scope = []
    with open(path) as f:
        t = 0
        in_defs = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if in_defs:
                if line.startswith('$scope'):
                    scope.append(line.split()[2])
                elif line.startswith('$upscope'):
                    scope.pop()
                elif line.startswith('$var'):
                    parts = line.split()
                    # $var <type> <width> <id> <name> [$end]
                    vid, vname = parts[3], parts[4]
                    full = '.'.join(scope + [vname])
                    ids[vid] = full
                elif line.startswith('$enddefinitions'):
                    in_defs = False
                    names = {vid: n for vid, n in ids.items()
                             if any(n == k or n.endswith('.' + k) for k in keys)}
                continue
            if line[0] == '#':
                t = int(line[1:])
                continue
            if line[0] == '$':
                continue
            # 值变化：标量 "0<id>" / 向量 "b1010 <id>"
            if line[0] in '01xzXZ' and len(line) > 1:
                val, vid = line[0], line[1:]
            elif line[0] == 'b' or line[0] == 'B':
                val, vid = line.split()
                val = val[1:]
            else:
                continue
            if vid in names:
                changes.append((t, names[vid], val))
    return changes

def fmt(changes):
    out = []
    for t, n, v in changes:
        out.append(f"t={t/1000:g}ns {n} = {v}")
    return out

if __name__ == '__main__':
    a, b = sys.argv[1], sys.argv[2]
    keys = sys.argv[3:] or DEFAULT_KEYS
    ca, cb = parse_vcd(a, keys), parse_vcd(b, keys)
    la, lb = fmt(ca), fmt(cb)
    print(f"[a] {a}: {len(ca)} 条变化（关注 {len(set(1 for _ in ca))} 信号）")
    print(f"[b] {b}: {len(cb)} 条变化")
    import difflib
    diff = list(difflib.unified_diff(la, lb, fromfile='iverilog', tofile='verilator', lineterm=''))
    if not diff:
        print("完全一致")
    else:
        for d in diff[:200]:
            print(d)
