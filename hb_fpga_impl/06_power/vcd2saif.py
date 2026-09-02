#!/usr/bin/env python3
# vcd2saif.py — VCD -> SAIF 转换器（替代 Vivado 自带 vcd2saif：Windows 版 2021.2 不带这个工具）
#
# 用法：
#   python vcd2saif.py tb_ae.vcd activity.saif --re-root tb_ae.dut
#   --re-root <路径>   把 SAIF 树根移到指定实例（点分隔），丢掉 tb 层。read_saif 侧
#                      用 -instance_name 把它挂到 netlist 对应实例（本项目：tb 的
#                      dut=ae_core -> netlist ae_top/u_core）
#   --exclude <正则>   丢弃匹配的信号名（可选，如大存储阵列字）
#
# 产出口径：标量和总线逐位一条 NET，含 T0/T1/TX（处于 0/1/x 的累计时长）与
# TC（0<->1 翻转次数）；z 记为 x；$dumpoff 区间按 x 计。时间单位沿用 VCD timescale。
# 算法：值变化 O(1) 结算，收尾统一把所有位推进到 DURATION，单遍线性扫描。
import re
import sys
import argparse
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("vcd")
    p.add_argument("saif")
    p.add_argument("--re-root", dest="reroot", default=None)
    p.add_argument("--wrap", default=None,
                   help="把重定位后的树包进这个点分实例链（如 tb_ae.dut.u_core）。"
                        "Vivado read_saif 默认剥掉 SAIF 最外两层，所以内容要挂在"
                        "第 3 层、且名字等于网表里的实例路径首段（本项目 u_core）")
    p.add_argument("--exclude", default=None)
    p.add_argument("--t-start", dest="tstart", type=int, default=0,
                   help="窗口起始时刻（VCD 时间单位）：之前的值变化只用来定当前值，"
                        "不计时长/翻转，DURATION 也从这里起算。限窗 dump 必须给，"
                        "否则窗口前的未知时间会把翻转率稀释掉")
    return p.parse_args()


class Node:
    __slots__ = ("children", "nets")

    def __init__(self):
        self.children = {}   # scope 名 -> Node
        self.nets = {}       # 信号名 -> 位槽


def main():
    a = parse_args()
    excl = re.compile(a.exclude) if a.exclude else None

    root = Node()
    stack = [root]
    scope_names = [""]                    # 记路径用
    id_slots = {}                         # id -> [位槽...]（标量=1 元素）
    timescale = "1ps"
    duration = 0
    pending_vec = None                    # 跨行的 b<val> 的值缓存
    t0 = a.tstart                         # 统计窗口起点（VCD 时间单位）

    def new_slot():
        return {"t0": 0, "t1": 0, "tx": 0, "tc": 0, "st": "x", "tl": 0}

    def settle_all(t):
        for slots in id_slots.values():
            for s in slots:
                base = s["tl"] if s["tl"] > t0 else t0
                if t > base:
                    s["t" + s["st"]] += t - base
                s["tl"] = t

    def set_all_x():
        for slots in id_slots.values():
            for s in slots:
                s["st"] = "x"

    with open(a.vcd, "r", errors="replace") as f:
        in_defs = True
        for line in f:
            tok = line.strip()
            if not tok:
                continue
            if in_defs:
                if tok.startswith("$timescale"):
                    p = tok.split()
                    if len(p) >= 2:
                        timescale = p[1]
                elif tok.startswith("$scope"):
                    name = tok.split()[2]
                    if name.startswith("$") or stack[-1] is None:
                        # iverilog 内部生成块（$ivl_for_loop0 等），网表侧不存在，跳过
                        stack.append(None)
                        scope_names.append(name)
                        continue
                    node = stack[-1]
                    stack.append(node.children.setdefault(name, Node()))
                    scope_names.append(name)
                elif tok.startswith("$upscope"):
                    stack.pop()
                    scope_names.pop()
                elif tok.startswith("$var"):
                    if stack[-1] is None:
                        continue                     # 落在被跳过的 scope 里
                    # $var <type> <width> <id> <name> [区间] $end
                    # （iverilog 把区间写成独立字段：cp_rdata [127:0]）
                    p = tok.split()
                    width, sid, nm = int(p[2]), p[3], p[4]
                    suffix = ""
                    if len(p) >= 6 and re.match(r"^\[[0-9]+:[0-9]+\]$", p[5]):
                        suffix = p[5]
                    elif len(p) >= 6 and re.match(r"^\[[0-9]+\]$", p[5]):
                        nm = nm + p[5]               # 单个位选别名，按标量名处理
                    if sid in id_slots:
                        continue                     # 别名 id：忽略
                    node = stack[-1]
                    full = nm + suffix
                    # 名字自带区间（data[7:0] 或切片 desc[255:252]）：逐位展开成
                    # data[7]..data[0] / desc[255]..desc[252]，与网表逐位网名一致；
                    # 位序按 VCD 惯例 MSB 在前（slots[0] 对应值的最高位）
                    m = re.match(r"^(.*)\[(\d+):(\d+)\]$", full)
                    bitnames = None
                    if m and width == abs(int(m.group(2)) - int(m.group(3))) + 1:
                        base, hi, lo = m.group(1), int(m.group(2)), int(m.group(3))
                        rng = range(hi, lo - 1, -1) if hi >= lo else range(hi, lo + 1, 1)
                        bitnames = ["%s[%d]" % (base, i) for i in rng]
                    elif width > 1:
                        bitnames = ["%s[%d]" % (nm, i) for i in range(width - 1, -1, -1)]
                    slots = []
                    for i in range(width):
                        bitname = full if width == 1 else bitnames[i]
                        if excl and excl.search(bitname):
                            continue
                        if bitname in node.nets:     # 同名已登记：跳过该位
                            continue
                        s = new_slot()
                        node.nets[bitname] = s
                        slots.append(s)
                    id_slots[sid] = slots
                elif tok.startswith("$enddefinitions"):
                    in_defs = False
                continue

            c = tok[0]
            if c == "#":
                duration = int(tok[1:])
            elif c == "$":
                if tok.startswith("$dumpoff"):
                    settle_all(duration)
                    set_all_x()
                elif tok.startswith("$dumpall"):
                    pass
                # $dumpon：之后的变化自然覆盖 x 状态
            elif c in "01xz":
                sid = tok[1:]
                slots = id_slots.get(sid)
                if slots:
                    s = slots[0]
                    if duration >= t0:
                        base = s["tl"] if s["tl"] > t0 else t0
                        if duration > base:
                            s["t" + s["st"]] += duration - base
                        s["tl"] = duration
                        if s["st"] != c:
                            if (s["st"] == "0" and c == "1") or (s["st"] == "1" and c == "0"):
                                s["tc"] += 1
                    s["st"] = c                     # 窗口前：只更新当前值
            elif c in "bB":
                p = tok.split()
                if len(p) == 1:                      # 值和 id 分行
                    pending_vec = (duration, p[0][1:])
                    continue
                v, sid = p[0][1:], p[1]
                apply_vec(sid, v, duration, id_slots, t0)
            elif c == "r":
                continue                             # 实数变量：跳过
            elif pending_vec is not None:
                # 上一行只有 b<val>，本行是 id
                t, v = pending_vec
                apply_vec(tok, v, t, id_slots, t0)
                pending_vec = None

    # 收尾：所有位推进到 DURATION
    for slots in id_slots.values():
        for s in slots:
            base = s["tl"] if s["tl"] > t0 else t0
            if duration > base:
                s["t" + s["st"]] += duration - base
            s["tl"] = duration

    # ---- 树根重定位 ----
    node = root
    root_name = scope_names[1] if len(scope_names) > 1 else "top"
    if a.reroot:
        parts = a.reroot.split(".")
        for nm in parts:
            if nm not in node.children:
                sys.exit("!! --re-root 路径不存在: %s（缺 %s）" % (a.reroot, nm))
            node = node.children[nm]
        root_name = parts[-1]

    # ---- 外包实例链（配合 Vivado read_saif 默认剥两层）----
    # 注意最外层名字由 SAIF 头部的 INSTANCE 行承担，这里只包 names[1:]，
    # 否则最外层名字会重复出现、层次多一层，剥两层后对不上网表路径
    if a.wrap:
        names = a.wrap.split(".")
        root_name = names[0]
        for nm in reversed(names[1:]):
            outer = Node()
            outer.children[nm] = node
            node = outer

    # ---- 输出 SAIF ----
    # 语法已按 Vivado 2021.2 read_saif 实测校准（与 XSim open_saif/log_saif 产物同族，
    # 差异仅排版）：文件必须是 (SAIFILE ... 括号包裹、DIRECTION "backward"、
    # 网条目形如 (名字 (T0 x)(T1 x)(TX x)(TC n))、网名不能加引号（加引号报
    # Power 33-52 syntax error）、括号 [ ] 可不转义。缺 (SAIFILE 外壳时解析器
    # 不报错但一条网都不注记（Design nets matched = 1，那 1 个是时钟约束给的）。
    # read_saif 默认剥掉 SAIF 最外两层 INSTANCE：tb_ae{dut{u_core{...}}} 剥完
    # 剩 u_core，正好挂到 netlist 的 u_core 实例上。
    out = []
    out.append("(SAIFILE")
    out.append('  (SAIFVERSION "2.0")')
    out.append('  (DIRECTION "backward")')
    out.append('  (DESIGN )')
    out.append('  (DATE "%s")' % datetime.now().strftime("%a %b %d %H:%M:%S %Y"))
    out.append('  (VENDOR "")')
    out.append('  (PROGRAM_NAME "vcd2saif.py")')
    out.append('  (VERSION "1.0")')
    out.append("  (DIVIDER /)")
    out.append("  (TIMESCALE %s)" % timescale)
    out.append("  (DURATION %d)" % (duration - t0))

    def emit(n, depth):
        ind = "  " * (depth + 2)
        if n.nets:
            out.append(ind + "(NET")
            for nm, s in n.nets.items():
                out.append("%s  (%s (T0 %d) (T1 %d) (TX %d) (TC %d))"
                           % (ind, nm.replace('"', "'").replace(" ", "_"),
                              s["t0"], s["t1"], s["tx"], s["tc"]))
            out.append(ind + ")")
        for cn, ch in n.children.items():
            out.append("%s(INSTANCE %s" % (ind, cn))
            emit(ch, depth + 1)
            out.append(ind + ")")

    out.append("(INSTANCE %s" % root_name)
    emit(node, 0)
    out.append(")")
    out.append(")")
    with open(a.saif, "w") as g:
        g.write("\n".join(out) + "\n")

    def count(n):
        return len(n.nets) + sum(count(c) for c in n.children.values())
    print("[vcd2saif] timescale=%s 窗口=[%d,%d] duration=%d(单位同VCD) 根实例=%s NET 总数=%d"
          % (timescale, t0, duration, duration - t0, root_name, count(node)))


def apply_vec(sid, val, t, id_slots, t0):
    slots = id_slots.get(sid)
    if not slots:
        return
    w = len(slots)
    v = val[-w:] if len(val) >= w else val.rjust(w, "x")
    v = v.translate(str.maketrans("zZ?", "xxx"))
    for s, b in zip(slots, v):
        if t >= t0:
            base = s["tl"] if s["tl"] > t0 else t0
            if t > base:
                s["t" + s["st"]] += t - base
            s["tl"] = t
            if s["st"] != b:
                if (s["st"] == "0" and b == "1") or (s["st"] == "1" and b == "0"):
                    s["tc"] += 1
        s["st"] = b                                 # 窗口前：只更新当前值


if __name__ == "__main__":
    main()
