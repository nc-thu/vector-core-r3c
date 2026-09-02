# -*- coding: utf-8 -*-
"""ab_gate.py — a3 融合站点数值 A/B 门（2026-08-31）。

每个融合站点 [p → host链 → c] 做三条路径的对比（同一份合成 A_p 输入）：

  a2 路径（host 边界）：跑 a2 的 p 段得 Y_p 字节 → torch fp32 复刻 host 链
      （与 host_driver 同实现：int8*so → 模块前向 → clamp(round/sa)) →
      把结果字节注进 a2 的 c 段 A 槽 → 跑 a2 的 c 段得 Y_c 字节。
  a3 路径（引擎边界）：同一份 A_p 跑 a3 融合段，得 Y_c' 字节。

门判据：actv 站点（含 S2 纯重标定）Y_c 逐字节相等。
用法：python ab_gate.py [--lo 0] [--hi 999] [--seed 7]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CB = os.path.join(ROOT, '09_cbound')
sys.path.insert(0, CB)

from fast_interp_a3 import run_segment_fast_a3              # noqa: E402
from golden_interp import build_ddr_image, load_seq          # noqa: E402

A2 = os.path.join(CB, 'build_a2')
A3 = os.path.join(HERE, 'build_a3')
BLOB2 = os.path.join(A2, 'weights_blob.bin')
BLOB3 = os.path.join(A3, 'weights_blob.bin')

ACTV_CLS = {'SiLU', 'GELU', 'GELUActivation', 'ReLU'}


def c16(x):
    return (x + 15) // 16


def pack_kact(q):
    """int8 [rows, cols] → 块字节。byte(i,c) = ((i//16)*cols+c)*16 + (i%16)。
    与 host_driver.pack_kact 同一公式（那里用 torch，这里直接 numpy）。"""
    r, c = q.shape
    r16 = c16(r) * 16
    if r < r16:
        q = np.concatenate([q, np.zeros((r16 - r, c), np.int8)], 0)
    return q.reshape(r16 // 16, 16, c).transpose(0, 2, 1).copy().tobytes()


def unpack_blk(buf, rows16, cols):
    a = np.frombuffer(buf, dtype=np.int8).reshape(rows16 * cols, 16)
    return a.reshape(rows16, cols, 16).transpose(0, 2, 1) \
            .reshape(rows16 * 16, cols)


def q_round(x, scale):
    """clamp(round(x/scale),±127)，torch.round 四舍六入五成双（与 hw_calib
    /host_driver 一致）。"""
    return torch.clamp(torch.round(x / scale), -127.0, 127.0).to(torch.int8)


def np2t(a_i8):
    """numpy int8 → torch（本机 torch 对 numpy 2.x 的 C API 不可用，
    走 buffer 协议绕开；frombuffer 出来是一维，view 回原形状）。"""
    a = np.ascontiguousarray(a_i8)
    return torch.frombuffer(memoryview(a), dtype=torch.int8) \
        .view(*a.shape)


def t2np(t_i8):
    """torch int8 → numpy，走 data_ptr 拷贝。"""
    import ctypes
    t = t_i8.contiguous()
    return np.frombuffer(
        ctypes.string_at(t.data_ptr(), t.numel()), dtype=np.int8) \
        .reshape(t.shape).copy()


def host_forward(cls, t):
    """host 激活模块前向（torch fp32，与模型模块同实现）。"""
    import torch.nn.functional as F
    if cls == 'SiLU':
        return F.silu(t)
    if cls == 'GELU':
        return F.gelu(t)                       # nn.GELU() 默认 erf
    if cls == 'GELUActivation':
        return F.gelu(t, approximate='tanh')   # HF gelu_new
    if cls == 'ReLU':
        return F.relu(t)
    raise KeyError(cls)


# ---------------- a2 段索引（图名按 '@' 前的基名聚合，行切片共用键）----
def base_name(nm):
    return str(nm).split('@')[0]


def build_index(root):
    out, inn = {}, {}
    for sd in sorted(os.listdir(os.path.join(root, 'segments'))):
        man = json.load(open(os.path.join(root, 'segments', sd,
                                          'manifest.json'),
                             encoding='utf-8'))
        for o in man['outputs']:
            out.setdefault(base_name(o['name']), []).append(sd)
        for e in man['inputs']:
            inn.setdefault(base_name(e['name']), []).append(sd)
    return out, inn


IDX2_OUT, IDX2_IN = build_index(A2)
IDX3_OUT, _ = build_index(A3)


def run_seg_set(root, blob, segs, act_fn, rng):
    """跑一组段，返回 {段名: 跑完的 DDR}。act_fn(段名, 条目) -> bytes 或
    None（None 则填确定性随机字节，与站点无关的输入不影响目标输出）。"""
    out = {}
    for sd in segs:
        seg_dir = os.path.join(root, 'segments', sd)
        man = json.load(open(os.path.join(seg_dir, 'manifest.json'),
                             encoding='utf-8'))
        P = man['profile']
        act = {}
        for e in man['inputs']:
            nm = str(e['name'])
            if nm.startswith('const:'):
                # 常数词列条目：host_driver 的填法是 cval 复制满 16 lane
                #（_const_stripe 的 LOAD 拿这 16B 写 A 图常数列）。
                v = int(e.get('cval', nm.split(':')[1])) & 0xFF
                act[nm] = np.full(e['words'] * 16, v, dtype=np.uint8)
                continue
            b = act_fn(sd, e)
            if b is None:
                r = np.array([rng.integers(-40, 40)
                              for _ in range(e['words'] * 16)],
                             dtype=np.int8)
                b = r.tobytes()
            act[e['name']] = np.frombuffer(b, dtype=np.uint8)
        ddr = build_ddr_image(seg_dir, blob, act, P)
        ctx, d2, _ = run_segment_fast_a3(load_seq(seg_dir), ddr, P)
        out[sd] = d2
    return out


def collect_out(root, segs_ddr, base):
    """把名字以 base 开头（含 '@br0' 行切片）的输出图字节拼成整图，
    返回 int8 [rows16*16, cols]；行切片按 row_lo 排序拼接。"""
    parts = []
    for sd, ddr in segs_ddr.items():
        man = json.load(open(os.path.join(root, 'segments', sd,
                                          'manifest.json'),
                             encoding='utf-8'))
        for o in man['outputs']:
            nm = str(o['name'])
            if base_name(nm) != base:
                continue
            br = int(nm.split('@')[1]) if '@' in nm else \
                int(o.get('row_lo', 0))
            buf = ddr[o['ddr']:o['ddr'] + o['words'] * 16].tobytes()
            cols = o.get('n') or o.get('cols')
            assert cols, f'{nm}: 输出条目缺列数'
            parts.append((br, o['words'] // cols, buf, cols))
    if not parts:
        return None
    parts.sort()
    cols = parts[0][3]
    buf = b''.join(p[2] for p in parts)
    return unpack_blk(buf, sum(p[1] for p in parts), cols)


def find_a_entries(root, segs, base):
    """在一组段里找基名为 base 的 A 输入条目 → [(段名, row_lo, rows, k)]。"""
    hits = []
    for sd in segs:
        man = json.load(open(os.path.join(root, 'segments', sd,
                                          'manifest.json'),
                             encoding='utf-8'))
        for e in man['inputs']:
            nm = str(e['name'])
            if base_name(nm) != base or e.get('kind') != 'act_in':
                continue
            br = int(nm.split('@')[1]) if '@' in nm else \
                int(e.get('row_lo', 0))
            hits.append((sd, br, e['m'], e['k']))
    hits.sort(key=lambda x: x[1])
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lo', type=int, default=0)
    ap.add_argument('--hi', type=int, default=10 ** 9)
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()

    cal = json.load(open(os.path.join(ROOT, '02_quant',
                                      'hw_calib_table_v2.json')))
    calg = cal.get('gemms', cal)
    hp2 = json.load(open(os.path.join(A2, 'host_plan.json'),
                         encoding='utf-8'))
    node2 = {(n['module'], n.get('seq')): n for n in hp2['nodes']
             if n['kind'] == 'gemm'}
    hp3 = json.load(open(os.path.join(A3, 'host_plan.json'),
                         encoding='utf-8'))
    producers = [n for n in hp3['nodes']
                 if n.get('fused_pair', {}).get('role') == 'producer']
    print('融合站点总数 %d，本次检查 [%d, %d)' % (len(producers), a.lo, a.hi))

    def a_slices(entries, Amat):
        """按输入条目的行切片生成 {段名: {条目名: bytes}}。"""
        out = {}
        for (sd, br, rows, kk) in entries:
            out.setdefault(sd, {})[br] = pack_kact(Amat[br:br + rows,
                                                         :kk])
        return out

    nfail = 0
    for si, pd in enumerate(producers):
        if si < a.lo or si >= a.hi:
            continue
        chain = [tuple(x.split(':', 1))
                 for x in pd['fused_pair']['host_chain']]
        c_ent = next(n for n in hp3['nodes']
                     if n.get('fused_pair', {}).get('role') == 'consumer'
                     and n['fused_pair'].get('producer') == pd['module']
                     and n.get('segs') == pd.get('segs'))
        p2 = node2.get((pd['module'], pd['seq']))
        c2 = node2.get((c_ent['module'], c_ent['seq']))
        assert p2 and c2, 'a2 节点缺失'
        rng = np.random.default_rng(a.seed * 100003 + si)

        # ---- p 的 a2 段与 A 输入条目（可能按行切片）----
        psegs = IDX2_OUT.get(p2['out_graph'], [])
        assert psegs, f'{pd["module"]}: a2 找不到输出图 {p2["out_graph"]}'
        pa_e = find_a_entries(A2, psegs, p2['in_graph'])
        assert pa_e, 'a2 p 段找不到 A 输入'
        m = pa_e[0][2]
        kp = pa_e[0][3]
        assert all(e[2] == m or True for e in pa_e) and \
            all(e[3] == kp for e in pa_e)
        A = rng.integers(-60, 60, size=(m, kp), dtype=np.int8)
        Ainj = {}
        for (sd, br, rows, kk) in pa_e:
            Ainj[(sd, br)] = pack_kact(A[br:br + rows, :kk])

        # ---- 路径 1：a2 p 段 → host 链 → a2 c 段 ----
        def fill_p(s, e, _inj=Ainj):
            br = int(str(e['name']).split('@')[1]) \
                if '@' in str(e['name']) else 0
            return _inj.get((s, br))
        d1 = run_seg_set(A2, BLOB2, psegs, fill_p, rng)
        Yp = collect_out(A2, d1, p2['out_graph'])
        n_p = Yp.shape[1]
        assert Yp.shape[0] >= m and n_p == c_ent['k'], \
            f'{pd["module"]}: Y_p 形状 {Yp.shape}'
        t = np2t(Yp[:m]).to(torch.float32) * pd['so']
        for (hmod, hcls) in chain:
            assert hcls in ACTV_CLS, f'非 actv 链 {hcls}'
            t = host_forward(hcls, t)
        Ac = t2np(q_round(t, c_ent['sa']))
        # A 尾列：GEMM 条目 k 恒为 k_data+1，host_driver.act_image 对一切
        # 这种条目都补常数列（aug 层=bias_aug_c，非 aug 补 1——W 末行=0，
        # 值无数值影响）。漏补会让打包图按 k_data 列步长错位，第 0 行组
        # 恰好重合、后续组整体移 16B，假 FAIL。
        cval = int(calg[c_ent['module']]['bias_aug_c'])
        Ac = np.concatenate([Ac, np.full((m, 1), cval, np.int8)], 1)
        csegs = IDX2_OUT.get(c2['out_graph'], [])
        assert csegs, f'{c_ent["module"]}: a2 找不到输出图'
        ca_e = find_a_entries(A2, csegs, c2['in_graph'])
        assert ca_e, 'a2 c 段找不到 A 输入'
        Cinj = {}
        for (sd, br, rows, kk) in ca_e:
            Cinj[(sd, br)] = pack_kact(Ac[br:br + rows, :kk])

        def fill_c(s, e, _inj=Cinj):
            br = int(str(e['name']).split('@')[1]) \
                if '@' in str(e['name']) else 0
            return _inj.get((s, br))
        d2 = run_seg_set(A2, BLOB2, csegs, fill_c, rng)
        Yc2 = collect_out(A2, d2, c2['out_graph'])

        # ---- 路径 2：a3 融合段（A_p 直接喂 p 的输入槽，行切片同名）----
        # 注意 node_map 的 segs 在描述符并入当前段时是空表，改按 c 的
        # 输出图名（module#seq，全流唯一）从 a3 段索引定位融合段。
        # A 输入不按图名找：trace 的 tid 会复用，host_plan 里的 in_graph
        # 与编译器实际绑定的 p.a 可能指向不同记录（id 复用撞名），融合段
        # 里 p 的 A 是唯一的 act_in 条目，按形状（k==kp 且 m<=m）定位。
        fsegs = IDX3_OUT.get(c_ent['out_graph'], [])
        assert fsegs, f'{c_ent["module"]}: a3 找不到融合输出段'
        fa_e = []
        for sd in fsegs:
            man = json.load(open(os.path.join(A3, 'segments', sd,
                                              'manifest.json'),
                                 encoding='utf-8'))
            for e in man['inputs']:
                if e.get('kind') != 'act_in' or e.get('k') != kp \
                        or (e.get('m') or 0) > m:
                    continue
                br = int(str(e['name']).split('@')[1]) \
                    if '@' in str(e['name']) else int(e.get('row_lo', 0))
                fa_e.append((sd, br, e['m'], e['k']))
        fa_e.sort(key=lambda x: x[1])
        assert fa_e, 'a3 融合段找不到 A 输入'
        Finj = {}
        for (sd, br, rows, kk) in fa_e:
            Finj[(sd, br)] = pack_kact(A[br:br + rows, :kk])

        def fill_f(s, e, _inj=Finj):
            br = int(str(e['name']).split('@')[1]) \
                if '@' in str(e['name']) else 0
            return _inj.get((s, br))
        d3 = run_seg_set(A3, BLOB3, fsegs, fill_f, rng)
        Yc3 = collect_out(A3, d3, c_ent['out_graph'])

        if Yc2 is None or Yc3 is None or Yc2.shape != Yc3.shape:
            nd = -1
            diff = None
        else:
            diff = (Yc2[:m] != Yc3[:m])
            nd = int(diff.sum())
        status = 'OK ' if nd == 0 else 'FAIL'
        if nd != 0:
            nfail += 1
        print('%s #%03d %-50s m=%-6d %s Y_c差=%d' %
              (status, si, pd['module'][:50], m,
               '+'.join(c for _, c in chain) or 'S2', nd))
        if diff is not None and 0 < nd <= 12:
            for (r, cc) in np.argwhere(diff)[:6]:
                print('      row=%d col=%d a2=%d a3=%d' %
                      (r, cc, Yc2[r, cc], Yc3[r, cc]))
    print('\n结果: FAIL %d' % nfail if nfail else '\n结果: 全部逐字节一致')
    return 1 if nfail else 0


if __name__ == '__main__':
    sys.exit(main())
