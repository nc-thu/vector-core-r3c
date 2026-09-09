# -*- coding: utf-8 -*-
"""host_driver.py — HB-GD 端到端数值链驱动（host 步骤 torch fp + PL 段 fast_interp）。

架构（模型驱动）：
  1. 加载真模型（eval，CPU）与 batch_sXXX.pt，按 fp32_ref 的种子固定噪声。
  2. 打补丁（按 trace 记录分类；已验证每个模块要么全部降级、要么全部原生）：
     - gemm 记录（Conv/Linear）→ 量化输入 → 跑该 out_graph 的写入段
       → 拼装 Y 图反量化（+host fp bias）→ 返回 fp。
     - 七家族注意力 → 定制 forward（成员段、QK 段、host softmax、PV 段全在这里编排）。
     - host 记录（norm/actv/其它）不改 forward，只挂钩子把输出登记进 tensor_reg
       （共享段可能提前运行，需要别人已产出的 fp 张量）。
     - 记录被编译器丢弃的模块（MSDA、注意力 relays、SwinBlock 壳、
       RotaryEmbedding、BERT qkv…）原生 fp 执行。
  3. 段执行：build_ddr_image（权重 blob + 输入图）→ fast_interp → 输出图入店。

图存储：
  - tensor_reg: 't{id}' → fp 张量（模块入口/出口 + host 钩子登记）。
  - blk_store: 图名 → FIFO（int8 块 [rows16*16, cols]）；生产 push、消费 pop，
    顺序 = 编译序 = 模型调用序。
  - act_store: act_out 图名 → 出现次序列表（int8 块 + 行列摆位 + so/bias），
    assemble 时拼成整图。

段边界量化约定（与 02_quant/hw_calib.py 一致）：
  a_int = clamp(round(a_fp/sa), -127, 127)（round = 四舍六入五成双）
  aug 层 A 图末列 = bias_aug_c（常数），权重末行 = w_bias_int8（在 blob 里）
  y_fp = y_int * so (+ fp bias if host_bias)
  P_int = clamp(round(P_fp*127), -127, 127)
  σ_S 见 attn_calib.py：S_int = plain(q@k)/σ_S（无家族 scale、无位置调制；
  BERT exact_temp 例外，scale 已折进 requant，host 不再乘）。

Run (server):
  cd ~/workspace/holobrain
  CUDA_VISIBLE_DEVICES="" ~/.conda/envs/holobrain/bin/python \
      /tmp/ae_hostdrv/host_driver.py \
      --build /tmp/ae_hostdrv/build_s000 --trace /tmp/ae_hostdrv/trace_s000.json \
      --batch /tmp/ae_hostdrv/batch_s000.pt --ref /tmp/ae_hostdrv/fp32_ref_000.npz \
      --out /tmp/ae_hostdrv/result_000.npz
"""
import argparse
import json
import os
import sys
import time
import types
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fast_interp import run_segment_fast          # noqa: E402
from golden_interp import build_ddr_image, load_seq  # noqa: E402
from rtl_seg import run_segment_rtl               # noqa: E402

FAMILIES = {'RotaryAttention', 'TemporalJointGraphAttention', 'JointGraphAttention',
            'MultiheadAttention', 'BiMultiHeadAttention', 'WindowMSA',
            'BertAttention'}

NEG_INF = float('-inf')


def c16(x):
    return (x + 15) // 16


def q_round(x, scale):
    """clamp(round(x/scale), -127, 127)，torch.round 与 hw_calib 一致（五成双）。"""
    if scale is None or scale <= 0:
        scale = 1.0
    return torch.clamp(torch.round(x / scale), -127.0, 127.0).to(torch.int8)


def pack_kact(q):
    """int8 [rows, cols] → 块字节（pad 到 16 行）。
    byte(i,c) = ((i//16)*cols + c)*16 + (i%16)。"""
    r, c = q.shape
    r16 = c16(r) * 16
    if r < r16:
        q = torch.cat([q, q.new_zeros((r16 - r, c))])
    a = q.numpy()
    return a.reshape(r16 // 16, 16, c).transpose(0, 2, 1).copy().tobytes()


def unpack_blk(buf, rows16, cols):
    """块字节 → int8 [rows16*16, cols]（pack_kact 的逆）。"""
    a = np.frombuffer(buf, dtype=np.int8).reshape(rows16 * cols, 16)
    return a.reshape(rows16, cols, 16).transpose(0, 2, 1).reshape(rows16 * 16,
                                                                 cols)


def _fix_kinematics_device(obj, seen=None):
    """batch 在 GPU 环境生成：map_location 只搬张量，Transform3d 的
    .device 属性仍是 'cuda'，chain 会误触发跨设备 clone（无 GPU 时崩）。
    递归找出带 chain 的对象，把整条链的 device/dtype 属性改回 cpu。"""
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
                print(f'[fix] kinematics chain {n} 处 device→cpu')


def _g(args, kw, i, name, default=None):
    """位置参数或关键字参数取值。"""
    if name in kw and kw[name] is not None:
        return kw[name]
    if i < len(args) and args[i] is not None:
        return args[i]
    return default


class HostDriver:
    """模型补丁器 + 段调度器。"""

    def __init__(self, build_dir, trace_path, calib_fp, model, engine='fast',
                 fp_after=None):
        self.dir = build_dir
        self.model = model
        self.engine = engine
        # 深度二分：全局第 fp_after 次「量化调用」(gemm/attn 补丁) 之后，
        # 后续调用全部走原生 fp 前向（输出照常 reg，下游段供给不断）
        self.fp_after = fp_after
        self.gidx = 0
        self.P = None
        hp = json.load(open(os.path.join(build_dir, 'host_plan.json'),
                            encoding='utf-8'))
        self.hp = hp
        self.seg_names = [s['name'] for s in hp['segments']]
        self.mans = {n: json.load(open(os.path.join(build_dir, 'segments', n,
                                                    'manifest.json'),
                                       encoding='utf-8'))
                     for n in self.seg_names}
        self.seqcache = {}
        self._t0 = time.perf_counter()
        self.wseg = defaultdict(list)
        self.rents = defaultdict(list)
        for n in self.seg_names:
            for o in self.mans[n]['outputs']:
                # 输出名 '@N' = 同一图的行切分，归一化到基名，
                # run_graph/assemble 都按基名找
                self.wseg[o['name'].split('@')[0]].append(n)
            for e in self.mans[n]['inputs']:
                self.rents[e['name']].append((n, e))
        self.node_by = {}
        self.gnodes = defaultdict(list)
        self.anodes = defaultdict(list)
        for n in hp['nodes']:
            self.node_by[(n['module'], n['seq'])] = n
            if n['kind'] == 'gemm':
                self.gnodes[n['module']].append(n)
            elif n['kind'] == 'attn':
                self.anodes[n['module']].append(n)
        self.ameta = {}
        for st in hp['host_steps']:
            a = st.get('attn')
            if a:
                self.ameta[(a['module'], a['seq'])] = a
        tr = json.load(open(trace_path, encoding='utf-8'))
        self.recs_by_mod = defaultdict(list)
        for r in tr['ops']:
            self.recs_by_mod[r['module']].append(r)
        self.cursor = defaultdict(int)
        self.calib = json.load(open(calib_fp, encoding='utf-8'))
        if '_meta' in self.calib:
            self.calib.pop('_meta')
        # v2 表的条目在 'gemms' 下；不展开会导致所有 aug 常数列
        # 查不到、c 退回 1（所有带偏置 GEMM 的 bias 全错）
        if 'gemms' in self.calib:
            self.calib = self.calib['gemms']
        self.tensor_reg = {}
        self.blk = defaultdict(list)
        self.act = defaultdict(list)
        self.done = set()
        self.bias_sd = model.state_dict()
        self.stats = defaultdict(int)
        self.missing = []
        self.warned = set()
        # aug 常数列：module → c 值（aug 是层属性）。注意 k_eff 恒为
        # k+1：host_bias 层的 W 末行全零，常数列乘零行、c 值无关紧要，
        # 不要对它们误报「缺 bias_aug_c」。
        self.augc = {}
        self.augmods = set()
        for (mod, _seq), n in self.node_by.items():
            if n['kind'] == 'gemm' and n.get('aug'):
                self.augmods.add(mod)
                e = self.calib.get(mod)
                if e and e.get('bias_aug_c'):
                    self.augc.setdefault(mod, e['bias_aug_c'])

    # ================= 基础 =================
    def reg(self, tid, x):
        if torch.is_tensor(x):
            self.tensor_reg[f't{tid}' if not isinstance(tid, str) else tid] = x

    def run(self, seg_name, need=None):
        if seg_name in self.done:
            return True
        man = self.mans[seg_name]
        act = {}
        missing = []
        for e in man['inputs']:
            nm = e['name']
            if nm.startswith('const:'):
                act[nm] = np.full(e['words'] * 16, int(e.get('cval', 0)) & 0xFF,
                                  dtype=np.uint8)
            elif e.get('kind') == 'act_in':
                b = self.act_image(e)
                if b is None:
                    missing.append(nm)
                    continue
                act[nm] = np.frombuffer(b, dtype=np.uint8)
            else:
                if not self.blk[nm]:
                    missing.append(nm)
                    continue
                blk = self.blk[nm].pop(0)
                act[nm] = np.frombuffer(pack_kact(torch.from_numpy(blk.copy())),
                                        dtype=np.uint8)
        if missing:
            self.missing.append((seg_name, missing, need))
            return False
        img = build_ddr_image(os.path.join(self.dir, 'segments', seg_name),
                              os.path.join(self.dir, 'weights_blob.bin'),
                              act, self.P)
        seq = self.seqcache.get(seg_name)
        if seq is None:
            seq = load_seq(os.path.join(self.dir, 'segments', seg_name))
            self.seqcache[seg_name] = seq
        if self.engine == 'rtl':
            _, ddr, _ = run_segment_rtl(
                os.path.join(self.dir, 'segments', seg_name), img, self.P)
        else:
            _, ddr, _ = run_segment_fast(seq, img, self.P)
        for o in man['outputs']:
            buf = ddr[o['ddr']:o['ddr'] + o['words'] * 16].tobytes()
            if o.get('kind') == 'act_out':
                self.act[o['name'].split('@')[0]].append((seg_name, o, buf))
            else:
                self.blk[o['name']].append(
                    unpack_blk(buf, o['rows16'], o['cols']))
        self.done.add(seg_name)
        self.stats['segments'] += 1
        if self.engine == 'rtl':
            print('[rtl] %d/%d %s %.1fs' % (self.stats['segments'],
                                            len(self.seg_names), seg_name,
                                            time.perf_counter() - self._t0),
                  flush=True)
            self._t0 = time.perf_counter()
        return True

    def act_image(self, e):
        """act_in 条目 → 量化字节（aug 层补常数列；行切分 '@N' 后缀去掉）。"""
        x = self.tensor_reg.get(e['name'].split('@')[0])
        if x is None:
            return None
        m, k = e['m'], e['k']
        rl = e.get('row_lo') or 0
        rh = e.get('row_hi') or m
        cl = e.get('col_lo') or 0
        ch = e.get('col_hi') or k
        W = x.shape[-1] if x.dim() > 1 else 1
        X2 = x.contiguous().reshape(-1, W)
        xs = X2[rl:rh, cl:ch] if rh > rl else torch.zeros((0, ch - cl))
        q = q_round(xs, e.get('sa'))
        if k == xs.shape[1] + 1:                  # aug 常数列
            c = self.augc.get(e.get('module'))
            if c is None:
                c = 1.0
                if e.get('module') in self.augmods and \
                        ('augc', e.get('module')) not in self.warned:
                    print(f'[warn] {e.get("module")}: 真 aug 层缺 bias_aug_c，'
                          f'常数列=1')
                    self.warned.add(('augc', e.get('module')))
            q = torch.cat([q, torch.full((q.shape[0], 1), float(c),
                                         dtype=torch.int8)], 1)
        return pack_kact(q)

    def run_graph(self, gname):
        ok = True
        for s in self.wseg.get(gname, ()):
            ok &= self.run(s, need=gname)
        return ok

    def assemble(self, gname):
        occ = self.act.get(gname) or []
        if not occ:
            return None
        mt = max(o.get('row_hi', o['m']) for _, o, _ in occ)
        nt = max(o.get('col_hi', o['n']) for _, o, _ in occ)
        Y = torch.zeros(mt, nt)
        for _, o, buf in occ:
            m, n = o['m'], o['n']
            cl, ch = o.get('col_lo', 0), o.get('col_hi', n)
            rl, rh = o.get('row_lo', 0), o.get('row_hi', m)
            pitch = o.get('pitch', ch - cl)
            a = np.frombuffer(buf, dtype=np.int8).reshape(-1, 16)
            g = a.reshape(-1, pitch, 16).transpose(0, 2, 1).reshape(-1, pitch)
            y = torch.from_numpy(g[:m].astype(np.float32)) * float(o['so'])
            if o.get('host_bias'):
                b = self.get_bias(o.get('bias_key'))
                if b is not None:
                    y = y + b[cl:ch].float()
            Y[rl:rh, cl:ch] = y
        return Y

    def get_bias(self, bias_key):
        if bias_key is None or self.bias_sd is None:
            return None
        if bias_key.endswith('_weight'):
            bk = bias_key[:-len('_weight')] + '_bias'
        elif bias_key.endswith('.weight'):
            bk = bias_key[:-len('.weight')] + '.bias'
        else:
            bk = bias_key + '.bias'
        t = self.bias_sd.get(bk)
        return t.detach().float() if t is not None else None

    # ================= 块存取 =================
    def pop_blks(self, name, cnt=None):
        q = self.blk[name]
        if cnt is None:
            cnt = len(q)
        return [q.pop(0) for _ in range(min(cnt, len(q)))]

    def deq(self, blk, sigma, rows=None):
        a = torch.from_numpy(np.ascontiguousarray(blk).astype(np.float32))
        if rows is not None:
            a = a[:rows]
        return a * float(sigma or 1.0)

    def push_blk(self, name, q, rows16=None):
        a = q.numpy()
        r = a.shape[0]
        r16 = rows16 if rows16 is not None else c16(r)
        if r < r16 * 16:
            a = np.concatenate([a, np.zeros((r16 * 16 - r, a.shape[1]),
                                            dtype=np.int8)], 0)
        self.blk[name].append(a)

    def push_batched(self, name, full):
        """full: np int8 [R, cols]。按消费段顺序（rows16*16 行一批）切片 push。"""
        a = full if isinstance(full, np.ndarray) else full.numpy()
        idx = 0
        for s in self.seg_names:
            for e in self.mans[s]['inputs']:
                if e['name'] == name:
                    rows = e['rows16'] * 16
                    if idx + rows > a.shape[0]:
                        print(f'[warn] {name}: 批切片越界 @{s}')
                        return
                    self.blk[name].append(a[idx:idx + rows])
                    idx += rows
        exp = sum(e['rows16'] * 16 for _, e in self.rents.get(name, ()))
        if idx != exp:
            print(f'[warn] {name}: push {idx} 行 ≠ 消费 {exp} 行')

    def count_consumers(self, name):
        return len(self.rents.get(name, ()))

    def must_assemble(self, gname):
        self.run_graph(gname)
        Y = self.assemble(gname)
        if Y is None:
            msg = self.missing[-1] if self.missing else None
            raise RuntimeError(f'{gname}: 无法拼装（缺输入 {msg}）')
        return Y

    # ================= 通用 GEMM =================
    def gemm_call(self, mod, node, rec, x):
        self.stats['gemm_calls'] += 1
        m, n = node['m'], node['n']
        in_shape, out_shape = rec['in_shapes'][0], rec['out_shapes'][0]
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Conv1d)):
            x2d = self.im2col(mod, x)
            self.reg(rec['in_ids'][0], x2d)
        else:
            self.reg(rec['in_ids'][0], x)
        try:
            Y = self.must_assemble(node['out_graph'])
        except RuntimeError as e:
            # 段输入暂时不齐（融合段跨了未 trace 的 functional 生产者）：
            # 原生 fp 兜底，保证链路走通；该图计为未覆盖
            self.stats['gemm_native_fallback'] = \
                self.stats.get('gemm_native_fallback', 0) + 1
            print(f'[fallback] {node["module"]}#{node["seq"]}: {e}')
            out = mod._hb_orig_fwd(x)
            self.reg(rec['out_ids'][0], out)
            return out
        out = self._shape_out(Y, mod, in_shape, out_shape)
        self.reg(rec['out_ids'][0], out)
        return out

    @staticmethod
    def im2col(mod, x):
        if isinstance(mod, torch.nn.Conv1d):
            # Conv1d → 假 2D：空间维垫在 H 上，dilation/padding/stride
            # 取标量（1 元组会让 unfold 的维度检查报错）
            x4 = x[:, :, None, :]                     # (N, C, 1, L)
            ks = mod.kernel_size[0] if isinstance(mod.kernel_size, tuple) \
                else mod.kernel_size
            dil = mod.dilation[0] if isinstance(mod.dilation, tuple) \
                else mod.dilation
            pad = mod.padding[0] if isinstance(mod.padding, tuple) \
                else mod.padding
            st = mod.stride[0] if isinstance(mod.stride, tuple) \
                else mod.stride
            cols = F.unfold(x4, (1, ks), dilation=dil, padding=pad,
                            stride=st)
        else:
            x4 = x
            kh, kw = mod.kernel_size[:2]
            cols = F.unfold(x4, (kh, kw), dilation=mod.dilation,
                            padding=mod.padding, stride=mod.stride)
        return cols.transpose(1, 2).reshape(-1, cols.shape[1])

    @staticmethod
    def _shape_out(Y, mod, in_shape, out_shape):
        if isinstance(mod, torch.nn.Conv1d) and len(out_shape) == 3:
            return Y.reshape(out_shape[0], out_shape[2],
                             out_shape[1]).permute(0, 2, 1).contiguous()
        if isinstance(mod, torch.nn.Conv2d) and len(out_shape) == 4:
            B, Cout, H, W = out_shape
            return Y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return Y.reshape(out_shape)

    # ================= 家族 forward =================
    def _member_node(self, mod_name):
        i = self.cursor[mod_name]
        lst = self.gnodes.get(mod_name, [])
        return lst[i] if i < len(lst) else None

    def _advance(self, mod_name):
        self.cursor[mod_name] += 1

    def _run_member(self, node):
        mod, seq = node['module'], node['seq']
        hm = node.get('heads_mode')
        if hm:
            H, _d, twin = hm
            for h in range(H):
                self.run_graph(f'{mod}#{seq}#h{h}')
            if twin:
                for h in range(H):
                    self.run_graph(f'{mod}#{seq}#vt{h}')
        else:
            self.run_graph(f'{mod}#{seq}')

    def _pop_heads(self, node, rows, peek=False):
        """成员 '#h{h}' 块 → [H] 个 fp [rows, d]。

        peek=True 只读不弹：S 段自己要消费同名块（JG 的 q/k 原块
        直接进 QK^T，host 只借读算 Δ）。"""
        H, d, _ = node['heads_mode']
        base = f"{node['module']}#{node['seq']}"
        out = []
        for h in range(H):
            if not self.blk[f'{base}#h{h}']:
                raise RuntimeError(
                    f'{base}#h{h} 块缺 missing='
                    f'{self.missing[-3:] if self.missing else []!r}')
            b = self.blk[f'{base}#h{h}'][0] if peek else \
                self.blk[f'{base}#h{h}'].pop(0)
            out.append(self.deq(b, node['so'], rows))
        return out, d

    # ---------- RotaryAttention ----------
    def fwd_rotary(self, mod, node, rec, args, kw):
        query = _g(args, kw, 0, 'query')
        key = _g(args, kw, 1, 'key')
        value = _g(args, kw, 2, 'value')
        value = key if value is None else value
        identity = _g(args, kw, 8, 'identity')
        identity = query if identity is None else identity
        query_pos = _g(args, kw, 3, 'query_pos')
        key_pos = _g(args, kw, 4, 'key_pos')
        attn_mask = _g(args, kw, 5, 'attn_mask')
        kpm = _g(args, kw, 7, 'key_padding_mask')
        B, N, C = query.shape
        M = key.shape[1]
        meta = self.ameta[(node['module'], node['seq'])]
        H, d, scale = meta['H'], meta['d'], mod.scale
        if B != 1:
            raise RuntimeError(f'rotary B={B}>1 未支持')
        qn = self._member_node(f'{node["module"]}.q_proj')
        kn = self._member_node(f'{node["module"]}.k_proj')
        vn = self._member_node(f'{node["module"]}.v_proj')
        self.reg(qn['in_graph'], query)
        self.reg(kn['in_graph'], key)
        self.reg(vn['in_graph'], value)
        for x in (qn, kn, vn):
            self._run_member(x)
            self._advance(x['module'])
        # host rotary：只在有 pos 时动 Q/K 块（否则原块直通，避免二次量化）
        if query_pos is not None:
            qs, _ = self._pop_heads(qn, N)
            q4 = torch.stack(qs, 1).view(B, H, N, d)
            q4 = mod.position_encoder(q4, query_pos)
            qs = [q4[0, h] for h in range(H)]
            for h in range(H):
                self.push_blk(f"{qn['module']}#{qn['seq']}#h{h}",
                              q_round(qs[h], qn['so']), c16(N))
        if key_pos is not None:
            ks, _ = self._pop_heads(kn, M)
            k4 = torch.stack(ks, 1).view(B, H, M, d)
            k4 = mod.position_encoder(k4, key_pos)
            ks = [k4[0, h] for h in range(H)]
            for h in range(H):
                self.push_blk(f"{kn['module']}#{kn['seq']}#h{h}",
                              q_round(ks[h], kn['so']), c16(M))
        # QK 段 → S → host softmax → P
        Sname = f'S:{node["module"]}#{node["seq"]}'
        self.run_graph(Sname)
        for h in range(H):
            S = self.deq(self.blk[Sname].pop(0), meta['sigma_s'], N)[:, :M]
            att = S * scale
            if attn_mask is not None:
                am = attn_mask.unsqueeze(1) if (attn_mask.dim() == 3 and
                                                attn_mask.shape[0] == B) \
                    else attn_mask
                att = torch.where(am, NEG_INF, att)
            if kpm is not None:
                kp = kpm[0] if kpm.dim() == 2 else kpm
                att = torch.where(kp[None, :], NEG_INF, att)
            P = att.reshape(N, M).softmax(-1)
            self.push_blk(f'P:{node["module"]}#{node["seq"]}#{h}',
                          q_round(P, 1.0 / 127), c16(N))
        Y = self.must_assemble(node['out_graphs'][0])
        out = Y.view(B, N, C) + identity
        self.reg(rec['out_ids'][0], out)
        self._advance(f'{node["module"]}.proj')
        return out

    # ---------- JointGraphAttention ----------
    def fwd_jg(self, mod, node, rec, args, kw):
        query = _g(args, kw, 0, 'query')
        key = _g(args, kw, 1, 'key')
        identity = _g(args, kw, 4, 'identity')
        identity = query if identity is None else identity
        query_pos = _g(args, kw, 3, 'query_pos')
        B, N, C = query.shape
        M = key.shape[1]
        meta = self.ameta[(node['module'], node['seq'])]
        H, d, scale = meta['H'], meta['d'], mod.scale
        if B != 1:
            raise RuntimeError(f'jg B={B}>1 未支持')
        qn = self._member_node(f'{node["module"]}.q_proj')
        kn = self._member_node(f'{node["module"]}.k_proj')
        vn = self._member_node(f'{node["module"]}.v_proj')
        self.reg(qn['in_graph'], query)
        self.reg(kn['in_graph'], key)
        self.reg(vn['in_graph'], key)             # v = v_proj(key)
        for x in (qn, kn, vn):
            self._run_member(x)
            self._advance(x['module'])
        qs, _ = self._pop_heads(qn, N, peek=True)   # S 段还要消费原块
        ks, _ = self._pop_heads(kn, M, peek=True)
        q4 = torch.stack(qs, 1).view(B, H, N, d)   # (b,h,n,c)
        k4 = torch.stack(ks, 1).view(B, H, M, d)
        qe = q4.unsqueeze(3)                       # (b,h,n,1,c)
        S_plain = (qe * k4.unsqueeze(2)).sum(-1) * scale
        if query_pos is not None:
            qp = mod.position_encoder(query_pos.flatten()).reshape(-1, N, M, C)
            qp = qp.unflatten(-1, (H, d)).permute(0, 3, 1, 2, 4)  # (b,h,n,m,c)
            if B != qp.shape[0]:
                qp = qp.tile(B // qp.shape[0], 1, 1, 1, 1)
            S_pos = (qe * qp * k4.unsqueeze(2)).sum(-1) * scale
        else:
            S_pos = S_plain
        delta = (S_pos - S_plain)[0]               # (h,n,m)
        Sname = f'S:{node["module"]}#{node["seq"]}'
        self.run_graph(Sname)
        for h in range(H):
            S = self.deq(self.blk[Sname].pop(0), meta['sigma_s'], N)[:, :M]
            P = (S * scale + delta[h]).softmax(-1)
            self.push_blk(f'P:{node["module"]}#{node["seq"]}#{h}',
                          q_round(P, 1.0 / 127), c16(N))
        Y = self.must_assemble(node['out_graphs'][0])
        out = Y.view(B, N, C) + identity
        self.reg(rec['out_ids'][0], out)
        self._advance(f'{node["module"]}.proj')
        return out

    # ---------- TemporalJointGraphAttention ----------
    def fwd_temporal(self, mod, node, rec, args, kw):
        query = _g(args, kw, 0, 'query')
        key = _g(args, kw, 1, 'key')
        value = _g(args, kw, 2, 'value')
        value = key if value is None else value
        identity = _g(args, kw, 7, 'identity')
        identity = query if identity is None else identity
        jd = _g(args, kw, 3, 'joint_distance')
        tpq = _g(args, kw, 4, 'temporal_pos_q')
        tpk = _g(args, kw, 5, 'temporal_pos_k')
        tam = _g(args, kw, 6, 'temporal_attn_mask')
        B, N, Tq, C = query.shape
        M, Tk = key.shape[1:3]
        U, mk, mq = N, M * Tk, Tq
        meta = self.ameta[(node['module'], node['seq'])]
        H, d, scale = meta['H'], meta['d'], mod.scale
        if B != 1:
            raise RuntimeError(f'temporal B={B}>1 未支持')
        qn = self._member_node(f'{node["module"]}.q_proj')
        kn = self._member_node(f'{node["module"]}.k_proj')
        vn = self._member_node(f'{node["module"]}.v_proj')
        self.reg(qn['in_graph'], query)
        self.reg(kn['in_graph'], key)
        self.reg(vn['in_graph'], value)
        # joint_pos_encoder.mlp.0 和 v 成员融在同一段：段的第二个
        # act_in 是 mlp.0 的输入（jd 经 sin/cos 展开成 256 维），它要等
        # joint_pos_encoder 被调用才存在。所以 jdis 必须先算——mlp.0 的
        # gemm_call 触发该段时 v 输入已注册，段一次算出 VT 孪生块。
        jdis_e = None
        if jd is not None:
            jdis = mod.joint_pos_encoder(jd.flatten()).reshape(B, N, M, C)
            jdis_e = jdis.unflatten(-1, (H, d)).permute(0, 3, 1, 2, 4)
            jdis_e = jdis_e.unsqueeze(3)               # (b,h,n,1,m,c)
        for x in (qn, kn, vn):
            self._run_member(x)
            self._advance(x['module'])
        qs, _ = self._pop_heads(qn, N * Tq)
        ks, _ = self._pop_heads(kn, M * Tk)
        q4 = torch.stack(qs, 1).view(B, H, N, Tq, d)
        k4 = torch.stack(ks, 1).view(B, H, M, Tk, d)
        if tpq is not None:
            q4 = mod.temporal_position_encoder(q4, tpq)
        if tpk is not None:
            k4 = mod.temporal_position_encoder(k4, tpk)
        for h in range(H):
            for j in range(U):
                self.push_blk(f"{qn['module']}#{qn['seq']}#h{h}u{j}",
                              q_round(q4[0, h, j], qn['so']), c16(Tq))
                self.push_blk(f"{kn['module']}#{kn['seq']}#h{h}u{j}",
                              q_round(k4[0, h].reshape(mk, d), kn['so']),
                              c16(mk))
        # VT：孪生块搬运（int 不动，逐头一份，条目名带 #vt{h}）
        twins = []
        for h in range(H):
            vk = f"{vn['module']}#{vn['seq']}#vt{h}"
            if not self.blk[vk]:
                raise RuntimeError(
                    f'{vk} 空：v 成员段未产出（missing='
                    f'{self.missing[-3:] if self.missing else []!r}）')
            twins.append(self.blk[vk].pop(0))
        for h in range(H):
            self.blk[f'VT:{node["module"]}#{node["seq"]}#h{h}'].append(
                twins[h])
        # joint Δ（σ_S 是无调制 q@k；jdis 已在成员段之前算好）
        qe = q4.unsqueeze(4)                       # (b,h,n,tq,1,c)
        S_plain = torch.einsum('bhnqmc,bhmkc->bhnqmk', qe, k4) * scale
        if jdis_e is not None:
            S_pos = torch.einsum('bhnqmc,bhmkc->bhnqmk', qe * jdis_e,
                                 k4) * scale
        else:
            S_pos = S_plain
        delta = (S_pos - S_plain)[0]               # (h,n,tq,m,tk)
        # temporal mask (Tq,Tk) → [Tq, M*Tk]（对 m 维广播；源码 where
        # 语义：True→-inf；float 掩码按加法）
        W = None
        Wadd = None
        if tam is not None:
            if tam.dtype == torch.bool:
                W = tam[:, None, :].expand(Tq, M, Tk).reshape(Tq, mk)
            else:
                Wadd = tam[:, None, :].expand(Tq, M, Tk).reshape(Tq, mk)
        Sname = f'S:{node["module"]}#{node["seq"]}'
        self.run_graph(Sname)
        for j in range(U):
            for h in range(H):
                S = self.deq(self.blk[Sname].pop(0), meta['sigma_s'],
                             Tq)[:, :mk]
                att = S * scale + delta[h, j].reshape(Tq, -1)
                if W is not None:
                    att = att.masked_fill(W, NEG_INF)
                if Wadd is not None:
                    att = att + Wadd
                P = att.softmax(-1)
                self.push_blk(f'P:{node["module"]}#{node["seq"]}#u{j}h{h}',
                              q_round(P, 1.0 / 127), c16(Tq))
        Y = self.must_assemble(node['out_graphs'][0])
        out = Y.view(B, N, Tq, C) + identity
        self.reg(rec['out_ids'][0], out)
        self._advance(f'{node["module"]}.proj')
        return out

    # ---------- MultiheadAttention（eager 语义）----------
    def fwd_mha(self, mod, node, rec, args, kw):
        query = _g(args, kw, 0, 'query')
        key = _g(args, kw, 1, 'key')
        value = _g(args, kw, 2, 'value')
        value = key if value is None else value
        attn_mask = _g(args, kw, 5, 'attn_mask')
        kpm = _g(args, kw, 3, 'key_padding_mask')
        E, H = mod.embed_dim, mod.num_heads
        hd = E // H
        N, B, _ = query.shape
        if B != 1 or key.shape[0] != N or key.shape[2] != E:
            raise RuntimeError(f'MHA 形状 {tuple(query.shape)} '
                               f'{tuple(key.shape)} 未支持')
        meta = self.ameta[(node['module'], node['seq'])]
        base = f'{node["module"]}#in_proj#{node["seq"]}'
        self.tensor_reg[f'in:{node["module"]}'] = query.reshape(N, E)
        Y = self.must_assemble(base)               # [N, 3E]（含 fp bias）
        q, k, v = Y[:, :E], Y[:, E:2 * E], Y[:, 2 * E:]
        Nq16 = c16(N)
        for h in range(H):
            self.push_blk(f'{base}#q{h}', q_round(q[:, h * hd:(h + 1) * hd],
                                                  meta['sigma_q']), Nq16)
            self.push_blk(f'{base}#k{h}', q_round(k[:, h * hd:(h + 1) * hd],
                                                  meta['sigma_k']), Nq16)
            vt = v[:, h * hd:(h + 1) * hd].t().contiguous()
            self.push_blk(f'{base}#vt{h}', q_round(vt, meta['sigma_vs'][0]),
                          c16(hd))
        Sname = f'S:{node["module"]}#{node["seq"]}'
        self.run_graph(Sname)
        for h in range(H):
            S = self.deq(self.blk[Sname].pop(0), meta['sigma_s'], N)[:, :N]
            att = S * (hd ** -0.5)               # [N, N]
            if attn_mask is not None:
                m = attn_mask
                if m.dim() == 3:                 # (B·H, Nq, Nk) 通道
                    m0 = m[0]
                    if not torch.allclose(m, m0.unsqueeze(0).expand_as(m)
                                          .to(m.dtype)):
                        print('[warn] MHA 3D mask 各通道不同，取通道 0')
                    m = m0
                if m.dtype == torch.bool:
                    att = att.masked_fill(m, NEG_INF)
                else:
                    att = att + m
            if kpm is not None:
                kp = kpm[0] if kpm.dim() == 2 else kpm
                att = att.masked_fill(kp[None, :], NEG_INF)
            P = att.softmax(-1)
            self.push_blk(f'P:{node["module"]}#{node["seq"]}#{h}',
                          q_round(P, 1.0 / 127), Nq16)
        Y2 = self.must_assemble(node['out_graphs'][0])
        out = Y2.view(N, B, E)
        self.reg(rec['out_ids'][0], out)
        self._advance(f'{node["module"]}.out_proj')
        return out, None

    # ---------- BiMultiHeadAttention ----------
    def fwd_bimha(self, mod, node, rec, args, kw):
        vision = _g(args, kw, 0, 'vision')
        lang = _g(args, kw, 1, 'lang')
        am_v = _g(args, kw, 2, 'attention_mask_v')
        am_l = _g(args, kw, 3, 'attention_mask_l')
        bsz, Nv, _ = vision.shape
        Nl = lang.shape[1]
        meta = self.ameta[(node['module'], node['seq'])]
        H, d, scale = meta['H'], meta['d'], mod.scale
        if bsz != 1:
            raise RuntimeError(f'bimha bsz={bsz}>1 未支持')
        vn = self._member_node(f'{node["module"]}.v_proj')
        ln = self._member_node(f'{node["module"]}.l_proj')
        vvn = self._member_node(f'{node["module"]}.values_v_proj')
        vln = self._member_node(f'{node["module"]}.values_l_proj')
        self.reg(vn['in_graph'], vision)
        self.reg(ln['in_graph'], lang)
        self.reg(vvn['in_graph'], vision)
        self.reg(vln['in_graph'], lang)
        for x in (vn, ln, vvn, vln):
            self._run_member(x)
            self._advance(x['module'])
        Qv = self.must_assemble(vn['out_graph']).view(bsz, Nv, H, d)
        Ql = self.must_assemble(ln['out_graph']).view(bsz, Nl, H, d)
        Vv = self.must_assemble(vvn['out_graph']).view(bsz, Nv, H, d)
        Vl = self.must_assemble(vln['out_graph']).view(bsz, Nl, H, d)
        for h in range(H):
            self.push_blk(f"{vn['out_graph']}#Qv{h}",
                          q_round(Qv[0, :, h], meta['sigma_q']), c16(Nv))
            self.push_blk(f"{vn['out_graph']}#Kv{h}",
                          q_round(Qv[0, :, h], meta['sigma_q']), c16(Nv))
            self.push_blk(f"{ln['out_graph']}#Ql{h}",
                          q_round(Ql[0, :, h], meta['sigma_k']), c16(Nl))
            self.push_blk(f"{ln['out_graph']}#Kl{h}",
                          q_round(Ql[0, :, h], meta['sigma_k']), c16(Nl))
        # 编译器侧名字是 f'S{nm}'，nm='S_v' → 'SS_v:'
        Sv_n = f'SS_v:{node["module"]}#{node["seq"]}'
        Sl_n = f'SS_l:{node["module"]}#{node["seq"]}'
        self.run_graph(Sv_n)
        self.run_graph(Sl_n)
        CL = mod.MAX_CLAMP_VALUE
        Nlp, Nvp = c16(Nl) * 16, c16(Nv) * 16
        Pv, Pl = [], []
        for h in range(H):
            if not self.blk[Sv_n] or not self.blk[Sl_n]:
                miss = self.missing[-6:] if self.missing else []
                raise RuntimeError(
                    f'bimha S 块缺 h{h} Sv={len(self.blk[Sv_n])} '
                    f'Sl={len(self.blk[Sl_n])} missing={miss!r}')
            Sv = self.deq(self.blk[Sv_n].pop(0), meta['sigma_s'],
                          Nv)[:, :Nl] * scale
            Sl = self.deq(self.blk[Sl_n].pop(0), meta['sigma_s'],
                          Nl)[:, :Nv] * scale
            Sv = torch.clamp(Sv, -CL, CL)
            SlT = torch.clamp(Sl, -CL, CL)
            SlT = SlT - SlT.max(-1, keepdim=True)[0]
            SlT = torch.clamp(SlT, -CL, CL)
            if am_v is not None:
                SlT = SlT.masked_fill(am_v[:, None, :], NEG_INF)
            if am_l is not None:
                add = torch.where(am_l == 0, -9e15,
                                  torch.zeros((), dtype=Sv.dtype))
                Sv = Sv + add[:, None, :]
            # am 掩码 [1,N] 广播会把 Sv/SlT 抬成 3 维，压回 2 维
            Pv.append(Sv.reshape(Nv, -1).softmax(-1))
            Pl.append(SlT.reshape(Nl, -1).softmax(-1))
        base = node['module']
        for h in range(H):
            self.push_blk(f'P_v:{base}#{node["seq"]}#{h}',
                          q_round(Pv[h], 1 / 127), c16(Nv))
            self.push_blk(f'P_l:{base}#{node["seq"]}#{h}',
                          q_round(Pl[h], 1 / 127), c16(Nl))
            z1 = torch.zeros(c16(d) * 16, Nlp, dtype=torch.int8)
            z2 = torch.zeros(c16(d) * 16, Nvp, dtype=torch.int8)
            z1[:d, :Nl] = q_round(Vl[0, :, h].t().contiguous(),
                                  meta['sigma_vs'][0])
            z2[:d, :Nv] = q_round(Vv[0, :, h].t().contiguous(),
                                  meta['sigma_vs'][1])
            self.blk[f'VT_v:{base}#{node["seq"]}#{h}'].append(z1.numpy().copy())
            self.blk[f'VT_l:{base}#{node["seq"]}#{h}'].append(z2.numpy().copy())
        Yv = self.must_assemble(node['out_graphs'][0]).view(bsz, Nv, -1)
        Yl = self.must_assemble(node['out_graphs'][1]).view(bsz, Nl, -1)
        self.reg(rec['out_ids'][0], Yv)
        self.reg(rec['out_ids'][1], Yl)
        self._advance(f'{node["module"]}.out_v_proj')
        self._advance(f'{node["module"]}.out_l_proj')
        return Yv, Yl

    # ---------- WindowMSA ----------
    def fwd_swin(self, mod, node, rec, args, kw):
        x = _g(args, kw, 0, 'x')
        mask = _g(args, kw, 1, 'mask')
        nWB, T, C = x.shape
        meta = self.ameta[(node['module'], node['seq'])]
        H, d, scale = meta['H'], meta['d'], mod.scale
        W_tot = nWB
        PW = W_tot * H
        T16 = c16(T)
        Tp = T16 * 16
        qkvn = self._member_node(f'{node["module"]}.qkv')
        self.reg(qkvn['in_graph'], x)
        self._run_member(qkvn)
        self._advance(qkvn['module'])
        Y = self.must_assemble(qkvn['out_graph'])  # [nWB*T, 3C]
        qkv = Y.view(W_tot, T, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]            # (W, H, T, d)
        Qfull = torch.zeros(PW * T16 * 16, d)
        Kfull = torch.zeros(PW * T16 * 16, d)
        VTfull = torch.zeros(PW * d, Tp)
        for pw in range(PW):
            w, h = divmod(pw, H)
            Qfull[pw * T16 * 16:pw * T16 * 16 + T] = q[w, h]
            Kfull[pw * T16 * 16:pw * T16 * 16 + T] = k[w, h]
            VTfull[pw * d:pw * d + d, :T] = v[w, h].t()
        sg = meta['sigma_q']
        self.push_batched(f'{qkvn["out_graph"]}#Q', q_round(Qfull, sg).numpy())
        self.push_batched(f'{qkvn["out_graph"]}#K', q_round(Kfull, sg).numpy())
        # VT 每批真实行数 = nb*d（d 可能不是 16 倍数），编译器按
        # pad16(nb*d) 收块：从 P 条目反推每批 nb，逐批补零
        VTq = q_round(VTfull, sg).numpy()
        vname = f'{qkvn["out_graph"]}#VT'
        pes = [e for _, e in self.rents.get(
            f'P:{node["module"]}#{node["seq"]}', ())]
        VTp = VTq
        if pes:
            nbs = [e['rows16'] // T16 for e in pes]
            total = sum(-(-nb * d // 16) * 16 for nb in nbs)
            if total != VTq.shape[0]:
                VTp = np.zeros((total, VTq.shape[1]), dtype=VTq.dtype)
                src = dst = 0
                for nb in nbs:
                    r = -(-nb * d // 16) * 16
                    VTp[dst:dst + nb * d] = VTq[src:src + nb * d]
                    src += nb * d
                    dst += r
                if src != VTq.shape[0]:
                    print(f'[warn] {vname}: VT 行数 {src} ≠ {VTq.shape[0]}')
        self.push_batched(vname, VTp)
        rpb = mod.relative_position_bias_table[
            mod.relative_position_index.view(-1)].view(T, T, -1).permute(2, 0, 1)
        Sname = f'S:{node["module"]}#{node["seq"]}'
        self.run_graph(Sname)
        Pfull = torch.zeros(PW * T16 * 16, T)
        for pw in range(PW):
            w, h = divmod(pw, H)
            S = self.deq(self.blk[Sname].pop(0), meta['sigma_s'], T)[:, :T]
            att = S * scale + rpb[h]
            if mask is not None:
                # mask 只有单张图的窗数（nW），backbone_3d 多帧 batch 时
                # 全局窗号要对 nW 取模（B=1 时退化为 mask[w]）
                att = att + mask[w % mask.shape[0]]
            Pfull[pw * T16 * 16:pw * T16 * 16 + T] = att.softmax(-1)
        self.push_batched(f'P:{node["module"]}#{node["seq"]}',
                          q_round(Pfull, 1.0 / 127).numpy())
        # proj 偏置：aug 层已折进 PL（常数列 + 权重末行）；host_bias 层
        # （v2 表 24 个 proj 里 14 个 fp_fallback）的 fp 偏置按编译契约
        # 由 host 在 Y 块反量化后补上——Y 块路径不走 assemble，2026-08-31
        # 之前漏了这步，swin 输出整体丢偏置
        pk = f'{node["module"]}.proj'
        proj_bias = None
        if not self.calib.get(pk, {}).get('bias_aug_c'):
            proj_bias = self.get_bias(pk + '.weight')
            if proj_bias is None and pk not in self.warned:
                print(f'[warn] {pk}: 无 bias_aug_c 且模型无 bias 参数')
                self.warned.add(pk)
        # swin 的 proj 融合在注意力段里，段输出是 Y 块（无 act 图），
        # 所以按 Y 输出名触发，而不是 out_graphs[0]
        self.run_graph(f'Y:{node["module"]}#{node["seq"]}')
        yb = self.pop_blks(f'Y:{node["module"]}#{node["seq"]}', W_tot)
        if len(yb) < W_tot:
            miss = self.missing[-6:] if self.missing else []
            raise RuntimeError(f'swin Y 块不足 {len(yb)}/{W_tot} '
                               f'missing={miss!r}')
        outs = torch.stack([self.deq(b, meta['sigma_os'][0], T) for b in yb])
        if proj_bias is not None:
            outs = outs + proj_bias.view(1, 1, -1)
        out = outs.reshape(nWB, T, C)
        self.reg(rec['out_ids'][0], out)
        self._advance(f'{node["module"]}.proj')
        return out

    # ---------- BertAttention ----------
    def fwd_bert(self, mod, node, rec, args, kw):
        hidden = _g(args, kw, 0, 'hidden_states')
        am = _g(args, kw, 1, 'attention_mask')
        if am is not None and 'bertmask' not in self.warned:
            bad = ((am.dtype == torch.bool and bool((~am).any())) or
                   (am.dtype != torch.bool and float(am.min()) < -1e-6))
            if bad:
                print('[warn] BERT attention_mask 非全通：PL softmax 无 mask，'
                      '数值有偏差')
                self.warned.add('bertmask')
        # X 图名从段的 act_in 条目取（q/k/v 记录已并入段内）
        xname = None
        for s in node['segs']:
            for e in self.mans[s]['inputs']:
                if e.get('kind') == 'act_in':
                    xname = e['name']
                    break
            if xname:
                break
        if xname is None:
            raise RuntimeError(f'{node["module"]}: 段里找不到 X act_in')
        if not isinstance(xname, str) or not xname.startswith(('t', 'in:')):
            xname = str(xname)
        T = hidden.shape[-2] if hidden.dim() == 3 else hidden.shape[0]
        self.tensor_reg[xname] = hidden.reshape(T, -1) if hidden.dim() > 2 \
            else hidden
        for og in node['out_graphs']:
            self.run_graph(og)
        Y = self.assemble(node['out_graphs'][0])
        if Y is None:
            raise RuntimeError(f'{node["out_graphs"][0]}: 段失败')
        out = mod.output.LayerNorm(Y.view(T, -1) + hidden.reshape(T, -1))
        out = out.view(hidden.shape)
        self.reg(rec['out_ids'][0], out)
        self._advance(f'{node["module"]}.output.dense')
        return out, None

    # ================= 补丁 =================
    def patch(self, only=None):
        """only: None=全部 PL；{'gemm'} 只 PL 化 GEMM（attn 原生 fp）；
        {'attn'} 只 PL 化注意力（GEMM 原生 fp）。被跳过的模块仍挂
        host hook 登记输出，保证下游段的 act_in 供给（二分定位用）。"""
        FAMFN = {'RotaryAttention': self.fwd_rotary,
                 'JointGraphAttention': self.fwd_jg,
                 'TemporalJointGraphAttention': self.fwd_temporal,
                 'MultiheadAttention': self.fwd_mha,
                 'BiMultiHeadAttention': self.fwd_bimha,
                 'WindowMSA': self.fwd_swin,
                 'BertAttention': self.fwd_bert}
        mods = dict(self.model.named_modules())
        for name, recs in sorted(self.recs_by_mod.items()):
            mod = mods.get(name)
            if mod is None:
                continue
            kinds = {self.node_by.get((r['module'], r['seq']), {}).get('kind')
                     for r in recs}
            kinds.discard(None)
            if not kinds:
                # 全丢弃的原生模块：也要挂 hook 登记输出张量——融合段
                # （如 bimha PV + MSDA GEMM 同段）的 act_in 需要它们的输出
                self.stats['native'] += 1
                self._patch_host(mod, name)
                continue
            if 'attn' in kinds:
                if only and 'attn' not in only:
                    self._patch_host(mod, name)
                    continue
                fam = recs[0]['cls']
                if fam not in FAMFN:
                    print(f'[warn] {name}: 家族 {fam} 无定制 forward，原生')
                    continue
                self._patch_attn(mod, name, FAMFN[fam])
            elif 'gemm' in kinds:
                if only and 'gemm' not in only:
                    self._patch_host(mod, name)
                    continue
                self._patch_gemm(mod, name)
            else:
                self._patch_host(mod, name)
        print(f'[patch] gemm={self.stats["patch_gemm"]} '
              f'attn={self.stats["patch_attn"]} host={self.stats["patch_host"]} '
              f'native={self.stats["native"]}')

    def _patch_gemm(self, mod, name):
        self.stats['patch_gemm'] += 1
        drv = self

        def fwd(module, x):
            i = drv.cursor[name]
            recs = drv.recs_by_mod[name]
            if i >= len(recs):
                raise RuntimeError(f'{name}: 调用次数超 trace')
            rec = recs[i]
            drv.cursor[name] = i + 1
            if drv.fp_after is not None and drv.gidx >= drv.fp_after:
                drv.gidx += 1
                out = module._hb_orig_fwd(x)
                drv.reg(rec['out_ids'][0], out)
                return out
            if drv.fp_after is not None:
                print(f'[bisect] #{drv.gidx} gemm {name}', flush=True)
            drv.gidx += 1
            node = drv.node_by.get((rec['module'], rec['seq']))
            if node is None:
                return module._hb_orig_fwd(x)
            return drv.gemm_call(module, node, rec, x)
        mod._hb_orig_fwd = mod.forward
        mod.forward = types.MethodType(fwd, mod)

    def _patch_attn(self, mod, name, fn):
        self.stats['patch_attn'] += 1
        drv = self

        def fwd(_self, *args, **kw):
            i = drv.cursor[name]
            recs = drv.recs_by_mod[name]
            if i >= len(recs):
                raise RuntimeError(f'{name}: 调用次数超 trace')
            rec = recs[i]
            drv.cursor[name] = i + 1
            if drv.fp_after is not None and drv.gidx >= drv.fp_after:
                drv.gidx += 1
                out = mod._hb_orig_fwd(*args, **kw)
                outs = out if isinstance(out, (tuple, list)) else (out,)
                ti = 0
                for o in outs:
                    if torch.is_tensor(o) and ti < len(rec['out_ids']):
                        drv.reg(rec['out_ids'][ti], o)
                        ti += 1
                return out
            if drv.fp_after is not None:
                print(f'[bisect] #{drv.gidx} attn {name}', flush=True)
            drv.gidx += 1
            node = drv.node_by.get((rec['module'], rec['seq']))
            if node is None:
                return mod._hb_orig_fwd(*args, **kw)
            return fn(mod, node, rec, args, kw)
        mod._hb_orig_fwd = mod.forward
        mod.forward = types.MethodType(fwd, mod)

    def _patch_host(self, mod, name):
        self.stats['patch_host'] += 1
        drv = self

        def hook(module, inputs, output):
            i = drv.cursor[name]
            recs = drv.recs_by_mod[name]
            if i >= len(recs):
                return
            rec = recs[i]
            drv.cursor[name] = i + 1
            outs = output if isinstance(output, (tuple, list)) else (output,)
            ti = 0
            for o in outs:
                if torch.is_tensor(o) and ti < len(rec['out_ids']):
                    drv.reg(rec['out_ids'][ti], o)
                    ti += 1
        mod.register_forward_hook(hook)


def run_sample(args, sample_id):
    HB = '~/workspace/holobrain'
    if HB not in sys.path:
        sys.path.insert(0, HB)
    import bringup  # noqa: F401
    from robo_orchard_lab.models.holobrain.processor import HoloBrainProcessor
    from robo_orchard_lab.models.mixin import ModelMixin
    import compiler

    P = compiler.PROFILES['full']
    seed = 20260830
    if args.ref and os.path.exists(args.ref):
        try:
            seed = int(np.load(args.ref)['seed'])
        except Exception:
            pass
    torch.manual_seed(seed)
    processor = HoloBrainProcessor.load(bringup.CKPT, bringup.PROCESSOR_JSON)
    model = ModelMixin.load_model(bringup.MODEL_DIR, load_impl='native')
    model = model.float().eval()
    batch = torch.load(args.batch, map_location='cpu', weights_only=False)
    _fix_kinematics_device(batch)
    drv = HostDriver(args.build, args.trace, args.calib, model,
                     engine=getattr(args, 'engine', 'fast'),
                     fp_after=getattr(args, 'fp_after', None))
    drv.P = P
    t0 = time.perf_counter()
    drv.patch()
    t1 = time.perf_counter()
    with torch.no_grad():
        outs = model(batch)
    pa = outs[0]['pred_actions'].detach().cpu().numpy()   # [1, 64, 14, 8]
    t2 = time.perf_counter()
    act = pa[0, :, :, 0]                                   # [64, 14]
    np.savez(args.out, action=act, pred_actions_raw=pa[0])
    print(f'[run] patch={t1 - t0:.1f}s forward={t2 - t1:.1f}s '
          f'segments={drv.stats["segments"]}/{len(drv.seg_names)} '
          f'gemm_calls={drv.stats["gemm_calls"]}')
    unrun = [s for s in drv.seg_names if s not in drv.done]
    if unrun:
        print(f'[warn] 未跑段 {len(unrun)}: {unrun[:8]}')
    if drv.missing:
        print(f'[warn] 缺输入 {len(drv.missing)}: {drv.missing[:5]}')
    report = dict(sample=sample_id, action_shape=list(act.shape),
                  segments_run=drv.stats['segments'],
                  segments_total=len(drv.seg_names),
                  patch_s=round(t1 - t0, 2), forward_s=round(t2 - t1, 2),
                  wall_s=round(t2 - t0, 2))
    if args.ref and os.path.exists(args.ref):
        ref = np.load(args.ref)
        ra = ref['action']
        d = np.abs(act - ra)
        report.update(
            jpos_mae=float(d.mean()), jpos_max=float(d.max()),
            jpos_mae_arm12=float(d[:, :12].mean()),
            jpos_mae_gripper=float(d[:, [6, 13]].mean()),
            per_joint_mae=[float(d[:, j].mean()) for j in range(d.shape[1])],
            ref_seed=int(ref['seed']) if 'seed' in ref.files else None)
        print(f'[vs-ref] jpos MAE={d.mean():.4f} max={d.max():.4f} '
              f'arm12={d[:, :12].mean():.4f} '
              f'gripper(6,13)={d[:, [6, 13]].mean():.4f}')
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', required=True)
    ap.add_argument('--trace', required=True)
    ap.add_argument('--batch', required=True)
    ap.add_argument('--ref', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--calib', default=None)
    ap.add_argument('--engine', default='fast', choices=['fast', 'rtl'])
    ap.add_argument('--sample-id', default='000')
    ap.add_argument('--fp-after', type=int, default=None,
                    help='深度二分：前 N 次量化调用走 PL，其后全部原生 fp')
    args = ap.parse_args()
    if not args.calib:
        for cand in (os.path.join(HERE, 'hw_calib_table_v2.json'),
                     '/tmp/ae_hostdrv/hw_calib_table_v2.json',
                     os.path.abspath(os.path.join(HERE, '..', '..',
                                                  '02_quant',
                                                  'hw_calib_table_v2.json'))):
            if os.path.exists(cand):
                args.calib = cand
                break
    rep = run_sample(args, args.sample_id)
    jf = args.out.replace('.npz', '_vs_fp32.json')
    with open(jf, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f'[done] {args.out} + {jf}')


if __name__ == '__main__':
    main()
