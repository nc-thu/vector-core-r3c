# -*- coding: utf-8 -*-
"""smoke_machinery.py — machinery 冒烟：迷你图 → 编译 → iverilog 逐段位精确对拍

不验证模型数值，只验证"编译器 ↔ RTL ↔ 黄金解释器"这套机器本身：
  描述符编码 / CTX·WRAM 寻址 / DMA 装载回写 / COPY 转置 / GEMM 普通+转置写回 /
  OP_ATTN_S softmax / 零槽预清零 / 半区交替 / 多 tile·多列组·chunk 拆段。
迷你图刻意覆盖：
  conv(im2col+aug) / aug / 无 bias / fp_fallback(host_bias) / k_eff>半区(全 WRAM) /
  pad 行预清零 / BertAttention(OP_ATTN_S 直通) / WindowMSA(键列组+每窗现载 proj) /
  ShiftWindowMSA 壳 / 豁免层整层 host / 容器壳跳过。
黄金 vs RTL 逐字节比对整个 DDR 终态——CTX 有任何被读未写的格子，RTL 会漏 X
出来，立刻暴露。

用法：python smoke_machinery.py [--keep] [--only N] [--skip-sim]
产物：build_smoke_mini/（迷你输入）  run_smoke/（编译输出+TB 运行目录）
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))      # …/vector_core_sim
HW_SIM = os.path.join(ROOT, 'hw_zcu104', 'sim')
HW_RTL = os.path.join(ROOT, 'hw_zcu104', 'rtl')
# 文件清单照抄 hw_zcu104/sim/regression.sh 的 sim_ae 行（不用通配，防卷进实验 RTL）
RTL_FILES = ['ae_pkg.sv', 'ae_dpram.sv', 'ae_ctx_ram.sv', 'ae_pe.sv',
             'ae_sysarr.sv', 'ae_requant.sv', 'rq_v2.sv', 'rq_ms.sv',
             'ae_exp_lut.sv', 'ae_gemm.sv', 'ae_softmax.sv', 'ae_copy.sv',
             'ae_dma.sv', 'ae_sched.sv', 'ae_core.sv', 'ae_top.sv']

MINI_DIR = os.path.join(HERE, 'build_smoke_mini')
RUN_DIR = os.path.join(HERE, 'run_smoke')

# Icarus Verilog 位置（Windows 本机装在 C:\iverilog）
IVL_BIN = r'C:\iverilog\bin'
IVL = os.path.join(IVL_BIN, 'iverilog.exe')
VVP = os.path.join(IVL_BIN, 'vvp.exe')


def _env():
    return {**os.environ,
            'PATH': IVL_BIN + os.pathsep + os.environ.get('PATH', '')}


def rng_of(name):
    return np.random.default_rng(zlib.crc32(name.encode('utf-8')) & 0xFFFFFFFF)


# ---------------------------------------------------------------------------
# 1. 迷你资产
# ---------------------------------------------------------------------------
def _rec(seq, module, cls, op, in_shapes, out_shapes, in_ids, out_ids,
         **kw):
    r = dict(seq=seq, module=module, cls=cls, op=op, in_shapes=in_shapes,
             out_shapes=out_shapes, in_ids=in_ids, out_ids=out_ids)
    r.update(kw)
    return r


def build_mini():
    ops = []
    nid = [1000]

    def ids(n=1):
        nid[0] += 7
        return [nid[0] + i for i in range(n)]

    def g(seq, module, cls, in_s, out_s, wkey, wshape, bias):
        ops.append(_rec(seq, module, cls, 'gemm', [in_s], [out_s],
                        ids(), ids(), weight_key=wkey, w_shape=wshape,
                        has_bias=bias))

    def e(seq, module, cls, in_s):
        ops.append(_rec(seq, module, cls, 'elem_norm', [in_s], [in_s],
                        ids(), ids()))

    ops.append(_rec(0, 'mini.pre', 'MiniPre', 'custom', [[8]], [[8]],
                    ids(), ids()))
    # conv：m=64 k=27 n=6，aug（im2col 由 host 出图）
    g(1, 'mini.cnn', 'Conv2d', [1, 3, 10, 10], [6, 8, 8],
      'mini.cnn.weight', [6, 3, 3, 3], True)
    e(2, 'mini.act0', 'SiLU', [64, 6])
    g(3, 'mini.fc0', 'Linear', [8], [8, 12], 'mini.fc0.weight',
      [12, 8], True)                       # aug c=3
    e(4, 'mini.act1', 'SiLU', [8, 12])
    g(5, 'mini.fc1', 'Linear', [24], [24, 12], 'mini.fc1.weight',
      [12, 12], False)                     # m%16=8 → pad 行预清零
    e(6, 'mini.act2', 'GELU', [24, 12])
    g(7, 'mini.fc2', 'Linear', [8], [8, 12], 'mini.fc2.weight',
      [12, 24], True)                      # fp_fallback → host_bias
    ops.append(_rec(8, 'mini.up', 'Upsample', 'custom', [[8, 12]], [[8, 40]],
                    ids(), ids()))
    g(9, 'mini.fc3', 'Linear', [8], [8, 12], 'mini.fc3.weight',
      [12, 40], False)                     # k_eff=41>32 → 全 WRAM 模式
    e(10, 'mini.norm0', 'RMSNorm', [8, 12])
    # ---- 迷你 BERT（children 先、attention summary 后）----
    B = 'mini.bert.attention'
    g(11, f'{B}.self.query', 'Linear', [8], [8, 16],
      f'{B}.self.query.weight', [16, 16], True)
    g(12, f'{B}.self.key', 'Linear', [8], [8, 16],
      f'{B}.self.key.weight', [16, 16], True)
    g(13, f'{B}.self.value', 'Linear', [8], [8, 16],
      f'{B}.self.value.weight', [16, 16], True)
    g(14, f'{B}.output.dense', 'Linear', [8], [8, 16],
      f'{B}.output.dense.weight', [16, 16], True)
    e(15, f'{B}.output.LayerNorm', 'LayerNorm', [8, 16])
    ops.append(_rec(16, B, 'BertAttention', 'attn', [[1, 8, 16]], [[1, 8, 16]],
                    ids(2), ids(),
                    kwargs_shapes={'attention_mask': [1, 1, 8, 8]}))
    e(17, 'mini.bert.output.LayerNorm', 'LayerNorm', [8, 16])
    # ---- 迷你 swin（4 窗 × 2 头 × 窗宽 13，C=16；13>COLS=12 → 2 键列组）----
    W = 'mini.blk.attn.w_msa'
    g(18, f'{W}.qkv', 'Linear', [52], [52, 48], f'{W}.qkv.weight',
      [48, 16], True)
    e(19, f'{W}.softmax', 'Softmax', [4, 2, 13, 13])
    g(20, f'{W}.proj', 'Linear', [52], [52, 16], f'{W}.proj.weight',
      [16, 16], True)
    ops.append(_rec(21, W, 'WindowMSA', 'attn', [[52, 16]], [[52, 16]],
                    ids(), ids(), kwargs_shapes={'mask': 'NoneType'}))
    ops.append(_rec(22, 'mini.blk.attn', 'ShiftWindowMSA', 'attn',
                    [[4, 13, 16], [2, 3]], [[4, 13, 16]], ids(2), ids()))
    e(23, 'mini.blk.norm2', 'LayerNorm', [52, 16])
    ops.append(_rec(24, 'mini.blk', 'SwinBlock', 'custom', [[52, 16]],
                    [[52, 16]], ids(), ids()))          # 容器壳 → 跳过
    g(25, 'spatial_enhancer.pts_prob_fc.layers.1', 'Linear', [14],
      [14, 16], 'spatial_enhancer.pts_prob_fc.layers.1.weight',
      [16, 32], True)                                   # 豁免 → 整层 host

    trace = dict(meta=dict(model='mini', note='machinery 冒烟'), ops=ops)

    # 权重（全 int8 随机，确定性）
    wdefs = [
        ('mini.cnn.weight', [6, 3, 3, 3]),
        ('mini.fc0.weight', [12, 8]),
        ('mini.fc1.weight', [12, 12]),
        ('mini.fc2.weight', [12, 24]),
        ('mini.fc3.weight', [12, 40]),
        (f'{B}.self.query.weight', [16, 16]),
        (f'{B}.self.key.weight', [16, 16]),
        (f'{B}.self.value.weight', [16, 16]),
        (f'{B}.output.dense.weight', [16, 16]),
        (f'{W}.qkv.weight', [48, 16]),
        (f'{W}.proj.weight', [16, 16]),
    ]
    tensors, w8dir = [], os.path.join(MINI_DIR, 'w8_mini')
    os.makedirs(w8dir, exist_ok=True)
    for i, (key, shape) in enumerate(wdefs):
        fn = f'w{i:02d}.bin'
        rng_of(key).integers(-128, 128, size=shape,
                             dtype=np.int8).tofile(os.path.join(w8dir, fn))
        tensors.append(dict(key=key, file=fn, shape=shape, dtype='int8',
                            extra=False))
    manifest = dict(tensors=tensors)

    # 校准（全部 s_shift=8：machinery 位精确档；数值含义不做要求）
    def wbias(key, n):
        return rng_of('bias:' + key).integers(-24, 24, size=n,
                                              dtype=np.int8).tolist()

    def ent(m, c=None, n=None, fp=False):
        d = dict(sa=1.0, sw=1.0, so=1.0, m_requant=m, s_shift=8,
                 bias_fp_fallback=fp, bias_aug_c=c,
                 w_bias_int8=(wbias(ent_i, n) if (c is not None and n)
                              else None))
        return d

    ent_i = ''      # wbias 的种子键占位
    cal = {}
    for key, m, c, n, fp in [
            ('mini.cnn', 220, 2, 6, False),
            ('mini.fc0', 180, 3, 12, False),
            ('mini.fc1', 600, None, None, False),
            ('mini.fc2', 300, None, None, True),
            ('mini.fc3', 90, None, None, False),
            (f'{B}.self.query', 140, 2, 16, False),
            (f'{B}.self.key', 150, -1, 16, False),
            (f'{B}.self.value', 160, 1, 16, False),
            (f'{B}.output.dense', 120, 2, 16, False),
            (f'{W}.qkv', 260, -1, 48, False),
            (f'{W}.proj', 110, 1, 16, False)]:
        ent_i = key
        cal[key] = ent(m, c, n, fp)

    with open(os.path.join(MINI_DIR, 'mini_trace.json'), 'w',
              encoding='utf-8') as f:
        json.dump(trace, f, indent=1)
    with open(os.path.join(MINI_DIR, 'mini_manifest.json'), 'w',
              encoding='utf-8') as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(MINI_DIR, 'mini_calib.json'), 'w',
              encoding='utf-8') as f:
        json.dump(dict(gemms=cal), f, indent=1)
    return trace, manifest, dict(gemms=cal), w8dir


# ---------------------------------------------------------------------------
# 2. 编译（import compiler，与 main() 同入口）
# ---------------------------------------------------------------------------
def do_compile(trace, manifest, w8dir, calib):
    sys.path.insert(0, HERE)
    import compiler as C
    C.HEADS['BertAttention'] = 2          # 迷你 BERT：C=16 ÷ 2 头 = d 8
    P = dict(C.PROFILES['smoke'])
    c = C.Compiler(trace, manifest, w8dir, calib, P, RUN_DIR,
                   rq_max_s=8)   # 对拍目标 = 现网 hw_zcu104 RTL（rq_v2 T_MAX=0）
    c.build_ir()
    c.lower()
    return c, c.emit(), P


# ---------------------------------------------------------------------------
# 3. iverilog 编一次 + 逐段跑
# ---------------------------------------------------------------------------
def build_sim():
    sim_dir = os.path.join(RUN_DIR, 'sim')
    os.makedirs(sim_dir, exist_ok=True)
    srcs = [os.path.join(HW_RTL, f) for f in RTL_FILES]
    vvp = os.path.join(sim_dir, 'runner.vvp')
    # 注意：iverilog 的 -P 覆盖必须放在源文件列表之前
    cmd = ([IVL, '-g2012', '-o', vvp, '-I', HW_RTL,
            '-Psegment_runner.COLS=12', '-Psegment_runner.CTX_WORDS=2048',
            '-Psegment_runner.W_WORDS=64', '-Psegment_runner.SEQ_N=64',
            '-Psegment_runner.DDR_BYTES=65536'] + srcs +
           [os.path.join(HERE, 'segment_runner.sv')])
    r = subprocess.run(cmd, capture_output=True, text=True, env=_env())
    if r.returncode:
        print(r.stdout + r.stderr)
        raise SystemExit('iverilog 编译失败')
    return vvp


def act_data_for(seg_dir):
    import golden_interp as G
    man = json.load(open(os.path.join(seg_dir, 'manifest.json'),
                         encoding='utf-8'))
    act = {}
    for e in man['inputs']:
        if e.get('kind') == 'const':
            act[e['name']] = np.full(16, e.get('cval', 0), dtype=np.int8)
        else:
            act[e['name']] = rng_of(e['name']).integers(
                -128, 128, size=e['words'] * 16, dtype=np.int8)
    return act


def run_segment_rt(vvp, seg_dir, tag):
    import golden_interp as G
    man = json.load(open(os.path.join(seg_dir, 'manifest.json'),
                         encoding='utf-8'))
    P = man['profile']
    act = act_data_for(seg_dir)
    img = G.build_ddr_image(seg_dir, os.path.join(RUN_DIR,
                                                  'weights_blob.bin'),
                            act, P)
    tmp = os.path.join(seg_dir, 'rt')
    os.makedirs(tmp, exist_ok=True)
    G.write_mem(os.path.join(tmp, 'ddr_init.mem'), img)
    shutil.copy(os.path.join(seg_dir, 'seq.mem'),
                os.path.join(tmp, 'seq.mem'))
    shutil.copy(os.path.join(HW_SIM, 'exp2_lut.mem'),
                os.path.join(tmp, 'exp2_lut.mem'))
    r = None
    try:
        r = subprocess.run([VVP, os.path.abspath(vvp), '+SEQ=seq.mem',
                            '+DDRIMG=ddr_init.mem', '+DUMP=ddr_dump.mem',
                            '+WDOG=8000000'],
                           cwd=tmp, capture_output=True, text=True, env=_env(),
                           timeout=180)
    except subprocess.TimeoutExpired:
        return False, 'vvp 超时 180s（段跑不完：描述符挂起或看门狗前未 done）', None
    log = r.stdout + r.stderr
    if r.returncode:
        return False, f'vvp 崩溃 rc={r.returncode}\n{log[-800:]}', None
    try:
        dump = G.read_dump(os.path.join(tmp, 'ddr_dump.mem'))
    except ValueError as ex:
        return False, f'dump 有 X（未初始化 CTX 泄漏）：{ex}\n{log[-500:]}', \
            None
    seq = G.load_seq(seg_dir)
    _, gddr, info = G.run_segment(seq, img, P)
    ok, msg = G.compare(gddr, dump)
    cyc = None
    for line in log.splitlines():
        if '[runner] done' in line:
            parts = dict(p.split('=') for p in line.split()[2:])
            cyc = int(parts['cycles'])
    return ok, msg, cyc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', type=int, default=-1,
                    help='只跑第 N 段（调试）')
    ap.add_argument('--skip-sim', action='store_true',
                    help='只编译迷你图，不跑 iverilog')
    ap.add_argument('--keep', action='store_true', default=True)
    a = ap.parse_args()

    for d in (MINI_DIR, RUN_DIR):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    trace, manifest, cal, w8dir = build_mini()
    c, st, P = do_compile(trace, manifest, w8dir, cal)
    print(f'[smoke] 编译: 段={st["segments"]} 描述符={st["d_total"]} '
          f'maxSEQ={st["max_seq"]} blob={st["blob_bytes"]}B '
          f'host_steps={st["host_steps"]}')
    print(f'[smoke] aug={st.get("aug_layers")} '
          f'host_bias={st["host_bias_total"]} '
          f'豁免={st.get("exempt_host_gemm")} '
          f'OP_ATTN_S={st.get("attn_op_attn_s", 0)} '
          f'swin={st.get("attn_2ph_swin", 0)}')
    if a.skip_sim:
        return
    vvp = build_sim()
    print(f'[smoke] iverilog 编译 OK → {os.path.basename(vvp)}')

    seg_names = sorted(os.listdir(os.path.join(RUN_DIR, 'segments')))
    n_pass = 0
    worst = []
    for i, name in enumerate(seg_names):
        if a.only >= 0 and i != a.only:
            continue
        seg_dir = os.path.join(RUN_DIR, 'segments', name)
        ok, msg, cyc = run_segment_rt(vvp, seg_dir, name)
        man = json.load(open(os.path.join(seg_dir, 'manifest.json'),
                             encoding='utf-8'))
        if ok:
            n_pass += 1
            print(f'  PASS {name}  descs={man["n_descs"]:3d} '
                  f'cyc={cyc} est={man["est_cycles"]}')
        else:
            print(f'  FAIL {name}  descs={man["n_descs"]}')
            print(f'       {msg}')
        if cyc:
            worst.append((cyc, man['est_cycles'], name))
    print(f'[smoke] 对拍: {n_pass}/{len(seg_names)} 段逐字节一致')
    if n_pass == len(seg_names):
        print('[smoke] MACHINERY SMOKE ALL PASS')


if __name__ == '__main__':
    main()
