# run_segs.py — R1 代表段验证：RTL vs fast_interp_a3.py
# 取含 STORE+GEMM 重叠的段，构建 DDR 初始映像，跑 iverilog，
# 比对位精确 DDR + 周期数 vs cycle_exact_a3 模型
import json, os, sys, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
A3   = os.path.join(os.path.dirname(os.path.dirname(HERE)), '12_actv', 'a3')
sys.path.insert(0, A3)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)),
                               '09_cbound'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)),
                               '12_actv', 'spec'))

from fast_interp_a3 import run_segment_fast_a3
from golden_interp import decode, load_seq
from cycle_exact_a3 import seg_exact, seg_account
from acct_a3 import CAL

BLOB = os.path.join(A3, 'build_a3', 'weights_blob.bin')
SEGS_DIR = os.path.join(A3, 'build_a3', 'segments')

# 选 3 个代表段：含 STORE->GEMM 重叠
TEST_SEGS = ['seg_0636', 'seg_0642', 'seg_0643']
COLS = 108
CTX_WORDS = 4096
W_WORDS = 4096

def build_ddr_image(seg_dir):
    """从 manifest 构建 DDR 初始映像（权重从 blob，输入用合成数据）"""
    man = json.load(open(os.path.join(seg_dir, 'manifest.json')))
    blob = np.fromfile(BLOB, dtype=np.uint8)

    max_ddr = 0
    for inp in man['inputs']:
        max_ddr = max(max_ddr, inp['ddr'] + inp['words'])
    for w in man['weights']:
        max_ddr = max(max_ddr, w['ddr'] + w['blob_len'])
    for out in man['outputs']:
        max_ddr = max(max_ddr, out['ddr'] + out['words'])

    # 512KB DDR
    ddr_size = max(max_ddr + 256, 524288)
    ddr = np.zeros(ddr_size, dtype=np.uint8)

    # 装载权重
    for w in man['weights']:
        ddr[w['ddr']:w['ddr']+w['blob_len']] = \
            blob[w['blob_off']:w['blob_off']+w['blob_len']]

    # 装载输入（合成数据：用确定性伪随机填充，方便对拍）
    rng = np.random.RandomState(42)
    for inp in man['inputs']:
        n = inp['words']
        # 生成有符号 int8 随机数据
        data = rng.randint(-128, 128, size=n).astype(np.int8).view(np.uint8)
        ddr[inp['ddr']:inp['ddr']+n] = data

    return ddr, max_ddr

def write_mem(path, arr):
    """numpy uint8 数组 → hex mem 文件"""
    with open(path, 'w') as f:
        for b in arr:
            f.write('%02X\n' % int(b))

def read_mem(path, n):
    """hex mem → numpy uint8"""
    lines = open(path).read().strip().split('\n')
    return np.array([int(l, 16) for l in lines[:n]], dtype=np.uint8)

def run_seg(seg_name):
    seg_dir = os.path.join(SEGS_DIR, seg_name)
    print('\n=== %s ===' % seg_name)

    # 1) 构建 DDR 映像
    ddr, max_ddr = build_ddr_image(seg_dir)
    print('  DDR size: %d bytes (max offset: %d)' % (len(ddr), max_ddr))

    # 2) 拷贝 seq.mem
    import shutil
    shutil.copy(os.path.join(seg_dir, 'seq.mem'),
               os.path.join(HERE, 'seq.mem'))
    write_mem(os.path.join(HERE, 'ddr_init.mem'), ddr)

    # 3) 跑 fast_interp_a3（黄金参考）
    seq = load_seq(seg_dir)
    P = {'COLS': COLS, 'CTX_WORDS': CTX_WORDS, 'W_WORDS': W_WORDS}
    ddr_gold = ddr.copy()
    run_segment_fast_a3(seq, ddr_gold, P)
    # 只比较 output 区域
    man = json.load(open(os.path.join(seg_dir, 'manifest.json')))
    out_regions = [(o['ddr'], o['words']) for o in man['outputs']]

    # 4) 跑 iverilog RTL
    rtl = '../rtl/ae_pkg.sv ../rtl/ae_dpram.sv ../rtl/ae_ctx_ram.sv ' \
          '../rtl/ae_pe.sv ../rtl/ae_sysarr.sv ../rtl/ae_requant.sv ' \
          '../rtl/rq_v2.sv ../rtl/rq_ms.sv ../rtl/ae_exp_lut.sv ' \
          '../rtl/ae_gemm.sv ../rtl/ae_softmax.sv ../rtl/ae_copy.sv ' \
          '../rtl/ae_dma.sv ../rtl/ae_actv.sv ../rtl/ae_sched.sv ' \
          '../rtl/ae_core.sv ../rtl/ae_top.sv'
    cmd = 'iverilog -g2012 -o /tmp/seg_r1.vvp -I ../rtl %s tb_ae_seg.sv' % rtl
    r = subprocess.run(cmd, shell=True, cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        print('  COMPILE FAIL:', r.stderr[:500])
        return False

    r = subprocess.run('vvp /tmp/seg_r1.vvp', shell=True, cwd=HERE,
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print('  SIM FAIL:', r.stderr[:500])
        return False

    # 解析周期
    for line in r.stdout.split('\n'):
        if '[seg]' in line:
            print('  RTL:', line.strip())
            parts = line.split()
            for p in parts:
                if p.startswith('cycles='):
                    rtl_cycles = int(p.split('=')[1])
                if p.startswith('gemm='):
                    rtl_gemm = int(p.split('=')[1])

    # 5) 模型周期
    model_cycles = seg_exact(seg_dir, cols=COLS, w_words=W_WORDS)
    print('  Model cycles: %d' % model_cycles)
    print('  RTL cycles:   %d' % rtl_cycles)
    print('  GEMM cycles:  %d (%.1f%% of total)' % (
        rtl_gemm, 100.0 * rtl_gemm / rtl_cycles if rtl_cycles else 0))

    # 6) 位精确比对（输出区域）
    ddr_rtl = read_mem(os.path.join(HERE, 'dump_ddr_seg.mem'), len(ddr))
    bit_ok = True
    for ddr_addr, n_words in out_regions:
        gold_slice = ddr_gold[ddr_addr:ddr_addr+n_words]
        rtl_slice = ddr_rtl[ddr_addr:ddr_addr+n_words]
        if not np.array_equal(gold_slice, rtl_slice):
            diffs = np.sum(gold_slice != rtl_slice)
            print('  BIT MISMATCH at ddr=%d: %d/%d bytes differ' % (
                ddr_addr, diffs, n_words))
            bit_ok = False
        else:
            print('  BIT OK at ddr=%d (%d bytes)' % (ddr_addr, n_words))

    if bit_ok:
        print('  >> BIT-EXACT PASS')
    else:
        print('  >> BIT FAIL')

    return bit_ok, rtl_cycles, model_cycles, rtl_gemm

if __name__ == '__main__':
    results = []
    for seg in TEST_SEGS:
        ok, rtl_c, model_c, gemm_c = run_seg(seg)
        results.append((seg, ok, rtl_c, model_c, gemm_c))

    print('\n=== 汇总 ===')
    total_rtl = 0
    total_model = 0
    for seg, ok, rtl_c, model_c, gemm_c in results:
        dev = 100.0 * (rtl_c - model_c) / model_c if model_c else 0
        g_pct = 100.0 * gemm_c / rtl_c if rtl_c else 0
        print('%s: %s  RTL=%d  Model=%d  偏差=%.1f%%  GEMM占比=%.1f%%' % (
            seg, 'PASS' if ok else 'FAIL', rtl_c, model_c, dev, g_pct))
        total_rtl += rtl_c
        total_model += model_c
    print('总计: RTL=%d  Model=%d  偏差=%.1f%%' % (
        total_rtl, total_model,
        100.0 * (total_rtl - total_model) / total_model))
