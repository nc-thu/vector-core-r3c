# -*- coding: utf-8 -*-
"""fix_streams.py — 修复 build_full 描述符流的 dma_len 编码溢出（2026-08-31）

问题（本测量轮发现）：compiler.desc() 把 dma_len<<61 塞进 18 位字段 [78:61]，
长度 > DMA_MAX(0x3FFF0=262128) 的 LOAD/STORE 高位溢到 is_loop_end(79)/
in_loop(80)/steps[0](81)（更大的还进 steps 高位）。RTL/黄金都只读 18 位：
- 窄化后 ≠0 → DMA 截断（少搬字节，不挂死，数据错）
- 窄化后 ==0 → remain=0 下溢 → DMA 无限循环 → 看门狗 FATAL（seg_0142 实测）

修法（外科手术，不动布局/清单/其余描述符）：把溢出描述符拆成 ≤DMA_MAX 的
分片序列，语义与"intended 长度一次搬完"逐字节等价：
  STORE(op5)      chunk_i: addr=A+i*C, y_base=Y+i*C/16, len=min(C, L-i*C)
  LOAD CTX(b0)    chunk_i: addr=A+i*C, b_base=B+i*C/16
  LOAD W(b≠0)     每 k 行 108B 路由，分片须 108 对齐：C_w=261792(=864*303)，
                  b_base 每 chunk +303（=C_w/108）
j0(62-77) 与 LOAD/STORE 的 dma_len(61-78) 本就共用位段（编码器历史上允许），
新分片 j0=0、steps/in_loop/is_end=0、其余字段原样拷贝。
intended L 重构 = desc[83:61]（23 位，可表到 8MB），断言 [91:84]==0。
产出 build_full_fixed/segments/*/seq.mem（未受影响段字节不变）+ fix_report.json。
"""
import json
import os
import shutil
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else '../03_compiler/build_full'
DST = sys.argv[2] if len(sys.argv) > 2 else '../03_compiler/build_full_fixed'
DMA_MAX = 262128                 # 0x3FFF0，16 的倍数
C_W = 261792                     # 108×8×303：LOAD W 的 108 对齐分片（8 倍数）

KEEP_MSK = (1 << 256) - 1
for sh, w in ((228, 16), (212, 16), (196, 16), (176, 20), (156, 20), (136, 20),
              (120, 16), (104, 16), (96, 8), (92, 4), (29, 32)):
    KEEP_MSK &= ~(((1 << w) - 1) << sh)
LEN_MSK = ((1 << 23) - 1) << 61          # desc[83:61]
LEN_CLEAR = ((1 << 31) - 1) << 61        # desc[91:61]：dma_len+j0 重叠段+循环位全清
HIGH_STEPS = ((1 << 8) - 1) << 84        # steps 高 8 位，必须为 0


def split_desc(d):
    """溢出则返回分片列表，否则 None。"""
    op = (d >> 252) & 0xF
    if op not in (4, 5):
        return None
    if d & HIGH_STEPS:
        raise AssertionError('steps 高位非零，无法安全重构 intended 长度')
    L = (d & LEN_MSK) >> 61
    narrow = L & 0x3FFFF
    if L == narrow:                       # 没溢出
        return None
    b_src = (d >> 246) & 7
    base_f = 'y_base' if op == 5 else 'b_base'
    base = (d >> (136 if op == 5 else 156)) & 0xFFFFF
    addr = (d >> 29) & 0xFFFFFFFF
    C = C_W if (op == 4 and b_src != 0) else DMA_MAX
    step_b = C // 16 if not (op == 4 and b_src != 0) else None
    out = []
    off = 0
    while off < L:
        n = min(C, L - off)
        nd = (d & KEEP_MSK & ~LEN_CLEAR) | ((n & 0x3FFFF) << 61) \
            | (((addr + off) & 0xFFFFFFFF) << 29)
        if op == 4 and b_src != 0:
            nd |= ((base + off // 108) & 0xFFFFF) << 156
        else:
            nd |= ((base + off // 16) & 0xFFFFF) << (136 if op == 5 else 156)
        out.append(nd)
        off += n
    return out


def main():
    segs = sorted(os.listdir(os.path.join(SRC, 'segments')))
    os.makedirs(os.path.join(DST, 'segments'), exist_ok=True)
    rep = dict(n_seg_affected=0, n_desc_split=0, n_chunks_added=0,
               hang_fixed=0, affected=[], errors=[])
    for s in segs:
        sp = os.path.join(SRC, 'segments', s, 'seq.mem')
        words = [int(l, 16) for l in open(sp) if l.strip()]
        out, changed, nsp, hang = [], False, 0, 0
        for d in words:
            ch = split_desc(d)
            if ch is None:
                out.append(d)
            else:
                changed = True
                nsp += 1
                if (d & 0x3FFFF << 61) == 0:
                    hang += 1
                out.extend(ch)
        dp = os.path.join(DST, 'segments', s)
        if not os.path.exists(dp):
            os.makedirs(dp)
            shutil.copy(os.path.join(SRC, 'segments', s, 'manifest.json'), dp)
        if changed:
            if len(out) > 2048:
                rep['errors'].append('%s: 拆分后 %d 条 > SEQ_N' % (s, len(out)))
            with open(os.path.join(dp, 'seq.mem'), 'w') as f:
                for d in out:
                    f.write('%064X\n' % d)
            rep['n_seg_affected'] += 1
            rep['n_desc_split'] += nsp
            rep['n_chunks_added'] += len(out) - len(words)
            rep['hang_fixed'] += hang
            rep['affected'].append(s)
        else:
            # 未受影响：硬链不住就复制（Windows/跨设备都兜住）
            dst_f = os.path.join(dp, 'seq.mem')
            if not os.path.exists(dst_f):
                shutil.copyfile(sp, dst_f)
    for f in ('weights_blob.bin', 'host_plan.json', 'model_summary.json'):
        p = os.path.join(DST, f)
        if not os.path.exists(p):
            os.copyfile if False else shutil.copyfile(os.path.join(SRC, f), p)
    json.dump(rep, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'fix_report.json'), 'w'), indent=1)
    print('受影响段 %d，拆分描述符 %d，新增分片 %d，其中窄化=0（原会挂死）%d'
          % (rep['n_seg_affected'], rep['n_desc_split'], rep['n_chunks_added'],
             rep['hang_fixed']))
    if rep['errors']:
        print('错误：', rep['errors'])


if __name__ == '__main__':
    main()
