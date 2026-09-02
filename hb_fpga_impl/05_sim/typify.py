# -*- coding: utf-8 -*-
"""typify.py — 2782 段的"拍数类型"去重（05_sim 收官测量，2026-08-31）

原理：段的拍数由描述符流里所有影响控制流/数据通路形状的字段决定，与数据值
和 DDR/CTX/WRAM 基址无关（DMA 延迟只看时间不看地址；LFSR 停顿是绝对拍数的
确定函数，同一条指令流 → 同一时刻的 DMA → 同一停顿序列）。因此把 256b 描述符
里的纯地址字段掩掉后，整段流相同的段共享同一拍数。

字段表（compiler.py desc()，与 golden_interp.decode() 一致）：
  保留（形状/控制）：op[255:252] a_src[248:249] b_src[248:246] sm_causal[245]
    y_tr[244] m[243:228] n[227:212] k[211:196] b_spad/n_loc[135:120]
    rq_m[119:104] rq_s[103:96] inv_idx[95:92] steps[91:81] in_loop[80]
    is_loop_end[79] dma_len[78:61] j0[77:62]（与 dma_len 重叠位，按原字保留）
  掩掉（纯地址）：   a_base[195:176] b_base[175:156] y_base[155:136] dma_addr[60:29]
  rq_m/rq_s 数值不影响时序（requant 是固定流水线），保守保留进签名。
"""
import json
import os
import sys

BUILD = sys.argv[1] if len(sys.argv) > 1 else 'build_full'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'types.json'

MASK_KEEP = (1 << 256) - 1
for sh, w in ((176, 20), (156, 20), (136, 20), (29, 32)):
    MASK_KEEP &= ~(((1 << w) - 1) << sh)

types = {}          # sig(tuple) -> [seg ids]
full_types = {}     # 全字段（不掩地址）类型数，只统计用
n_desc_total = 0
for i in range(2782):
    seg = f'seg_{i:04d}'
    words = []
    with open(os.path.join(BUILD, 'segments', seg, 'seq.mem')) as f:
        for line in f:
            line = line.strip()
            if line:
                words.append(int(line, 16))
    # 流以 DONE（op=0xF）结尾，截到 DONE 与黄金/RTL 语义一致
    for j, w in enumerate(words):
        if (w >> 252) & 0xF == 15:
            words = words[:j + 1]
            break
    n_desc_total += len(words)
    full_types.setdefault(tuple(words), []).append(i)
    sig = tuple(w & MASK_KEEP for w in words)
    types.setdefault(sig, []).append(i)

instances = sorted((len(v) for v in types.values()), reverse=True)
out = {
    'n_segments': 2782,
    'n_descs_total': n_desc_total,
    'n_types_full_field': len(full_types),
    'n_types_shape': len(types),
    'max_instances_one_type': instances[0],
    'types': [
        {
            'type_id': t,
            'n_descs': len(sig),
            'instances': segs,
            'rep': segs[0],
            'sig': [hex(w) for w in sig],   # 掩地址后的描述符流（hex 串）
        }
        for t, (sig, segs) in enumerate(sorted(
            types.items(), key=lambda kv: (-len(kv[1]), kv[1][0])))
    ],
}
with open(OUT, 'w') as f:
    json.dump(out, f)
print(f"段 {out['n_segments']}  描述符总数 {n_desc_total}")
print(f"全字段类型数 {len(full_types)}  形状类型数 {len(types)}  "
      f"最大实例数 {instances[0]}")
print(f"实例数 top10: {instances[:10]}")
