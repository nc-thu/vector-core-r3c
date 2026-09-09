# -*- coding: utf-8 -*-
"""查 build 里 v_proj 孪生 VT 输出的段与输入（服务器一次性诊断脚本）。"""
import json
import os
import sys
import collections

build = sys.argv[1] if len(sys.argv) > 1 else 'build_s000'
segdir = os.path.join(build, 'segments')
items = []
for n in sorted(os.listdir(segdir)):
    p = os.path.join(segdir, n, 'manifest.json')
    if os.path.exists(p):
        items.append((n, json.load(open(p, encoding='utf-8'))))
d = dict(items)

hits = collections.defaultdict(list)
for nm, s in items:
    for o in s.get('outputs', ()):
        if '#vt' in o['name'] and 'decoder.layers.1.v_proj' in o['name']:
            hits[o['name']].append(nm)
for k in sorted(hits):
    print(k, '->', hits[k])

print('---')
tgt = hits.get('decoder.layers.1.v_proj#1206#vt0', [])
print('target segs:', tgt)
for t in tgt[:2]:
    s = d[t]
    print('SEG', t)
    print('  inputs :', [(e['name'], e.get('kind'), e.get('m'), e.get('k'))
                         for e in s['inputs']])
    print('  outputs:', [(o['name'], o.get('kind'), o.get('rows16'),
                          o.get('cols')) for o in s['outputs']])

qh = collections.defaultdict(list)
for nm, s in items:
    for o in s.get('outputs', ()):
        if o['name'].startswith('decoder.layers.1.q_proj#') \
                and o['name'].endswith('#h0'):
            qh[o['name']].append(nm)
print('---')
for k in sorted(qh):
    print(k, '->', qh[k])
    for t in qh[k][:1]:
        print('  inputs :', [(e['name'], e.get('kind'), e.get('m'), e.get('k'))
                             for e in d[t]['inputs']])
        print('  outputs:', [(o['name'], o.get('kind')) for o in d[t]['outputs']])

# host_plan 里 temporal 注意力节点与 host_steps 的 ameta
hp = json.load(open(os.path.join(build, 'host_plan.json'), encoding='utf-8'))
print('---')
for n in hp['nodes']:
    if n['kind'] == 'attn' and 'decoder.layers.1' == n['module']:
        print('attn node:', n)
print('---')
for st in hp['host_steps']:
    a = st.get('attn')
    if a and a['module'] == 'decoder.layers.1':
        print('host_step attn:', a)
