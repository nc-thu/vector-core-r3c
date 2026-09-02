# -*- coding: utf-8 -*-
"""aggregate.py — 把 sweep/aux/golden 结果汇成 cycles_by_type.json + TOTAL.json
（本地跑：需要 05_sim/{types.json,seg_stage.json,results/} 和 03_compiler/build_full）
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, '..', '03_compiler', 'build_full_fixed')
F198_5, F250 = 198.5e6, 250e6


def main():
    types = json.load(open(os.path.join(HERE, 'types.json')))['types']
    stage = json.load(open(os.path.join(HERE, 'seg_stage.json')))
    sweep = json.load(open(os.path.join(HERE, 'results', 'sweep.json')))
    ests = {e['type_id']: e for e in
            json.load(open(os.path.join(HERE, 'est_by_type.json')))}
    by_type = {r['type_id']: r for r in sweep}

    rows = []
    skipped = []
    for t in types:
        tid, rep, inst = t['type_id'], t['rep'], len(t['instances'])
        r = by_type.get(tid)
        cyc = {}
        if r:
            for u in r['runs']:
                if 'cycles' in u:
                    cyc['ref' if u['mode'] == 0 else 'pf1'] = u['cycles']
        if 'ref' not in cyc or 'pf1' not in cyc:
            skipped.append(dict(type_id=tid, rep=rep, n_instances=inst,
                                runs=r['runs'] if r else 'sweep 缺该类型'))
            continue
        man = json.load(open(os.path.join(
            BUILD, 'segments', 'seg_%04d' % rep, 'manifest.json')))
        rows.append(dict(type_id=tid, rep=rep, n_instances=inst,
                         n_descs=t['n_descs'], stage=stage['seg_%04d' % rep],
                         est_cycles=ests[tid]['est_v0'],
                         est_lenfix=ests[tid]['est_lenfix'],
                         est_fixed=ests[tid]['est_fixed'],
                         macs_v0=man['macs'],
                         macs_padded=ests[tid]['macs_padded'],
                         macs_useful=ests[tid]['macs_useful'], cycles=cyc,
                         mac_total=next(u['mac_total'] for u in r['runs']
                                        if u['mode'] == 1)))

    tot_ref = sum(r['cycles']['ref'] * r['n_instances'] for r in rows)
    tot_pf1 = sum(r['cycles']['pf1'] * r['n_instances'] for r in rows)
    est_sum = sum(r['est_cycles'] * r['n_instances'] for r in rows)
    est_lenfix = sum(r['est_lenfix'] * r['n_instances'] for r in rows)
    est_fixed = sum(r['est_fixed'] * r['n_instances'] for r in rows)
    macs_sum = sum(r['macs_padded'] * r['n_instances'] for r in rows)
    macs_useful = sum(r['macs_useful'] * r['n_instances'] for r in rows)
    macrtl_sum = sum(r['mac_total'] * r['n_instances'] for r in rows)

    by_stage = {}
    for r in rows:
        s = by_stage.setdefault(r['stage'], dict(segments=0, ref=0, pf1=0,
                                                 est=0, est_fixed=0,
                                                 types=0, macs=0))
        s['segments'] += r['n_instances']
        s['types'] += 1
        s['ref'] += r['cycles']['ref'] * r['n_instances']
        s['pf1'] += r['cycles']['pf1'] * r['n_instances']
        s['est'] += r['est_cycles'] * r['n_instances']
        s['est_fixed'] += r['est_fixed'] * r['n_instances']
        s['macs'] += r['macs_padded'] * r['n_instances']
    for s in by_stage.values():
        s['ms_198p5'] = s['pf1'] / F198_5 * 1e3
        s['ms_250'] = s['pf1'] / F250 * 1e3

    # 对账：三档 est（v0 窄长+双除 / 修长 / 修长+修 mt）分别与实测比
    for key in ('ref', 'pf1'):
        meas = sum(r['cycles'][key] * r['n_instances'] for r in rows)
        for ek, ev in (('est_v0', est_sum), ('est_lenfix', est_lenfix),
                       ('est_fixed', est_fixed)):
            print('[%s vs %s] %+d (%+.2f%%)' % (key, ek, meas - ev,
                                                (meas - ev) / ev * 100))
        devs = sorted(rows, key=lambda r: -(r['cycles'][key] - r['est_fixed']))
        print('[%s vs est_fixed] 偏差最大的 5 个类型：' % key)
        for r in devs[:5]:
            print('  type %3d seg %4d %-16s meas=%8d est_fixed=%8d (%+.1f%%) ×%d'
                  % (r['type_id'], r['rep'], r['stage'], r['cycles'][key],
                     r['est_fixed'],
                     (r['cycles'][key] - r['est_fixed']) / r['est_fixed'] * 100
                     if r['est_fixed'] else 0, r['n_instances']))

    json.dump(rows, open(os.path.join(HERE, 'cycles_by_type.json'), 'w'),
              indent=1)
    if skipped:
        print('警告：%d 个类型无完整实测（不计入总数）：' % len(skipped))
        for s in skipped[:5]:
            print('  type %d seg %04d inst=%d' %
                  (s['type_id'], s['rep'], s['n_instances']))

    total = dict(
        total_cycles_ref=tot_ref, total_cycles_pf1=tot_pf1,
        n_types_measured=len(rows), types_skipped=skipped,
        est_cycles_v0=est_sum, est_lenfix=est_lenfix, est_fixed=est_fixed,
        ms_ref_at_198p5=tot_ref / F198_5 * 1e3,
        ms_pf1_at_198p5=tot_pf1 / F198_5 * 1e3,
        ms_ref_at_250=tot_ref / F250 * 1e3,
        ms_pf1_at_250=tot_pf1 / F250 * 1e3,
        n_types=len(rows), n_segments=2782,
        macs_padded=macs_sum, macs_useful=macs_useful,
        macs_rtl=macrtl_sum, macs_match=(macs_sum == macrtl_sum),
        stage_table=by_stage,
        heaviest_types=sorted(
            [dict(type_id=r['type_id'], rep=r['rep'], stage=r['stage'],
                  cycles_pf1=r['cycles']['pf1'], n=r['n_instances'],
                  contrib_pf1=r['cycles']['pf1'] * r['n_instances'])
             for r in rows], key=lambda x: -x['contrib_pf1'])[:10],
    )
    aux_p = os.path.join(HERE, 'results', 'aux.json')
    if os.path.exists(aux_p):
        total['aux'] = json.load(open(aux_p))
    gold_p = os.path.join(HERE, 'results', 'golden.json')
    if os.path.exists(gold_p):
        total['golden'] = json.load(open(gold_p))
    json.dump(total, open(os.path.join(HERE, 'TOTAL.json'), 'w'), indent=1)

    print('总拍数 REF=%d PRIM-pf1=%d（est: v0=%d lenfix=%d fixed=%d）' %
          (tot_ref, tot_pf1, est_sum, est_lenfix, est_fixed))
    print('毫秒 @198.5MHz: ref=%.1f pf1=%.1f | @250MHz: ref=%.1f pf1=%.1f' %
          (tot_ref / F198_5 * 1e3, tot_pf1 / F198_5 * 1e3,
           tot_ref / F250 * 1e3, tot_pf1 / F250 * 1e3))
    print('MAC 对账（padded 口径）：修正模型 %d vs RTL mac_total %d 全等=%s' %
          (macs_sum, macrtl_sum, macs_sum == macrtl_sum))
    print('有效 MAC（m*n_loc*k，去 padding）= %d' % macs_useful)
    print('阶段 top5（按 pf1 拍数）:')
    for s, v in sorted(by_stage.items(), key=lambda kv: -kv[1]['pf1'])[:5]:
        print('  %-18s seg=%4d types=%3d pf1=%10d (%.1f%%) est=%10d' %
              (s, v['segments'], v['types'], v['pf1'],
               v['pf1'] / tot_pf1 * 100, v['est']))


if __name__ == '__main__':
    main()
