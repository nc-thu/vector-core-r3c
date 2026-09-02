# -*- coding: utf-8 -*-
"""bench.py — iverilog vs Verilator 吞吐基准（两参数档 × 两工具），写出 bench.json。

前提：run_gates.sh 已跑过（向量已生成、仿真器已编译；缺了会自己补编译）。
测量口径：
  冒烟档（sim/，COLS=12）   iverilog = tb_ae.sv 一次 vvp 连跑 REF+PRIM 两遍
                            verilator = tb_ae_v.sv 两次进程（+MODE=0/1）
  全参数档（sim108/）       iverilog = tb_ae_p.sv 一次 vvp 两遍（若 sim108/ 存在
                            full_iv_timing.json 则复用验收门 g2_iv_base 那次运行
                            的墙钟——同命令/同向量，省 2-3 小时重复跑）
                            verilator = tb_ae_v.sv 两次进程
  大负载（sim_big/，COLS=108，~1.2M 拍）Verilator 单边跑 +MODE=0（iverilog
                            全参数档单拍成本过高，吞吐用 sim108 冒烟档代表）
  cycles/秒 = 计数器 cycles 总和 / 墙钟总耗时（不含编译，含进程启动）
外推：54e6 拍 / rate -> 小时数。
用法：python bench.py            # 三档全测，写 bench.json 到本目录
"""
import json, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
V = '~/.conda/envs/vsim/bin'
RTL = ['ae_pkg.sv', 'ae_dpram.sv', 'ae_ctx_ram.sv', 'ae_pe.sv', 'ae_sysarr.sv',
       'ae_requant.sv', 'rq_v2.sv', 'rq_ms.sv', 'ae_exp_lut.sv', 'ae_gemm.sv',
       'ae_softmax.sv', 'ae_copy.sv', 'ae_dma.sv', 'ae_sched.sv', 'ae_core.sv']
FULLDEF = ['-DV_COLS=108', '-DV_CTX_WORDS=131072', '-DV_W_WORDS=4096',
           '-DV_SEQ_N=2048', '-DV_DDR_BYTES=8388608', '-DV_WDG_CYC=20000000']
CYC_RE = re.compile(r'cycles=(\d+)')

def run(cmd, cwd, capture=True):
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=cwd, capture_output=capture, text=True)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        print(r.stdout[-3000:] if capture else '')
        print(r.stderr[-3000:] if capture else '')
        raise SystemExit(f"FAIL ({dt:.1f}s): {' '.join(cmd)}")
    return dt, (r.stdout or '') + (r.stderr or '')

def parse_cycles(out):
    """[tb]/[tb_v] 行里的 cycles=<n>（每模式一行），返回 [ref, prim] 或 [ref]"""
    return [int(m) for m in CYC_RE.findall(out)]

def compile_iv(workdir, defs, tb, out):
    if os.path.exists(os.path.join(workdir, out)):
        return
    cmd = [f'{V}/iverilog', '-g2012'] + defs + ['-o', out,
            '-I', '../rtl'] + ['../rtl/' + f for f in RTL] + [tb]
    dt, _ = run(cmd, workdir)
    print(f"  [compile] iverilog {tb} -> {out} ({dt:.0f}s)")

def compile_vl(workdir, defs):
    if os.path.exists(os.path.join(workdir, 'obj_dir', 'Vtb_ae_v')):
        return
    # 走 vlbuild.sh：内部 make 用系统 g++-10 覆盖 conda 工具链（-fcoroutines/ctime 问题）
    cmd = ['bash', '../vlbuild.sh'] + defs + ['--top-module', 'tb_ae_v', 'tb_ae_v.sv'] + \
          ['../rtl/' + f for f in RTL]
    dt, _ = run(cmd, workdir)
    print(f"  [compile] verilator ({dt:.0f}s)")

def bench_tier(name, workdir, defs, iv_tb, iv_out, modes=(0, 1), iv_marker=None):
    print(f"== {name}")
    res = {}
    # iverilog 三种口径：
    #   iv_marker 给定且文件存在 → 复用一次已完成 vvp 运行的墙钟（全参数档 2 遍要
    #     跑 2-3 小时，与验收门 g2_iv_base 是同命令/同向量/同 vvp 文件，重跑一次
    #     纯属浪费机器——marker 由人工用该次运行的起止时间戳写好放进 workdir）
    #   tb_ae/tb_ae_p → 一次 vvp 连跑两模式（验收门同款 TB）
    #   tb_ae_v → 每模式一个独立进程
    marker_path = os.path.join(workdir, iv_marker) if iv_marker else None
    if marker_path and os.path.exists(marker_path):
        with open(marker_path) as mf:
            m = json.load(mf)
        cy, tot_t = m['cycles'], m['wall_s']
        tot_c = sum(cy)
        res['iverilog'] = dict(wall_s=round(tot_t, 3), cycles=cy, total_cycles=tot_c,
                               cycles_per_s=round(tot_c / tot_t, 1),
                               basis='复用验收门 g2_iv_base 同一 vvp 运行的墙钟（同命令/同向量）')
        print(f"  iverilog : {tot_c} cycles / {tot_t:.0f}s = {tot_c/tot_t/1000:.3f} kcy/s （复用 {iv_marker}）")
    else:
        compile_iv(workdir, defs, iv_tb, iv_out)
        if iv_tb == 'tb_ae_v.sv':
            tot_t, cy = 0.0, []
            for m in modes:
                dt, out = run([f'{V}/vvp', iv_out, f'+MODE={m}'], workdir)
                c = parse_cycles(out)
                assert len(c) == 1, f"iverilog +MODE={m} 输出异常: {out[-500:]}"
                tot_t += dt; cy += c
            tot_c = sum(cy)
        else:
            tot_t, out = run([f'{V}/vvp', iv_out], workdir)
            cy = parse_cycles(out)
            assert len(cy) == len(modes), f"iverilog 输出异常: {out[-500:]}"
            tot_c = sum(cy)
        res['iverilog'] = dict(wall_s=round(tot_t, 3), cycles=cy, total_cycles=tot_c,
                               cycles_per_s=round(tot_c / tot_t, 1))
        print(f"  iverilog : {tot_c} cycles / {tot_t:.2f}s = {tot_c/tot_t/1000:.1f} kcy/s")
    compile_vl(workdir, defs)
    # verilator：每模式一个进程
    tot_c, tot_t, vcy = 0, 0.0, []
    for m in modes:
        dt, out = run([os.path.join('.', 'obj_dir', 'Vtb_ae_v'), f'+MODE={m}'], workdir)
        c = parse_cycles(out)
        assert len(c) == 1, f"verilator +MODE={m} 输出异常: {out[-500:]}"
        tot_c += c[0]; tot_t += dt; vcy += c
    res['verilator'] = dict(wall_s=round(tot_t, 3), cycles=vcy, total_cycles=tot_c,
                            cycles_per_s=round(tot_c / tot_t, 1))
    print(f"  verilator: {tot_c} cycles / {tot_t:.2f}s = {tot_c/tot_t/1000:.1f} kcy/s")
    # 两工具 cycles 对账：REF 必须严格相等；PRIM 允许 |Δ|≤64——tb_ae/tb_ae_p 同进程
    # 连跑两遍，第二遍的 LFSR 停顿相位与 fresh 进程不同（dma 读停顿由绝对时间的
    # LFSR 决定），实测差 ~8 拍，属 TB 残留不是工具差异（详见 run_gates.sh 头部根因说明）
    cy_i, cy_v = res['iverilog']['cycles'], res['verilator']['cycles']
    assert len(cy_i) == len(cy_v), f"{name}: 模式数不一致 iv={cy_i} vl={cy_v}"
    d = [a - b for a, b in zip(cy_i, cy_v)]
    assert d[0] == 0, f"{name}: REF cycles 不一致 iv={cy_i[0]} vl={cy_v[0]}"
    assert all(abs(x) <= 64 for x in d[1:]), \
        f"{name}: PRIM cycles 偏差超容忍 iv-vl={d}（LFSR 相位残留应在 ±64 内）"
    res['cycles_match'] = (f"REF 严格一致（{cy_i[0]}）；PRIM Δ={d[1:]}（同进程 LFSR 相位残留，"
                           f"|Δ|≤64 容忍）" if len(d) > 1 else f"严格一致（{cy_i}）")
    return res

def bench_big():
    """大负载档（sim_big/，~1.2M 拍）—— Verilator 单边。

    2026-08-30 实测：全参数档 iverilog 单拍成本过高（8MB ddr_init.mem 的
    readmemh 文本解析 + 1728 PE 4 态阵列），1.2M 拍要跑几十小时，没有可操作性。
    iverilog 的吞吐用全参数冒烟档（full_col108）的实测速率代表；Verilator 用
    大负载档实测（含长跑的 cache/分支效应），两者口径在 bench.json 里分别标注。
    """
    print("== 大负载 COLS=108 ~1.2M 拍（Verilator 单边）")
    compile_vl('sim_big', FULLDEF)
    dt, out = run([os.path.join('.', 'obj_dir', 'Vtb_ae_v'), '+MODE=0'], 'sim_big')
    c = parse_cycles(out)
    assert len(c) == 1, f"verilator big 输出异常: {out[-500:]}"
    res = dict(verilator=dict(wall_s=round(dt, 3), cycles=c, total_cycles=c[0],
                              cycles_per_s=round(c[0] / dt, 1)))
    print(f"  verilator: {c[0]} cycles / {dt:.2f}s = {c[0]/dt/1000:.1f} kcy/s")
    return res

def main():
    os.chdir(ROOT)
    report = {'meta': dict(
        verilator=subprocess.run([f'{V}/verilator', '--version'], capture_output=True,
                                 text=True).stdout.strip(),
        iverilog=subprocess.run([f'{V}/iverilog', '-V'], capture_output=True,
                                text=True).stdout.splitlines()[0],
        nproc=os.cpu_count(), note='cycles/s 不含编译；54M 拍外推用全参数档大负载口径')}

    report['smoke_col12'] = bench_tier('冒烟档 COLS=12 (sim/)',
                                       'sim', [], 'tb_ae.sv', 'bench_iv_smoke.vvp')
    report['full_col108'] = bench_tier('全参数档 COLS=108 冒烟负载 (sim108/)',
                                       'sim108', FULLDEF, 'tb_ae_p.sv',
                                       'bench_iv_full.vvp',
                                       iv_marker='full_iv_timing.json')
    report['big_col108'] = bench_big()

    ex = {}
    r_vl = report['big_col108']['verilator']['cycles_per_s']   # 大负载实测
    r_iv = report['full_col108']['iverilog']['cycles_per_s']   # 全参数冒烟实测
    ex['verilator'] = dict(rate_kcycles_per_s=round(r_vl / 1000, 1),
                           basis='sim_big ~1.2M 拍实测',
                           hours_54M=round(54e6 / r_vl / 3600, 2),
                           wall_s_54M=round(54e6 / r_vl, 1))
    ex['iverilog'] = dict(rate_kcycles_per_s=round(r_iv / 1000, 1),
                          basis='sim108 冒烟负载实测（大负载不可操作）',
                          hours_54M=round(54e6 / r_iv / 3600, 2),
                          wall_s_54M=round(54e6 / r_iv, 1))
    report['extrapolate_54M_cycles'] = ex

    with open('bench.json', 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[bench] 写出 bench.json")
    for tool, e in ex.items():
        print(f"  {tool:9s}: {e['rate_kcycles_per_s']} kcy/s -> 54M 拍 ≈ "
              f"{e['hours_54M']} h ({e['wall_s_54M']} s)")

if __name__ == '__main__':
    main()
