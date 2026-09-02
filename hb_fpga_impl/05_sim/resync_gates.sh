#!/usr/bin/env bash
# ============================================================================
# resync_gates.sh — 新 RTL 重同步的本地驱动（在本地 05_sim/ 下运行）
# 做三件事：
#   1. 在服务器搭运行树 $A（默认 /tmp/aers——2026-08-31 实测 /home 17T 盘 100%
#      满、0 字节可写，scp/sed 全失败，运行树只能放 /tmp；rtl 从服务器旧快照
#      复制后用本地新 RTL 覆盖）
#   2. 上传：01_rtl/rtl/*.sv → $A/rtl/；01_rtl/sim/{tb_ae.sv,gen_vectors.py,
#      gem_cycles.py,regression.sh} → $A/sim/；05_sim/{tb_ae_v.sv(带 iverilog-only
#      CTX/WRAM 清零),emit_big.py(生成后把向量搬回 sim_big),resync_server.sh,
#      vlbuild.sh} → $A 及 sim108/sim_big 副本；sim108 还要 gen_vectors_108.py
#      （emit_big 依赖，从服务器旧 sim108 拿）
#   3. 对新 tb_ae.sv 施加 NBA 竞态修复（start/rst_n/hoist_en/pf_en 的阻塞赋值
#      会在 iverilog/Verilator 间错相，见 run_gates.sh 头部根因说明）
#   然后服务器上跑 resync_server.sh（4 用例 × 三遍跨工具对拍 + 大负载吞吐重测）
# 用法：bash resync_gates.sh          # A=/tmp/aers；换目录 A=/tmp/xxx bash resync_gates.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
H=nc23@101.6.64.77
A=${A:-/tmp/aers}
L_RTL=../01_rtl/rtl
L_SIM=../01_rtl/sim

# --- 服务器侧运行树（/home 满，树放 /tmp；rtl 从旧快照实拷贝再覆盖） ---
ssh $H "mkdir -p $A/sim $A/sim108 $A/sim_big && \
  [ -d $A/rtl ] || cp -rL /tmp/aerun/rtl $A/rtl 2>/dev/null || \
    cp -r /home/nc23/workspace/holobrain/ae_sim/rtl $A/rtl"

# --- 上传 ---
scp $L_RTL/*.sv $H:$A/rtl/
scp $L_SIM/tb_ae.sv $L_SIM/gen_vectors.py $L_SIM/gem_cycles.py $L_SIM/regression.sh $H:$A/sim/
scp tb_ae_v.sv resync_server.sh vlbuild.sh emit_big.py $H:$A/
scp tb_ae_v.sv $H:$A/sim/tb_ae_v.sv
scp tb_ae_v.sv $H:$A/sim108/tb_ae_v.sv
scp tb_ae_v.sv emit_big.py $H:$A/sim_big/
scp tb_ae_v.sv $H:$A/sim_big/tb_ae_v.sv
ssh $H "[ -f $A/sim108/gen_vectors_108.py ] || \
  cp /tmp/aerun/sim108/gen_vectors_108.py $A/sim108/ 2>/dev/null || \
  cp /home/nc23/workspace/holobrain/ae_sim/sim108/gen_vectors_108.py $A/sim108/"
# compare.py / exp2_lut.mem：emit 对拍与 LUT 装载要用，旧版未动直接复用
ssh $H "[ -f $A/sim/compare.py ] || cp /tmp/aerun/sim/compare.py $A/sim/; \
  [ -f $A/sim/exp2_lut.mem ] || cp /tmp/aerun/sim/exp2_lut.mem $A/sim/"

# --- 新 tb_ae.sv 的 NBA 竞态修复（与 2026-08-30 修旧版 TB 同一处方） ---
ssh $H "cd $A/sim && sed -i 's|    rst_n = 1; repeat (4) @(posedge clk);|    rst_n <= 1; repeat (4) @(posedge clk);  // NBA：复位释放竞态（见 run_gates.sh 头部）|; s|    start = 1; @(posedge clk); start = 0;|    start <= 1; @(posedge clk); start <= 0;  // NBA：启动竞态|; s|    hoist_en = prim;|    hoist_en <= prim;|; s|    pf_en = pf;|    pf_en <= pf;|' tb_ae.sv && grep -n 'rst_n <= 1\|start <= 1\|hoist_en <=\|pf_en <=' tb_ae.sv"

# --- 清掉旧构建产物（新 RTL 必须重编）+ 跑服务器侧复验 ---
ssh $H "cd $A && rm -rf sim/obj_dir sim108/obj_dir sim_big/obj_dir && \
  (setsid nohup bash -c 'ulimit -c 0; bash resync_server.sh' > $A/resync_run.log 2>&1 < /dev/null &); \
  echo resync_started"
echo "进度看：ssh $H tail -f $A/resync_run.log"
