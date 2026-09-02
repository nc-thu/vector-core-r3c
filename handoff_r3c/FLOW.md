# 验证与综合流程（动代码前必读）

## 实验速度硬规矩（用户裁决，写进 CLAUDE.md 第 9 节）

1. **每轮实验 ≤10 分钟**。超时就缩范围：抽 3~5 个代表段、降规模，不干等全扫。
2. **RTL 仿真优先 Verilator**（服务器，快一个量级）。iverilog 只做秒级语法/微观冒烟。
3. 拍数评估默认走解析模型（est × 校准系数），位精确门只抽代表段。

## 验证矩阵（动了什么就必须跑什么）

| 动了哪里 | 必跑 | 为什么 |
|---|---|---|
| ae_pe / ae_sysarr / ae_gemm（阵列/量化/读出） | 全部 12 项 | 乘加和读出位精确是命根子 |
| ae_requant / rq_ms / rq_v2 | tb_rq + sim_ae + compare | 量化位图错一个字节全盘皆输 |
| ae_dma / ae_core（DMA/握手/仲裁） | sim_ae + compare + **actv 三件套**（含 PF=1） | R2 死锁就是 actv+pf1 负载踩出来的 |
| ae_sched | sim_ae 双遍（REF/PRIM）+ PF=0/1 两档 | skip 和预取都在调度器里 |
| ae_softmax / ae_sm16 | tb_sm16 | softmax 位精确 |
| ae_actv | tb_ae_actv + compare_actv | 片上算子 |
| 只改 COLS/参数 | gem_cycles 对账 + 综合 DSP 数核对 | NGRP=COLUMNS/4 要跟着对 |
| 任何 RTL | 综合（DSP 恰好 1536、WNS 不劣化） | use_dsp 属性丢了他不会报错，只会悄悄挤进 LUT |

## 回归（一键 12 项）

```bash
cd "e:/GPU ARCH/vector_core_sim/hb_fpga_impl/22_r3c_rtl/sim"
bash regression.sh          # 日志逐项存 reg_logs/<步骤名>.log
```

内容：gen_vectors → sim_ae（全芯片双遍 REF/PRIM，COLS=12 冒烟）→ compare（CTX/DDR 全量逐字节）→ tb_rq → tb_sm16 → tb_pe_pack → tb_pe_pack_dsp（要 Vivado unisim）→ gem_cycles 对账 → actv 三件套（gen_actv/sim_ae_actv/compare_actv/tb_ae_actv）。

**注意：iverilog 串行全套约 1 小时。日常迭代不要整套跑**——单跑你动的那部分（上表），全量回归留给收官。

## Verilator 快路径（服务器，推荐）

```bash
# 本机同步 RTL 到服务器（首次）
scp -r "hb_fpga_impl/22_r3c_rtl/rtl" nc23@101.6.64.77:/tmp/ae_vgate/rtl_r3c
scp "hb_fpga_impl/14_rtl_r1r2/sim/tb_ae_v.sv" nc23@101.6.64.77:/tmp/ae_vgate/

# 服务器上（模板：14_rtl_r1r2/sim/vlbuild_vgate.sh）
ssh nc23@101.6.64.77
cd /tmp/ae_vgate
/home/nc23/.conda/envs/vsim/bin/verilator --binary --timing -j 0 -Wno-fatal \
  --top-module tb_ae_v -DV_COLS=96 -DV_CTX_WORDS=131072 -DV_W_WORDS=4096 \
  -DV_SEQ_N=2048 -DV_DDR_BYTES=524288 -DV_WDG_CYC=20000000 \
  rtl_r3c/ae_*.sv rtl_r3c/rq_*.sv tb_ae_v.sv
make -C obj_dir -f Vtb_ae_v.mk -j 32 CXX=/usr/bin/g++-10
```

构建一次约 2~3 分钟，之后每段位精确检查秒级——这就是「≤10 分钟一轮」的正确姿势。服务器纪律：/home 满只写 /tmp，跑完不留进程。

## 综合（一轮 20~40 分钟）

工作区模式：`E:\ae_syn\<轮次名>\`（**无空格路径**，仓库路径含 "GPU ARCH" 不能直接跑）。模板抄 `E:\ae_syn\r3c_c96\synth.tcl`：

```bash
cd /e/ae_syn/<新轮次> && /d/software/Vivado/2021.2/bin/vivado.bat -mode batch -source synth.tcl
```

synth.tcl 要点（都是踩出来的）：
- **OOC（out-of-context）+ RuntimeOptimized**：1728 PE 级设计非 OOC 会在 Final Netlist Cleanup 崩溃。
- **显式 read_verilog 清单**（17 个 .sv），别用通配符。
- **-flatten_hierarchy none**：保层级 utilization（不然 util_hier_d2 出不来）。
- 时钟 4.000 ns（250 MHz 目标）。

产物核对清单：
1. `out/utilization.rpt`：**DSP48E2 恰好 1536**。如果少了（PE 掉进 LUT）= use_dsp 属性丢了；如果 LUT 暴涨同理。
2. `out/timing.rpt`：WNS 对比上一轮（当前基线 −1.363）。
3. `out/util_hier_d2.rpt`：各模块 LUT/FF/DSP/BRAM 拆账。

## 周期模型对账（gem_cycles）

改了 RTL 时序行为（不动功能）也要重对账：`22_r3c_rtl/sim/gem_cycles.py`。本轮刚用命令级探针重标过（写引擎每 16B 行 6 拍、命令尾 4 拍、行为级从机开销 3 拍），模型与 RTL 偏差 <5%。**旧常量别信**——RTL 一改两边一起过期，用命令级探针（dut.u_sched.dma_c_r，注意不在顶层）逐条实测再写死。

## 改 COLS 的完整清单

COLS 是参数（ae_pkg/ae_top/ae_core/ae_gemm/ae_dma/ae_copy 默认值），但改了不等于完事：

1. RTL 默认值六处 + grep 硬编码残留。
2. NGRP=COLUMNS/RQ_SH（4 列一套 rq_ms）自动推导，确认无硬编码 27/108。
3. **重跑编译器**：`12_actv/a3/compiler_a3.py`（WRAM 路由按「每 k 恰好 COLS 字节」）——不重跑就吃窄尾组浪费（本模型实测 +40%）。
4. 重跑回归 + 综合（DSP 数 = 16×COLS）。

## 全帧性能复算（模型脚本，秒级）

```bash
cd "hb_fpga_impl/19_r3c_arch"
/d/ana/python r3c_model.py    # R3C 全帧拍数（GEMM/TB/HP64 三口径）
/d/ana/python pe_sizing.py    # 任意 COLS 的选型对比（双口径）
```

输入是 `12_actv/a3/build_a3/segments/`（2580 段真实流）。模型假设变了改脚本重跑即可，别手算。
