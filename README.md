# vector-core-r3c：INT8 脉动阵列加速器（ZCU104）R3C 交接仓库

这是 FPGA 硬件线的交接仓库，对应 2026-09-01 的 R3C 收官形态。**入口是 [handoff_r3c/README.md](handoff_r3c/README.md)**——一天上手路线、目录地图、环境要求都在那里。

## 一句话现状

16×96 INT8 脉动阵列（1536 DSP / 88.9%），R1（DMA 双引擎）+ R2（CTX 预取）+ R3C（行组流水）三代优化全部落地：12 项回归位精确全绿、GEMM 全帧 336→185M 拍（−45%）、1.39~1.67 ms/帧。只综合未上板。

## 仓库范围

| 目录 | 内容 |
|---|---|
| `handoff_r3c/` | 交接文档五件套（README/STATUS/ARCHITECTURE/FLOW/PITFALLS） |
| `hb_fpga_impl/22_r3c_rtl/` | **当前主线 RTL + 仿真回归** |
| `hb_fpga_impl/14_rtl_r1r2/` | R1+R2 轮 RTL 存档（含服务器 Verilator 模板） |
| `hb_fpga_impl/12_actv/` | 编译器 + 周期账本 + 真实流数据（模型复算依赖） |
| `hb_fpga_impl/01~23_*/` | 各轮报告与模型脚本（WORKLOG.md 是全项目时间线权威） |
| `research_holobrain/` | 算法侧调研（性能预算的源头） |
| `CLAUDE.md` | 项目工作规约（含第 9 节实验速度硬约束） |

**未包含**（在内部完整库，按「只推核心代码和结果、不推数据块」原则排除）：
- 数据块：`a3/build_a3` 的权重 blob（188MB）与各段 ctx/w/ddr mem（只保留了 `segments/*/seq.mem`，模型脚本 r3c_model/pe_sizing 复算的最小依赖）、a3 根的 t7_*.npy 快照、`04_dataset` 的 npz 样本、量化 fixture.pt
- 第三方库：`research_holobrain/robo_orchard_lab`（HoloBrain 上游代码，从上游自行克隆）
- 历史重轮：`03_compiler`（5.6 万文件）、`09_cbound`、`09_int4_impl`、`12_actv/a4`、`a3/tmp_*`，以及 `hw_zcu104` 旧硬件线、Evo-1/SwiftVLA 调研目录

**注意**：`handoff_r3c/FLOW.md` 等文档里含内网服务器地址（Verilator 验证用），本仓库保持 private。
