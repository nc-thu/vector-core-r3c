# vector-core-r3c：INT8 脉动阵列加速器（ZCU104）交接仓库

FPGA 硬件线交接仓库。2026-09-01 R3C 收官时建立，2026-09-08 起扩展为全项目三线结构（入口 [LINES.md](LINES.md)：algo 算法 / arch 架构 / hw 硬件 + compiler 共享底座 + plans 跨线讨论）。R3C 交接文档入口仍是 [handoff_r3c/README.md](handoff_r3c/README.md)。

## 一句话现状（2026-09-09）

**hw v5 / arch v6 定版 B8-48**：16×48 物理列 Pack2 PE（768 DSP / 44.4%），303.215MHz 布线收敛（WNS +0.009），帧时间模型 0.909 s（预算 2.13 s、余量 2.3×），比 R3C-96（1536 DSP、1.388 s）少一半 DSP、快 34.5%。R2 双读引擎段级实测 −7.4%；B8-64 采点两条件不过出局。各线版本历史看对应目录的 CHANGELOG.md，全项目时间线权威是 `hb_fpga_impl/WORKLOG.md`。

## 仓库范围

| 目录 | 内容 |
|---|---|
| `LINES.md` | 三线结构与版本规则（0908 起的全项目地图） |
| `hw/` | **硬件线（当前主线）**：v4 B8×R3C 集成门、v5 H2 时序收敛 + R2 双读引擎 + B8-64 采点（每版含 rtl/sim/synth/results + HTML 报告） |
| `arch/` | **架构线**：v5 B8 选型扫描、v6 LUT 锚点修正 + 定版收口 + 时间账分析页（b8_scan.py / b8_final.py 可复算） |
| `algo/` | **算法线**：CHANGELOG 总账（工作产物在 `hb_fpga_impl/24~26_*`） |
| `compiler/` | 三线共享编译器/仿真器底座（状态登记） |
| `plans/` | 跨线讨论页 |
| `handoff_r3c/` | R3C 交接文档五件套（README/STATUS/ARCHITECTURE/FLOW/PITFALLS） |
| `hb_fpga_impl/22_r3c_rtl/` | R3C 主线 RTL + 仿真回归（B8 之前的部署基线） |
| `hb_fpga_impl/01~26_*/` | 各轮报告与模型脚本（WORKLOG.md 是全项目时间线权威） |
| `research_holobrain/` | 算法侧调研（性能预算的源头） |
| `CLAUDE.md` | 项目工作规约（含第 9 节实验速度硬约束） |

**未包含**（在内部完整库，按「只推核心代码和结果、不推数据块」原则排除）：
- 数据块：`a3/build_a3` 的权重 blob（188MB）与各段 ctx/w/ddr mem（只保留了 `segments/*/seq.mem`，模型脚本 r3c_model/pe_sizing 复算的最小依赖）、a3 根的 t7_*.npy 快照、`04_dataset` 的 npz 样本、量化 fixture.pt、24~26 轮的 .mem 回归向量转储（44MB）、hw R2 的段级 DDR dump
- 第三方库：`research_holobrain/robo_orchard_lab`（HoloBrain 上游代码，从上游自行克隆）
- 本地工具产物：`arch/figures/`（Visio 工具链产物）、仿真二进制（.vvp/.vcd）与 Vivado 生成物（综合工作区在库外 E:\ae_syn）
- 历史重轮：`03_compiler`（5.6 万文件）、`09_cbound`、`09_int4_impl`、`12_actv/a4`、`a3/tmp_*`，以及 `hw_zcu104` 旧硬件线、Evo-1/SwiftVLA 调研目录、`pe_w8a8_sota` 等 round_report 调研目录

**说明**：文档里的服务器地址已替换为 `<SERVER>` 占位符（内部同事从内部渠道获取）；Verilator/ssh 命令中的 `~` 即该服务器用户 home。
