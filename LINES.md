# 三线工作区入口（2026-09-08 13:42 建立）

项目分三条线推进，各自独立目录、独立版本号、独立 CHANGELOG。这页是总入口。

| 线 | 中文名 | 英文名 | 目录 | 管什么 | 当前版本 |
|---|---|---|---|---|---|
| 算法线 | 算法线（量化与部署精度） | Algorithm & Accuracy | [algo/](algo/CHANGELOG.md) | 误差归因、量化方案、部署链数值；判据 jpos ≤ 0.045 rad | v4（0.0500/0.0622） |
| 架构线 | 架构线（拍数与调度建模） | Architecture & Performance | [arch/](arch/CHANGELOG.md) | 周期模型、pe_sizing、调度与读压缩，产出 RTL 需求 | v4（R3C 模型 1.39 s） |
| 硬件线 | 硬件线（RTL 电路与实现） | Hardware & RTL | [hw/](hw/CHANGELOG.md) | RTL 改动、综合/布线、B8 PE、功耗、回归 | v3（R3C 定版 = 部署基线） |
| 共享底座 | 编译器与仿真器 | Compiler & Simulator | [compiler/](compiler/CHANGELOG.md) | 三条线共同改动的基础设施（语义/调度/指令编码） | v5（SSA trace + 修复链） |
| 跨线讨论 | — | — | [plans/](plans/README.md) | 跨线讨论页、路线拍板记录 | — |

## 版本与命名规则

- **版本 = 一次有定论的收口**：正结果落地或负结果定案都立一版；纯过程性中间实验不立版，写进当版文件夹或 CHANGELOG 条目。
- **版本文件夹**：`v{N}_{YYYY-MM-DD}_{HHMM}_{ascii说明}/`，纯 ASCII（中文文件名在 IDE 链接里会被转义出问题，中文只进页内标题）。
- **版本报告**：`{vN}_{YYYY-MM-DD}_{HHMM}_{ascii说明}.html`，页内中文标题、头部时间戳到秒、优化必须写提升百分比、页尾复现命令。
- **每版收口必须更新对应线的 CHANGELOG.md**：版本号、时间戳到秒、一句话改动、关键数字（含百分比）、本地/服务器产物路径、复现命令。
- **服务器约定**：`/tmp/{algo|arch|hw|compiler}_v{N}/`，版本号与本地对齐。
- **版本号续历史里程碑**（不重新起号）：负结果轮和过程轮在 CHANGELOG 里记条目但不占号。

## 历史目录（原地不动，用 CHANGELOG 映射表溯源）

`hb_fpga_impl/01~26_*`、`round_report_*/`、`research_*/`、`handoff_r3c/`、`pe_w8a8_sota/` 等历史文件夹保持原样不搬迁——搬迁会弄断几十个 HTML 的相对链接、git 历史也对不上。各线 CHANGELOG 的"历史映射"节给出 版本号 ↔ 历史文件夹 ↔ 服务器目录 ↔ 关键数字 的对照。2026-09-08 之前的工作全部按此方式溯源；之后的新工作一律进三线新目录。
