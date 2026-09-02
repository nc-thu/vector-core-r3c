# 交接包总入口（2026-09-01，R3C 收官形态）

**这个包取代根目录旧的 `handoff/`（2026-08-30 版）**——旧包描述的 16×108、R3 收官形态已过时。旧包可删，留着只作历史参考。

交接对象：接手 `hb_fpga_impl/` FPGA 硬件线的同事。这一版把 R1/R2/R3C 三代优化全部收进来了。文档讲「怎么干、坑在哪、现在到哪了」，数字和细节一律指向仓库里的原始报告（每轮一个 HTML，带时间戳），不在交接文档里复制第二遍。

## 一句话现状

INT8 脉动阵列加速器（跑 HoloBrain-0.2B 推理的 GEMM 主线），R3C 行组流水 + 16×96 阵列：**1536 DSP（88.9%）、120,619 LUT（52.4%）、WNS −1.363 ns（OOC 口径）、12 项回归位精确全绿、只综合未上板**。全帧模型口径 1.67 ms/帧（TB）/ 1.39 ms（HP64），距 60 Hz 实时预算 30 ms 有 20 倍余量。

## 一天上手路线

1. **先跑一遍回归确认环境没坏**（iverilog 串行较慢，约 1 小时；等不及就只跑前三步看位精确）：
   ```bash
   cd "e:/GPU ARCH/vector_core_sim/hb_fpga_impl/22_r3c_rtl/sim" && bash regression.sh
   # 期望末尾 12 项 ALL PASS（日志在 reg_logs/）
   ```
2. 读 [STATUS.md](STATUS.md)——现在停在哪、权威数字看哪个文件。
3. 读 [ARCHITECTURE.md](ARCHITECTURE.md)——模块地图 + R1/R2/R3C 三代优化各自干了什么（为什么 DMA 有两个引擎、为什么 PE 有快照）。
4. 要动代码前，读 [FLOW.md](FLOW.md)——验证矩阵（动了哪类模块必须跑哪些对拍）+ 综合流程 + **实验速度规矩**。
5. 撞到奇怪问题先查 [PITFALLS.md](PITFALLS.md)——五代人踩过的坑都在。

## 目录地图（相对仓库根）

| 路径 | 里面是什么 |
|---|---|
| `hb_fpga_impl/22_r3c_rtl/` | **当前主线 RTL + 仿真**（R3C+96 列，从 14_rtl_r1r2 复制起步）。rtl/ 17 个 .sv 是综合清单，sim/ 是回归。 |
| `hb_fpga_impl/14_rtl_r1r2/` | R1+R2 轮的 RTL 存档（R3C 的前身，零改动保留）。服务器 Verilator 模板 vlbuild_vgate.sh 也在这。 |
| `hb_fpga_impl/12_actv/a3/` | 编译器 + 周期账本（compiler_a3.py / acct_a3.py / gem_cycles 对账）。改 COLS 要重跑这里的编译器。 |
| `hb_fpga_impl/01~21_*/` | 历史轮次存档（模型、量化、调研、报告），逐轮说明见 WORKLOG.md。 |
| `hb_fpga_impl/19_r3c_arch/` | R3C 架构定案 + 模型脚本（r3c_model.py / pe_sizing.py，两个可复算）。 |
| `E:\ae_syn\r3c_c96\` | **当前权威综合工作区**（DSP=1536 那一轮）。基线对照在 `actv_v122_fullchip/`。 |
| `research_holobrain/` | 算法侧调研（为什么是这个模型、算力账、量化门），性能预算的源头。 |

## 环境要求

- **Vivado 2021.2**：`D:\software\Vivado\2021.2\`。综合必须在无空格路径跑（仓库路径含 "GPU ARCH"），工作区模式 `E:\ae_syn\<轮次名>\`。
- **iverilog**：`C:\iverilog`（regression.sh 自己加 PATH）。只用于回归和秒级冒烟。
- **Verilator（推荐）**：在服务器 nc23@101.6.64.77，`/home/nc23/.conda/envs/vsim/bin`，工作区 `/tmp/ae_vgate`，模板 `14_rtl_r1r2/sim/vlbuild_vgate.sh`。比 iverilog 快一个量级，日常验证首选。
- **python**：本机 `/d/ana/python`（numpy）；服务器 `/home/nc23/.conda/envs/holobrain/bin/python`。注意服务器 `/home` 满，只写 `/tmp`。

## 硬规矩（用户裁决，写进 CLAUDE.md 第 9 节）

1. **每轮实验 ≤10 分钟**：超时就缩范围（抽代表段、降规模），不干等。
2. **RTL 仿真优先 Verilator**（服务器），iverilog 只做秒级冒烟。
3. 每轮工作新文件夹新 HTML，文件名/页头/条目时间戳到时分秒；量化结论必须写「提升了 xx%」。
4. 服务器跑完不留后台进程。

## 报告索引（近四轮，全在 hb_fpga_impl/ 下）

| 轮次文件夹 | 内容 |
|---|---|
| `19_r3c_arch/` | R3C 架构定案 + 模型（GEMM 336→185M，−45%） |
| `20_read_compress/` | LOAD 压缩三路线（下一步的方向） |
| `21_r1r2_explain/` | R1/R2 原理说人话（DMA 双引擎 + CTX 预取） |
| `22_r3c_rtl/` | **RTL 落地 + 综合**（本轮收官页）+ PE 选型页 |
| `23_dsp_int8/` | DSP48E2 算 int8 的原理页（给新人扫盲用） |

更早轮次看 `hb_fpga_impl/WORKLOG.md`（每个条目带时间戳，是全项目的时间线权威）。
