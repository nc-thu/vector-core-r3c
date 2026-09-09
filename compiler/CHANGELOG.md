# 编译器与仿真器 CHANGELOG — Compiler & Simulator（共享底座）

管什么：三条线共同改动的基础设施——编译器（compiler.py，trace→段流→manifest）、黄金/快速解释器（golden_interp / fast_interp）、host 驱动（host_driver.py）、trace 工具（trace_sample.py）。**归属纪律：算法线动数值语义、架构线动调度布局、硬件线动指令编码描述符格式**；动哪类就记到哪条线的版本里，同时在本文件登记底座状态，避免再次出现副本失联。
版本规则与命名见根目录 [LINES.md](../LINES.md)。服务器约定 `/tmp/compiler_v{N}/`。

---

## v6 ——（下一版，规划中）权威副本收拢 + 描述符编码分叉收敛
- 内容：把分散的权威副本（hb_fpga_impl/03_compiler、24_pcw_rtn/sw、/tmp/alg_fix/sw）收拢到 compiler/ 一处并版本化；收敛 RTL/SW 描述符编码分叉（RTL=NCW 扩展字+bit28 vs SW=op=14 OP_SF+标志位，对齐清单见 24_pcw_rtn/REPORT_SW.md §10 八条）。

## v5 —— 2026-09-08 11:14 SSA trace + host_driver 三 bug 修复（部署链数值 v4 的载体）
- 改动：trace_sample.py 边名从 id() 改 SSA 唯一编号（4753 边/0 多写/0 无主）+ host 前导注册；host_driver.py 头行错排修复 6 处 + conv im2col padding 一行修。
- 关键数字：编译指纹与正主逐位一致（3118 段）；公平性门不变（fp 回退 27/缺输入 59）；e2e 0.0500/0.0622。
- 权威副本：服务器 /tmp/alg_fix/sw/（host_driver.py 含 .bak_stackview 备份）。
- 本地报告：round_report_routing_bug/2026-09-08_1252_*.html。

## v4 —— 2026-09-07 02:40 --attn-calib 数据通道（attention 实测常数上链，零代码改动）
- 权威副本：26_ref_denoise 的 swfix 链（副本三文件 diff 逐字节相同）。服务器 /tmp/pcw_rtn（未动）+ /tmp/alg_refq。

## v3 —— 2026-09-04 10:31 pcW+RTN 扩展：op=14 OP_SF + 逐列系数
- 改动：新增 op=14（OP_SF 系数装载，每 256b 字 10 个 24b 槽）；GEMM 标志位 rq_s bit7=逐列系数、rq_m bit15=RTN；golden/fast 两解释器真实段逐位一致（fast_selftest 9 桶全过）。
- 关键数字：3118 段 / 362,091 描述符（OP_SF 101,294 字）；权重 blob 196.46 MB；OP_SF 开销 ≈0.13% 整帧。
- 权威副本：hb_fpga_impl/24_pcw_rtn/sw/（03_compiler 26 文件的全量拷贝+改动）。服务器 /tmp/pcw_rtn。

## v2 —— 2026-08-31 13:5x 两大 bug 修复：dma_len 溢出 + 多 tile STORE 偏移
- 改动：dma_len 18 位字段超长溢进 loop 字段（按 DMA_MAX 拆分，+4,599 描述符）；多 tile STORE 漏 byte0 偏移（832 处覆盖，修复后 31,991 输出图零重叠零缺口）。
- 关键数字：ΣSTORE 751,525,888 字节对 manifest；3118 段（v0 为 2782 段）。build_full_v3。
- 权威副本：hb_fpga_impl/03_compiler/。

## v1 —— 2026-08-31（凌晨）编译器 v0：2782 段全量切分，机器位精确
- 关键数字：档 A 机器冒烟 10/10 段逐字节一致（iverilog RTL vs numpy 黄金）；档 B 真实 trace 全量 203,378 描述符零越界；host 步骤 1,424 条。
- 权威副本：hb_fpga_impl/03_compiler/。

## 未占号条目
- 2026-08-31 17:2x compute-bound 调度改造（a1/a2 三档布局）——架构语义归架构线 v2，底座代码在 09_cbound/compiler.py 副本。
- 史前：fast_interp / mk_ideal_calib / probe_so / rtl_seg 工具族（08-31 端到端轮沉淀，03_compiler/NOTES.txt 八节）。
