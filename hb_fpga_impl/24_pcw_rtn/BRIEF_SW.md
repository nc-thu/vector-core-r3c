# BRIEF_SW — pcW+RTN 移植全深度仿真链（阶段1+3，软件线）

生成时刻：2026-09-04 09:46:26。主会话已确认服务器恢复（<SERVER>，主机 ic），/tmp/ae_hostdrv、/tmp/w8a8_ab、~/workspace/holobrain 完整。

## 背景（3 分钟版）

上一轮（research_w8a8_error/REPORT.md）已定位：W8A8 部署后端到端 0.29 rad 红灯的根因是**权重逐 tensor（per-tensor）INT8 量化**——离群通道层误差 7~11%，430 层相干累积。解法 A（已选定）：**逐通道权重（pcW）+ requant 就近舍入（RTN）**，模块边界深度实测 jpos 0.189→0.04715（−75%）。

本轮任务：把 pcW+RTN 移植进**全深度**仿真链（host_driver/fast_interp），在真实样本 s000/s001 上对拍 fp32 参考。判据 **jpos ≤ 0.045**。注意：边界深度 0.047 是模块边界数字，全深度预测区间 **0.07~0.16**，必须实测——这是上一轮结论的诚实边界。

关键基线数字（都要在结果里对账）：
- 全深度部署语义 V0：s000=0.2993，s001=0.2108（判据 0.045，超 5~7×）
- 边界深度 pcW+RTN（V2_rn）：0.04715
- fp 重采样噪声地板：0.0457
- 判据：jpos ≤ 0.045

## 工作目录纪律

- **本轮新文件夹 hb_fpga_impl/24_pcw_rtn/，不改旧文件**。把 hb_fpga_impl/03_compiler/ 的 *.py（26 个，不含 build_*）复制到 24_pcw_rtn/sw/ 后在副本上改。
- 结果 json 落 24_pcw_rtn/results/。
- WORKLOG：每完成一个 Phase 在 hb_fpga_impl/WORKLOG.md 追加时间戳条目（`date "+%Y-%m-%d %H:%M:%S"` 取本地时刻；写清做了什么、数字、下一步）。

## 服务器纪律（硬约束）

- `ssh <SERVER>`（~/.ssh/config 已配好）。python 一律 `~/.conda/envs/holobrain/bin/python`，跑链加 `CUDA_VISIBLE_DEVICES=`（CPU 口径，与 0.2993 基线一致）。
- **只写 /tmp**（推荐 /tmp/pcw_rtn/，别动 /tmp/ae_hostdrv 里的旧 build）。跑完杀干净残留进程。
- ckpt 在 ~/workspace/holobrain/ckpt，工作参照 host_driver.py 头部 "Run (server)" 说明（在 ~/workspace/holobrain 下起 python）。
- **每轮实验 ≤10 分钟**：全深度 host_driver 一遍约几分钟，OK；超时的先缩小（比如先跑 s000 再 s001）。别干等，轮询间隔做点别的分析。

## 语义锚（最重要的一节）

**数值语义的唯一权威是 research_w8a8_error/analysis_02_server_ab.py 里 V2c_pcW / V2_rn_pcW 的实现**（它产出了 0.04715）。移植时逐条对齐：

1. pcW：权重按**输出通道**（GEMM 的列方向）absmax 定 scale，q_w = clamp(round(w/s_pc), -127, 127)。
2. RTN：requant 从截断 `y=sat8((acc·m)>>>s)` 改为就近舍入。ab_quant.py 里怎么写 rn（舍入常数与移位量的关系）就怎么搬——先读懂它，再在报告里**用一段话+公式写清 rn 的精确整数语义**（RTL 同学要照这个改 rq_v2，两处必须逐位一致）。
3. per-column requant：m/s（和 rn）从每 GEMM 一组变成**每输出列一组**。fast_interp 的 GEMM op `y=sat8(acc·m>>s)` 相应改成按列取系数；golden_interp 同步改（两者必须保持逐位一致——fast_selftest.py 就是干这个的，改完必须重跑过）。
4. 描述符/host_plan 怎么扩展（每 tile 多载 (m,s,rn)×列数组）由你设计，但要在报告里写清格式变化，RTL/编译器同学要照此对齐。

## 步骤（建议）

1. **读懂**（本地，30 分钟）：03_compiler/compiler.py（权重怎么量化进 blob、requant 系数怎么算怎么发）、fast_interp.py + golden_interp.py（GEMM op）、host_driver.py（链怎么跑、jpos 哪里算——找 verify_outputs.py 或等价物）、analysis_02_server_ab.py 的 pcW+RTN 数学。
2. **改代码**（本地副本）：per-channel 量化进 build 流程（标定表 absmax 按列算；沿用既有 hw_calib/标定脚本的 8 样本扰动流程的话更好，见 02_quant/hw_calib.py 在服务器上的位置）、描述符扩展、fast/golden_interp 的 per-column+RTN。
3. **位一致自检**：fast_selftest.py（fast vs golden 逐位）必须全绿——这是移植正确性的第一道门。
4. **服务器重建 build**：用 ckpt 重建 pcW 版 build_s000/s001（放 /tmp/pcw_rtn/）。重建耗时先估（编译器全量 build 在服务器上跑过多久，WORKLOG/NOTES 里有线索；超 10 分钟就先只建 s000）。
5. **全深度对拍**：host_driver 跑 s000、s001，jpos 对 fp32_ref。报 s000/s001 两个数 + 与 0.2993/0.2108/0.045 的对账。
6. **若超 0.045**：按 REPORT §5 顺序上杠杆 F（BERT/fusion 少数层 exempt fp 走既有 exempt 机制，零 RTL），每上一档报一次数字；时间不够就把"下一步该做什么"写清楚，诚实收尾。

## 交付

- 24_pcw_rtn/sw/ 改后代码 + 24_pcw_rtn/results/*.json（jpos、逐层 rel 摘要）
- 24_pcw_rtn/REPORT_SW.md：结论一句话、rn 精确语义（RTL 对齐用）、改了哪些文件哪些函数、s000/s001 数字与判据对账、标定流程变化、复现命令（本地+服务器）、诚实边界（没做什么/口径限制）
- WORKLOG 时间戳条目
- 报告末尾给 RTL 线的对齐清单：描述符格式、rn 公式、per-column 系数个数与位宽

## 判据提醒

判据是 jpos ≤ 0.045。预测区间 0.07~0.16 意味着**很可能不过**——不过就如实报数字+分析差在哪（模块间通路？哪条路径的 rel 还高？diag 思路可以复用：服务器 diag 工具的输出格式见 research_w8a8_error/server_data/diag_000_v3b.json）。不要为了过判据改语义。
