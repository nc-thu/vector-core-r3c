# BRIEF_RTL — pcW+RTN 的 RTL 落地（阶段2，硬件线）

生成时刻：2026-09-04 09:46:26。与软件线（BRIEF_SW.md，另一个代理）并行；你不动软件，只做 RTL+TB。

## 背景（3 分钟版）

上一轮根因研究（research_w8a8_error/REPORT.md §4）选定的方案 A 需要三处 RTL 改动，全部**不碰脉动阵列、累加器、时序关键路径**：

1. **rq_v2 加 rn 输入**：requant 从截断改就近舍入。现语义 `y=sat8((x·m)>>>s)`（见 rtl/rq_v2.sv 头注释的整数恒等式改写：sum = x·mh + ((x·ml)>>>8)，再 sum>>>t，t=s-8）。RTN = 移位前加舍入常数。REPORT §4 的口径："sum_r 从 (sum >>> t_r) 改为 ((sum + rn_r) >>> t_r)，rn = 2^(s-9)（s≥9；s=8 退化为截断或把该列编到 s=9）"。
2. **rq_ms 每列系数总线**：m/s 单输入改成 m_bus[SHARE*16]/s_bus[SHARE*8] 按槽（slot）选择（REPORT 说 x 方向已经是这个模式，照抄）。
3. **ae_gemm SF_INIT 载 96 组系数**：从锁 1 组改为载 96 组（24 个 rq_ms 各 4 slot）。描述符每 tile 多约 9 个 256b 字。

## 语义锚（与软件线逐位一致）

软件代理会在 hb_fpga_impl/24_pcw_rtn/REPORT_SW.md 里写清 rn 的精确整数语义和描述符格式——**写完你的 TB 前先读它**；如果它还没写完，先做代码结构改动和 floor 回归，最后再对拍 RTN 语义。两边不一致 = 白干。

## 工作目录纪律

- **本轮新文件夹 hb_fpga_impl/24_pcw_rtn/rtl/ + 24_pcw_rtn/sim/，不改 22_r3c_rtl 里的旧文件**（R3C 已交付冻结）。
  - 把 22_r3c_rtl/rtl/ 整目录复制到 24_pcw_rtn/rtl/，在副本上改。
  - 把 22_r3c_rtl/sim/ 的脚本+TB（*.py *.sh *.v，**不要** .mem/.vvp/日志，那些是生成物）复制到 24_pcw_rtn/sim/，改 regression.sh 里的相对路径指向 ../rtl（结构同款，应该只改文件清单里加新模块）。
- WORKLOG：每完成一个 Phase 在 hb_fpga_impl/WORKLOG.md 追加时间戳条目。

## 实现要求

1. **rq_v2（24_pcw_rtn/rtl/rq_v2.sv）**：加 `input rn_en`（或等价使能）+ `input [T_MAX:0] rn` 端口……具体形式你定，但必须满足：
   - **floor 模式（rn=0）与旧版逐位一致**——这是回归对照的锚。
   - RTN 模式：`sum_r <= (sum + rn_r) >>> t_r`，rn_r 与 t_r 同拍流水（照抄 t_r 的对齐处理，rq_v2.sv L71-73 有注释讲为什么移位量必须与积流水对齐）。
   - rn 在 T0 与乘积一起寄存，别在移位那拍才组合加（时序上那是加在关键路径上）。
2. **rq_ms**：m/s 总线化 + per-slot 选择；系数载入接口与 ae_gemm 的 SF_INIT 流程对接。
3. **ae_gemm**：SF_INIT 从 1 组扩到 96 组（24 rq_ms × 4 slot），描述符侧多约 9 个 256b 字——具体怎么从描述符搬进来，读现有 SF_INIT 的实现照葫芦画瓢。
4. 位宽/上界自查：报告里给一句"rn 加入后 sum 位宽是否需要 +1"的论证（rn ≤ 2^(t-1)，sum 是 PW=XW+8 位有符号——会不会溢出？不会的话为什么）。

## 验证（全部在 24_pcw_rtn/sim/）

1. **floor 回归**：改完后跑 regression.sh（iverilog 在 /c/iverilog/bin，脚本自己加 PATH；tb_pe_pack_dsp 需要 Vivado unisim，路径写死在脚本里）。rn=0 时与 R3C 基线**位精确一致**——tb_rq（60k 向量，rq_v1 神谕对拍）必须原样全绿。
2. **RTN 向量门**：扩展 tb_rq.sv（或新建 tb_rq_rtn.sv）：
   - 神 oracle：用 python 生成器（参考 sim/gen_rq_vec.py）按软件线的 rn 语义产期望向量，rq_v2 RTN 模式对拍，位精确。
   - 覆盖：随机 m/s/x + rn 满格（s≥9）、rn=0 退化、饱和边界（sat8 上下溢同时撞舍入）。
3. **ae_gemm 级**：SF_INIT 载 96 组后跑既有 GEMM TB（找 sim/ 里对应的 tb，比如 tb_ae.sv 的 GEMM 段或 tb_ae_seg.sv），per-column 系数下输出对拍 python 神谕。
4. 每轮仿真 ≤10 分钟（iverilog 秒级冒烟 OK；分钟级长仿真优先考虑服务器 Verilator——服务器已恢复，ssh <SERVER>，只写 /tmp；iverilog 跑分钟级也行，别跑更长的）。

## 交付

- 24_pcw_rtn/rtl/（改后 RTL）+ 24_pcw_rtn/sim/（改后 TB+脚本+新生成器）
- 24_pcw_rtn/REPORT_RTL.md：三处改动逐个说清（改了什么、怎么验证）、rn 语义与软件线对齐确认（引用 REPORT_SW.md 的公式并确认逐位一致）、资源影响定性估计（REPORT §4 预估：rq_ms ~2.3kbit LUTRAM + slot mux；rq_v2 ~24×35 LUT 加法器；给一句"是否与预估一致"）、复现命令、诚实边界
- WORKLOG 时间戳条目

## 顺序建议

先结构（rq_ms 总线 + ae_gemm 96 组）→ floor 回归绿 → 再 RTN（读软件报告对齐语义）→ RTN 向量门 → ae_gemm 级对拍。每步 WORKLOG 记一次。
