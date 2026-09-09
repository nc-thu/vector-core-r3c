# 硬件线 CHANGELOG — Hardware & RTL（电路与实现）

管什么：RTL 改动（ae_top 全家）、综合/布线、功耗、B8 PE 探索、回归验证（12 项位精确）。部署基线的唯一权威 = 本线当前版本。
版本规则与命名见根目录 [LINES.md](../LINES.md)。服务器约定 `/tmp/hw_v{N}/`；综合工作区 E:\ae_syn\。验证纪律：每轮实验 ≤10 分钟、RTL 仿真优先服务器 Verilator（iverilog 只做秒级冒烟）。

---

## v5 —— 2026-09-08 22:08 H2 时序收官 + R2 双读引擎 + B8-64 采点（**当前实现基线**）：eng48 WNS +0.009 @303.215MHz 收敛，定版 B8-48
- 改动：**H2 两条读出腿**——①drain_row 扇出腿换 12 份同值副本寄存器（NREP=(PCOLS+3)/4，每份 4 物理列，打包向量端口过 iverilog）；②requant 入口腿换 rq_ms_x（slot 选择输出寄存一拍 x_sel_r 再进乘法核心，数据滞后 2 拍，FSM 相位前移一拍补偿：rq_v 发射 slot==3→2、换行 slot==2→1、停发/走尾 slot==3→2，requant 消费窗口与 H1 逐拍重合，走读仍精确 64 拍）。**eng64 采点**：top_eng64 包装 + 综合流程加 eng64 目标 + route 级 report_power。**R2 双读引擎**（在 r2/ 子目录、R3C 底本 22_r3c_rtl 上做，底本 git 核查一字未动）：ae_dma 读引擎抽成 ae_rd_eng ×2（CTX/W 各一台，tag 路由）、顶层第二组 AXI 读主口、调度器前台 LOAD 改 fire-and-forget（消费点等空闲）——**架构修正：单改 DMA 不改调度器重叠≈0，光复制引擎没用**。
- 关键数字：eng48 布线 **WNS −0.169→+0.009（+0.178ns，303.215MHz 收敛）**，净代价 +208 LUT（+0.18%）/+733 FF（=24 套×28b rq_ms_x 流水 + 48 副本 FF 的账）；位精确 tb_gemm PCOLS=4/48/64 全 PASS + tb_sys PD=5/6；拍数纪律 48 列 9 描述符 6 个一拍不差、3 个 +4 拍（+0.9~1.5%，DALIGN 入口相位例外），帧级 DRAIN 64→65 敏感性 ≤+0.27%（HP64 0.909→0.910 s）。eng64：LUT 155,629（OOC 67.55%，与锚点外推差 1.2%）、WNS −0.696（拥塞型 7,313 失败端点/TNS −1,531.6ns，非微调可救）、功耗 10.709W vs eng48 8.321W（+28.7%，vectorless）。**定版裁决：B8-48**——三条件两不过（WNS/功耗），降频 250.4MHz 用 HP64=0.970 s 反而比 0.909 s 慢。R2：4 段位精确逐字节一致，段级拍数 −7.4%（141,647→131,235，读服务 −8.1%，两口重叠率最高 15.5%），对照模型理想 −19.1% 的差距=单 outstanding 从机+段形态+无 W 段白付开销；本地全芯片三遍 REF/PRIM/PRIM-pf1 全绿、断言零触发。
- 本地：[v5_2026-09-08_2000_h2_r2_b64_pwr/](v5_2026-09-08_2000_h2_r2_b64_pwr/)（rtl/sim/synth/results + r2/ 子目录 + 2026-09-08_2140 HTML）；综合工作区 E:\ae_syn\hb_h1\；服务器 /tmp/ae_v5r2/（段级对拍）。
- 下一版：H3 B8-48 系统集成（R2 改动合流回底本 + v4 三处底本修复回移 + WRAM 接入）；COPY 段分解交编译器线先量化；R5 写侧确认不进 H3（B8-48 下写腿非瓶颈）。

## v4 —— 2026-09-08 18:10 B8×R3C 集成 H1 门收官：位精确全绿 + 全宽 768DSP 引擎落地，布线差 5%（两条读出路径已定位），修出底本三个潜伏问题
- 改动：ae_pe_p2（Pack2 脉动 PE：1 DSP 出 2 逻辑列积，2×32b acc + 2×27b 快照，积落地 6 拍）、ae_sysarr_p2（16×PCOLS，逻辑列展平读出）、ae_gemm_p2（PULSE_DLY=5 延迟线；requant 套数=逻辑列/4；**修底本三处**：①互锁潜伏 bug——k<59 时脉冲经 row≥12 逃生舱发射后 pend 被旧 walk 完成清零 → 下组脉冲过早发射覆盖未读快照，svc_r 修复，生产 k≥64 从未触发；②地址乘法——喂数侧（行组号+1）×k 与写回侧行组号×n 的 16×16 乘法在 250MHz 从未暴露、@3.298ns 是综合全部 4 条违例（−0.280ns），换基址寄存器 +k/+n 增量；③读出长路径——行选择 mux→requant 桶形移位跨模块单拍路径布线 −1.097ns（R3C 同族弱路径），引擎侧加一拍 acc_rq_r 读出寄存、换行提前到 slot==2 保相位契约、r15_seen 防第 16 行漏读。三处修完位精确与拍数逐项不变）。底本未动。
- 关键数字：末脉冲安全窗口实测 {5,6}（PD=7 丢下组首积——脉冲拍优先级高于累加，比纸面推导窄 1 拍）；位精确 tb_sys 16×4 背靠背 + tb_gemm 15 描述符·轮（PCOLS=4 六个 + 全宽 48 九个：小 k/尾巴/转置/负 rq_m/j0 偏移，两轮修复各复跑全过、拍数逐项相同）；综合 @3.298ns：PE 120 LUT/675MHz 布线后、16×4 条带 7,703 LUT/395MHz 布线后、16×48 阵列 92,055 LUT（39.95%）+768 DSP+WNS+2.054 综合后、**每对 119.9 LUT（kill 线 250 的一半）**；全宽引擎布线后 118,148 LUT（51.28%）+181,333 FF+768 DSP+WNS −0.169（=288MHz OOC 悲观口径，比目标差 5%；读出寄存器修法从 −1.097 收回 0.93ns，探索档 phys_opt 不再改善，剩 drain_row 扇出 + requant 入口两条腿，H2 收尾；R3C 先例真机比 OOC 快 ~6%）；requant+FSM+写回开销 26,093 LUT，全引擎比 R3C-96（120,619）少 2.1%。
- 模型修正（model_corr.py）：v5 读腿公式漏计 ptap 放行（R3C-96 真实 167+wb 拍、B8-48 真实 124+wb，B8-48 每读出界行组反少 43 拍）；帧区间 B8-48 0.909~0.995s vs R3C-96 1.388~1.627s（相对 −34.5%→−38.9%，预算 2.13s 余量 ≥2.1×）；**requant 无需加倍坐实**（24 套=DRAIN 64 拍不变，v5 的 +10.8% 惩罚场景取消）。
- 简化发现：B8-48 映射下 Pump2 第二相闲置 → **无需双时钟**（v4 规划的 600/303 双域 wrapper 取消），单时钟 303.2MHz 每拍每 DSP 2 有效 MAC，0.909s 模型账不变。
- 本地：hw/v4_2026-09-08_1503_b8_h1_gate/（rtl/sim/synth/results + 2026-09-08_1600 HTML）；综合工作区 E:\ae_syn\hb_h1\。
- 下一版：H2 第一优先收掉 0.169ns（drain_row 分组寄存 + requant 入口一拍，位精确/拍数不变纪律重验）+ 集群扩展采点（64→256→768 退化曲线）+ pcW 并入 ae_gemm_p2 + 三处底本修复回移 22_r3c_rtl。

## v3 —— 2026-09-01 20:00 R3C 定版（**当前部署基线**）：16×96 / 1536 DSP，GEMM 引擎拍数 −24%
- 改动：ae_pe（27b 快照寄存器 snap_r，乘加驻留 DSP）、ae_sysarr（快照侧读出）、ae_gemm（FSM 拆喂数/读出道两段并行，行组周期 = max(k+2, DRAIN+DALIGN+2+wb)）、COLS 108→96 全家改；附带修 R2 前置死锁（pf_ctx_stall 的 eng_dma 项）。
- 关键数字：GEMM 引擎拍 REF 5975→4526（−24.2%）/ PRIM −24.7%；总周期 REF 8255→6814（−17.5%）；综合 LUT 120,619（−10.5%）/ FF 150,221（+26.9%，快照成本）/ BRAM 114.5 / **DSP 1536（88.9%）** / WNS −1.363 @250 MHz OOC；12 项回归 ALL PASS。
- 本地：[hb_fpga_impl/22_r3c_rtl/](../hb_fpga_impl/22_r3c_rtl/)、[handoff_r3c/](../handoff_r3c/)（交接包：README/STATUS/ARCHITECTURE/FLOW/PITFALLS）、23_dsp_int8/、2026-09-01_2002 R3C HTML。
- 远端：综合 E:\ae_syn\r3c_c96\。

## 支线（不占版本号）
- **v3-pcw** 2026-09-04 11:27：pcW/RTN 三处 RTL（rq_v2 加 rn_en 就近舍入、rq_ms 单输入改 per-slot 总线、ae_gemm SF_INIT 锁 96 组系数 + NCW 系数字泵）。17 项回归全绿、芯片级 floor 锚逐字节一致、拍数 +1.7%。**已验证未并入基线**（等 compiler 描述符编码分叉收敛）。→ hb_fpga_impl/24_pcw_rtn/REPORT_RTL.md。注意描述符编码与软件线分叉（RTL=NCW 扩展字 vs SW=op=14），合并轮需收敛。
- **B8 研究支线** 2026-09-04 15:48：W8A8 SOTA PE 探索收官——B8 = Pack2×Pump2 + 4 份精确 INT32 状态 + 两级校正累加。位精确（Verilator 10,251 dots / 1,032,501 事务 / 0 错）；单核 OOC 771.6 MHz / post-route 688.7 MHz / 171 LUT / 1 DSP；**64 PE 集群 606.4 MHz，1.213 GMAC/s/DSP = 现 R3C PE 的 6.1×**（含集群退化 11.9%）。→ [pe_w8a8_sota/](../pe_w8a8_sota/)（REPORT.md、results/summary.csv、2026-09-04_154806 B8 总结 HTML）。
- 2026-09-02 18:25 B0~B7 演进史（XtraMAC/朴素抽取/HA1/HA2/ring 各代对比，B4 面积黑马、HA2 频率收益）→ pe_w8a8_sota/2026-09-02_1823 HTML。

## v2 —— 2026-09-01 10:50 R1+R2+AE_ACTV 四模式：读写双引擎 + CTX 预取
- 改动：ae_dma 拆读/写双引擎（AXI 全双工，fire-and-forget 写）；ae_ctx_ram 分区预取 + pf FSM 扩 TAG_C；AE_ACTV 引擎四模式（ACTV/BIAS/NORM 含 AdaRMS/ELTWISE，微观 221,184 字节位精确）。
- 关键数字：全芯片 134,762 LUT / DSP 1728 满片 / WNS −1.038；R2 预取 bug 修复（CTX B 口静默丢写，pf_ctx_drop_cnt 7→0）；seg_0602 PF=1 全 DDR diff 0。
- 本地：hb_fpga_impl/09_onchip_rtl/、12_actv/、14_rtl_r1r2/、17_r2fix_report/。

## v1 —— 2026-08-30 RTL 定版：调度器双发射预取 + T_MAX=39（16×108 / 1728 DSP）
- 改动：ae_sched pf 子状态机（PF_RD/LAT/ISSUE）+ 半区互斥守卫；ae_gemm rq_ms T_MAX 0→39（s∈[21,27] 全链路位精确）。
- 关键数字：预取实测省 404~408 拍（预测/实测差 3.47%）；LUT 110,465（双发射 +650 / T_MAX +3,286）；功耗基线 4,465 mW 布线后 vectorless（85% 在 GEMM 数据通路）；8/8 回归零漂移。
- 本地：hb_fpga_impl/01_rtl/、06_power/。

## 未占号条目
- 2026-09-09 09:48 v5 报告重写版（用户反馈 0908 版不够说人话：数据全部不变，第零节加术语表、一句一件事、图景先行）→ [v5_2026-09-08_2000_h2_r2_b64_pwr/2026-09-09_0948_H2时序_R2双读_B8定版_重写版.html]，页头注明取代 2140 版。
- 2026-09-01 10:37 架构框图（ae_top 寄存器/描述符/引擎/仲裁全景图）→ hb_fpga_impl/16_arch_diagram/。
- 2026-08-31 16:24 INT4 零改动档接线验证（36 段差异全含 text_encoder；nibble 打包 −26.6% 字节已备未上）→ hb_fpga_impl/09_int4_impl/，等算法线基座站稳后重评。
- 史前：hw_zcu104/（2026-08-26~08-28，Evo-1 时代第一代加速器：SM16、rq_ms 集成、packed INT8 证伪、use_dsp 三坑）→ [hw_zcu104/](../hw_zcu104/) 与 handoff/（旧交接包，已被 handoff_r3c 取代）。
