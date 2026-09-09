# R2 读通路定位 findings（底本 22_r3c_rtl，只读不改）

开始：2026-09-08 20:07:03

## 结论先行

ctx 装载（TAG_CTX，tag=0）与 w 装载（TAG_W，tag≠0）**没有并行**，两条流在
**三个串联的排队点**上逐一串行：

1. **同一台读引擎**（ae_dma.sv 单 rd FSM，前台命令与后台预取共用）；
2. **同一组 AR/R 通道**（单 AXI 读主口，TB 从机还是单 outstanding）；
3. **调度器主 FSM 串行发射 + T_RUN_DMA 阻塞等待**（ae_sched.sv）。

另外发现一个**模型侧没提的结构性限制**：即使把引擎和通道翻倍，调度器
现有发射策略下两条流几乎没有机会同时在飞（详见第 4 节）。所以本任务
除"复制引擎+第二主口"外，还必须动调度器的发射策略（前台 LOAD 改
fire-and-forget + 消费点等待），否则收益接近零。这是对架构线 R2 路线
假设的一次修正，属于"重要结论"。

## 1. 排队点一：单台读引擎（ae_dma.sv）

- hb_fpga_impl/22_r3c_rtl/rtl/ae_dma.sv:88 读引擎只有一个 FSM
  rd_st（R_IDLE/R_AR/R_R/R_R2/R_FIN）。
- ae_dma.sv:115 命令锁存：`wire rd_go = start & ~cmd_is_wr | bg_start;`
  —— 前台 LOAD_CTX、前台 LOAD_W、后台预取（bg_start，TAG_W 或 TAG_CTX）
  三类命令全部进这一个 FSM，一次只能跑一条。
- ae_dma.sv:146 在 R_R 拍按 `r_tag_r == 3'd0` 分叉写 CTX B 口或 WRAM，
  说明两流只是同一 FSM 里的两个数据目的地，物理通路从头共享。
- ae_dma.sv:52-59 AR/R 通道只有一组，写引擎独占 AW/W/B，与读全双工
  ——这是 v2 轮已拿到的收益，读侧没动过。

## 2. 排队点二：单组 AXI 读通道 + TB 单 outstanding 从机

- ae_top.sv:38-45 顶层只有一个 AXI 读主口（m_axi_ar*/m_axi_r*）。
- TB 从机（22_r3c_rtl/sim/tb_ae.sv:69-89，tb_ae_v.sv 同构）：接住一个
  AR 后 arready<=0，直到 rlast 消费完才恢复——单 outstanding。
  就算 RTL 侧塞两台引擎进同一通道，也会在从机处串行化。这正是任务书
  "共享一条 64b 总线做字节级交织拿不到收益"的出处。结论：必须两个独立
  主口 + TB 两个独立从机模型。

## 3. 排队点三：调度器发射策略（ae_sched.sv）

- 前台 LOAD 阻塞：ae_sched.sv:222 T_EXEC op=4 未命中预取时要求
  !dma_busy 才发 dma_start，然后 ae_sched.sv:260 T_RUN_DMA 死等
  dma_done 才 T_ADV。dma_busy 是整个 DMA（含 wr 引擎，ae_dma.sv:310），
  所以一条在飞的 STORE 也会挡住前台 LOAD。
- 后台预取单命令在飞：ae_sched.sv:90 pf_v 单 bit；ae_sched.sv:110
  `pf_idle_ok = pf_win && (pf_st == PF_IDLE) && !pf_v && !rd_busy`
  ——预取发射同样被整台 rd 引擎忙挡住。
- 预取窗口：ae_sched.sv:106-107 pf_win = T_RUN_G || T_RUN_SM || T_RUN_A
  ——只有 GEMM/softmax/ACTV 在跑时才发预取。前台 LOAD 期间（T_RUN_DMA）
  窗口关闭，什么都发不了。
- 预取目标只有 pc_next 一条（lookahead=1，ae_sched.sv:104），每窗口
  最多藏一条 LOAD。

## 4. 关键新发现：翻倍引擎后，现有发射策略拿不到重叠

真实段指令流（12_actv/a3/build_a3/segments，2026-09-08 20:05 实测解码）：

- seg_0636：`LOAD ctx(57.5KB), LOAD w(27.7KB), GEMM(k=257), LOAD w,
  STORE, GEMM, ...`——大 ctx 装载全部在段头，前面没有任何 GEMM 窗口，
  预取帮不上；段头 ctx+w 两条装载严格串行。
- seg_0001：`LOAD ctx(262KB), LOAD ctx(212KB), LOAD w(10.4KB), GEMM,
  LOAD w, ...`——同样段头串行。
- 段中 `GEMM → LOAD w → STORE → GEMM` 循环里，LOAD w 已被现有 lookahead=1
  预取部分藏在 GEMM 窗口后（V=108.1M 拍是没藏完的暴露部分）。

调度器不变、只复制引擎的话：

- 前台 LOAD 仍在 T_RUN_DMA 死等自己完成，主 FSM 不会推进到下一条
  LOAD 的 T_EXEC → 两台引擎几乎不会同时在飞（唯一"同时"是后台预取 +
  前台装载，而预取不变量（ae_sched.sv:79-86 注释、tb_ae.sv:146-160
  断言 A）保证 pf_v=1 期间主 FSM 到达的下一条必是预取目标本身——那条
  走 pf_hit 等待，不发新前台命令）。
- 所以必须改发射策略：前台 LOAD 改成 fire-and-forget（发完即推进），
  在消费点（GEMM/COPY/ACTV/STORE 的 T_EXEC）等两台读引擎都空闲。
  依赖正确性：GEMM 是 ctx 和 w 的共同消费者，"等两台都空"保守但安全；
  CTX B 口/WRAM B 口仲裁随之从 eng_dma 单点改成"引擎在跑就选它"。

## 5. 仲裁点清单（改动波及面）

- CTX B 口写仲裁：ae_core.sv:267-296（优先级 eng_g>eng_sm>eng_a>
  eng_dma 前台>bg_wran）→ 后两臂合并成"ctx 读引擎忙"一臂。
- pf_ctx_stall（CTX 预取写让拍）：ae_core.sv:314-315 → 只对 ctx 引擎的
  后台命令生效（前台装载经消费点等待，不可能与 eng_g/sm/a 重叠）。
- WRAM B 口写仲裁：ae_core.sv:332-339（eng_cp > eng_dma||bg_wran）→
  改成 eng_cp > rd_w 忙。
- bg_wran（后台 DMA 授权）：ae_core.sv:243-246 → 按引擎拆。
- 预取 W 半区隔离约束：ae_sched.sv:116-119 pf_issue_ok——语义不动，
  只把 rd_busy 检查改成目标引擎检查。

## 6. 实测结果（2026-09-08 20:36:00 回填，服务器 Verilator，4 代表段，MODE=1 PRIM + PF=1）

| 段 | base 拍数 | r2 拍数 | Δ总拍 | 读侧 union base→r2 | 两口重叠率 |
|---|---|---|---|---|---|
| seg_0600 | 2365 | 2061 | −12.9% | 1993→1719（−13.7%） | 15.5% |
| seg_0529 | 4381 | 4221 | −3.7% | 2284→2310（+1.1%） | 0%（该段无 W 装载流） |
| seg_0602 | 51057 | 48761 | −4.5% | 49787→47020（−5.6%） | 4.8% |
| seg_0636 | 83844 | 76192 | −9.1% | 76886→69299（−9.9%） | 10.9% |
| 合计 | 141647 | 131235 | **−7.4%** | 130950→120348（−8.1%） | — |

- 读侧 union = 任一读口在服务的拍数（TB 端口计数），即读侧总服务拍。
- GEMM 拍数逐段 ±5 拍不变（seg_0602 3165→3160，seg_0636 30723→30722）：双引擎不扰阵列。
- 预判的三条边界全部证实：
  1. **同流仍串行**：seg_0602 的 W 流 45870 拍自身几乎不重叠。该段 r2 union=47020，
     距理论下界 max(rd1,rd2)=45870 只差 2.5%——W 主导段已接近 max(X,V) 上界。
  2. **无 W 流的段白付拆分开销**：seg_0529 rd2=0，union 反而 +26 拍（+1.1%，引擎
     路由/预取簿记开销），但前台 LOAD 不再被在飞 STORE 挡（!dma_busy 含 wr_busy），
     总拍数仍 −3.7%。
  3. **小段离上界远**：seg_0600 union 1719 vs 下界 1176（+46%）——段太小，段头装载
     占主导，重叠机会有限；大段（seg_0636）离下界 13.6%。
- 架构模型（b8_scan.py 读通道 X+V→max(X,V)，帧级 −19.1%）是"两流随时可重叠"的
  带宽上界；RTL 实测 4 段读侧 −8.1%，差距来自真实程序形态（段头大 ctx 装载前无
  GEMM 窗口、循环体只有单条 LOAD w、单 outstanding 从机）与消费点等待约束。

## 7. 段级 golden 对拍的预先存在问题（与 R2 无关，如实记录）

4 个 staging 段的 RTL dump 与 fast_interp_a3 golden 存在非零 diff（2199/64/3350/3928
字节），但 **base 与 r2 的 diff 逐段完全相同**，且与 MODE=0/1、PF=0/1 无关——是
底本 22_r3c_rtl 时代就存在的口径问题，不是双读引擎引入的。排查记录：

- diff 集中在 STORE 写出的 CTX 数据区（STORE 写 W×16 字节，如 seg_0600 在
  addr=144 写 4096B，而 manifest outputs 只标 256 词——首轮分区统计用错尺度）。
- 服务器上 9 月 1 日的旧 vgate 二进制（R3C 定稿前一代 RTL）今天跑同一 staging
  同样 diff>0（seg_0600 diff=4245，更大）→ 段级 golden 门在此环境下从未绿过。
- R3C 定稿的正式门是 12 项回归（gen_vectors/sim_ae/compare + actv 三件套），
  不含段级 golden 对拍（见 22_r3c_rtl 同日 HTML 报告验证矩阵）。
- 本轮 r2 的位精确主张因此落在两层：① 本地 12 项口径中的 gen_vectors 全芯片
  三遍（default + actv）全绿；② 服务器 4 段 r2 与 base 的 dump 逐字节 cmp 一致。
  段级"对 golden 位精确"对 base 和 r2 都不成立，属于待查的独立问题（怀疑
  fast_interp_a3 的 GEMM/requant 语义与 22_r3c_rtl 有出入，需单开一轮）。
