# R2 双读引擎：RTL 实现与验证报告

生成时刻：2026-09-08 20:40:00 | 执行人：硬件线 v5 R2 子任务 | 底本：hb_fpga_impl/22_r3c_rtl/（零改动）

## 一句话结论

DMA 读侧拆成两台引擎（ctx 一台、w 一台），配第二个 AXI 读主口，再把调度器的
前台 LOAD 从"死等完成"改成"发完就走、消费点再等"。结果：4 个代表段总拍数
**−7.4%**、读侧服务拍 **−8.1%**，位精确与基线逐字节一致（本地合成向量对拍全绿 +
服务器 4 段 dump 逐字节 cmp 一致）。架构模型预期的帧级 −19.1% 是理想上界，
RTL 实测拿不到那么多，原因见"收益边界"。

## 1. 为什么必须动调度器（定位阶段的最重要的发现）

底本里两条装载流在三个点排队（详见 notes/findings.md，带 file:line 证据）：

1. 只有一台读引擎 FSM（ae_dma.sv:88/115/146）——TAG_CTX 和 TAG_W 命令进同一个状态机；
2. 只有一组 AR/R 通道（ae_top.sv:38-45），TB 从机还是单 outstanding；
3. 调度器 T_RUN_DMA 死等（ae_sched.sv:222/260），且 !dma_busy 里含写引擎——
   一条在飞的 STORE 也能挡住下一条前台 LOAD。

只复制引擎不动调度器的话，主 FSM 在 T_RUN_DMA 里等着，根本走不到下一条 LOAD
的发射点，两台引擎几乎不会同时在飞——收益接近零。所以本轮改动 = 引擎翻倍 +
第二主口 + 前台 LOAD 改 fire-and-forget（消费点 GEMM/COPY/ACTV/STORE 的 T_EXEC
等两台读引擎都空闲再启动）。

## 2. 改了什么（4 个 RTL 文件，全部在 r2/rtl/，文件头有 diff 说明）

| 文件 | 改动 |
|---|---|
| rtl/ae_dma.sv | 读 FSM 抽成子模块 ae_rd_eng，例化两台：u_rd_c（tag=0 → CTX B 口）走读口 1，u_rd_w（tag≠0 → WRAM）走新增读口 2。前台命令按 cmd_tag 路由，后台预取按 bg_tag 路由（bg 脉冲拍让位，同底本 bg 优先语义）。写引擎逐字保留。 |
| rtl/ae_sched.sv | 预取簿记按引擎拆（pf_v_c/pf_v_w 等）；op=4 前台 LOAD 改 fire-and-forget，命中预取则等对应引擎的 pf_done；消费点（op=3/5/6 与 GEMM 默认臂）等 rd_idle=!rd_c_busy&&!rd_w_busy（op=5 还要 !wr_busy）。 |
| rtl/ae_core.sv | 新增第二组读主口；CTX B 口仲裁的 eng_dma+bg_wran 两臂合并成 rd_c_busy 一臂；WRAM B 口 DMA 臂改成 rd_w_busy；pf_ctx_stall 收窄为"ctx 引擎的后台命令 + eng_g/sm/a 在跑"；bg_wran 触发器删除。 |
| rtl/ae_top.sv | 新增 m_axi2_ar*/m_axi2_r* 端口并连线（寄存器映射零改动）。 |

TB（r2/sim/）：tb_ae.sv（iverilog 三遍冒烟）与 tb_ae_v.sv（Verilator 段执行）各加
第二套独立单 outstanding 从机（独立 LFSR 停顿源，seed 31C3）、LFSR 快照/装载对齐、
预取不变量断言按引擎拆两套，新增断言 C（前台 ctx 装载不得与 eng_g/sm/a 重叠——
消费点等待被破坏时才会触发）+ 读活动计数探针。

死锁/丢写论证（写在 findings.md §4-5）：消费点等待发生在 T_EXEC，此刻 eng_*=0，
所以 pf_ctx_stall=0、CTX B 口仲裁一定选得到读引擎臂；前台 ctx 装载因消费点等待
不可能与串行引擎重叠（断言 C 守卫）；WRAM 半区隔离约束（b_base 高位 ≠ 当前
GEMM 半区）语义不动。

## 3. 位精确证据（两层）

层 1——本地 iverilog 全芯片三遍（R3C 12 项回归的 gen_vectors+compare 口径，
2026-09-08 20:38:45，各约 40 秒）：
- default 用例：CTX 16384B + DDR 65536B 逐字节一致；PRIM-pf1 dump 与 PRIM-pf0 逐位一致。
- actv 用例：同上，全部 PASS；prim2==prim 逐位一致。
- 断言 A/B（预取不变量）+ 新增断言 C 全程零触发。

层 2——服务器 Verilator 段级（2026-09-08 20:25-20:36，4 段 × base/r2，每段秒级）：
- `cmp dump_seg_XXXX_base.mem dump_seg_XXXX_r2.mem`：**4 段全部逐字节一致**。
- 这是"双读引擎不改变计算结果"的直接证据（时序变了、数据一位不变）。

已知边界：4 个 staging 段的 dump 对 fast_interp_a3 golden 有非零 diff（2199/64/
3350/3928 字节），但 base 与 r2 逐段**完全相同**，且与 MODE=0/1、PF=0/1 无关；
用 9 月 1 日的旧 vgate 二进制复跑同样 diff>0（甚至更大）。即段级 golden 门在
22_r3c_rtl 时代就没绿过（R3C 定稿门的 12 项里也不含它）。这是独立于 R2 的
预存问题（怀疑 fast_interp_a3 的 GEMM/requant 语义与 RTL 有出入），本轮不认领、
建议单开一轮查。详见 findings.md §7。

## 4. 收益数字（服务器 4 代表段，MODE=1 +PF=1）

| 段 | base 拍数 | r2 拍数 | Δ总拍 | 读侧服务拍 base→r2 | 两口重叠率 |
|---|---|---|---|---|---|
| seg_0600（头段小） | 2365 | 2061 | −12.9% | 1993→1719 | 15.5% |
| seg_0529（无 W 流） | 4381 | 4221 | −3.7% | 2284→2310 | 0% |
| seg_0602（W 主导） | 51057 | 48761 | −4.5% | 49787→47020 | 4.8% |
| seg_0636（头段大） | 83844 | 76192 | −9.1% | 76886→69299 | 10.9% |
| 合计 | 141647 | 131235 | **−7.4%** | 130950→120348（−8.1%） | — |

GEMM 拍数逐段 ±5 拍不变——双引擎不扰动阵列。本地合成负载：default REF 6814→6554
（−3.8%）、PRIM 6618→6342（−4.2%）；actv PRIM 9848→9592（−2.6%）。

对照架构模型（b8_scan.py：读通道 X+V=301M→max=184M，帧级 −19.1%）：模型假设
两流随时可重叠，是带宽上界。RTL 实测到不了，三个原因——
1. 同流装载仍串行（每口单 outstanding）：seg_0602 的 W 流 45870 拍自身无法重叠；
   该段 r2 读侧 47020 已贴近下界 max(rd1,rd2)=45870（差 2.5%）。
2. 真实程序形态限制重叠面：段头大 ctx 装载前没有 GEMM 窗口，循环体每轮只有一条
   LOAD w，预取 lookahead=1 就够（lookahead=2 在真实段里找不到两个 GEMM 间的
   双 LOAD，做了也是死代码，故未做）。
3. 无 W 流的段（seg_0529）白付拆分开销（读侧 +1.1%），但总拍数仍降——收益来自
   前台 LOAD 不再被在飞 STORE 挡（底本 !dma_busy 含写引擎）。

## 5. 诚实边界（没验证的项与假设）

- **综合/资源未跑**：架构线预估 +2.4k LUT（rtl_requirements_h3.md），本轮未做
  Vivado 综合，LUT/DSP/时序影响未实证。CTX B 口与 WRAM B 口仲裁逻辑变简单了
  （两臂并一臂），预期不增反减，但这是推断。
- **12 项回归只跑了 2 项**（gen_vectors default + actv 三遍）；tb_rq/tb_sm16/
  tb_pe_pack/gem_cycles 等 10 项与读通路无关且 RTL 未动它们的模块，未重跑。
- **段级 golden 对拍不绿**（预存问题，见 §3 边界）；r2 的段级正确性主张是
  "与基线逐字节一致"，不是"与 golden 一致"。
- **pf_stall 探针在两个合成负载里都是 0**：没碰到"ctx 后台预取撞 B 口被占"的
  场景，pf_ctx_stall 冻结路径只被断言和代码走查覆盖，没有被负载实际压过
  （底本 R2 轮的 actv 大块预取负载 4096B CTX 也许能压到，未验证）。
- **TB 从机是行为级单 outstanding**：真 PS-DDR 的仲裁/重排/带宽共享没建模，
  实机重叠率会低于 TB 读数。
- **帧级外推未做**：4 段合计 −7.4% 不能直接乘到整帧（段形态分布不均），
  要按 acct_a3 全帧账重算才算数。
- 服务器工作目录 /tmp/ae_v5r2（rtl_base/rtl_r2/两套 obj/脚本）保留未删，可复跑。

## 6. 文件清单

```
hw/v5_2026-09-08_2000_h2_r2_b64_pwr/r2/
├── rtl/   ae_dma.sv  ae_sched.sv  ae_core.sv  ae_top.sv     （★ 改动，头部有 diff 注）
├── sim/   tb_ae.sv   tb_ae_v.sv   stage_seg.py              （★ 新/改）
│          gen_vectors.py compare.py rsqrt_lut.mem exp2_lut.mem（底本拷贝，脚本相对路径需要）
│          stage/seg_{0600,0529,0602,0636}/                   （staging，0636 为新加）
├── notes/ findings.md（定位+实测回填） REPORT.md（本文）
└── results/ seg_compare_2026-09-08_2036.txt  smoke_local_2026-09-08_2038.txt
```

## 7. 时间线（2026-09-08）

| 时刻 | 事项 |
|---|---|
| 20:05-20:07 | 读底本代码 + 段指令流解码，定位三个排队点 |
| 20:08:24 | findings.md 落盘（§1-§5 定位结论 + 预期边界） |
| 20:11-20:14 | rtl 四文件改写（ae_dma → ae_sched → ae_core → ae_top） |
| 20:15-20:19 | 本地 iverilog 冒烟 default 三遍 + compare 全绿（含 prim2==prim） |
| 20:21 | stage_seg.py + seg_0636 新 staging；服务器 rtl/obj 部署（/tmp/ae_v5r2） |
| 20:22-20:25 | 服务器 Verilator 4 段 × base/r2 首轮：拍数 −7.4%，bitdiff 逐段相同 |
| 20:25-20:34 | diff 定位 + MODE/PF 无关性核查 + 旧 vgate 二进制复跑 → 判定段级 golden 为预存问题 |
| 20:36:00 | 4 段复跑取完整 rdstat；cmp 证明 base/r2 dump 逐字节一致 |
| 20:38:45 | 本地 default + actv 双用例三遍冒烟重跑全绿（落 results/） |
| 20:40:00 | findings §6/§7 回填、results 存档、本报告 |

## 8. 复现

- 本地：`cd r2/sim && python gen_vectors.py && /c/iverilog/bin/vvp.exe smoke.vvp && python compare.py`（actv 加 `--case actv`；smoke.vvp 为 r2 RTL 预编译产物）
- 服务器：`/tmp/ae_v5r2/run_r2.py`（4 段 base vs r2 全表；dump 直比 `cmp runwd/dump_seg_*_base.mem runwd/dump_seg_*_r2.mem`）
- 服务器流程遵 handoff_r3c/FLOW.md（Verilator --timing --binary，conda env vsim）
