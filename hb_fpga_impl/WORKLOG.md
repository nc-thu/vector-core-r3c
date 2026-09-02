# WORKLOG —— hb_fpga_impl

规则：每完成一块工作追加一节；问题单列"碰到的问题"；子代理产出由主会话汇总落盘。

---

## 2026-08-30 · 开工：摸底与工作区

**任务拆解**（用户指令）：
1. RTL 加调度器双发射 + WRAM 双缓冲，硬件电路定版。
2. 服务器上把 HB-GD 模型和 RoboTwin 数据集编译成 runtime 二进制。
3. 二进制喂加速器：性能拍数 + 功耗 + 推理结果，与数据集真值对比。
4. FPGA 实现级（用户拍板：不搭 Vitis/bitstream，上板后置）。
5. 新文件夹分子文件夹组织，维护工作记录，子代理并行。

**摸底结论**（3 个 Explore 代理，详细数字见各代理报告）：

- RTL：描述符 256b，OP_GEMM/ATTN_S/HOIST/COPY/LOAD/STORE/DONE；requant 是**每个 GEMM 一组静态对称 scale**（Q8.8 乘 m，s=8 固定），无 bias 端口、无 per-channel——编译器必须逐 GEMM 校准，量化口径和上一轮 per-token 实验不同，要先校准。WRAM 真双口、SEQ 执行期读口空闲、CTX B 口有真冲突 → 双发射 v1 只预取权重（TAG_W）。已知陷阱：DMA 命令寄存器是当前描述符组合切片（ae_core.sv:173-177），双发射要加影子寄存器。
- 仿真基建：回归 8 项全绿 ~3 分钟；TB 是 COLS=12 冒烟档、DDR 模型 64KB、看门狗 2M 拍——跑真模型要扩容 + 分段。50M 拍级仿真无先例，是本轮最大风险。功耗从未跑过，从 synth.tcl 加 report_power 起步。
- 模型侧：HB-GD=BIP3D，forward 链已理清；怪算子（deformable 采样、PSE 反投影、rotary、norm/gelu、FK/DPMSolver）走 CPU 侧，PL 只跑 GEMM+ATTN_S——用"分段指令流 + 段间 host 编排"，不加新 RTL 原语。量化资产（hb_quant.py、fixture.pt、weight_stats.json、117 条 gemm_items 形状表）都是现成地基。
- 数据集：用户选 RoboTwin 官方数据集小子集（HF sdvkasc/robotwin，100K+ HDF5 轨迹）；HB 仓库 README 说数据要用官方仿真器自生成，服务器无 sapien → 格式兼容性是风险项，验证失败则退 bringup 合成样本。

**工作区**：hb_fpga_impl/ 01_rtl ~ 08_report 建好。

**并行开工**（6 个子代理）：
1. ~~Plan 代理：RTL 双发射信号级设计~~（已完成，见下）。
2. V100 校准实验（02_quant）：硬件口径 per-tensor 静态 scale + K+1 bias 增广 + Q8.8 requant，顺产编译器常数表。
3. RoboTwin 数据集（04_dataset）：HF sdvkasc/robotwin 小子集 + 格式兼容验证 + 样本/真值/fp32 基准。
4. 执行序 trace（03_compiler 服务器侧）：forward hook 抓算子执行序 IR + INT8 权重导出 + 服务器装 iverilog。
5. Verilator 仿真引擎（05_sim）：iverilog→Verilator 移植 + 全参数位精确对拍 + 吞吐基准（54M 拍可行性判定）。
6. RTL 双发射实现（01_rtl）：照设计文档改 ae_sched/ae_core/ae_top + pf 用例 + 四道验收门（零漂移回归/逐位一致/综合 LUT≤+3000/gem_cycles 更新）。

**RTL 双发射设计裁决**（Plan 代理产出，要点）：
- WRAM 双缓冲走 **b_base 半区对分**（bit11=缓冲号）：零 BRAM、零 864b mux、GEMM/DMA 地址通路零改动；代价 k≤2048——HB 全部 GEMM 的 k 最大恰好 2048（FFN fc2），零余量可用。ping-pong 整片复制被否（+108 BRAM 且在 864b 喂数通路上加 mux，时序盲区）。
- 前瞻深度 1（双缓冲结构上只容 1 个在飞预取）；发射窗口 T_RUN_G/T_RUN_SM；pf 子状态机 PF_RD→PF_LAT→PF_ISSUE 共 3 拍，SEQ 读口执行期空闲复用。
- **陷阱①②的解**：DMA 命令影子寄存器组在 PF_LAT 拍锁存（发射前一拍），start 拍 mux 选择，无竞争；pf_en=0 时 mux 结构性退化直通。
- 等待点=被预取 LOAD 自己的 T_RUN_DMA（pf_hit 命中→等 pf_done），单 DMA 引擎零改动复用（消费门控保证 D_IDLE 永不见第二条 pending 命令）。
- 硬件守卫：半区互斥 + d_k≤2048 违纪自动退化串行（编译器纪律错误不会变数据错误）。
- 预估 300-700 LUT、~95 FF、copy 交叉路径（WNS 临界）零改动。

**编译器架构定案**（COMPILER_SPEC.md）：分段指令流+host 编排；段内全 PL（GEMM+ATTN_S），段间 host 跑 CPU 算子（norm/gelu/deformable/solve/rotary/窗口重排/去噪外循环）；每段 = seq.mem + ddr_init.mem + manifest；host 驱动在服务器跑（torch 做 CPU 算子）；三道验证门（单段位精确/端到端 vs fp32/拍数 vs 模型 <10%）。

**仿真引擎决策**：54M 拍全模型 iverilog 串行撑不住（十几小时起），段间又有数据依赖没法并行——上 Verilator（快 10-50 倍），仿真搬服务器；iverilog 位精确对拍做验收，Verilator 不达标就退 iverilog 分段跑。

---

## 2026-08-30 · 执行序 trace 完成（03_compiler 第一步）

- ops_trace.json：一次完整推理 3003 条算子（gemm 1459 = Linear 1429 + Conv 30；elem_norm 778；custom 498；attn 268）。**去噪循环严格 227 条/步 × 10 步**，周期性对上；traced forward 输出与 fixture 真值逐位一致（trace 的就是真实推理）。
- 阶段边界：backbone(155)→neck(164)→backbone_3d(319)→neck_3d(328)→text_encoder(465)→text_feat_map(466)→feature_enhancer(651)→spatial_enhancer(662)→decoder(3002)。
- 权重导出：431 个 int8 + 403 bias + 16 额外（含 6 个 MHA in_proj 裸 Parameter）= 160MB 留服务器；421/421 Linear 全带权重键。
- iverilog 装进 holobrain 环境（Icarus 13.0），没破坏现有包。
- **坑**：①quant_harness 实际在根目录不在子目录；②fixture 只存张量、模型还要 text/kinematics 列表，得 processor 重建对拍；③MHA 必须 eager + with_kwargs=True 才挂得上钩；④id 数据流边只有 578 条严格边，编译器以形状匹配为准；⑤**服务器 /home 只剩 55GB**（17T 盘 100% 挂载报警），后续大文件要小心。

## 2026-08-30 · RoboTwin 数据集完成（04_dataset）

- **来源**：HF TianxingChen/RoboTwin2.0（第一作者官方仓库）place_empty_cup 任务，aloha-agilex 本体，50 集×179 步。用 zip HTTP Range 只抽中央目录+5 集（~55MB），没下整包。
- **兼容性：映射后兼容（实证）**。官方是 e71140e 之后的 v1.0 格式，四项映射全通：相机改名（cam_third_view→front 等）、外参换算 T_world2cam=inv(cam2world_gl@diag(1,-1,-1,1))（真实数据验证误差 1.6e-7）、关节序 [左臂6,左夹爪,右臂6,右夹爪] 直接对上、RGB 通道序坑（RoboTwin 把 RGB 当 BGR 编 jpeg，HB decode 后 flip 恰好还原）确认无害。
- **唯一合成项：深度**——官方数据集根本没存深度（pointclouds 也是空的），用真实内外参+桌面平面合成，已标注。fp32 MAE 仍低 → 模型对该先验不敏感，短期够用。
- 3 个样本（不同 episode/指令）：sample_00K.npz（imgs [4,256,320,3]、depths、hist_robot_state [14,8] FK 后 link 位姿、input_ids 18-21 token）、truth_00K.npz（future_actions [64,14] 与 action 字段逐帧全等）、fp32_ref_00K.npz。
- **fp32 策略偏差基线 ≈ 0.044-0.064 rad**（2.3-3.7°/关节，样本间稳定）。含义：加速器输出 vs fp32 参考若远小于这个数 → 加速器复现了策略；加速器 vs 真值的偏差应与 fp32 vs 真值同量级（都是策略偏差不是数值错）。
- 坑：桌面高度估计顶到 0.90m 上限（该任务双臂不深探桌面）；HB packer 的文件名 int() 解析会卡官方 episode_0000000.hdf5（只影响训练管线，不影响推理用例）。

## 2026-08-30 · 硬件口径量化校准完成（02_quant）——判绿，带出两个硬发现

**结论**：部署口径（per-tensor 静态 A8 + per-tensor W8 + 整数 requant + K+1 bias 增广）端到端 jpos MAE **0.02881**（3 seed 均值，噪声底 0.0113 的 2.54 倍）≤ 0.030 门限，**绿**。模式 A 复跑与上轮逐位一致（环境没变）。

**硬发现 1：requant 乘数方向**。任务书公式 r=so/(sa·sw) 是反量化方向（写反了）；正确 **r=(sa·sw)/so**（缩小，实测 1.55e-4~1.56e-2）。第一版照抄跑出全模型输出砸死 ±127；用 RTL 自带 FA 测试向量（m=64,s=8→r=0.25）证实缩小方向。

**硬发现 2：s=8 固定移位跑不了这个模型 → RTL 必须改一个参数**。s=8 时 172 层 m 取整成 0（输出全零）、258 层精度饥饿、0 层达标；实测全模型需要 **s∈[21,27]、m∈[16424,32754]**。rq_v2 模块本来就支持 s∈[8,47]（T_MAX=39），ae_gemm 实例化写死 T_MAX=0——**改成 39 即可，已追加给 RTL 代理**（连同 s>8 位精确新用例+综合增量）。

**bias 三分支**：314 层 K+1 增广成功（c=2~64 分布）；**88 层放不下**（|b|/(sa·sw)=8262~93836，int8×c 容量只有 8128——根源 b/(sa·sw)=16129·|b|/(|a|max·|w|max)，bias 与输入×权重同量级就爆）→ host 段补 fp bias；28 层无 bias。45 层若 c 放宽 128（用 -128 码）能塞下、43 层怎么都塞不下。

**其他**：acc 最大 818k ≪ 27b 上限；8 扰动样本校准后 eval 超校准 5% 层只剩 6/430、饱和层 0；top 误差层= Swin s2/s3 ffn.layers.1 + BERT output.dense（K=3072 输出远小于输入的层，相对漂移 0.58-0.84）；requant 纯 floor 无舍入 → 每 GEMM -0.5 LSB 系统偏置会累积（已含在 0.0288 里；想再压，RTL 加 2^(s-1) 舍入常数是最便宜的下一步）。

**交付**：hw_calib_table.json（421+10 全量，编译器常数源：m_requant/s_shift/bias_aug_c/w_bias_int8/bias_fp_fallback）+ hw_calib_results.json + hw_calib.py（含 fp64 核心与纯整数参考逐位一致自测）+ log，本地 02_quant/ 与服务器 hw_calib/ 双份。集成契约已发编译器代理。

## 2026-08-30 · RTL 定版：双发射 + T_MAX 四道门全过（01_rtl）

**电路冻结**。改动：ae_sched（pf 子状态机 PF_IDLE/RD/LAT/ISSUE、硬件守卫半区互斥+k≤半区、命中/消费路径）、ae_core（DMA 命令源 2:1 mux + bg_wran 授权 WRAM B 口）、ae_top（CTRL bit2=pf_en）、ae_gemm（rq_ms T_MAX 0→39 一行）。回退点 rtl_bak_pf0/。

- **门 1 零漂移**：regression 8/8 PASS，REF 8405/5973/844、PRIM 8149/5720/843 与 MEASURED 逐数相等；pf 不发射的用例三遍 dump 逐位一致（结构性零漂移证明）。
- **门 2 预取**：PRIM-pf1 8009 < pf0 8413（省 404 拍），dump 逐位一致且 golden 位精确；后台发射 11 次全命中零串行兜底；预测省 418 vs 实测 404，差 3.47%（门限 10%）。
- **门 3 综合（一次过）**：WNS −1.038 与基线完全相同、最差路径仍是 copy 交叉；BRAM 122.5/DSP 1728/URAM 64 持平；LUT 106,525→110,465：**双发射 ≈+650（预估 300-700 内）、T_MAX=39 = +3,286**（27 套 rq_ms×35b 桶形移位；T_MAX=19 可省 1/6，留作选项）。占片 47.9%。
- **门 4**：gem_cycles pf 感知记账（PF_CMD_OVH=1、dma_busy 双口径），smoke --pf PASS。
- **T_MAX 验证**：rqs 用例 m∈[9000,32754]、s∈[21,27] 全链路位精确；s=8 老用例拍数不变。
- 坑：①LFSR 停顿相位跨 run 漂移造成 pf1 假多 24 拍（快照/回放对齐解决）；②pf1 改变到达相位使 gemm 计数 +3 拍（rq_ms slot 自由轮转对齐，良性）；③iverilog $time 按模块单位缩放；④dump 大小写；⑤Windows python 写不了 /tmp。
- 设计偏差 1 条：pf_raddr 端口未引出（内部 mux 完成，core 层无人消费）。

**功耗线开工**（RTL 定版解除阻塞）：A 档 vectorless 基线 + B 档 SAIF 流程（vcd2saif + read_saif 链路用冒烟 VCD 先打通，模型活动 VCD 后补）。

## 2026-08-30 · 功耗完成：基线 4465 mW + SAIF 流程打通（06_power）

- **三个口径**（xczu7ev-ffvc1156-2-e，250MHz，typical/25°C）：综合态 vectorless 4285 mW；**布线后 vectorless 4465 mW（无活动数据时的引用口径，实现置信度 High）**；冒烟 SAIF 6058 mW（仅流程验证）。真实翻转比 vectorless 假设费电（DSP 808→1838 mW），真实数等模型 VCD 注记。
- 布线做法：OOC 全 opt/place/route，时钟放松 10ns 跑 router（4ns 极慢），报功耗前改回 4ns。布线后 DCP 存 E:\ae_syn\pwr\ae_top_impl_ooc.dcp 供后续注记。
- **电在哪**（布线后）：u_core 3841 mW 里 u_gemm 2924（u_arr 2473 + 27 套 requant 270）+ WRAM 540 + u_ctx 123 + u_cp 101 + 控制 <85——**85% 在 GEMM 数据通路**。按资源 Signals 956 / DSP 808 / CLB 806 / BRAM 599 / Clocks 557。
- **SAIF 链路坑**（重要，已解）：SAIF 必须整体包 (SAIFILE ...) 且 DIRECTION "backward"，缺外壳解析器不报错但注记 0%；read_saif 2021.2 无 -input/-instance_name，默认剥两层 INSTANCE，层次用 INSTANCE 嵌套表达。自写 vcd2saif.py（限窗 VCD→SAIF），冒烟注记 1540 网。
- 注意：OOC 网表无真实 IO 负载（Vcco 全 0），上板会略高；tb_ae.sv 的 VCD 采样改动全部 ifdef 保护（默认关，回归 cycles 复核一致）。

## 2026-08-31 · 编译器 v0 完成（03_compiler）：2782 段全量切完，机器位精确

- **档 A 机器冒烟 10/10 段逐字节一致**（iverilog RTL vs numpy 黄金解释器，全 DDR dump 对拍）。覆盖：conv im2col+K+1 增广、无 bias、fp_fallback、k 超半区、末行组 pad 预清零、BertAttention 走 OP_ATTN_S、WindowMSA 每窗现载、豁免层整层 host。冒烟在本轮救过一次场（重构丢"末行组预清零"立刻 9/9→5/10 抓住）。
- **档 B 真实 trace 全量**：2782 段 / 203,378 条描述符（GEMM 69,414 / COPY 54,032 / LOAD_CTX 42,246 / LOAD_W 15,494 / STORE 19,410）；最大段 SEQ 1636≤2048 ✓、最大段权重 3.65MB≤16MB ✓；weights_blob 196.46MB（同权重跨段缓存复用）；host 步骤 1,424 条。独立后检全过（DDR 区间零重叠、203,378 条寻址零越界、半区断言零触发）；26 个真实段（1,647~236,504 拍）numpy 黄金完整执行无错。
- **预估 313M 拍 ≈ 1.58s @198.5MHz**（Fmax=1/(4ns−1.038ns WNS)，copy 交叉路径限制，250MHz 修复一直明确不在范围）。这是 v0 朴素调度的**下界估计**：段独立 → 激活反复 LOAD_CTX（42K 条）、swin 每窗现载 proj、全部输出 STORE 再读回；实测是估计的 2-3.4 倍（估计模型不含 LFSR 读停顿/仲裁），真实值大概率 3-5s 量级。与账本 CONC 0.27s 的差距主因=无跨段权重驻留+段切分开销，优化空间明确（BERT 缓存/hoist/驻留是后续性能轮）。
- **k 走 host-bias：119 层**（fp_fallback 88 + k>2048 增广超限 31；按去噪 10 步实例算 290）。
- **注意力分配**：OP_ATTN_S 12 个（BertAttention）；两相 GEMM+host softmax 220 个（rotary 120/temporal 60/swin 24/jg 4/mha 6/bimha 6）；MSDeform 整段 host 6 个。
- **缺口**：26 个 decoder.head.convs.* requant 常数占位（补丁代理已发）。
- **坑 12 条**（NOTES.txt），严重的：①黄金解释器把 DDR 字节按无符号进 CTX（负 int8 变 128-255），golden 全错 RTL 对——差点带偏排查；②trace 里同 module 执行 10 步，owner 归属要按"时间上随后第一条匹配实例"而不是全局最深前缀；③Linear in_shapes 三维时行数口径（通道维错算进 m，qkv 的 m 达 225 万）；④大激活（backbone FFN A 图 7.9MB）超单列组要按行分块独立落段；⑤swin 权重段内复用（不然 64 窗 7.6MB 撑爆段）。

## 2026-08-31 · requant 常数补丁（02_quant）：26 警告 = 8 个唯一 key，全补齐

- 26 条占位警告去重后是 8 个模块：decoder.head.convs.0/1（**nn.Conv1d**，上轮只 hook Linear/Conv2d）+ feature_enhancer.text_attn_blocks.{0..5}.self_attn.attn.**in_proj_weight**（MHA 融合权重无子 Linear 可 hook；key 带下划线后缀，补丁已按此对齐）。
- 8 个全部走 K+1 增广（无 fp_fallback），m∈[17704,32268]、s∈[23,25]、acc 最大 132k（27b 上限的 1/500）、|m/2^s − r| ≤ 4.8e-08。合并表 **hw_calib_table_v2.json 共 439 条**（431+8，无重复 key 已 assert）。下一轮编译 --calib 指到 v2，警告清零。
- **服务器 /home 100% 满**（1KB 写不进）——本次全程在 /tmp 跑（根盘 111G 空闲），完事已清。后续服务器工作一律先查磁盘，必要时用 /tmp。已通知 Verilator 代理。

## 2026-08-31 · Verilator 迁移收尾：新 RTL 跨工具对拍全绿 + 吞吐基准（05_sim）

**结论**：iverilog→Verilator 位精确迁移在新 RTL（双发射 pf + T_MAX=39）上验收通过。resync_server.sh **39/39 PASS**：4 用例（default/tail/pf/rqs）× 三遍（REF/PRIM-pf0/PRIM-pf1），两工具 DDR dump 逐字节一致、与 golden 位精确、pf1==pf0 数据逐字节（预取只省拍不改数）、cycles 跨工具全等。预取实测：pf/rqs 用例 8409→8001 拍（**省 408，~4.9%**）；default/tail 无收益（负载里没有可后台预取的权重装载——零漂移的结构性佐证）。

**吞吐**（64 核服务器，他组负载 ~5.4）：Verilator 大负载档（COLS=108，1.49M 拍 GEMM 密集）**6845 拍/s**（218s）→ 54M 拍 ≈ **2.2 h**、全链 313M ≈ **12.7 h**；iverilog 全参数档 ~10-30 拍/s（先期低负载口径）→ 54M 要 21-63 天，**全模型仿真只能 Verilator**。冒烟档两工具差 320×。全部数字与口径在 05_sim/bench.json。

**两处跨工具分歧根因都是 TB 写法，RTL 零改动**：start/rst_n 用阻塞赋值在 posedge 后与 always_ff 竞态（iverilog 先评估 DUT、Verilator 先跑 TB 进程）。调度器早 1 拍不改变固定流水线拍数，但 DMA 读停顿由绝对时间上的 LFSR 决定 → LOAD 路径 cycles 漂移；复位释放竞态使 rq_ms 的 slot 自由轮转计数器错相 → S_DALIGN 等待拍数变化。修法=全改 NBA。同进程连跑第二遍的 LFSR 相位残留（PRIM Δ≈8 拍，数据不变）是 TB 属性非工具差异，判据放行并在脚本头写明。

**新坑（新向量才踩到）**：负载会读少量从未写过的 CTX padding 字。基线 tb_ae.sv 层级清零=00（golden 亦 00）、Verilator 2 态=0、iverilog fresh 进程=X → "xx" 漏进 dump（default 用例 168 字节，四个新用例全中）。修法：tb_ae_v.sv 加 `ifndef VERILATOR` 的 t=0 层级清零，三环境语义对齐。

**段执行接口已就位**（对接编译器 segment_runner 的 +SEQ/+DDRIMG/+DUMP）：+SEQ 复位后经 seq_we/seq_waddr/seq_wdata 端口装满 SEQ RAM（PS 装载通道，不改冻结 RTL），+DDRIMG/+DUMP 直读直写；双工具验证 dump 与基线逐字节一致。注意点（./seq.mem 固定文件名占位、+SEQ 推迟 start 使 LFSR 相位差几拍、看门狗参数、每进程 readmemh 固定开销 ~2.9s）写在 05_sim/NOTES.txt。

**提速空间判断**（协调者问题，NOTES 有细节）：conda verilator 默认 OPT 是 **-Os 不是 -O3**；现实空间 2-4×（去 --timing 换 C++ main 最大单项 1.5-2.5×；-O3+march=native 1.15-1.4×；--threads 0-2× 不稳）。但 ~300 段并发 16-32 进程等效 10-20× 且零工程量——**优先段并行**，引擎优化只在核填不满或单段成尾延迟时做。

**服务器环境**：/home（17T）100% 满 0 字节可写——运行树全部迁 /tmp（aers=新链、aerun=旧链）；旧 RTL 验收链的 iverilog 全参数档在负载下慢 3×（单遍 REF 9300s 未完），按指令废弃（日志有 [skipped-deprecated] 标记，新链同套 TB 更严用例已覆盖）。/home 的 ae_sim 清到 19MB（删 1.1G pip 缓存 .mmroot、VCD、vvp/dump 可再生物）。交付物（tb_ae_v.sv/bench.py/bench.json/emit_big.py/resync_*/NOTES）已回本地 05_sim。

**坑**：①pkill -f 的模式串出现在自己 ssh 命令行里会自杀会话（退出码 255，杀进程用 pkill -x 精确名）；②iverilog 不接受 string 变量与字面量混用的三元（报 string/bool 类型错），改 if/else；③emit_big 借 gen_vectors_108 的 wmem 写文件，落点是 gen_vectors_108 所在目录——会把 sim108 的冒烟向量覆盖成大负载档（加 os.replace 搬回 sim_big，覆盖过一次已恢复）；④iverilog 的 `ls dump_*.mem` 探查要防 tail/awk 管道把无匹配吞掉造成误判。

## 2026-08-31 · 全模型分层测拍完成（05_sim）：1.08G 拍 @ 部署口径，挖出编译器 dma_len 溢出

**总量**：2782 段按描述符流字节全等去重 = 441 类，每类代表段 Verilator 双模式实测（REF 与 PRIM+预取），总数 = Σ(类拍数 × 实例数)。**ref 1,114,545,402 拍 / pf1 1,080,821,046 拍（预取省 33.7M，3.0%）= 5.44s @198.5MHz / 4.32s @250MHz**。MAC 对账 RTL=223.76G（padded）=compiler 逐类全等，useful（去 padding）145.8G——**有效 MAC 占峰值 7.8%**，v0 朴素调度搬运占大头（如最重类 seg_0625：3.34M 拍里 DMA 2.27M）。阶段分解（pf1）：decoder 537.4M（2117 段）、feature_enhancer 226.5M、backbone 145.6M、spatial_enhancer 101.9M（仅 10.4G MAC 却吃 9.4% 拍数）、backbone_3d 36.4M、text_encoder 30.4M、其余 <2M。

**dma_len 编码溢出（本轮最大 bug，测拍时发现）**：desc() 把 dma_len<<61 塞 18 位字段 [78:61]，超长 LOAD/STORE 高位溢进 is_loop_end/in_loop/steps。窄化≠0→静默少搬（数据错不挂死）、==0→remain 下溢死循环（看门狗 FATAL）。影响 1276/2782 段、2932 条描述符，其中必挂死 2 条（seg_0142/0158 的 1.31MB STORE）。fix_streams.py 外科拆分（不重编译）：STORE/LOAD-CTX 按 DMA_MAX=262128 切、LOAD-W 按 108 对齐 261792 切；校验 Σ修复后 STORE 字节=660,897,024==manifest 输出 words×16，0 段不一致。**compiler.py 正式修法待回填**（本轮只有流级修复）。教训：18 位字段装 18 位以上值，RTL 与黄金解释器读法一致所以对拍抓不到——只有跑超长段才暴露。

**est 对账三档**（est_check.py 按 441 类逐条）：est_v0=313.0M（口径复现无误）→ est_lenfix=544.5M（修长度读法 +74%）→ est_fixed=854.8M（再修 gemm mt 双重除法 +57%）→ 实测 1080.8M。剩余 226M（21%）= LFSR 读停顿/仲裁/段内固定开销，估计模型仍缺。**gem_cycles mt=ceil16(m)//16 双重除法**还坑了 compiler macs 字段（Σ=8.03G vs RTL 223.76G），修正口径 ceil16(m)×16×COLS×k 后逐拍全等。

**验证**：①数据无关性 5 类×激活全零 vs 随机×两模式拍数全等（拍数=描述符流的确定函数，实证）；②确定性 3 类×2 遍全等；③11 代表段全参数黄金对拍 **10/11 位精确**（覆盖 9 阶段+最长流+80 实例类）。**唯一失配 seg_0221**：11 段中唯一 op=1 ATTN_S 段（BERT 12 头×8×8 非因果 softmax），5923/12288 输出字节系统性偏大 ~5%，MAC/拍数全等→softmax 数值路径；已排除 exp LUT/argmax/mx±1/除法差 1 等单变量假设，**根因未定，需 SM16 内部探针**（39/39 验收没覆盖 8×8 非因果形状）。拍数不受影响。

**SAIF**：top3 贡献类（t010/t027 全段、t625 前 10GB 窗口）已转 SAIF 在服务器 /tmp/ae_cycles/vcd/，**尚未 scp 回本地、未 read_saif 出数**（被中断）。

**坑**：sim_fast 旧二进制只认 +MODE/+PF（%s plusarg 静默无效靠 cwd 兜底，口径没坏但 +DUMP=/dev/null 是空话，多耗 ~20GB dump）；vcd2saif 必须 --t-start=VCD 首时间戳（给 0 会把 SEQ 装载段补成 X 稀释翻转率 25%，vcd_run.py 已自动探测）；seg_0625 全量 VCD 52.9GB→切前 10GB 转 SAIF 后删（回收 53GB）。产物清单见 05_sim/MEASURE_NOTES.txt 第五节（types/sweep/aux/golden/cycles_by_type/TOTAL + 脚本，多数还在服务器 /tmp，TOTAL/MEASURE_NOTES/cycles_by_type/typify/measure 已回本地）。

## 2026-08-31 · host 驱动端到端数值（03_compiler）：**进行中，被中断**

host_driver.py（52KB）已写好：装段→Verilator 跑→解析 dump→反量化→CPU 算子→量化进下一段；样本 000/001 已 trace（trace_s000/s001.json）；build_full 已用 calib v2 重编译（08-31 03:54）。**result_*.npz 未产出**——进程退出时数值链没跑完。恢复时注意：build_full_fixed（流修复版）与本地 build_full（v2 校准版）是两层修复，**合并=compiler.py 正式修 dma_len 后用 v2 校准重编译**。判据：action MAE ≤0.045（fp32 策略偏差基线 0.044-0.064 rad 同量级）。

## 2026-08-31 10:30 问题讨论页（实测 vs 账本 16×）+ 模块级资源账

- 10:17 从布线后 DCP 跑层级资源账（E:/ae_syn/pwr/hier_util.tcl → hier_util.rpt）：
  u_arr 42,267 LUT/1728 DSP、u_cp 18,410、requant 27 套 16,940、u_gemm 胶水 12,117、
  u_sm 10,164、u_dma 1,469、u_sched 999、u_ctx 260（+64 URAM）、WRAM 108×RAMB36、
  u_core 合计 103,846（45.1%）。布线后比综合阶段少 ~6.5k LUT（110,465→103,958）。
- 10:30 问题讨论页落盘：08_report/2026-08-31_1030_实测vs账本差距问题讨论.html
  - 16× = 1.98（口径：4 相机 vs 2 视角，四科目比值 1.97–2.06 齐刷刷指向线性扩展）
    × 1.53（padding）× 5.27（搬运停顿，63%→12% 利用率）。
  - est 三级台阶澄清：两个估算 bug 修掉（+74%/+57%）+ 真开销 26%；est_fixed×1.26≈实测。
  - 新规范第一版图：模块 LUT × 阶段时间占比（qa_modules/qa_time）+ 对表比值图 + 16× 瀑布图。
  - 行动项：per-opcode 忙拍计数器（Verilator 侧统计，不改 RTL，半天）→ 把 5.27× 归因到根因。
  - 决策点 D1–D4 待拍板：口径对齐（建议 C 双口径 + A 对照实验）、搬运优化顺序、padding、
    调研结果并入后是否开新一轮综合。
- 生成脚本：08_report/_gen_qa_page.py（模板 + 内联 SVG）、check_html.js（自检，新页 PASS）。

## 2026-08-31 10:55 问题讨论页 v2：并入 on-chip 调研结果

- on-chip 激活/归一化调研完成（07_onchip_ops/，未动 RTL/编译器）：
  - 归因账（2782 段指令流解码 × RTL 常数，与 441 类实测差 1.8%）：GEMM 599.1M（55.4%）/
    STORE 206.6M（19.1%，3.2 B/拍最慢）/ LOAD_CTX 164.9M（15.3%）/ LOAD_W 75.4M（7.0%）/
    调度+LFSR 19.6M / COPY 15.2M（1.4%——按指令条数排序会看错重点，按拍数 STORE 才是第一大项）。
  - host 算子边界往返共 148.8M 拍（13.8%），norm 族占 49%。
  - 方案：AE_ACTV 统一行引擎（norm/actv/rotary/bias/softmax-bias 五模式）+ swin 散射；
    MVP 净省 82.8M 拍=7.7%（5.44→5.03s），+swin 97.0M=9.0%（→4.96s）；
    资源 ~16k LUT / 7 BRAM / 0 DSP（预算 60k LUT）。
  - 下一堵墙量化：编译器行分块边界 STORE+LOAD_CTX 222.8M 拍（20.6%）＞全部算子融合之和；
    搬运全消后利用率上限 14.1%（形状匹配问题，对应 D3）。
  - 四步落地路线（actv+bias 打底 → V100 定点验 norm → softmax 描述符 → rotary/swin），
    编译器配套（op=6 段合并/表装载）必须同步否则无收益。
- v2 页面：08_report/2026-08-31_105*_实测vs账本差距问题讨论.html（7 图 41 tip，自检 PASS；
  v1 10:30 保留）。新增 2 图：qa_cycle_mix（总拍归因）、qa_onchip_gain（净省拍/算子）。
  根因表按拍数重排；D2 推荐改为 BERT 缓存 → AE_ACTV MVP → 编译器大段/流式 → 预取窗口扩展。

## 2026-08-31 11:28 问题讨论页 v3：并入 INT4 调研结果

- INT4 权重量化调研完成（07_int4/，服务器无遗留进程）：
  - 推荐档：BERT per-tensor INT4 + Swin RGB 塔 g=128 + neck 卷积 per-tensor，其余 W8。
    权重字节 159.5→102.6 MB（−35.6%）；合成 0.0231 rad / 真实最差 0.0315，与 W8 基线
    （0.0259/0.0288）同噪声带，低于 0.04 红线（端到端判据 0.045）。
  - 三档：零改动（BERT only，−26.7%，0.0187，只换常数表今天可上）/ 推荐（−35.6%）/
    激进（42–49%，真实出现 0.056–0.065 坏例，不建议）。
  - 关键发现：占 53% 字节的 BERT 恰好最不敏感（85M 大块分布平整）；g64≈g128；
    QoQ 式解包回 INT8 的零 RTL 路线在 Swin 上不行（0.079 rad）；逐层贪心不可靠
    （单层边际仅 +0.0003~0.002，伤害是扩散头累积+混沌放大，必须组合实测）；
    W8 基线跨进程漂移 10%（0.0288 vs 0.0259），对比必须同进程。
  - 方法筛选：冻结 per-tensor 激活筛掉 SmoothQuant/AWQ/QuIP/SpQR；可用 GPTQ/HQQ/RTN+组scale。
  - 硬件：−35.6% 字节直接改善 LOAD_W/驻留；推荐档需组 scale 累加时反量化（RTL 项，
    建议与 AE_ACTV 凑同一批综合——都动 u_core）。
- v3 页面：08_report/2026-08-31_1128_实测vs账本差距问题讨论.html（8 图 47 tip，自检 PASS；
  v1/v2 保留）。新增 qa_w4_modules 图；决策点扩到 D1–D5（D5=INT4 档位，推荐中间档）。
- 待收：端到端数值链（最后一个后台代理）→ 到货后出 v4。

## 2026-08-31 12:53 答疑页：数据集/视角/CTX/指令集

- 回答用户四个问题，产出 08_report/2026-08-31_1253_数据集视角与指令集答疑.html
  （自检 PASS，1 图 6 tip，内联复用 sw_desc_mix）。
- 核心内容：
  - RoboTwin vs LIBERO 三原因：LIBERO 后训权重未开源（自训 V100 数天级）；
    HB-GD 必须喂深度而论文 LIBERO 96.7 用 MuJoCo GT 深度，RoboTwin 用真实内外参
    +桌面平面合成（唯一合成项），两边公平性等价；官方小子集 HTTP Range 55MB 当天跑通。
    并入新洞察：实测 145.8G=账本 RoboTwin 档 149.8G 的 97.4%，"1.98×"是档位差不是实测丢东西。
  - 视角表：4 相机（third+左右腕+head，149.8G 档）/LIBERO 2 视角（73.7G 档，拍数约减半）/
    1 视角（无官方档，算法决定）。相机间无交叉注意力→乘加线性扩展（四科目 1.97–2.06 指纹）。
  - 换 LIBERO 拆两问：硬件同口径数字→不用换，跑 D1-A 双视角对照（一天）；
    benchmark 成功率→算法线的事（自训权重+MuJoCo 管线）。建议现阶段不换。
  - CTX 白话：64 URAM 激活缓存（16 路 bank 交织喂 16 行阵列），与 WRAM（108 BRAM 权重）
    分工表；LOAD_CTX=op4/b_src=0；42,246 条多的根因=段独立用完即扔。
  - 指令集总览：7 opcode 语义表（含各自条数与拍数）+ 256b 字段位表
    （op/a_src/b_src/sm_causal/y_tr/m/n/k/a_base/b_base/y_base/b_spad/rq_m/rq_s/inv_idx/
    steps/in_loop/is_loop_end/dma_len/j0[77:62 与 dma_len 共段]/dma_addr/保留 29b）
    + 三机制（硬件循环/inv 跳过/requant 折进指令）；op=6 已留给 AE_ACTV。
- 同时派出两个新后台代理：compute-bound 编译器改造（09_cbound/，门=−≥20% 拍+GEMM≥70%
  +位精确）、AE_ACTV RTL+INT4 落地（09_onchip_rtl/、09_int4_impl/）。
  端到端数值链代理仍在跑。三个都回来后出"解决方案与效果"页（全部写提升了 xx%）。

## 2026-08-31 13:5x 端到端数值链定案：链路零误差，量化方案红灯

- 端到端代理完成（3.6h，服务器无遗留进程，慢档 RTL 全链 nohup 继续在跑 47/3118 段 ~38h）。
- 编译器修掉两个 bug：
  - dma_len 18 位溢出（>262,128 字节高位泄进 loop 字段，静默少搬/死循环）——
    编码时按 DMA_MAX 拆分，与 05_sim 外科修复精确一致，描述符 +4,599。
  - 多 tile STORE 漏 tile 偏移，后 tile 覆盖前 tile（静态扫描 832 处）——
    _emit_store 加 byte0，修复后 31,991 个输出图零重叠零缺口。
- 三道校验门全过：ΣSTORE 字节=751,525,888 对 manifest；RTL vs 黄金解释器 6 段位精确；
  段输出与 int64 精算 0/1,966,080 不一致。fast_interp 与黄金解释器逐位一致（快档可信）。
- **红灯（核心结论）**：全链 per-tensor W8A8 + 每条 GEMM 输出回 INT8（残差流也是 INT8，
  全深度量化）下，样本 000 MAE 0.2993 rad / 001 0.2108 rad，超 0.045 判据 5–7×
  （fp 自身重采样底 0.0457/0.0439）。
- 五步定界：不是实现 bug，是方案本身——
  ① 输出坍缩：动态关节（8/9/10/13）方差归零、joint1 恒 0.669、左夹爪恒 0.5（先验中心）；
  ② 理想校准表（按本样本各层 absmax 重造，192/438 层被放大）后 13/14 关节逐位不变；
  ③ 单层 rel 0.1755 中 0.1640 来自 A/W int8 本身，requant+输出 int8 只加 0.011，
     段链实测==量化理论仿真精确相等；
  ④ 只量化 GEMM、注意力全走 fp → 0.2942，几乎全差；
  ⑤ BERT mask 排除（21 token 全有效）。
- 与 08-30 软件量化门"W8A8 绿灯 0.0110"不矛盾：软件门是浅深度（模块边界 fake-quant、
  残差流 fp），硬件是全深度（每条 GEMM 输入输出都 INT8）。绿灯不能外推到部署口径。
  与 SwiftVLA"激活 INT8 掉 12pp"同源。
- 影响：D5（INT4 档位）价值被 gate——基座 W8A8 都过不了判据，INT4 字节收益要等
  量化方案升级后才有意义；compute-bound 改造不受影响（拍数与位精确口径不变）。
- 交付物：build_full_v3 / build_s000_v3 / build_s001_v3 / build_s000_ideal、
  result_000_v3b.npz / result_001_v3.npz、03_compiler/NOTES.txt（八节）、
  新工具 verify_outputs.py / mk_ideal_calib.py / probe_so.py / rtl_seg.py；
  服务器 /tmp/ae_hostdrv/、/tmp/ae_v3/。
- 新决策点 D6（量化方案升级，等用户拍板）：A 逐 token/逐通道动态激活量化
  （需 RTL absmax，与 AE_ACTV 同族）；B W8A16（数据通路改宽，requant 重做）；
  C action_head 前特征通路混合精度；D 先用 fast_interp 做纯软件档位扫描再选。

## 2026-08-31 15:3x AE_ACTV 引擎 + INT4 零改动档落地（09_onchip_rtl/、09_int4_impl/）

- AE_ACTV 片上算子引擎（op=6，ACTV/BIAS 两模式）：
  - 微观对拍：8 随机用例（行组尾数/列宽尾数/表长尾数/负乘子/饱和角）
    20480/20480 字节精确。调通中抓两个真 RTL bug：尾组行掩码读了未锁存的 m_r
    （拿到上一条描述符的行数，修法 row_mask 显式传 m）；iverilog genvar 位选择
    求值不可靠+武装写时序差一拍。黄金脚本自身一个 bug（把执行完的 CTX 终态当初态
    dump）修掉后用例 6 假通过变真通过。
  - 全芯片回归：op=6 黄金语义扩展进 gen_vectors.py 副本，默认用例向量逐字节不变
    （md5 一致），--case actv 加 3 处 op=6 + 3 张表 LOAD；四项位精确全过，
    regression.sh 12/12 ALL PASS。
  - 编码：b_src 复用为子模式（0=ACTV 1=BIAS），256b 布局不动。坑：ACTV 表映像必须
    把表项 x 复制到字 tbl+x 全部 16 槽位（512B）——CTX 广播读按槽位对号，
    "每 lane 只收 x%16==L"的第一版布局是错的；BIAS lo/hi 分区槽位恰好正确。
  - 综合 OOC（xczu7ev-2，目标 250MHz）：整引擎 4507 逻辑 LUT + 640 LUTRAM + 985 FF
    + 2 BRAM36 + 0 DSP，WNS −0.229ns（≈238MHz，高于实测 198.5MHz 时钟）。
    门槛两模式 ≤2.5k LUT，实测 5147，超 106%——只 ACTV 1307 达标；BIAS 4410 是大头
    （16 份 8b×16b 无 DSP 乘法，每 lane ~234 LUT）。下一轮换乘法结构
    （查表分解/共享乘法器分时复用）。相对全片 ~120k 空余 LUT 不构成容量问题，
    关键是扩到五模式时的缩放。
- INT4 零改动档（BERT per-tensor，07_int4 推荐档的零 RTL 部分）：
  - 接线全链验证：编译产物与 W8 对照结构完全一致（3118 段、260,797 描述符），
    差异恰好 36 段且全部含 text_encoder 权重、无误伤。requant 链实证：
    BERT GEMM 的 rq_m/rq_s 按新尺度重算，有效乘子恰好放大 127/7=18.14 倍。
  - 零改动档本身字节节省 0%（设计使然：int4 网格值装在 int8 容器里，
    15,494 条 LOAD_W 与 W8 逐字节全同）。真节省在 nibble 打包（w4_packed/ 已备好）：
    BERT 84.9MB→42.5MB（−50.0%），折合 int8 导出总量 −26.6%（与 07_int4 的 26.7% 对上）。
  - 拍数（est 模型同口径）：搬运字节 597.5→545.2MB（−8.75%）；LOAD_W est
    147.49M→134.11M 拍（−9.07%）；全链 est 626.91M→613.53M（−2.13%，
    3158.2→3090.9ms @198.5MHz）。全链只有 2.1% 是因为 LOAD_W 占总拍 23.5%、
    BERT 又只占 LOAD_W 的 17.5%。
  - 端到端精度（服务器 fast 链，样本 000，3118/3118 段跑满）：W8 对照 0.2993 rad
    与已发布基线逐位复现；W4-BERT 0.3010 rad，Δ+0.0017（+0.57%），14 关节里
    12 个逐位不差（仅 joint4 自身 +19.6%、joint13 +1.7%）。BERT 不敏感在真实链上
    坐实——INT4 对输出几乎无感，0.29 底噪是 W8A8 基座固有的（端到端代理已定界）。
- 服务器：/tmp/ae_w4/（build+双方 result+日志），代理进程全部退出，
  慢档 RTL 跑未受影响。新坑：host_driver 动态 import compiler 需整目录拷贝
  /tmp/ae_hostdrv/*.py；Vivado 变体综合必须各用独立 out 目录。

## 2026-08-31 16:24 三份总结页（架构/算法/电路，09_summary/）

- 应用户口令："图文 HTML 总结距上次总结以来的工作，分架构、算法、电路三份"。
  新文件夹 09_summary/（遵守新轮次新文件夹规矩），三页自检 PASS：
  - 2026-08-31_1624_架构总结.html（6 图 31 tip）：16× 拆解+口径对表、拍数归因账
    （含"按条数排序看错重点"的自我修正）、est 阶梯、on-chip 方案、编译器两 bug、
    指令集与 op=6、compute-bound 进行中、决策点 D1–D6 状态表。
  - 2026-08-31_1624_算法总结.html（3 图 15 tip）：端到端定案（三门+位精确+红灯数字）、
    五步定界、软件门 vs 全深度口径对照表、D6 四选项（推荐软件扫档先行）、INT4 三档+
    端到端 A/B+零改动档字节 0% 更正、数据集维持原判。
  - 2026-08-31_1624_电路总结.html（4 图 26 tip）：模块资源账完整表（LUT/DSP/BRAM+
    拍数占比，u_dma 1469 LUT 承担 41.4% 拍 vs u_cp 18410 LUT 承担 1.4% 拍的反差）、
    AE_ACTV 位精确落地与两个 RTL bug、综合 5147 LUT 超门槛 106% 拆解、
    INT4 nibble 打包与 requant ×127/7 实证、慢档 RTL 在跑。
- 新图表 3 张（09_summary/gen_charts2.js，复用 08_report 规范）：algo_mae
  （四变体 MAE vs 判据）、algo_w4_tiers（INT4 三档字节）、hw_actv_lut（引擎 LUT 拆解）。
  复用内联 9 张（从 08_report/charts/ 拷入 09_summary/charts/）。
- 覆盖范围 = 08-31 上午三份工作记录之后；compute-bound 代理仍在跑，明确标注不在本份内。

## 2026-08-31 16:5x 方向聚焦：停硬件/算法线，只留架构

- 用户拍板：硬件和算法工作全部停掉，专注架构（compute-bound 改造）。
- 硬件线：无在跑代理（AE_ACTV+INT4 已于下午收工）；服务器慢档 RTL 全链
  经查已不在进程表（最后已知 47/3118 段），不重启；AE_ACTV 乘法结构重构
  等后续硬件活不排期。算法线：无在跑代理（端到端上午已定案）；D6 软件扫档
  不启动，决策点挂起。
- 服务器现状：唯一在跑的是架构代理的验证集群（/tmp/ae_cb，xargs -P 18
  逐段 Verilator 实测拍数，约 2 小时），保留。
- 架构中间态（主会话直接观察，代理未交卷）：已产出 build_a1 / build_a2 /
  build_repro 三套编译产物 + cycles_build_a1.tsv（369 段）/cycles_build_a2.tsv
  （517 段）逐段拍数表 + gate_a1/gate_a2 逐段位精确门目录；已向代理发状态
  询问（A1/A2 改动内容、拍数降幅、门进度、ETA）。

## 2026-08-31 17:0x 实验速度裁决：杀 Verilator 集群，改 ≤10 分钟模型评估

- 用户裁决：18 路逐段实测太慢（跑了约 2 小时）；之后**每个子实验 ≤10 分钟，
  周期预测准确率 ≥95% 即够**。已存为长期规矩（memory/experiment-speed-rule.md），
  以后派代理必写进硬约束。
- 已执行：服务器 /tmp/ae_cb 全部进程（q.sh×2、xargs×2、cyc.sh、Vtb_ae_v）
  杀干净并确认 ALL_KILLED；服务器无我们的遗留进程。
- 重规划（已下发给 compute-bound 代理）：
  - 已测数据不浪费——cycles_build_a1.tsv（369 段）/ a2（517 段）/ repro 转为
    校准+验收集，解析/快档模型对齐它们，报告逐段偏差分布+总量偏差，≥95% 即收；
  - A1/A2/repro 三套全部软件模型评（总拍、GEMM 占比、各项 vs 基线百分比），
    每评一次 ≤10 分钟；
  - 位精确门只抽 3–5 代表段/变体（≤10 分钟），不再全扫；
  - 先交快报（A1/A2 改动内容+已测段初步拍数对比+GEMM 占比），不等新实验跑完。

## 2026-08-31 17:2x compute-bound 快报：编译器两档合计 −18.9% 拍，位精确全过

- 三档（只改 09_cbound/compiler.py 调度/布局，数值语义零改动）：
  - build_repro=基线复刻：3118 段 seq.mem 与服务器 new_full 逐一 md5 相同，
    已测段 cycles 与 gate2 实测逐段分毫不差（锚点闭合）。
  - build_a1=杠杆A（段界/列块合并）：DDR 档 8MB→64MB（8MB 是 TB 遗产，
    dma_addr 32b 本可寻址 4GB），3118→2762 段，消 A 图重复搬运与 M-feed 重复。
  - build_a2=a1+杠杆B/C/D：多头段 A 图整段一次装载（原每头重装 16 次，
    heads-outer 重排）；V-T twin 段 A 单装；WRAM 权重驻留表（半区命中不重发）；
    LOAD_W 紧贴 GEMM 发射（pf 窗口内 100% 遮蔽）。
- 模型口径（≤10 分钟，用 repro 已实测 1468 段拟合每拍成本
  gemm×1.05/store×1.18/load_ctx×2.14/load_w×2.13/copy×1.10）：
  逐段偏差中位 0.7%、p90 6.8%、总量偏差 0.00%——达标（≥95% 准确率规矩）。
  - 总拍：repro 1216.5M → a1 1134.0M（−6.8%）→ a2 986.4M（−18.9%，−230M）。
  - 分量 repro→a2：LOAD_CTX 417→226M（−46%）、LOAD_W 132→104M（−21%）、
    STORE 276M 不变（751MB 全是唯一输出，编译器剪不动）、GEMM 346→336M、COPY 40M。
  - GEMM 占比 28.5%→29.7%→34.1%。
- **两个诚实更正**：
  ① 原任务书的"GEMM 55.4%→≥70%"目标口径有误——55.4% 来自老账本计数 bug
  （用了全局 n 而非 n_loc），RTL 实测 gemm 计数器占比只有 ~30%；单 DMA 引擎
  串行下 STORE+LOAD 占 ~63%，调度器救不动，属 RTL 需求主项。
  ② 基线 1216.5M 是逐段实测锚定口径；此前 05_sim 的 1080.8M 是老模型估计值，
  与本口径不可直接混用。旧归因账（GEMM 55.4%/STORE 19.1%/LOAD_CTX 15.3%，
  2782 段旧流）在新口径下重排为 LOAD_CTX 34.3% 第一、GEMM 28.5%、STORE 22.7%
  ——已要求代理在最终报告里把口径差说清。
- 位精确门全过：静态 hazard 门（check_streams 逐存储格写者追踪）三档 ERR=0
  WARN=0；Verilator 数值门 a2 10 段+a1 5 段全 PASS（覆盖 patch/swin/ffn/SM+COPY/
  V-T/rotary/32MB 大段/重 COPY），STORE 全窗口逐字节 vs 黄金一致；
  golden==fast_interp==RTL 三方逐位一致（3 段）；拍数与数据无关已证（空 DDR 同拍数）。
- 下一步（≤10 分钟）：最终报告=分杠杆账+残余流量下限（LOAD_CTX 首装 784MB+
  段内重装 53MB、LOAD_W 重装 148MB、STORE 751MB 全唯一）+ RTL 需求清单带量化
  收益（异步 STORE 队列 ~276M、CTX 预取 ~226M、GEMM 行组流水 ~186M 占固定
  开销 58%、COPY 40M）+复现命令。服务器集群已死透，无遗留进程，数据保留。

## 2026-08-31 17:4x compute-bound 最终报告收官（−18.9%，模型 99.9%，位精确 15/15）

- 杠杆归因：A（段界/列块合并，repro→a1）−82.5M（−6.8%）；B/C/D（多头 A 单装/
  V-T 单装/WRAM 驻留表/LOAD_W 贴发射，a1→a2）−147.6M（−13.0%）。
  字节口径 LOAD_CTX 1543.3→837.6MB（−45.7%），其中段内重装 527.0→53.5MB（−89.8%）。
- 模型外样验证：a2 已测 676 段总量偏差 −0.11%、a1 −0.37%——对重构后的指令流同样准，
  986.4M 可信度 ±1%（≥95% 门实际 99.9%）。
- 口径对账定案：旧账 1080.8M 与本轮 1216.5M 差三源（2782 vs 3118 段流不同、
  GEMM 用全局 n 高估 66%、DMA 项没算 LFSR+burst 从机开销）。
  新口径排序：LOAD_CTX 34.3% > GEMM 28.5% > STORE 22.7% > LOAD_W 10.9% > COPY 3.3%
  （a2 档：GEMM 34.1% > STORE 28.0% > LOAD_CTX 22.9% > LOAD_W 10.5% > COPY 4.1%）。
  旧 HTML 页（问题讨论页/三份总结）的归因图数字作废，以新页为准（不改旧文件）。
- 残余流量下限：LOAD_CTX 837.6=首装 784.1（段自包含契约不可免）+双装 53.5
  （可再消 ~15M 拍）；LOAD_W 587.1=首装 438.6+重装 148.5；STORE 751.5 零冗余
  （覆盖重写 0 字节，调度层已到头）；GEMM 理想拍 319.7M 里 58.1%（185.8M）是
  行组固定开销（feed/drain/写回，69.2 万行组）。
- RTL 需求清单（性价比序）：R1 异步 STORE 队列 276M（~3-5k LUT+6-10 BRAM36，低风险）；
  R2 CTX A 预取 226M（中风险）；R3 行组间流水可回收 ~130M（面积近零，时序风险）；
  R4 WRAM 2→4 组 10-20M；R5 COPY 消除 40M；R6 真机 DDR 口径（HP 64B/cyc 下 DMA
  服务 606M→150-250M，GEMM 才成主线）。R1+R2 逐项上限 502.7M，受单 DMA 服务时间
  约束组合可实现 ~340M（986.4→~646M=3.25ms）。
- 证伪清单：STORE 合并/pad 修剪证伪（零冗余）；跨段 CTX 驻留未做（要改段自包含
  契约，非纯调度）；"GEMM 70%"口径不存在；段内 CTX 图缓存（~15M）已识别未实施。
- 事故两起已修：TB 默认全量 dump 64MB（192MB 文本/段）写满 /tmp（删 93GB+
  +DUMP=/dev/null 防护）；双队列重复跑同段（杀重，tsv 去重无影响）。
- 服务器无遗留进程，/tmp/ae_cb 1.3GB 数据保留；本地 gate_a1 5 段、gate_a2 10 段、
  三套 cycles tsv 齐全。复现命令在最终报告第十节。

## 2026-08-31 17:25 "解决方案与效果"页落盘（10_cbound_report/）

- 2026-08-31_1725_compute-bound改造方案与效果.html（5 图 23 tip，自检 PASS，
  新文件夹 10_cbound_report/）。内容：三档方案与降幅表（−6.8%/−18.9%）、
  分量对账三图（节省/基线分量/a2 分量）、口径对账与旧页作废清单、
  模型 99.9% 达标过程、15 段位精确门、残余流量下限、RTL 需求清单 R1–R6
  （含组合上限 340M→646M/3.25ms 与建议拍板顺序 R1→R2→R3）、证伪清单、复现命令。
- 全部代理收官：架构（compute-bound）、硬件（AE_ACTV+INT4）、算法（端到端）
  三线均有定案；当前活跃线仅架构。待拍板：R1/R2 重启硬件线与否；D6 挂起。

## 2026-09-01 00:49 — 全项目优化总结页（GEMM 主线达成）

### 本轮（架构线推进到 GEMM 主要计算时间）
- **片上算子引擎全模式落地（12_actv/）**：AE_ACTV 引擎四模式——ACTV（直查表）、BIAS、NORM（含 AdaRMS，两遍扫描+rsqrt）、ELTWISE（双输入残差加）。微观 221,184/221,184 字节位精确；NORM 真实站点 148/148、ELTWISE 148/148 ≤1 LSB；全芯片回归 12/12；全芯片综合 u_actv 24,852 LUT / 0 DSP / WNS −1.038。
- **a3 融合（12_actv/a3/）**：192 站 actv 融合，986.7→899.4M（−8.8%），GEMM 37.4%。数值 A/B 192/192 逐字节一致。但发现理想夹心只有 192 处，norm 全长在注意力旁。
- **a4 驻留+残差（12_actv/a4/）**：实测未降（901M），resid 98% 结构性跳过（c_in_pair_fusion/tensor_link）。编译器路线到头。
- **R1+R2 收益重算（13_rtl_plan/）**：基于 a3 真实流，计算通道 387.2M > 读 292.1M > 写 218.4M。C2 R1+R2 在 TB 口径下 427.2M / GEMM 78.9%。关键发现：TB 从机物理上是 AXI 全双工（不是共口），写被计算完全藏住。R1 单独不够（46.8%），必须 R2 把 LOAD_CTX 也藏进 GEMM。
- **R1+R2 RTL 改造（14_rtl_r1r2/）**：ae_dma 双引擎（读/写并发）+ ae_ctx_ram 分区预取 + pf FSM 扩 TAG_C + NORM 时序修复。全芯片回归 12/12；综合 LUT 134,762（+28）/ DSP 1,728 满片 / WNS −1.038 达标。Verilator 代表段门服务器跑中。
- **总结页（15_summary/）**：2026-09-01_0049_全项目优化总结.html，覆盖口径翻案→compute-bound 三档→a3 融合→引擎扩展→R1+R2 全部优化点，每项带量化百分比+时间戳。累计 1216.5M→427.2M（−64.9%），GEMM 28.5%→78.9%。

### 状态
- 架构线 GEMM 主线达成（模型预测 78.9%，RTL 落地过综合，Verilator 门跑中）。
- INT4 线停（用户令），算法线停，硬件上板后置。
- 下一轮可选：R3 行组流水（427→~297M，GEMM ~88%）。

## 2026-09-01 01:3x — Verilator 代表段门结果（R1 过，R2 有 bug 修复中）

- **环境**：服务器 /tmp/ae_vgate/，Verilator 5.050 + g++-10，DDR=512KB 档、COLS=108，RTL 用 14_rtl_r1r2 版（双引擎 ae_dma 12564B）。
- **TB 两个缺陷已修**：①$readmemh 与 +DDRIMG 竞争（ddr_init.mem 拷 cwd 兜底）；②R1 改 fire-and-forget 后 TB 在 wait(done) 后需 repeat(200000) 排空写引擎，否则末条 STORE 丢失（部署侧接口语义，已记交接）。
- **R1（独立写通道）验证通过**：3 代表段（seg_0600/0529/0602）PF=0/PF=1 全位精确（256/256、32/32、4096/4096 字节，全 DDR 0 diff）。STORE 与 GEMM 真并发在计数器可见：seg_0600 重叠 2026 拍（45.5%）。
- **R2 周期收益实测**：seg_0600 省 548 拍（−18.4%）、seg_0602 省 2256 拍（−4.2%）、seg_0529 省 0（半区撞，符合契约）。
- **R2 数据正确性 bug（修复中）**：seg_0602（k=257 长窗口）PF=1 时 83 字节错（STORE 区、lane 0 row 0、±1 LSB）；PF=0 同段逐位一致 → 非数值问题，是长 GEMM 写回窗口下预取写与 GEMM 抢 CTX B 口的让拍边界 bug。短 k 段不触发。修复设计（代理已定位）：ae_dma 暴露 rd_tag_o + pf_ctx_stall 握手，CTX 预取 R_DATA 拍 B 口被占时 rready=0 暂停读引擎而非丢写。
- 周期模型对拍：seg_0529/0602 偏差 −2.7%/−2.1%（±15% 门内）；seg_0600 −38%（R2 预取额外省拍，低于模型——模型没算并发红利）。

## 2026-09-01 10:37 — 架构框图交付（16_arch_diagram/）

- 产物：2026-09-01_1012_架构框图.html，内联 SVG（viewBox 1480×1330），黑白灰配色，无交互无花哨。
- 覆盖：ae_top 寄存器映射（0x00 CTRL / 0x04 STATUS / 0x08-0x1C 计数器 / 0x20-0x60 SEQ 接口）、
  离线编译器与 256b 描述符字段（op/b_src/m/n/k/三基址/rq/spad/j0）、u_sched（SEQ RAM 2048×256b、
  主 FSM T_FETCH→T_LATCH→T_EXEC→T_RUN_*→T_ADV→T_FIN、pf FSM）、四引擎行（u_gemm/u_sm/u_actv/u_cp）、
  CTX A 口 one-hot 仲裁（GEMM▸SM▸ACTV▸COPY▸STORE）、u_ctx 16-bank 2MB、WRAM 442KB、
  双 B 口写仲裁（GEMM-Y▸SM▸ACTV▸前台 LOAD▸后台 CTX 预取）、DMA rd/wr 双引擎 FSM、DDR4。
  附图例（实线数据流/虚线控制/粗灰总线）+ 模块索引表（图↔RTL 对应）+ 典型段执行序走查。
- 校验：check_html.js PASS；SVG 19 主 rect/101 text 坐标逐个核对；Edge --dump-dom 确认加载；
  Edge --screenshot 渲染新标签页失败（Bing 壁纸），换 Chrome --headless=new 截图成功（1560×1450 PNG），
  视觉确认顶行三盒/四引擎行/DMA/DDR4 全部在位。

## 2026-09-01 10:50 — R2 预取 bug 修复收官（全门通过）

- **根因翻案（iverilog 探针实证）**：原怀疑的 gemm_wb_active 让拍是死门；真因是 CTX B 口
  纯优先级仲裁静默丢后台预取写 + ae_dma 读引擎 rready 恒真不回压，rd_done 照发 → 调度器误判预取完成。
  探针：修复前 pf_ctx_drop_cnt=7，修复后=0。
- **修法（两文件）**：ae_dma 加 pf_ctx_stall/rd_tag_o，R_R 全动作挂 rvalid&&rready，stall 即冻结；
  ae_core 算 pf_ctx_stall = bg_wran && tag==CTX && B口被引擎占。写一个不丢，AXI 自然背压。
- **全门通过**：iverilog 微观 21 用例 221,184/221,184 位精确；全芯片回归 12/12；
  Verilator seg_0602 PF=1 全 DDR diff 0/524,288（修前 83 字节错）；seg_0600/0529 不回退；
  PF=0 三段复验位精确。综合 +349 LUT（+0.26%）、WNS −1.038 持平、DSP/URAM 不变。
- **收益确认**：seg_0602 PF=1 净省 2,256 拍（−4.2%）。R1+R2（427.2M/78.9%）硬件实证闭环。
- 报告页：17_r2fix_report/2026-09-01_1050_R2预取bug修复报告.html。
  RTL 已同步服务器 /tmp/ae_vgate/rtl/ 与综合区 /e/ae_syn/r1r2_fullchip/rtl/。

## 2026-09-01 11:17 — R3 行组流水方案页（未实施）

- 产物：18_r3_plan/2026-09-01_1117_R3行组流水方案.html。
- 模型账（a3 真实流 69.2 万行组逐组重算）：G 336.9M = 稳态乘加 133.9M + 行组固定开销 185.8M
  （每组排空等待 127 + 量化读出 64 + 写回均值 70 + 零碎 8）；R3 周期 = max(k+2, 68+wb)。
- 方案 C（推荐）：PE 加 27b 快照寄存器（末脉冲随 A 链传播、到拍快照+清零），DSP 累加不动；
  +4.9 万 FF、LUT<+1k、DSP/BRAM 零增。G → 185.5M（−45.0%），阵列利用率 39.7%→73%。
  方案 A（保底，不动 PE）：G → 240M（−28.7%）。
- 总账修正（重要）：TB 口径 427.2 → 332.1M（−22.3%，读通道 292.1M 变新瓶颈）；HP64 口径
  275.8M（−35.4%）。旧总结页"297M/88%"系口径混淆：297M 是 HP64 保守值，88% 仅在
  全藏理想口径成立（140.8/(140.8+19)），真实占比 TB 55.9% / HP64 67.3%。
- 排期依赖：R3 是读侧改造（第二读引擎等）的前置条件；不做 R3 读侧零收益。

## 2026-09-01 11:38 — R3C 架构定案 + LOAD 压缩路线 + R1/R2 原理页（三交付，未动 RTL）

- **19_r3c_arch/（R3 方案 C 定案 + 模型落地）**：r3c_model.py 对 a3 流 69.2 万行组逐条重算，
  行组周期改 max(k+2, 68+wb)。GEMM 336.4→185.3M（−44.9%），TB 总拍 426.7→332.1M（−22.2%），
  HP64 275.6M（−35.4%）。数据通路（PE 27b 快照 + 末脉冲 A 链传播 + tile_buf 双缓冲 + FSM 两段化）
  与验收门定死，RTL/综合按用户指示未启动。结果落 r3c_model.json。
- **20_read_compress/（LOAD_CTX/LOAD_W 压缩路线）**：跑通 a4 普查脚本拿真实字节构成——
  LOAD_CTX 680MB 里 81%（548.6MB）是引擎自产图 DDR 往返；容量内驻留候选 191.6MB。
  三条路线：①CTX 双区驻留+host 折链（a4 教训=必须折链联动，单驻留只剩 20 站）；
  ②双读引擎（292.1→184.0M，单刀过 235.6M 线，TB 混合速率待测）；③WRAM 4 组（239.0M 不够线）。
  过线后三口径归一 275.6M/1.39ms（自项目起点 −77.3%）。顺序依赖：先 R3 再读压缩。
- **21_r1r2_explain/（R1/R2 原理说人话页）**：R1=DMA 单 FSM 拆读/写双引擎走 AXI 全双工
  （GEMM 让 A 口规则 + fire-and-forget）；R2=pf FSM 扩 TAG_CTX + 半区预取 + GEMM 喂数期 B 口
  空档写入，含 pf_ctx_stall bug 修复始末。899.4→426.7M（−52.6%）、GEMM 78.9%。

## 2026-09-01 14:27:08　PE 阵列规模数据驱动选型定案 + HTML 风格全局 skill

**PE 选型（COLS 108→96，1536 PE，−192 DSP）**：新脚本 `19_r3c_arch/pe_sizing.py`（输出 `pe_sizing.json`）。
- 双口径建模：固定流重切（按 108 编译的流硬切，偏悲观）+ 重编译（逻辑 GEMM 重建后按新 COLS 均衡分组）。
- 逻辑 GEMM 重建规则：同段内连续 GEMM 描述符，(m,k,a_base,y_base,y_tr) 相同且 j0 链式递进、中间只夹 op=3/4 → 合并（im2col 自带独立 a_base 不合并）。49569 个逻辑 GEMM（83767 条描述符）。
- 关键结果（重编译口径，校准×1.0514）：108→176.1M；104→181.8M(+3.2%)；100→183.8M(+4.4%)；**96→185.2M(+5.2%)**；92→192.4M(+9.2%)。
- **定案 COLS=96**：与现状 108 固定流（185.3M）几乎持平（+0.05%），HP64 1.39ms 不变，TB 1.67ms 不变（读通道 292M 仍为瓶颈）；利用率 49.6%→53.0%（反而提高）；−192 DSP（88.9% 占用，留 192 个余量）；96 是 16 的倍数（j0 组边界永在 lane 组边界、转置写回不跨组）、96/8=12 拍整（B 装载无残拍）、NGRP=24。
- 阵列口径 MAC 总量 150.8 G/帧（83767 条 GEMM 描述符实测汇总）。

**HTML 风格全局 skill**：`~/.claude/skills/html-report/SKILL.md` 落盘（CSS 模板+时间戳/新文件夹/复现命令/资源占比图表规矩），后续 HTML 报告直接套用。

## 2026-09-01 14:29:11　DSP48E2 int8 乘法原理页

`23_dsp_int8/2026-09-01_1429_一个DSP怎么算int8乘法.html`：说人话拆解一颗 DSP48E2 算 int8×int8——补码/符号扩展（−5×100 手算）、27×18 乘法器只用 8×8 角落、P 寄存器驻留与 27 位无损覆盖（|acc|≤3306 万<2^26）、use_dsp="yes" 属性的坑（不加→1728 PE 全进 LUT=125% 超载）、一拍只能一对 int8（打包乘法交叉项污染，int4 才能两对）、R3C 加快照的根源、1536/88.9% 资源账。

## 2026-09-01 14:32:40　PE 阵列选型决策页

`22_r3c_rtl/2026-09-01_1432_PE阵列选型1728减到1536.html`：双口径模型对比（固定流硬切 +40% 是错误口径；重编译口径 96 列只 +5.2%，与现状 185.3M 持平）、逻辑宽度分布表（108 满组仅占行组 6%）、96 vs 100 结构判据表（lane 对齐/整拍对齐/NGRP/余量），定案 16×96=1536 PE。

## 2026-09-01 20:00:41　R3C 方案 C 落 RTL：行组流水 + 16×96 阵列，GEMM 引擎拍数 −24%

**改动范围（22_r3c_rtl/，从 14_rtl_r1r2 复制起步，原目录零改动）**：只动 ae_pe/ae_sysarr/ae_gemm 三个文件 + 一处前置死锁修复（ae_core 一行）。
- **ae_pe.sv**：加 27 位快照寄存器 snap_r；末脉冲随 A 链传播，脉冲到拍 `snap_r<=acc_r; acc_r<=0`——清零不再依赖整排排空，乘加仍驻留 DSP（use_dsp="yes" 保留在 acc_r 上）。
- **ae_sysarr.sv**：acc_row 组合读改从快照侧出（符号扩展回 32 位），clr 退化为复位兜底；旧 busy/done 波前排空检测删除。
- **ae_gemm.sv**：FSM 拆喂数道（st_f）与读出道（st_r）两段并行；行组周期 = max(k+2, DRAIN+DALIGN+2+wb)；requant NGRP=COLS/4 推导（96→24 套 rq_ms）。
- **COLS 默认 108→96**（ae_pkg/ae_top/ae_core/ae_gemm/ae_dma/ae_copy），16×96=1536 PE；grep 确认 RTL 无残留硬编码 108。
- **ae_core.sv 一行修复（前置 R2 缺陷，非 R3C 引入）**：pf_ctx_stall 原式含 (eng_dma && !dma_iswr) 项，主 FSM 走 LOAD 命中预取路径时会冻结自己在等的后台预取 → 死锁。actv+pf1 负载实测 R2 基线（14_rtl_r1r2 副本）同样挂死，删项后 pf1 正常完成且 dump 与 pf0 逐位一致。真前台 LOAD 与 bg_wran 互斥（T_EXEC 发 dma_start 要求 !dma_busy），删项无副作用。

**回归（12 项 ALL PASS，硬门位精确全过）**：gen_vectors / sim_ae / compare / tb_rq / tb_sm16 / tb_pe_pack / tb_pe_pack_dsp / gem_cycles / gen_actv / sim_ae_actv / compare_actv / tb_ae_actv。
- GEMM 引擎拍数：REF 5975→4526（**−24.2%**）、PRIM 5723→4310（**−24.7%**）（14_rtl_r1r2/sim/reg_logs 同日基线 vs 22_r3c_rtl）。
- 总周期：REF 8255→6814（−17.5%）、PRIM 7999→6618（−17.3%）。
- gem_cycles.py 重标定：R3C 行组流水公式 max(k+2, DRAIN+DALIGN+2+wb)×行组数（GEMM 偏差 −0.53%/−0.65%）；DMA 模型按写引擎实测节奏重标（每 16B 行 6 拍、命令尾 4 拍、行为级从机开销 3 拍，末条 STORE 按 RTL 口径只计 4 拍），偏差 REF +0.56%/PRIM −2.46%；PF0/PF1 档实测填入（pf_bg_start 探针 11 次，与模型 n_pf=11 精确一致，省拍量偏差 −3.88%）。

**Vivado 2021.2 综合（e:/ae_syn/r3c_c96/，OOC + RuntimeOptimized + flatten_hierarchy none，4.000 ns 时钟）**：
| 项 | 基线 actv_v122（16×108） | R3C r3c_c96（16×96） | Δ |
|---|---|---|---|
| LUT | 134734（58.5%） | 120619（52.4%） | **−10.5%** |
| FF | 118336（25.7%） | 150221（32.6%） | +26.9%（1536×27b 快照+延迟线） |
| BRAM | 126.5 | 114.5 | −9.5% |
| DSP48E2 | 1728（100%） | **1536（88.9%，验收恰好命中）** | −192 |
| WNS | −1.359 ns | −1.363 ns | 持平（同样欠 250 MHz 约 1.36 ns，OOC 口径） |

u_gemm 模块 LUT 76176→65452（−14.1%）、FF 105657→137097（+29.8%）。只综合，未跑 place/route/bitstream。

## 2026-09-01 20:02:46　R3C RTL 轮总结页

`22_r3c_rtl/2026-09-01_2002_R3C行组流水RTL验证与综合.html`：三文件改动表、12 项回归表、周期账（基准 −24%/全帧模型 −45% 口径说明）、综合资源表（DSP 恰好 1536、LUT −10.5%、FF +26.9% 快照成本、WNS 持平）、每模块资源×时间占比表、R2 旧死锁修复始末、坑清单 5 条（含 iverilog→Verilator 裁决）。

## 2026-09-01 20:05:16　R3C 交接包

`handoff_r3c/`（仓库根）：README（一天上手+目录地图+环境+硬规矩）、STATUS（当前权威形态 1536 DSP/12 项全绿/五轮演进/下一步优先级）、ARCHITECTURE（模块地图+R1/R2/R3C 三代优化+口径体系）、FLOW（验证矩阵+Verilator 快路径+综合流程+改 COLS 清单）、PITFALLS（15 条，含本轮 5 条新坑）。取代旧 handoff/（08-30 版，保留作历史）。

## 2026-09-02　R3C 交接包同步 GitHub

仓库 github.com/nc-thu/vector-core-r3c（private），main=23c20e4。按用户裁决只推核心：仿真器/算法/RTL 代码+核心结果（3653 文件，最大单文件 4.6MB）；数据块全部不入库（权重 blob 188MB、segments 的 ctx/w/ddr mem、t7_*.npy、04_dataset npz、fixture.pt、第三方 robo_orchard_lab），segments/*/seq.mem 保留供 r3c_model/pe_sizing 复算。白名单式 .gitignore 落在仓库根。坑两条：①aux.json 是 Windows 保留设备名，git 打不开，gitignore 掉（measure.py 可重新生成）；②校园网到 github.com:443 时通时断，推送需趁连接窗口抢推重试。
