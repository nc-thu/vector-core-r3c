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

## 2026-09-02　仓库转公开

github.com/nc-thu/vector-core-r3c 转 PUBLIC（用户裁决）。转公开前全库清理内网信息：内网 IP、登录串、服务器家目录路径全部占位符化（29 个文件）；Vivado 报告头裸主机名保留；token/密钥扫描零命中。

## 2026-09-02 13:40:54　双代理并行启动：W8A8 SOTA PE 探索 + W8A8 部署错误根因调查

- **代理 A**：pe_w8a8_sota/（独立目录，PLAN.md=用户 42 节计划原文 + TASK.md=环境/纪律）。目标：DSP48E2 上 bit-space×clock-pumping 联合利用（1 DSP 服务 2×2 logical MAC contexts）+ hierarchical exact accumulation（INT24 chunk→INT32 后台 flush，HA1/HA2 两版），baseline 矩阵 B0~B8（XtraMAC/FPL-ring/UDP），Phase 1~4 必做、OOC ≤40 次、kill criteria 诚实触发。
- **代理 B**：research_w8a8_error/（BRIEF.md）。目标：逐层误差分解（0.29 rad 超判据 6.6× 的来源）+ 量化粒度主因判定 + 解法对比（动态激活 scale 可能硬件零改动——requant 描述符已带 rq_m/rq_s；W8A16；敏感层保精度；舍入改就近）。
- 两项各出一 HTML（代理完成后主会话写）。

## 2026-09-02　双代理重启（会话中断接续）

上会话 13:40 启动的两个代理随会话终止而中断（A 只剩目录骨架+XtraMAC clone，B 零开工）。本会话重新派发，任务书不变（pe_w8a8_sota/PLAN.md+TASK.md、research_w8a8_error/BRIEF.md）。

### 2026-09-02 15:27:55 面试速成教程
- 新建 tutorial/，产出《架构模拟器速成：从指令解码到周期模型》图文 HTML。
- 内容全部取自仓库真实代码：golden_interp.decode（指令解码）、fast_interp_a3（功能模拟）、acct_a3（周期公式+校准表）、cycle_exact_a3（事件模拟+预取并发）、r3c_model/pe_sizing（两个架构决策案例）。
- 附 60 秒电梯陈述、高频追问预演表、event-driven simulator 手撕骨架（20 行）。

## 2026-09-02 16:46:30　W8A8 SOTA PE：Phase 0 全绿（golden + TB + 7 variant 位精确对拍）

- **golden**：model/golden.py 生成 full suite（10251 dots / 1,032,501 txs；K∈{1..4096} 23 档 × 5 分布 + K=1 全枚举 2401 块）+ smoke（55 dots）。文件格式 B/T/X/E 四行型，TB 统一消费。
- **TB 三件套**（tb/）：tb_pe.sv（B1/B4/B5/B6/B7 通用，negedge 命令队列驱动、每 slot 恰 2 posedge）、tb_b0.sv（4-pass 重放）、tb_quad_raw.sv（raw 核逐 beat 对拍 FIFO）+ glbl.v stub。
- **RTL 七件全部 iverilog 位精确全绿**（unisim DSP48E2）：B0 scalar、B1 XtraMAC 移植、B4 naive、B5 HA1、B6 HA2、B7 窗口 ring、quad_raw（raw Pack2×Pump2 乘法核，2,065,002 beats 零误差）——**Phase 2 前提（乘法核 bit-exact）已成立**。
- 修复记录：B7 三处（cap 双拍/满判据差一/乒乓 rdy 同拍清零竞态 + 处理-填充节奏失配改为 st3 背靠背换 bank）；B4 符号扩展 30b 截断；`local`/`void` 关键字；TB slot 驱动多采样 1 posedge；对拍 FIFO 满判据。
- 证据：runs/<variant>/sim_full/{command.txt,sim.log,result.json}。
- 下一步：Vivado OOC（工作区 E:\ae_syn\pe_w8a8_sota\，tcl/syn.tcl + synth/run_synth.sh + sweep.py 已备），按 PLAN §38 顺序 Phase 1（B0/B1/B7）→ Phase 2（quad_raw@fast）→ Phase 3/4。

---

## 2026-09-02 16:57:56 W8A8 部署错误根因定案：逐 tensor 权重量化是主因；逐通道权重+就近舍入实测 -75%

**背景**：全深度 W8A8 链在真实 RoboTwin 样本上 jpos 0.2993 rad（判据 0.045，超 6.6x）。本轮在真实 s000 上做同硬件语义的逐项切换 A/B（模块边界深度，与 modeB 门禁同口径），把误差来源拆开。全部材料在 research_w8a8_error/（REPORT.md + analysis_00~05 + results/）。

**根因（实测证据链）**：
- 只把权重改逐通道（激活/输出/requant 全不动）：0.189 → 0.059（**-69%**）。
- 激活侧一切改动（逐帧/逐 token/dyadic 动态 scale、SmoothQuant 折入）端到端差 ≤0.007，**全部无效**——激活噪声零均值、在 K 项 MAC 里平均掉；权重误差是相干扰动（每个 token 同向），430 层 + 10 步 DPM 同向累积。
- 输出 INT8 轨道/截幅：无效（V0b fp 输出 0.196 ≈ V0 0.189）。
- requant 截断的 -0.5 LSB 偏置：权重修好后可见，改就近舍入再拿 21%（0.0595 → 0.0472）。
- 组合（逐通道权重 + RTN）= **0.047**，贴近 W8A16 天花板路径（0.0049）；W8A16 判定为过度设计。
- 最差层三方对齐（bringup 实测 rel / 部署权重解析误差 / 真实样本 sentinel）：Swin stages.2/3 ffn.layers.1（逐通道幅值差 15.9x、e_pt 0.111）、downsample.reduction、BERT output.dense、fusion ffn/out_l_proj。
- 首层 patch_embed rel 0.70 之谜解开：全部来自输出 requant 网格太粗（合成批 absmax 标定的 so，真实输出重尾），输入侧 sa 超量程 1.42x/2.17x 只占 0.164；**端到端无害**（V2_rn 到 0.047 时首层仍 0.705，LayerNorm 吸收）。
- 门禁教训：合成 bringup 批逐层 rel 也有 0.8+ 却端到端绿（0.029）——该门禁对真实部署无预测力，评估集必须换真实样本。

**解法与硬件改动量**（推荐 = A）：
- A. 逐通道权重 + RTN：rq_ms 的 m/s 单输入改 per-slot 总线（x 已是 mux，24 核共 2.3 kbit 系数 RAM）+ rq_v2 加 35b 舍入常数加法器（rn=2^(s-9) 随描述符锁存）+ ae_gemm SF_INIT 载 96 组系数；累加器界不变，不碰脉动阵列。编译器：逐通道 sw 标定 + 逐列 (m,s,rn) 描述符 + 逐通道 bias aug。
- 下一步第 1 任务（软件先行）：pcW+RTN 移植进全深度仿真链，s000/s001 对拍 fp32_ref，判据 jpos ≤0.045（边界深度预测区间 0.07~0.16，需实测）。

## 2026-09-02 17:02:21　W8A8 错误根因调查汇报页

`research_w8a8_error/2026-09-02_1702_W8A8部署错误根因.html`：A/B 切换对照图（只动权重 −69%/−75%，激活侧全部无效）、相干 vs 零均值误差机理、最差层三方对齐表、推荐方案硬件改动量表（rq_ms 每列系数 + rq_v2 舍入加法器，不碰脉动阵列）、否决路线各一条理由、下一步四步路线。诚实边界：0.047 是模块边界深度，全深度预测 0.07~0.16 待下轮实测。

## 2026-09-02 17:07:40 — W8A8 PE 探索 Phase 1 完成：B0/B1/B7 OOC 全绿
- B0 scalar baseline：33 LUT / 164 FF / 1 DSP，Fmax ≈ 401.8 MHz（三档 period WNS 一致收敛）
- B1 pack2 baseline：191 LUT / 236 FF / 1 DSP，Fmax ≈ 471.0 MHz
- B7 ring（pack2+pump2+窗口环+层次累加）：217 LUT / 332 FF / 1 DSP，Fmax ≈ 421.4 MHz @ 2× 快钟
- 全部 iverilog full-suite 位精确全绿（10251 dots，B0 为 4-pass 重放对拍）
- 备注：Vivado 偶发内部错误 "rt-undefined"（synth_design 崩溃），重试即恢复，非 RTL 问题
- 下一步：Phase 2 quad_raw OOC 时序验证（失败即停）

## 2026-09-02 17:10:10 — W8A8 PE 探索 Phase 2 完成：quad_raw gate 通过
- raw Pack2×Pump2 乘法核（1 DSP 服务 2×2 逻辑 MAC）：30 LUT / 76 FF / 1 DSP
- 三档 period 全部大裕量：2.5ns WNS +1.256 / 2.222ns +0.978 / 2.0ns +0.756 → Fmax ≈ 803.9 MHz
- 结论：DSP48E2 位空间并行与时钟倍频可同时利用，raw 核时序完全不是瓶颈
- 瓶颈在外围：B7 全核 421.4 MHz vs raw 核 803.9 MHz，差距来自抽取校正+层次累加逻辑
- 下一步：Phase 3 B4 naive OOC sweep

## 2026-09-02 17:15:00 — W8A8 PE 探索 Phase 3 完成：B4 naive OOC
- B4（pack2×pump2 + 每拍朴素抽取 + 4×INT32 独立累加）：127 LUT / 172 FF / 1 DSP，Fmax ≈ 495.3 MHz
- 对比 B7 ring（217 LUT / 332 FF / 421.4 MHz）：B4 面积近半、频率更高——窗口环的抽取折叠 FSM 反而拖累时序
- 注：B4 三档中 2.5ns 两次撞上 Vivado 偶发文件读取故障（unimacro_verilog.tcl 读失败，与 rt-undefined 同源），第三次重试成功；非 RTL 问题
- 下一步：Phase 4 B5（HA1）/ B6（HA2）OOC sweep

## 2026-09-02 17:25:30 — W8A8 PE 探索 Phase 4 完成：B5/B6 HA 结构 OOC
- B5（HA1：DSP-P 窗口环 + INT32 全局）：406 LUT / 520 FF / 1 DSP，Fmax ≈ 501.0 MHz
- B6（HA2：两级累加 INT24 chunk→INT32 global）：386 LUT / 553 FF / 1 DSP，Fmax ≈ 672.95 MHz
  - 加测 1.818ns（WNS +0.332）与 1.667ns（WNS +0.181）：600 MHz 目标仍通过，Fmax 收敛一致
  - 折算吞吐：672.95 MHz × 2 MAC/拍 = 1346 GMAC/s / DSP，为 B0 scalar（401.8 × 1）的 3.35 倍
- 结论：两级累加（HA2）胜出——chunk 累加保持窄位宽进 DSP 反馈路径，全局累加仅慢速域承担
- Vivado rt 瞬态故障复现两次（rtSynthParallelPrep.tcl 读失败、.Xil realtime tcl 读失败），清 .Xil 后恢复
- 下一步：汇总 summary.csv；Phase 5 cluster 视时间决定

## 2026-09-02 18:25:00 — W8A8 PE 探索 Phase 5 完成（缩水版）+ 全任务收官
- B6 单 PE post-route（synth+opt+place+route @1.667ns）：WNS −0.065 → Fmax 577.4 MHz，较 OOC 672.9 退化 14.1%
- 16×B6 广播 cluster post-route：6692 LUT / 16 DSP，WNS −0.412 → 481.0 MHz（单核 83.3%）
- 64×B6 广播 cluster post-route：26707 LUT / 64 DSP，WNS −0.386 → 487.1 MHz（单核 84.4%）
  - 16→64 吞吐/DSP 基本持平（0.96 vs 0.97 GMAC/s/DSP）：cluster 扩展性良好，退化集中在单核→多核一跳
  - 违例路径在 PE 内部（route 占 64%），非广播扇出：floorplan（PBLOCK）有希望挽回，未做（缺口）
- B0 post-route @2.5ns：WNS +0.025 → 404.0 MHz（近零退化）；B6/B0 post-route 口径吞吐比 2.86×
- cluster 等价性冒烟：pe_b6_cluster(N=2) vs 单体 pe_b6，63 块 PASS（runs/cl16/sim_smoke/）
- 环境备注：Vivado 2021.2 本机随机文件读失败（rt-undefined 家族）已用自动重试+清 .Xil 根治；综合 tcl 加 set_param general.maxThreads 1
- Phase 6 power 按指令未做；CHUNK sweep、floorplan 对比、B1 post-route 未做（缺口记入 REPORT）
- 交付：results/summary.csv（27 行，OOC 23 + post-route 4，全脚本生成）、REPORT.md 内容（因子代理写文件限制，由主会话落盘）
- 探索汇报 HTML：pe_w8a8_sota/2026-09-02_1823_W8A8SOTA_PE探索结果.html（2026-09-02 18:23:53 落盘）。核心结论：4 积/DSP 位精确可行但布线后吞吐 2.86×（判据 3×，边缘未达）；损失分解——朴素累加 −38%、布线 −14%、集群密度 −16%、乘法核与 packing 零损失；意外发现 B4 为面积黑马（127 LUT 面积效率第一）、HA2 真实收益是频率（+34%）非面积；最大缺口为 floorplan（PBLOCK）实验未做

## 2026-09-04 10:30:30 — 24_pcw_rtn 阶段1：语义移植与编译器扩展（代码全部落地）
- BRIEF_SW 执行：pcW+RTN（逐输出通道 INT8 权重 + requant 就近舍入）从模块边界 A/B 移植进全深度链（compiler → fast/golden_interp → host_driver），语义权威 = research_w8a8_error/analysis_02_server_ab.py 的 V2c_pcW / V2_rn_pcW，逐位对齐
- sw/ 目录：03_compiler 26 个 py 全量拷贝，只改副本。新增描述符 op=14（OP_SF 系数装载：每 256b 字 10 个 24b 槽 m<<8|s，bits[251:240]=槽号）；GEMM 标志位 rq_s bit7=逐列系数、rq_m bit15=RTN；两解释器（golden 纯 Python / fast numpy 向量化）在真实 build 段逐位一致
- RTN 整数语义定稿：y = sat8((acc·m + 2^(s-1)) >>> s)，s≥9；与 RTL rq_v2 内部口径 y = sat8((acc·mh + ((acc·ml)>>>8) + 2^(s-9)) >>> (s-8)) 逐位相等（整数恒等式，本地 2 万随机 + 负数边界用例零失配）
- 服务器 pcw_export.py 从 fp32 ckpt 重导逐通道 INT8 权重 427 个 tensor（逐 tensor 的 w8_export 信息已丢不可反推），通道 scale 最大比值 41.2×；mk_pcw_calib.py 合成 hw_calib_table_pcw.json：426 层 pcW + 12 conv 保持逐 tensor + 全部量化层 bias 改 host fp（aug 关闭，A/B 实测两者差 <0.001）
- 修复一个键名失配：pcw_scales.json 按 manifest 键（带 .weight）存、v2 校准表键不带后缀，首轮只匹配 6 层；双向查找后 427 层全覆盖（唯一未匹配层 spatial_enhancer.pts_prob_fc.layers.1 本就是豁免层）

## 2026-09-04 10:31:00 — 24_pcw_rtn 阶段2：编译 + fast_selftest 全绿
- build_s000_pcw / build_s001_pcw（服务器 /tmp/pcw_rtn/）：3118 段、362091 描述符（其中 OP_SF 101294 字）、最大 SEQ=1009（预算 2048）、权重 blob 196.46MB 与 v3 相同、预估 3162ms@198.5MHz（v3 为 3102 段；增量全部来自 OP_SF 字与逐列系数重发）
- fast_selftest（fast vs golden 逐位对拍，特征桶抽段）：9 桶全 PASS，其中 8 桶含 OP_SF + 逐列 GEMM（最大一段 470 个 SF 字 + 44 个逐列 GEMM），含 y_tr/softmax/COPY 混合路径
- 判据锚点：全深度部署基线 V0 jpos=0.2993/0.2108（s000/s001），边界 A/B pcW+RTN=0.04715，fp 重采样噪声 0.0457，判据 0.045

## 2026-09-04 10:39:34 — 24_pcw_rtn 阶段2（RTL 线）：三处 RTL 改动 + RTN 向量门全绿
- rq_v2.sv：加 rn_en 输入，RTN 就近舍入。rn 由 s 在模块内译码（one-hot 取 2^(t-1)，t=s-8≥1 才使能；t≥PW 时钳位 2^(PW-1)，两侧都移位到 0 无损），T0 与乘积同拍寄存 rn_r/t_r（不加在 T1 移位关键路径）。sum 容器 PW→PW+1（36b）：sext(sum)+zext(rn) ∈ (−2^34, 2^35) 不回绕。floor（rn_en=0）与 R3C 逐位一致
- rq_ms.sv：m/s 单输入 → m_bus[SHARE×16]/s_bus[SHARE×8] 按 slot 选择（照抄 x_bus 模式），rn_en 直通
- ae_gemm.sv：SF_INIT 从锁 1 组 (m,s) 改为锁 rq_coeff[COLS×24]（列 c 的 {m[16b],s[8b]} 在 [c·24+8]/[c·24]）+ rq_rn_en；24 套 rq_ms × 4 slot = 96 组
- ae_sched.sv/ae_core.sv：GEMM 族描述符（op∉{3,4,5,6,15}）扩成 1+NCW 字（NCW=ceil(COLS·24/256)，96 列=9 字），新增 T_CFETCH/T_CLATCH 泵（2 拍/字，枚举追加保持旧编码）；pc 推进 = pc+1+NCW（与预取 pc_next 同式，预取只看紧邻描述符无多步前瞻，天然安全）；rn_en 走头字 bit 28（原空闲位）
- tb_rq.sv 新增相位 D：rq_v2(rn_en=1) 60k 向量对 python 神谕 sat8((x·m+2^(s-1))>>>s) 位精确（s 覆盖 HB 标定域 [21,27]/全域 [9,47]/s=8 退化 7500 行/小 s 饱也撞舍入，饱和命中 36288 行）；相位 A/B（floor 锚，rq_v1 神谕）60k+30k 全绿不变
- 修 R3C 遗留向量坑两处：①gen_rq_vec.py 写 ctrl.mem 但 tb_rq 读 rq_ctrl.mem（靠手工改名凑合），生成侧改名对齐；②x 裁剪上界 2^26 → 2^26−1（+2^26 越出 27b 有符号域，R3C 相位 A/B 双侧同错抵消未暴露，对 python 神谕才炸，1555 失配全在 x=+2^26/128·128·4096）。两 TB 加向量空载守卫（缺 .mem 时 $fatal 而非空跑假 PASS），regression.sh 补 gen_rq 步骤、复制 sm16 孤本向量与 spec/norm_gold
- gen_vectors.py：golden 逐列 requant + RTN、描述符 +NCW 系数字（平铺 blob）、--case pcw（每 GEMM 独立 12 组随机 (m,s)、每层掷 RTN/floor、s 域含 8/9/11 边界列）；tb_ae.sv SEQ_N 64→128（默认用例 64 字恰好顶满）
- pcw 用例全链路三遍（REF/PRIM/PRIM-pf1）dump 与 golden 逐位一致，prim2==prim 逐字节；默认用例引擎计数 vs R3C 实测：gemm 4532 vs 4526（+6，slot 相位抖动同 R3C 的 pf1−pf0=+3 性质）、dma 712 vs 710、macs/skip 精确一致；总周期 6814→6930（+116 = 系数泵 2 拍×2 字×29 条 GEMM，纯调度器侧）
- 与软件线语义对齐：RTN 整数公式两侧已各自从相反方向验证逐位等价（SW 侧 2 万随机含负数边界零失配 + 本线 60k 向量门）；描述符编码两侧不同（RTL：头字后挂 NCW 字 + bit 28；SW：op=14 OP_SF 独立字 + rq_s bit7/rq_m bit15 标志）——两条 BRIEF 各自授权的格式，REPORT_RTL.md 写清偏离与收敛建议

## 2026-09-04 11:06:45 — 24_pcw_rtn 阶段3（RTL 线）：回归 17 项全绿 + 芯片级 floor 锚 + tb_ae_actv X 根因

- 回归 run-4（11:02 完，约 5 分钟）：ALL PASS 17 项（新增 gen_lut 步骤）。tb_rq 相位 D 60000 向量对 python 神谕 err=0（s∈[21,27]/全域/7500 行 s=8 退化/小 s，饱和命中 36288 行）；相位 A/B floor 锚 err=0。
- 芯片级 floor 锚（11:00）：本 rtl/ 重跑 actv 用例，CTX/DDR 四个终态 dump 与 R3C 存档 cmp 逐字节一致（ddr_init 输入亦一致）。seq.mem 40→70 字 = 15 GEMM × 2 系数字，行为零变化。
- pcW 全链路：逐列 (m,s) + 每 GEMM 掷 RTN，REF/PRIM/prim2 全部位精确；默认用例位精确 + 引擎 MAC 精确一致；总拍 REF 6814→6930（+1.7%，泵 108 + 相位抖动 +8）、PRIM 6618→6702（+84）。
- tb_ae_actv 全 X 根因（耗时最久的一坑）：ae_actv.sv 运行期 $readmemh("rsqrt_lut.mem")，NORM 子模式的 rsqrt 表是输入孤本（spec/norm_gold.py --dump-lut 生成），拷目录时按"*.mem 是生成物"漏拷。定位手段：R3C 编译产物 vvp 在两目录交叉运行隔离出 CWD 变量 → 枚举运行期文件 IO。修复：regression.sh 加 gen_lut 步骤，表与 R3C 孤本逐字节一致；tb_ae_actv 恢复 221184 字节位精确 PASS。
- REPORT_RTL.md 全文已完成（三处改动、验证、语义对齐含描述符编码分叉裁决、资源对照、复现命令、6 条诚实边界）；子代理写文件被工具策略拦截，全文已发回主会话落盘。

## 2026-09-04 11:27:00 — 24_pcw_rtn 阶段4（门禁线）：真实样本门禁交付 + 两次翻案（一次自查、一次根因级）

- gate_real.py（服务器 /tmp/pcw_rtn/）：评估集换真实样本 s000/s001（batch 来自 host_driver 同款加载 + kinematics device 修复），协议其余与 hw_calib mode B_v1 完全一致（import 复用，不复制数值代码）；4 种子 × 2 样本，判据 0.045
- 翻案一（自查）：初版从 v2 表重建 params 时 88 个 bias_fp_fallback 层 w_acc 置 None（bias 丢失，假红 0.2415），二版塞原始 bias（单位错，应为累加器域 b/(sa·sw)，0.1849）。修复后逐字段核对：v2 表与今日 fresh 标定 sa/so/sw/m 全部精确一致（430 层比值 1.000）——此前"v2 表是独立误差源"的说法撤回，标定表没有问题
- 最终数字：host_bias（compiler.py:31 部署约定：aug 层 K+1 进累加器、fallback 层 PL 出 int8 后 host 加 fp bias）s000=0.02167 / s001=0.02437 双绿；acc_bias（bias 进累加器理想口径）0.01855/0.02523。fp 参考与 fp32_ref_000.npz 逐位一致（jpos=0.00000），排除参考漂移
- 翻案二（根因级）：ab_quant.py（research_w8a8_error 的语义权威）pt 权重路径单位 bug——get_w 返回 qz(W,sw)*sw（反量化权重）进 acc，requant r=(sa·sw)/so 却按整数 acc 写 → 矩阵项压 sw 倍、bias 项正确，层输出≈只剩 bias；make_conv_fwd 同款，10 个 conv 输出全部缩 sw 倍（视觉流死亡）。单层实测：ab 输出 rel_vs_fp=1.04，我的实现贴 fp。上一轮"per-tensor 权重量化是 0.29 主因/pcW −75%/激活侧全无效"三个结论全部作废（都测在坏基线上）；V2c/V2_rn 的 pc 线性路径单位正确但 conv 仍坏，"视觉 sentinel 0.8~1.0 但动作到地板"的旧解释（LayerNorm 吸收）实为 conv 坏了
- 新根因陈述：0.29 = GEMM 量化在 815 次调用链上的逐级 requant 复利（only_gemm 0.2942 ≈ 全链 0.2975，非 GEMM op 只占 0.003；模块间 fp 复位时同样 430 个 GEMM 只剩 0.022，13.8× 全部来自深度）+ 合成标定对真实输入失配（patch_embed 输入 80% 超量程削顶 in_sat=0.802，105~115/430 模块超 calib×1.05；注意力层输入在量程内时单 GEMM max_rel 仅 2.0~2.4%）
- 影响与去向：RTL 线三处改动正确且验证完备（保留，基础设施不浪费），但 pcW+RTN 修的粒度在全深度账本里占小头；软件线全深度实测变为"测深度代价"本身，深度二分价值上升；门禁下版应加真实样本标定通道（train/test 分离）与 in_proj 覆盖（部署表含 in_proj_weight，门禁/hw_calib 目前留 fp）
- 交付：24_pcw_rtn/REPORT_GATE.md（完整调查过程+复现命令）、results/gate_real_results.json、results/gate_real_full3.log、REPORT_RTL.md 已落盘（RTL 线全文）

## 2026-09-04 13:30:00 — 24_pcw_rtn 阶段5（软件线）：pcW+RTN 全深度 bias 根因修复 + 深度二分曲线

- bias 根因（0.333 元凶）：mk_pcw_calib 首版把所有 pcW 层偏置改 host fp，但注意力 qkv 的 int8 输出在 PL 段内被 QK^T/PV 直接消费不回 host，q/k/v 偏置全部静默丢失（另 6 个 MHA in_proj 偏置键别名丢失）。修复 = 逐通道 K+1 增广 w_bias_j=round(b_j/(sa·swc_j·c))，c∈2^0..2^6 每层统一；注意力内部 134 层 c=64 溢出改逐通道饱和 ±127（真饱和 37 层 3373 通道），host 边界 99 层保持 fp fallback（精确）；compiler apply_calib pcW 分支改落穿 aug 判定
- 修复效果：s000 0.3328→0.2383（−28.5%）、s001 0.2698→0.1828（−32.2%）；对逐 tensor 基线 0.2992/0.2165 低 20.3%/15.6%；仍超判据 0.045 约 5.3×/4.1×（如实报红）。fast_selftest 9/9 全绿复跑通过。数字已与 results/result_*_pcwv3_vs_fp32.json 逐个核对一致
- 深度二分（host_driver 新增 --fp-after N；N=815 与全量跑 0.2383 完全一致，机制交叉验证过）：判据穿越 N≈180（特征增强/robot_encoder 入口，BERT 段 N=150 仍 0.030 绿）；主跳变 200–250（0.054→0.134）；250–750 平台 0.133；N=809 反常 0.6765——只留最后 5 个输出头调用走 fp 反而爆炸，全量（含头量化）0.238：静态标定在输出头的重量化把漂移拉回量程，混合精度尾巴不单调更好（记账已排除：5 个未跑段=5 个 native 模块，4 个缺输入为全量同样存在的预存行为）
- lever F 无效：F1（12 BERT FFN 前缀）0.2376、F2（38 前缀/261 调用点豁免）0.2384 vs 基线 0.2383——豁免 20% 调用点纹丝不动，证实误差是传播复利+输出头回拉、非 per-GEMM 贡献
- RTN 恒等式再验证：20 万随机向量（x 全域 ±2^26）RTN/floor 双模式 0 失配（12:04:42）；OP_SF 开销 101,294 字×8 拍≈4.1ms@198.5MHz=整帧 0.13%；表 v3 统计 s∈[21,30] m∈[16384,32767] 无钳位
- REPORT_SW.md 已落盘（全文由子代理交付、主会话转写）；含 RTL 团队用的 rn 公式+证明、OP_SF 格式、RTL 对齐 8 条清单、复现命令、10 条诚实边界；服务器 /tmp/pcw_rtn 进程已清理（gate_real 未动）。下一轮最值得跟进：N=809 输出头回拉现象（fp 头 0.68 vs 量化头 0.24，0.44 差值全来自 5 个调用的重量化回拉）

## 2026-09-04 15:13:41 — W8A8 B8 正确性门通过

- 新增 B8：Pack2 × Pump2、4 份 exact INT32 state、乘积校正和 INT32 recurrence 分两级，row0/row1 交错访问；不使用 INT24 chunk、snapshot 或 global flush。
- 服务器 Verilator 全量对拍通过：10,251 个 dots、1,032,501 个 transaction、0 错误，仿真用时 7.87 秒。Icarus + UNISIM 冒烟 55 dots、0 错误。

## 2026-09-04 15:23:06 — W8A8 B8 单核同流程比较完成

- B8 初版因 accumulator 上多余 keep 属性占 238 LUT；去掉后降到 143 LUT，减少 39.9%，OOC Fmax 保持 771.6 MHz，硬门槛全部通过。
- 同一 post-route 流程下，B8 为 688.7 MHz / 171 LUT / 1 DSP。相对 B4，Fmax 提高 14.3%、LUT 增加 11.8%；相对 B6，Fmax 提高 23.6%、LUT 减少 57.1%。
- B8 关键路径已经不是“乘积校正 + INT32 累加”串联。所有 route run 的 routing error、未约束内部 endpoint、DRC Error/Critical Warning 均为 0。

## 2026-09-04 15:43:42 — W8A8 B8 cluster 规模实验完成

- B8 16/32/64 PE AUTO 分别达到 636.5/619.2/606.4 MHz。64 PE 总吞吐 77.623 GMAC/s，每 DSP 1.213 GMAC/s，相对单核频率退化 11.9%。
- 16 PE SOFT PBLOCK 为 625.4 MHz，比 AUTO 低 1.8%，因此 32/64 PE 继续使用 AUTO。
- 四线程稳定完成，最大单次实验为 64 PE route，耗时 285.67 秒，所有实验均低于 10 分钟。

## 2026-09-04 15:48:06 — W8A8 B8 实验总结生成

- 生成单文件 HTML、Markdown 报告和 CSV 汇总。公式、必填字段、时间戳、DRC/路由状态和 HTML 结构自动检查通过。
- Power、真实 250/500 MHz 双时钟 wrapper 和 R3C 集成本轮未做。B6 只重跑到 16 PE；旧 64 PE 数据为 487 MHz，报告没有把 16 PE 结果冒充大规模结论。

## 2026-09-04 16:56:00 — 25_alg 三线（标定/SmoothQuant/输出头）：两线证伪 + N=809"回拉"翻案为输出坍缩

- 任务来源：用户 0904 提出的 12 实验矩阵（S0-S11，主流 PTQ 经验对照：SmoothQuant/RPTQ/FQ-ViT/LLM.int8/HAWQ-V3/BRECQ/PACT）。三代理并行（/tmp/alg_calib、/tmp/alg_smooth、/tmp/alg_head），S0 复用基线 0.2383/0.1828 不重跑
- 标定线（REPORT_CALIB.md，25_alg_calib/）：假设证伪。S1 全网真实 absmax 标定 0.2710/0.2097（+13.7%/+14.7% 更差，in/out 一致）；p99.9 口径 0.2313/0.1855（−2.9%/+1.5%）；仅 patch_embed −1.0%、仅穿越段 −0.4%、段内乘子扫描 ±0.5% 噪声带。机制：削顶被下游 LayerNorm 幅度归一化吸收，为消削顶放大 sa 反而让步长变粗、在 583 次 GEMM 复利下代价更大——分辨率比削顶值钱但量级 ≤3%。副产品 S10：全网 22.2 万通道 outlier 清单（top 0.1% 比值 57-115×，聚集 backbone.stages.3.blocks.0.ffn.layers.1 等；BERT 段典型通道仅用量程 ~15%），在 diag_real_s000.json
- SmoothQuant 线（REPORT_SMOOTH.md，25_smooth_quant/）：机制生效、误差不动。S5 robot_encoder 10 组 0.238131（−0.09%，fp 等价门 6.6e-08）；S6 全域 102 组 0.235129/0.180371（−1.3%，混 0.0064 fp 地板要打折）。通道比值压平真实（中位 3.3→1.9/4.7→2.2，超标率→0），但 per-tensor sa 被最大通道钉死（0.76-1.11）——压平小通道换不来分辨率，权重侧 swc 比值还恶化到 26×。结论：激活逐通道动态范围不是主因，RPTQ/FQ-ViT 型激活整形在本链不成立
- 输出头线（REPORT_HEAD.md，25_alg_head/）：① 调用图修正：全链 815 次调用，5 个头调用=#810-814（convs.0/convs.1/ol.1/ol.3/ol.4），头纯读出全流程调 10 次、末次主导。② S7 瀑布：#811 convs.1 主回拉（−0.318，占 73%），convs.0 −0.071、ol.4 −0.051，ol.1/ol.3 轻微有害。③ S8 A2（权重fp+激活requant）=0.6825、A3（权重W8+不requant）=0.6832——任一成分单独拿掉都塌回 0.68，回拉是 W8 舍入×requant 舍入的交互效应。④ S9 so 乘子：×2.0→0.2059（−13.6%，全部来自 convs.0 单层），×3.0 平台，距判据 4.6×。⑤ **机制翻案：回拉=输出坍缩**——部署基线头输出 72/112（关节,参数）单元格在 64 步上精确恒定（9/14 关节全 8 参数单一值）；fp32 参考 9/14 关节近常数（std≤0.02）、坍缩常数在 6 个押中，0.68→0.24 是记分运气不是精度恢复；该动的关节 8/9/10/13 仍死（j8 误差 0.86）；out_sat 全 0、rail 0——"饱和即收缩"假说被 T4 直接否定。0904 根因陈述第 2 条的输出头部分应改写
- 三线合计：主流 PTQ 三板斧（标定/激活平滑/头策略）在本链全部测完、全部无效或边际（最好 0.2059=so×2 反坍缩，距判据 4.6×）；每条死路都有机制级解释，唯一存活的 main hypothesis=逐级 requant 复利把信号 SNR 磨到头输出常数化。下一档杠杆（按用户文献框架）：逐级 requant 本身动刀（少数关键边界 int16/int24 中间态保留、HAWQ 式 mixed boundary policy）或 QAT/块重建（BRECQ/OmniQuant，jpos 目标函数）
- 交付：REPORT_{CALIB,SMOOTH,HEAD}.md 三份全部落盘（数字主会话逐个从服务器 json 复核）；脚本本地副本 25_alg_calib/、25_smooth_quant/sw/、24_pcw_rtn/sw/head_probe.py；三代理进程清理确认，/tmp/pcw_rtn 与 /tmp/ae_hostdrv 零改动

## 2026-09-06 22:50:00 — 25_alg 轮 HTML 重写为白话版

- 原因：0904 首版（2026-09-04_1658_算法三线探索与回拉翻案.html）术语密度过高被用户退回（"我看不懂"），且中文文件名在 IDE 链接中被 URL 转义后解析失败
- 重写版：round_report_alg_ptq/2026-09-06_2249_plain_rewrite.html（纯 ASCII 文件名；实验数据与首版完全相同，仅表述重写：术语首现即解释、一句一事、数字带意义、图表配"怎么看"）
- 旧文件已删除

## 2026-09-07 02:48:10 — 26_ref_denoise：四参照梯定案"8 位格式够用、整数执行背锅"，去噪迭代证伪放大，缺口 87% 定位 attention 占位标定常数，真尺度上链 −19.3%/−23.5%

- 任务来源：用户 0906 评审——五条结论收回意见 + 下轮目标四参照对比 + 去噪"步内误差 vs 跨步放大"分离 + INT16 先软后硬。两代理并行（/tmp/alg_refq、/tmp/alg_denoise），全部头条数字主会话从服务器 json 逐个复核，A6=0.02577 由主会话本人复跑确认
- **四参照梯（REPORT_REFQ.md）**：R0 fp32=0（定义）；R1 部署图全 fp=0.0021（既有 --fp-after 0）；**R2 独立假量化（同位置同 scale 同网格、浮点执行舍入乘加、量化数学独立重写不复用 requant 代码）= 0.0208/0.0209（s000 双种子）、0.0295/0.0220（s001）——低于 0.045 判据**；R3 整数链 0.23834/0.18275。三验证门：fp 直通 6.99e-08；round(W/swc) 与 pcw_export 逐位一致（19 万元素差 1 LSB）；单模块 vs 整数链 rel 6e-4（15-bit 乘子预期量级）。**判据树落 ③≪④ 分支：0.2383 的 85~90% 是整数执行细节，不是 8 位格式**。变体：W8Afp 0.0085（权重几乎免费）、A-only 0.016、int16 网格 −18.9%、floor 差 3 倍（RTN 对）、bias 增广 +0.003、BERT mask ≤0.002、requant 乘子全链定界 0.031（alg_denoise drfix 0.0306/0.0232）。两套独立 harness 互证（refq V1=0.0208 / denoise free=0.01969）
- **去噪解剖与隔离（REPORT_DENOISE.md）**：10 步 DPMSolverMultistepScheduler（dpmsolver++ order2 sample，配置 model.config.json /decoder/base_cfg/test_noise_scheduler），反馈只走 noisy_action dim0 一条窄通道（dim1-6 每步由 recompute() 正运动学重建）。**"815 层串行复利"双杀**：结构上 815=215 一次性前缀+60/步×10 步（275 边界×调用次数），"decoder.layers 走两遍"是误记（66 槽每层每步一次）；机制上每步传导比全程<0.5（回灌先衰减一半以上）、步 0-7 单独量化 ≈3e-05、只量化末步=TF=TF+=0.00889008 逐位相同（绿，调度器 model_outputs 历史零贡献）、状态存储单独 int8 仅 4.69e-05（**16 位预算不必给去噪状态**）、整数链 bisect 第 2-9 步 500 调用零增量。free=0.01969 构成：前缀条件偏差 77% + 末步出口 45%（近似可加小幅抵消）
- **attention 三档梯子+实修（REPORT_ATTENTION.md）**：A1 S 压 int8 +1.6%、A2 整数 exp 表 +1.2%、A3 P 压 1/127 ≈0——**A4（v 码 σvs+PV 段 requant 用部署占位常数）一档占缺口 87.4%（0.0208→0.2110，与部署动作 corr 0.947）**；A5 同结构换真 absmax=0.02722、A6 完整整数执行复刻（V1 背景）=0.02577。家族：temporal 75.6%、rotary 29.7%、WindowMSA 6.9%、其余噪声（temporal_A4 输出波动放大 542 倍）。机制=compiler.py L1139/L1163 占位常数（PV 码欠程约 10 倍）非实测。**真尺度上链（工具链零代码改动，--attn-calib 数据通道，fast_selftest 全过、不加开关重编逐字节等于原 build）：s000 0.23834→0.19222（−19.3%）、s001 0.18275→0.13991（−23.5%，out-of-sample 更大）**。附带：query-thrice 接线 e2e 增量恰 0（不修，归档）；输出头 11/112 恒定格与注意力无关不随修复解除
- **主会话警告（下一轮必做）**：代理判"真链残差主因=GEMM requant 复利"所引旧证据（base bisect N=250→0.133、only_gemm≈全链）都是在注意力常数还坏着时测的，被混淆；且两套 harness 均未仿真 AE_ACTV int8 NORM/ELTWISE（148+148 站点）、SM16/SM32 softmax、rotary 读 int8 中间值——A6 与真链间一整类语义差未定界。下一轮：修复链重测 bisect + 仿真 ACTV 族，归因关闭后再排 int16 边界/QAT 顺序
- so×2 终局（C 线）：s001 上 +6.1% 反噬（0.1940），样本特定不泛化，放弃、不并入基线（与用户 0906 裁决一致）
- 交付：hb_fpga_impl/26_ref_denoise/REPORT_{REFQ,DENOISE,ATTENTION}.md；HTML round_report_ref_denoise/2026-09-07_*.html；/tmp/pcw_rtn 与 /tmp/ae_hostdrv 零改动（swfix 为副本且链上三文件 diff 逐字节相同）；记忆已更新

## 2026-09-08 13:12:00 — 27 轮收官：0.16 全归因（接线 78% + 头行错排 + conv im2col padding），修复链 0.1922→0.0500（−74%）/0.1399→0.0622（−56%）；600MHz PE 后硬件/架构路线讨论页

- **0.16 归因关闭（三根因，148 旗标全归因无第四类）**：①id() 地址复用错喂（0.192 中的 0.150=78%，SSA 修复四门全过）；②头/行错排（_pop_heads 返回 2D 块、stack(qs,1).view 必错排，6 处改 stack().reshape，canonical 老链同带——0.1922/0.13991 锚点是带病成绩单）；③conv im2col padding 一行 bug（标量 pad 垫穿尺寸 1 的假维→A 矩阵 2/3 零行，在动作出口+自回归放大；`padding=(0,pad)` 一行修）。
- **修复链终局**（服务器 /tmp/alg_fix/，主会话从 fix_summary.json 逐项复核）：SSA 0.2486/0.2571 → stackfix 0.2450 → convfix **0.04998（−74.0% vs 锚）/0.0622（−55.6%）**；旗标 24→0；公平性门全程不变（fallback 27/missing 59/segments 3118）。距判据 0.045 差 11%；语义天花板 B5b 0.04247。
- **三个负结果定案**：JG S×1.5=打乱参照伪影（对正确逐头积 α=0.998-0.999，requant 常数未动）；k 侧 1.07=文本指令路径 6 层量化深度损伤（feature_enhancer 21 行文本特征，LayerNorm 放大成去相关；b5b 同指纹=在 0.04247 天花板内，不修）；input_layers 簇=纯继承（fp 回退从未量化，conv 修后自然落 0.053）。修后 decoder 族中位 rel 0.0659 vs b5b 0.0656——修复链与语义天花板贴合。
- **下一档杠杆**（归因关闭后首次定义良好）：6 个文本注意力块边界保 int16 的软件 A/B → 不够再 QAT。剩 0.0050 差距无驱动侧位点。
- 交付：/tmp/alg_fix/REPORT_FIX.md §9~§15、results/fix_summary.json phase3、walk_table_v2.json、rotcap/kproj_hook/segcap_kproj 探针；破案过程页 round_report_routing_bug/2026-09-08_1252_*.html。
- **路线讨论页**（用户口令：600MHz PE 之后硬件/架构两线分工）：round_report_hw_arch_next/2026-09-08_1312_hw_arch_plan_after_600mhz_pe.html。要点：B8（1.213 GMAC/s/DSP=现 R3C PE 6.1×，位精确）把计算需求 933ms→~153ms，但读需求 1.47s 没动→帧时间只省 10-25%，墙全在喂数；硬件线 H1=B8×R3C 快照 micro 门（生死题）、H2=集群 64→256 采点、H4=600MHz 功耗；架构线 A1=pe_sizing/r3c_model 加 B8 旋钮（列数×PE 类型扫描）、A2=读压缩重排（双读引擎 292→184M、CTX 驻留消 548MB 往返）；现状 1.39s 已过 2.13s 实时预算（裕量 1.5×），性能线买的是裕量/功耗/面积。决策点 D1（先 H1）/D2（A1 含 R3C 对照线）/D3（B8 失败退路=R3C+读压缩）待拍板。

## 2026-09-08 14:28:11 — 架构线 v5：B8 列数选型定案 B8-48（帧时间 −35%、DSP −50%），B8-96 因 LUT 放不下排除；读压缩三路线重算（真机无帧时间收益、TB 写墙新暴露）；三线目录 + 全局工作流 skill

- **A1（B8 旋钮进周期模型）**：B8 = 4 MAC/接口拍/DSP（Pack2×Pump2）+ 接口时钟 303.2 MHz（cl64 606.43 实测/2）+ 逻辑列=2×物理列 + PE 169 LUT/颗（cl64 实测），塞进 pe_sizing 重编译口径，扫物理列 {24,32,48,64,96}×{R3C,B8}。**定案 B8-48（逻辑 96 列）：HP64 帧时间 0.909s（−34.5%）、DSP 768 颗（−50%）、LUT 估 76%、每帧拍数与 R3C-96 一拍不变只换时钟**。B8-96 光 PE 核 259,584 LUT=器件 113% 放不下——**B8 第一约束从 DSP 变成 LUT（第二功耗）**；R3C 窄到 48 列以下破 2.13s 预算而 B8-24（22% DSP）仍 −9%，窄阵列第一次划算。B8 利用率 26.5%（R3C 53%）= 行组下限 68 拍接口域地板（结构性，非 bug）。
- **敏感性两条（进 RTL 需求 B3/B4）**：requant 不加倍（DRAIN 128）→ GEMM +16%/帧 +10.8%，必须躲；读出链 2× 宽 → 只 −6~8%，可选。
- **A2（读压缩三路线 B8 语境重算）**：真机 HP64 读侧仅 20.1M 拍从来不是墙，三路线帧时间收益全在 TB 口径——路线 2 双读引擎仍是单刀（B8-48 TB 1.125→0.909s 两口径归一）；路线 1 价值重定位为 ctx 字节 680→469MB（−31%，功耗项）；**新暴露 TB 写墙 W=218.4M 拍**（B8-64/96 读压缩后 TB 卡 0.852s），R5 写侧整形成 H3 下一张牌。
- 交付：arch/v5_2026-09-08_1414_b8_sizing_read_compress/（b8_scan.py ~2s / b8_scan.json / rtl_requirements_h3.md B1-B7+R1-R3 / 2026-09-08_1428 HTML）；arch/CHANGELOG.md v5 条目；**全局 skill ~/.claude/skills/lines-workflow**（三线版本纪律+实验规矩+报告规矩）；三线目录（algo/arch/hw/compiler/plans + LINES.md + 四份 CHANGELOG 历史映射）本轮 13:42 用户拍板建立。

## 2026-09-08 18:10:00 — hw v4 / H1 门：B8×R3C 集成收官，位精确全绿（阵列+引擎双宽度）、全宽 768DSP 引擎布线落地，时序差 5% 定位两条读出路径；修出底本三个潜伏问题
- 任务来源：用户 1440 口令"去做一版硬件的实现吧，看看当前的方案是否合适的"——把架构线 v5 定案的 B8-48（48 物理列 × Pack2，768 DSP，303.2MHz，0.909s）写成真 RTL 过 H1 微门，kill 线=位精确不过/WNS 收不了/LUT 每对超 250。
- **RTL 三件套**（hw/v4_2026-09-08_1503_b8_h1_gate/rtl/，底本 22_r3c_rtl 未动）：ae_pe_p2（Pack2 脉动 PE，积落地 6 拍，脉冲拍优先级=快照>清零>累加）、ae_sysarr_p2（逻辑列展平读出）、ae_gemm_p2（PULSE_DLY=5 脉冲延迟线；requant 24 套=逻辑列/4）。
- **位精确**：末脉冲安全窗口实测 {5,6}（PD=7 时脉冲拍丢弃下组首积，比纸面推导窄 1 拍）；引擎级 PCOLS=4 六描述符 + 全宽 48 九描述符全 PASS（全宽本地 ~100s）；三轮 RTL 修复（见下）每轮双宽度复跑，逐位且逐拍不变。
- **底本三个潜伏问题（全部修复在本轮文件，待回移）**：①互锁 bug——k<59 时脉冲经 drain row≥12 逃生舱发射后 pend 被旧 walk 完成清零，下组脉冲过早发射覆盖未读快照（m=35/k=37 用例击穿，row5..15 全零），svc_r（读侧消费放行才重臂）修复；生产 k≥64 从未触发。②地址乘法——(行组号+1)×k / 行组号×n 的 16×16 乘法 18 级逻辑，250MHz 无事、@3.298ns 综合全 4 条违例（−0.280ns），换基址寄存器 +k/+n 增量。③读出长路径——行选择 mux→requant 桶形移位跨模块单拍，布线 −1.097ns（R3C 同族弱路径，v3 OOC 也是 −1.363），引擎侧加一拍 acc_rq_r 读出寄存 + 换行提前 slot==2 + r15_seen 防第 16 行漏读（第一版踩坑：停发挂 drain_row==15 提前一个窗口触发，requant 捕获停在 15/16 卡死，探针定位）。
- **综合/布线 @3.298ns OOC**：PE 120 LUT/1 DSP/布线后 675MHz；16×4 条带 7,703 LUT/布线后 395MHz；16×48 阵列 92,055 LUT（39.95%）+768 DSP（44.4%）/综合 +2.054——**每对 119.9 LUT，kill 线 250 的一半**；全宽引擎布线后 118,148 LUT（51.28%）/181,333 FF/768 DSP/WNS −0.169（读出修法从 −1.097 收回 0.93ns，违例端点 7543→716；AggressiveExplore phys_opt 不再改善）。剩余两条腿：drain_row 扇出（96 列 16:1 mux 一拍）、requant 入口一拍塞 4:1 mux+27×8 乘——H2 修法明确；R3C 先例真机比 OOC 快 ~6%，288MHz 保守口径帧 0.957s（vs R3C-96 −31%），303.2MHz 收口则 0.909s（−34.5%）。
- **模型修正**（results/model_corr.py）：v5 读腿公式漏计 ptap 放行——R3C-96 真实 167+wb 拍、B8-48 124+wb（每读出界行组反少 43 拍），帧区间 B8-48 0.909~0.995s vs R3C-96 1.388~1.627s（相对 −34.5%→−38.9%）；requant 无需加倍坐实（24 套 DRAIN 64 拍不变，v5 的 +10.8% 惩罚场景取消）；**无需双时钟**（B8-48 映射 Pump2 第二相闲置，单时钟 303.2MHz 每拍每 DSP 2 有效 MAC）。
- 交付：hw/v4_2026-09-08_1503_b8_h1_gate/（rtl/sim/synth/results + 2026-09-08_1600 HTML，18:10 回填布线终数）；hw/CHANGELOG.md v4；综合工作区 E:\ae_syn\hb_h1\（含 phsopt/phsopt2 探索档证据）。H1 判定：位精确 ✓、LUT ✓、时序差 5% 非结构性（kill 线"收不了"指结构性失败，本例有明确收尾路径）——**B8-48 方案成立，进 H2**。

## 2026-09-08 22:08:53 — hw v5 / arch v6：H2 时序收官（eng48 WNS +0.009 @303.215MHz 收敛）+ R2 双读引擎落地（段级 −7.4%）+ B8-64 采点关死翻案窗口——定版 B8-48
- 任务来源：用户 1945 口令"完成你说的这四项任务……架构和电路都要更新一版，做完之后整理成 HTML"——上一版架构报告点名的四件事：R2 双读引擎 RTL、H2 时序收口、B8-48/64 定版采点、R5/COPY 进 H3 清单。
- **H2 两条读出腿（hw/v5 rtl/，底本 v4 改造）**：①drain_row 扇出腿——4 位寄存器一拍驱动 96 逻辑列的 16:1 快照选择，换 12 份同值副本（NREP=(PCOLS+3)/4 每份 4 物理列，iverilog 不支持数组端口→打包向量 [ri*4+:4]）；②requant 入口腿——累加快照→4 选 1 slot→27×8 窄乘挤一拍，换 rq_ms_x 输入流水（x_sel_r 先寄一拍），数据滞后 2 拍，FSM 相位前移一拍补回（rq_v 发射 slot==3→2、换行 slot==2→1、停发/走尾同步前移，requant 消费窗口与 H1 逐拍重合、走读仍精确 64 拍；例外=进 DALIGN 撞 slot==3 时多绕 1~4 拍）。
- **H2 验证**：位精确 PCOLS=4/48/64 + tb_sys PD=5/6 全 PASS；拍数纪律 48 列 9 描述符 6 个一拍不差、D3/W0/W2 +4 拍（+0.9~1.5%）；帧级 DRAIN 64→65 敏感性 GEMM +0.25%/TB +0.0%/HP64 0.909→0.910s。**布线终数：eng48 WNS −0.169→+0.009（303.215MHz 收敛），代价 +208 LUT（+0.18%）/+733 FF**（=24 套×28b rq_ms_x 流水+48 副本 FF 的账）；功耗 8.321W vectorless。
- **B8-64 采点与定版**：eng64（64 物理列/1024 DSP）位精确 9 描述符 PASS；综合 LUT 155,629（OOC 67.55%，与修正锚点外推 157,530 差 1.2%）、WNS −0.696（拥塞型 7,313 失败端点/TNS −1,531.6ns）、功耗 10.709W（+28.7% vs eng48）。**裁决：定版 B8-48**——三条件两不过（WNS/功耗），补刀：降频 250.4MHz 用 HP64=0.970s 反而比 B8-48 的 0.909s 慢，翻案只剩深度重流水（不可预期工作量）。
- **arch v6**：LUT 模型锚点修正（v5 每列 3140 高估 27.6%→2461.4 LUT/列实测反推，B8-64 从 98% 贴线变 78.4% 有余量，这正是要采 eng64 点的原因；采完关死）；R2 模型修正——**−19.1% 是理想上界，4 段实测 −7.4%**；R5 写侧压缩 B8-48 下收益为零不进 H3、COPY 40.3M+ACTV 10.0M 拍=计算腿 18.3% 先编译器侧分解。
- **R2 双读引擎（hw/v5 r2/ 子目录，R3C 底本 22_r3c_rtl 一字未动）**：光复制读引擎没用——三个串行点：单 rd FSM、单 AXI 读通道+单 outstanding 从机、调度器 T_RUN_DMA 死等（**架构实质修正：必须同时改调度器发射策略为 fire-and-forget**）。改 ae_rd_eng ×2+第二读主口+消费点等待；4 段位精确逐字节一致（服务器 Verilator，base 与 r2 DDR dump cmp 全同），段级 −7.4%（141,647→131,235 拍，读服务 −8.1%，两口重叠率最高 15.5%）；与模型 −19.1% 的差距=单 outstanding+段形态（无 W 段白付、段头大 ctx 无重叠窗）+帧级外推明确不做。
- 交付：hw/v5_2026-09-08_2000_h2_r2_b64_pwr/（rtl/sim/synth/results + r2/ + 2026-09-08_2140 HTML 22:08 定稿）、arch/v6_2026-09-08_2000_h1anchor_r2_b64_final/（b8_final.py/json + measured.json + h3_r5_copy.md）；hw/CHANGELOG v5、arch/CHANGELOG v6；综合工作区 E:\ae_syn\hb_h1\（eng48/eng64 route 级含 power.rpt）、服务器 /tmp/ae_v5r2/。
- 遗留：R2 的 12 项回归只重跑 2 项；段级 golden 门 fast_interp_a3 预存 diff（与 R2 无关，建议单开一轮查）；R2 验证从机行为级单 outstanding，实机重叠率会更低。

## 2026-09-09 09:52:42 — hw v5 报告重写（说人话）+ arch v6 架构分析页：墙序列 计算→写→读；修正 COPY 全消地板 −18%→−6.3%
- 用户 0941 反馈 0908 的 hw v5 报告"不够参考说人话 skill"→ 重写版（数据全部不变、行文重排：第零节加术语表、一句一件事、图景先行、每节"怎么看"），新文件不改旧页：hw/v5_2026-09-08_2000_h2_r2_b64_pwr/2026-09-09_0948_H2时序_R2双读_B8定版_重写版.html（页头注明取代 2140 版）。
- **架构分析页**（arch/v6_2026-09-08_2000_h1anchor_r2_b64_final/2026-09-09_0952_架构分析_B8-48定版后的时间账与优化方向.html，5 张 SVG 图，数字全部从 b8_scan.json/b8_final.json/综合报告核对）：
  - **修正一处上版账目错误**：H3 清单"COPY 全消地板 0.909→0.744s（−18%）"只做了计算腿减法、没套帧模型 max()——写腿 218.4M 先拦住，正确帧级地板 0.852s（−6.3%）；COPY 的真实价值必须与写压缩捆绑评估。
  - 墙序列（TB 口径、R2 后起步 0.909s）：计算墙 235.5M → 写墙 218.4M → 读墙 184.0M；真机口径无读墙（20.1M）。
  - 杠杆按解锁顺序：①停顿吸收 −14.5%（→0.777s，调度重拍不动数据通路）→ ②COPY 消除（①后再 −7.3%，被写墙封顶）→ ③R5 写压缩（②后再 −15.2% → 0.611s 三牌全兑现地板，−32.8%，零 DSP）→ ④requant 加倍远期（真机终段 0.565s；TB 被读墙拦在 0.607s）。
  - **R5 裁决细化**：从"不进 H3"改为"条件启动"（现在做零收益维持原判，计算腿杠杆落地后它是唯一解锁牌，进 H4 候选）；GEMM 利用率 26.5% vs MAC 地板 49.1M 拍（3.8×）为结构性地板。
  - 建议路线：H3 系统集成（全帧实测校准模型+验证①）+ 编译器 COPY 分解并行（决定②值不值得动 RTL）。

## 2026-09-09 10:52:00 — 三线结构 + hw v4/v5 + arch v5/v6 + 算法线 24~26 推 GitHub

- 提交 70b9635（296 文件，6.2MB）推 github.com/nc-thu/vector-core-r3c main：三线目录（LINES.md/hw/arch/algo/compiler/plans + 四份 CHANGELOG）、hw v4（B8×R3C H1 门）+ v5（H2 收敛/R2 双读/eng64 采点定版 B8-48 + 0948 重写版）、arch v5/v6（b8_scan/b8_final + 0952 架构分析页）、hb_fpga_impl 24~26（pcW/RTN RTL+软硬联合、算法校准/头排/平滑量化/去噪归因）、README 改三线结构说明。
- 公开前清理沿用 0902 约定：24~26 目录与 WORKLOG 中的内网 IP/登录串/服务器家目录路径占位符化（<SERVER>/~）；.mem 回归转储（44MB）、r2 段级 DDR dump、arch/figures（Visio 产物）、.vvp/.pyc 不推；token/密钥扫描零命中；Vivado 报告头裸主机名按 0902 裁决保留（31 个 .rpt，非 .rpt 文件零命中）。
