# HoloBrain-0（HB-GD 0.2B）调研报告

调研时间：2026-08-30。方法：论文（arXiv 2602.12062 HTML 全文）+ 官方开源代码逐文件核实（RoboOrchardLab，master 分支）+ 官方 checkpoint 配置（HF HorizonRobotics/HoloBrain_v0.0_GD）三方交叉。所有结论带出处；算不死的给区间。

## 0. 结论速览

1. **模型真开源了**（不像 SwiftVLA 只有论文）：代码、预训练权重、RoboTwin/GraspAnything 后训权重全部可下载。但 **LIBERO 的后训权重和评测配置没放**——96.7% 无法用公开材料复现。
2. **悬案①定案（高置信）：LIBERO 96.7% 的输入含 MuJoCo 渲染的完美 GT 深度图**。已发布权重结构上强制吃 depth（不喂直接 crash）；repo 专门写了 LIBERO 数据生成器开 `camera_depths=True` 渲染两路深度。拿 96.7% 和 RGB-only 的 π0/OpenVLA 横比要打折扣。
3. **"GroundingDINO" 只是半个**：检测头整个删掉了，留下 Swin-T 骨干 + BERT 文本编码器（冻结）+ 6 层特征增强器。真正的新东西是 PSE（深度空间增强器，很小）和 20.8M 扩散动作专家。
4. **计算量约为 SwiftVLA 的 1/5.7**（LIBERO 主档 73.7 GMAC/chunk vs 417.8），权重流约 1/4.2（334 MB vs 1.41 GB）。108 列阵列 CONC 口径 0.27 s/chunk，30 Hz 预算 2.13 s——裕量近 8 倍，实时性不是问题。
5. **真正的难点不在乘加，在算子杂**：deformable attention（数据相关的稀疏采样）、3D 反投影、深度分布 softmax、关节图注意力、rotary——这些占 MAC 不到 5%，但都落不到阵列上，要 CPU/软逻辑兜着。工程量在喂料和数据重排，不在算。
6. 对 FPGA 路线的建议：**值得作为 SwiftVLA 的替代/并行对象推进**，但先想清楚谁来算 deformable attention 和深度前处理。

## 1. 调研来源

| 材料 | 位置 | 说明 |
|---|---|---|
| 论文 HTML 全文 | arxiv.org/html/2602.12062v1 | 结构图表、超参表（Table A1）、LIBERO/LIBERO-Plus 数字 |
| RoboOrchardLab 源码 | `research_holobrain/robo_orchard_lab/` | 官方训练/评测框架，含 holobrain 模型全部定义与 LIBERO env/dataset |
| 官方 checkpoint 配置 | `research_holobrain/hf_cfg/` | 从 hf-mirror 直下的 model.config.json + 三个 processor（RoboTwin×2、真机） |
| 权重文件 | HF HorizonRobotics/HoloBrain_v0.0_GD | model.safetensors = 735,423,856 B，fp32 存储即 **183.9M 参数**，与论文 0.2B 对上 |

GitHub 直连被墙，仓库经代理取 master 分支 tarball；HF 经 hf-mirror。

## 2. 模型结构（逐项核实版）

HB-GD 类名是 `BIP3D`（建在地平线自家 BIP3D/SEM 系列上），五段式：

```
RGB ×N视角 (320×256) ─→ Swin-T 骨干 (27.5M) ─→ ChannelMapper (2.1M) ─┐
                                                                    ├→ 6层特征增强器 (21.9M, 含 deformable attn)
指令文本 ─→ BERT-base (108.9M, 冻结) ─→ text_feat_map (0.2M) ─────────┘        │
                                                                              ▼
深度图 ×N视角 ─→ 微型 Swin (0.79M, embed仅16) ─→ neck_3d ─→ PSE 深度融合 (2.3M 合计)
                                                                │
joint state (每关节8维) ─→ 状态编码器 4层 (6.1M) ──────────────→ 动作专家解码器 6层 (14.7M)
                                                    + UpsampleHead (0.13M) = 20.85M
                                                    扩散去噪 ×10 步 (DPMSolver++, x-prediction)
```

参数对账：160.6M（2D+文本）+ 2.28M（PSE）+ 20.85M（专家）= **183.7M** ≈ safetensors 推算 183.9M ≈ 论文 0.2B。三方吻合，账可信。

### 逐模块说明

- **Swin-T 骨干**（每视角）：320×256 输入，patch 4×4 后 80×64 token，4 个 stage，输出 stride 8/16/32 三个 level（40×32、20×16、10×8）。8.48 GMAC/视角。
- **BERT-base**（108.9M，占全模型参数 59%）：冻结，只在指令变化时跑一次（LIBERO 每任务一条指令 → 可缓存，摊薄后近零）。指令按 16 token 估 1.36 GMAC。
- **特征增强器 6 层**（21.9M）：每层 = 图像 deformable 自注意力（4 个尺度、每查询点采 8 头×4 尺度×4 点=128 个采样点）+ 文本自注意力 + 文本↔图像双向注意力（只在最后一个尺度，每视角仅 20 token）。2 视角合计 26.4 GMAC——**这是全模型最重的单块**，也是 FPGA 最不友好的（deformable 采样）。
- **PSE 深度增强器**（合计 2.28M，全模型 1.2%）：微型 Swin（embed 16）吃深度图 → 预测每个图像位置的深度分布（128 个 bin，0.01–1.2m）→ 用相机内外参把 2D 特征反投影成 3D 位置编码 → 与图像特征融合（fusion_fc 1.18M 是大头）。约 2.6 GMAC/视角。相机固定时内外参反投影可整体离线缓存。
- **动作专家 20.85M**（论文 20.8M，差 0.3%）：token = 关节数 × 16（64 步动作按 4 步一组压成 16 个 chunk token；RoboTwin 14 关节 = 224 token）。每层 = 时间-关节图注意力 → 图像 cross-attn（K/V 长 1600/视角）→ 文本 cross-attn → AdaRMSNorm（扩散时间步条件）→ FFN2048。UpsampleHead 把输出展成 64 步 × 8 维混合动作（Δ关节角 + Δ末端位姿）。去噪 10 步每步全过一遍解码器。

### 部署形态（视角数/关节数随场景变，同一权重通吃）

| 场景 | 视角 | 分辨率 | 关节 token | 备忘 |
|---|---|---|---|---|
| LIBERO（Franka） | 2（agentview + 腕部） | 320×256 | 8（推断，无官方 processor） | 唯一没放权重的场景 |
| RoboTwin（双臂 piper） | 4 | 320×256 | 14 | 已放后训权重 |
| 真机（双臂 + RealSense） | 3 | 384×256 | 14 | 已放后训权重 + 部署指南 |

## 3. 三个悬案的答案

### ① LIBERO 96.7% 用没用 GT 深度？——用了（高置信）

公开代码无法直接复现 LIBERO 成绩（无 eval 脚本、无权重、无训练配置），但证据链五环全部指向同一方向：

1. **已发布权重在结构上强制吃深度**：`with_feature_3d=true` 时 `DepthFusionSpatialEnhancer` 对 None 深度直接抛 TypeError（`models/bip3d/structure.py` 的 extract_feat → deformable_format）。不喂深度不是"降级运行"而是 crash。
2. **repo 专门写了 LIBERO 数据生成器并开深度渲染**：`dataset/libero/generate_dataset.py:201` 显式 `camera_depths=True`，特征表硬编码存 `agentview_depth` 和 `robot0_eye_in_hand_depth` 两列。
3. **深度来源是渲染器直出的完美 GT**：robosuite `get_real_depth_map` 把 MuJoCo OpenGL 深度 buffer 反变换成米制深度，无噪声、无空洞、无效点（`envs/robosuite.py:30-51`）。单测还验证过投影对拍 <5cm。
4. **三个已发布 processor 全部 `load_depth=true`**，且推理管线 assert `data.depth is not None`——深度是部署硬性输入。
5. 若 LIBERO 不喂深度，就得专门训一个 `with_depth=False` 的变体——与论文"深度是正式输入"的设定矛盾，且没有任何变体发布的迹象。

**含义**：96.7% 是"完美深度加持"下的数字。真机深度相机（RealSense，有噪声/空洞/量程）存在 sim-to-real 质量差——他们真机能跑（GraspAnything 93.5%），说明对深度质量有一定鲁棒性，但 LIBERO 榜面数字与 RGB-only 方法不可直接横比。

### ② GroundingDINO 实际是什么？——半个（骨干+文本+增强器，无检测头）

- 检测相关的 query/box head 全部不在结构里（config 无对应字段，forward 不经过）。
- 实际用的三件：Swin-T（27.5M）+ BERT-base（108.9M，冻结）+ 6 层 TextImageDeformable2DEnhancer（21.9M）。
- 输入 320×256（Table A1 原文），token 网格 80×64；四个尺度（stride 4/8/16/32/64 里取 8/16/32/64 共 1,700 token/视角）。
- 这一路合计 **45.1 GMAC（LIBERO 2 视图 + 16 token 指令）**，其中 BERT 只有 1.4 GMAC 但参数占大头。（核验修正：bi-attn 双向注意力作用在 stride-64 级、每视角仅 20 token，此前按 80 token 算高了 0.6 GMAC。）

### ③ PSE + 深度头 + 动作专家的真实计算量（账本定稿：LIBERO 主档 = 2 视图 + N_j=8）

| 模块 | 参数 | MAC | 备注 |
|---|---|---|---|
| PSE 全部（深度 Swin + 分布头 + 3D 融合） | 2.28M | 5.2 GMAC（2 视角） | 其中非 GEMM（反投影/加权聚合/深度 softmax）约 0.4 G，量级清单在账本 not_modeled |
| 动作专家（含 10 步去噪、状态编码 ×1） | 20.85M | 23.4 GMAC（N_j=8） | K/V 每步全量重算无缓存（核验员确认 action_decoder.py:654-666）；hoist 优化可省 8% MAC |
| 视觉 2D（Swin-T+neck ×2） | 27.5M | 17.3 GMAC | 含 window padding 浪费（padded token 5880/1470/441/196） |
| 增强器 6 层（2 视角+文本块） | 21.9M | 26.4 GMAC | 全模型最大单块 |
| BERT+映射（16 token 指令） | 108.9M | 1.4 GMAC | 每 chunk 仅 1 次，冻结可按指令缓存 |
| **全模型（主档）** | **183.8M** | **73.7 GMAC/chunk** | SwiftVLA 417.8 → **1/5.7** |

敏感度（同账本换档）：N_j=14（已发布权重口径）87.4 / 3 视图 102.7 / 4 视图（RoboTwin）131.8 / RoboTwin 全口径（4 视图+N_j=14+L=32）149.8 / RGB-only（去 depth 分支）68.5 / 去噪 4 步 59.7 / 真机分辨率 384×256 85.3 GMAC。

## 4. 16×108 阵列映射账（定稿）

账本脚本：`profile/holobrain_spec.py` + `holobrain_hw.py`（复用 SwiftVLA 线的共用 account 流水线，主会话已重跑验收，数字与核验后口径一致）。SwiftVLA/Evo-1 对照行与其各自 hw.json 逐位一致。逐条目/分阶段拆解与优化杠杆账另见 `profile/holobrain_breakdown.py`（14 道验收门全过 + 独立核验 9/9 确认），图文汇报页 `../round_report_hb_prof/hb_profiling.html`——核心读数：喂料墙 54.21 M 拍占 CONC 99.8%（W 49.32 M 拍 90.8% 被遮蔽，BW128/SM32 零收益）；软件杠杆 BERT 缓存+K/V hoist 组合 −8.7%；216 列 BW128 −35.8% 但 packed PE 已证 LUT 不可行；W 流有约 13 MB/chunk 的窄 n 整组装载浪费可由固件白捡。

### 4.1 时延主表（LIBERO 主档 73.7 GMAC）

| 配置 | M 拍 | s@198.5MHz | s@250MHz | 阵列利用率 |
|---|---|---|---|---|
| **108 列 SM16 BW64（现行硅片）LWreal** | 89 | **0.45** | 0.36 | 48% |
| **108 列 SM16 BW64（现行硅片）CONC** | 54 | **0.27** | 0.22 | 78% |
| 216 列 SM16 BW128 CONC（R3 packed 档） | 35 | 0.18 | 0.14 | 61% |

实时线：chunk=64 → 30 Hz 预算 2.13 s、50 Hz 预算 1.28 s。**108 列现行硅片两档口径都过线**：CONC 0.27 s 对 30 Hz 裕量 7.8×、对 50 Hz 4.7×；保守 LWreal 0.45 s 也有 4.7×/2.8×。即使 RoboTwin 全口径（149.8 GMAC）也只有 0.57 s@108 列，照样过 30 Hz。**HB-GD 在现行阵列上不是算力问题。**

### 4.2 谁是墙（108 列 CONC，M 拍）

喂料 feed 54.2 > W 流 49.3 > softmax÷16 18.6 > COPY 8.6 > elem÷16 7.9 > drain52 0.4 —— **墙=喂料下限**（16 行 m-tile 串行），和 SwiftVLA 不同（那边第一墙是 expert W 流重装）。216 列 +128b 后 W 降到 27.4M，feed 34.7M 成唯一墙；利用率掉到 61%，加宽收益有限（0.27→0.18 s）——**不需要 216 列，108 列就是答案**。

### 4.3 权重流与缓存

- W 流 334 MB/chunk（INT8 按次装载，忠实上游无缓存口径）：BERT 编码器 85 MB 是最大单项（嵌入表 23.8M 另驻 DDR）——指令不变可整条缓存掉；解码器 15.5 MB/步 × 10 步 = 155 MB；视觉/增强器权重每视角重装。
- K/V + 距离嵌入 hoist（数学等价的软件优化）后：67.4 GMAC / 312 MB，省 7-8%。
- 对照：SwiftVLA 1.41 GB、Evo-1 448 档 4.26 GB。

### 4.4 同账本横向对照（108 列 / 64b / CONC）

| 负载 | GMAC | W 流 | 时延 |
|---|---|---|---|
| **HB-GD LIBERO 主档** | **73.7** | 0.33 GB | **0.27 s** |
| HB-GD RoboTwin 全口径 | 149.8 | 0.41 GB | 0.57 s |
| SwiftVLA 部署 3 视图 | 417.8 | 1.41 GB | 1.40 s |
| Evo-1 C1 224/N10 | 443.3 | 1.90 GB | 1.49 s |
| Evo-1 448/N32（v2 终点） | 1096.1 | 4.26 GB | 3.65 s |

### 4.5 账本没算什么（诚实边界）

非 GEMM 算子 11 项未计入 0.27 s（量级与归宿逐项列在 `holobrain_hw.json` 的 not_modeled）：deformable 采样加权和（约 84 MMAC/2 视角 + 335M 元素操作）、PSE 4×4 反投影（217,600 次/视角，相机固定可离线缓存）、深度分布加权聚合（M=1 病态，16 行阵列利用率只有 1/16）、temp_joint 6D einsum、UpsampleHead 插值、FK/DPMSolver 步进、RoPE/norm/gate（1.24 G-ops 粗账）、Swin 窗口切分 gather-scatter、BERT 查表、深度解码 resize。**实际延迟会比 0.27 s 高一些**——deformable 采样是每视角 6 层的核心算子，向量化效率未知，且这部分相对 SwiftVLA 占比更高。CPU 侧（tokenizer、图像/深度解码、Flask 服务器）也不在账内。

## 5. FPGA 部署可能性评估

### 5.1 算子分类：哪些落阵列、哪些落不上去

**能落阵列（纯 GEMM/注意力，占 MAC ≥95%）**：
- Swin-T 的全部投影和 MLP（window attention 的 QKV/proj 是规则 GEMM，注意力分数可走 SM16——和 SwiftVLA 的 ViT softmax 同路数）
- BERT 全部（且冻结 → INT8 离线校准容易，还能按指令缓存）
- 增强器里的投影/FFN、PSE 的 fusion_fc、动作专家全部
- 10 步去噪就是同一组 GEMM 跑 10 遍，权重每次重装（WRAM 442KB 装不下解码器的 15.5 MB/步，和 SwiftVLA 同样格局但轻 6 倍——它 expert 是 98 MB/步）

**落不上去（数据相关寻址/逐元素，占 MAC <5% 但工程量不小）**：
| 算子 | 量级（每视角每次） | 建议归宿 |
|---|---|---|
| deformable attention 采样与加权（增强器 6 层） | 每层 1,700 查询点 × 4 尺度 × 采样点数 | CPU/软逻辑，或改结构（换普通 cross-attn 需重训） |
| PSE 3D 反投影（内外参矩阵乘 + 4×4 求解） | 1700 点 × 小矩阵；相机固定可离线缓存成查表 | 预处理离线做掉 |
| PSE 深度分布 softmax（128 bin）+ 加权聚合 | 1700 × 128 softmax + 向量加权 | 软逻辑/SM16 复用 |
| 关节图注意力 mask、rotary 位置编码、gating、上采样插值 | 逐元素，量级 0.1M 级 | 软逻辑 |

**判断**：MAC 上完全装得下（主档 73.7 GMAC 对 108 列是小活，账本实测 CONC 0.27 s），难点是喂料编排——比 SwiftVLA 的"纯 transformer 流水"碎得多。Swin window 划分、deformable 采样索引、深度分布三样各自要一条预处理通路。

### 5.2 权重流与内存

- 全模型 INT8 ≈ 184MB，其中 BERT 编码器 85MB 冻结可按指令缓存（LIBERO 每任务一条指令，摊薄近零；23.8M 的词嵌入表驻 DDR 查表，不占 W 流）。
- 每 chunk 实际 W 流 **334MB**（账本忠实口径：视觉/增强器权重每视角重装、BERT 一次、解码器 15.5MB×10 步）。SwiftVLA 是 1.41GB → **1/4.2**。
- 激活：2 视角 3,400 token × 256 维进增强器，CTX 2MB 装得下（SwiftVLA 3 视图 KV 都能忍，这里更小）。

### 5.3 量化风险（从 SwiftVLA W8A8 线外推）

- 权重 INT8 大概率无害（SwiftVLA w8a16 只掉 2pp；BERT 冻结更好校准）。
- **激活 per-token INT8 是已知风险**（SwiftVLA 掉 12pp）。HoloBrain 的 deformable 采样和深度分布对数值精度是否更敏感——没实测过，是量化线的第一个实验。
- 深度输入本身是 16bit 量纲（米制原值进网络，无归一化）——深度通路的量化方案要单独设计。

### 5.4 总评

**可行性：高（算力维度），中（工程维度）。**

- 算力：主档 73.7 GMAC/chunk（最重档 RoboTwin 149.8）对 108 列 @198.5MHz 宽裕——CONC 口径 0.27~0.57 s，全部过 30 Hz 线（预算 2.13 s）。墙是喂料下限不是算力。
- 带宽：334MB W 流/chunk，比 SwiftVLA 轻 4 倍，现行 BW64 就够；216 列+BW128 只再省 0.09 s，不值得。
- 工程量：算子杂（五类非阵列算子）+ 深度输入通路（LIBERO 之外的部署要配深度源）+ SimpleRTC 异步逻辑（软逻辑级，量小）。
- 与硬件现状完全兼容：不需要 216 列、不需要 packed PE、不需要动 DMA。

## 6. 与 SwiftVLA / Evo-1 同口径对比

| 维度 | Evo-1 | SwiftVLA | HB-GD |
|---|---|---|---|
| 参数 | 1.0B | 0.45B | 0.18B |
| MAC/chunk | 1,096 GMAC（C1 档 443） | 417.8 GMAC | **73.7 GMAC**（RoboTwin 全口径 149.8） |
| W 流/chunk | 4.26GB | 1.41GB | **334MB** |
| 108 列时延（CONC） | 3.65s（C1 1.49s） | 1.40s | **0.27s**（0.57s RoboTwin 档） |
| LIBERO 成绩 | 93.5%（自复现） | 94.7%（论文，代码未发布） | 96.7%（论文，**含 GT 深度**） |
| 代码/权重 | 有 | **无** | **有**（LIBERO 除外） |
| 阵列外算子 | 少（纯 transformer） | 少（纯 transformer + flow） | **多**（deformable/3D/深度分布/图注意力） |
| 第一瓶颈 | expert W 流 | expert W 流重装 | 喂料下限 |

趋势很清楚：三代模型一路变小变杂。HB-GD 用"深度传感器 + 3D 先验"换掉了大 LLM——对 FPGA 是好事（乘加少、W 流轻），代价是流水线工程量。

## 7. 风险与建议

**风险**：
1. LIBERO 权重/配置未公开——想验证 96.7 只能等，或拿 RoboTwin 后训权重先跑通全流程。
2. 深度依赖：对标数字建立在完美 GT 深度上；换真实深度源（或去掉深度分支重训）成绩未知。
3. 激活量化敏感性未实测（deformable 采样 + 深度分布可能比纯 transformer 更脆）。
4. 本报告的 MAC 账是纸面推导（结构维度全部经代码核实），没跑过模型实测延迟——上 V100 实测前，时延数字以账本口径为准。

**建议下一步**（按性价比排序）：
1. 账本已定稿（§4）：108 列现行硅片 0.27 s/chunk，30 Hz 裕量 7.8 倍，墙=喂料下限——硬件侧不需要任何改动，前期工作全在非 GEMM 算子的归宿设计。
2. 用 RoboTwin 后训权重在 V100 跑通推理 + 延迟拆分（复用 SwiftVLA 线的 harness 思路），顺带实测"喂全零深度"到底掉多少——这直接回答"深度分支值多少钱"。
3. 量化线第一个实验：W8A8 全模型 fake-quant + deformable attention 数值敏感性；深度输入米制定标可参考 bin 宽 9.4mm（0.01–1.2m÷128）。
4. 若决定立项做 FPGA 原型：第一刀砍向喂料编排（Swin window 重排 + deformable 索引生成），不是乘加阵列。

---
*附：调研过程产物——`_research_phase1.json`（四路代码调研结构化结果）、`_workflow_final.json`（Verify 逐条判定 + R5 账本结构化结论）、`profile/`（spec/hw 账本脚本与 JSON，重跑可复现本报告全部数字）。*
