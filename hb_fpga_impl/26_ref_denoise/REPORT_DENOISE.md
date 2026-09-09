# REPORT_DENOISE — 去噪迭代结构拆解：步内量化误差 vs 跨步反馈放大

生成：2026-09-07 00:48（本地；服务器实验完成于 2026-09-07 00:48，<SERVER>）
服务器产物目录：/tmp/alg_denoise/（results_final_summary.json 为总入口）。
**主会话已复核**：门 a、free、prefix/loop、步隔离 10 档、TF/TF+、stateq、drfix（修正版 0.0306/0.0232）、k20c0 s001=0.19398 均从 json 重读一致。注意 results_denoise.json 里 attribution/dr_floor=0.157 是首版单位 bug 的作废值，修正版在 results_drfix.json。

## 一句话结论

**去噪迭代维度是收缩的，不是放大的。"815 层串行复利"在结构和机制上都死了。**

- 结构：815 = 275 个量化边界 × 调用次数（215 个一次性前缀 + 每步 60 × 10 步）；有效串行深度 = 单遍 275，不是 815。
- 机制：浮点模块级全量量化自由迭代（free）jpos 只有 0.01969；教师强制 TF = 0.00889 绿；步 0–7 单独量化贡献 ≈ 3e-05 量级；每步传导比全程 < 0.5（回灌先衰减一半以上）。
- 整数链 0.2383 与浮点执行 0.0197 差 12 倍，这 12 倍不属于"W8A8 深度"、不属于"迭代反馈"，属于整数链自己的执行机制（attention 内部整数 softmax 为头号候选，见 REPORT_REFQ 账本与本目录 attention 拆解）。

## 解剖（A 线）

### 去噪循环本体

- **步数 10**，timesteps [999, 899, …, 100]。
- 调度器 **DPMSolverMultistepScheduler**（diffusers 0.35.2）：algorithm_type=dpmsolver++，solver_order=2（跨步保存 2 个 model_outputs），prediction_type="sample"（直接预测 x0），lower_order_final=True。配置出处 model.config.json /decoder/base_cfg/test_noise_scheduler。
- **步间状态 noisy_action [1,64,14,8]**：dim0（归一化关节角）是调度器唯一反馈通道，也是判据通道；dim1–6（6D 姿态）每步由 recompute() 正运动学从 dim0 重建（宿主 fp，不量化）；dim7 夹爪。**迭代反馈只走 dim0 一条窄通道。**

### 815 的正确分解账

815 = **215 一次性前缀**（161 gemm + 54 attn 事件，每模块恰调一次）+ **每步 60 × 10 步**（42 gemm + 18 attn 事件，10 步集合完全同构）。

| 前缀组 | 调用数 (gemm/attn) |
|---|---|
| backbone (Swin 2D) | 40 (28/12) |
| neck | 3 (3/0) |
| backbone_3d (Swin 3D) | 40 (28/12) |
| text_encoder (BERT 12 层) | 36 (24/12) |
| text_feat_map | 1 (1/0) |
| feature_enhancer (6 img + 6 text 块) | 60 (48/12) |
| spatial_enhancer | 6 (6/0) |
| robot_encoder | 25 (21/4) |

每步 60 = t_embed 2 + input_layers 5 + 6 块 × 8（adaLN 1 + temp_joint_attn 1 + joint_pos_encoder 2 + img_cross_attn 1 + text_cross_attn 1 + ffn 2）+ head 5。**head 每步执行一次**，末步 output_layers.4 = 全链最后调用 #814。

### "decoder.layers 56 层走两遍"证伪

实测 66 槽位（6 块 × 11；上一轮记 65 也是误数）= 30 带权（t_norm/temp_joint_attn/img_cross_attn/text_cross_attn/ffn）+ 18 RMSNorm + **18 个 None 占位**（gate_msa/scale_shift/gate_mlp 融合实现）。每层每步恰执行一次。
口径对账：标定表 438 键 = 324 循环外 + 114/步；部署 275 边界 = 215+60；循环内 72 个 q/k/v/proj 线性层被融合成 18 个 attn 事件；前缀约 55 个表键在部署走宿主 fp。

## 验证门

- 门 a（fp 直通）：jpos = 6.99e-08（PASS；二次重跑同值，协议确定）。
- 门 b（free 全量量化）：0.01969，比整数基线 0.2383 低 92%。偏差是信息不是 bug：浮点执行 + fp bias + attention 核 fp（且比部署更重：438 键全挂）。与 /tmp/alg_refq 的独立实现 V1=0.0208 同量级，两套 harness 互证。

## 四实验数字表（jpos，vs fp32_ref_000）

| 配置 | jpos MAE | 相对 free |
|---|---|---|
| fp 直通 | 6.99e-08 | — |
| **free 全量化自由迭代** | **0.019688** | 1.00× |
| 只量化一次性前缀（循环 fp） | 0.015243 | 0.77× |
| 只量化循环（前缀 fp） | 0.009018 | 0.46× |
| 只量化第 0 步 ~ 第 7 步（逐个） | 1.6e-05 – 4.1e-05 | ≈0 |
| 只量化第 8 步 | 0.000845 | 0.04× |
| **只量化第 9 步（末步）** | **0.008890** | 0.45× |
| **教师强制 TF** | **0.008890** | 0.45× |
| TF+（再恢复调度器 model_outputs 历史） | 0.008890 | 0.45× |
| 仅步间状态过 int8（absmax/127，模块全 fp） | 4.69e-05 | 0.002× |
| 定点 requant（floor，修正刻度） | 0.030643 | 1.56× |
| 定点 requant（round，修正刻度） | 0.023196 | 1.18× |
| 部署整数链全量（对照） | 0.2383 | 12.1× |

TF = TF+ = step9-only **三个数逐位一致**：调度器历史（model_outputs）一点额外误差都不带，反馈通道里唯一有效的就是状态本身。

首版定点 requant 0.15726 已作废（把 m/2^s 直接乘在反量化累加值上、缺 1/(sa·swc) 因子——正是 0904 ab 轮那个单位 bug 的复刻，独立复现了历史 bug 也算交叉验证）。修正后 rq_scale = m/(2^s·sa·swc) ≈ 1/so，逼近误差实测 3.03e-05（前 80 键最大值）。

## 逐 step 传导比（free vs fp 缓存逐步对拍）

| step | 状态入口 dim0 相对偏差 | 预测 dim0 相对偏差 | 传导比 |
|---|---|---|---|
| 0 | 0.00000 | 0.0863 | — |
| 1 | 0.00346 | 0.0848 | 0.040 |
| 2 | 0.00674 | 0.0851 | 0.080 |
| 3 | 0.01005 | 0.0849 | 0.118 |
| 4 | 0.01355 | 0.0827 | 0.160 |
| 5 | 0.01709 | 0.0849 | 0.207 |
| 6 | 0.02142 | 0.0890 | 0.252 |
| 7 | 0.02695 | 0.0828 | 0.303 |
| 8 | 0.03166 | 0.0884 | 0.383 |
| 9 | 0.04017 | 0.0887 | 0.455 |

读法：状态偏差线性加性（每步约 +0.004，复利放大应是指数——没有）；预测偏差平坦（0.083–0.089，每步误差主体是步内新产生的）；传导比全程 < 0.5（回灌先衰减一半以上）。步 0 入口条件偏差（一次性前缀造成）：图像特征 21.5%、文本 24.8%、机器人状态 12.0%（相对 L2）。

## 整数链 bisect 重读（佐证）

N=250 → 0.1335；N=750 → 0.1321。**第 2–9 步共 500 个量化调用加进去，误差零增长**——整数链自己也没有逐步放大；误差在前缀+首步就饱和。（N≥810 尾段有缺输入回退告警，数值不稳，只看趋势。）

## C 线：so×2 跨样本泛化核验

| 样本 | pcw_v3 基线 | k20c0 (so×2) | 变化 |
|---|---|---|---|
| s000 | 0.2383 | 0.2059 | −13.6% |
| s001 | 0.1828 | **0.19398** | **+6.1%（反噬）** |

**不泛化。** 样本特定杠杆（s000 的头部输出坍缩模式在 s001 上没有出现；s001 误差集中在关节 4/6/13 = 0.52/0.50/0.59，同一批腕/爪通道）。除非逐样本自适应，放弃作为通用优化项——0.2059 不并入基线（与用户 0906 裁决一致）。

## 判定与机理

1. **迭代维度收缩不放大**，三条独立证据：传导比全程 < 0.5 且状态偏差线性加性；步隔离步 0–7 ≈ 0、离末步每远一步衰减约 10 倍；整数链 bisect 第 2–9 步零增量。
2. **free=0.01969 构成**：前缀条件偏差 ≈77%（0.01524），末步出口 ≈45%（0.00889），近似可加小幅抵消（0.0152+0.0090=0.0242 vs 实测 0.0197）。
3. **整数链 0.2383 的主体不在"W8A8 + 迭代"**：同刻度浮点执行全量 0.01969，换修正刻度定点 requant 也只有 0.03064，与 0.2383 还差 7.8 倍。剩余候选：int8 bias 增广（refq 实测 e2e 只 +0.003）、**attention 核 σ_S/exp/P 整数化（头号，refq 三档拆解中）**、段间激活单次量化摆位。
4. **状态存储无关紧要**：只让状态过 int8 = 4.69e-05（free 的 0.24%）。硬件把去噪状态存 int8 几乎不掉精度——16 位预算不必花在状态上。

## 复现命令

```
PY=~/.conda/envs/holobrain/bin/python   # ssh <SERVER>，模型 ~/workspace/holobrain
# A 线解剖（秒级）
$PY /tmp/alg_denoise/ana_prefix.py; $PY /tmp/alg_denoise/layer_inv.py
$PY /tmp/alg_denoise/diff_steps.py; $PY /tmp/alg_denoise/cnt_layers.py
# B 线四实验（约 70s；量化数学独立实现）
cd ~/workspace/holobrain && nice -n 19 $PY /tmp/alg_denoise/exp_denoise.py --exp all
nice -n 19 $PY /tmp/alg_denoise/exp_denoise.py --exp drfix
# C 线 so×2 泛化（编译 ~2min + forward ~518s）
cp /tmp/alg_head/hw_calib_table_k20c0.json /tmp/alg_denoise/
$PY /tmp/ae_hostdrv/compiler.py --trace /tmp/ae_hostdrv/trace_s001.json \
    --manifest /tmp/pcw_rtn/manifest.json --w8 /tmp/pcw_rtn/w8_full \
    --calib /tmp/alg_denoise/hw_calib_table_k20c0.json --out /tmp/alg_denoise/build_s001_k20c0
cd ~/workspace/holobrain && nice -n 19 $PY /tmp/ae_hostdrv/host_driver.py \
    --build /tmp/alg_denoise/build_s001_k20c0 --trace /tmp/ae_hostdrv/trace_s001.json \
    --batch /tmp/ae_hostdrv/batch_s001.pt --ref /tmp/ae_hostdrv/fp32_ref_001.npz \
    --out /tmp/alg_denoise/result_001_k20c0.npz --sample-id 001
```

## 文件清单（/tmp/alg_denoise/）

results_final_summary.json（总入口）、results_denoise.json、results_drfix.json、result_001_k20c0_vs_fp32.json、deploy_calls_815.json（#814 已补齐）、exp_denoise.py、ana_prefix.py / layer_inv.py / diff_steps.py / cnt_layers.py、run2.log / drfix.log / c_build.log / c_run.log / smoke.log / run1.log、hw_calib_table_k20c0.json、build_s001_k20c0/、result_001_k20c0.npz。

时间线（服务器）：解剖账本 23:52:34 → 冒烟 00:09:42 → run1（步计数 bug）00:16:39 → run2 四实验 00:25:47–00:26:53 → C 线 00:29:38 → drfix 修正复跑 00:32:44 → 槽位核实 00:38:42 → 报告 00:48:00。
