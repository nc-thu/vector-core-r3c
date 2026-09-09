# 24_pcw_rtn 阶段 4 报告：真实样本门禁 + 两次翻案

生成时间：2026-09-04 11:35:00（北京时间）
作者：主会话（阶段 4：门禁修正）
任务：把 W8A8 门禁的评估集从合成 bringup 批换成真实 RoboTwin 样本（s000/s001），判据 jpos ≤ 0.045 rad。
脚本：`24_pcw_rtn/gate_real.py`（服务器 /tmp/pcw_rtn/，协议 = hw_calib mode B_v1，仅换评估集）

一句话结论：**模块边界 per-tensor W8A8 在部署 bias 约定下于真实样本是绿的（s000=0.0217 / s001=0.0244）**；上一轮"per-tensor 权重量化是 0.29 红灯主因"的结论来自 ab 实验的单位 bug，**撤回**；全深度 0.29 的真实构成是 GEMM 量化在 815 次调用链上的逐层复利 + 合成标定对真实输入的量程失配（首层 80% 削顶），不是量化粒度。

---

## 1. 最终数字

| 测量 | s000 | s001 | 判定（判据 0.045） |
|---|---|---|---|
| acc_bias：bias 进累加器（理想口径，硬件做不出） | 0.01855 | 0.02523 | 绿（下界参考） |
| **host_bias：compiler.py 部署约定（判据口径）** | **0.02167** | **0.02437** | **绿** |
| （对照）部署全深度 fast_interp | 0.2993 | 0.2108 | 红 |
| （对照）部署 only_gemm（非 GEMM op 全 fp） | 0.2942 | — | 红 |
| （撤回）ab V0_deployed | ~~0.18937~~ | — | 无效 |
| （撤回）ab V2_rn_pcW "−75%" | ~~0.04715~~ | — | 基线无效，见 §3 |

协议：4 种子（1000/1100/1200/20260830）均值；fp 参考 = 同批同种子现场重算（已验证与 fp32_ref_000.npz **逐位一致**，jpos=0.00000，排除参考漂移）；CPU 口径；标定 = 合成扰动 8 样本（与 hw_calib 完全同款）。层覆盖：420 Linear + 10 Conv = 430 模块。

## 2. 调查过程（两次翻案）

### 2.1 第一次翻案（我的 bug，自查自纠）

初版 gate_real 从 v2 标定表重建 params 时，88 个 bias_fp_fallback 层的 w_acc 被置 None（bias 整个丢失）→ 假红灯 0.2415；第二版塞了原始 bias（单位错，应为累加器域 b/(sa·sw)）→ 0.1849。期间我曾宣布"v2 表是独立误差源（13× 翻转）"——**该结论撤回**。修复后逐字段核对：v2 表与今日 fresh 标定的 sa/so/sw/m **全部精确一致**（430 层比值 1.000，同一确定性流程产物），表本身没有问题。教训：从表重建数值路径时，bias 的单位域（累加器域 vs 输出域）是最容易错的地方。

### 2.2 第二次翻案（ab 实验的单位 bug，影响上一轮根因结论）

修好 bias 后双变体都绿（0.019/0.022），与 ab V0 的 0.189 矛盾。排除法收口：参考一致（0.00000）、表值一致、覆盖一致（ab 也不量化 in_proj）、seed 噪声 ±0.006。最后单层实测对拍暴露根因：

**ab_quant.py（服务器 /tmp/w8a8_ab/ab_quant.py，与本地 analysis_02_server_ab.py 同版）的 pt 权重路径**：
`get_w` 返回 `qz(Wf,sw)*sw`（反量化回真实值的权重）→ `acc = xq @ wqᵀ` 带上了 sw 单位（≈y/sa）；但 requant 的 `r=(sa*sw)/so` 是按纯整数 acc 写的。结果：**矩阵项被压 sw 倍、bias 项正确** → 每个层输出≈"只剩 bias"。`make_conv_fwd` 同款（`wq = qz(Wf,sw)*sw` 后进 conv），**10 个 conv 输出全部缩 sw 倍 ≈ 视觉流死亡**。

单层证据（backbone.stages.2.blocks.3.ffn.layers.0.0，表值 sa/sw/so 同源）：ab 输出 rel_vs_fp=1.04（bias 垃圾），我的实现贴近 fp（前 3 元素 1.338/-4.225/-5.282 vs fp 1.316/-4.212/-5.267）。

由此解释上一轮报告里两处反常：
- "V2_rn 动作 0.047 时视觉 sentinel 全程 0.8~1.0"——不是"LayerNorm 吸收"，是 conv 路径坏了，动作靠状态/DPM 先验撑着；
- "激活侧改动全部无效（0.189~0.196 钉死）"——所有激活变体都跑在坏基线上，钉死在"常数传播"水平，**激活/标定故事从未被真正检验**。

pc 线性路径（V2c/V2_rn 的权重部分）单位正确，0.059/0.047 作为"pc 线性 + 坏 conv"的混合测量仍有效，但"pcW 提升 69%/75%"的对比无效（分子是真量子化，分母是 bug）。

### 2.3 与部署链 diag 的交叉验证

- only_gemm（只量化 GEMM、其余 fp）：jpos 0.2942 ≈ 全链 0.2975 → 非 GEMM op 只贡献 ~0.003；
- 部署首层（clean 输入）rel=0.175；我的语义复算同层 max_rel=0.25 / mae_rel=0.16（同量级，门禁忠实）；
- **首层 in_sat=0.802：patch_embed 的输入 80% 超出合成标定量程被削顶**；注意力层 GEMM（输入在量程内）max_rel 仅 2.0~2.4%。coverage：105~115/430 模块真实输入超 calib×1.05。

## 3. 新的根因陈述（替代 research_w8a8_error/REPORT.md §2 结论）

0.29 红灯的构成（按证据强度）：
1. **GEMM 量化在深度链上的复利**（主项）：单 GEMM 在量程内只有 ~2% 误差，但部署链每个 op 边界都以静态 scale 重新量化到 int8，815 次调用逐层锁死/放大漂移（only_gemm 已复现全部 0.294）。模块间 fp 复位（我的边界门禁）时同样 430 个 GEMM 只留 0.022——13.8× 的差全部来自"没有 fp 复位"。
2. **合成标定对真实输入失配**（注入项）：首层 80% 削顶是实锤；上一轮"标定/激活侧无效"的否证全部无效（坏基线）。
3. per-GEMM 量化粒度（per-tensor 权重、floor/RTN）：量程内 ~2%/层，**不是主因**——pcW+RTN 的硬件动机需要重估（本轮 RTL 已落地且验证，基础设施不浪费，但预期收益的参照系变了）。

## 4. 对本轮三条线的影响

- RTL 线（已完成）：改动正确、验证完备（17 项全绿、floor 逐字节锚、RTN 双向位精确），保留；但它修的 requant 粒度在全深度账本里只占小头。
- 软件线（进行中）：全深度 pcW 实测仍是关键实验（现在测的是"深度代价"本身）；深度二分（前 N op 量化扫描）价值上升。
- 门禁：阶段 4 交付完成；下一版应考虑（a）真实样本标定通道（注意 train/test 分离）、（b）in_proj 覆盖（部署表含 in_proj_weight 条目，门禁/hw_calib 目前留 fp，覆盖口径有差）。

## 5. 复现命令

```bash
# 双变体门禁（约 160 s，服务器 CPU）
ssh <SERVER>
cd ~/workspace/holobrain && CUDA_VISIBLE_DEVICES= \
  ~/.conda/envs/holobrain/bin/python /tmp/pcw_rtn/gate_real.py
# 结果 /tmp/pcw_rtn/gate_real_results.json（本地已回传 24_pcw_rtn/results/）

# 单层对拍（ab pt 路径 vs 我的语义 vs fp）：见本报告 §2.2，脚本为一次性
# heredoc，参数：backbone.stages.2.blocks.3.ffn.layers.0.0

# ab 结果复核
python3 -c "import json; d=json.load(open('/tmp/w8a8_ab/ab_result.json')); \
  print(d['V0_rn_ptW']['metrics'], d['V2c_pcW']['metrics'])"
```

本地产物：`24_pcw_rtn/gate_real.py`、`results/gate_real_results.json`、`results/gate_real_full3.log`。

## 6. 诚实边界

1. 本门禁是**模块边界**测量（模块间 fp），不能替代全深度判定；0.0217 的含义是"per-GEMM 语义干净"，不是"部署会绿"。
2. 标定仍是合成扰动批（沿上一轮口径，本轮只换评估集）；in_sat=0.802 说明该标定对真实输入失配，acc_bias/host_bias 两变体都带着这个失配——0.022 是"失配标定下"的绿。
3. in_proj/text_feat_map/ConvTranspose 在门禁中保持 fp（部署表含这些条目）；部署覆盖更广，门禁口径偏乐观。
4. 种子数 4；ab 的对比数字是单种子（20260830），seed 噪声 ±0.006，不影响本报告的定性结论。
5. 0.29 的分解（§3）是基于边界/深度/only_gemm 三点证据的推断，逐 op 的完整记账要等软件线的深度二分实验。
