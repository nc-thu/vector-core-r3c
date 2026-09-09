# REPORT_REFQ — 独立假量化参照：全深度误差的主体是整数执行细节，不是 8 位格式下限

生成：2026-09-07 00:46（本地；服务器实验完成于 2026-09-07 00:31，<SERVER>，全程 CPU，nice -n 19）
服务器工作目录：/tmp/alg_refq/（结果 json 在 /tmp/alg_refq/results/）。
**主会话已逐位复核**：V1/V1b/V2/V3/V4/V5/V1_nomask 逐种子 jpos、门 a 6.99e-08、整数基线 0.23834/0.18275 均从 json 重读，与代理报告一致。

## 一句话结论

量化位置和 scale 与部署逐层相同、只把舍入和乘加换成浮点（fp64）实现后：

- s000：jpos MAE 从 0.23834 降到 **0.02080**（误差降低 91.3%）
- s001：从 0.18275 降到 **0.02948**（降低 83.9%，双种子 0.0220/0.0295）

**8 位格式本身的下限在 0.02~0.03——低于 0.045 判据。当前全深度误差里 85~90% 来自整数执行细节。**
按用户判据树走 ③≪④ 分支：预算应投向定位并修掉整数执行侧的误差源，而不是换量化格式/位置或上训练补偿。

这一档（同位置同 scale、浮点执行、独立实现不复用 requant 代码）此前从未跑过——用户 0906 评审点名要的正是它。

## 主表：各变体 jpos MAE（rad，越小越好）

整数链基线读自 /tmp/pcw_rtn/result_000|001_pcwv3_vs_fp32.json。

| 变体 | s000 部署种子 | s000 种子1000 | s001 部署种子 | s001 种子1000 | vs 整数基线 |
|---|---|---|---|---|---|
| 整数链（部署实测） | 0.23834 | — | 0.18275 | — | 基线 |
| **V1 W8A8 假量化 RTN（主角，即 R2 参照）** | **0.02080** | 0.02091 | **0.02948** | 0.02196 | s000 −91.3% / s001 −83.9% |
| V1b bias 按部署整数增广 | 0.02345 | — | 0.02959 | — | 比 V1 差 +0.003/+0.000 |
| V2 输入+requant 改 floor | 0.06201 | — | — | — | V1 的 3.0 倍（舍入规则重要，RTN 选对了） |
| V3 只量化权重（激活 fp） | 0.00853 | 0.00885 | 0.00842 | 0.00703 | 权重量化几乎免费 |
| V4 只量化激活（权重 fp） | 0.01615 | — | — | — | 激活一份 ≈ 权重的 1.9 倍 |
| V5 激活 int16 网格（同量程） | 0.01686 | — | — | — | 比 V1 好 18.9% |
| V1_nomask 丢 BERT mask | 0.01939 | — | 0.02760 | — | 与 V1 差 <0.002（噪声级，排除） |

两种子下数字稳定（波动 ±0.004），结论不依赖单种子。
与 /tmp/alg_denoise 的独立第二实现（free=0.01969）同量级——两套独立写的假量化 harness 互相印证。

## 误差账本：s000 的 0.23834 怎么拆

| 项 | 份额（jpos） | 怎么测的 |
|---|---|---|
| 8 位格式下限（量化位置/scale/网格） | 0.0208 | V1（同位置同 scale，浮点舍入乘加） |
| bias 整数增广列 | +0.0027（s001 +0.0001） | V1b − V1（e2e 两次差） |
| BERT attention_mask 丢失 | ≤0.002，噪声级 | V1_nomask − V1；text 序列 90.9% 是 pad 也就这个量级 |
| conv 逐 tensor scale | 单层 rel 1.3e-4，小 | 门 c conv 模块 |
| 15-bit requant 乘子 + 舍入 | 单层 rel ~6e-4 ×437 层 | 门 c V1b-vs-oracle 逐层实测；全链定界见 alg_denoise 的 drfix（0.0306/0.0232） |
| **剩余未定界 ≈0.19** | 0.2383 − 0.0208 − 0.003 − 0.002 | 指向 attention 内部整数执行（S 压 int8、整数 exp 表、P 压 1/127 网格），三档拆解见本目录 REPORT_ATTENTION.md（若已生成） |

嫌疑排序（写明哪些是推测）：

1. **attention 内部整数执行（头号嫌疑，本轮三档拆解中）**：部署把打分 S 压 int8、softmax 用整数 exp 表、概率 P 量化到 1/127 网格，backbone MSA / img·text attention / decoder cross·temporal 每条 attention 路径都这么走。假量化把这部分当精确 fp，是两侧最大的语义缺口。
2. 15-bit 乘子 + 逐层 requant 复利：单层 ~6e-4（门 c 实测），随机游走口径 sqrt(437)×6e-4≈1.3e-2，不足以单独解释 10 倍差距——且 alg_denoise 的 drfix 全链实测也只有 0.0306。
3. bias 增广、conv 逐 tensor：已实测，小（上表）。
4. 部署 4 个模块调用实例回退原生 fp：方向相反（让部署更准），只会低估差距。

机制注记：门 c 里 aug 层的 V1（精确 fp bias）对 oracle 单层 rel_l2 高达 0.22（qkv），但 e2e 只差 0.003——bias 对每行加同一常数，q·k 打分整体平移，softmax 后基本不变。V1b 还原整数增广语义后单层 rel_l2 掉到 6e-4。

## 三个验证门实测

**门 a：fp 直通 = 参照协议。PASS。** s000 jpos=6.99e-08、s001=6.81e-08（判 <1e-5）。非严格 0：npz 参照在 V100 GPU 生成，CPU eager 复算有 ~7e-8 跨设备浮点噪声，比指标小 6 个量级。

**门 b：权重逐位一致。PASS。** 5 个 tensor（qkv/MHA in_proj/text_feat_map/decoder.t_embed.mlp.2/v_proj），4 个与 pcw_export 部署 int8 二进制完全一致；in_proj 1/196,608 元素差 1 LSB（fp32 vs fp64 在 .5 边界分歧）。swc 与 absmax/127 相对差 5.6~5.9e-8（fp32 存储截断）。

**门 c：同输入 fake-quant vs 整数链真段（fast_interp=RTL 逐位）。PASS 5/5。**

| 模块 | bias 语义 | 自建整数仿真 vs oracle | V1 rel_l2 | V1b rel_l2 |
|---|---|---|---|---|
| backbone qkv（早期 aug） | 整数增广 | 17/6,773,760 差 1 LSB（半进位边界） | 2.24e-01 | **5.89e-04** |
| v_proj（中期 aug） | 整数增广 | 逐位一致 | 1.05e-01 | **6.86e-04** |
| decoder input_fc.5（晚期 aug） | 整数增广 | 逐位一致 | 1.59e-03 | **6.51e-04** |
| ffn.layers.0.0（host fp bias） | requant 后 host 加 | 逐位一致 | 5.94e-04 | 5.94e-04 |
| patch_embed conv（逐 tensor） | — | 逐位一致 | 1.29e-04 | 1.29e-04 |

数学自检：round(acc/so) ≡ 逐列 round(acc·sa·swc_j/so)，qkv 全部 6,773,760 输出逐位一致。

## 与整数链的语义差异清单（fake 侧有意为之，逐条记账）

1. requant 乘子：部署 15-bit m_j/2^s_j 编码；fake 直接除 so。单层 rel ~6e-4。
2. bias：部署 aug 层整数增广列（134 条目存在列饱和）；V1 精确 fp，V1b 还原部署语义（e2e +0.003）。
3. 舍入：部署 RTN 半进位；fake torch.round（仅 .5 边界不同）。V2 floor 差 3 倍——RTN 是对的。
4. clamp 边界两侧一致：输入 [-127,127]、输出 [-128,127]（部署实测口径）。非差异项。
5. attention 内部：部署整数 softmax（int8 S、整数 exp、P 压 1/127、BERT 无 mask）；fake 用模型原生 fp softmax（带 mask）。**两侧最大未定界差异**，归"执行细节"侧。
6. 部署 4 个模块调用实例回退 fp；fake 全部 438 条目挂量化（420 Linear + 12 conv + 6 MHA in_proj，1 条 exempt 保持 fp）。
7. V5 口径：激活 int16 网格但量程仍锁 ±127·sa（步长细化 ~258 倍），权重保持 int8——纯粒度诊断，不松裁剪。

## 时间线（服务器本地时间）

- 00:04:48 门 a+b 完成（11.1s）
- 00:14:29 V1 完成（两种子两样本）；00:14:43~00:15:08 V3/V2/V4/V5
- 00:15:20 V1b（全部变体一轮 103s，每变体恢复后验 fp 前向与缓存逐位一致）
- 00:24:26 门 c（22.1s；修三处：列分块段 manifest 索引、conv 4D 展平、CRLF）
- 00:28:35 V1_nomask；00:29:54 mask pad 占比探针（21 token、4 头、90.9% pad）

## 复现命令

```
PY=~/.conda/envs/holobrain/bin/python
cd ~/workspace/holobrain
CUDA_VISIBLE_DEVICES= nice -n 19 $PY /tmp/alg_refq/gates_a_b.py     # 门 a+b
CUDA_VISIBLE_DEVICES= nice -n 19 $PY /tmp/alg_refq/gate_c.py        # 门 c
CUDA_VISIBLE_DEVICES= nice -n 19 $PY /tmp/alg_refq/run_variants.py  # V1..V5,V1b,V1_nomask
# 单跑：run_variants.py --only V1_w8a8_rtn
```

代码：refq_lib.py（独立量化数学+变体+挂载）、gates_a_b.py、gate_c.py、run_variants.py。量化数学未 import 部署侧任何 quant/requant/encode 函数，只复用模型加载/批处理/jpos 协议。

结果文件（/tmp/alg_refq/results/）：gate_a/b/c.json、V1_w8a8_rtn.json、V1b_augbias.json、V2_floor.json、V3_w8afp.json、V4_aonly.json、V5_w8a16.json、V1_nomask.json——每个含逐样本逐种子 jpos_mae/jpos_max/分关节分解/前向耗时。
