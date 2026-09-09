# REPORT_CALIB — 真实样本标定线：假设证伪，标定失配是二阶项
生成：2026-09-04 16:01:21（实验窗口 14:34–16:01，服务器 <SERVER>）

## 0. 结论（先读这五条）

1. 假设"合成标定对真实输入失配是重要误差注入项，真实样本标定能压掉一部分"
   **被证伪**。全网换成真实 absmax 标定，s000 从 0.2383 恶化到 0.2710（+13.7%），
   s001 从 0.1828 恶化到 0.2097（+14.7%）；p99.9 口径 s000 −2.9%、s001 持平。
   最好的一档也只有 −2.9%，离 0.045 判据还差 5 倍以上。
2. 只修失配最狠的段近乎无效：仅改 patch_embed（−1.0%）、仅改 #140–250 段
   （特征增强+robot_encoder 入口，−0.4%）。段内 sa 乘 0.8/0.9/1.1/1.25 扫描
   全部落在 ±0.5% 的噪声带里，没有敏感性。
3. 失配本身是真实存在的（不是测量假象）：437 个量化模块里 109 个（s000）/
   115 个（s001）真实输入超合成量程 ×1.05，patch_embed 输入 80% 被削顶，
   与此前 in_sat=0.802 的证据逐位对上。但它对端到端误差贡献很小——
   削顶发生在第一个 conv 的输入端，下游 LayerNorm 把幅度归一化吸收掉了；
   而为消除削顶而放大 sa 会让每个值的量化步长变粗，在全深度 583 次 GEMM
   调用的逐级 requant 复利下，粗步长的代价反而更大。这就是 absmax 口径
   恶化的机制。p99.9 口径把步长收细（sa 中位数只有 v2 的 0.648），
   换 0.1% 削顶，净效果微好——分辨率比削顶更值钱，但量级 ≤3%。
4. 主误差项维持 0904 翻案后的结论：GEMM int8 逐级 requant 的深度复利
   （only_gemm 0.294 ≈ 全链 0.298）。标定线修的注入项在全深度账本里占小头，
   这条线可以收了；下一杠杆应看向逐级 requant 本身（如更高精度累加器/
   部分层 fp16 requant/通道级激活定标）。
5. 通道级结构是真实的（这是本轮最有复用价值的副产品）：top 0.1% 通道比值
   57–115×，聚集在 backbone.stages.3.blocks.0.ffn.layers.1、
   decoder.robot_encoder.layers.1.position_encoder.mlp.2、
   decoder.layers.*.adaLN_modulation.1、t_embed.mlp.2 等少数模块。
   BERT 段典型通道只用掉逐 tensor 量程的 ~15%。若未来做激活侧 per-channel
   或 outlier-aware 定标，这些就是目标清单（S10 交付，
   见 diag_real_s000.json 的 top_chan_01pct/top_chan_1pct 与 _chan.npz）。

## 1. 方法与口径

- 链路：compiler → fast_interp（host_driver 全深度仿真）→ 对 fp32_ref 算
  jpos MAE（判据 ≤0.045 rad）。基线 build_s000_pcwv3 / build_s001_pcwv3
  （pcW+RTN，合成 v3 表）。
- 真实统计采集（diag_real.py）：fp 模型在真实 batch 上前向，hook 437 个
  量化模块（v3 表 438 键中 437 个可挂；text_feat_map 是常量映射保持 v2）。
  hook 口径与 02_quant hw_calib.calibrate 完全一致（同样取 args[0]，
  force_mha_slow_path 开启），只换数据不换语义。6 个 MHA in_proj 别名挂
  父模块：输入=query，输出=F.linear(q, in_proj_weight) 现算（真输出，
  不是注意力输出）。absmax/饱和计数精确，p99/p99.9/p99.99 用均匀采样
  reservoir（每调用 ≤4000 值、每模块 ≤16 万值）。
- 表生成（mk_real_tables.py）：sa=真实 in 统计/127、so=真实 out 统计/127
  （absmax 或 p99.9 两种口径），r_star/m_requant/s_shift/m_s8/偏置增广
  （w_bias=round(b/(sa·sw)/c)，c 重选）全部按 hw_calib.build_hw_params
  公式整体重算，再过 mk_pcw_calib.py v3 流程重生成 rq_ms_col 与逐通道
  增广偏置——没有手改任何 bias 字段。权重 sw/swc 不动。
- 有效性门槛：每个新表重编 build 后 fast_selftest 9 桶对拍，
  10 个 build 全部 ALL PASS。所有 requant 编码 s∈[20,30]、s_clamped=0。
- 评估：host_driver --calib 新表 + 对应新 build。S1 两口径跑 s000
  （in-sample，标定=评估同一批，有泄漏）与 s001（out-of-sample），
  分开列。S2/S3/S4 只跑 s000。

## 2. 数字总表（jpos MAE，rad；全部从 result json 读出，主会话已复核）

| 实验 | s000 (in) | 对基线 | s001 (out) | 对基线 |
|---|---|---|---|---|
| 基线 pcwv3 合成表 | 0.2383 | — | 0.1828 | — |
| S1 absmax 全网真实 | 0.2710 | +13.7% | 0.2097 | +14.7% |
| S1 p99.9 全网真实 | 0.2313 | −2.9% | 0.1855 | +1.5% |
| S2 仅 patch_embed | 0.2360 | −1.0% | — | — |
| S3 仅 #140–250 段 | 0.2373 | −0.4% | — | — |
| S4 sa×0.8/0.9/1.1/1.25 | 0.2359/0.2384/0.2364/0.2383 | ±0.5% 噪声 | — | — |

S1p 的分项（s000）：arm12=0.2408（基线 0.2431），gripper=0.3625
（基线 0.4476）——微小改善主要来自夹爪通道。

## 3. 诊断仪表盘（T1）

逐段（s000，util=真实 absmax/合成量程，sat=|x|>127·sa 占比）：

| 段 | 调用区间 | 模块数 | util 中位 | util max | >1.05 | sat max | 通道比值中位 |
|---|---|---|---|---|---|---|---|
| patch_embed | #0–43 | 2 | 1.797 | 2.17 | 2 | 0.802 | 1.7 |
| 视觉主干(swin) | #1–86 | 109 | 0.951 | 1.45 | 23 | 0.0004 | 2.6 |
| BERT 文本 | #87–122 | 72 | 1.003 | 1.57 | 13 | 0.0004 | 6.5 |
| 特征增强 | #124–189 | 102 | 0.983 | 1.44 | 20 | 0.0055 | 5.0 |
| robot_encoder 入口 | #190–214 | 37 | 1.116 | 1.84 | 23 | 0.0070 | 3.3 |
| 解码器主体 | #222–809* | 102 | 1.000 | 1.72 | 23 | 0.0027 | 4.5 |
| 输出头 | #215–814* | 12 | 1.000 | 1.21 | 5 | 0.0033 | 3.0 |

（*按模块首次调用归段；decoder 头几次调用与 robot_encoder 段交叠。）
- sa real/v2 全网分布：p10=0.810 / p50=1.000 / p90=1.280 / min 0.479 /
  max 2.172——合成表中位数无偏，失配集中在尾部。
- s001 复检：同构（115/437 超量程，patch_embed sat 67.8%）；样本间逐模块
  absmax 比值 p50=1.000、p90=1.071——真实输入分布跨样本稳定。
- top outlier 通道（S10 交付，全网 222,093 通道按 chan_absmax/模块中位排序）：
  - `decoder.robot_encoder.layers.1.position_encoder.mlp.2`：通道 113/81/101/138/195/247/140，比值 115/107/101/73/70/65/60×
  - `backbone.stages.3.blocks.0.ffn.layers.1`：通道 2734/957/2166/2318/2717/1423…，比值 90/86/80/76/75/71×（top-1% 里 500 条中占 243 条，最大 outlier 聚集地）
  - `decoder.t_embed.mlp.2` 通道 145（88×）、`text_encoder...layer.11.output.dense` 通道 2947/51（68/57×）、`decoder.layers.*.adaLN_modulation.1`（每层约 30 条进 top-1%）
  完整清单在 diag_real_s000.json（top_chan_01pct 全量、
  top_chan_1pct 截前 500）与 diag_real_s000_chan.npz（每模块通道 absmax）。

## 4. 诚实边界

- 标定集只有 1 个真实样本（s000）。服务器上现成 batch 只有 s000/s001
  （robotwin_subset 有原始数据但无现成批，未再抽）。s000 数字有泄漏，
  s001 是干净对照——两口径方向一致，结论不靠泄漏。
- S2/S3/S4 只跑了 s000（in-sample），其"无效"结论按 S1 的 in/out 一致性
  外推；段扫描 ±0.5% 的抖动就是单跑噪声底，别解读出形状。
- S1p 口径有 135 层退 host fp bias（v3 为 99）：更小的 sa 让偏置增广在
  c=64 更易溢出。注意力内部层仍走饱和增广不变（aug_sat=134 与 v3 同），
  但两层口径的 host 边界 bias 放置有差异，−2.9% 里混着这一项。
- text_feat_map（常量映射）保持 v2；6 个 MHA in_proj 的输出统计是
  F.linear 现算的真输出，但该 6 层的 so 口径与其余层来源不同源。
- jpos 单指标、2 个样本、判据 0.045 全线红；本轮不改变"5 倍红线"现状。
- 每个单跑是标准 9 分钟全深度协议；因同期机器上有另一批任务
  （load 140–160），实际墙钟 ~30 分钟/跑，数字本身不受影响。

## 5. 复现命令（服务器，完整可粘贴）

```bash
PY=~/.conda/envs/holobrain/bin/python
# T1 诊断（s000；s001 换 batch/out 路径）
cd ~/workspace/holobrain && CUDA_VISIBLE_DEVICES= $PY /tmp/alg_calib/diag_real.py \
    --batch /tmp/ae_hostdrv/batch_s000.pt --out /tmp/alg_calib/diag_real_s000.json
# 表生成（S1a 例；S1p 改 --mode p999，S2 改 --scope patch_embed，
# S3 改 --scope seg140_250，S4 加 --mult 0.8 等）
cd /tmp/pcw_rtn && $PY /tmp/alg_calib/mk_real_tables.py \
    --stats /tmp/alg_calib/diag_real_s000.json --v2 /tmp/ae_hostdrv/hw_calib_table_v2.json \
    --biases fp_biases.json --mode absmax --scope all --mult 1.0 --out /tmp/alg_calib/v2_S1a.json
$PY sw/mk_pcw_calib.py --v2 /tmp/alg_calib/v2_S1a.json --scales pcw_scales.json \
    --biases fp_biases.json --out /tmp/alg_calib/t_S1a.json
# 编译 + 自检 + 评估（s001 换 trace/batch/ref/build）
$PY sw/compiler.py --trace /tmp/ae_hostdrv/trace_s000.json --manifest manifest.json \
    --w8 w8_full --calib /tmp/alg_calib/t_S1a.json --pcw-w8 pcw_export --out /tmp/alg_calib/b_S1a_s000
$PY sw/fast_selftest.py /tmp/alg_calib/b_S1a_s000
$PY sw/host_driver.py --build /tmp/alg_calib/b_S1a_s000 --trace /tmp/ae_hostdrv/trace_s000.json \
    --batch /tmp/ae_hostdrv/batch_s000.pt --ref /tmp/ae_hostdrv/fp32_ref_000.npz \
    --calib /tmp/alg_calib/t_S1a.json --sample-id 000 --out /tmp/alg_calib/result_000_S1a.npz
# 一键编排：bash /tmp/alg_calib/run_all.sh tables|builds|selftest|roundA|roundB|roundC
# 读数（从 json，不凭记忆）：
$PY -c "import json; print(json.load(open('/tmp/alg_calib/result_000_S1a_vs_fp32.json'))['jpos_mae'])"
```

## 6. 文件清单（服务器 /tmp/alg_calib/，2.6 GB）
- diag_real_s000.json / diag_real_s001.json（+_chan.npz）：诊断仪表盘
- v2_S1a/S1p/S2/S3/S4k{08,09,11,125}.json：中间 v2' 表
- t_S1a/.../t_S4k125.json：8 张 v3 格式新表（评估用表）
- b_*/：10 个重编 build（含 s001 两个）
- results/：12 个 result_*_vs_fp32.json（含拷入的两个基线；主会话已逐个复核）
- *.log：tables/builds/selftest/roundA/B/C、diag_s000/s001、run_*_S*.log
- 脚本：diag_real.py / mk_real_tables.py / run_all.sh / collect.py
  （本地副本 e:\GPU ARCH\vector_core_sim\hb_fpga_impl\25_alg_calib\）
