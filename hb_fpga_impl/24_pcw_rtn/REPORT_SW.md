# 24_pcw_rtn 阶段 5 报告（软件线）：pcW+RTN 全深度移植、bias 根因修复与深度二分

生成时间：2026-09-04 13:25:11（北京时间）
任务来源：`24_pcw_rtn/BRIEF_SW.md`
执行范围：把模块边界 A/B 实验的 pcW+RTN（逐输出通道 INT8 权重 + requant 就近舍入）移植进
编译器 → fast/golden 解释器 → host_driver 全深度链，在真实样本 s000/s001 上对 fp32 判定，
判据 jpos ≤ 0.045 rad。
代码：`24_pcw_rtn/sw/`（03_compiler 全套复制后改副本，原文件零改动）
结果：`24_pcw_rtn/results/`（21 个 json + 运行日志 + summary_matrix.json）

一句话结论：**pcW+RTN 全深度落地且逐位自检全绿；首轮 0.333 的根因是移植时 bias 路径接错
（不是 pcW 语义问题），修复后 s000=0.2383 / s001=0.1828，比同表逐 tensor 基线低
20.3% / 15.6%；仍超判据 4~5 倍。深度二分显示判据穿越在特征增强/robot_encoder 段
（N≈180），且"只留输出头走 fp"反而把误差推到 0.68——全链一致的静态标定在输出头
把漂移拉回量程，混合精度尾巴不单调更好。**

---

## 1. 全深度数字矩阵（vs fp32，真实样本）

| 配置 | s000 | s001 | 与逐 tensor 基线比 |
|---|---|---|---|
| 逐 tensor 权重 + RTN（ptF，部署粒度对照） | 0.2992 | 0.2165 | — |
| pcW+RTN v1（**bias 路径 bug 版**，v2 表） | 0.3328 | 0.2698 | +11.2% / +24.6% |
| pcW+RTN v1（同 bug，换 fresh 表） | 0.3319 | 0.2683 | 表等价，数字几乎不动 |
| **pcW+RTN v3（bias 修复版，v2 表）** | **0.2383** | **0.1828** | **−20.3% / −15.6%** |
| 判据 | 0.045 | 0.045 | 仍超 5.3× / 4.1× |

配套锚点（引自 `REPORT_GATE.md`，模块边界、模块间 fp 复位）：per-tensor W8A8 部署 bias
约定 s000=0.02167 / s001=0.02437 双绿。全深度 0.24~0.30 与边界 0.022 的差距就是
「每个 op 边界都静态重量化到 int8、没有 fp 复位」的链上复利代价——本轮深度二分（§8）
画出了这个代价的分布。

## 2. 移植内容（BRIEF_SW 交付项）

1. **编译器**（`sw/compiler.py`）：
   - 新 op=14 `OP_SF`：段内先发射系数装载字（每字 10 个 24b 槽），再发射逐列 GEMM；
   - `_emit_gemm(..., rq_pc=, col0=)`：GEMM 标志位 `rq_s bit7`=逐列模式、`rq_m bit15`=RTN
     使能；flags 全 0 = 部署 floor 语义（RTL 回归锚）；
   - `col0` 参数把「描述符 j0（Y 散射偏移，分块内相对值）」和「逐通道系数切片需要的
     全局层列号」分开——这是最容易错位的点，已显式传参 + `(id(rq_pc), c0, n_loc)`
     状态去重防错发；
   - 7 处发射点全部接通：主 wgemm 分块、VT 孪生、多头拆块、BERT q/k/v 与 out.dense、
     relay 输出、swin proj；
   - est_desc_cycles：op=14 记 8 拍。
2. **解释器**（`sw/golden_interp.py` 黄金 + `sw/fast_interp.py` 向量化，逐位一致）：
   OP_SF 装载 sf 寄存器组；逐列 requant 读 `sf[0..n_loc)`（设计 b：sf 按本段 WRAM
   局部列索引，槽 0 = 该 GEMM 第 0 列）。
3. **host_driver**：逐列 requant 全在 PL 段内，host 数值路径不变；本轮新增
   `--fp-after N` 深度二分开关（§8）。
4. **校准链**：`sw/pcw_export.py`（逐通道权重导出）+ `sw/mk_pcw_calib.py`（合成表）+
   `sw/dump_bias.py`（fp 偏置导出）+ `sw/recali.py`（现标对照）。
5. **fast_selftest**：修 bias 后在 v3 build 复跑 9/9 全绿（13~1009 描述符段，覆盖
   op=14、逐列、RTN、aug 列共存）。

## 3. bias 根因复盘（0.333 → 0.2383，本轮最大 bug）

**症状**：pcW+RTN 首轮 0.3328/0.2698，比逐 tensor 基线还差 11%/25%。pcW 不可能让结果
变差——一定是移植 bug（协调者判断正确）。

**根因**：`mk_pcw_calib.py` 首版把所有 pcW 层偏置改成 host fp 偏置（bias_fp_fallback）。
但注意力 qkv 类层的 int8 输出在 PL 段内被 QK^T/PV 直接消费，不回 host——host 的 fp
偏置加不进去，等于每个注意力的 q/k/v 偏置被静默丢弃。部署 v2 语义里这些层走 K+1 增广
（偏置以整数列进累加器），换粒度时我把放置位置一起换错了。这与门禁线第一次翻案
（REPORT_GATE.md §2.1）是同一族错误：**bias 的三个属性——单位域（累加器域 b/(sa·sw)，
不是原始 b）、放置（累加器内 vs host）、粒度（逐 tensor c vs 逐通道）——换任何一个都
要重新过数据流约束**。

**修复**（表 v3 `hw_calib_table_pcw_v3.json`）：
- pcW 层偏置改回 K+1 增广、逐通道化：`w_bias_j = round(b_j / (sa·swc_j·c))`，c 为每层
  统一的 2 的幂（≤64，取满足 max_j |b_j/(sa·swc_j·c)| ≤ 127 的最小者）；
- 编译器 `apply_calib` 的 pcW 分支原来提前 return、跳过 aug 判定——改为落穿；
- 注意力内部生产层（qkv/q/k/v/in_proj，134 层有偏置）c=64 仍溢出时不退 fp fallback，
  而是逐通道饱和到 ±127：近似偏置好过零偏置。真饱和 37 层 3373 通道（占 134 层 36288
  通道的 9.4%，幅度被低估、方向不变）；
- host 边界层（99 层有偏置的 fp fallback）保持部署约定：PL 出 int8、host 反量化后加
  fp 偏置，精确；
- 顺带修 in_proj 偏置键别名（manifest 带 `_weight` 后缀、bias 表已归一，首版 6 个 MHA
  in_proj 查不到偏置被静默丢）；
- 22 个 k_proj 在 v2 表里就没有偏置（Qwen2 系 q/k/v 是 bias=False），fallback 标记是
  无害空操作。

**效果**：s000 0.3328→0.2383（−28.5%），s001 0.2698→0.1828（−32.2%）；编译统计
aug 层调用 0→893、host_bias(pcW) 1244→306。

## 4. RTN 整数语义（写给 RTL 团队）

部署/软件口径（m∈[1,32767]、s∈[9,47]，`>>>`=算术右移=floor 除，sat8=饱和到[−128,127]）：

```
floor 模式（flags 全 0，R3C 回归锚）:  y = sat8((x·m) >>> s)
RTN 模式（rq_m bit15=1）:              y = sat8((x·m + 2^(s-1)) >>> s)，s ≥ 9
逐列模式（rq_s bit7=1）:               第 j 列用 (m_j, s_j)（sf 寄存器组），按列独立
```

RTL `rq_v2` 内部口径（t=s−8，m = mh·2^8+ml，mh=m[15:8] 有符号、ml=m[7:0] 无符号）：

```
sum = x·mh + ((x·ml) >>> 8) = floor(x·m / 2^8)   （嵌套 floor 恒等式，精确）
y   = sat8((sum + rn) >>> t)，rn = 2^(t-1) = 2^(s-9)
```

**两式逐位相等的证明**（全部整数 x、m、s≥9 成立，含负数）：记 p = x·m = 256·sum + r，
r = p mod 256 ∈ [0,256)，则
```
y_sw = floor((p + 2^(s-1)) / 2^s)
     = floor((256·sum + r + 256·2^(s-9)) / 2^s)
     = floor((sum + 2^(s-9) + r/256) / 2^(s-8))
```
sum + 2^(s-9) 是整数（s≥9）、0 ≤ r/256 < 1，被外层 floor 精确吸收，故
y_sw = (sum + 2^(s-9)) >>> (s-8) = y_rtl。floor 模式同理（嵌套 floor 恒等式直接推论）。
s=8（t=0）时 2^(s-9) 非整数，RTL 退化为截断——编译器契约 = RTN 列一律编 s≥9
（本轮标定实际 s∈[21,30]，rq_s 抬升=0，天然满足）。

**验证**：本地 20 万随机向量（x 取满累加器全域 (−2^26,2^26)，2026-09-04 12:04:42）
RTN/floor 双模式 0 失配；RTL 线 60k 向量对 python 神谕位精确（REPORT_RTL.md）；
fast vs golden 真实 build 9 段逐位一致。

**语义权威说明**：analysis_02_server_ab.py 的 pc 线性路径单位正确，是本轮移植锚点；
其 pt/conv 路径有单位 bug（矩阵项被压 sw 倍），相关基线已作废（REPORT_GATE.md §2.2
第二次翻案）。本轮判据不变（0.045），参照系改为门禁线模块边界 0.0217/0.0244 与部署
全深度 0.2993/0.2108。

## 5. OP_SF 描述符格式（SW 线编码）

```
256b 字，op=14：
  bits[251:240]  slot0   本字填充的起始槽号（段内局部列号，[0, COLS)）
  槽 i = bits[24i+23 : 24i]，i=1..10，值 = m<<8 | s（m 16b，s 8b）
```
装载：`sf[slot0+i] = 槽 i`，越界丢弃。GEMM 置 rq_s bit7 后第 j 列用 sf[j]（局部列
[0, n_loc)）——设计 b，分块/多头发射不用回算全局列号；跨 GEMM 复用系数时重发
（编译器按 `(id(rq_pc), c0, n_loc)` 去重，系数未变的连续 GEMM 不重发）。

**与 RTL 线编码分叉**：RTL 用「GEMM 头字后挂 NCW 系数扩展字 + 头字 bit28 使能」。
语义等价、位级不同，各自验证完备；收敛建议见 REPORT_RTL.md，落地前不要混用。
`host_driver --engine rtl` 在 RTL 侧合并 op=14 前不可用（本轮全走 fast）。

## 6. 逐列系数量统计

- s∈[21,30]、m∈[16384,32767]、s_clamped=0（S_MIN_RT=9 下限从未触发）；
- 编码 r_j=(sa·swc_j)/so=m·2^-s：15b m 相对编码误差 ≤2^-14≈6e-5，远小于 INT8 权重
  噪声，视为无损；
- 发射量：OP_SF 字 101,294 个（1434 个逐列 GEMM 调用 × 平均 ~70 列；426 个 pcW 层），
  开销 ≈0.81M 拍 ≈ **4.1ms @198.5MHz，占整帧预估 3162ms 的 0.13%**；
- 权重字节不变（INT8 逐通道与逐 tensor 同为 1 字节/系数），blob 196.46MB。

## 7. 校准流程改动（mk_pcw_calib.py → 表 v3）

输入 v2 部署表（sa/so 不动——激活静态逐 tensor，权威 V2c_pcW 口径）+ 逐通道 swc +
fp 偏置。pcW 条目新增 `"pcw": true, "swc": [...], "rq_ms_col": [[m,s]]×n,
"bias_aug_c": c, "w_bias_int8": [...]`。

统计：438 量化层 = pcW 426（linear/in_proj）+ conv 12（保持逐 tensor + v2 增广）；
偏置 aug 299 层（含 134 注意力内部、37 层饱和）、fp fallback 99 层（host 边界精确）、
无偏置 28 层；c 分布 {64:119, 32:64, 16:67, 8:35, 4:13, 2:1}。

表等价：fresh 现标表与 v2 表 430 层 sa/so/sw/m 比值 1.000（门禁线核对），pcwF 与 pcw
两列差 <0.001——标定表不是误差源（第一次翻案结论，已撤回）。

## 8. 深度二分（前 N 个量化调用走 PL、其后全 fp）

方法：`host_driver --fp-after N`（新增）。全局第 N 次量化调用（gemm/attn 补丁前向，
按真实执行顺序，全链共 816 次）之后全部走原生 fp32，输出照常登记。N=0 纯 fp =
0.0021（机制 sanity）；N=815 与无开关全量跑 **0.2383 完全一致**（交叉验证）。
s000、pcwv3 build：

| N | jpos | 覆盖区段 |
|---|---|---|
| 0 | 0.0021 | 纯 fp |
| 100 | 0.0196 | BERT 前 4 层（绿） |
| 150 | 0.0303 | BERT 12 层 + 特征增强前段（绿） |
| 200 | 0.0542 | **判据穿越点**（特征增强完 + robot_encoder 前 7 层） |
| 250 | 0.1335 | 主跳变完成（0.05→0.13 发生在 200–250） |
| 300 | 0.1330 | 解码器前段 |
| 500 | 0.1366 | 解码器中段 |
| 700 | 0.1337 | 解码器第一遍完 |
| 750 | 0.1321 | + 第一遍 head/t_embed/input_layers |
| 809 | **0.6765** | 只剩最后 5 个输出头调用走 fp——**fp 岛反而爆炸** |
| 816 | 0.2383 | 全量 |

调用映射（run000bisect815.log 的 816 行 [bisect] 记录）：#0–140 文本编码器 BERT 12 层；
#140–190 特征增强（MSDeform/img/text/text_img attn）；#190–816 解码器（robot_encoder 24 层
+ decoder.layers 56 层 **走两遍** + head/t_embed/input_layers 各两遍）。

**读法**：
1. 判据穿越在 N≈180（特征增强/robot_encoder 入口），主跳变 200–250——BERT 段几乎
   免疫（N=150 仍绿）；
2. 250 之后是 0.133 平台一直到 750——解码器主体加量化只抬了 0.001 量级；
3. **N=809 反常**：把最后 5 个输出头调用（decoder.head.convs.0/1、output_layers.1/3/4）
   留 fp，误差反而从 0.24 暴涨到 0.68。记账已排除（5 个未跑段恰好 = 5 个 native 模块；
   4 个缺输入警告在全量跑里同样存在、tensor id 相同，是预存部署行为）。机制推断：
   全链一致的静态标定在输出头的重量化把漂移的激活拉回标定量程（饱和即收缩），
   fp 头则把特征漂移线性放大到输出——**混合精度尾巴不单调更好，误差不是
   "量化越少越好"的函数**。这直接解释 lever F 为什么无效；
4. N=809→816 的 0.44 差值全部来自输出头 5 个调用的重量化"回拉"效应，值得单独
   研究（可能是通往判据的钥匙，也可能是 DPM 头的噪声鲁棒性在起作用）。

## 9. lever F（敏感层豁免，零 RTL）

机制 `compiler --exempt-extra`（复用 exempt_fp 通道）。F1=12 个 BERT intermediate.dense
前缀；F2=38 前缀（BERT FFN + 解码器 FFN 族，展开 261 个调用点豁免）。均 v3 表重编。
s000：**F1=0.2376、F2=0.2384，基线 0.2383——无效**（F2 豁免了 20% 的调用点仍纹丝不动）。
与 §8 读法一致：误差不是这些模块的 per-GEMM 贡献，是链上传播 + 输出头回拉。

## 10. RTL 对齐清单（合并 op=14 时逐项核对）

1. [ ] 描述符编码二选一收敛（§5 分叉：SW op=14 字 vs RTL NCW 扩展字）；
2. [ ] rq_s bit7 / rq_m bit15 与 RTL 头字 bit28/rn_en 的映射表；
3. [ ] sf 槽 24b（m16+s8）与 RTL rq_coeff[COLS×24] 一致；
4. [ ] slot0 局部列语义（设计 b）在 RTL 侧对应（RTL 现按全局列平铺，无 slot 概念）；
5. [ ] RTN 常数 2^(s_j−1) 逐列 vs RTL rn=2^(t−1)、t=s−8 换算；
6. [ ] sat8 饱和口径（R3C 已锚）；
7. [ ] op=14 est_desc_cycles=8 vs RTL 系数泵 2 拍/字的时序账；
8. [ ] fast_interp 的 sf 未装载断言（m=0 即 assert）对应 RTL 上电复位值约定。

## 11. 复现命令

```bash
# ===== 服务器（<SERVER>，CPU，只写 /tmp）=====
PY=~/.conda/envs/holobrain/bin/python
# 1) 逐通道权重导出 + 偏置导出（一次性）
cd ~/workspace/holobrain && CUDA_VISIBLE_DEVICES= $PY /tmp/pcw_rtn/sw/pcw_export.py \
    --manifest /tmp/pcw_rtn/manifest.json --out /tmp/pcw_rtn/pcw_export
CUDA_VISIBLE_DEVICES= $PY /tmp/pcw_rtn/sw/dump_bias.py \
    --manifest /tmp/pcw_rtn/manifest.json --out /tmp/pcw_rtn/fp_biases.json
# 2) 合成 v3 表
cd /tmp/pcw_rtn && $PY sw/mk_pcw_calib.py \
    --v2 /tmp/ae_hostdrv/hw_calib_table_v2.json --scales pcw_scales.json \
    --biases fp_biases.json --out hw_calib_table_pcw_v3.json
# 3) 编译（17s；与既有 build 逐字节对拍过命令重建）
cd /tmp/pcw_rtn && $PY sw/compiler.py --trace /tmp/ae_hostdrv/trace_s000.json \
    --manifest manifest.json --w8 w8_full --calib hw_calib_table_pcw_v3.json \
    --pcw-w8 pcw_export --out build_s000_pcwv3
# 4) 逐位自检
cd /tmp/pcw_rtn && $PY sw/fast_selftest.py build_s000_pcwv3
# 5) E2E（~9 分钟/样本；s001 同理换 trace/batch/ref）
cd /tmp/pcw_rtn && $PY sw/host_driver.py --build build_s000_pcwv3 \
    --trace /tmp/ae_hostdrv/trace_s000.json --batch /tmp/ae_hostdrv/batch_s000.pt \
    --ref /tmp/ae_hostdrv/fp32_ref_000.npz --calib hw_calib_table_pcw_v3.json \
    --sample-id 000 --out result_000_pcwv3.npz
# 6) 深度二分
cd /tmp/pcw_rtn && $PY sw/host_driver.py --build build_s000_pcwv3 ... --fp-after 809 \
    --out result_000_bisect809.npz
# 7) lever F
cd /tmp/pcw_rtn && $PY sw/compiler.py ... --exempt-extra "$(cat exempt_f2.txt)"
# 本地 sw 副本 = 24_pcw_rtn/sw/（含全部脚本）；结果 json 在 24_pcw_rtn/results/
```

## 12. 诚实边界

1. **判据未过**（0.2383/0.1828 vs 0.045）：按新根因陈述，全深度误差主体是链上传播
   复利 + 输出头回拉效应，per-GEMM 粒度在全深度账本占小头——−20%/−16% 真实但不足；
2. conv 保持逐 tensor（12 层，权威 V2_rn_pcW 口径 + v2 增广）；
3. in_proj 的 pcW 是外推（边界 A/B 权威未覆盖，部署表有、门禁留 fp）；
4. 37 层 3373 通道偏置饱和（幅度低估、方向不变），无独立误差项量化；
5. r_j 用 15b 定点网格（相对误差 ≤6e-5），非 fp64 精确；
6. RTN 作用于所有 GEMM（含 conv 与逐 tensor 层）——「pcW+RTN 包」与部署语义差两处
   （粒度+舍入），不是单变量；
7. --engine rtl 不可用（RTL 侧 op=14 未合并）；
8. 深度二分测的是前缀量化误差（非孤立单层），N=809 的 0.68 机制是推断（回拉/饱和
   收缩假说），未做独立验证实验；
9. 标定仍为合成扰动批（8 样本），对真实输入量程失配（首层 in_sat=0.802）原样带入；
10. N=809 与全量的对比依赖"4 个缺输入为预存无害行为"的判断（全量跑同样警告、同样
    tensor id、结果与历史 build 一致）。

## 13. 与两轮翻案的关系

- 第一次（v2 表是否独立误差源）：撤回，本轮 pcw vs pcwF <0.001 差再证；
- 第二次（ab pt/conv 单位 bug）：本轮锚定其 pc 线性路径（未受影响）；全深度预期参照
  系改为"深度代价主导"——实测 0.2383/0.1828 落在预期内；
- 详见 `24_pcw_rtn/REPORT_GATE.md`。

## 14. 交付物清单

- `sw/`：compiler / fast_interp / golden_interp / host_driver / fast_selftest /
  mk_pcw_calib / pcw_export / dump_bias / recali（+ 03_compiler 全套副本）；
- `results/`：result_{000,001}_{pcw,pcwF,ptF,pcwv3}_vs_fp32.json（8）、
  result_000_bisect{0,100,150,200,250,300,500,700,750,809,815}_vs_fp32.json（11）、
  result_000_pcwv3_{F1,F2}_vs_fp32.json（2）、summary_matrix.json、diag_000_pcw.json、
  exempt_f{1,2}.txt、gate_real_results.json、关键运行日志（run000bisect815.log 含
  816 调用映射）；
- 服务器 /tmp/pcw_rtn/：脚本、表 v3、build_s{000,001}_pcwv3 及 F1/F2、全部日志；
  本轮进程已清理（gate_real 未动）。
