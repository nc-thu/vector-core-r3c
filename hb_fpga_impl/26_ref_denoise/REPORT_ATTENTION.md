# REPORT_ATTENTION — attention 内部三档拆解 + 真尺度上链实修：缺口 87% 在占位标定常数

生成：2026-09-07 02:47（本地；服务器梯子完成 01:54、实修完成 02:40，<SERVER>）
服务器产物：/tmp/alg_refq/results/{attn_ladder,attn_family,attn_diag,attn_calib_s000,result_000_fix_vs_fp32,result_001_fix_vs_fp32,task2_query_thrice,task3_head_cells}.json
**主会话复核**：梯子 A0-A5、家族、实修 0.19222/0.13991、query-thrice 增量恰为 0、常数格 0/11/11/0 均从 json 重读；A6=0.02577 由主会话本人复跑 run_full_repro.py 确认（fp 门 0.00000、restore guard 干净）。

## 一句话结论

R2（假量化 0.0208）到 R3（整数链 0.2383）之间 0.2175 的缺口，**87.4% 集中在一对部署尺度常数上：attention 内部 v 码与 PV 段 requant 用的是公式占位值，不是实测值**（compiler.py L1139 占位 `_ph_rq` r=2048/(k·127²)、L1163 `_attn_rq`；PV r=σv/(127·σo)，占位只有真值的 0.1~0.4 倍——temporal 中位 0.107，即 PV 码欠程约 10 倍）。同一结构换真 absmax 尺度（A5）误差立刻跌回 0.027。真尺度上链实修：**s000 0.23834→0.19222（−19.3%）、s001 0.18275→0.13991（−23.5%，out-of-sample 改善反而更大）**。

## 三档梯子（+A4/A5/A6；s000、部署种子、CPU）

| 档位 | 加什么 | jpos MAE | 相对 V1 增量 | 占 0.2175 缺口 |
|---|---|---|---|---|
| A0=V1 | 假量化基线（attention 内部全 fp） | 0.02080 | — | — |
| A1 | S 按部署 so 压 int8 回浮点，softmax 仍 fp | 0.02437 | +0.0036 | 1.6% |
| A2 | A1 + BERT 整数 exp 查表（round(2^(−d/16)·4096)） | 0.02336 | +0.0026 | 1.2% |
| A3 | A2 + P 压 1/127 网格进 PV | 0.02023 | −0.0006 | −0.3%（噪声） |
| **A4** | **A3 + v 压码 σvs + PV 部署 requant（完整部署数值链）** | **0.21102** | **+0.1902** | **87.4%** |
| A5 | A4 结构不变，S/PV 换本块真 absmax 尺度 | 0.02722 | +0.0064 | 3.0% |
| A6 | 码+真常数+σ_S deq 的完整整数执行复刻（V1 GEMM 背景） | 0.02577 | +0.0050 | 2.3% |

S 网格、整数 exp 表、P 网格三档合计 ≤1.6%，全是噪声级。A4 与部署动作输出最像（rel_l2 0.307 vs 其他档 ~0.68、corr 0.947）——部署的动作输出确实被这一档失真主导。

复刻过程本身做了一次 first-divergence：恒等对拍暴露并修复了复刻侧 3 个掩码/组装 bug（temporal 头交错、bimha 掩码广播打乱 v 侧组装、rotary where 广播出 (1,N,M)），修复后七家族恒等全部 ≤4.5e-07，梯子数字才可信。

## 家族拆解（单家族走 A4，其余 V1；attn_family.json）

| 家族 | A4 单家族 jpos | 占缺口 |
|---|---|---|
| **TemporalJointGraphAttention** | **0.18500** | **75.6%** |
| RotaryAttention | 0.08522 | 29.7% |
| WindowMSA | 0.03580 | 6.9% |
| BertAttention / BiMHA / MHA / JGA | 0.019~0.023 | 噪声级 |

（家族间有交互，占比之和可超 100%。）所有家族 A5 都回落 ~0.02。temporal_A4 输出波动放大 542 倍——去噪循环里每步都跑的 temporal attention 是重灾区，与解剖结论（temporal 在每步 60 调用里占大头）自洽。

## 真尺度上链实修（2026-09-07 02:40）

测量口径：σ_S = 每模块一个静态常数，absmax(plain q@k)/127，取 s000 一次前向该模块全部调用的最大值（70 模块/262 调用，attn_calib_s000.json）；so_q/k/v/o 沿用 v3 表原值。工具链零代码改动：`cp -r sw → swfix` 后 compiler/host_driver/fast_interp 与原件逐字节相同（diff 验证），变化全走数据——compile 既有开关 `--attn-calib`，232 个注意力步全部从占位切到校准路径。正确性锚：不加 attn-calib 重编的 build 与 build_s000_pcwv3 的 host_plan/weights_blob sha256/model_summary 逐字节一致；fast_selftest 两个修复 build ALL PASS。

| 样本 | 部署基线 | 实修（真尺度上链） | 改善 |
|---|---|---|---|
| s000（尺度来源，in-sample） | 0.23834 | **0.19222** | **−19.3%** |
| s001（out-of-sample） | 0.18275 | **0.13991** | **−23.5%** |

编码无损（m 全部 >16000，相对误差 0）、静态化无罪（A5 静态 0.02738 ≈ 动态 0.02722）。

## 为什么没到 0.03~0.05 锚点（差异分析 + 主会话警告）

同语义复刻侧 A6=0.02577 说明注意力修复本身有效（复刻侧 0.211→0.026，−88%）。真链只兑现 −19%/−24%，剩余 ≈0.16 的去向：

1. 代理判读：非注意力 GEMM 逐级 requant 复利是真链残差主因（引 base 链 bisect N=250→0.133、0904 only_gemm≈全链）。
2. **主会话警告：上述两条旧证据都是在"注意力占位常数还坏着"的链上测的**——base bisect 前 250 个调用含 54 个带坏常数的 attention 事件；only_gemm 同样包含 attention GEMM 边界。"GEMM 复利是大头"要在修复链上重测（fixed 链 bisect / 修复背景下整数 vs 浮点 GEMM 对换）才能成立。
3. **两套假量化 harness 都没有仿真非 GEMM 算子的整数执行**：部署里 NORM/ELTWISE 走 AE_ACTV int8 引擎（a3 轮 148+148 真实站点）、softmax 走 SM16/SM32、rotary 读的是 int8 存储的中间值；harness 里这些全是 torch fp。这是 A6 与真链之间一整类未覆盖的语义差，候选份额未定界。
4. 修复链 N=250 混合态 0.2938 非单调（与 N=809 同族的拼接病态），单点不过度解读。
5. 参考底噪 fp_resample_floor=0.0457：0.03 锚点本身贴着噪声下界。

## 附带定案（任务 2/3）

- **query-thrice 接线 e2e 免费**：V1 + 仅 6 个 text MHA 按 fwd_mha L744 接线（in_proj 三份全喂 query）= 0.02080，增量恰 0.00000（五位小数全同）。模块输出级差 163% 被下游完全吸收——不是第二个要修的 bug，归档。
- **输出头常数格不随注意力修复解除**：112 格 64 步恒定数 = fp 0 / base 链 11 / fix 链 **11 且同一批**（多为 0/7 边缘分箱）。这 11 格的恒定来自输出头前的 GEMM 量化，跟注意力无关，随 GEMM 线走。

## 判定（2026-09-07 02:40）

1. 占位尺度常数机制在真链上方向正确、部分兑现（−19.3%/−23.5%，out-of-sample 更大）；注意力从第一瓶颈退到第二。
2. 真链残差 ≈0.16 的归因未关闭：代理押"GEMM requant 复利"，主会话指出其支撑证据被坏常数混淆、且 AE_ACTV/SM16 整数算子从未被仿真——下一轮在修复链上重测 bisect + 仿真 ACTV 族，两件事做完才能下结论。
3. 0.192/0.140 仍超判据 4.3×/3.1×；下一档杠杆回到 GEMM requant 链（int16 边界保留 / QAT），但排序等上一条归因关闭后再定。

## 复现命令

```
PY=~/.conda/envs/holobrain/bin/python   # ssh <SERVER>
# 梯子/家族/diag（/tmp/alg_refq/attn_emul.py，一轮 227s+73s）
cd ~/workspace/holobrain && CUDA_VISIBLE_DEVICES= nice -n 19 $PY /tmp/alg_refq/run_attn_ladder.py
# A6 复刻（~2min）
cd ~/workspace/holobrain && CUDA_VISIBLE_DEVICES= nice -n 19 $PY /tmp/alg_refq/run_full_repro.py
# σ_S 实测（CPU 副本 attn_calib.py，仅 3 处 cuda→cpu，不在部署链上）
cd ~/workspace/holobrain && CUDA_VISIBLE_DEVICES= nice -n 19 $PY /tmp/alg_refq/swfix/attn_calib.py --batch /tmp/ae_hostdrv/batch_s000.pt --out /tmp/alg_refq/results/attn_calib_s000.json
# 实修：重编 + e2e（s001 换 trace/batch/ref/build）
cd /tmp/alg_refq/swfix && $PY compiler.py --trace /tmp/ae_hostdrv/trace_s000.json \
    --manifest /tmp/pcw_rtn/manifest.json --w8 /tmp/pcw_rtn/w8_full \
    --calib /tmp/pcw_rtn/hw_calib_table_pcw_v3.json --pcw-w8 /tmp/pcw_rtn/pcw_export \
    --attn-calib /tmp/alg_refq/results/attn_calib_s000.json --out /tmp/alg_refq/build_s000_fix
$PY fast_selftest.py /tmp/alg_refq/build_s000_fix
cd ~/workspace/holobrain && CUDA_VISIBLE_DEVICES= nice -n 19 $PY /tmp/alg_refq/swfix/host_driver.py \
    --build /tmp/alg_refq/build_s000_fix --trace /tmp/ae_hostdrv/trace_s000.json \
    --batch /tmp/ae_hostdrv/batch_s000.pt --ref /tmp/ae_hostdrv/fp32_ref_000.npz \
    --calib /tmp/pcw_rtn/hw_calib_table_pcw_v3.json --sample-id 000 --out /tmp/alg_refq/results/result_000_fix.npz
```
