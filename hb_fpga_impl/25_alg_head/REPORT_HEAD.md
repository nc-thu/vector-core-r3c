# 输出头解剖：N=809→815 "回拉效应"成分拆解

生成时间：2026-09-04 16:41:20（CST）。样本 s000，判据 jpos ≤0.045 rad，基线 build_s000_pcwv3 = 0.2383。
全部数字从 result json 读取（jpos_mae 字段），路径见文末清单。工作目录 /tmp/alg_head/（服务器 <SERVER>），/tmp/pcw_rtn 与 /tmp/ae_hostdrv 未改动。

## 一、结论（先读这个）

1. **主回拉调用是 #811 decoder.head.convs.1**。瀑布扫描里，把 convs.1 从 fp 换回量化，jpos 从 0.6036 降到 0.2854（−0.318），占 0.436 总回拉的 73%。其次 convs.0（#810，−0.071）、output_layers.4（#814，−0.051）。output_layers.1/3 的 requant 反而轻微有害（各 +0.002）。

2. **回拉无法归因到单一成分**。S8 解耦：A2（头 5 层权重 fp、激活 requant 照常）= 0.6825；A3（权重照常 W8、输出不 requant 不 clamp）= 0.6832。两者都塌回 fp 头水平（0.6744）。只有"int8 权重 × 输出 requant"同时在场才有 0.2383。这是交互效应。

3. **"clamp 把漂移拉回量程"的直接形式被否定**。T4 实测：头 5 层输出饱和率全为 0（pre-clamp 最大值只有满量程 44%~77%），部署侧无任何元素碰 ±127 rail；输入饱和率≈0（convs.0 最大 0.34%，过冲 1.19×）。回拉期间没有发生饱和。

4. **真实机制：requant 网格把小幅度时间结构压成"输出常数"**。部署基线的头输出里，112 个（关节×参数）单元格有 72 个在全部 64 步上精确恒定（9/14 关节的全部 8 参数是单一值，如 j0 全 0.0、j6 全 0.5）；A2/A3 为 0/112，14 关节全漂移（std≈0.07）。fp32 参考 14 关节中 9 个本来就近常数（std≤0.02），所以"恒定输出"在其中 6 个关节误差 <0.03，而"漂移输出"处处 ~0.6。0.68→0.24 的回拉本质是**输出坍缩成常数恰好押中近常数参考**。真正要动的关节（8/9/10/13）基线里仍死/错（j8 误差 0.86）。

5. **机制链**：全链量化漂移让头特征直流大、时间变化分量小——AC 幅度 ~requant 网格 1 LSB。W8 舍入与 requant RTN 联合把 AC 压进同一网格 bin（输出恒定）；任一成分换回精确值（A2 精确权重 / A3 无 requant），AC 跨 bin 抖动，输出变漂移轨迹。convs.1 在瀑布里最关键与之一致：它是头里网格最粗的第一层（so=0.067），AC 最先在它这里被压死。

6. **显式 clamp/so 策略：能微调，不成立为精度手段**。so×k 全 5 层：0.5→0.3867、0.7→0.2686、1.0→0.2383、1.4→0.2388、2.0→0.2059（−13.6%）、3.0→0.2065（平台）。收益定位在 convs.0 单层（convs.0-only ×2=0.2060≈全 5 层 0.2059）；convs.1 单层 ×2=0.2383、ol.4 单层 ×2=0.2388，均无变化。放宽 convs.0 让部分时间结构逃出坍缩，但仍在"坍缩解邻域"微调，离 0.045 判据差 4.6 倍，且方向是"少一点回拉"而非"恢复精度"。

对主根因陈述的修正建议：0904 记录"输出头回拉把漂移激活拉回标定量程（饱和即收缩）"应改写为"输出头 requant 网格把亚-LSB 时间结构舍位成常数输出；该常数在本样本 9 个近常数关节中 6 个押中，故 jpos 下降；这不是精度恢复，是输出坍缩"。

## 二、调用图（T0）

全链 815 次量化调用（#0..#814）。--fp-after N：第 N 次起走原生 fp。最后 12 个调用：

| gidx | op | 模块 |
|---|---|---|
| #803 | attn | decoder.layers.56 |
| #804 | gemm | decoder.layers.56.joint_pos_encoder.mlp.0 |
| #805 | gemm | decoder.layers.56.joint_pos_encoder.mlp.2 |
| #806 | attn | decoder.layers.59 |
| #807 | attn | decoder.layers.61 |
| #808 | gemm | decoder.layers.64.layers.0.0 |
| #809 | gemm | decoder.layers.64.layers.1 |
| #810 | gemm | decoder.head.convs.0（Conv1d 256→128, k=768）|
| #811 | gemm | decoder.head.convs.1（Conv1d 128→64, k=384）|
| #812 | gemm | decoder.head.output_layers.1（Linear 64→64, pcW）|
| #813 | gemm | decoder.head.output_layers.3（Linear 64→64, pcW）|
| #814 | gemm | decoder.head.output_layers.4（Linear 64→8, pcW）|

N 档对应（fp 的调用）与结果：

| N | fp 的调用 | jpos |
|---|---|---|
| 809 | layers.64.layers.1 + 5 个头调用 | 0.6765 |
| 810 | convs.0/1 + ol.1/3/4 | 0.6744 |
| 811 | convs.1 + ol.1/3/4 | 0.6036 |
| 812 | ol.1/3/4 | 0.2854 |
| 813 | ol.3/4 | 0.2870 |
| 814 | ol.4 | 0.2888 |
| 815 | 无（全量化）| 0.2383 |

结构说明：head（纯读出，无反馈）在全流程里被调 10 次；末 5 个 gidx 属于第 10 次（末次）调用，末次主导最终动作（N=810 只 fp 末次即 0.6744 ≈ A2/A3 全 10 次改动的 0.68）。

## 三、S7 瀑布（T1）

| 加回量化的调用 | jpos 变化 | 说明 |
|---|---|---|
| #809 layers.64.layers.1 | 0.6765→0.6744（−0.0021）| 近中性 |
| #810 head.convs.0 | 0.6744→0.6036（−0.0708）| 次要 |
| **#811 head.convs.1** | **0.6036→0.2854（−0.3182）** | **主回拉，73%** |
| #812 ol.1 | 0.2854→0.2870（+0.0016）| 轻微有害 |
| #813 ol.3 | 0.2870→0.2888（+0.0018）| 轻微有害 |
| #814 ol.4 | 0.2888→0.2383（−0.0505）| 末层贡献 |

## 四、S9 so 乘子扫描（T2）

表生成：mk_pcw_calib_so.py（mk_pcw_calib.py 副本 + --so-mult/--so-mods）；命中层 so×k，requant 系数 r=(sa·swc_j)/(so·k) 联动重算（conv 走 v1_encode 口径），bias 列经 v3 流程重算后与原表逐位相同。k=1.0 再生成表的 gemms 段与 hw_calib_table_pcw_v3.json 完全一致，重编译 build 与 build_s000_pcwv3 逐字节一致（diff -r 为空）。

| 配置 | jpos | vs 基线 |
|---|---|---|
| ×0.5（5 层）| 0.3867 | +62% |
| ×0.7（5 层）| 0.2686 | +13% |
| ×1.0 | 0.2383 | 基线 |
| ×1.4（5 层）| 0.2388 | ±0 |
| **×2.0（5 层）** | **0.2059** | **−13.6%** |
| ×3.0（5 层）| 0.2065 | −13.3%（平台）|
| ×2.0 仅 convs.0 | 0.2060 | −13.6%（全部收益来自此层）|
| ×2.0 仅 convs.1 | 0.2383 | 0 |
| ×2.0 仅 ol.4 | 0.2388 | 0 |

arm12/gripper 拆分：×2.0 改善全在 arm12（0.2431→0.2053），gripper 不动（0.4476→0.4475）。

## 五、S8 权重/激活解耦（T3）

head_probe.py 三模式（子类化 HostDriver，只旁路 5 个头模块，其余照部署段执行）：

| 模式 | 定义 | jpos | 读法 |
|---|---|---|---|
| stats | 部署照跑 + 宿主复算对拍 | 0.2383 | 与基线同分，复现成功 |
| A2 | 权重 fp（W/swc 不取整）+ 激活量化/requant 照常 | 0.6825 | ≈ fp 头 |
| A3 | 权重照常 W8 + 输出不 requant/clamp（acc×sa×swc_j 直出）| 0.6832 | ≈ fp 头 |
| （参照）fp 头 | 全 fp（--fp-after 810）| 0.6744 | |

判读：任务书预设的分支（A2≈0.2383 → 激活 requant 是答案；A3≈0.68 → 权重量化无害）都不成立——任一成分单独拿掉都回 0.68，回拉是 (W8 舍入 × requant 舍入) 联合效应。

## 六、T4 饱和统计 + 轨迹解剖

stats 模式实测（全量化链、10 次头调用累计）：

| 模块 | in_sat | in_max/127 | out_sat | out_max/127 | 对拍不一致 |
|---|---|---|---|---|---|
| convs.0 | 0.34% | 1.19 | 0 | 0.77 | 490239/573440（dm_mean 7.3 LSB）|
| convs.1 | 0 | 0.91 | 0 | 0.65 | 540196/573440（dm_mean 17.5）|
| ol.1 | 0 | 0.61 | 0 | 0.36 | 0（逐位一致）|
| ol.3 | 0 | 0.26 | 0 | 0.44 | 0（逐位一致）|
| ol.4 | 0 | 0.44 | 0 | 0.44 | 54960/71680（dm_mean 9.6）|

部署侧 dep_rail=0：头 5 层输出无任何元素触 ±127。回拉假说预言的"头输入饱和率不低"不成立（≤0.34%）。

轨迹解剖（64 步 × 14 关节 × 8 参数）：
- 基线：72/112 个（关节,参数）单元格精确恒定（9 关节的全部 8 参数恒定；j0 全 0.0、j6 全 0.5、j8 全 1.2056）。参考里 9 个关节 std≤0.02，其中 6 个误差 <0.03。
- A2/A3：0/112 恒定，全部漂移（std≈0.07），逐关节误差均匀 ~0.6-0.8。
- 基线误差构成：9 恒定关节贡献 1.736/14，5 非恒定关节贡献 1.601/14；最大单项 j8=0.86（参考 std=0.13 在动，输出恒定）。

## 七、诚实边界

1. 单样本 s000；s001 未跑（时间盒）。so×k 未跨样本验证。
2. head_probe 对 convs.0/1、ol.4 的宿主复算与部署段不逐位一致（dm_mean 7~18 LSB；ol.1/ol.3 逐位一致）。差异来自部署段 conv 走 1344 行伪窗口 im2col 约定（Conv1d 垫成 (N,C,1,L) 后标量 padding 把 H=1 垫成 3），宿主走真值 448 行 im2col——数学等价的真值口径，非位级复刻。convs 的 in/out_sat 读数是真值口径指示值；A2/A3 的 conv 路径同口径。关键结论不依赖此口径：部署侧 rail=0 与 72/112 恒定单元格全部直接测自部署输出。
3. --fp-after 只作用末次 head 调用；A2/A3 作用全部 10 次。两者一致指向 0.68，末次主导。
4. "AC 幅度 ~1 LSB、W8 偏移决定是否跨 bin"是对三方结果的机制解释，未做独立显微测量（时间盒）；恒定性与 rail 统计是实测。
5. so×k 只改 5 个头模块 so 与联动 requant 系数，bias 列不变；不改变上游标定失配来源（合成标定 vs 真实输入）。
6. 服务器有兄弟代理 4 个 E2E 并发，单轮 E2E 从 ~8 分钟拖到 ~15-20 分钟；自有并发始终 ≤5。

## 八、复现命令（可粘贴）

PY=~/.conda/envs/holobrain/bin/python（ssh <SERVER> 后）

（1）T1 瀑布（S7）：
  cd /tmp/pcw_rtn
  for N in 810 811 812 813 814; do
    $PY sw/host_driver.py --build build_s000_pcwv3 \
      --trace /tmp/ae_hostdrv/trace_s000.json --batch /tmp/ae_hostdrv/batch_s000.pt \
      --ref /tmp/ae_hostdrv/fp32_ref_000.npz --calib hw_calib_table_pcw_v3.json \
      --sample-id 000 --fp-after $N --out /tmp/alg_head/result_000_bisect${N}.npz \
      > /tmp/alg_head/run000bisect${N}.log 2>&1 &
  done

（2）T2 so 乘子（S9）表 + build + E2E：
  MODS5='decoder.head.convs.0,decoder.head.convs.1,decoder.head.output_layers.1,decoder.head.output_layers.3,decoder.head.output_layers.4'
  $PY /tmp/alg_head/mk_pcw_calib_so.py --v2 /tmp/ae_hostdrv/hw_calib_table_v2.json \
      --scales /tmp/pcw_rtn/pcw_scales.json --biases /tmp/pcw_rtn/fp_biases.json \
      --so-mult 2.0 --so-mods "$MODS5" --out /tmp/alg_head/hw_calib_table_k20.json
  cd /tmp/pcw_rtn
  $PY sw/compiler.py --trace /tmp/ae_hostdrv/trace_s000.json --manifest manifest.json \
      --w8 w8_full --calib /tmp/alg_head/hw_calib_table_k20.json --pcw-w8 pcw_export \
      --out /tmp/alg_head/build_k20
  $PY sw/host_driver.py --build /tmp/alg_head/build_k20 \
      --trace /tmp/ae_hostdrv/trace_s000.json --batch /tmp/ae_hostdrv/batch_s000.pt \
      --ref /tmp/ae_hostdrv/fp32_ref_000.npz --calib /tmp/alg_head/hw_calib_table_k20.json \
      --sample-id 000 --out /tmp/alg_head/result_000_sok20.npz
  # 单层验证：--so-mods 换成单模块名（k20c0/k20c1/k20o4 同法）

（3）T3/T4（S8）：三模式探针：
  for M in stats A2 A3; do
    $PY /tmp/alg_head/head_probe.py --build /tmp/pcw_rtn/build_s000_pcwv3 \
      --trace /tmp/ae_hostdrv/trace_s000.json --batch /tmp/ae_hostdrv/batch_s000.pt \
      --ref /tmp/ae_hostdrv/fp32_ref_000.npz --calib /tmp/pcw_rtn/hw_calib_table_pcwv3.json \
      --mode $M --sample-id 000 --out /tmp/alg_head/result_000_${M}.npz &
  done

（4）轨迹解剖：
  $PY -c "import numpy as np; raw=np.load('/tmp/pcw_rtn/result_000_pcwv3.npz')['pred_actions_raw']; print(sum(1 for j in range(14) for k in range(8) if len(np.unique(raw[:,j,k]))==1), '/112 const cells')"

## 九、文件清单（/tmp/alg_head/）

- 表与 build：hw_calib_table_{k100,k05,k07,k14,k20,k30,k20c0,k20c1,k20o4}.json、build_k{100,05,07,14,20,30,20c0,20c1,20o4}/
- 结果 json：result_000_{bisect810..814, sok05..sok30, sok20c0/c1/o4, A2, A3, headstats, headstats2, stats}_vs_fp32.json（stats/headstats2 含 probe_stats 字段）
- 日志：run000*.log（与结果同名）
- 脚本：mk_pcw_calib_so.py（表生成器，k=1.0 已验证复现 v3 表与 build）、head_probe.py（stats/A2/A3 三模式探针）、patch_mk.py（补丁留档）
- 本报告：REPORT_HEAD.md
