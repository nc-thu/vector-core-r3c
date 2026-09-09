# 算法线 CHANGELOG — Algorithm & Accuracy（量化与部署精度）

管什么：HoloBrain-0（HB-GD 0.2B）W8A8 全深度部署的数值精度——误差归因、量化方案、部署链 bug。判据 jpos ≤ 0.045 rad（≈2.6°），双样本 s000 定位 / s001 验证。
版本规则与命名见根目录 [LINES.md](../LINES.md)。服务器约定 `/tmp/algo_v{N}/`（历史轮沿用当时的 `/tmp/alg_*` 目录名）。

---

## v5 ——（下一版，规划中）int16 边界保留 / QAT
- 内容：6 个文本注意力块（feature_enhancer text_attn_blocks）的边界中间态保 int16 的软件 A/B；不够再上 QAT。归因已在 v4 关闭，这两个工具第一次定义良好。
- 依据：v4 定案 k 侧 1.07 倍偏差 = 文本路径 6 层量化深度损伤，且在语义天花板 0.04247 之内；v4 终局距判据差 11%（0.0050 rad），已无驱动侧位点。

## v4 —— 2026-09-08 11:14 三 bug 修复收官：0.1922→0.0500 / 0.1399→0.0622（−74.0% / −55.6%）
- 改动：①SSA trace 修复（trace 边名从 id() 内存地址改唯一编号，消灭 64.8% 多写边的错喂，占旧误差 78%）；②头/行错排修复（`stack(qs,1).view` → `stack().reshape`，6 处）；③conv im2col padding 一行修（`padding=pad` → `padding=(0,pad)`，A 矩阵 2/3 零行消除，动作出口+自回归放大）。
- 关键数字：修复链 SSA 0.2486/0.2571 → stackfix 0.2450 → convfix **0.04998/0.0622**；148 个旗标全部归因到 3 个根因（无第四类），修后 24→0；公平性门全程不变（fp 回退 27 / 缺输入 59 / 段数 3118）。
- 负结果同日定案：JG S×1.5 = 打乱参照伪影（对正确逐头积 α=0.998~0.999）；k 侧 1.07 = 文本路径量化深度（b5b 同指纹，在天花板内）；input_layers 簇 = 纯继承。
- 本地：[round_report_routing_bug/2026-09-08_1252_接线bug破案与修复进展.html](../round_report_routing_bug/)、[hb_fpga_impl/WORKLOG.md](../hb_fpga_impl/WORKLOG.md) 0908 节。
- 服务器：/tmp/alg_fix/（REPORT_FIX.md §1~§15、results/fix_summary.json）、/tmp/alg_actv/（B 梯 consolidated.json：B5b=0.04247 语义天花板）、/tmp/alg_attr/、/tmp/main_verify/（基线复跑）。
- 重要事实：canonical 老链（0.1922/0.13991 锚点）同样带 bug ②③——那是带病成绩单；B5b 0.04247 是无 bug 口径天花板。

## v3 —— 2026-09-07 02:40 attention 占位常数换实测 σ_S：0.2383→0.1922 / 0.1828→0.1399（−19.3% / −23.5%）
- 改动：compiler.py L1139/L1163 的占位标定常数（S/PV 反量化）换实测 absmax，经 `--attn-calib` 数据通道上链（工具链零代码改动）。
- 关键数字：四参照梯定案 8 位格式无罪（同位置同刻度浮点执行 0.0208/0.0295，低于判据）；去噪迭代证伪放大（传导比<0.5、状态 int8 仅 4.7e-05）。
- 本地：[hb_fpga_impl/26_ref_denoise/REPORT_{REFQ,DENOISE,ATTENTION}.md](../hb_fpga_impl/26_ref_denoise/)、round_report_ref_denoise/2026-09-07_*.html。
- 服务器：/tmp/alg_refq/、/tmp/alg_denoise/。

## v2 —— 2026-09-04 13:30 pcW+RTN 全深度 + bias 三连修：0.2993→0.2383 / 0.2108→0.1828（−20.3% / −15.6% vs 逐 tensor 基线）
- 改动：逐输出通道 INT8 权重 + requant 就近舍入移植进全深度链；修注意力 qkv 偏置静默丢失（逐通道 K+1 增广）；深度二分定位主跳变 N=200~250。
- 本地：[hb_fpga_impl/24_pcw_rtn/](../hb_fpga_impl/24_pcw_rtn/)（REPORT_SW/REPORT_GATE/REPORT_RTL.md）、round_report_pcw_rtn/2026-09-04_1331_*.html。
- 服务器：/tmp/pcw_rtn/。

## v1 —— 2026-08-31 13:5x 全深度 W8A8 部署定界：0.2993 / 0.2108（红灯，链路本身零误差）
- 结论：三门全过（ΣSTORE 对账、RTL vs 黄金位精确、段输出 0 错）、五步定界确认是量化方案不是实现 bug；输出坍缩形态（动态关节方差归零）。
- 本地：hb_fpga_impl/03_compiler/NOTES.txt 八节、WORKLOG 08-31 节。
- 服务器（已清理，产物本地有）：build_full_v3 / build_s000_v3 / build_s001_v3。

## 未占号条目（负结果与过程轮）
- 2026-09-04 16:56 **算法三线全证伪**（25_alg）：真实样本标定最好 −2.9%（粗步长比削顶贵）；SmoothQuant 机制生效误差不动（sa 被最大通道钉死）；N=809"回拉"翻案为输出坍缩。→ hb_fpga_impl/25_alg_{calib,smooth,head}/、round_report_alg_ptq/。
- 2026-09-04 11:27 **0904 大翻案**：0903"根因=逐 tensor 权重量化 / pcW −75% / 激活侧全无效"三结论全部作废（ab_quant.py 单位 bug 测在坏基线上）。→ hb_fpga_impl/24_pcw_rtn/REPORT_GATE.md。
- 2026-09-02 16:57 research_w8a8_error 首轮根因报告（后被上面翻案覆盖，脚本保留）。→ research_w8a8_error/。
- 史前：2026-08-30 V100 量化门（W8A8 绿灯 0.0110，浅深度口径，不能外推到全深链）；SwiftVLA 时代（round_report_hb_quant 之前）。
