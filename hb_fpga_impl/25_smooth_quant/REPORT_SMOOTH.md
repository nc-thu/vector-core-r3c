# REPORT_SMOOTH — SmoothQuant 等价变换（LN 折叠）：机制生效，误差不动

生成：2026-09-04 16:40 ｜ 实验目录：服务器 /tmp/alg_smooth/ ｜ 本地脚本：hb_fpga_impl/25_smooth_quant/sw/
（结果数字主会话已从 result json 逐个复核）

## 结论

**SmoothQuant 式 LN 折叠在 HoloBrain-0 W8A8 部署链上没有降误差。**

| 配置 | s000 | s001 | fp 等价门 |
|---|---|---|---|
| 基线 pcW v3 | 0.238341 | 0.182750 | — |
| S5：robot_encoder 10 组 | 0.238131（−0.09%） | 0.182595（−0.08%） | 6.6e-08 PASS |
| S6：全域 102 组 | 0.235129（−1.35%） | 0.180371（−1.30%） | 0.006431 PASS |

判据 0.045 全红，变化在噪声量级（S6 的 −1.3% 双样本一致）。但变换机制上生效：通道比值中位 3.3→1.9（S5）/4.7→2.2（S6），通道超标率 ~2%→0%。**压平发生、误差不动 ⇒ 部署误差主体不在激活逐通道动态范围**，推翻 RPTQ/FQ-ViT 型主因假设在本链的占比，与 0904 根因（GEMM int8 逐级 requant 复利 + 输出头回拉）一致。下一轮预算建议投向输出头回拉机制（bisect N=809→816 的 0.44 差值段）或逐级 requant 累积拆解，激活分布整形不再作主线。

## 方法（v1：LN 生产者折叠，零 RTL）

数学：y = x·W 改写为 y = (x/s)·(s·W)。对输入来自 LayerNorm/RMSNorm 的目标 Linear 组：γ'=γ/s、β'=β/s（RMSNorm 无 β）；W'=W·diag(s)（按输入通道右乘）后重新走 pcW 导出；**bias 不动**，增广偏置整数全部经 mk_pcw_calib.py v3 流程整表重算（未手改表）。s_i = Xmax_i^α / Wmax_i^(1−α)，α=0.5，除以几何均值，clamp [2⁻⁶,2⁶]；X 来自真实 s000 批 hook（seed 20260830 与 fp32_ref 同协议）。平滑层 sa 用 v2 同款合成扰动流（n_cal=8）在修改版模型上重收、拼回 v2 表（hw_calib_table_v2mod_*.json）再进 mk_pcw_calib；其余环节（RTN、conv、豁免表）不动。

多消费者检查两级：probe（真实前向 data_ptr 跟张量流；Linear 输入的模块级生产者是普通 LN 且该 LN 全部模块消费者同组 → 102 组全过，0 组不纯）+ allowlist（robot_encoder joint_self_attn 前有通道不变的 permute+flatten，源码核实 op 顺序 [norm, joint_self_attn, None, None, norm, ffn]×4+[norm] 后放行 4 组）。S5=robot_encoder 10 组（#190–214 全覆盖）；S6=S5+decoder.layers 12 组 q + enhancer 26 组 + backbone FFN/downsample 28 组 + BERT 段 + input_layers.5（共 157 成员层）。

## 时间线（全真实执行）

14:46 probe（421 层 hook，102 组）→ 14:51 s 归一化修正（首版几何均值公式 bug，通道比值不受影响）→ 14:53 fp 门 S5=6.6e-08 → 15:09 fp 门 S6=0.0064 → 15:22 S5 导出+重标定（18 层 sa，比例 0.76–1.11）→ 15:27 S5 表+编译+自检全绿 → 15:43 S5 s000=0.238131 → 15:35 S6 导出（157 层 sa，比例 0.11–1.98）→ 15:56 S6 s000=0.235129 → 16:08/16:14 s001：S6=0.180371、S5=0.182595。

## 为什么无效（T5 机制诊断）

1. **压平是真的**：目标 Linear 输入逐通道 absmax 比值（max/median）S5 中位 3.3→1.9（最大 8.2→3.8）、S6 中位 4.7→2.2；实测与预测逐组吻合（input_fc.4 8.2→3.8，pred 3.8）。
2. **饱和清零也是真的**：对部署 sa 的通道超标率 S5 中位 1.6%→0%、S6 最大 6.2%→≤1.2%。
3. **但 sa 几乎没变**（S5 比例 0.76–1.11）：per-tensor 静态 scale 由最大通道决定，压平小通道不动最大通道，量化分辨率没变，收益兑换不出来。
4. **权重侧代价付了**：成员层 swc 通道比值 S5 中位 2.3（input_fc.5 从 4.4×升到 7.0×）、S6 中位 2.9（backbone.stages.2.downsample.reduction 12.9×→26.3×）。
5. S6 fp 地板 0.0064：probe 看不到 functional 加法消费者（如 mmdet 层 query+query_pos 后进 self_attn），BERT 段 after 实测 28.2 vs pred 53.9 证实漂移，S6 的 −1.3% 归因要打折；S5 结构经源码核实无此问题（fp 门 1e-8）。
6. **跳过的 LN**：不可折叠（输入非直接来自 LN）而跳过的：backbone w_msa qkv（窗口 reshape）、enhancer img/text attn qkv（query+pos functional 加法）、MSDeform 四投影、out_proj、FFN 第二层、decoder.layers FFN（scale_shift 调制后）、t_embed 主体。α=0.75 按"无效即砍"约定砍掉（计划已生成 /tmp/alg_smooth/probe_out_alpha75.json 未执行）。

## 诚实边界

1. 判据未过；−0.1%/−1.3% 不能声称有效或有害。
2. 折叠只覆盖"输入直接来自 LN"的 Linear（157/426 量化层）；bisect 主跳变段（#190–250）可折叠部分已由 S5 全覆盖。
3. X 统计单样本 s000；sa 数值走合成流（与 v2 同源，保单变量可比）。
4. S6 数字混入 0.0064 fp 地板。
5. s 的 clamp 触发未逐组审计（不影响等价性）。
6. 全程 CPU fast_interp 仿真，RTL 零改动零验证（按设计）。

## 复现命令（<SERVER>，完整可粘贴）

```bash
PY=~/.conda/envs/holobrain/bin/python
A=/tmp/alg_smooth          # 依赖 /tmp/pcw_rtn/sw + manifest/w8_full/fp_biases + /tmp/ae_hostdrv 数据（均只读未动）
# 1) probe+统计+计划（~4 min）
cd ~/workspace/holobrain && CUDA_VISIBLE_DEVICES= $PY $A/smooth_probe_collect.py \
    --batch /tmp/ae_hostdrv/batch_s000.pt --v2 /tmp/ae_hostdrv/hw_calib_table_v2.json \
    --out $A/probe_out.json --alpha 0.5
# 2) s 归一化修正（离线）：s_new=s_raw/geomean，clamp，重算 ratio_after_pred → probe_out_fixed.json
# 3) 改 state_dict
CUDA_VISIBLE_DEVICES= $PY $A/smooth_build_sd.py --plan $A/probe_out_fixed.json --scope s5 \
    --out $A/smoothed_sd_s5.pt --edits $A/edits_s5.json        # s6 换 --scope s6
# 4) T1 fp 门（<0.01 才许往下走）
CUDA_VISIBLE_DEVICES= $PY $A/smooth_fp_gate.py --sd $A/smoothed_sd_s5.pt \
    --plan $A/probe_out_fixed.json --scope s5 --gate-json $A/fpgate_s5.json --stats $A/stats_after_s5.json
# 5) 导出+重标定（pcW/fp 偏置核对/sa 拼 v2mod）
cd ~/workspace/holobrain && CUDA_VISIBLE_DEVICES= $PY $A/smooth_export.py \
    --sd $A/smoothed_sd_s5.pt --plan $A/probe_out_fixed.json --scope s5 --tag s5 --out-dir $A
# 6) mk_pcw_calib v3 整表 + 双样本编译 + fast_selftest
bash $A/stage2.sh s5
# 7) e2e（~9-11 min/样本）
cd $A && $PY sw/host_driver.py --build build_s000_s5 --trace /tmp/ae_hostdrv/trace_s000.json \
    --batch /tmp/ae_hostdrv/batch_s000.pt --ref /tmp/ae_hostdrv/fp32_ref_000.npz \
    --calib hw_calib_table_smooth_s5.json --sample-id 000 --out result_000_smooth_s5.npz
#    （s001 换 trace_s001/batch_s001/fp32_ref_001/build_s001_s5/sample-id 001）
# 8) α=0.75 变体：s75=s05·sqrt(xmax/s05)，计划已生成 $A/probe_out_alpha75.json，走 3)-7)（本轮未执行）
```

## 文件清单

- 服务器 /tmp/alg_smooth/（2.8G）：脚本（smooth_probe_collect / smooth_build_sd / smooth_fp_gate / smooth_export.py、stage2.sh、e2e.sh、sw/ 副本）；probe_out.json、probe_out_fixed.json（主用）、probe_out_alpha75.json、fpgate_s5/s6.json、stats_after_s5/s6.json、export_summary_s5/s6.json、edits_s5/s6.json、smoothed_sd_s5/s6.pt、hw_calib_table_v2mod_s5/s6.json、hw_calib_table_smooth_s5/s6.json、pcw_export_s5/s6/、pcw_scales_s5/s6.json、fp_biases_s5/s6.json、build_s000/s001_s5/s6、result_000/001_smooth_s5/s6.npz + _vs_fp32.json、全部 run log。
- 本地：e:\GPU ARCH\vector_core_sim\hb_fpga_impl\25_smooth_quant\sw\（四个 smooth_*.py + stage2.sh + e2e.sh，已同步为最终修正版）。

**尾注**：fp_biases 与基线逐位一致（bias_diff=[]）、S5 fp 门 6.6e-08，说明 bias 纪律和折叠数学都干净；S6 的 0.0064 地板定位在 enhancer/BERT 段的 functional-add 消费者，若下一轮要复用 S6 范围，建议先给 probe 加"LN 输出进 functional 加法"的检测再放行。
