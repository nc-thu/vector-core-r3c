# hb_fpga_impl：HB-GD 上加速器——RTL 双发射 + 模型编译 + FPGA 实现级验证

本轮目标（2026-08-30 起）：

1. RTL 定版：调度器双发射（权重后台预取）+ WRAM 双缓冲，pf_en 开关保证 pf_en=0 与旧电路逐拍一致。
2. 服务器侧把 HB-GD 模型 + RoboTwin 官方数据集小子集编译成 runtime 二进制（指令流 + 权重 + 输入）。
3. 位精确 RTL 仿真跑真实模型：性能拍数 + 推理结果，与数据集真值和 fp32 参考对拍。
4. Vivado 综合功耗报告（OOC 级；bitstream/Vitis/上板按用户决定后置）。
5. HTML 报告 + 全程工作记录。

## 目录

| 目录 | 内容 |
|---|---|
| 01_rtl/ | RTL 双发射改动副本 + 回归记录（源头 hw_zcu104/ 不动） |
| 02_quant/ | 硬件口径量化校准（V100）：per-tensor 静态 scale + requant 常数表 |
| 03_compiler/ | 模型 → 描述符指令流 + 权重/输入二进制的编译器 |
| 04_dataset/ | RoboTwin 官方数据集小子集 + 输入样本提取 + 真值 |
| 05_sim/     | 位精确仿真 harness（TB 扩容、分段跑、拍数统计） |
| 06_power/   | 功耗流程（synth report_power + SAIF） |
| 07_eval/    | 推理结果 vs 真值对拍 |
| 08_report/  | 结果 HTML（新轮次新文件夹，不改旧文件） |

工作记录看 [WORKLOG.md](WORKLOG.md)。
