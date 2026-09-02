# PPA — Vivado 2021.2 逻辑综合（ae_top，COLS=108，250 MHz 目标）

器件 xczu7ev-ffvc1156-2-e（ZCU104）；`synth_design -directive RuntimeOptimized`；
综合后（未布局布线）时序。板容量：LUT 230400 / FF 460800 / BRAM36 312 / URAM 96 / DSP 1728。

| 资源 | 用量 | 占用率 |
|---|---|---|
| LUT | 110465 | 47.9% |
| LUTRAM | 1952 | — |
| FF | 112260 | 24.4% |
| BRAM | 122.5 | 39.3% |
| URAM | 64 | 66.7% |
| DSP | 1728 | 100.0% |

WNS = -1.038 ns @ 4.000 ns 时钟（TNS = -605.934 ns，违例路径 864 条，建立）；保持 WHS = 1.3 ns。
综合阶段（未布局布线、慢工艺角）保守 Fmax ≈ 198.5 MHz；最差路径 = copy 引擎 16→108 lane 重排交叉矩阵（纯 LUT，19 级），布局布线 + phys_opt 后有望进一步收敛。

CTX（2 MB，131072×128b SDP）推断为 URAM 级联（cascade height 8）；WRAM = 108 lane × 4096×8 BRAM；1728 DSP = 16×108 MAC 阵列，requant 32×16 乘法与 softmax 乘法走 LUT（use_dsp=no）。
