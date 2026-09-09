# 24_pcw_rtn RTL 线报告：逐通道权重量化（pcW）+ RTN 就近舍入

生成时间：2026-09-04 11:04:38
作者：RTL 线（24_pcw_rtn 轮）
任务书：`hb_fpga_impl/24_pcw_rtn/BRIEF_RTL.md`
上游依据：`research_w8a8_error/REPORT.md` §4 方案 A
基线：22_r3c_rtl（R3C 已交付冻结，本轮零改动，全部在副本上做）

一句话总结：三处 RTL 改动全部落地并验证——floor 模式与 R3C 基线**逐字节一致**（含芯片级终态对拍），RTN/逐列系数模式与软件口径的黄金模型**位精确一致**，回归 17 项全绿。

---

## 1. 这轮解决什么问题

R3C 交付后，全深度硬件链的量化误差是 0.29 rad，超部署判据（0.045 rad）6.6 倍，根因是逐层累积的 requant 误差。research_w8a8_error 的 §4 给出方案 A：权重改逐通道量化（pcW），requant 从截断改就近舍入（RTN），预期把真实样本误差从 0.189 降到 0.047（-75%）。

方案 A 的约束是**不碰脉动阵列、不碰累加器、不碰时序关键路径**——所有改动都落在 requant 单元、系数装载和描述符格式上。本轮 RTL 线把这个方案做进 RTL 并验证。

## 2. 三处 RTL 改动

改动文件都在 `24_pcw_rtn/rtl/`（从 22_r3c_rtl 复制后修改，R3C 原件未动）。

### 2.1 rq_v2.sv：加 RTN 舍入常数（2026-09-04）

原来 requant 是截断：`y = sat8(sum >>> t)`，其中 `sum = x·mh + ((x·ml)>>>8)`（m 拆高低字节后的精确分解，PW=35b）。截断对负数系统性偏大，是量化误差的一个固定来源。

现在加一个常数再移位：`y = sat8((sum + rn) >>> t)`，`rn = 2^(t-1) = 2^(s-9)`（s≥9 时）。

整数恒等式（两头独立验过，见 §4.2）：`p = 256·sum + r, r∈[0,256)`，则 `floor((p + 2^(s-1)) / 2^s)` 与 `(sum + 2^(s-9)) >>> (s-8)` 逐位相等，负数同样成立。也就是说 RTL 里的"加半再移位"和软件公式 `(x·m + 2^(s-1)) >> s` 是同一个数。

实现要点：

- `rn` 在 T0 拍组合译码成 one-hot（`rn_c[t-1] = 1`），随乘积一起寄存进 `rn_r`，T1 拍才参与加法——**没有新增时序级数**，加法器本来就在。
- 加法容器从 35b 加宽到 36b（`logic signed [PW:0]`）：sum∈(−2^34, 2^34)，rn≤2^34，加起来不回卷。符号扩展用 `$signed({1'b0, rn_r})`，裸拼接会把加法变无符号移位出错（这条在代码注释里有标记）。
- `t ≥ PW` 时 rn 钳位到 2^(PW-1)：两边移位后都是 0，无损。
- `rn_en=0` 时 rn 恒 0，**位精确回到 R3C 行为**——这是回归锚。
- `s=8`（t=0）时 rn=0，RTN 退化回截断。编译器侧契约：要 RTN 的列 s 必须 ≥9（REPORT §4 也建议把 s=8 的列编到 s=9）。
- T_MAX=0 配置（s=8 专用核）不支持 RTN，rn_en 被忽略。

### 2.2 rq_ms.sv：系数单输入改总线 + per-slot 选择（2026-09-04）

原来 rq_ms 的 m/s 是每 GEMM 一组单输入，24 个 slot 共享。现在改成总线 + 槽内选择：

```systemverilog
assign m_sel = $signed(m_bus[slot*16 +: 16]);
assign s_sel =        s_bus[slot*8  +: 8];
```

x 本来就是这么做的（`x_bus[slot*XW +: XW]`），m/s 照抄同一模式。选择器是纯组合 mux，不在脉动阵列路径上。`rn_en` 是每组共享（每 GEMM 一个舍入开关，不是每列——见 §6 分叉说明）。

### 2.3 ae_gemm.sv + ae_sched.sv + ae_core.sv：SF_INIT 载 96 组逐列系数（2026-09-04）

系数从哪来是这轮最大的结构改动。R3C 的 GEMM 描述符只占 1 个字，requant 系数在头字里（全列共享一组）。现在每输出列一组 `{m[16b], s[8b]}`，96 列 = 2304 bit，头字装不下，所以：

**描述符格式**（GEMM 族 = op∉{3,4,5,6,15} 的算子）：

- 头字：沿用 R3C 布局，空闲的第 28 位给 `rn_en`（dma_addr 在 [60:29]，不冲突）。
- 头字之后跟 `NCW = ceil(COLS·24/256)` 个 256b 系数字：COLS=96 时 9 个字，COLS=12（仿真）时 2 个字。
- 系数平铺成 blob：列 c 的 s 在 `blob[c*24 +: 8]`，m（有符号 Q8.8）在 `blob[c*24+8 +: 16]`，字 w 取 `blob[w*256 +: 256]`。

**取指泵**（ae_sched.sv）：FSM 加两个状态 T_CFETCH/T_CLATCH，T_LATCH 认出 GEMM 头字后逐字泵系数（每字 2 拍），锁进 `coeff_r[NCW×256b]`，经 `coeff_o` 总线送 ae_gemm。pc 步进对 GEMM 族改推 `1+NCW`（T_ADV 和预取的 pc_next 同式）。被 skip 的 GEMM 同样泵——省不掉但无害，语义简单。

**ae_gemm.sv**：SF_INIT 从锁 1 组改锁 `rq_coeff_r[COLS*24-1:0]`，generate 循环里把列 `gq*RQ_SH+gc` 的系数接到第 gq 个 rq_ms 的总线上。列映射闭环：列 c 的系数处理列 c 的累加器、写进 tile_buf 的列 c，两端同源不会错位。

**预取安全性**：预取 FSM 只看 pc_next 单条描述符，pc_next 已经跳过系数字，所以预取永远不会再把系数字当头字解析；PF_RD 只在 pf_win=T_RUN_G/SM/A 期间进入，与 T_CFETCH 时序不相交。这两点论证写在 ae_sched.sv 头注释里。

## 3. 描述符流和拍数的代价

- 描述符变长：仿真 actv 用例 seq.mem 从 40 字涨到 70 字（15 个 GEMM × 2 系数字，+75%）；硬件 COLS=96 时每个 GEMM 多 9 个字（+288 B/tile），和 REPORT §4 的预估完全一致。
- 取指拍数：每个被取指的 GEMM 多 2×NCW 拍。仿真默认用例每遍取指 27 个 GEMM 族描述符 → +108 拍。
- 实测（默认用例，COLS=12）：REF 总拍 6814→6930（+1.7%），其中 +108 是泵的算术值，+6/+2 是引擎相位抖动；PRIM 总拍 6618→6702（+84 拍，泵 +108、GEMM 引擎 −4、DMA −20——泵插入改变了引擎与预取的相位对齐，幅度 <0.5%，和 R3C 自身 pf 档间抖动同类）。MAC 计数逐项精确一致（86016/82944/skip 2048）。
- 真实帧量级：COLS=96 时每 GEMM 泵 18 拍，相对 GEMM 本体的 K 维流水时长可忽略。

## 4. 验证

### 4.1 回归全绿（2026-09-04 11:02，run-4，约 5 分钟）

`cd 24_pcw_rtn/sim && bash regression.sh` → **ALL PASS（17 项）**：gen_vectors / sim_ae / compare / gen_rq / tb_rq / tb_sm16 / tb_pe_pack / tb_pe_pack_dsp / gem_cycles / gen_actv / sim_ae_actv / compare_actv / **gen_lut（本轮新增）** / tb_ae_actv / gen_pcw / sim_ae_pcw / compare_pcw。

### 4.2 RTN 语义位精确（tb_rq 相位 D，2026-09-04）

60000 向量对 python 神谕（神谕公式 = 软件线公式 `(x·m + 2^(s-1)) >> s`，含 sat8），**err=0**。覆盖：s∈[21,27]（HB 真实标定域）30000 行、全域 [9,47] 15000 行、s=8 退化 7500 行、小 s [9,11] 7500 行；饱和命中 36288 行（舍入与饱和同拍撞界）。t≥PW 钳位分支被 s∈[35,47] 的随机行覆盖。

反向验证：软件线在 WORKLOG（2026-09-04）记录了 2 万随机含负数边界的同公式检查，同样全绿。**两侧从各自方向独立确认了同一整数语义**。

floor 锚（相位 A/B）：60000+30000 向量对 rq_v1 硬件神谕，err=0。

### 4.3 芯片级 floor 锚：与 R3C 存档逐字节一致（2026-09-04 11:00）

用我的 RTL 重跑 actv 用例，CTX/DDR 四个终态 dump（dump_ctx_ref/prim、dump_ddr_ref/prim，共 64KB+256KB×2 模式）与 R3C 交付存档 **cmp 逐字节一致**，ddr_init 输入也一致。这是最强形式的回归锚：rn_en=0 时新 RTL 的整芯片行为和 R3C 完全相同。

### 4.4 pcW+RTN 全链路位精确（sim_ae_pcw）

新用例（`gen_vectors.py --case pcw`）：每个 GEMM 独立抽 12 组逐列 (m,s)（s 混 HB 标定域 [21,27]、s=8、小 s 边界；|m|∈[16424,32754] 带符号），每 GEMM 掷 RTN/floor。结果：CTX/DDR 全部位平面对黄金模型位精确（compare_pcw PASS），REF==PRIM，prim2==prim（预取不改变结果）。默认用例（全列同 (m,s)、rn_en=0）同样位精确——即新描述符格式 + 泵在无逐列系数时也不改变行为。

### 4.5 黄金模型对账（gem_cycles smoke）

模型 vs R3C 实测基准：GEMM 引擎偏差 −0.53%/−0.65%，DMA +0.56%/−2.46%，MAC 精确命中；"系数泵"作为独立科目入账（108 拍），对账口径注释里写明实测基准是 R3C 旧格式、差值≈泵开销属预期。

### 4.6 回归基础设施的三处修复（本轮踩坑实录）

1. **tb_rq 空载假 PASS**：R3C 交接版 regression.sh 漏跑向量生成器，且生成器写 `ctrl.mem`、TB 读 `rq_ctrl.mem`（R3C 靠手工改名凑合）。已对齐文件名、补生成步骤、TB 加 `$fatal` 空载守卫（tb_sm16 同）。
2. **向量域 bug**：x 分布的 +2^26 上界越出 27b 有符号域，相位 A/B 对硬件神谕双侧同错抵消，相位 D 对 python 神谕才暴露（err=1555）。已修（上界取 2^26−1）。
3. **tb_ae_actv 全 X**：ae_actv.sv 运行期 `$readmemh("rsqrt_lut.mem")`——NORM 子模式的 rsqrt 查找表，属"输入孤本"（`spec/norm_gold.py --dump-lut` 生成），拷目录时被"*.mem 是生成物"规则误伤。缺失时表全 X、NORM 输出全 X。已显式加 `gen_lut` 回归步骤，表内容与 R3C 孤本逐字节一致（生成器复现）。exp2_lut.mem 同类，由 gen_vectors.py 再生成，已确认与 R3C 一致。

## 5. 资源影响与 REPORT §4 预估对照

| 项 | §4 预估 | 本轮实现 | 对齐说明 |
|---|---|---|---|
| rq_v2 舍入加法器 | 约 24×35 LUT | 常数加进已有 35b 饱和加法路径，+1b 宽 + one-hot 常数 mux，rn_r 随流水寄存 | 同量级或更低（没有新增独立加法器，复用既有 sum 加法） |
| rq_ms 逐列系数 | 24 核共 2.3 kbit LUTRAM + slot mux | 系数放 FF（ae_gemm rq_coeff_r），slot mux 为纯 LUT 4:1（24×16b m + 24×8b s） | **偏离**：没用 LUTRAM，用寄存器。SF_INIT 是一次性锁存语义，不需要寻址逻辑；2304 FF 在 ZU7EV 上代价可接受 |
| SF_INIT 载 96 组 | 多载 96 组系数 | rq_coeff_r 2304 FF（COLS=96） | 一致，这是主导成本 |
| 调度器（§4 未单列） | — | coeff_r 9×256b=2304 FF + 2 个 FSM 状态 + 5b 计数器 | 泵在取指通路，不在数据通路 |
| 描述符 | 每 tile 多约 9 个 256b 字 | 实测 COLS=96 恰 9 字（288 B/tile），actv 仿真用例 40→70 字 | 与预估完全一致 |

本轮未跑综合（Vivado 流程不在范围），资源数字是结构级对照；时序上改动都不在脉动阵列/累加器路径，rq_v2 没加流水级。**建议下轮跑一次 synth 对拍 R3C 的 LUT/FF 基线。**

## 6. 与软件线的语义对齐（含一处已裁决分叉）

**RTN 整数语义：对齐，双方向确认**（§4.2）。这是"两侧语义不一致=白干"里最要命的一条，已经闭环。

**描述符编码：两侧设计不同，本轮按任务书走 RTL 侧格式**。钢人论证后裁决如下：

- 软件线（BRIEF_SW 口径）：op=14 OP_SF 描述符，每字 10 个 24b 槽（`m<<8|s`），槽号在字内 [251:240]，逐列开关 rq_s/rq_m 放标志位。
- RTL 线（BRIEF_RTL + REPORT §4 口径，本轮实现）：GEMM 头字 + NCW 个平铺系数字，rn_en 在头字 bit 28。96 列 9 字（软件格式同容量要 10 字）。
- 选 RTL 侧的理由：任务书明文授权且格式与 §4 的"描述符每 tile 多约 9 字"预估精确吻合；GEMM 描述符原子自包含（系数跟描述符走，host 不会写散）；不占 op 编码空间。
- **收敛建议**（给合并轮）：两种编码可以共存——host_plan 侧保留 OP_SF（软件线内部表示），装板前由 host_driver 翻译成 RTL 平铺格式；翻译是纯打包变换，无语义歧义。rn 的粒度 RTL 侧是每 GEMM 一个开关，若标定后需要"同一 GEMM 内部分列 RTN/部分截断"，把 rn_en 也下放到槽内即可（需扩槽宽，本轮不做，REPORT §4 的标定方案也没要求）。
- 软件线的 REPORT_SW.md 是其交付物（截至本报告生成时尚未落盘）；本报告的语义对齐结论基于 BRIEF_SW + 双向数值验证，REPORT_SW.md 落盘后以其最终口径复核一遍即可（公式层面不会再变）。

## 7. 复现命令

```bash
# 全量回归（约 5 分钟，ALL PASS 17 项）
cd hb_fpga_impl/24_pcw_rtn/sim && bash regression.sh

# 只看 RTN 门（60k 向量 vs python 神谕）
cd hb_fpga_impl/24_pcw_rtn/sim && python gen_rq_vec.py && \
  iverilog -g2012 -o reg_rq.vvp tb_rq.sv ../rtl/rq_v1.sv ../rtl/rq_v2.sv \
    ../rtl/rq_ms.sv ../rtl/rq_m6.sv && vvp reg_rq.vvp

# pcW+RTN 全链路（逐列系数 + 逐 GEMM RTN 掷签，位精确比对）
cd hb_fpga_impl/24_pcw_rtn/sim && python gen_vectors.py --case pcw && \
  iverilog -g2012 -o reg_ae_pcw.vvp -I ../rtl <RTL 文件清单见 regression.sh> tb_ae.sv && \
  vvp reg_ae_pcw.vvp && python compare.py

# 芯片级 floor 锚（actv 用例，dump 与 R3C 存档 cmp）
# 见 §4.3；在干净目录用本 rtl/ 编译 tb_ae.sv，diff 22_r3c_rtl/sim/dump_*.mem
```

环境：iverilog 在 /c/iverilog/bin（regression.sh 自己加 PATH）；tb_pe_pack_dsp 需要 D:/software/Vivado/2021.2 的 unisim（路径写死在脚本里）。

## 8. 诚实边界

1. **COLS=96 只做了结构验证**：NCW=9、96 组系数的 generate 和泵逻辑是参数化的，本轮全部仿真在 COLS=12（NCW=2）跑。96 列的位精确验证需要 COLS=96 的向量与仿真（拍数 ×8，本轮 10 分钟实验约束内放不下），留给下轮。
2. **描述符编码与软件线分叉未合并**（§6）：合并前两侧不能直接共 host_plan，翻译层建议已给出。
3. **老格式向量全部失效**：tb_ae_seg / 服务器 seg 向量（run_segs.py 系）还是 R3C 描述符格式，用前要按新格式再生成。
4. **s=8 + rn_en=1 静默退化截断**：RTL 不报错，靠编译器契约保证 RTN 列 s≥9。若要硬保证，可在编译器侧断言（软件线责任）。
5. **未跑综合**（§5）：资源/时序结论是结构级论证 + §4 预估对照。
6. **skip 的 GEMM 也泵系数**：每帧浪费 2×NCW×skip 数拍（量级可忽略），换取取指通路语义简单。优化空间留档。
