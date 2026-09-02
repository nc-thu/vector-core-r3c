# -*- coding: utf-8 -*-
# 2026-08-31 问题讨论页生成器：HTML 模板 + 内联 4 张新图 + est 阶梯图，时间戳取实际写入时刻
import datetime, io, os

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
fn = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")

html = u"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>实测与账本差 16 倍：拆成三笔账 · 2026-08-31</title>
<style>
:root{color-scheme:light}
*{box-sizing:border-box}
body{margin:0;background:#f9f9f7;color:#0b0b0b;font-family:system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;line-height:1.75;font-size:15px}
.wrap{max-width:920px;margin:0 auto;padding:28px 22px 80px}
header.page{border-bottom:1px solid #e1e0d9;padding-bottom:18px;margin-bottom:10px}
h1{font-size:24px;margin:0 0 10px;line-height:1.4}
.meta{font-size:13px;color:#52514e;display:flex;flex-wrap:wrap;gap:4px 20px}
nav.pagenav{margin:12px 0 0;font-size:13.5px}
nav.pagenav a{color:#2a78d6;text-decoration:none;margin-right:16px}
h2{font-size:19px;margin:40px 0 4px}
.hint{font-size:13px;color:#898781;margin:0 0 10px}
.entry{background:#fcfcfb;border:1px solid rgba(11,11,11,.10);border-radius:10px;padding:16px 20px;margin:14px 0}
p{margin:8px 0}
ul,ol{margin:8px 0;padding-left:22px}
li{margin:4px 0}
.tile-row{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}
.tile{flex:1 1 170px;background:#fcfcfb;border:1px solid rgba(11,11,11,.10);border-radius:10px;padding:12px 16px}
.tile .v{font-size:21px;font-weight:700}
.tile .k{font-size:12.5px;color:#52514e;margin-top:2px;line-height:1.5}
figure.chart{margin:18px 0;background:#fcfcfb;border:1px solid rgba(11,11,11,.10);border-radius:10px;padding:14px 18px 8px}
figure.chart h4{margin:2px 0 2px;font-size:15px}
figure.chart .how{font-size:13px;color:#52514e;margin:0 0 8px}
.figrow{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}
.figrow figure.chart{flex:1 1 400px;margin:0}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}
th,td{border-bottom:1px solid #e1e0d9;text-align:left;padding:6px 8px;vertical-align:top}
th{color:#52514e;font-weight:600;border-bottom:1px solid #c3c2b7}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.warn{border-left:3px solid #fab219;background:#fffdf3;padding:10px 14px;border-radius:0 8px 8px 0;margin:12px 0}
.ok{border-left:3px solid #3e9c5c;background:#f6fbf7;padding:10px 14px;border-radius:0 8px 8px 0;margin:12px 0}
#tip{position:fixed;opacity:0;pointer-events:none;background:#0b0b0b;color:#fff;font-size:12.5px;line-height:1.5;padding:6px 10px;border-radius:6px;max-width:280px;z-index:9}
[data-tip]{cursor:default}
svg path[data-tip]:hover,svg [data-tip]:hover{filter:brightness(.92)}
footer{margin-top:44px;padding-top:14px;border-top:1px solid #e1e0d9;font-size:12.5px;color:#898781}
code{background:#f0efec;border-radius:4px;padding:1px 5px;font-size:12.5px}
.running{display:inline-block;font-size:12px;font-weight:600;color:#8a6100;background:#fff4dd;border-radius:5px;padding:1px 8px;vertical-align:2px;white-space:nowrap}
</style>
</head>
<body>
<div id="tip"></div>
<div class="wrap">

<header class="page">
<h1>实测 5.44 秒 vs 账本 0.27 秒：16 倍差距拆成三笔账，两件事怎么办</h1>
<div class="meta">
<span>生成时刻：<b>{{NOW}}</b>（v3：10:30 初稿 + 10:49 并入 on-chip + 11:27 并入 INT4）</span>
<span>数据截至：<b>2026-08-31 10:17</b>（布线后资源账）</span>
<span>讨论对象：三份方向工作记录（硬件 / 软件 / 架构，2026-08-31）</span>
</div>
<nav class="pagenav">
<a href="2026-08-31_硬件工作记录.html">→ 硬件方向记录</a>
<a href="2026-08-31_软件工作记录.html">→ 软件方向记录</a>
<a href="2026-08-31_架构工作记录.html">→ 架构方向记录</a>
<a href="../WORKLOG.md">完整工作日志</a>
</nav>
</header>

<p>方向记录发出来后，里面"账本 0.27 秒、实测 5.44 秒"的对照引出两个疑问：① 工作量口径是不是本来就不一样；② v0 调度的搬运债到底多大、怎么还。这一页把 16 倍差距完整拆开，每个因子给证据，最后列出要拍板的五个决策点。三个后台调研里，<b>片上激活/归一化（第六节：净省 7.7% 拍数）和 INT4 权重量化（第七节：权重字节减 35.6%）已经到了</b>，端到端数值链还在跑。</p>

<div class="ok">
<p style="margin:4px 0"><b>三句话结论。</b></p>
<p style="margin:4px 0">① 16.0 倍 = <b>1.98 × 1.53 × 5.27</b>：工作量口径差一倍、矩阵补零多算一半、剩下 5.27 倍是搬运和停顿把阵列饿着了。</p>
<p style="margin:4px 0">② 前两笔<b>不是硬件变慢</b>：1.98 是这次跑的任务比账本假设的大（4 个相机 vs 账本按 2 个视角算），1.53 是补零的算术代价。真正要修的只有第三笔。</p>
<p style="margin:4px 0">③ 估算从 3.1 亿拍涨到 10.8 亿拍，是<b>两个估算 bug 修掉后估算变准了</b>，不是硬件退化（第四节）。</p>
</div>

<h2>一、16 倍怎么拆：三个乘法因子，各归各的账</h2>
<p class="hint">数据：Verilator 全模型实测（441 类型代表段，部署口径），研究阶段账本 research_holobrain/profile/holobrain_hw.json。</p>
<div class="entry">
<p><b>先说 0.27 秒是怎么来的。</b>研究阶段做账本时算出一次推理 73.7 GMAC（十亿次乘加）。芯片峰值算力是 432 GMAC/s（1728 个乘法器 × 250 MHz）。账本假设阵列 63% 的时间在算，即有效算力 273 GMAC/s，73.7 ÷ 273 = 0.27 秒。</p>
<p><b>实测是 4.32 秒</b>（250 MHz 口径；现在时序 198.5 MHz，所以实际是 5.44 秒）。4.32 ÷ 0.27 = 16.0 倍。这 16 倍可以拆成三个因子相乘，每个因子都能单独验证：</p>
<figure class="chart">
<h4>从账本到实测：每一跳差在哪</h4>
<p class="how">怎么看：四根柱子从左到右是"账本估计 → 换成实测工作量 → 加上补零 → 加上搬运停顿"。柱子上的 × 就是上一跳的倍数，三跳连乘 = 16.0。鼠标悬停看每一步的算法。</p>
{{SVG:qa_gap}}
</figure>
<table>
<tr><th>因子</th><th class="num">倍数</th><th>一句话原因</th><th>定性</th></tr>
<tr><td>工作量口径</td><td class="num">×1.98</td><td>实测跑的是 4 相机任务，账本按 2 视角算——乘加正好差一倍</td><td>口径问题，不是硬件问题（第二节）</td></tr>
<tr><td>padding（补零）</td><td class="num">×1.53</td><td>矩阵边长不是 16 的倍数时补零，补的零也在占用乘法器</td><td>可优化但有下界（第三节）</td></tr>
<tr><td>搬运与停顿</td><td class="num">×5.27</td><td>阵列真正在算的时间只占 12%，账本假设的是 63%</td><td><b>真问题</b>（第三节）</td></tr>
</table>
<p>第三个因子的算法：带零乘加共 223.8 G，占满 4.32 秒 × 432 GMAC/s = 1866 G 个槽位里的 12%；63% ÷ 12% = 5.27。换句话说，<b>阵列 88% 的时间在等数据</b>——这就是搬运债的定量说法。</p>
</div>

<h2>二、问题①：两边说的"一次推理"不是同一次</h2>
<p class="hint">结论：1.98 倍几乎全部来自相机数（4 vs 2），不是模型变重、也不是硬件偷懒。</p>
<div class="entry">
<p><b>两个口径分别是什么。</b>账本是研究阶段给硬件评估用的模型统计：双视角输入、指令按 16 个 token 算，共 73.66 GMAC。实测这次跑的是 RoboTwin 任务配置：<b>4 个相机</b>、指令 tokenize 出来的 token 数不到 16、激活没有任何缓存。两边都叫"一次推理"，但任务配置不同。</p>
<p><b>逐科目对表</b>（实测取有效乘加，即不含补零的 useful MAC）：</p>
<table>
<tr><th>账本科目</th><th>对应实测阶段</th><th class="num">账本 GMAC</th><th class="num">实测 GMAC</th><th class="num">比值</th></tr>
<tr><td>动作头 ×10 去噪步</td><td>decoder</td><td class="num">23.41</td><td class="num">48.24</td><td class="num">2.06×</td></tr>
<tr><td>融合层 Fusion</td><td>feature_enhancer</td><td class="num">26.42</td><td class="num">52.73</td><td class="num">2.00×</td></tr>
<tr><td>视觉 2D</td><td>backbone + neck</td><td class="num">17.26</td><td class="num">33.93</td><td class="num">1.97×</td></tr>
<tr><td>视觉 3D + PSE</td><td>spatial_enhancer + backbone_3d + neck_3d</td><td class="num">5.20</td><td class="num">10.24</td><td class="num">1.97×</td></tr>
<tr><td>文本 BERT</td><td>text_encoder</td><td class="num">1.37</td><td class="num">0.68</td><td class="num">0.50×</td></tr>
<tr><td><b>总计</b></td><td></td><td class="num"><b>73.66</b></td><td class="num"><b>145.82</b></td><td class="num"><b>1.98×</b></td></tr>
</table>
<figure class="chart">
<h4>四个大科目全是约 2 倍——这是相机数的指纹</h4>
<p class="how">怎么看：每根条是"实测 ÷ 账本"。1.0 处的竖线是账本基准。四个大科目挤在 1.97–2.06，只有文本在 0.5——两个异常用同一个原因解释不了，但两个原因都很干净（见下）。</p>
{{SVG:qa_recon}}
</figure>
<p><b>为什么相机翻倍、乘加正好翻倍。</b>每个相机的画面独立过主干和融合，相机之间没有交叉注意力。所以乘加随相机数<b>线性</b>涨：4 ÷ 2 = 2 倍。四个互不相干的科目比值全落在 1.97–2.06，就是"线性扩展"这件事的指纹——如果是别的原因（比如模型改了、硬件多算了），很难四个科目一起这么齐。</p>
<p><b>文本为什么反而只有一半。</b>文本乘加只和 token 数成正比。账本按 16 token 算，实测这条指令 tokenize 出来大约 8 个，所以是 0.5 倍。这个科目只占总乘加不到 1%，不影响大局。</p>
<p><b>顺带澄清"BERT 重跑"。</b>账本其实也把 BERT 算在内（1.37 G），所以"每次推理重跑 BERT"<b>不在这 1.98 倍里</b>。它影响的是时间：重跑多花 3040 万拍，占总时间 2.8%——归第三节的搬运债管。</p>
<p><b>一处没对齐的科目，如实说明。</b>账本把 3D 相关计算合成一个"视觉 3D + PSE"科目（5.20 G），实测拆成了 spatial_enhancer、backbone_3d、neck_3d 三项（合计 10.24 G）。两边划法不完全一样，这一行的 1.97× 只作参考。它只占 7% 乘加，不动摇总结论。</p>
<p><b>把口径摘干净后的尺度感：</b>如果按账本口径（2 视角、短指令）重跑一遍，在效率不变的前提下，时间约为 4.32 ÷ 1.98 ≈ <b>2.2 秒</b>（250 MHz 口径）。这是除法推算，不是实测，也不是承诺——但说明口径修正后，剩下要解释的就是搬运债那一笔了。</p>
<p><b>三个选项，等你拍板：</b></p>
<table>
<tr><th>选项</th><th>做什么</th><th>成本</th><th>得到什么</th></tr>
<tr><td><b>A. 对照实验（推荐）</b></td><td>把 trace 配置改成 views=2，重跑一遍编译和测拍</td><td>约一天机器时间</td><td>和账本同口径的实测数字，顺带实证"线性扩展"假设</td></tr>
<tr><td>B. 只折算</td><td>所有数字按 1.98 除，不重跑</td><td>零</td><td>快，但"线性"停留在推断；四个科目比值很齐，风险不大</td></tr>
<tr><td>C. 双口径报告（推荐）</td><td>以后性能数字同时给"4 相机实测"和"每相机归一"两个口径</td><td>零（写法约定）</td><td>读的人自己选口径，不会再出现"0.27 对 5.44"的裸对照</td></tr>
</table>
<p>我的建议：<b>C 现在就做</b>（写报告的约定问题），<b>A 排进队列</b>（一次实验把口径钉死，还能当回归用例）。</p>
</div>

<h2>三、问题②：搬运债 5.27 倍——"先跑对"路线的已知代价</h2>
<p class="hint">结论：这笔债是架构决定（段独立）带来的。归因账已由 on-chip 调研给出（下表），不用再等计数器——按拍数排，STORE 写回是第一大项，COPY 只有 1.4%。</p>
<div class="entry">
<p><b>债是怎么欠下的。</b>为了让验证简单，我们把一次推理切成 2782 个独立的段，每段自带全套数据搬运：权重装进来、激活装进来、算完写回去、下一段重头再来。这个决定让整面验证墙垒得很稳（架构页第三节），代价就是搬运指令满天飞、阵列经常空转。架构页已经认了这笔账，这里把它拆细。</p>
<p><b>归因账怎么来的。</b>on-chip 调研（第六节）把全部 2782 段的指令流逐条解码，用 RTL 实测常数折算每段各占多少拍，加总 10.61 亿拍 vs 实测 10.81 亿（差 1.8%）——两套独立方法落在同一个数上，这套分解可信。原始账在 07_onchip_ops/boundary_account.json。</p>
<figure class="chart">
<h4>10.81 亿拍都花在哪（按拍数归因）</h4>
<p class="how">怎么看：矩阵乘本体只占 55%，剩下 45% 是搬运和停顿。上一版（10:30 初稿）这张图的位置是"等计数器"——现在归因账到了，直接填上。悬停看每一项的细节。</p>
{{SVG:qa_cycle_mix}}
</figure>
<p><b>五个根因，按拍数重排。</b>注意排序变了：初稿按指令条数排，COPY（5.4 万条）看着像大杠杆；按拍数算它只有 1.4%，<b>STORE 写回才是第一大单项</b>——这就是"先归因再动手"的意义：</p>
<table>
<tr><th>#</th><th>根因</th><th class="num">拍数（占比）</th><th>证据</th><th>对应的解法</th></tr>
<tr><td>1</td><td>STORE 写回慢</td><td class="num">206.6 M（19.1%）</td><td>19,410 条 · 660.9 MB · 折算只有 3.2 字节/拍</td><td>编译器加大段 + 流式写回——比任何算子融合都大，下一主战场</td></tr>
<tr><td>2</td><td>激活反复装</td><td class="num">164.9 M（15.3%）</td><td>LOAD_CTX 42,246 条 · 1.3 GB</td><td>on-chip 算子融合（第六节，方案已出：norm/actv/softmax 直接削这笔）</td></tr>
<tr><td>3</td><td>权重每段重装</td><td class="num">75.4 M（7.0%）</td><td>LOAD_W 15,494 条；预取已吃 33.7 M，还暴露约 42 M</td><td>热层驻留 + 预取窗口扩到 COPY/新引擎运行期（约 10 个 LUT，顺手修）</td></tr>
<tr><td>4</td><td>补零与形状匹配</td><td class="num">引擎忙时效率仅 22%</td><td>有效 145.8 G → 带零 223.8 G（+53%）</td><td>段合并拼 K（D3）</td></tr>
<tr><td>5</td><td>转置与停顿</td><td class="num">34.8 M（3.2%）</td><td>COPY 15.2 M + 调度/LFSR 19.6 M</td><td>COPY 编译期化降级为顺手项（拍数上界只有 1.4%）</td></tr>
</table>
<p><b>代表段的 DMA 忙时是直接证据</b>（哪一段 DMA 忙时占比高，哪一段阵列就在挨饿）：BERT 注意力段 97.7%（几乎全程在搬）、最重段 67.9%、decoder 高频段 64.6%；反过来计算密度高的段只有 13.3%。同一颗芯片，段与段之间差别这么大，说明瓶颈在调度和搬运，不在乘法器——乘法器数量一个没变。这张图在架构页第四节，这里不重复贴。</p>
<p><b>能立刻做的低风险事（更新后三件）：</b></p>
<ul>
<li><b>BERT 结果缓存</b>：同一条指令的文本特征算一次就够。省 3040 万拍，<b>总时间直接减 2.8%</b>，host 侧改几行。</li>
<li><b>预取窗口扩展</b>：预取目前只在 GEMM/SM 运行期发射，COPY 和新引擎运行时它闲着——把发射窗口扩进去，约 10 个 LUT，吃回 LOAD_W 暴露 42M 拍的一部分。</li>
<li><b>COPY 编译期化</b>：降级为顺手项。归因账显示 5.4 万条 COPY 只占 15.2M 拍（1.4%）；它的真实罪状是打断权重预取（上一条顺手治）。</li>
</ul>
<div class="ok">
<p style="margin:4px 0"><b>初稿的行动项"加 per-opcode 忙拍计数器"已经有了更好的答案：归因账直接做出来了。</b></p>
<p style="margin:4px 0">方法不是硬件计数器，而是把 2782 段指令流逐条解码、用 RTL 实测常数折算（上面那张图）。与 441 类实测对账差 1.8%，排序结论稳。Verilator 计数器降级为可选的验证工具；真要上板长期用，再考虑把它做进 u_sched。</p>
</div>
<p><b>和后台调研的关系。</b>on-chip 激活/归一化<b>方案已出</b>（第六节）：MVP 净省 82.8M 拍（7.7%），直接削根因 2 的账；INT4 权重量化让同样的 BRAM 装下更多权重（驻留更容易）或换成更宽的阵列（padding 比例下降），结果待回填。</p>
</div>

<h2>四、est 对账三级台阶：硬件没变慢，是估算变准了</h2>
<p class="hint">回应"修两次、涨两次"的疑问：两次修复修的都是估算器的 bug，不是硬件行为变了。</p>
<div class="entry">
<figure class="chart">
<h4>拍数估算 vs 实测：三级台阶，每级一个已修复的原因</h4>
<p class="how">怎么看：四根柱子是同一颗硬件、同一个模型的拍数。前两根是估算器带 bug 时的读数，第三根是 bug 修完的估算，第四根是 Verilator 实测。</p>
{{SVG:sw_est_ladder}}
</figure>
<p><b>第一级 313M → 544.5M（+74%）。</b>编译器算搬运长度时读错了字段的读法（dma_len 编码 bug，软件页第七条），很多搬运被低估。修的是<b>估算公式</b>，硬件每拍干的事没变。</p>
<p><b>第二级 544.5M → 854.8M（+57%）。</b>估算器算 GEMM 段数时把组数除了两次 16（ceil16(m) 又 //16），计算量口径被低估。同样是估算 bug。</p>
<p><b>第三级 854.8M → 1080.8M 实测（+26%）。</b>这一级不是 bug：是估算模型本来就不含的真开销——LFSR 随机数读的停顿、总线仲裁等待这些。它们一直在硬件里发生，只是以前没人算它们。</p>
<p><b>可复用的推论：</b>修完 bug 的估算（est_fixed）× 1.26 ≈ 实测。以后新配置可以用这个系数做快速预估，半天出数，不用等两天实测。注意这是单模型标定的系数，先当经验值用，别当物理常数。</p>
<p><b>第二套口径互证。</b>on-chip 调研的边界账模型（指令流解码 × RTL 常数）加总 10.61 亿拍，与实测差 1.8%——两套独立方法落在同一个数上，互为旁证。</p>
</div>

<h2>五、同一颗芯片：资源花在哪 × 时间花在哪</h2>
<p class="hint">新图表规范（2026-08-31 起）：硬件 Breakdown 必须同时给每个模块的 LUT/DSP/BRAM 占用和模型计算时间占比。这是第一版。</p>
<div class="entry">
<p><b>资源账的来源。</b>从布线后的设计检查点（DCP）跑 Vivado 层级报表（2026-08-31 10:17:33），逐模块提取。数字是布线后的真实占用，不是综合阶段的估算。</p>
<div class="figrow">
<figure class="chart">
<h4>每个模块占多少 LUT</h4>
<p class="how">怎么看：条子越长 LUT 越多。乘法阵列是最大的一块，但注意——LUT 排第二的 copy 重排网络和排第五的 softmax，干的是"伺候阵列"的活。</p>
{{SVG:qa_modules}}
</figure>
<figure class="chart">
<h4>模型各阶段的计算时间占比</h4>
<p class="how">怎么看：decoder 一个阶段吃掉一半时间。阵列（上一张图最大的模块）在所有阶段都是主要执行者，所以这张图也就是阵列的时间去向。模块级忙拍计数器还没有（第三节行动项），目前时间只能按模型阶段切。</p>
{{SVG:qa_time}}
</figure>
</div>
<table>
<tr><th>模块</th><th>干什么</th><th class="num">LUT</th><th class="num">FF</th><th class="num">DSP</th><th class="num">BRAM36</th><th class="num">URAM</th></tr>
<tr><td>u_arr（PE 阵列）</td><td>16×108 脉动阵列，所有矩阵乘法</td><td class="num">42,267</td><td class="num">87,544</td><td class="num">1,728</td><td class="num">–</td><td class="num">–</td></tr>
<tr><td>u_cp（copy 交叉）</td><td>转置/重排（swin 窗口、im2col）</td><td class="num">18,410</td><td class="num">1,180</td><td class="num">–</td><td class="num">–</td><td class="num">–</td></tr>
<tr><td>requant（27 套）</td><td>INT32 累加结果压回 INT8</td><td class="num">16,940</td><td class="num">≈3,970</td><td class="num">–</td><td class="num">–</td><td class="num">–</td></tr>
<tr><td>u_gemm 胶水</td><td>行缓冲 tile_buf + 排水控制</td><td class="num">12,117</td><td class="num">14,243</td><td class="num">–</td><td class="num">–</td><td class="num">–</td></tr>
<tr><td>u_sm（softmax）</td><td>16 条并行查表车道</td><td class="num">10,164</td><td class="num">2,541</td><td class="num">–</td><td class="num">–</td><td class="num">–</td></tr>
<tr><td>u_dma</td><td>DDR 和片上之间搬数据</td><td class="num">1,469</td><td class="num">1,672</td><td class="num">–</td><td class="num">–</td><td class="num">–</td></tr>
<tr><td>u_core 胶水</td><td>FIFO / 行缓冲</td><td class="num">1,224</td><td class="num">1</td><td class="num">–</td><td class="num">14（另 1 块 RAMB18）</td><td class="num">–</td></tr>
<tr><td>u_sched</td><td>读描述符、发指令</td><td class="num">999</td><td class="num">619</td><td class="num">–</td><td class="num">–</td><td class="num">–</td></tr>
<tr><td>u_ctx</td><td>激活上下文缓存（本体在 URAM）</td><td class="num">260</td><td class="num">6</td><td class="num">–</td><td class="num">–</td><td class="num">64</td></tr>
<tr><td>WRAM（g_w × 108）</td><td>当前层的权重 bank</td><td class="num">≈0</td><td class="num">–</td><td class="num">–</td><td class="num">108</td><td class="num">–</td></tr>
<tr><td><b>合计（u_core）</b></td><td></td><td class="num"><b>103,846</b></td><td class="num"><b>111,671</b></td><td class="num"><b>1,728</b></td><td class="num"><b>122</b></td><td class="num"><b>64</b></td></tr>
<tr><td><b>占整片 xczu7ev</b></td><td></td><td class="num"><b>45.1%</b></td><td></td><td class="num"><b>100%（满）</b></td><td class="num"><b>39.3%</b></td><td class="num"><b>66.7%</b></td></tr>
</table>
<p><b>三件事值得说：</b></p>
<ol>
<li><b>40.7% 的 LUT（u_arr）承担 100% 的矩阵乘</b>，时间上它是所有模型阶段的主角。资源大和时间大在这里是一致的。</li>
<li><b>u_dma 只占 1.4% 的 LUT，却是性能瓶颈</b>（第三节）。资源占比和性能瓶颈不是一回事——修瓶颈不需要花很多 LUT，这是好消息。</li>
<li><b>DSP 已经 100% 用满，LUT 还有 55% 余量。</b>想靠加列扩算力，DSP 先撞墙，出路只有"一个 DSP 算两个 INT8 乘"或更低位宽——这正是 INT4 调研的动机之一。</li>
</ol>
<p style="font-size:13px;color:#52514e">注：硬件页引的综合阶段 LUT 是 110,465（47.9%），本表布线后是 103,958（45.1%）——布线优化掉了约 6.5k。两页数字口径不同属正常，以布线后为准。</p>
</div>

<h2>六、调研结果①：片上激活/归一化（已到）<span class="date">2026-08-31 10:49 并入</span></h2>
<p class="hint">结论：值得做。MVP 方案净省 7.7% 总时间，约 16k LUT、7 块 BRAM、零 DSP——LUT 余量 60k 里只花四分之一。</p>
<div class="entry">
<p><b>这笔账是什么。</b>现在 host（CPU/ARM 侧）跑的 norm、gelu、softmax 这些算子，每一步都要"结果写回 DDR → host 算 → 下一段再装回来"。调研把全部 2782 段指令流解码后算出：挂在 host 算子上的段边界往返共 <b>148.8M 拍（占总时间 13.8%）</b>，其中 norm 一族（RMS/LN/Ada/GN 共 423 步）占 49%——同一个归一化结果还会被下游多个段反复装载 2.5 次，融合的红利比"省一次往返"更大。</p>
<figure class="chart">
<h4>每个算子族搬进片上各能净省多少拍</h4>
<p class="how">怎么看：条子是净省拍（已经扣掉新引擎自己跑的时间）。norm 最大但性价比不是最高；注意单项相加（108M）大于 MVP 合计（82.8M），因为 MVP 用严格口径——只算整条边界上全部算子都被覆盖而真正消失的往返。</p>
{{SVG:qa_onchip_gain}}
</figure>
<div class="tile-row">
<div class="tile"><div class="v">−7.7%</div><div class="k">MVP（norm+actv+softmax+rotary）净省 82.8M 拍 = 417 ms，5.44s → 5.03s；MAC 利用率 7.81% → 8.46%</div></div>
<div class="tile"><div class="v">−9.0%</div><div class="k">再加 swin 窗口散射引擎：净省 97.0M 拍 = 489 ms，5.44s → 4.96s；消灭 88% 的 host 边界</div></div>
<div class="tile"><div class="v">~16k LUT</div><div class="k">AE_ACTV 统一行引擎 12–15k + swin 散射 2.5k + 调度扩展 0.3k（含 50% 余量共 24.5k，预算 60k）；BRAM 7 块；DSP 零新增</div></div>
<div class="tile"><div class="v">1424 → 447</div><div class="k">host 算子步骤从 1424 步减到约 447 步（69% 被融合）；留下的多是 deform/PSE 这类专用引擎性价比最低的</div></div>
</div>
<p><b>方案怎么落地（照抄既有先例，不发明新结构）。</b>所有新引擎照 softmax 单元 SM16 的方式接进阵列：读激活缓存的广播口、16 车道写回、调度器串行调度；描述符用空闲的 op=6。四个关键设计判断：</p>
<ul>
<li><b>查表完胜分段线性。</b>INT8 输入只有 256 种取值，SiLU/GELU 用 256 项查找表预存"算好再量化"的结果，与 host 的浮点算完再量化<b>同舍入位精确</b>，新增误差约等于零；分段线性反而多 1–2 个最低位误差，还不省资源。</li>
<li><b>归一化的精度根基是整数累加。</b>第一遍扫描的 Σx、Σx² 用纯整数累加器，零舍入；统计和缩放在 32–48 位定点里做。对端到端精度的影响预计 ≤0.003 rad（当前误差 0.0288，门限 0.045，余量够）——但动 RTL 前先在 V100 仿真里用定点模型跑一遍验证，零成本。</li>
<li><b>softmax 不用新硬件。</b>SM16 已经在片上，只缺"独立调用"的描述符；220 处小 softmax 里 184 处零新数据通路。</li>
<li><b>必须配套改编译器，否则只有硬件没有收益。</b>被融合的算子改发新描述符、两侧段合并、γ/cos-sin/bias 表进段内存映像——硬件和编译器是一对，缺一不可。</li>
</ul>
<p><b>四步走（每步独立验收门，风险从低到高）：</b>① 先做 actv+bias 两个模式（约 2k LUT，可与 host 位精确对拍），把描述符/表装载的地基调通；② norm 上 RTL 前，先跑 V100 定点模型验证（判据：jpos MAE ≤ 0.033）；③ 加 softmax 独立描述符，纯 rotary/temporal 路径先上；④ rotary 与 swin 散射补完，吃下 backbone。</p>
<p><b>调研同时把下一堵墙看清了：</b>编译器行分块边界上还压着 <b>222.8M 拍（20.6%）</b>的 STORE+LOAD_CTX——比本轮全部算子融合加起来还大。治它要靠编译器加大段、流式装载，那是下一个主战场（已并入 D2）。另外就算搬运全消，有效利用率上限也只有 14.1%（GEMM 引擎忙时每拍平均只做 373 个乘加 = 峰值的 22%）——那是阵列形状与层形状的匹配问题，对应 D3。</p>
<p style="font-size:13px;color:#52514e">产物：07_onchip_ops/（NOTES.txt 完整方案、benefit_estimation.json 收益表、boundary_account.json 归因原始账、四个可复跑分析脚本）。未改任何 RTL/编译器。</p>
</div>

<h2>七、调研结果②：INT4 权重量化（已到）<span class="date">2026-08-31 11:27 并入</span></h2>
<p class="hint">结论：值得做，推荐分三档里的中间档——权重字节减 35.6%，精度与 W8 在同一噪声带。</p>
<div class="entry">
<p><b>一句话方案：</b>BERT 文本编码器降到 per-tensor INT4、Swin RGB 塔降到 g=128 组内 INT4、neck 卷积降 per-tensor INT4，其余保持 W8。<b>权重字节从 159.5 MB 降到 102.6 MB，减少 35.6%</b>；合成口径偏差 0.0231 rad、真实样本最差 0.0315 rad，与 W8A8 基线（0.0259 / 0.0288）在同一噪声带，低于 0.04 rad 红线（端到端判据 0.045 rad，还有余量）。</p>
<figure class="chart">
<h4>哪些模块降 INT4、哪些保持 W8（按字节占比）</h4>
<p class="how">怎么看：条子是模块占的权重字节。最大的一块（BERT，53%）恰恰是最不敏感的——这是本次调研最大的惊喜。悬停看每个模块的偏差数字和判断依据。</p>
{{SVG:qa_w4_modules}}
</figure>
<p><b>三档方案，按硬件改动量选：</b></p>
<table>
<tr><th>档位</th><th>内容</th><th class="num">字节节省</th><th class="num">合成口径</th><th>真实样本</th><th>硬件代价</th></tr>
<tr><td>零改动</td><td>只有 BERT 72 层 → per-tensor INT4</td><td class="num">26.7%</td><td class="num">0.0187 rad</td><td>与 W8 持平</td><td><b>零</b>：只换常数表（sw→absmax/7，requant 重算 r），439 层全部可编码无溢出</td></tr>
<tr><td><b>推荐</b></td><td>上面 + Swin RGB 塔 g=128 + neck 4 卷积</td><td class="num"><b>35.6%</b></td><td class="num">0.0231 rad</td><td>0.029 / 0.032 / 0.018</td><td>需要组 scale 在累加时反量化——下一轮 RTL 项</td></tr>
<tr><td>激进</td><td>再加 action_head（42%）或全模型（49%）</td><td class="num">42–49%</td><td class="num">0.037–0.047</td><td>出现 0.056–0.065 坏例</td><td>超 0.04 红线，<b>不建议</b></td></tr>
</table>
<p><b>方法是怎么筛的。</b>我们的约束很硬：模型只有 0.2B、激活已冻结成 per-tensor 静态 INT8、每个矩阵乘只有一组 requant 常数。所以只能动权重的办法可用（GPTQ 需标定、HQQ 免标定、RTN+组 scale 零成本——本轮用的就是它）；SmoothQuant、AWQ、QuIP 这些都要动激活侧或改激活分布，跟冻结的 per-tensor 缩放直接冲突，筛掉。文献侧证也对得上：0.5B 级模型 W4A8 普遍掉 9–14%（整模型 W4 红灯不是我们的 bug）；但 ViT 塔单独 W4 只掉零点几点的先例存在，和我们"伤害集中在部分模块"的画像一致。</p>
<p><b>一个反直觉的实验发现：</b>单层降到 INT4 的边际代价中位数只有 +0.0003~0.002 rad，模块级的伤害是几百个小误差在扩散头里累积再被混沌放大的结果。所以逐层贪心选层不可靠（第一轮贪心组合的预估与实测差了一个量级），最终方案靠"模块级锚点定方向、组合实测定方案"。同理，对比实验必须在同进程做——W8 基线跨进程有 10% 的浮点漂移（0.0288 vs 0.0259），不同进程的数字不能直接比。</p>
<p><b>对硬件意味着什么。</b>权重字节减 35.6% 直接改善第三节根因 3（权重装载/驻留）：LOAD_W 的数据量变小，同样的 WRAM/CTX 容量能驻留更多层。零改动档今天就能换上（只动常数表）；推荐档的组 scale 反量化是 RTL 项，建议和第六节的 AE_ACTV 凑同一批综合（都动 u_core，一次布线验证两件事）。</p>
<p style="font-size:13px;color:#52514e">产物：07_int4/（NOTES.txt 调研+实验+8 条坑、cfgs/FINAL_E.json 推荐配置、final_w4_layers.csv 128 层清单、dump/scan_table.csv 876 行逐层扫描表）。服务器无遗留进程。</p>
</div>

<h2>八、仍在跑的一件 <span class="running">进行中 · 结果将补充</span></h2>
<div class="entry">
<table>
<tr><th>方向</th><th>要回答的问题</th><th>状态</th></tr>
<tr><td>端到端数值链</td><td>compiler dma_len 正式修复 + 重编译 + 样本 000 全链对 fp32 参考，判断误差是否在 0.045 rad 内</td><td><span class="running">后台运行中 {{NOW}}</span></td></tr>
</table>
</div>

<h2>九、等你拍板的五个决策点</h2>
<div class="entry">
<table>
<tr><th>#</th><th>决策</th><th>选项</th><th>我的建议</th></tr>
<tr><td>D1</td><td>工作量口径怎么对齐</td><td>A 双视角对照实验（约一天）/ B 按 1.98 折算 / C 双口径报告</td><td><b>C 立刻 + A 排队</b></td></tr>
<tr><td>D2</td><td>搬运优化的动手顺序（v2 已按归因账更新）</td><td>五个根因先做哪个</td><td>① BERT 缓存（−2.8%，host 几行，立刻）；② AE_ACTV 的 MVP（净省 7.7%，+swin 9.0%；先 actv+bias 打底、norm 动 RTL 前先跑 V100 定点验证）；③ <b>编译器加大段 + 流式装载</b>治 222.8M 拍（20.6%）的行分块边界——单项最大，下一主战场；④ 顺手：预取窗口扩到 COPY/新引擎运行期（~10 LUT）。COPY 编译期化降级（上界仅 1.4%）</td></tr>
<tr><td>D3</td><td>padding（×1.53）动不动</td><td>段合并拼 K 能大幅减少补零，但调度复杂度上升</td><td>和 D2 的段合并同源，等计数器归因后一起决定</td></tr>
<tr><td>D4</td><td>调研结果并入后是否开新一轮综合</td><td>INT4 / 片上算子都会改 RTL</td><td>数值链结果一到就凑批：AE_ACTV + swin 散射 + INT4 组 scale 反量化一次综合，都动 u_core，一次布线验证三件事</td></tr>
<tr><td>D5</td><td>INT4 上哪一档</td><td>零改动（BERT only，−26.7% 字节，0.0187）/ 推荐（+Swin g128+neck，−35.6%，0.0231）/ 激进（42–49%，超红线）</td><td><b>推荐档</b>；零改动档可以不等综合先换上（只动常数表）。组 scale 反量化 RTL 与 D4 凑批</td></tr>
</table>
</div>

<footer>
<p>数据来源：05_sim/cycles_by_type.json（441 类型实测，2026-08-30/31）；research_holobrain/profile/holobrain_hw.json（账本）；E:/ae_syn/pwr/hier_util.rpt（布线后 DCP 层级账，2026-08-31 10:17:33，Vivado 2021.2 xczu7ev）；07_onchip_ops/boundary_account.json + benefit_estimation.json（归因账与收益表，2026-08-31）；图表为 gen_charts.js 生成的内联 SVG。</p>
<p>生成时刻：<b>{{NOW}}</b>（v3：10:30 初稿 + 10:49 on-chip + 11:27 INT4；端到端数值链到达后出 v4）。</p>
</footer>

</div>
<script>
(function(){
  var tip = document.getElementById('tip');
  document.querySelectorAll('[data-tip]').forEach(function(el){
    el.addEventListener('mousemove', function(e){
      tip.textContent = el.getAttribute('data-tip');
      tip.style.opacity = 1;
      var x = Math.min(e.clientX + 14, window.innerWidth - 300);
      tip.style.left = x + 'px';
      tip.style.top = (e.clientY + 18) + 'px';
    });
    el.addEventListener('mouseleave', function(){ tip.style.opacity = 0; });
  });
})();
</script>
</body>
</html>
"""

for name in ['qa_gap', 'qa_recon', 'qa_cycle_mix', 'qa_onchip_gain', 'qa_w4_modules', 'qa_modules', 'qa_time', 'sw_est_ladder']:
    with io.open(os.path.join('charts', name + '.svg'), 'r', encoding='utf-8') as f:
        html = html.replace('{{SVG:' + name + '}}', f.read().strip())
html = html.replace('{{NOW}}', now)

assert '{{' not in html, 'unresolved placeholder'

out = u'%s_实测vs账本差距问题讨论.html' % fn
with io.open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print('written:', out, len(html), 'chars')
