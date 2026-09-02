# -*- coding: utf-8 -*-
# 2026-08-31 答疑页生成器：RoboTwin vs LIBERO / 视角区别 / CTX / 指令集
import datetime, io, os

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
fn = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")

html = u"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>数据集、视角与指令集答疑 · 2026-08-31</title>
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
.date{display:inline-block;font-size:12px;font-weight:600;color:#52514e;background:#f0efec;border-radius:5px;padding:1px 8px;margin-left:8px;vertical-align:2px;white-space:nowrap}
p{margin:8px 0}
ul,ol{margin:8px 0;padding-left:22px}
li{margin:4px 0}
figure.chart{margin:18px 0;background:#fcfcfb;border:1px solid rgba(11,11,11,.10);border-radius:10px;padding:14px 18px 8px}
figure.chart h4{margin:2px 0 2px;font-size:15px}
figure.chart .how{font-size:13px;color:#52514e;margin:0 0 8px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}
th,td{border-bottom:1px solid #e1e0d9;text-align:left;padding:6px 8px;vertical-align:top}
th{color:#52514e;font-weight:600;border-bottom:1px solid #c3c2b7}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.ok{border-left:3px solid #3e9c5c;background:#f6fbf7;padding:10px 14px;border-radius:0 8px 8px 0;margin:12px 0}
.warn{border-left:3px solid #fab219;background:#fffdf3;padding:10px 14px;border-radius:0 8px 8px 0;margin:12px 0}
code{background:#f0efec;border-radius:4px;padding:1px 5px;font-size:12.5px}
.running{display:inline-block;font-size:12px;font-weight:600;color:#8a6100;background:#fff4dd;border-radius:5px;padding:1px 8px;vertical-align:2px;white-space:nowrap}
footer{margin-top:44px;padding-top:14px;border-top:1px solid #e1e0d9;font-size:12.5px;color:#898781}
</style>
</head>
<body>
<div class="wrap">

<header class="page">
<h1>答疑：为什么用 RoboTwin 不用 LIBERO、视角差在哪、CTX 是什么、指令集长什么样</h1>
<div class="meta">
<span>生成时刻：<b>{{NOW}}</b></span>
<span>依据：research_holobrain 调研档案、04_dataset/DATASET_NOTES.txt、hw_zcu104/sim/gen_vectors.py（描述符编码权威）</span>
</div>
<nav classagenav>
<nav class="pagenav">
<a href="2026-08-31_1128_实测vs账本差距问题讨论.html">→ 问题讨论页（v3）</a>
<a href="2026-08-31_架构工作记录.html">→ 架构方向记录</a>
<a href="../WORKLOG.md">完整工作日志</a>
</nav>
</header>

<div class="ok">
<p style="margin:4px 0"><b>四句话先答完：</b></p>
<p style="margin:4px 0">① 选 RoboTwin 是因为 <b>LIBERO 的后训权重没有开源</b>，而 HB-GD 必须喂深度、RoboTwin 的相机参数齐全能合成深度，官方小子集当天就能离线跑通。</p>
<p style="margin:4px 0">② 4 相机 vs 2 视角的区别就是<b>工作量差一倍</b>（每个相机独立过网络，乘加随相机数线性涨）；1 视角没有官方配置，是算法决定不是硬件决定。</p>
<p style="margin:4px 0">③ CTX 是片上的<b>激活缓存</b>（64 块 URAM），GEMM 的输入和中间结果都住这里；LOAD_CTX 就是"把激活从内存装进这片缓存"。</p>
<p style="margin:4px 0">④ 指令集一共 <b>7 条指令</b>（256 位一条），第五节给了完整字段表和语义。</p>
</div>

<h2>一、为什么当时选 RoboTwin，不是 LIBERO</h2>
<div class="entry">
<p>选数据集是在 2026-08-30 定的，三个原因按重要性排：</p>
<p><b>原因一：LIBERO 的权重拿不到。</b>HB-GD 官方开源了代码和 RoboTwin 后训权重，但<b>LIBERO 后训权重明确不在开源范围内</b>。要用 LIBERO 就得自己拿官方代码在 LIBERO 上重新后训——V100 上数天级的活，而且不保证复现论文的 96.7 分。硬件验证要的是"尽快有一个能跑的真模型"，等不起这个。</p>
<p><b>原因二：深度输入的来源问题，两边其实一样。</b>HB-GD 的推理必须喂深度（开了 3D 特征开关后不喂深度直接报错，不是降级）。调研时确认了一个关键事实：论文 LIBERO 96.7 分用的是 MuJoCo 仿真器渲染的完美深度。RoboTwin 官方数据集虽然没存深度，但四个相机的内外参齐全，我们用真实内外参加桌面平面求交合成了深度（四个输入模态里唯一合成的项，其余全真实）。也就是说，<b>两条路用的都是仿真器深度，谁也不比谁"作弊"更多</b>；RoboTwin 这条路当天就能走通。</p>
<p><b>原因三：工程速度。</b>RoboTwin 官方 place_empty_cup 任务 50 集，zip 支持 HTTP Range 断点下载，实际只拉了 55MB 就离线重建出样本和 fp32 参考。LIBERO 的数据管线要搭 MuJoCo 渲染，重得多。</p>
<p><b>顺带说清一个这周对表才看清的事实：</b>研究阶段的账本其实算过 RoboTwin 档——149.8 GMAC（LIBERO 2 视角档 73.7 是主档，因为要对表论文 benchmark）。我们这次实测 145.8 GMAC 有效乘加，是账本 RoboTwin 档的 <b>97.4%</b>，差的那点在文本 token 数和科目划分。<b>所以实测和账本本来就是对的</b>，之前说的"1.98×"是账本主档（LIBERO 2 视角）和 RoboTwin 档之间的档位差，不是实测丢了什么。</p>
</div>

<h2>二、4 个相机、2 个视角、1 个视角，区别在哪</h2>
<div class="entry">
<p><b>先说结论：对硬件来说，区别就是工作量。</b>HB-GD 的网络里，每个相机/视角的画面独立过 2D 主干和融合，相机之间没有交叉注意力。所以乘加随相机数<b>线性</b>涨——4 个相机就是 2 视角的两倍乘加、约两倍拍数。这也是上一页对表时四个科目比值全挤在 1.97–2.06 的原因。</p>
<table>
<tr><th>配置</th><th>相机/视角</th><th class="num">账本档（GMAC）</th><th>拍数（相对本次）</th><th>备注</th></tr>
<tr><td><b>RoboTwin 4 相机</b>（本次）</td><td>第三视角 + 左腕 + 右腕 + 头部</td><td class="num">149.8（实测 145.8）</td><td>基准：5.44s @198.5MHz</td><td>双臂操作任务需要左右腕各看一眼</td></tr>
<tr><td>LIBERO 2 视角（账本主档）</td><td>全局相机 + 腕部相机</td><td class="num">73.7</td><td>约一半（除法推算 ≈2.2s @250MHz，未实测）</td><td>单臂任务的典型配置，论文 benchmark 口径</td></tr>
<tr><td>1 视角</td><td>—</td><td class="num">无此档</td><td>—</td><td>官方没有这个配置；3D 表征按多视角+深度设计，砍到 1 个视角表征质量无保证，掉多少点是算法问题，与硬件无关</td></tr>
</table>
<p><b>为什么不能靠砍视角省算力：</b>砍视角确实能让硬件工作量减半再减半，但模型精度会掉（2D 检测、深度三角化、抓取位姿估计都吃多视角冗余）。这是算法侧的决定，硬件侧只能如实反映它带来的乘加变化。真想要"2 视角的硬件数字"，正确做法是跑一次双视角对照实验（RoboTwin 数据只喂 2 个相机，一天），而不是改模型。</p>
</div>

<h2>三、有必要换成 LIBERO 数据集吗</h2>
<div class="entry">
<p><b>把"换 LIBERO"拆成两个独立的问题，答案不一样：</b></p>
<table>
<tr><th>你想要什么</th><th>需要换数据集吗</th><th>实际要做什么</th></tr>
<tr><td>和论文同口径的<b>硬件性能数字</b>（0.27s 那张账能直接对上）</td><td><b>不用</b></td><td>跑双视角对照实验（问题讨论页 D1-A）：RoboTwin 数据只喂 2 个相机，重跑 trace + 测拍，约一天。相机位姿/场景和 LIBERO 不同，但工作量口径一致</td></tr>
<tr><td>LIBERO benchmark 的<b>任务成功率</b>（96.7 那个数）</td><td>要，但这是算法线的事</td><td>自己后训 LIBERO 权重（V100 数天级）+ MuJoCo 深度渲染管线。和 FPGA 仿真器无关；等算法线出了权重，硬件侧只需重跑一遍 trace</td></tr>
</table>
<p><b>我的建议：现阶段不换。</b>硬件验证的目标（跑对、测准拍数）不依赖数据集；口径问题用 D1-A 双视角对照解决，成本一天。LIBERO 成功率是另一条线的事，需要时再切——切的时候编译器输入换个 trace 就行，架构不动。</p>
</div>

<h2>四、CTX 是什么（顺便说清它和 WRAM 的分工）</h2>
<div class="entry">
<p><b>CTX 就是片上的激活缓存。</b>芯片上有两块主要存储，分工很清楚：</p>
<table>
<tr><th></th><th>CTX（上下文缓存）</th><th>WRAM（权重存储）</th></tr>
<tr><td>存什么</td><td><b>激活</b>：GEMM 的输入矩阵、段内算出来的中间结果</td><td><b>权重</b>：当前段正在算的那一层的 INT8 权重</td></tr>
<tr><td>在哪</td><td>64 块 URAM（u_ctx 模块）</td><td>108 块 BRAM（每列一块 bank）</td></tr>
<tr><td>组织</td><td>16 路交织：一拍能读 16 行的同一列，正好喂 16 行的脉动阵列</td><td>108 列各读各的</td></tr>
<tr><td>谁往里装</td><td><b>LOAD_CTX</b> 指令（从 DDR 搬进来）</td><td><b>LOAD_W</b> 指令（从 DDR 搬进来）</td></tr>
</table>
<p><b>为什么 LOAD_CTX 有 42,246 条那么多。</b>现在的架构是"段独立"：一次推理切成 2782 个段，每段的激活用完就扔，下一段要用的再从 DDR 装一遍。同一个归一化结果甚至会被下游几个段反复装 2.5 次。这就是搬运债里 164.9M 拍（15.3%）的来源——不是 CTX 太小，是调度没让数据<b>留下来</b>。正在跑的 compute-bound 改造（CTX 驻留分析 + 段合并）就是治这个的。</p>
</div>

<h2>五、我们目前的指令集：7 条指令、256 位一条</h2>
<div class="entry">
<p><b>整套架构只有 7 条指令</b>（硬件行话叫描述符，descriptor）。每条 256 位，编译器生成，按顺序写进 seq.mem 文件，调度器一条条读、译码、执行。这个设计是刻意的：指令越少，验证墙越矮。</p>
<table>
<tr><th class="num">op 编码</th><th>指令</th><th>干什么</th><th class="num">一次推理条数</th><th class="num">拍数（归因账）</th></tr>
<tr><td class="num">0</td><td><b>GEMM</b></td><td>矩阵乘：A 从 CTX 读、B 从 WRAM 读，32 位累加，requant（乘常数、右移、饱和）压回 INT8 写回 CTX</td><td class="num">69,414（含 op 0/1/2）</td><td class="num">GEMM 共 599.1M</td></tr>
<tr><td class="num">1</td><td><b>ATTN_S</b></td><td>GEMM 之后紧接硬件 softmax（SM16 单元，16 行并行），原地写回——注意力路径专用</td><td class="num">（同上）</td><td class="num">（同上）</td></tr>
<tr><td class="num">2</td><td><b>HOIST</b></td><td>GEMM + "循环不变"标记：去噪 10 步里结果不变的部分，第二步起直接跳过重算</td><td class="num">（同上）</td><td class="num">（同上）</td></tr>
<tr><td class="num">3</td><td><b>COPY</b></td><td>CTX → WRAM 的列重排：把激活摆成下一 个 GEMM 要的 B 阵形状（转置 K/V 之类）</td><td class="num">54,032</td><td class="num">15.2M</td></tr>
<tr><td class="num">4</td><td><b>LOAD</b></td><td>DDR → 片上。目的地由 b_src 字段选：0 = CTX（激活，即 LOAD_CTX）、1 = WRAM（权重，即 LOAD_W）</td><td class="num">42,246 + 15,494</td><td class="num">164.9M + 75.4M</td></tr>
<tr><td class="num">5</td><td><b>STORE</b></td><td>CTX → DDR：结果 16 路解交织写回内存。目前只有 3.2 字节/拍，是最大的单项搬运债</td><td class="num">19,410</td><td class="num">206.6M</td></tr>
<tr><td class="num">15</td><td><b>DONE</b></td><td>段结束。每个段的最后一条</td><td class="num">2,782</td><td class="num">≈0</td></tr>
</table>
<p>剩下的编码（6、7–14）空闲。正在落地的 on-chip 算子引擎（SiLU/GELU 查表、bias、norm）就用 <b>op=6</b>，256 位布局不动，只在 b_src 字段里放子模式——新旧指令集兼容。</p>
<p><b>256 位里的字段怎么排</b>（位切片从高位到低位，编码权威是 gen_vectors.py 的 desc() 函数）：</p>
<table>
<tr><th>字段（位段）</th><th>宽度</th><th>含义</th></tr>
<tr><td>op [255:252]</td><td>4</td><td>指令编码（上表的 0/1/2/3/4/5/15）</td></tr>
<tr><td>a_src [251:249]</td><td>3</td><td>A 操作数来源</td></tr>
<tr><td>b_src [248:246]</td><td>3</td><td>B 来源；LOAD 指令里当地址目的地（0=CTX，1=WRAM）</td></tr>
<tr><td>sm_causal [245]</td><td>1</td><td>softmax 是否因果掩码（只算下三角）</td></tr>
<tr><td>y_tr [244]</td><td>1</td><td>输出是否写转置布局（PV 那类要转置的 GEMM 用）</td></tr>
<tr><td>m / n / k [243:228 / 227:212 / 211:196]</td><td>各 16</td><td>矩阵乘形状：输出 m×n、 contracting 维 k</td></tr>
<tr><td>a_base / b_base / y_base [195:176 / 175:156 / 155:136]</td><td>各 20</td><td>A、B、输出在 CTX/WRAM 里的起始地址</td></tr>
<tr><td>b_spad [135:120]</td><td>16</td><td>GEMM 里是 n_loc（本组实际列数）；COPY 里复用为跨度</td></tr>
<tr><td>rq_m [119:104]</td><td>16</td><td>requant 乘数（Q8.8 定点）；COPY 复用为源列号 src_j0</td></tr>
<tr><td>rq_s [103:96]</td><td>8</td><td>requant 右移位数</td></tr>
<tr><td>inv_idx [95:92]</td><td>4</td><td>循环不变槽号（仅 GEMM 族可用，0xF = 不是不变量）</td></tr>
<tr><td>steps [91:81] / in_loop [80] / is_loop_end [79]</td><td>11+1+1</td><td>段内循环：去噪 10 步这种重复用硬件循环跑，不用编译器展开 10 份指令</td></tr>
<tr><td>dma_len [78:61]</td><td>18</td><td>LOAD/STORE 的搬运字节数，上限 262,128——<b>就是那个溢出 bug 的字段</b>（超长搬运曾被静默截断，已修）</td></tr>
<tr><td>j0 [77:62]</td><td>16</td><td>全局列起始号。注意它和 dma_len <b>共用位段</b>：GEMM 族不用 DMA 所以借这块地方放列号，LOAD/STORE 不用列号所以放长度——一条 256 位塞下两种指令的省地设计</td></tr>
<tr><td>dma_addr [60:29]</td><td>32</td><td>DDR 侧地址</td></tr>
<tr><td>[28:0]</td><td>29</td><td>保留未用</td></tr>
</table>
<figure class="chart">
<h4>一次推理 203,378 条指令都在干什么</h4>
<p class="how">怎么看：把上表的"条数"画出来。LOAD 家族（CTX+W）和 COPY 加起来超过一半——这就是 memory-bound 的直观样子。compute-bound 改造的目标就是把这三根条子压短。</p>
{{SVG:sw_desc_mix}}
</figure>
<p><b>三个值得知道的机制</b>（都是为省指令数/拍数设计的）：</p>
<ul>
<li><b>段内硬件循环</b>：去噪 10 步的指令流完全一样，用 in_loop/steps 让硬件自己循环，不用编译器复制 10 份——不然指令条数直接 ×10。</li>
<li><b>循环不变跳过（HOIST）</b>：标了 inv_idx 的 GEMM，第二步起硬件直接跳过发射。BERT 的指令可整段复用也是同一思想。</li>
<li><b>requant 折进指令</b>：每条 GEMM 自带 (rq_m, rq_s)，算完立刻压回 INT8，不用单独一条量化指令。</li>
</ul>
</div>

<h2>六、现在在跑的三个后台活 <span class="running">进行中</span></h2>
<div class="entry">
<table>
<tr><th>活</th><th>目标</th><th>状态</th></tr>
<tr><td>compute-bound 改造</td><td>编译器/调度侧把搬运债压下去：行分块段界合并、CTX 驻留、STORE 合并、装载提前。目标总拍数 −20% 以上、GEMM 占比 55.4%→≥70%，数值逐位不变</td><td><span class="running">后台进行中 {{NOW}}</span></td></tr>
<tr><td>on-chip 引擎 + INT4 落地</td><td>AE_ACTV 引擎（actv+bias 模式起步，op=6）RTL 实现 + 位精确对拍；INT4 零改动档（BERT，权重字节 −26.7%）落地验证</td><td><span class="running">后台进行中 {{NOW}}</span></td></tr>
<tr><td>端到端数值链</td><td>compiler dma_len 正式修复 + 重编译 + 样本 000 全链对 fp32，判 ≤0.045 rad</td><td><span class="running">后台进行中 {{NOW}}</span></td></tr>
</table>
<p>三个活回来后出下一份 HTML，按规矩给<b>具体的解决方案和效果</b>（每项优化写清"提升了 xx%"，前后对比）。</p>
</div>

<footer>
<p>数据来源：research_holobrain 调研档案（2026-08-30，RoboOrchardLab 源码核实）；04_dataset/DATASET_NOTES.txt（数据集来源与映射）；hw_zcu104/sim/gen_vectors.py（描述符编码与黄金语义）；05_sim 拍数归因（07_onchip_ops/boundary_account.json）。</p>
<p>生成时刻：<b>{{NOW}}</b></p>
</footer>

</div>
</body>
</html>
"""

for name in ['sw_desc_mix']:
    with io.open(os.path.join('charts', name + '.svg'), 'r', encoding='utf-8') as f:
        html = html.replace('{{SVG:' + name + '}}', f.read().strip())
html = html.replace('{{NOW}}', now)
assert '{{' not in html
html = html.replace('<nav classagenav>\n', '')  # 清掉笔误的空标签

out = u'%s_数据集视角与指令集答疑.html' % fn
with io.open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print('written:', out, len(html), 'chars')
