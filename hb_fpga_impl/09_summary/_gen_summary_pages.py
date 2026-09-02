# -*- coding: utf-8 -*-
# 2026-08-31 三份总结页生成器：架构 / 算法 / 电路（覆盖上份工作记录以来的工作）
# 规矩：秒级时间戳、量化写 xx%、图表内联、悬停 tooltip、新轮次新文件夹（09_summary/）
import datetime, io, os

NOW = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
FN = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")

CSS = """
:root{color-scheme:light}
*{box-sizing:border-box}
body{margin:0;background:#f9f9f7;color:#0b0b0b;font-family:system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;line-height:1.75;font-size:15px}
.wrap{max-width:920px;margin:0 auto;padding:28px 22px 80px}
header.page{border-bottom:1px solid #e1e0d9;padding-bottom:18px;margin-bottom:10px}
h1{font-size:23px;margin:0 0 10px;line-height:1.4}
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
.warn{border-left:3px solid #d64545;background:#fdf6f6;padding:10px 14px;border-radius:0 8px 8px 0;margin:12px 0}
.info{border-left:3px solid #2a78d6;background:#f5f9fe;padding:10px 14px;border-radius:0 8px 8px 0;margin:12px 0}
code{background:#f0efec;border-radius:4px;padding:1px 5px;font-size:12.5px}
.running{display:inline-block;font-size:12px;font-weight:600;color:#8a6100;background:#fff4dd;border-radius:5px;padding:1px 8px;vertical-align:2px;white-space:nowrap}
#tip{position:fixed;opacity:0;pointer-events:none;background:#0b0b0b;color:#fff;font-size:12.5px;line-height:1.5;padding:6px 10px;border-radius:6px;max-width:280px;z-index:9;transition:opacity .12s}
footer{margin-top:44px;padding-top:14px;border-top:1px solid #e1e0d9;font-size:12.5px;color:#898781}
"""

JS = """
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
"""


def make_page(fname, title, scope, tldr, body, sources):
    html = u"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>%s</title>
<style>%s</style>
</head>
<body>
<div id="tip"></div>
<div class="wrap">

<header class="page">
<h1>%s</h1>
<div class="meta">
<span>生成时刻：<b>%s</b></span>
<span>本页范围：%s</span>
</div>
<nav class="pagenav">
<a href="2026-08-31_%s_架构总结.html">架构总结</a>
<a href="2026-08-31_%s_算法总结.html">算法总结</a>
<a href="2026-08-31_%s_电路总结.html">电路总结</a>
<a href="../08_report/2026-08-31_1128_实测vs账本差距问题讨论.html">问题讨论页 v3</a>
<a href="../08_report/2026-08-31_1253_数据集视角与指令集答疑.html">答疑页</a>
<a href="../WORKLOG.md">完整工作日志</a>
</nav>
</header>

<div class="ok">%s</div>

%s

<footer>
<p>%s</p>
<p>生成时刻：<b>%s</b>。上一份总结：《2026-08-31 架构/硬件/软件工作记录》（08_report/）。compute-bound 改造代理仍在跑，结果出来后另出页，不在本份总结里。</p>
</footer>

</div>
%s
</body>
</html>
""" % (title, CSS, title, NOW, scope, FN, FN, FN, tldr, body, sources, NOW, JS)

    for name in CHARTS.get(fname, []):
        with io.open(os.path.join('charts', name + '.svg'), 'r', encoding='utf-8') as f:
            html = html.replace('{{SVG:' + name + '}}', f.read().strip())
    assert '{{' not in html, 'unresolved placeholder in ' + fname
    with io.open(fname, 'w', encoding='utf-8') as f:
        f.write(html)
    print('written:', fname, len(html), 'chars')


CHARTS = {}

# ============================================================ 架构页
CHARTS['%s_架构总结.html' % FN] = ['qa_gap', 'qa_recon', 'qa_cycle_mix', 'sw_est_ladder', 'qa_onchip_gain', 'sw_desc_mix']

arch_tldr = u"""<p style="margin:4px 0"><b>架构这半天干了四件事：</b></p>
<p style="margin:4px 0">① 实测 5.44s 对账本 0.27s 的 <b>16× 差距拆完了</b>：约一半是口径（RoboTwin 4 相机 vs 账本 2 视角、补零放大），另一半是真问题——搬运和停顿吃掉大头，阵列利用率只有 12%。</p>
<p style="margin:4px 0">② 拍数账拆到<b>每条指令</b>：写回（STORE）占 19.1% 是第一大项、激活反复装载占 15.3% 第二——memory-bound 确诊。</p>
<p style="margin:4px 0">③ 编译器修掉<b>两个真 bug</b>（超长搬运溢出、多 tile 写回覆盖），链路验证做到零实现误差。</p>
<p style="margin:4px 0">④ on-chip 算子引擎方案定型（净省 7.7%~9.0% 拍）且 RTL 已位精确落地；<b>compute-bound 编译器改造正在跑</b>，目标再砍 20% 以上拍数。</p>"""

arch_body = u"""
<h2>一、16× 差距拆解：三跳连乘，口径和真问题各占一半 <span class="date">2026-08-31 10:30</span></h2>
<div class="entry">
<p>实测一次推理 1080.8M 拍（5.44s @198.5MHz），账本说只要 0.27s。这个 16 倍差距拆成三跳：</p>
<ol>
<li><b>工作量口径 ×1.98</b>：账本主档算的是 LIBERO 2 视角（73.7 GMAC），我们跑的是 RoboTwin 4 相机（账本里本来就有这一档：149.8 GMAC）。</li>
<li><b>补零放大 ×1.53</b>：有效乘加 145.8G，补零到形状对齐后变成 223.8G——列组取整、窗口对齐的代价。</li>
<li><b>搬运与停顿 ×5.27</b>：真正的问题。理想流水利用率 63%，实测只有 12%——阵列大部分时间在等数据。</li>
</ol>
<figure class="chart">
<h4>从账本到实测：每一跳差在哪</h4>
<p class="how">怎么看：四根柱子从左到右是"账本估计 → 换成实测工作量 → 加上补零 → 加上搬运停顿"。柱子上的 × 就是上一跳的倍数，三跳连乘 = 16.0。鼠标悬停看每一步的算法。</p>
{{SVG:qa_gap}}
</figure>
<p><b>口径对表（这轮的新结论）</b>：实测有效乘加 145.8G 是账本 RoboTwin 档 149.8G 的 <b>97.4%</b>。所以实测和账本本来就是对的，1.98× 是两档之间的差，不是我们丢了东西。四个大科目的实测÷账本比值全挤在 1.97–2.06（相机数线性扩展的指纹），只有文本是 0.50（tokenize 后 8 vs 16 个 token）。</p>
<figure class="chart">
<h4>四个大科目全是约 2 倍——这是相机数的指纹</h4>
<p class="how">怎么看：每根条是"实测 ÷ 账本"。1.0 处的竖线是账本基准。四个大科目挤在 1.97–2.06，只有文本在 0.5。</p>
{{SVG:qa_recon}}
</figure>
</div>

<h2>二、拍数归因账：写回是第一大项，不是计算 <span class="date">2026-08-31 10:30–10:50</span></h2>
<div class="entry">
<p>方法：把一次推理的 2782 段指令流全部解码，用 RTL 常数逐条算拍数，再和 441 类 GEMM 的实测对账——两者差 1.8%，互为印证。结果（总 1080.8M 拍）：</p>
<table>
<tr><th>去向</th><th class="num">拍数</th><th class="num">占比</th><th>说明</th></tr>
<tr><td>GEMM 计算</td><td class="num">599.1M</td><td class="num">55.4%</td><td>矩阵乘本体，含 requant 和 softmax 计入路径</td></tr>
<tr><td><b>STORE 写回</b></td><td class="num">206.6M</td><td class="num"><b>19.1%</b></td><td>结果写回 DDR，660.9MB 只有 3.2 字节/拍——<b>第一大单项</b></td></tr>
<tr><td><b>LOAD_CTX 激活装载</b></td><td class="num">164.9M</td><td class="num"><b>15.3%</b></td><td>1.3GB 激活反复从 DDR 装进片上缓存，平均每个被装 2.5 次</td></tr>
<tr><td>LOAD_W 权重装载</td><td class="num">75.4M</td><td class="num">7.0%</td><td>预取已吃掉 33.7M，剩余约 42M</td></tr>
<tr><td>调度+LFSR</td><td class="num">19.6M</td><td class="num">1.8%</td><td>描述符调度与伪随机停顿</td></tr>
<tr><td>COPY 重排</td><td class="num">15.2M</td><td class="num">1.4%</td><td>5.4 万条指令但只占 1.4% 拍</td></tr>
</table>
<figure class="chart">
<h4>10.81 亿拍都花在哪（按拍数归因）</h4>
<p class="how">怎么看：矩阵乘本体只占 55%，剩下 45% 是搬运和停顿。悬停看每一项的细节。</p>
{{SVG:qa_cycle_mix}}
</figure>
<p><b>一处自我修正</b>：10:30 初稿曾按"指令条数"排序，COPY 5.4 万条看着像大杠杆；归因账到了以后按拍数重排——COPY 其实只占 1.4%，STORE 才是第一位。看错排序会把力气花错地方。</p>
<figure class="chart">
<h4>拍数估算 vs 实测（单位：百万拍）</h4>
<p class="how">怎么看：估算模型逐次逼近实测，最后一级差 21% 是 LFSR 停顿、仲裁这些模型没算的开销。以后用"修正后估算 ×1.26"做快速预测。</p>
{{SVG:sw_est_ladder}}
</figure>
</div>

<h2>三、on-chip 算子融合：方案定型，净省 7.7%~9.0% 拍 <span class="date">2026-08-31 10:49</span></h2>
<div class="entry">
<p>host 边界往返一共 148.8M 拍（13.8%），其中 norm 族占 49%。方案是 <b>AE_ACTV 统一行引擎</b>：norm、激活、rotary、bias、softmax-bias 五种模式共用一条通路，照 SM16 的样子接进阵列。查表实现完胜分段线性——INT8 输入只有 256 种取值，查表能和 host 逐位一致。</p>
<ul>
<li>MVP（actv+bias 起步）：净省 82.8M 拍 = <b>−7.7%</b>（5.44→5.03s）；</li>
<li>再加 swin 散射：净省 97.0M 拍 = <b>−9.0%</b>（→4.96s）；</li>
<li>资源预算：约 16k LUT / 7 BRAM / 0 DSP（第一版实测见电路总结页——两模式 5147 LUT，超自设门槛）。</li>
</ul>
<figure class="chart">
<h4>每个算子族搬进片上各能净省多少拍</h4>
<p class="how">怎么看：条子是净省拍（扣掉新引擎自己跑的时间）。单项相加（108M）大于 MVP 合计（82.8M），因为 MVP 只算整条边界上算子全被覆盖而真正消失的往返。</p>
{{SVG:qa_onchip_gain}}
</figure>
<p><b>下一堵墙提前量化了</b>：就算算子融合全做满，编译器行分块边界上的 STORE+LOAD_CTX 还有 222.8M 拍（20.6%）——比全部算子融合加起来还大。所以编译器大段化是比算子融合更大的杠杆，这正是 compute-bound 改造（第六节）主攻的方向。</p>
</div>

<h2>四、编译器修掉两个真 bug，链路做到零实现误差 <span class="date">2026-08-31 上午（代理 3.6 小时）</span></h2>
<div class="entry">
<ol>
<li><b>dma_len 18 位字段溢出</b>：一条 LOAD/STORE 最多搬 262,128 字节，超长搬运的高位会泄进循环控制字段，造成静默少搬数据或 DMA 死循环。修法：编码时就按上限拆成多条。和 05_sim 之前的外科修复<b>精确一致</b>（描述符数 +4,599 一致）。</li>
<li><b>多 tile STORE 覆盖</b>：CTX 放不下时 GEMM 分行 tile 执行，LOAD 带了 tile 偏移但 STORE 没带——后面的 tile 把前面的结果覆盖掉，静态扫描发现 832 处。修复后 31,991 个输出图<b>零重叠零缺口</b>，段输出与 int64 精算 0/1,966,080 处不一致。</li>
</ol>
<p>三道校验门全过：总回写字节 751,525,888 对得上 manifest；RTL 对黄金解释器 6 段位精确；描述符数变化精确。数值结论（W8A8 超判据）见算法总结页——那是量化方案的问题，不是链路的问题。</p>
</div>

<h2>五、指令集：7 条指令，第 8 条（op=6）已被算子引擎占用 <span class="date">2026-08-31 12:53</span></h2>
<div class="entry">
<p>整套架构只有 7 条 256 位指令：GEMM、ATTN_S（矩阵乘+硬件 softmax）、HOIST（循环不变跳过）、COPY（片上重排）、LOAD（DDR→片上，目的地选 CTX 或 WRAM）、STORE（写回）、DONE（段结束）。完整字段位表在答疑页，这里只说本轮的变化：<b>op=6 空位已被 AE_ACTV 引擎占用</b>，b_src 字段复用为子模式（0=ACTV 查表激活、1=BIAS），256 位布局一个没动，新旧指令集兼容。</p>
<figure class="chart">
<h4>一次推理 203,378 条指令都在干什么</h4>
<p class="how">怎么看：LOAD 家族（CTX+W）加 COPY 超过一半条数——memory-bound 在指令构成上的样子。compute-bound 改造目标就是压短这三根条子。</p>
{{SVG:sw_desc_mix}}
</figure>
</div>

<h2>六、compute-bound 改造：正在跑 <span class="running">进行中</span></h2>
<div class="entry">
<p>针对归因账的两大项（STORE 19.1%、LOAD_CTX 15.3%），编译器/调度侧四个杠杆同时改：行分块段界合并、CTX 驻留（算完不扔）、STORE 合并延迟写回、装载提前。<b>通过门：总拍数 −20% 以上、GEMM 占比从 55.4% 提到 ≥70%、数值逐位不变。</b>代理在 09_cbound/ 独立目录里做，不污染主线；结果出来另出页。</p>
</div>

<h2>七、决策点状态</h2>
<div class="entry">
<table>
<tr><th>编号</th><th>问题</th><th>状态</th></tr>
<tr><td>D1</td><td>跑双视角对照拿 LIBERO 档同口径数字（一天）</td><td>待排</td></tr>
<tr><td>D2</td><td>改造路线排序</td><td>已定：on-chip MVP → 编译器大段（compute-bound 代理在跑）</td></tr>
<tr><td>D3</td><td>形状匹配（窄条 GEMM 列组取整）——搬运全消后利用率上限也只有 14.1% 的根源</td><td>待排</td></tr>
<tr><td>D4</td><td>host 算子边界往返（13.8%）</td><td>方案已定型（AE_ACTV），RTL 已落地</td></tr>
<tr><td>D5</td><td>INT4 档位</td><td>推荐档定为 BERT g128 组合（−35.6% 字节），但价值被 D6 gate 住</td></tr>
<tr><td><b>D6</b></td><td><b>量化方案升级（W8A8 端到端超判据 5–7 倍）</b></td><td><b>待拍板</b>，见算法总结页</td></tr>
</table>
</div>
"""

# ============================================================ 算法页
CHARTS['%s_算法总结.html' % FN] = ['algo_mae', 'algo_w4_tiers', 'qa_w4_modules']

algo_tldr = u"""<p style="margin:4px 0"><b>算法这半天一个大红灯、一个绿灯、一个维持原判：</b></p>
<p style="margin:4px 0">① <b>红灯</b>：端到端数值链跑通且链路零实现误差，但 W8A8 全深度量化下关节角误差 0.2993 rad，<b>超 0.045 判据 5–7 倍</b>。五步定界确认是量化方案本身逐层累积，不是 bug——要过判据必须升级量化方案（决策点 D6，待拍板）。</p>
<p style="margin:4px 0">② <b>绿灯</b>：INT4 把 BERT 权重（占字节 53%）换成 4 位，端到端只差 <b>+0.57%</b>（14 个关节 12 个逐位不差）。BERT 不敏感在真实链上坐实，等基座方案修好随时可上。</p>
<p style="margin:4px 0">③ <b>维持原判</b>：数据集继续用 RoboTwin。实测口径对上了账本 RoboTwin 档的 97.4%，想拿 LIBERO 档数字跑双视角对照即可，不用换数据集。</p>"""

algo_body = u"""
<h2>一、端到端数值链定案：链路干净，方案超差 <span class="date">2026-08-31 上午（代理 3.6 小时）</span></h2>
<div class="entry">
<p>编译器两处 bug 修复（见架构总结页）之后，三道校验门全过：</p>
<ul>
<li>回写总字节 751,525,888 与 manifest 逐段一致；</li>
<li>RTL 对黄金解释器 6 段位精确（MAC 对账一致）；</li>
<li>段输出与 int64 精算 <b>0/1,966,080 处不一致</b>；快档引擎与黄金解释器抽 6 段逐位一致。</li>
</ul>
<p>链路本身零实现误差。但全链跑完的结果（判据 0.045 rad）：</p>
<table>
<tr><th>样本</th><th class="num">PL 链 MAE</th><th class="num">fp32 自身重采样底</th><th class="num">超判据</th></tr>
<tr><td>000</td><td class="num">0.2993 rad</td><td class="num">0.0457 rad</td><td class="num">6.6 倍</td></tr>
<tr><td>001</td><td class="num">0.2108 rad</td><td class="num">0.0439 rad</td><td class="num">4.7 倍</td></tr>
</table>
<figure class="chart">
<h4>四种量化变体的端到端误差 vs 判据（样本 000，单位 rad）</h4>
<p class="how">怎么看：上面四根条子（0.29–0.30）不管怎么改校准、怎么放注意力浮点，都挤在一起——说明误差不来自这些地方；它们是底下判据（0.045）的 6.5 倍以上。悬停看每个变体改了什么。</p>
{{SVG:algo_mae}}
</figure>
</div>

<h2>二、五步定界：为什么断定是方案、不是 bug</h2>
<div class="entry">
<ol>
<li><b>看输出形态</b>：fp 输出里动态变化的 4 个关节（8/9/10/13）在量化链下方差几乎归零；joint1 恒 0.669（fp 是恒 0）、左夹爪恒 0.5——正好是先验中心。<b>输出在 64 步时间维上坍缩成常数，动作头退化成输出先验。</b></li>
<li><b>换理想校准表</b>：按本样本各层实际 absmax 重造校准表（192/438 层被放大，说明分布失配是全网性的）重跑——MAE 0.2968，13/14 个关节<b>逐位不变</b>。校准失配不是主因。</li>
<li><b>拆单层误差</b>：第一层 0.1755 的相对误差里 0.1640 来自激活和权重的 int8 本身，requant 和输出量化只加 0.011。且段链实测与量化理论仿真<b>精确相等</b>——链路忠实执行了量化语义。</li>
<li><b>类别二分</b>：只量化 GEMM、注意力全走浮点——MAE 0.2942，和全链几乎一样。<b>GEMM 量化链单独就足以造成全部超差。</b></li>
<li><b>排除 BERT mask</b>：本样本 21 个 token 全有效无 padding，mask 缺失影响为零。</li>
</ol>
</div>

<h2>三、为什么软件量化门是绿灯、硬件链是红灯</h2>
<div class="entry">
<p>08-30 的软件量化门实验给过 W8A8 绿灯（动作 MAE 0.0110）。两个结论不矛盾，差别在<b>量化深度</b>：</p>
<table>
<tr><th></th><th>软件量化门（08-30）</th><th>硬件全链（本轮）</th></tr>
<tr><td>量化位置</td><td>模块边界 fake-quant，算子内部走浮点</td><td>每条 GEMM 的输入和输出都压 INT8</td></tr>
<tr><td>残差流</td><td>保持浮点</td><td>住 INT8 的 CTX，也被量化</td></tr>
<tr><td>结论</td><td>单点量化无害 → 绿灯</td><td>20 多层逐层累积把特征磨到信噪比 1:1 → 红灯</td></tr>
</table>
<p>这和 SwiftVLA 那次"激活 INT8 掉 12pp、权重 INT8 无损"是同一个根：<b>激活量化是主要风险源</b>。软件门的绿灯口径不能外推到部署口径——这条已写进研究记忆，以后量化门一律要补全深度口径。</p>
<div class="warn"><p style="margin:4px 0"><b>D6 待拍板——过判据要换方案，选项四个：</b>①逐 token/逐通道动态激活量化（要 RTL 加 absmax 归约，和 AE_ACTV 引擎同族）；②W8A16（激活保 16 位，数据通路改宽最贵）；③action_head 前特征通路混合精度（误差 73% 集中在 action_head，对症）；④<b>先用 fast_interp 做纯软件扫档（推荐，一天级成本）</b>——快档引擎已证与黄金解释器逐位一致，能把①②③各档在软件里跑一遍，选过了判据且 RTL 代价最小的再落硬件。</p></div>
</div>

<h2>四、INT4：调研三档 + 真实链落地 <span class="date">调研 11:27 / 端到端 A/B 15:05</span></h2>
<div class="entry">
<p>调研结论（07_int4/）：占字节 53% 的 BERT 恰好最不敏感；逐层贪心不可靠（伤害是累积+混沌放大，必须组合实测）；W8 基线跨进程漂移 10%，对比必须同进程。三档方案：</p>
<figure class="chart">
<h4>INT4 三档的权重要多少字节（打包后口径）</h4>
<p class="how">怎么看：推荐档省 35.6% 字节、精度还在 W8 噪声带里；激进档（42–49%，真实坏例 0.056–0.065 rad 超红线）不建议，图中不画。</p>
{{SVG:algo_w4_tiers}}
</figure>
<figure class="chart">
<h4>按模块拆：INT4 动哪里、留哪里</h4>
<p class="how">怎么看：text（BERT）和 vision_2d（Swin）是能动的两块；fusion、action_head、vision_3d 保持 INT8——尤其 action_head 是误差集中地，不要再压。</p>
{{SVG:qa_w4_modules}}
</figure>
<p><b>真实链落地（零改动档，BERT 换 INT4）</b>：编译产物与 W8 对照结构完全一致（3118 段、260,797 描述符），差异恰好 36 段、全部含 text_encoder 权重、无误伤；requant 常数按新尺度重算，有效乘子精确放大 127/7=18.14 倍。端到端 A/B（服务器快档，3118 段跑满）：</p>
<table>
<tr><th></th><th class="num">MAE</th><th class="num">差</th><th>逐关节</th></tr>
<tr><td>W8 对照</td><td class="num">0.2993 rad</td><td class="num">—</td><td>与已发布基线逐位复现（同机同日 A/B 干净）</td></tr>
<tr><td>W4-BERT</td><td class="num">0.3010 rad</td><td class="num">+0.57%</td><td>14 个关节 12 个逐位不差（仅 joint4 自身 +19.6%、joint13 +1.7%）</td></tr>
</table>
<p><b>一个如实更正</b>：之前说"零改动档 −26.7% 字节"，落地时发现只换常数和权重值的话字节节省是 <b>0%</b>——INT4 网格值装在 INT8 容器里，搬运量和 W8 逐字节相同。−26.6% 字节属于 nibble 打包（半字节压缩，w4_packed/ 目录已备好），折合 LOAD_W 拍数 −9.07%、全链 est 拍数 <b>−2.13%</b>（626.9M→613.5M，3158.2→3090.9ms）。全链只有 2.1% 是因为 LOAD_W 只占总拍 23.5%，BERT 又只占 LOAD_W 的 17.5%。</p>
</div>

<h2>五、数据集与视角：RoboTwin 维持原判 <span class="date">2026-08-31 12:53 答疑页</span></h2>
<div class="entry">
<p>三个原因维持 RoboTwin：LIBERO 后训权重未开源（自训要 V100 数天）；HB-GD 必须喂深度而两边用的都是仿真器深度（RoboTwin 用真实内外参+桌面平面合成，是四个输入模态里唯一合成项）；官方小子集 55MB 当天离线跑通。视角区别就是工作量：相机间无交叉注意力，乘加随相机数线性涨——4 相机（149.8G 档）约是 LIBERO 2 视角（73.7G 档）的两倍。想拿 LIBERO 档硬件数字，跑双视角对照（D1，一天）即可，不用换数据集；要 LIBERO benchmark 成功率是算法线的事。完整论证在答疑页。</p>
</div>
"""

# ============================================================ 电路页
CHARTS['%s_电路总结.html' % FN] = ['qa_modules', 'qa_cycle_mix', 'qa_time', 'hw_actv_lut']

hw_tldr = u"""<p style="margin:4px 0"><b>电路这半天三件事：</b></p>
<p style="margin:4px 0">① 从布线后的设计里拉出<b>每个模块的资源账</b>：DSP 1728 个全满、全在乘法阵列；1,469 LUT 的 DMA 承担 41% 的拍数，18,410 LUT 的 copy 重排网络只承担 1.4%——面积和产出的反差一眼可见。</p>
<p style="margin:4px 0">② <b>AE_ACTV 算子引擎 RTL 落地</b>：微观 20480/20480 字节精确、全芯片回归 12/12 位精确，调通过程抓出两个真 RTL bug；综合 5147 LUT + 0 DSP，超自设门槛 106%——BIAS 模式的 16 份无 DSP 乘法是大头，下一轮换乘法结构。</p>
<p style="margin:4px 0">③ INT4 的电路侧准备就绪：nibble 打包目录备好（LOAD_W 拍数 −9.07%），requant 乘子 ×127/7 已实证；慢档 RTL 全链在服务器上跑（约 38 小时）。</p>"""

hw_body = u"""
<h2>一、模块资源账：每个模块占多少面积、承担多少计算 <span class="date">2026-08-31 10:17:33（布线后 DCP）</span></h2>
<div class="entry">
<p>从布线后的 checkpoint 提层级利用率报表，u_core 合计 103,846 LUT（占全片 230,400 的 45.1%）。DSP 一个不剩全在阵列，BRAM 主体是权重存储：</p>
<table>
<tr><th>模块</th><th class="num">LUT</th><th class="num">DSP</th><th>BRAM / URAM</th><th>干什么</th><th class="num">对应拍数占比</th></tr>
<tr><td>u_arr（PE 阵列）</td><td class="num">42,267</td><td class="num">1,728（满片）</td><td>—</td><td>16×108 脉动阵列的乘加</td><td class="num">GEMM 599.1M · 55.4%</td></tr>
<tr><td>u_cp（copy 交叉）</td><td class="num">18,410</td><td class="num">0</td><td>—</td><td>列间转置重排网络</td><td class="num">COPY 15.2M · 1.4%</td></tr>
<tr><td>requant（27 套）</td><td class="num">16,940</td><td class="num">0</td><td>—</td><td>INT32 累加压回 INT8</td><td class="num">计入 GEMM</td></tr>
<tr><td>u_gemm（胶水）</td><td class="num">12,117</td><td class="num">0</td><td>—</td><td>行馈给、写回控制</td><td class="num">计入 GEMM</td></tr>
<tr><td>u_sm（SM16）</td><td class="num">10,164</td><td class="num">0</td><td>—</td><td>16 行并行 softmax</td><td class="num">计入 ATTN_S</td></tr>
<tr><td>u_dma</td><td class="num">1,469</td><td class="num">0</td><td>—</td><td>DDR 搬运引擎</td><td class="num">LOAD+STORE 446.9M · 41.4%</td></tr>
<tr><td>u_core（胶水）</td><td class="num">1,224</td><td class="num">0</td><td>14 RAMB36</td><td>顶层连接</td><td class="num">—</td></tr>
<tr><td>u_sched</td><td class="num">999</td><td class="num">0</td><td>—</td><td>描述符调度状态机</td><td class="num">调度 19.6M · 1.8%</td></tr>
<tr><td>u_ctx</td><td class="num">260</td><td class="num">0</td><td>64 URAM</td><td>激活缓存（CTX）</td><td class="num">被 LOAD_CTX/GEMM 用</td></tr>
<tr><td>WRAM</td><td class="num">—</td><td class="num">0</td><td>108 RAMB36</td><td>权重 bank</td><td class="num">被 LOAD_W/GEMM 用</td></tr>
</table>
<figure class="chart">
<h4>每个模块占多少 LUT</h4>
<p class="how">怎么看：条子越长 LUT 越多。乘法阵列是最大的一块；LUT 排第二的 copy 重排网络干的是"伺候阵列"的活。</p>
{{SVG:qa_modules}}
</figure>
<figure class="chart">
<h4>10.81 亿拍都花在哪（按拍数归因）</h4>
<p class="how">怎么看：和上一张图对照着看——拍数的大头在搬运（LOAD+STORE 合计 41.4%），而搬运引擎 u_dma 只有 1,469 LUT；反过来 copy 交叉 18,410 LUT 只承担 1.4% 的拍。<b>面积大小和干活多少不是一回事。</b></p>
{{SVG:qa_cycle_mix}}
</figure>
<figure class="chart">
<h4>模型各阶段的计算时间占比</h4>
<p class="how">怎么看：decoder 一个阶段吃掉一半时间，阵列（面积最大的模块）在所有阶段都是主要执行者——这张图就是阵列时间的去向。</p>
{{SVG:qa_time}}
</figure>
</div>

<h2>二、AE_ACTV 引擎 RTL：位精确落地，抓出两个真 bug <span class="date">2026-08-31 下午（代理 2.1 小时）</span></h2>
<div class="entry">
<p>on-chip 方案（架构总结页第三节）的第一版 RTL 落地，先做 ACTV（查表激活）+ BIAS 两种模式：</p>
<ul>
<li><b>微观对拍</b>：8 个随机用例（覆盖行组尾数、列宽尾数、表长尾数、负乘子、饱和角落），RTL 与 numpy 黄金逐字节比对 <b>20480/20480 全部精确</b>。</li>
<li><b>全芯片回归</b>：op=6 黄金语义加进 gen_vectors.py，默认用例向量逐字节不变（md5 一致），新增用例在三个位置插 ACTV/BIAS 指令加表装载——四项位精确全过，regression.sh <b>12/12 ALL PASS</b>。</li>
<li><b>抓出的 RTL bug</b>：①尾组行掩码读了还没锁存的行数寄存器，拿到的是上一条描述符的值，掩码全错（修法：行掩码把行数当显式参数传）；②仿真器对 genvar 位选择求值不可靠、表武装写时序差一拍。黄金脚本自己也有一个 bug（把执行完的终态当初态 dump 给测试台），修掉后一个用例从假通过变真通过。</li>
</ul>
<p><b>编码</b>：op=6，b_src 字段复用为子模式（0=ACTV、1=BIAS），其余字段原位复用，256 位布局不动。一个坑记下：<b>ACTV 表映像必须把每个表项复制到所在字的全部 16 个槽位</b>（512B）——CTX 是广播读、按槽位对号，第一版"每个 lane 只收自己那 16 项"的布局是错的。</p>
</div>

<h2>三、AE_ACTV 综合：功能达标，面积超自设门槛一倍</h2>
<div class="entry">
<figure class="chart">
<h4>AE_ACTV 引擎综合后用了多少 LUT（OOC，目标 250MHz）</h4>
<p class="how">怎么看：整引擎 5147 LUT，是调研时自设门槛（2500）的两倍。超在哪：BIAS 模式的 16 份 8b×16b 无 DSP 乘法（4410 LUT）；查表的 ACTV 模式（1307）单看达标。下一轮给 BIAS 换乘法结构——查表分解或共享乘法器分时复用。</p>
{{SVG:hw_actv_lut}}
</figure>
<ul>
<li>DSP 用量为 <b>0</b>（门槛达标——不动满片的乘法器资源）；2 块 BRAM36 存表。</li>
<li>时序 WNS −0.229ns @250MHz，折合约 238MHz，高于实测运行的 198.5MHz 时钟——当前够用。</li>
<li>全片还有约 120k LUT 空余，5k LUT 不构成容量问题；真正的关注点是扩到五模式（norm/rotary/softmax-bias）时的缩放，BIAS 这种乘法结构必须先治。</li>
</ul>
</div>

<h2>四、INT4 的电路侧：打包目录备好，requant 链实证 <span class="date">2026-08-31 下午</span></h2>
<div class="entry">
<ul>
<li><b>nibble 打包</b>（w4_packed/ 已备好）：BERT 权重两个 4 位装一个字节，字节 −50.0%，折合导出总量 −26.6%；LOAD_W 拍数 est −9.07%（147.49M→134.11M），全链 est −2.13%。描述符条数不变——打包对指令流零侵入，只动 DMA 解包。</li>
<li><b>requant 链实证</b>：BERT GEMM 的 requant 常数按 INT4 新尺度重算，有效乘子精确放大 127/7=18.14 倍（例：32390/2<sup>25</sup> → 18364/2<sup>20</sup>），aug bias 幅度同步缩小约 18 倍——常数表通路不用改 RTL。</li>
</ul>
</div>

<h2>五、慢档 RTL 全链在跑，位精确是本轮所有改动的验收标准 <span class="running">进行中</span></h2>
<div class="entry">
<p>慢档（RTL 逐段）全链在服务器 nohup 跑样本 000：47/3118 段（1.5%），约 45 秒/段，预计还需约 38 小时。快慢档段级已证位精确，数值结论预期与快档一致，跑完留完整记录。本轮电路侧的每一步改动（AE_ACTV 引擎、op=6 语义、编译器修复）都走了同一条验收路径：<b>微观对拍 → 全芯片位精确回归 → 综合实测</b>，不带病前进。</p>
</div>
"""

SOURCES_ARCH = u"数据来源：05_sim 拍数实测与归因（cycles_by_type.json）；research_holobrain 账本（holobrain_hw.json）；07_onchip_ops/boundary_account.json（host 边界往返）；03_compiler/NOTES.txt（编译器修复与三门校验）；图表为 gen_charts.js / gen_charts2.js 生成的内联 SVG。"
SOURCES_ALGO = u"数据来源：03_compiler/NOTES.txt 与 build_s000_v3 / build_s001_v3 / build_s000_ideal / build_s000_w4（端到端各变体，服务器 /tmp/ae_v3、/tmp/ae_w4）；07_int4/（三档调研）；02_quant（08-30 软件量化门）；04_dataset/DATASET_NOTES.txt。"
SOURCES_HW = u"数据来源：E:/ae_syn/pwr/hier_util.rpt（布线后 DCP 层级账，Vivado 2021.2 xczu7ev）；09_onchip_rtl/（AE_ACTV RTL、微观与全芯片对拍、OOC 综合报告）；09_int4_impl/w4_packed/（nibble 打包）；05_sim（慢档 RTL 全链）。"

SCOPE = u"2026-08-31 上午三份工作记录之后 → 本页生成时刻"

make_page(u'%s_架构总结.html' % FN, u'架构总结：16× 差距拆完，memory-bound 确诊，compute-bound 改造在跑（2026-08-31）',
          SCOPE, arch_tldr, arch_body, SOURCES_ARCH)
make_page(u'%s_算法总结.html' % FN, u'算法总结：链路零误差但 W8A8 全深链超判据 5–7 倍，INT4 落地无感（2026-08-31）',
          SCOPE, algo_tldr, algo_body, SOURCES_ALGO)
make_page(u'%s_电路总结.html' % FN, u'电路总结：模块资源账、AE_ACTV 引擎位精确落地、INT4 打包备好（2026-08-31）',
          SCOPE, hw_tldr, hw_body, SOURCES_HW)
