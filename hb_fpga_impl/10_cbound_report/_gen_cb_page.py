# -*- coding: utf-8 -*-
# 2026-08-31 compute-bound"解决方案与效果"页生成器（10_cbound_report/，新轮次新文件夹）
import datetime, io, os

NOW = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
FN = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")

html = u"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>compute-bound 改造：解决方案与效果 · 2026-08-31</title>
<style>
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
pre{background:#0b0b0b;color:#e8e6df;font-size:12px;line-height:1.6;padding:12px 14px;border-radius:8px;overflow-x:auto}
pre code{background:none;color:inherit}
#tip{position:fixed;opacity:0;pointer-events:none;background:#0b0b0b;color:#fff;font-size:12.5px;line-height:1.5;padding:6px 10px;border-radius:6px;max-width:280px;z-index:9;transition:opacity .12s}
footer{margin-top:44px;padding-top:14px;border-top:1px solid #e1e0d9;font-size:12.5px;color:#898781}
</style>
</head>
<body>
<div id="tip"></div>
<div class="wrap">

<header class="page">
<h1>compute-bound 改造：解决方案与效果</h1>
<div class="meta">
<span>生成时刻：<b>{{NOW}}</b></span>
<span>快报 17:2x / 最终报告 17:4x（代理收工）</span>
<span>产物：09_cbound/（本地）+ /tmp/ae_cb（服务器，1.3GB 保留）</span>
</div>
<nav class="pagenav">
<a href="../09_summary/2026-08-31_1624_架构总结.html">架构总结</a>
<a href="../09_summary/2026-08-31_1624_电路总结.html">电路总结</a>
<a href="../08_report/2026-08-31_1128_实测vs账本差距问题讨论.html">问题讨论页 v3</a>
<a href="../WORKLOG.md">完整工作日志</a>
</nav>
</header>

<div class="ok">
<p style="margin:4px 0"><b>一段话效果：</b>只改编译器的调度、布局和段界（数值语义一行没动），一次推理总拍数从 1216.5M 降到 <b>986.4M（−18.9%）</b>，折算时间 6.13ms→<b>4.97ms</b> @198.5MHz；激活装载字节 <b>−45.7%</b>（1543→838MB）。数值逐位不变的验证全过：静态冲突检查三档零错误，Verilator 抽 15 段位精确全 PASS。拍数模型外样准确率 <b>99.9%</b>（用户门是 95%）。再往下砍要动 RTL——需求清单已带量化收益排好序，前两项（异步写回 + 激活预取）组合还能再降约 <b>340M 拍（−34%）到 646M/3.25ms</b>，等拍板。</p>
</div>

<h2>一、三档方案：改了什么、各降多少 <span class="date">2026-08-31 17:4x</span></h2>
<div class="entry">
<table>
<tr><th>档</th><th>改了什么</th><th class="num">段数</th><th class="num">总拍</th><th class="num">降幅</th></tr>
<tr><td><b>repro</b>（基线复刻）</td><td>没改调度，只复刻基线做对照。3118 个指令流文件与服务器原版 <b>md5 逐一全同</b>，已测段的拍数与此前实测逐拍一致——对照锚点是闭合的</td><td class="num">3118</td><td class="num">1216.5M</td><td class="num">—</td></tr>
<tr><td><b>a1</b>（段界合并）</td><td>DDR 档位 8MB→64MB（8MB 只是测试台遗留，地址线 32 位本来就能寻 4GB）；列块与行分块的段界合并，消掉激活图重复搬运和重复喂行</td><td class="num">2762</td><td class="num">1134.0M</td><td class="num"><b>−6.8%</b></td></tr>
<tr><td><b>a2</b>（a1+装载优化）</td><td>多头段（rotary、k/v_proj 等 66 段）的激活图从<b>每头重装 16 次</b>改成整段装一次；V-T 双子段单装；权重驻留表（同内容不重发）；权重装载贴着 GEMM 发射、被预取窗口 100% 遮蔽</td><td class="num">2762</td><td class="num">986.4M</td><td class="num"><b>−18.9%</b></td></tr>
</table>
<figure class="chart">
<h4>三档总拍数（百万拍）</h4>
<p class="how">怎么看：两步改造一共砍掉 230M 拍。杠杆 A（段界合并）贡献 −82.5M，杠杆 B/C/D（装载优化）贡献 −147.6M——装载优化比段界合并更值钱。悬停看每档细节。</p>
{{SVG:cb_total}}
</figure>
</div>

<h2>二、钱从哪省的：一项一项对账</h2>
<div class="entry">
<figure class="chart">
<h4>repro → a2 每一项省了多少拍（百万拍）</h4>
<p class="how">怎么看：省的 230M 拍里 83% 来自激活装载（LOAD_CTX −46%）。STORE 和 COPY 一拍没省——不是没做，是证伪了（见第八节）：写回的 751.5MB 全是唯一输出，没有冗余可删。</p>
{{SVG:cb_savings}}
</figure>
<figure class="chart">
<h4>基线的拍数都花在哪（新锚定口径）</h4>
<p class="how">怎么看：新口径下第一大项是激活装载（34.3%），不是之前页子里写的 GEMM 55.4%——那是一个账本 bug，第三节专门说清。</p>
{{SVG:cb_mix_repro}}
</figure>
<figure class="chart">
<h4>改造后拍数花在哪（a2）</h4>
<p class="how">怎么看：搬运压下去之后，GEMM（34.1%）自然升为第一大项，写回（28.0%）第二。单 DMA 引擎串行工作下，GEMM 占比推不过 ~34%——再往上只能靠 RTL。</p>
{{SVG:cb_mix_a2}}
</figure>
<p>字节口径的账：LOAD_CTX 从 1543.3MB 降到 837.6MB（<b>−45.7%</b>），其中段内重复装载从 527.0MB 降到 53.5MB（<b>−89.8%</b>）——多头重装和段界重复这两类浪费基本清干净了。</p>
</div>

<h2>三、口径对账：旧页数字作废清单（重要更正）</h2>
<div class="entry">
<div class="warn">
<p style="margin:4px 0"><b>之前页面里的这几组数字作废，以本页为准：</b></p>
<ul style="margin:4px 0">
<li>"GEMM 599.1M / 55.4%"——老账本用<b>全局列数</b>而不是<b>实际列数</b>算写回拍，GEMM 高估约 66%；RTL 实测计数器里 GEMM 只有 ~330-360M（占 ~30%）。</li>
<li>"STORE 206.6M / 19.1%、LOAD_CTX 164.9M / 15.3%"——旧账只算理想拍，没算 DDR 从机（1 拍/2 周期）和 LFSR 停顿的真实开销，DMA 项被系统性低估。</li>
<li>"实测 1080.8M 拍"——那是老模型的估计值，不是本轮逐段实测口径的和。本轮锚定口径基线是 <b>1216.5M</b>（差异三源：指令流本身不同 2782 vs 3118 段、GEMM 高估、DMA 低估）。</li>
<li>归因排序翻案：<b>旧口径 GEMM&gt;STORE&gt;LOAD_CTX；新口径 LOAD_CTX 34.3% &gt; GEMM 28.5% &gt; STORE 22.7%</b>。方向结论不变（memory-bound），但第一大项认错了。</li>
</ul>
</div>
<p>旧页面（问题讨论页 v1–v3、三份总结）按"不改旧文件"的规矩原样保留作历史，其中归因图的数字以本页第三节为准。当时基于旧口径给的改造优先级（"STORE 是第一杠杆"）相应修正为：<b>激活装载才是编译器侧最大的杠杆（本轮已兑现 −46%），写回是 RTL 侧第一杠杆（R1）</b>。</p>
</div>

<h2>四、拍数模型：95% 的门，实际 99.9%</h2>
<div class="entry">
<p>按新规矩（子实验 ≤10 分钟、预测准确率 ≥95%），拍数对比全部走软件模型：用基线已实测的 1468 段最小二乘拟合 5 个每拍成本系数（GEMM×1.05、STORE×1.18、LOAD_CTX×2.14、LOAD_W×2.13、COPY×1.10，每段 −916 固定）。</p>
<ul>
<li>拟合段内：逐段偏差中位数 0.7%、p90 6.8%、<b>总量偏差 0.00%</b>；</li>
<li><b>外样检验</b>（关键——模型对改造后的指令流没见过）：a2 已实测 676 段总量偏差 <b>−0.11%</b>，a1 为 −0.37%；</li>
<li>结论：986.4M 这个数可信度 ±1%。以后方案对比全走这个模型，每次 ≤10 分钟。</li>
</ul>
</div>

<h2>五、数值逐位不变：15 段位精确门全过</h2>
<div class="entry">
<p>改造只动调度/布局，数值语义零改动——两道门验证这一点：</p>
<ul>
<li><b>静态冲突门</b>（逐存储格写者追踪）：repro 3118 段、a1/a2 各 2762 段，全部 <b>ERR=0 WARN=0</b>；</li>
<li><b>Verilator 位精确门</b>（随机输入，写回窗口逐字节对黄金解释器）：a2 抽 10 段 + a1 抽 5 段，<b>15/15 PASS</b>。抽样覆盖了所有改造触到的特征：patch_embed 大图（m=20480）、swin 多头、ffn 列块合并、softmax+COPY、V-T 双子段、rotary 头外重排、32MB 大段寻址（24.4MB 写回窗口）、重搬运段（336 条 COPY）。</li>
</ul>
<p>另验三个闭环：纯 Python 黄金 == fast_interp == RTL 三方逐位一致（3 段）；拍数与数据内容无关（空 DDR 跑出相同拍数）；exp 查表与黄金语义全同。</p>
</div>

<h2>六、编译器已经到头了：残余流量的下限</h2>
<div class="entry">
<table>
<tr><th>项</th><th class="num">改造后剩余</th><th>构成与下限</th></tr>
<tr><td>LOAD_CTX</td><td class="num">837.6MB</td><td>首装 784.1MB（段自包含契约：每段 host 送新输入，免不了）+ 段内双装 53.5MB（还能再消，约 −15M 拍，本轮没做——收益小于再过一轮门的成本）</td></tr>
<tr><td>LOAD_W</td><td class="num">587.1MB</td><td>首装 438.6MB + 重装 148.5MB（WRAM 只有 2 个半区，≥3 组权重的段结构性重装；暴露的只有 48.6M 拍，大部分被预取藏住）</td></tr>
<tr><td>STORE</td><td class="num">751.5MB</td><td><b>零冗余</b>——覆盖重写 0 字节。调度层已到头，再降只能 RTL</td></tr>
<tr><td>GEMM</td><td class="num">336.4M 拍</td><td>理想拍 319.7M 里只有 133.9M 是稳态乘加，<b>58.1%（185.8M）是每个行组 ~199 拍的固定开销</b>（喂行/排空/写回），69.2 万个行组——R3 的靶子</td></tr>
</table>
</div>

<h2>七、RTL 需求清单：下一步的量化菜单（等拍板）</h2>
<div class="entry">
<p>编译器尽力后的 a2 = 986.4M（4.97ms）。全部 RTL 项做完的理论下限推演：单 DMA 引擎的服务时间 = 写回 276 + 激活装载 226 + 权重装载 104 = 606M 拍，所以 DMA 满载、其余全藏的下限约 <b>646M（3.25ms）</b>。</p>
<figure class="chart">
<h4>RTL 需求清单：每项的拍数上限（百万拍）</h4>
<p class="how">怎么看：R1+R2 逐项上限合计 502.7M，但受单 DMA 引擎服务时间约束（606M），<b>组合可实现上限约 340M</b>（986.4→~646M）。要突破 646M 必须 R3（GEMM 内部行组开销）+R6（真机带宽）。</p>
{{SVG:cb_rtl}}
</figure>
<table>
<tr><th>#</th><th>需求</th><th class="num">拍数上限</th><th>代价粗估</th><th>风险</th></tr>
<tr><td>R1</td><td>异步 STORE 队列（写回与计算重叠，独立写通道）</td><td class="num">276M</td><td>~3-5k LUT + 6-10 BRAM36，改 ae_dma/ae_sched</td><td>低</td></tr>
<tr><td>R2</td><td>CTX A 预取（预取状态机扩激活标签+输入双缓冲）</td><td class="num">226M</td><td>预取解码 + CTX 端口仲裁</td><td>中</td></tr>
<tr><td>R3</td><td>GEMM 行组间流水（免逐组排空/重排）</td><td class="num">~130M</td><td>控制状态机，面积近零</td><td>中（时序收敛）</td></tr>
<tr><td>R4</td><td>WRAM 驻留 2→4 组</td><td class="num">10-20M</td><td>每列 +2Mb，约 +36 BRAM36 等效；主要价值是给 R1/R2 腾带宽</td><td>低</td></tr>
<tr><td>R5</td><td>COPY 消除（DMA 列主序路由）</td><td class="num">40M</td><td>DMA 路由模式或第二 CTX 读口</td><td>中</td></tr>
<tr><td>R6</td><td>真机 DDR 口径</td><td class="num">口径项</td><td>本报告系数基于测试台从机（1 拍/2 周期+LFSR）；真机 HP 口径 64B/拍下 DMA 服务 606M→约 150-250M，GEMM 才成主线</td><td>—</td></tr>
</table>
<p><b>建议的拍板顺序</b>：R1（低风险、单刀 276M）→ R2（与 R1 合计兑现 ~340M）→ 视真机口径决定 R3。R1/R2 都只动 DMA/调度，不碰阵列和数值通路，位精确门现成。</p>
</div>

<h2>八、证伪清单（如实报告，防止后人再试）</h2>
<div class="entry">
<ol>
<li><b>STORE 合并 / 补零修剪：证伪</b>。751.5MB 全是唯一输出（覆盖重写 0 字节）；补零在 16 字节字内的 lane 上，没有多余字节可省。</li>
<li><b>跨段 CTX 驻留：未做</b>。要改段自包含契约（host 驱动和黄金模型都得跟着改），不是纯调度变换——列入 host/RTL 杠杆，不在本轮。</li>
<li><b>"GEMM 占比提到 70%"：口径不存在</b>。55.4% 基数本身是账本 bug；单 DMA 串行下 STORE+LOAD 占 ~63%，调度器无解，本轮实测最高推到 34.1%。</li>
<li><b>段内 CTX 图缓存</b>（消最后 53.5MB 双装，~15M 拍/1.5%）：已识别未实施，收益小于再过一轮门的成本。</li>
</ol>
</div>

<h2>九、复现命令</h2>
<div class="entry">
<pre><code># 本地（hb_fpga_impl/09_cbound/）
python compiler.py --w8 w8_export --calib ../02_quant/hw_calib_table_v2.json \
  --attn-calib ../02_quant/attn_calib.json --profile full --out build_a2
python check_streams.py build_a2        # 静态冲突门（三档全过）
python calib.py                         # 模型系数 + 外样准确率（读 cycles_build_*.tsv）
python gate_rtl.py gen build_a2 seg_0000 ...   # 数值门输入
python gate_rtl.py cmp build_a2 seg_0000 ...   # 逐字节比对

# 服务器（/tmp/ae_cb，数据保留）
bash vlbuild_cb.sh                      # Verilator（64MB DDR 模型 + 看门狗 4e7）
cd gate/&lt;seg&gt; &amp;&amp; ../obj_dir/Vtb_ae_v +MODE=1 +PF=1 +SEQ=seq.mem +DDRIMG=ddr_init.mem \
  +DUMP=dump.mem +MASK=mask.mem         # 数值门单段（+MASK 稀疏 dump 为本轮 TB 新增）
bash cyc.sh &lt;build&gt; &lt;seg&gt;               # 拍数实测（+DUMP=/dev/null 防全量 dump）</code></pre>
<p>坑两条已记录：TB 不给 +DUMP 时默认全量 dump（曾写满 /tmp 93GB，已删并加防护）；双队列重复跑同段（已杀重，拍数表去重后无影响）。</p>
</div>

<h2>十、当前状态与下一步</h2>
<div class="entry">
<ul>
<li>服务器无遗留进程；/tmp/ae_cb 1.3GB 数据保留（三套拍数表 + 门产物）。</li>
<li><b>等拍板一</b>：要不要重启硬件线做 R1+R2（异步写回 + 激活预取，组合 ~340M 拍 → 646M/3.25ms）。都只动 DMA/调度，不碰数值通路。</li>
<li><b>等拍板二（挂起）</b>：D6 量化方案升级——算法线已停，此决策挂起，重启时从 fast_interp 软件扫档开始。</li>
<li>还有一台账没动：双视角对照（D1，一天）可以拿 LIBERO 档同口径数字，与 compute-bound 无冲突。</li>
</ul>
</div>

<footer>
<p>数据来源：09_cbound/（compiler.py 三档、check_streams.py 静态门、calib.py 模型系数与外样、gate_rtl.py 数值门、cycles_build_{repro,a1,a2}.tsv 实测拍数）；服务器 /tmp/ae_cb（改版 TB：64MB DDR 模型、稀疏 dump；1.3GB 保留）。口径基线 1216.5M 为逐段 RTL cycles 计数器之和（MODE=1+PF=1，含 DDR 从机与 LFSR 开销），与旧账 1080.8M 不可混用（第三节）。</p>
<p>生成时刻：<b>{{NOW}}</b>。图表为 gen_charts3.js 生成的内联 SVG，悬停看细节。</p>
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

for name in ['cb_total', 'cb_savings', 'cb_mix_repro', 'cb_mix_a2', 'cb_rtl']:
    with io.open(os.path.join('charts', name + '.svg'), 'r', encoding='utf-8') as f:
        html = html.replace('{{SVG:' + name + '}}', f.read().strip())
html = html.replace('{{NOW}}', NOW)
assert '{{' not in html

out = u'%s_compute-bound改造方案与效果.html' % FN
with io.open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print('written:', out, len(html), 'chars')
