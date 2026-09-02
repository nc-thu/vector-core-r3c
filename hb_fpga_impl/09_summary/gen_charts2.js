// 2026-08-31 三份总结页的新增图表（复用 08_report/gen_charts.js 的规范与样式）
// 规范：单系列量级对比 → 单一蓝 #2a78d6、细条、数据端 4px 圆角、基线锚定、
// 网格退隐 #e1e0d9、刻度灰 #898781、悬停 tooltip 由页面 JS 提供（data-tip）。
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'charts');
fs.mkdirSync(OUT, { recursive: true });

const C = {
  series: '#2a78d6',
  grid: '#e1e0d9',
  base: '#c3c2b7',
  tick: '#898781',
  label: '#52514e',
  ink: '#0b0b0b',
};
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// 水平条形图。rows: [{name, value, label, tip}]，axis = x 轴最大值
function hbar(id, title, rows, axis, unit, ticks, vw = 680, plotW = 330) {
  const x0 = 175, x1 = x0 + plotW;
  const rowH = 36, barH = 18;
  const vh = rows.length * rowH + 46;
  let s = `<svg viewBox="0 0 ${vw} ${vh}" role="img" aria-label="${esc(title)}" style="width:100%;height:auto;display:block">`;
  for (const t of ticks) {
    const x = x0 + (t / axis) * plotW;
    s += `<line x1="${x}" y1="10" x2="${x}" y2="${vh - 30}" stroke="${C.grid}" stroke-width="1"/>`;
    s += `<text x="${x}" y="${vh - 14}" text-anchor="middle" font-size="11" fill="${C.tick}">${t}${unit}</text>`;
  }
  s += `<line x1="${x0}" y1="10" x2="${x0}" y2="${vh - 30}" stroke="${C.base}" stroke-width="1"/>`;
  rows.forEach((r, i) => {
    const y = 14 + i * rowH;
    const w = Math.max(1, (r.value / axis) * plotW);
    const rad = Math.min(4, w / 2);
    const d = `M${x0},${y} L${x0 + w - rad},${y} A${rad},${rad} 0 0 1 ${x0 + w},${y + rad} L${x0 + w},${y + barH - rad} A${rad},${rad} 0 0 1 ${x0 + w - rad},${y + barH} L${x0},${y + barH} Z`;
    s += `<path d="${d}" fill="${C.series}" data-tip="${esc(r.tip)}"/>`;
    s += `<text x="${x0 - 10}" y="${y + 13.5}" text-anchor="end" font-size="12.5" fill="${C.label}">${esc(r.name)}</text>`;
    s += `<text x="${x0 + w + 8}" y="${y + 13.5}" font-size="12.5" font-weight="600" fill="${C.ink}">${esc(r.label)}</text>`;
  });
  s += `</svg>`;
  fs.writeFileSync(path.join(OUT, id + '.svg'), s);
}

// ---------- 算法页 ----------

// ALGO1 端到端 MAE：四个"超差"变体 vs fp 噪声底 vs 判据
hbar('algo_mae', '端到端关节角误差（样本 000，单位 rad）', [
  { name: 'W4-BERT 全链', value: 0.3010, label: '0.3010', tip: 'INT4 只换 BERT 权重：0.3010 rad，比 W8 基线 +0.57%；14 个关节 12 个逐位不差——BERT 不敏感在真实链上坐实' },
  { name: 'W8A8 全链（基线）', value: 0.2993, label: '0.2993', tip: 'per-tensor W8A8 全深度量化：样本 000 MAE 0.2993 rad（样本 001 为 0.2108）——超判据 5～7 倍' },
  { name: '理想校准表重跑', value: 0.2968, label: '0.2968', tip: '按本样本各层实际 absmax 重造校准表（192/438 层被放大）再跑：13/14 关节逐位不变——校准失配不是主因' },
  { name: '只量化 GEMM（注意力全 fp）', value: 0.2942, label: '0.2942', tip: '注意力全走浮点：0.2942 rad，与全链几乎一样——几乎全部误差来自 GEMM 量化链' },
  { name: 'fp32 自身重采样底', value: 0.0457, label: '0.0457', tip: 'fp32 输出重采样的噪声底：0.0457（样本 001 为 0.0439）——判据就定在这个量级' },
  { name: '端到端判据', value: 0.0450, label: '0.045', tip: '判据 0.045 rad。上面四根条子是它的 6.5～6.7 倍' },
], 0.34, '', [0, 0.1, 0.2, 0.3]);

// ALGO2 INT4 三档：字节（打包后口径）
hbar('algo_w4_tiers', 'INT4 三档的权重要多少字节（打包后口径）', [
  { name: 'W8 基线（全部 INT8）', value: 159.5, label: '159.5 MB', tip: '当前部署：全部权重 INT8，导出 159.5MB' },
  { name: '零改动档（BERT INT4）', value: 117.0, label: '117.0 MB · −26.7%', tip: '只换 BERT 常数表。仿真精度 0.0187，端到端实测 +0.57%（无感）。注意：int4 值装在 int8 容器里不打包则字节为 0 节省，−26.7% 是 nibble 打包后的口径' },
  { name: '推荐档（BERT+Swin g128+neck）', value: 102.6, label: '102.6 MB · −35.6%', tip: '0.0231 rad（真实最差 0.0315），与 W8 同噪声带，低于 0.04 红线。代价：组 scale 要在累加时反量化（RTL 项）' },
], 170, ' MB', [0, 50, 100, 150]);

// ---------- 电路页 ----------

// HW-ACTV AE_ACTV 综合拆解
hbar('hw_actv_lut', 'AE_ACTV 引擎综合后用了多少 LUT（OOC，目标 250MHz）', [
  { name: '两模式合计（整引擎）', value: 5147, label: '5147 · 超门槛 106%', tip: '4507 逻辑 LUT + 640 LUTRAM + 985 FF + 2 BRAM36 + 0 DSP；WNS −0.229ns @250MHz ≈ 238MHz，高于实测 198.5MHz 时钟' },
  { name: '其中 BIAS 模式', value: 4410, label: '4410', tip: '16 份 8b×16b 无 DSP 乘法，每 lane 约 234 LUT——超支的大头，下一轮换乘法结构（查表分解或共享乘法器分时复用）' },
  { name: '自设门槛（两模式合计）', value: 2500, label: '2500', tip: '调研时给 ACTV+BIAS 两模式定的预算' },
  { name: '其中 ACTV 模式（查表）', value: 1307, label: '1307 · 达标', tip: '667 逻辑 LUT + 640 LUTRAM，查表实现，单看在门槛内' },
], 5600, '', [0, 1000, 2000, 3000, 4000, 5000]);

console.log('charts written to', OUT);
