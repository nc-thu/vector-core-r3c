// 2026-08-31 compute-bound 解决方案与效果页图表（规范同 08_report/gen_charts.js）
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

function vbar(id, title, rows, axis, unit, ticks, vw = 560) {
  const plotH = 200, yBase = 240, top = 40;
  const bw = 76, gap = 44;
  const total = rows.length * bw + (rows.length - 1) * gap;
  const x0 = (vw - total) / 2;
  const vh = 285;
  let s = `<svg viewBox="0 0 ${vw} ${vh}" role="img" aria-label="${esc(title)}" style="width:100%;height:auto;display:block">`;
  for (const t of ticks) {
    const y = yBase - (t / axis) * plotH;
    s += `<line x1="${x0 - 30}" y1="${y}" x2="${vw - x0 + 30}" y2="${y}" stroke="${C.grid}" stroke-width="1"/>`;
    s += `<text x="${x0 - 36}" y="${y + 4}" text-anchor="end" font-size="11" fill="${C.tick}">${t}${unit}</text>`;
  }
  s += `<line x1="${x0 - 30}" y1="${yBase}" x2="${vw - x0 + 30}" y2="${yBase}" stroke="${C.base}" stroke-width="1"/>`;
  rows.forEach((r, i) => {
    const x = x0 + i * (bw + gap);
    const h = Math.max(1, (r.value / axis) * plotH);
    const y = yBase - h;
    const rad = Math.min(4, h / 2);
    const d = `M${x},${yBase} L${x},${y + rad} A${rad},${rad} 0 0 1 ${x + rad},${y} L${x + bw - rad},${y} A${rad},${rad} 0 0 1 ${x + bw},${y + rad} L${x + bw},${yBase} Z`;
    s += `<path d="${d}" fill="${C.series}" data-tip="${esc(r.tip)}"/>`;
    s += `<text x="${x + bw / 2}" y="${y - 8}" text-anchor="middle" font-size="12.5" font-weight="600" fill="${C.ink}">${esc(r.label)}</text>`;
    const lines = r.name.split('\n');
    lines.forEach((ln, j) => {
      s += `<text x="${x + bw / 2}" y="${yBase + 18 + j * 15}" text-anchor="middle" font-size="12" fill="${C.label}">${esc(ln)}</text>`;
    });
  });
  s += `</svg>`;
  fs.writeFileSync(path.join(OUT, id + '.svg'), s);
}

// CB1 总拍阶梯
vbar('cb_total', '三档总拍数（百万拍）', [
  { name: '基线复刻\nrepro', value: 1216.5, label: '1216.5 M', tip: '基线复刻：3118 段与服务器原版 md5 逐一全同，已测段拍数与 gate2 实测逐拍一致（=6.13ms @198.5MHz）' },
  { name: '杠杆A\na1 段界合并', value: 1134.0, label: '1134.0 M · −6.8%', tip: '杠杆A：DDR 档 8MB→64MB（TB 遗产，32b 地址本可寻 4GB）+ 列 chunk 与行分块段界合并，3118→2762 段' },
  { name: 'A + B/C/D\na2 装载优化', value: 986.4, label: '986.4 M · −18.9%', tip: 'a2 = a1 + 多头段 A 图整段单装（原每头重装 16 次）+ V-T 单装 + WRAM 驻留表 + LOAD_W 贴发射吃满预取窗口（=4.97ms @198.5MHz）' },
], 1400, '', [0, 400, 800, 1200]);

// CB2 每项省了多少
hbar('cb_savings', 'repro → a2 每一项省了多少拍（百万拍）', [
  { name: 'LOAD_CTX（激活装载）', value: 190.3, label: '−190.3 M · −46%', tip: '多头段 A 图从每头重装 16 次改为整段一次装载 + 段界合并消重复搬运。字节口径 1543.3→837.6MB（−45.7%），段内重装 −89.8%' },
  { name: 'LOAD_W（权重装载）', value: 28.6, label: '−28.6 M · −22%', tip: 'WRAM 驻留表（同内容半区命中不重发）+ LOAD_W 紧贴 GEMM 发射，预取命中 6318→9702 次' },
  { name: 'GEMM', value: 10.8, label: '−10.8 M · −3%', tip: '段界合并消掉 M-feed 重复喂行' },
  { name: 'STORE（写回）', value: 0, label: '0 · 持平', tip: '751.5MB 回写全是唯一输出（覆盖重写 0 字节）——调度层已到头，再降只能靠 RTL 异步队列' },
  { name: 'COPY（重排）', value: 0, label: '0 · 持平', tip: '重排结构没变，本轮不动它' },
], 220, '', [0, 50, 100, 150, 200]);

// CB3 基线分量（新口径）
hbar('cb_mix_repro', '基线的拍数都花在哪（新锚定口径，repro=1216.5M）', [
  { name: 'LOAD_CTX', value: 416.6, label: '416.6 M · 34.3%', tip: '激活装载是新口径下的第一大项——旧账把它记成 15.3% 是因为没算 DDR 从机与 LFSR 开销' },
  { name: 'GEMM', value: 347.2, label: '347.2 M · 28.5%', tip: '矩阵乘本体。旧账的 599.1M/55.4% 用了全局列数而非实际列数，高估约 66%' },
  { name: 'STORE', value: 276.5, label: '276.5 M · 22.7%', tip: '写回 DDR，751.5MB 全是唯一输出' },
  { name: 'LOAD_W', value: 132.2, label: '132.2 M · 10.9%', tip: '权重装载，大部分已被预取遮蔽' },
  { name: 'COPY', value: 40.3, label: '40.3 M · 3.3%', tip: '片上重排' },
], 450, '', [0, 100, 200, 300, 400]);

// CB4 a2 分量
hbar('cb_mix_a2', '改造后拍数花在哪（a2=986.4M）', [
  { name: 'GEMM', value: 336.4, label: '336.4 M · 34.1%', tip: '改造后 GEMM 升为第一大项——搬运被压下去，计算占比自然上来' },
  { name: 'STORE', value: 276.4, label: '276.4 M · 28.0%', tip: '写回一项没动：751.5MB 全唯一，下一步是 RTL 异步 STORE 队列（上限 276M）' },
  { name: 'LOAD_CTX', value: 226.3, label: '226.3 M · 22.9%', tip: '从 34.3% 压到 22.9%；剩余=首装 784MB（段自包含契约）+双装 53.5MB' },
  { name: 'LOAD_W', value: 103.6, label: '103.6 M · 10.5%', tip: '重装 148.5MB 大部分被预取藏住' },
  { name: 'COPY', value: 40.3, label: '40.3 M · 4.1%', tip: '不变' },
], 450, '', [0, 100, 200, 300, 400]);

// CB5 RTL 需求清单
hbar('cb_rtl', 'RTL 需求清单：每项的拍数上限（百万拍）', [
  { name: 'R1 异步 STORE 队列', value: 276, label: '276 M', tip: '写回与 GEMM/DMA 读重叠，独立写通道。代价 ~3-5k LUT + 6-10 BRAM36，改 ae_dma/ae_sched，风险低' },
  { name: 'R2 CTX A 预取', value: 226, label: '226 M', tip: 'pf 状态机扩 TAG_C + CTX 输入双缓冲。风险中' },
  { name: 'R3 GEMM 行组流水', value: 130, label: '~130 M（可回收）', tip: '行组固定开销 185.8M 的 70%：免逐组 drain/重排。面积近零，风险在时序收敛' },
  { name: 'R5 COPY 消除', value: 40, label: '40 M', tip: 'DMA 列主序路由把 COPY 变 LOAD_W 变体' },
  { name: 'R4 WRAM 2→4 组', value: 15, label: '10-20 M', tip: '主要价值是给 R1/R2 腾 DMA 带宽（+36 BRAM36 等效）' },
], 320, '', [0, 100, 200, 300]);

console.log('charts written to', OUT);
