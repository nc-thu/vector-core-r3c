// 2026-08-31 工作记录报告的图表生成器（一次性，产物内联进三个 HTML）
// 规范：单系列量级对比 → 单一蓝色 #2a78d6、细条、数据端 4px 圆角、基线锚定、
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

function fmt(n) { return n.toLocaleString('en-US'); }
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// 水平条形图。rows: [{name, value, label, tip, pct}]，axis = x 轴最大值
function hbar(id, title, rows, axis, unit, ticks, vw = 680, plotW = 330) {
  const x0 = 175, x1 = x0 + plotW;
  const rowH = 36, barH = 18;
  const vh = rows.length * rowH + 46;
  let s = `<svg viewBox="0 0 ${vw} ${vh}" role="img" aria-label="${esc(title)}" style="width:100%;height:auto;display:block">`;
  // 网格 + 刻度
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

// 垂直条形图（顶端圆角）
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
    s += `<text x="${x0 - 36}" y="${y + 4}" text-anchor="end" font-size="11" fill="${C.tick}">${fmt(t)}</text>`;
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
    // 两行类目标签
    const lines = r.name.split('\n');
    lines.forEach((ln, j) => {
      s += `<text x="${x + bw / 2}" y="${yBase + 18 + j * 15}" text-anchor="middle" font-size="12" fill="${C.label}">${esc(ln)}</text>`;
    });
  });
  s += `</svg>`;
  fs.writeFileSync(path.join(OUT, id + '.svg'), s);
}

// ---------- 硬件页 ----------

// HW1 资源占用（% of 总量）
hbar('hw_resource', 'FPGA 资源占用（xczu7ev）', [
  { name: 'DSP（乘法器）', value: 100.0, label: '100% · 1728 / 1728（满）', tip: 'DSP 乘法器：1728 个全部用满。扩列只能靠一个 DSP 算两个 INT8 乘（packed PE，未实测）' },
  { name: 'URAM', value: 66.7, label: '66.7% · 64 / 96', tip: 'URAM：64 / 96。主要给上下文缓存（CTX）' },
  { name: 'LUT（逻辑）', value: 47.9, label: '47.9% · 110,465 / 230,400', tip: 'LUT：110,465 / 230,400。双发射+T_MAX 修复共 +3,286' },
  { name: 'BRAM', value: 39.3, label: '39.3% · 122.5 / 312', tip: 'BRAM：122.5 / 312。108 列权重存储（WRAM）等' },
], 100, '%', [0, 25, 50, 75, 100]);

// HW2 功耗三口径
vbar('hw_power', '功耗三口径（mW，整片）', [
  { name: '综合后\n(未布线)', value: 4285, label: '4285', tip: '综合后、无真实活动：4285 mW' },
  { name: '布线后\n(推荐基线)', value: 4465, label: '4465', tip: '布线后 vectorless：4465 mW。推荐的上板前基线' },
  { name: '冒烟 SAIF\n(真实活动)', value: 6058, label: '6058', tip: '冒烟用例 SAIF 注记（真实翻转）：6058 mW。全模型 SAIF 待注记' },
], 6500, '', [0, 2000, 4000, 6000]);

// HW3 布线后功耗分解（u_core 3841 mW）
hbar('hw_power_breakdown', 'GEMM 子系统里电都花在哪（u_core 共 3841 mW）', [
  { name: 'PE 阵列（1728 乘法）', value: 2473, label: '2473 mW · 64%', tip: 'u_arr 2473 mW：1728 个 PE 的加乘与连线翻转，占 u_core 的 64%' },
  { name: 'WRAM（权重存储）', value: 540, label: '540 mW · 14%', tip: 'WRAM 540 mW：108 个权重 bank 的读写翻转' },
  { name: 'requant（量化回收）', value: 270, label: '270 mW · 7%', tip: '27 套 requant 270 mW：INT32 累加结果压回 INT8 的移位乘法' },
  { name: '控制及其他', value: 334, label: '334 mW · 9%', tip: '调度器/守卫等控制逻辑，<85 mW，其余为杂项连线' },
  { name: 'CTX（上下文缓存）', value: 123, label: '123 mW · 3%', tip: 'u_ctx 123 mW：URAM 激活缓存' },
  { name: 'copy 交叉网络', value: 101, label: '101 mW · 3%', tip: 'u_cp 101 mW：列间转置交叉网络。LUT 大头但翻转低' },
], 2500, '', [0, 500, 1000, 1500, 2000, 2500], 680, 330);

// ---------- 软件页 ----------

// SW1 拍数对账阶梯
hbar('sw_est_ladder', '拍数估算 vs 实测（单位：百万拍）', [
  { name: 'v0 估算（原口径）', value: 313.0, label: '313.0 M', tip: 'compiler manifest 的估算：313.0M 拍 ≈ 1.58s @198.5MHz' },
  { name: '修长度读法后', value: 544.5, label: '544.5 M · +74%', tip: '同一模型在修复 dma_len 后的流上重估：544.5M（+74%）' },
  { name: '再修乘法器除法后', value: 854.8, label: '854.8 M · +57%', tip: '修掉 gemm 组数被除小 16 倍的 bug 后：854.8M（+57%）' },
  { name: 'Verilator 实测', value: 1080.8, label: '1080.8 M · +26%', tip: '441 类型全实测：1080.8M 拍（部署口径 pf1）。剩余 21% 差 = LFSR 读停顿/仲裁等模型未含的开销' },
], 1200, '', [0, 300, 600, 900, 1200], 680, 330);

// SW2 描述符构成
hbar('sw_desc_mix', '一次推理 203,378 条描述符都在干什么', [
  { name: 'GEMM（矩阵乘）', value: 69414, label: '69,414 · 34.1%', tip: 'GEMM 计算描述符 69,414 条（34.1%）——真正"算"的部分' },
  { name: 'COPY（转置搬运）', value: 54032, label: '54,032 · 26.6%', tip: 'COPY 运行时转置 54,032 条（26.6%）：swin 窗口重排、im2col 布局调整' },
  { name: 'LOAD_CTX（激活装载）', value: 42246, label: '42,246 · 20.8%', tip: '激活从 DDR 装进片上 CTX：42,246 条（20.8%）。段独立 → 反复装' },
  { name: 'STORE（结果写回）', value: 19410, label: '19,410 · 9.5%', tip: '结果写回 DDR：19,410 条（9.5%），共 660.9 MB/次推理' },
  { name: 'LOAD_W（权重装载）', value: 15494, label: '15,494 · 7.6%', tip: '权重装载 15,494 条（7.6%）：双发射预取优化的对象' },
  { name: '段收尾（DONE 等）', value: 2782, label: '2,782 · 1.4%', tip: '每段一条收尾描述符，2782 段共 2,782 条' },
], 75000, '', [0, 25000, 50000, 75000], 680, 330);

// ---------- 架构页 ----------

// ARCH1 阶段拍数分解
hbar('arch_stage', '拍数按模型阶段分解（部署口径，共 1080.8 M 拍）', [
  { name: 'decoder（去噪解码）', value: 537.4, label: '537.4 M · 49.7%', tip: 'decoder：2117 段、537.4M 拍（49.7%）。10 个去噪步 × 每步约 212 段' },
  { name: 'feature_enhancer', value: 226.5, label: '226.5 M · 21.0%', tip: 'feature_enhancer：366 段、226.5M 拍（21.0%）' },
  { name: 'backbone（2D 主干）', value: 145.6, label: '145.6 M · 13.5%', tip: 'backbone：124 段、145.6M 拍（13.5%）' },
  { name: 'spatial_enhancer', value: 101.9, label: '101.9 M · 9.4%', tip: 'spatial_enhancer：只有 41 段、10.4G MAC，却吃 101.9M 拍（9.4%）——小矩阵+大搬运的典型' },
  { name: 'backbone_3d', value: 36.4, label: '36.4 M · 3.4%', tip: 'backbone_3d：90 段、36.4M 拍' },
  { name: 'text_encoder（BERT）', value: 30.4, label: '30.4 M · 2.8%', tip: 'text_encoder：36 段、30.4M 拍。本轮每次推理都重跑，缓存是后续优化' },
  { name: 'neck / neck_3d / 其他', value: 2.7, label: '2.7 M · 0.2%', tip: 'neck 1.9M + neck_3d 0.8M + text_feat_map 0.08M' },
], 600, '', [0, 150, 300, 450, 600], 680, 330);

// ARCH2 代表段 DMA 忙时占比
hbar('arch_dma_busy', '代表段里 DMA 搬运忙时占总拍数的比例', [
  { name: 'seg_0221（BERT 注意力）', value: 97.7, label: '97.7%', tip: 'seg_0221：104.7 万拍里 DMA 忙 102.3 万拍（97.7%），GEMM 只忙 5.3 万拍——典型"喂不饱"' },
  { name: 'seg_0625（最重段）', value: 67.9, label: '67.9%', tip: 'seg_0625：334.4 万拍里 DMA 忙 226.9 万拍（67.9%），GEMM 忙 107.4 万拍——部分重叠' },
  { name: 'seg_0718（decoder 高频段）', value: 64.6, label: '64.6%', tip: 'seg_0718：全模型跑 80 次的段类型。23.4 万拍里 DMA 忙 15.1 万拍（64.6%）' },
  { name: 'seg_0134（backbone 大段）', value: 30.5, label: '30.5%', tip: 'seg_0134：38.6 万拍里 DMA 忙 11.8 万拍（30.5%），GEMM 忙 27.7 万拍——计算为主' },
  { name: 'seg_0162（长计算段）', value: 13.3, label: '13.3%', tip: 'seg_0162：96.4 万拍里 DMA 只忙 12.8 万拍（13.3%），GEMM 忙 78.9 万拍——计算密度高的好学生' },
], 100, '%', [0, 25, 50, 75, 100], 680, 330);

// ---------- 问题讨论页（2026-08-31 10:2X）----------

// QA1 16× 差距瀑布：0.27s 账本 → 4.32s 实测，三个乘法因子
vbar('qa_gap', '从账本估计到实测：0.27s 怎么变成 4.32s（单位：秒，250MHz 口径）', [
  { name: '账本估计\n(双视角)', value: 0.27, label: '0.27s', tip: '研究阶段账本：73.7 GMAC ÷ 273 GMAC/s = 0.27s。273 = 假设阵列 63% 时间在算' },
  { name: '×1.98\n工作量口径', value: 0.53, label: '0.53s', tip: '换成实测这次跑的工作量 145.8 GMAC（4 相机 vs 账本 2 视角），效率不动：0.53s' },
  { name: '×1.53\npadding', value: 0.82, label: '0.82s', tip: '矩阵边长不是 16 的倍数时补零：有效 145.8G 变成带零 223.8G，×1.53' },
  { name: '×5.27\n搬运与停顿', value: 4.32, label: '4.32s', tip: '阵列真正在算的时间只占 12%：权重反复装、激活反复装、5.4 万条转置指令把阵列饿着。63%÷12% = ×5.27' },
], 4.8, 's', [0, 1, 2, 3, 4]);

// QA2 工作量对表：实测有效乘加 / 账本乘加
hbar('qa_recon', '实测工作量是账本的几倍（有效乘加之比，1.0 = 和账本一致）', [
  { name: 'decoder ↔ 动作头', value: 2.06, label: '2.06×', tip: 'decoder 实测 48.24G vs 账本 ActionHead 23.41G' },
  { name: 'feature_enhancer ↔ 融合', value: 2.00, label: '2.00×', tip: 'feature_enhancer 实测 52.73G vs 账本 Fusion 26.42G' },
  { name: 'backbone+neck ↔ 视觉2D', value: 1.97, label: '1.97×', tip: 'backbone+neck 实测 33.93G vs 账本 Vision2D 17.26G' },
  { name: '3D+空间增强 ↔ 视觉3D', value: 1.97, label: '1.97×', tip: 'spatial_enhancer+backbone_3d+neck_3d 实测 10.24G vs 账本 Vision3D+PSE 5.20G。两边科目划法不完全一样，比值仅供参考' },
  { name: 'text_encoder ↔ 文本', value: 0.50, label: '0.50×', tip: 'text_encoder 实测 0.68G vs 账本 Text 1.37G：实测指令 tokenize 出的 token 数比账本假设的 16 个短，约一半' },
  { name: '总计', value: 1.98, label: '1.98×', tip: '实测有效乘加 145.82G vs 账本 73.66G。四个大科目全是 ≈2×，指向同一个原因：相机数 4 vs 视角数 2' },
], 2.4, '×', [0, 0.5, 1, 1.5, 2], 680, 330);

// QA3 模块 LUT 占用（布线后 DCP 层级账，u_core 共 103,846）
hbar('qa_modules', '每个模块占多少 LUT（布线后实测，共 103,958 已用）', [
  { name: 'PE 阵列 u_arr', value: 4.2267, label: '42,267 · 40.7%', tip: '16×108 脉动阵列：全部 1728 个 DSP（乘法器）都在这里，所有矩阵乘法由它完成。87,544 个寄存器存部分积' },
  { name: 'copy 重排 u_cp', value: 1.841, label: '18,410 · 17.7%', tip: '转置/重排网络：swin 窗口重排、im2col 都靠它在列间搬数据。一次推理要执行 54,032 条 COPY 指令' },
  { name: 'requant ×27', value: 1.694, label: '16,940 · 16.3%', tip: '量化回收：把 32 位累加结果压回 INT8。27 套时分复用服务 108 列（每套管 4 列）' },
  { name: 'GEMM 胶水', value: 1.2117, label: '12,117 · 11.7%', tip: 'u_gemm 内部行缓冲 tile_buf + 输出排水（drain）控制' },
  { name: 'softmax u_sm', value: 1.0164, label: '10,164 · 9.8%', tip: '16 条并行查表车道算注意力 softmax' },
  { name: 'DMA u_dma', value: 0.1469, label: '1,469 · 1.4%', tip: '搬数引擎：在 DDR 和片上之间倒数据。逻辑不大，但它是性能瓶颈所在' },
  { name: 'core 胶水', value: 0.1224, label: '1,224 · 1.2%', tip: 'u_core 一级的 FIFO/行缓冲，14 块 BRAM' },
  { name: '调度器 u_sched', value: 0.0999, label: '999 · 1.0%', tip: '按段读描述符、发指令。双发射改造就在这里' },
  { name: 'CTX 缓存 u_ctx', value: 0.026, label: '260 · 0.3%', tip: '上下文缓存，本体放在 64 块 URAM 里，逻辑只有 260 LUT' },
], 4.5, '万', [0, 1, 2, 3, 4], 680, 330);

// QA4 模型各阶段计算时间占比（与 QA3 配对：资源在哪 × 时间在哪）
hbar('qa_time', '模型各阶段的计算时间占比（实测 1080.8M 拍，部署口径）', [
  { name: 'decoder（去噪解码）', value: 49.7, label: '49.7%', tip: '537.4M 拍。10 个去噪步，全在动作头 GEMM 上' },
  { name: 'feature_enhancer', value: 21.0, label: '21.0%', tip: '226.5M 拍。特征增强/融合层' },
  { name: 'backbone（2D 主干）', value: 13.5, label: '13.5%', tip: '145.6M 拍。每个相机各过一遍主干' },
  { name: 'spatial_enhancer', value: 9.4, label: '9.4%', tip: '101.9M 拍。只有 10.4G 带零乘加却吃 9.4% 时间——小矩阵+大搬运的典型受害者' },
  { name: 'backbone_3d', value: 3.4, label: '3.4%', tip: '36.4M 拍' },
  { name: 'text_encoder（BERT）', value: 2.8, label: '2.8%', tip: '30.4M 拍。本轮每次推理都重跑；缓存后这 2.8% 可以直接省掉' },
  { name: 'neck 等', value: 0.2, label: '0.2%', tip: 'neck 1.9M + neck_3d 0.8M + text_feat_map 0.08M' },
], 55, '%', [0, 10, 20, 30, 40, 50], 680, 330);

// QA5 总拍构成归因（on-chip 调研边界账，2026-08-31 到货）
hbar('qa_cycle_mix', '10.81 亿拍都花在哪（按拍数归因，不再按指令条数）', [
  { name: 'GEMM 引擎', value: 599.1, label: '599.1 M · 55.4%', tip: '矩阵乘本体。忙时每拍平均 373 个乘加 = 峰值的 22%：小矩阵和排水填充摊薄了引擎——这是 padding/形状匹配问题，不是搬运' },
  { name: 'STORE（写回 DDR）', value: 206.6, label: '206.6 M · 19.1%', tip: '算完的结果写回 DDR，共 660.9 MB，折算只有 3.2 字节/拍——最慢的搬运，第一大单项。治它要靠编译器加大段+流式写回' },
  { name: 'LOAD_CTX（激活重装）', value: 164.9, label: '164.9 M · 15.3%', tip: '激活从 DDR 重装片上，共 1306.5 MB。on-chip 算子融合（第六节）直接削这笔' },
  { name: 'LOAD_W（权重装载）', value: 75.4, label: '75.4 M · 7.0%', tip: '权重装载。预取已吃掉 33.7M，还暴露约 42M 拍——预取窗口扩到 COPY/新引擎运行期可再吃回一部分' },
  { name: '调度+LFSR 停顿', value: 19.6, label: '19.6 M · 1.8%', tip: '段间隙、随机数读停顿、总线仲裁' },
  { name: 'COPY（运行时转置）', value: 15.2, label: '15.2 M · 1.4%', tip: '5.4 万条看着吓人，拍数只占 1.4%。真实罪状是打断权重预取（预取只在 GEMM/SM 运行期发射），是条数问题不是拍数问题' },
], 650, '', [0, 150, 300, 450, 600], 680, 330);

// QA6 on-chip 算子融合收益（净省拍，均已扣新引擎自身拍数）
hbar('qa_onchip_gain', '算子搬进片上各能净省多少拍（已扣引擎自身开销）', [
  { name: 'norm 归一化族', value: 57.9, label: '57.9 M · 291 ms', tip: 'RMS/LN/Ada/GN 共 423 步。占全部边界代价的 49%，最大单项。约 9,000 LUT，两级累加器保整数精度' },
  { name: 'softmax 独立调用', value: 15.5, label: '15.5 M · 78 ms', tip: 'SM16 已在片上，只缺独立调用的描述符。184/220 处零新数据通路，约 1,800 LUT' },
  { name: 'actv（SiLU/GELU）', value: 14.1, label: '14.1 M · 71 ms', tip: 'INT8 输入只有 256 种取值，256×8 直查表与 host 可同舍入位精确，新增误差≈0。约 1,500 LUT' },
  { name: 'swin 窗口散射', value: 13.9, label: '13.9 M · 70 ms', tip: '硬件散射引擎约 2,500 LUT。不做它就覆盖不了 backbone 的 26.5M 边界代价' },
  { name: 'rotary 旋转', value: 6.6, label: '6.6 M · 33 ms', tip: 'cos/sin 表进 BRAM + 成对双乘，边际约 1,000 LUT' },
], 65, '', [0, 15, 30, 45, 60], 680, 330);

// QA7 INT4 敏感度×字节占比（哪些模块降、哪些不降）
hbar('qa_w4_modules', '六个模块各占多少权重字节，哪些降到 INT4（总 159.5 MB）', [
  { name: 'text_encoder（BERT）', value: 84.9, label: '84.9 · 降 INT4', tip: '占 53% 字节的模块恰恰最不敏感：per-tensor INT4 偏差 0.0187 rad，还低于 W8 基线 0.0259（同一噪声带，不是更准，是不可区分）。85M 参数的大块权重分布平整，小模型 W4 必须分组的普遍规律里它是例外' },
  { name: 'vision_2d（Swin RGB）', value: 27.4, label: '27.4 · 降 INT4 g128', tip: 'per-tensor 会掉到 0.049（真实样本 0.075+），必须组内 scale（g=128）。g=64 与 g=128 实测等效，组宽可按硬件方便选。注意：QoQ 式装载时解包回 INT8 的零 RTL 路线在这里不行，组 scale 折进 per-tensor 网格后真实样本最差 0.079' },
  { name: 'fusion', value: 23.3, label: '23.3 · 保持 W8', tip: 'g=128 也要 0.0382，离红线太近，不动' },
  { name: 'action_head', value: 20.7, label: '20.7 · 保持 W8', tip: '单独 g=128 是 0.0354 尚可，但激进档（42% 节省）合成 0.037–0.047、真实出现 0.056–0.065 坏例，超 0.04 红线。想拉它进来优先 GPTQ/HQQ，且标定集必须覆盖全部去噪 timestep' },
  { name: 'neck_convs', value: 2.1, label: '2.1 · 降 INT4', tip: 'per-tensor 0.0226，免费' },
  { name: 'vision_3d', value: 0.9, label: '0.9 · 保持 W8', tip: '只占 0.6% 字节，真实样本 s2 会崩到 0.067——不值得冒险' },
], 110, '', [0, 25, 50, 75, 100], 680, 330);

console.log('charts written to', OUT);
