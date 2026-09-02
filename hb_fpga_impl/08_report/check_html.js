// 2026-08-31 HTML 自检：标签配平、data-tip 完整、时间戳、未解析占位符
// 用法：node check_html.js <file.html> [...]
const fs = require('fs');
let bad = 0;
for (const f of process.argv.slice(2)) {
  const h = fs.readFileSync(f, 'utf8');
  const problems = [];
  for (const t of ['div', 'figure', 'table', 'tr', 'td', 'th', 'p', 'svg', 'h1', 'h2', 'h4', 'header', 'footer', 'script', 'style', 'body', 'html']) {
    const o = (h.match(new RegExp('<' + t + '[\\s>]', 'g')) || []).length;
    const c = (h.match(new RegExp('</' + t + '>', 'g')) || []).length;
    if (o !== c) problems.push(`tag ${t}: ${o} open vs ${c} close`);
  }
  const tipOpen = (h.match(/data-tip="/g) || []).length;
  if ((h.match(/data-tip="[^"]*"/g) || []).length !== tipOpen) problems.push('data-tip attribute broken');
  if (/data-tip="[^"<>]*[<>]/.test(h)) problems.push('raw < or > inside data-tip');
  if (h.includes('{{')) problems.push('unresolved {{placeholder}}');
  if (!/20\d\d-\d\d-\d\d \d\d:\d\d:\d\d/.test(h)) problems.push('missing second-precision timestamp');
  console.log(problems.length ? `FAIL ${f}\n  ` + problems.join('\n  ') : `PASS ${f} (svg ${(h.match(/<svg/g) || []).length}, data-tip ${tipOpen})`);
  if (problems.length) bad = 1;
}
process.exit(bad);
