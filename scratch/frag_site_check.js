/* Cross-check: the site's fragility-adjuster math vs the python study.
   Expect (63d, MA10, step thr50 floor0.5, non-OVS trades with scores):
   totR-adjusted ~ +601, avgR-per-unit ~ +0.592 (scratch/frag_sizing_verify.py). */
const fs = require("fs");
const frag = JSON.parse(fs.readFileSync("dist/data/fragility.json", "utf8"));
const tj = JSON.parse(fs.readFileSync("dist/data/trades.json", "utf8"));

const rows = [];
for (let i = 0; i < tj.n; i++) {
  const r = {};
  for (const k of Object.keys(tj.columns)) r[k] = tj.columns[k][i];
  rows.push(r);
}

// fragSeries (dial 63d, ma 10) — same algorithm as portfolio.js
const raw = frag.dials["63d"], ma = 10;
const vals = new Array(raw.length).fill(null);
{
  const win = [];
  let sum = 0;
  for (let i = 0; i < raw.length; i++) {
    if (raw[i] != null) {
      win.push(raw[i]); sum += raw[i];
      if (win.length > ma) sum -= win.shift();
    }
    if (win.length) vals[i] = sum / win.length;
  }
}
const dates = frag.dates;
function upperBound(arr, x) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const m = (lo + hi) >> 1; arr[m] <= x ? lo = m + 1 : hi = m; }
  return lo;
}
function scoreFor(d) {
  const i = upperBound(dates, d) - 1;
  if (i < 0) return null;
  if (Date.parse(d) - Date.parse(dates[i]) > 7 * 86400e3) return null;
  return vals[i];
}
const multOf = s => (s == null ? 1 : (s >= 50 ? 0.5 : 1.0));   // step thr50 floor0.5

let n = 0, totRadj = 0, multSum = 0, throttled = 0;
for (const t of rows) {
  if (t.Strategy === "Overbot Vol Spike") continue;
  if (t.R == null || !t.Signal_Date) continue;
  const s = scoreFor(t.Signal_Date);
  if (s == null) continue;
  const m = multOf(s);
  n++; totRadj += t.R * m; multSum += m;
  if (m < 1) throttled++;
}
console.log(`covered non-OVS trades: ${n}`);
console.log(`totR adjusted: ${totRadj.toFixed(1)}  (python: ~+601)`);
console.log(`avgR per unit risk: ${(totRadj / multSum).toFixed(4)}  (python: ~+0.592)`);
console.log(`throttled: ${(100 * throttled / n).toFixed(1)}%  (python: 242/1153 = 21.0%)`);
console.log(`today: score ${vals[vals.length - 1].toFixed(2)} -> mult ${multOf(vals[vals.length - 1])}`);
