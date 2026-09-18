/* Headless smoke test for the OVS hold-extension toggle in portfolio.js —
   shims a minimal DOM + fetch over the real dist/data payloads, runs init(),
   then flips S.showOvsExt and asserts the row swap, basis fallback, and the
   ext-lab section render. Pattern: scratch/test_orders_render.js. */
const fs = require("fs");
const vm = require("vm");

function mkEl() {
  const el = {
    innerHTML: "", textContent: "", value: "", checked: false, style: {},
    dataset: {}, classList: { toggle() {}, add() {}, remove() {} },
    children: [], type: "", title: "",
    addEventListener() {}, appendChild(c) { this.children.push(c); return c; },
    querySelector() { return mkEl(); }, querySelectorAll() { return []; },
    setAttribute() {}, contains() { return false; },
  };
  return el;
}
const document = {
  _els: {},
  getElementById(id) { return (this._els[id] ||= mkEl()); },
  addEventListener(ev, fn) { if (ev === "DOMContentLoaded") this._ready = fn; },
  createElement() { return mkEl(); },
  createTextNode(t) { return { text: t }; },
  querySelectorAll() { return []; },
};
const fetch = async (path) => {
  const p = "dist/" + String(path);
  if (!fs.existsSync(p)) return { ok: false, status: 404, json: async () => ({}) };
  return { ok: true, status: 200, json: async () => JSON.parse(fs.readFileSync(p, "utf8")) };
};
const Plotly = { react() {}, newPlot() {}, Plots: { resize() {} } };
const sandbox = { document, fetch, console, setTimeout, clearTimeout, Plotly,
                  URL: { createObjectURL() { return ""; }, revokeObjectURL() {} },
                  Blob: function () {}, requestAnimationFrame(fn) { fn(); } };
sandbox.window = sandbox;
vm.createContext(sandbox);
const src = fs.readFileSync("site/assets/common.js", "utf8") + "\n" +
            fs.readFileSync("site/assets/portfolio.js", "utf8") +
            "\n;globalThis.__test = { S, filteredTrades, curveExact, renderExtLab, dailySeries, dailyMetrics };";
vm.runInContext(src, sandbox);

(async () => {
  let pass = true;
  const check = (ok, label) => { console.log((ok ? "OK   " : "MISS ") + label); if (!ok) pass = false; };
  try {
    await document._ready();   // init()
  } catch (e) {
    console.log("init threw:", e.message);
  }
  const T = sandbox.__test; const S = T.S;
  check(S && S.trades && S.trades.length > 3000, `trades loaded (${S && S.trades && S.trades.length})`);
  check(S.extById.size === 351, `extById size 351 (${S.extById.size})`);

  const before = T.filteredTrades();
  const rSum = rows => rows.reduce((a, t) => a + (t.R || 0), 0);
  const r0 = rSum(before);
  check(T.curveExact() === true, "curveExact true before toggle");

  S.showOvsExt = true;
  const after = T.filteredTrades();
  check(after.length === before.length, `row count unchanged (${before.length} -> ${after.length})`);
  const nSwapped = after.filter(t => t.OvsExt).length;
  check(nSwapped === 351, `351 rows swapped (${nSwapped})`);
  const dR = rSum(after) - r0;
  check(Math.abs(dR - 81.9) < 1.5, `delta R ~ +81.9 (${dR.toFixed(1)})`);
  check(T.curveExact() === false, "curveExact false while toggle on");

  // gate + ext compose without id collisions (gate rows have no trade_id)
  S.showBlocked = true;
  const both = T.filteredTrades();
  check(both.length === before.length + S.gateBlockedRows.length,
        `gate rows still additive (${both.length})`);
  check(both.filter(t => t.GateBlocked && t.OvsExt).length === 0, "no gate/ext cross-contamination");
  S.showBlocked = false; S.showOvsExt = false;

  // per-trade MTM vector path: must reproduce the aggregated exact path to
  // rounding, and toggles must show only their true effect (no basis jump)
  check(S.mtmDates && S.mtmMain.size >= 3000, `trade_mtm loaded (${S.mtmMain.size} vectors)`);
  const tot = a => a.reduce((x, y) => x + y, 0);
  const dsExact = T.dailySeries(T.filteredTrades());
  const shExact = T.dailyMetrics(dsExact).sharpe;
  // force the vector path with nothing swapped (extById emptied)
  const saved = [...S.extById];
  S.extById.clear(); S.showOvsExt = true;
  const dsVec = T.dailySeries(T.filteredTrades());
  check(dsVec.exact === true, "vector path reports exact basis");
  check(Math.abs(tot(dsVec.pnl) - tot(dsExact.pnl)) < 100,
        `vector total == aggregated total (${tot(dsVec.pnl).toFixed(0)} vs ${tot(dsExact.pnl).toFixed(0)})`);
  const shVecBase = T.dailyMetrics(dsVec).sharpe;
  check(Math.abs(shVecBase - shExact) < 0.06,
        `vector Sharpe == aggregated Sharpe (${shVecBase.toFixed(2)} vs ${shExact.toFixed(2)})`);
  for (const [k, v] of saved) S.extById.set(k, v);
  // extension on: Sharpe moves by its true effect only (~flat), not a basis jump
  const shVecExt = T.dailyMetrics(T.dailySeries(T.filteredTrades())).sharpe;
  check(Math.abs(shVecExt - shVecBase) < 0.12,
        `toggle Sharpe effect ~flat on MTM basis (${shVecBase.toFixed(2)} -> ${shVecExt.toFixed(2)})`);
  S.showOvsExt = false;
  // direction filter now rides the vector path too, totals preserved
  S.f.dir = "Short";
  const shorts = T.filteredTrades();
  const dsShort = T.dailySeries(shorts);
  const realized = shorts.reduce((a, t) => a + (t.PnL_flat || 0), 0);
  check(dsShort.exact === true, "direction filter exact basis");
  check(Math.abs(tot(dsShort.pnl) - realized) < 100,
        `short-only MTM total == realized total (${tot(dsShort.pnl).toFixed(0)} vs ${realized.toFixed(0)})`);
  S.f.dir = "All";

  T.renderExtLab();
  const kpis = document._els["xlKpis"] ? document._els["xlKpis"].innerHTML : "";
  check(kpis.includes("Rebooked trades"), "ext-lab KPIs rendered");
  check(kpis.includes("351"), "KPI shows 351 rebooked");
  const cap = document._els["xlCaption"] ? document._els["xlCaption"].textContent : "";
  check(cap.includes("T+5"), "caption mentions the rule");

  console.log(pass ? "\nALL PASS" : "\nFAILURES");
  process.exit(pass ? 0 : 1);
})();
