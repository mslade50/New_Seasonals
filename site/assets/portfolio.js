/* portfolio.js — client-side filterable analytics over the trade ledger.

   Sizing basis: FLAT $750k (PnL_flat). Per-trade dollars are additive, so any
   filtered subset yields an exact equity curve / Sharpe / DD on this basis.

   Daily-curve resolution:
   - strategy / tier / date filters  -> exact daily MTM (sum of per-strategy
     daily series computed server-side)
   - direction / ticker filters      -> fallback: realized PnL booked on exit
     date (step curve; vol metrics approximate). A notice badge appears.
*/
"use strict";

const S = {
  meta: null, trades: [], sd: null, positions: null, exposure: null, corr: null,
  dateIdx: [],            // sd.dates as strings
  dateToI: new Map(),
  f: { strategies: null, tier: "All", dir: "All", preset: "All", from: null, to: null, tickerQ: "" },
  tradeLogTable: null,
};
const START_EQ = 750000;

document.addEventListener("DOMContentLoaded", init);

async function init() {
  renderNav("index.html");
  try {
    const [meta, trades] = await Promise.all([
      fetchJSON("data/meta.json"), fetchJSON("data/trades.json")]);
    S.meta = meta;
    S.trades = rowsFromColumnar(trades);
    const [sd, pos, exp, corr] = await Promise.all([
      meta.payloads.strategy_daily ? fetchJSONOrNull("data/strategy_daily.json") : null,
      meta.payloads.positions ? fetchJSONOrNull("data/positions.json") : null,
      meta.payloads.exposure ? fetchJSONOrNull("data/exposure.json") : null,
      meta.payloads.correlation ? fetchJSONOrNull("data/correlation.json") : null,
    ]);
    S.sd = sd; S.positions = pos; S.exposure = exp; S.corr = corr;
    if (sd) {
      S.dateIdx = sd.dates;
      sd.dates.forEach((d, i) => S.dateToI.set(d, i));
    }
    setAsof(`ledger thru ${meta.ledger_last_signal} · built ${meta.built_at}`);
    document.getElementById("subtitle").textContent =
      `${meta.n_trades.toLocaleString()} trades · ${meta.n_tickers} tickers · ` +
      `${meta.date_min} to ${meta.date_max} · flat $750k risk basis (filter-exact)`;
    buildFilterBar();
    renderStatic();
    apply();
  } catch (e) {
    document.getElementById("kpis").innerHTML = `<div class="err">Failed to load data: ${e.message}</div>`;
    console.error(e);
  }
}

/* ================= filters ================= */
function allStrategyNames() {
  return [...new Set(S.meta.strategies.map(s => s.Strategy))].sort();
}

function buildFilterBar() {
  const el = document.getElementById("filters");
  el.innerHTML = "";

  // strategy multiselect
  const msel = document.createElement("div");
  msel.className = "msel";
  const mbtn = document.createElement("button");
  mbtn.className = "btn"; mbtn.id = "stratBtn";
  msel.appendChild(mbtn);
  const panel = document.createElement("div");
  panel.className = "panel"; panel.style.display = "none";
  msel.appendChild(panel);
  el.appendChild(msel);

  const names = allStrategyNames();
  S.f.strategies = new Set(names);
  const acts = document.createElement("div");
  acts.className = "acts";
  acts.innerHTML = `<button class="btn ghost" data-a="all">All</button>
                    <button class="btn ghost" data-a="none">None</button>
                    <span class="cnt"></span>`;
  panel.appendChild(acts);
  const boxes = [];
  for (const n of names) {
    const row = document.createElement("label");
    row.className = "row";
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.checked = true; cb.value = n;
    cb.addEventListener("change", () => {
      cb.checked ? S.f.strategies.add(n) : S.f.strategies.delete(n);
      syncStratBtn(); apply();
    });
    boxes.push(cb);
    row.appendChild(cb);
    row.appendChild(document.createTextNode(" " + n));
    panel.appendChild(row);
  }
  acts.querySelector('[data-a="all"]').addEventListener("click", () => {
    S.f.strategies = new Set(names); boxes.forEach(b => b.checked = true); syncStratBtn(); apply();
  });
  acts.querySelector('[data-a="none"]').addEventListener("click", () => {
    S.f.strategies.clear(); boxes.forEach(b => b.checked = false); syncStratBtn(); apply();
  });
  mbtn.addEventListener("click", () => {
    panel.style.display = panel.style.display === "none" ? "block" : "none";
  });
  document.addEventListener("click", ev => {
    if (!msel.contains(ev.target)) panel.style.display = "none";
  });
  function syncStratBtn() {
    mbtn.textContent = `Strategies (${S.f.strategies.size}/${names.length})`;
  }
  syncStratBtn();

  // tier + direction segments
  el.appendChild(makeSeg("Tier", ["All", "Liquid", "Overflow"], v => { S.f.tier = v; apply(); }));
  el.appendChild(makeSeg("Direction", ["All", "Long", "Short"], v => { S.f.dir = v; apply(); }));

  // date presets
  const presets = ["All", "10Y", "5Y", "3Y", "1Y", "YTD"];
  const presetSeg = makeSeg("Range", presets, v => {
    S.f.preset = v;
    const max = S.meta.date_max;
    const maxD = new Date(max + "T00:00:00Z");
    let from = null;
    if (v === "YTD") from = max.slice(0, 4) + "-01-01";
    else if (v !== "All") {
      const yrs = parseInt(v);
      const d = new Date(maxD); d.setUTCFullYear(d.getUTCFullYear() - yrs);
      from = d.toISOString().slice(0, 10);
    }
    S.f.from = from; S.f.to = null;
    fromInp.value = from || ""; toInp.value = "";
    apply();
  });
  el.appendChild(presetSeg);

  const fromInp = document.createElement("input");
  fromInp.type = "date"; fromInp.title = "From";
  const toInp = document.createElement("input");
  toInp.type = "date"; toInp.title = "To";
  fromInp.addEventListener("change", () => { S.f.from = fromInp.value || null; S.f.preset = "Custom"; markSeg(presetSeg, null); apply(); });
  toInp.addEventListener("change", () => { S.f.to = toInp.value || null; S.f.preset = "Custom"; markSeg(presetSeg, null); apply(); });
  el.appendChild(fromInp); el.appendChild(toInp);

  // ticker search
  const tInp = document.createElement("input");
  tInp.type = "text"; tInp.placeholder = "Tickers (e.g. NVDA, XLE)";
  tInp.style.width = "170px";
  let tmr = null;
  tInp.addEventListener("input", () => {
    clearTimeout(tmr);
    tmr = setTimeout(() => { S.f.tickerQ = tInp.value.trim(); apply(); }, 250);
  });
  el.appendChild(tInp);

  // reset
  const rb = document.createElement("button");
  rb.className = "btn ghost"; rb.textContent = "Reset";
  rb.addEventListener("click", () => {
    S.f = { strategies: new Set(names), tier: "All", dir: "All", preset: "All", from: null, to: null, tickerQ: "" };
    boxes.forEach(b => b.checked = true); syncStratBtn();
    fromInp.value = ""; toInp.value = ""; tInp.value = "";
    el.querySelectorAll(".seg").forEach(seg => {
      seg.querySelectorAll("button").forEach((b, i) => b.classList.toggle("on", i === 0));
    });
    apply();
  });
  el.appendChild(rb);
}

function makeSeg(label, values, onpick) {
  const box = document.createElement("span");
  box.appendChild(Object.assign(document.createElement("label"), { textContent: label }));
  const seg = document.createElement("span");
  seg.className = "seg";
  values.forEach((v, i) => {
    const b = document.createElement("button");
    b.textContent = v;
    if (i === 0) b.classList.add("on");
    b.addEventListener("click", () => {
      seg.querySelectorAll("button").forEach(x => x.classList.remove("on"));
      b.classList.add("on");
      onpick(v);
    });
    seg.appendChild(b);
  });
  box.appendChild(seg);
  box.seg = seg;
  return box;
}
function markSeg(segBox, value) {
  segBox.seg.querySelectorAll("button").forEach(b =>
    b.classList.toggle("on", b.textContent === value));
}

/* ================= filtering ================= */
function tickerTokens() {
  if (!S.f.tickerQ) return null;
  return S.f.tickerQ.toUpperCase().split(",").map(s => s.trim()).filter(Boolean);
}

function filteredTrades() {
  const toks = tickerTokens();
  const { strategies, tier, dir, from, to } = S.f;
  return S.trades.filter(t => {
    if (!strategies.has(t.Strategy)) return false;
    if (tier !== "All" && t.Tier !== tier) return false;
    if (dir !== "All" && t.Direction !== dir) return false;
    const d = t.Entry_Date || t.Signal_Date;
    if (from && d < from) return false;
    if (to && d > to) return false;
    if (toks && !toks.some(tok => (t.Ticker || "").toUpperCase().startsWith(tok))) return false;
    return true;
  });
}

function curveExact() {
  return S.sd && S.f.dir === "All" && !tickerTokens();
}

/* daily pnl array + matching date subarray for current filters */
function dailySeries(trades) {
  if (curveExact()) {
    const keys = Object.keys(S.sd.series).filter(k => {
      const [strat, tier] = k.split("||");
      if (!S.f.strategies.has(strat)) return false;
      if (S.f.tier !== "All" && tier !== S.f.tier) return false;
      return true;
    });
    let i0 = 0, i1 = S.dateIdx.length - 1;
    if (S.f.from) i0 = lowerBound(S.dateIdx, S.f.from);
    if (S.f.to) i1 = upperBound(S.dateIdx, S.f.to) - 1;
    if (i1 < i0) return { dates: [], pnl: [], exact: true };
    const n = i1 - i0 + 1;
    const pnl = new Float64Array(n);
    for (const k of keys) {
      const arr = S.sd.series[k];
      for (let i = 0; i < n; i++) pnl[i] += arr[i0 + i];
    }
    return { dates: S.dateIdx.slice(i0, i1 + 1), pnl: Array.from(pnl), exact: true };
  }
  // fallback: realized PnL on exit dates
  const map = new Map();
  for (const t of trades) {
    const d = t.Exit_Date;
    if (!d || t.PnL_flat == null) continue;
    map.set(d, (map.get(d) || 0) + t.PnL_flat);
  }
  const dates = [...map.keys()].sort();
  return { dates, pnl: dates.map(d => map.get(d)), exact: false };
}

function lowerBound(arr, x) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const m = (lo + hi) >> 1; arr[m] < x ? lo = m + 1 : hi = m; }
  return lo;
}
function upperBound(arr, x) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const m = (lo + hi) >> 1; arr[m] <= x ? lo = m + 1 : hi = m; }
  return lo;
}

/* ================= metrics ================= */
function tradeMetrics(tr) {
  const n = tr.length;
  const rs = tr.map(t => t.R).filter(v => v != null);
  const pnls = tr.map(t => t.PnL_flat).filter(v => v != null);
  const wins = pnls.filter(v => v > 0), losses = pnls.filter(v => v < 0);
  const sum = a => a.reduce((x, y) => x + y, 0);
  const mean = a => a.length ? sum(a) / a.length : null;
  const std = a => {
    if (a.length < 2) return null;
    const m = mean(a);
    return Math.sqrt(sum(a.map(v => (v - m) ** 2)) / (a.length - 1));
  };
  const totR = sum(rs), avgR = mean(rs), stdR = std(rs);
  const sortedR = rs.slice().sort((a, b) => a - b);
  const q = p => sortedR.length ? sortedR[Math.min(sortedR.length - 1, Math.floor(p * sortedR.length))] : null;
  let maxConsecL = 0, run = 0;
  const chron = tr.slice().sort((a, b) => (a.Exit_Date || "").localeCompare(b.Exit_Date || ""));
  for (const t of chron) {
    if (t.PnL_flat != null && t.PnL_flat < 0) { run++; maxConsecL = Math.max(maxConsecL, run); }
    else if (t.PnL_flat != null) run = 0;
  }
  const holds = tr.map(t => t.Hold_Days).filter(v => v != null);
  return {
    n, winRate: pnls.length ? wins.length / pnls.length : null,
    totR, avgR, stdR,
    sqn: (avgR != null && stdR) ? Math.sqrt(Math.min(n, 100)) * avgR / stdR : null,
    pf: losses.length ? sum(wins) / Math.abs(sum(losses)) : null,
    expectancy: mean(pnls),
    payoff: (wins.length && losses.length) ? mean(wins) / Math.abs(mean(losses)) : null,
    totPnl: sum(pnls),
    tail: (q(0.95) != null && q(0.05) != null && q(0.05) !== 0) ? Math.abs(q(0.95) / q(0.05)) : null,
    maxConsecL,
    avgHold: mean(holds),
  };
}

function dailyMetrics(ds) {
  const { pnl } = ds;
  const n = pnl.length;
  if (!n) return {};
  const rets = pnl.map(v => v / START_EQ);
  const sum = a => a.reduce((x, y) => x + y, 0);
  const m = sum(rets) / n;
  const sd = n > 1 ? Math.sqrt(sum(rets.map(v => (v - m) ** 2)) / (n - 1)) : 0;
  const downside = rets.filter(v => v < 0);
  const dsd = downside.length > 1 ?
    Math.sqrt(sum(downside.map(v => v * v)) / downside.length) : 0;
  // equity + max drawdown vs running peak
  let eq = START_EQ, peak = START_EQ, maxDD = 0;
  const equity = new Array(n);
  for (let i = 0; i < n; i++) {
    eq += pnl[i];
    equity[i] = eq;
    if (eq > peak) peak = eq;
    const dd = eq / peak - 1;
    if (dd < maxDD) maxDD = dd;
  }
  const annRet = m * 252, annVol = sd * Math.sqrt(252);
  return {
    equity, annRet, annVol,
    sharpe: sd ? (m / sd) * Math.sqrt(252) : null,
    sortino: dsd ? (m / dsd) * Math.sqrt(252) : null,
    maxDD,
    mar: maxDD ? annRet / Math.abs(maxDD) : null,
    bestDay: Math.max(...pnl), worstDay: Math.min(...pnl),
  };
}

/* ================= render ================= */
function apply() {
  const tr = filteredTrades();
  const ds = dailySeries(tr);
  const tm = tradeMetrics(tr);
  const dm = dailyMetrics(ds);
  renderKPIs(tm, dm, ds.exact);
  renderEquity(ds, dm);
  renderCumR(tr);
  renderMonthly(ds);
  renderRolling(ds);
  renderHist(tr);
  renderSeasonality(tr);
  renderHoldBuckets(tr);
  renderStratTable(tr);
  renderYearTable(tr, ds);
  renderTradeLog(tr);
}

function kpiCard(label, value, cls, sub) {
  return `<div class="kpi"><div class="l">${label}</div>
    <div class="v ${cls || ""}">${value}</div>${sub ? `<div class="s">${sub}</div>` : ""}</div>`;
}

function renderKPIs(tm, dm, exact) {
  const el = document.getElementById("kpis");
  const basisNote = document.getElementById("basisNote");
  basisNote.style.display = exact ? "none" : "inline-block";
  el.innerHTML = [
    kpiCard("Trades", tm.n.toLocaleString()),
    kpiCard("Win Rate", tm.winRate == null ? "-" : fmt.pct(tm.winRate, 1)),
    kpiCard("Total R", fmt.num(tm.totR, 0), clsSign(tm.totR)),
    kpiCard("Avg R", tm.avgR == null ? "-" : fmt.num(tm.avgR, 3), clsSign(tm.avgR)),
    kpiCard("Profit Factor", tm.pf == null ? "-" : fmt.num(tm.pf, 2)),
    kpiCard("Expectancy", tm.expectancy == null ? "-" : fmt.money(tm.expectancy), clsSign(tm.expectancy), "per trade"),
    kpiCard("SQN", tm.sqn == null ? "-" : fmt.num(tm.sqn, 2)),
    kpiCard("Payoff", tm.payoff == null ? "-" : fmt.num(tm.payoff, 2), null, "avg win / avg loss"),
    kpiCard("Total PnL", fmt.money(tm.totPnl), clsSign(tm.totPnl)),
    kpiCard("Ann Return", dm.annRet == null ? "-" : fmt.pct(dm.annRet, 1), clsSign(dm.annRet), "of $750k"),
    kpiCard("Ann Vol", dm.annVol == null ? "-" : fmt.pct(dm.annVol, 1)),
    kpiCard("Sharpe", dm.sharpe == null ? "-" : fmt.num(dm.sharpe, 2), clsSign(dm.sharpe)),
    kpiCard("Sortino", dm.sortino == null ? "-" : fmt.num(dm.sortino, 2)),
    kpiCard("Max Drawdown", dm.maxDD == null ? "-" : fmt.pct(dm.maxDD, 1), "neg"),
    kpiCard("MAR", dm.mar == null ? "-" : fmt.num(dm.mar, 2)),
    kpiCard("Tail Ratio", tm.tail == null ? "-" : fmt.num(tm.tail, 2), null, "|p95 / p5| of R"),
    kpiCard("Max Consec Losses", tm.maxConsecL),
    kpiCard("Avg Hold", tm.avgHold == null ? "-" : fmt.num(tm.avgHold, 1), null, "trading days"),
  ].join("");
}

function renderEquity(ds, dm) {
  const eqEl = document.getElementById("eqChart");
  const ddEl = document.getElementById("ddChart");
  if (!ds.dates.length) { Plotly.purge(eqEl); Plotly.purge(ddEl); return; }
  const equity = dm.equity;
  Plotly.react(eqEl, [{
    x: ds.dates, y: equity, mode: "lines", name: "Equity",
    line: { color: "#00d18f", width: 1.8 },
  }], plotLayout({
    height: 330, yaxis: { tickformat: "$,.0f", title: { text: "Equity (flat $750k)", font: { size: 11 } } },
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: START_EQ, y1: START_EQ,
               line: { color: "#444c5c", width: 1, dash: "dot" } }],
  }), PLOT_CFG);
  // underwater
  let peak = -Infinity;
  const dd = equity.map(v => { peak = Math.max(peak, v); return (v / peak - 1) * 100; });
  Plotly.react(ddEl, [{
    x: ds.dates, y: dd, mode: "lines", name: "Drawdown",
    fill: "tozeroy", line: { color: "#ff5d5d", width: 1 }, fillcolor: "rgba(255,93,93,.18)",
  }], plotLayout({ height: 150, margin: { t: 8 }, yaxis: { ticksuffix: "%" } }), PLOT_CFG);
}

function renderCumR(tr) {
  const el = document.getElementById("cumRChart");
  const byStrat = new Map();
  for (const t of tr) {
    if (t.R == null || !t.Exit_Date) continue;
    if (!byStrat.has(t.Strategy)) byStrat.set(t.Strategy, []);
    byStrat.get(t.Strategy).push(t);
  }
  let entries = [...byStrat.entries()].map(([s, arr]) => {
    return [s, arr, Math.abs(arr.reduce((x, t) => x + t.R, 0))];
  }).sort((a, b) => b[2] - a[2]);
  const traces = [];
  const top = entries.slice(0, 12), rest = entries.slice(12);
  top.forEach(([s, arr], i) => {
    arr.sort((a, b) => a.Exit_Date.localeCompare(b.Exit_Date));
    let c = 0;
    traces.push({
      x: arr.map(t => t.Exit_Date), y: arr.map(t => (c += t.R)),
      mode: "lines", name: s, line: { width: 1.4, color: PALETTE[i % PALETTE.length] },
    });
  });
  if (rest.length) {
    const all = rest.flatMap(([, arr]) => arr)
      .sort((a, b) => a.Exit_Date.localeCompare(b.Exit_Date));
    let c = 0;
    traces.push({
      x: all.map(t => t.Exit_Date), y: all.map(t => (c += t.R)),
      mode: "lines", name: `Other (${rest.length})`, line: { width: 1.2, color: "#5a6272", dash: "dot" },
    });
  }
  Plotly.react(el, traces, plotLayout({
    height: 330, yaxis: { title: { text: "Cumulative R", font: { size: 11 } } },
    legend: { font: { size: 10 } },
  }), PLOT_CFG);
}

function renderMonthly(ds) {
  const el = document.getElementById("monthlyChart");
  if (!ds.dates.length) { Plotly.purge(el); return; }
  const agg = new Map();
  for (let i = 0; i < ds.dates.length; i++) {
    const y = ds.dates[i].slice(0, 4), m = +ds.dates[i].slice(5, 7);
    const k = y + "-" + m;
    agg.set(k, (agg.get(k) || 0) + ds.pnl[i]);
  }
  const years = [...new Set(ds.dates.map(d => d.slice(0, 4)))].sort().reverse();
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const z = years.map(y => months.map((_, mi) => {
    const v = agg.get(y + "-" + (mi + 1));
    return v == null ? null : +(v / START_EQ * 100).toFixed(2);
  }));
  Plotly.react(el, [{
    z, x: months, y: years, type: "heatmap",
    colorscale: [[0, "#c0392b"], [0.5, "#10141d"], [1, "#0e9f6e"]],
    zmid: 0, showscale: false,
    texttemplate: "%{z:.1f}", textfont: { size: 9 },
    hovertemplate: "%{y} %{x}: %{z:.2f}%<extra></extra>",
  }], plotLayout({
    height: Math.max(280, 17 * years.length + 70),
    margin: { l: 46, t: 8 }, xaxis: { side: "top", gridcolor: "rgba(0,0,0,0)" },
    yaxis: { gridcolor: "rgba(0,0,0,0)", autorange: "reversed", type: "category" },
    hovermode: "closest",
  }), PLOT_CFG);
}

function renderRolling(ds) {
  const el = document.getElementById("rollSharpeChart");
  const W = 252;
  if (ds.pnl.length < W + 10) { Plotly.purge(el); el.innerHTML = '<p class="cap">Not enough days in window for rolling 252d Sharpe.</p>'; return; }
  el.innerHTML = "";
  const rets = ds.pnl.map(v => v / START_EQ);
  // prefix sums for O(n) rolling mean/std
  const n = rets.length, ps = new Float64Array(n + 1), ps2 = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) { ps[i + 1] = ps[i] + rets[i]; ps2[i + 1] = ps2[i] + rets[i] * rets[i]; }
  const xs = [], ys = [];
  for (let i = W; i <= n; i++) {
    const s = ps[i] - ps[i - W], s2 = ps2[i] - ps2[i - W];
    const m = s / W, varr = (s2 - W * m * m) / (W - 1);
    const sd = varr > 0 ? Math.sqrt(varr) : 0;
    xs.push(ds.dates[i - 1]);
    ys.push(sd ? +(m / sd * Math.sqrt(252)).toFixed(3) : null);
  }
  Plotly.react(el, [{
    x: xs, y: ys, mode: "lines", name: "Sharpe (252d)",
    line: { color: "#4da3ff", width: 1.4 },
  }], plotLayout({
    height: 280,
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0,
               line: { color: "#444c5c", width: 1, dash: "dot" } }],
  }), PLOT_CFG);
}

function renderHist(tr) {
  const el = document.getElementById("histChart");
  const rs = tr.map(t => t.R).filter(v => v != null).map(v => Math.max(-3, Math.min(5, v)));
  Plotly.react(el, [{
    x: rs, type: "histogram", xbins: { start: -3, end: 5, size: 0.2 },
    marker: { color: "#4da3ff", line: { color: "#0b0e14", width: 0.5 } },
  }], plotLayout({
    height: 280, hovermode: "closest", bargap: 0.02,
    xaxis: { title: { text: "R multiple (clipped at -3 / +5)", font: { size: 11 } } },
    yaxis: { title: { text: "Trades", font: { size: 11 } } },
  }), PLOT_CFG);
}

function renderSeasonality(tr) {
  const monthEl = document.getElementById("monthSeasChart");
  const wdEl = document.getElementById("weekdaySeasChart");
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const mAgg = new Array(12).fill(0), mCnt = new Array(12).fill(0);
  const wAgg = new Array(7).fill(0), wCnt = new Array(7).fill(0);
  for (const t of tr) {
    if (t.R == null || !t.Entry_Date) continue;
    const d = new Date(t.Entry_Date + "T00:00:00Z");
    mAgg[d.getUTCMonth()] += t.R; mCnt[d.getUTCMonth()]++;
    wAgg[d.getUTCDay()] += t.R; wCnt[d.getUTCDay()]++;
  }
  Plotly.react(monthEl, [{
    x: months, y: mAgg.map(v => +v.toFixed(1)), type: "bar",
    marker: { color: mAgg.map(v => v >= 0 ? "#00d18f" : "#ff5d5d") },
    customdata: mCnt, hovertemplate: "%{x}: %{y:.1f}R (%{customdata} trades)<extra></extra>",
  }], plotLayout({
    height: 250, hovermode: "closest",
    yaxis: { title: { text: "Total R by entry month", font: { size: 11 } } },
  }), PLOT_CFG);
  const wdNames = ["Mon", "Tue", "Wed", "Thu", "Fri"];
  const wy = [1, 2, 3, 4, 5].map(i => +wAgg[i].toFixed(1));
  Plotly.react(wdEl, [{
    x: wdNames, y: wy, type: "bar",
    marker: { color: wy.map(v => v >= 0 ? "#00d18f" : "#ff5d5d") },
    customdata: [1, 2, 3, 4, 5].map(i => wCnt[i]),
    hovertemplate: "%{x}: %{y:.1f}R (%{customdata} trades)<extra></extra>",
  }], plotLayout({
    height: 250, hovermode: "closest",
    yaxis: { title: { text: "Total R by entry weekday", font: { size: 11 } } },
  }), PLOT_CFG);
}

function renderHoldBuckets(tr) {
  const el = document.getElementById("holdChart");
  const buckets = [[0, 2], [3, 5], [6, 10], [11, 21], [22, 42], [43, 9999]];
  const labels = ["0-2d", "3-5d", "6-10d", "11-21d", "22-42d", "43d+"];
  const agg = buckets.map(() => ({ r: 0, n: 0 }));
  for (const t of tr) {
    if (t.R == null || t.Hold_Days == null) continue;
    const i = buckets.findIndex(([a, b]) => t.Hold_Days >= a && t.Hold_Days <= b);
    if (i >= 0) { agg[i].r += t.R; agg[i].n++; }
  }
  Plotly.react(el, [{
    x: labels, y: agg.map(a => +a.r.toFixed(1)), type: "bar",
    marker: { color: agg.map(a => a.r >= 0 ? "#00d18f" : "#ff5d5d") },
    customdata: agg.map(a => a.n),
    hovertemplate: "%{x}: %{y:.1f}R (%{customdata} trades)<extra></extra>",
  }], plotLayout({
    height: 250, hovermode: "closest",
    yaxis: { title: { text: "Total R by holding period", font: { size: 11 } } },
  }), PLOT_CFG);
}

function renderStratTable(tr) {
  const el = document.getElementById("stratTable");
  const groups = new Map();
  for (const t of tr) {
    const k = t.Strategy + "||" + t.Tier;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(t);
  }
  const totalR = tr.reduce((x, t) => x + (t.R || 0), 0);
  const rows = [...groups.entries()].map(([k, arr]) => {
    const [s, tier] = k.split("||");
    const m = tradeMetrics(arr);
    return {
      Strategy: s, Tier: tier, Trades: m.n,
      Win: m.winRate, TotR: m.totR, AvgR: m.avgR, PF: m.pf,
      PnL: m.totPnl, Share: totalR ? m.totR / totalR : null,
      AvgHold: m.avgHold,
    };
  });
  makeTable(el, {
    columns: [
      { key: "Strategy", label: "Strategy", align: "l" },
      { key: "Tier", label: "Tier", align: "l" },
      { key: "Trades", label: "Trades", fmt: v => v.toLocaleString() },
      { key: "Win", label: "Win %", fmt: v => fmt.pct(v, 1) },
      { key: "TotR", label: "Total R", fmt: v => fmt.num(v, 1), cls: clsSign },
      { key: "AvgR", label: "Avg R", fmt: v => fmt.num(v, 3), cls: clsSign },
      { key: "PF", label: "PF", fmt: v => v == null ? "" : fmt.num(v, 2) },
      { key: "PnL", label: "PnL ($)", fmt: v => fmt.money(v), cls: clsSign },
      { key: "Share", label: "% of R", fmt: v => v == null ? "" : fmt.pct(v, 1) },
      { key: "AvgHold", label: "Avg Hold", fmt: v => v == null ? "" : fmt.num(v, 1) },
    ],
    rows, defaultSort: { key: "TotR", dir: -1 },
  });
}

function renderYearTable(tr, ds) {
  const el = document.getElementById("yearTable");
  const years = new Map();
  for (let i = 0; i < ds.dates.length; i++) {
    const y = ds.dates[i].slice(0, 4);
    if (!years.has(y)) years.set(y, { pnl: [], trades: 0, r: 0, wins: 0, closed: 0 });
    years.get(y).pnl.push(ds.pnl[i]);
  }
  for (const t of tr) {
    const y = (t.Entry_Date || "").slice(0, 4);
    if (!years.has(y)) continue;
    const g = years.get(y);
    g.trades++;
    if (t.R != null) g.r += t.R;
    if (t.PnL_flat != null) { g.closed++; if (t.PnL_flat > 0) g.wins++; }
  }
  const rows = [...years.entries()].sort((a, b) => b[0].localeCompare(a[0])).map(([y, g]) => {
    const n = g.pnl.length;
    const tot = g.pnl.reduce((x, v) => x + v, 0);
    const rets = g.pnl.map(v => v / START_EQ);
    const m = n ? rets.reduce((x, v) => x + v, 0) / n : 0;
    const sd = n > 1 ? Math.sqrt(rets.reduce((x, v) => x + (v - m) ** 2, 0) / (n - 1)) : 0;
    let eq = START_EQ, peak = START_EQ, maxDD = 0;
    for (const v of g.pnl) { eq += v; peak = Math.max(peak, eq); maxDD = Math.min(maxDD, eq / peak - 1); }
    return {
      Year: y, Trades: g.trades, Win: g.closed ? g.wins / g.closed : null,
      TotR: g.r, PnL: tot, Ret: tot / START_EQ,
      MaxDD: maxDD, Sharpe: sd ? m / sd * Math.sqrt(252) : null,
    };
  });
  makeTable(el, {
    columns: [
      { key: "Year", label: "Year", align: "l" },
      { key: "Trades", label: "Trades" },
      { key: "Win", label: "Win %", fmt: v => fmt.pct(v, 1) },
      { key: "TotR", label: "Total R", fmt: v => fmt.num(v, 1), cls: clsSign },
      { key: "PnL", label: "PnL ($)", fmt: v => fmt.money(v), cls: clsSign },
      { key: "Ret", label: "Return %", fmt: v => fmt.pct(v, 1), cls: clsSign },
      { key: "MaxDD", label: "Max DD", fmt: v => fmt.pct(v, 1), cls: () => "neg" },
      { key: "Sharpe", label: "Sharpe", fmt: v => v == null ? "" : fmt.num(v, 2) },
    ],
    rows,
  });
}

function renderTradeLog(tr) {
  const el = document.getElementById("tradeLog");
  const columns = [
    { key: "Entry_Date", label: "Entry", align: "l" },
    { key: "Exit_Date", label: "Exit", align: "l" },
    { key: "Strategy", label: "Strategy", align: "l" },
    { key: "Tier", label: "Tier", align: "l" },
    { key: "Ticker", label: "Ticker", align: "l" },
    { key: "Direction", label: "Dir", align: "l",
      fmt: v => `<span class="badge ${v === "Short" ? "dirS" : "dirL"}">${v || ""}</span>` },
    { key: "Entry_Price", label: "Entry $", fmt: v => fmt.num(v, 2) },
    { key: "Exit_Price", label: "Exit $", fmt: v => fmt.num(v, 2) },
    { key: "Return_Pct", label: "Ret %", fmt: v => v == null ? "" : fmt.signed(v, 2), cls: clsSign },
    { key: "R", label: "R", fmt: v => v == null ? "" : fmt.signed(v, 2), cls: clsSign },
    { key: "PnL_flat", label: "PnL ($)", fmt: v => fmt.money(v), cls: clsSign },
    { key: "Hold_Days", label: "Hold", fmt: v => v == null ? "" : v + "d" },
    { key: "Exit_Type", label: "Exit Type", align: "l" },
    { key: "Entry_Criteria", label: "Criteria", align: "l" },
  ];
  const rows = tr.slice().sort((a, b) => (b.Entry_Date || "").localeCompare(a.Entry_Date || ""));
  if (S.tradeLogTable) S.tradeLogTable.setRows(rows);
  else S.tradeLogTable = makeTable(el, {
    columns, rows, pageSize: 25, search: true, csvName: "trades_filtered.csv",
  });
}

/* ---------- static (full-book) sections ---------- */
function renderStatic() {
  // open positions
  const posEl = document.getElementById("positionsTable");
  if (S.positions && S.positions.positions.length) {
    const rows = S.positions.positions;
    const totLong = rows.filter(r => r.Direction === "Long").reduce((x, r) => x + (r.Mkt_Value || 0), 0);
    const totShort = rows.filter(r => r.Direction === "Short").reduce((x, r) => x + (r.Mkt_Value || 0), 0);
    const opnl = rows.reduce((x, r) => x + (r.Open_PnL || 0), 0);
    document.getElementById("posCards").innerHTML = [
      kpiCard("Open Positions", rows.length),
      kpiCard("Long Mkt Value", fmt.money(totLong)),
      kpiCard("Short Mkt Value", fmt.money(totShort)),
      kpiCard("Net", fmt.money(totLong - totShort), clsSign(totLong - totShort)),
      kpiCard("Open PnL", fmt.money(opnl), clsSign(opnl)),
    ].join("");
    makeTable(posEl, {
      columns: [
        { key: "Entry_Date", label: "Entry", align: "l" },
        { key: "Time_Stop", label: "Time Stop", align: "l" },
        { key: "Strategy", label: "Strategy", align: "l" },
        { key: "Tier", label: "Tier", align: "l" },
        { key: "Ticker", label: "Ticker", align: "l" },
        { key: "Direction", label: "Dir", align: "l",
          fmt: v => `<span class="badge ${v === "Short" ? "dirS" : "dirL"}">${v || ""}</span>` },
        { key: "Entry_Price", label: "Entry $", fmt: v => fmt.num(v, 2) },
        { key: "Current_Price", label: "Last $", fmt: v => v == null ? "" : fmt.num(v, 2) },
        { key: "Shares", label: "Shares", fmt: v => fmt.num(v, 1) },
        { key: "Mkt_Value", label: "Mkt Value", fmt: v => fmt.money(v) },
        { key: "Open_PnL", label: "Open PnL", fmt: v => fmt.money(v), cls: clsSign },
      ],
      rows, defaultSort: { key: "Entry_Date", dir: -1 },
    });
  } else {
    posEl.innerHTML = '<p class="cap">No open positions.</p>';
  }

  // compounded reference curve
  const refEl = document.getElementById("refChart");
  if (S.sd && S.sd.equity_compounded) {
    Plotly.react(refEl, [{
      x: S.sd.dates, y: S.sd.equity_compounded, mode: "lines",
      name: "Compounded", line: { color: "#b07cff", width: 1.5 },
    }], plotLayout({
      height: 300, yaxis: { type: "log", tickformat: "$,.0s" },
    }), PLOT_CFG);
  } else refEl.innerHTML = '<p class="cap">No compounded series in this build.</p>';

  // exposure
  const expEl = document.getElementById("expChart");
  if (S.exposure) {
    const e = S.exposure;
    const mk = (y, name, color) => ({ x: e.dates, y, mode: "lines", name, line: { color, width: 1 } });
    Plotly.react(expEl, [
      mk(e.long, "Long", "#00d18f"), mk(e.short, "Short", "#ff5d5d"),
      mk(e.net, "Net", "#4da3ff"), mk(e.gross, "Gross", "#ffc14d"),
    ], plotLayout({ height: 300, yaxis: { ticksuffix: "%" } }), PLOT_CFG);
  } else expEl.innerHTML = '<p class="cap">No exposure series in this build.</p>';

  // correlation
  const corrEl = document.getElementById("corrChart");
  const divEl = document.getElementById("divTable");
  if (S.corr) {
    const c = S.corr;
    const z = c.matrix.map((row, i) => row.map((v, j) => i === j ? null : v));
    Plotly.react(corrEl, [{
      z, x: c.strategies, y: c.strategies, type: "heatmap",
      colorscale: "RdBu", reversescale: true, zmin: -1, zmax: 1,
      texttemplate: "%{z:.2f}", textfont: { size: 8.5 },
      hovertemplate: "%{y} vs %{x}: %{z:.2f}<extra></extra>", showscale: false,
    }], plotLayout({
      height: Math.max(380, 30 * c.strategies.length + 120),
      margin: { l: 140, b: 120 }, hovermode: "closest",
      xaxis: { tickangle: 45, gridcolor: "rgba(0,0,0,0)", tickfont: { size: 9.5 } },
      yaxis: { gridcolor: "rgba(0,0,0,0)", tickfont: { size: 9.5 } },
    }), PLOT_CFG);
    makeTable(divEl, {
      columns: [
        { key: "strategy", label: "Strategy", align: "l" },
        { key: "avg_corr", label: "Avg Corr", fmt: v => fmt.num(v, 2),
          cls: v => v < 0.2 ? "pos" : v < 0.45 ? "neu" : "neg" },
        { key: "max_corr", label: "Max Corr", fmt: v => fmt.num(v, 2) },
        { key: "max_with", label: "Most Correlated With", align: "l" },
      ],
      rows: c.diversification,
    });
  } else {
    corrEl.innerHTML = '<p class="cap">No correlation matrix in this build.</p>';
    divEl.innerHTML = "";
  }
}
