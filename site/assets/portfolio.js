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
  sizing: "scaled",       // "scaled" (compounds with equity) | "flat" (fixed base)
  lev: 1.0,               // portfolio leverage multiplier
  mult: new Map(),        // strategy -> risk multiplier (default 1)
  nativeBps: new Map(),   // strategy -> median Risk_bps from the ledger
  rangeTouched: false,    // user has interacted with the Range control
  tradeLogTable: null,
};
// Portfolio-level math runs on the ledger's $750k flat allocation; equity
// curves and portfolio dollars are DISPLAYED from a $10k start (returns,
// Sharpe, DD are scale-invariant). Trade-level dollars (trade log, open
// positions, expectancy, per-strategy PnL) stay at the $750k allocation.
const START_EQ = 750000;
const DISPLAY_EQ = 10000;
const DSCALE = DISPLAY_EQ / START_EQ;

function multFor(strat) {
  const m = S.mult.get(strat);
  return (m == null ? 1 : m) * S.lev;
}
function anyMultActive() {
  if (S.lev !== 1) return true;
  for (const v of S.mult.values()) if (v !== 1) return true;
  return false;
}

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
      `${meta.date_min} to ${meta.date_max} · $750k base, filter-exact recompute`;
    computeNativeBps();
    buildFilterBar();
    buildSizingSeg();
    buildRiskPanel();
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
    S.rangeTouched = true;
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
  fromInp.addEventListener("change", () => { S.f.from = fromInp.value || null; S.f.preset = "Custom"; S.rangeTouched = true; markSeg(presetSeg, null); apply(); });
  toInp.addEventListener("change", () => { S.f.to = toInp.value || null; S.f.preset = "Custom"; S.rangeTouched = true; markSeg(presetSeg, null); apply(); });
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
    S.rangeTouched = false;
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

/* ================= sizing toggle + risk panel ================= */
function buildSizingSeg() {
  const host = document.getElementById("sizingSeg");
  const seg = document.createElement("span");
  seg.className = "seg";
  for (const [val, label] of [["scaled", "Scaled with portfolio"], ["flat", "Flat $750k"]]) {
    const b = document.createElement("button");
    b.textContent = label;
    if (val === S.sizing) b.classList.add("on");
    b.addEventListener("click", () => {
      S.sizing = val;
      seg.querySelectorAll("button").forEach(x => x.classList.remove("on"));
      b.classList.add("on");
      apply();
    });
    seg.appendChild(b);
  }
  host.appendChild(seg);
}

function computeNativeBps() {
  const byStrat = new Map();
  for (const t of S.trades) {
    if (t.Risk_bps == null) continue;
    if (!byStrat.has(t.Strategy)) byStrat.set(t.Strategy, []);
    byStrat.get(t.Strategy).push(t.Risk_bps);
  }
  for (const [s, arr] of byStrat) {
    arr.sort((a, b) => a - b);
    S.nativeBps.set(s, arr[Math.floor(arr.length / 2)]);
  }
}

function buildRiskPanel() {
  const names = allStrategyNames();
  for (const n of names) if (!S.mult.has(n)) S.mult.set(n, 1);

  // leverage slider row
  const levRow = document.getElementById("levRow");
  levRow.className = "levrow";
  levRow.innerHTML = `<label>Portfolio leverage</label>
    <input type="range" id="levSlider" min="0" max="3" step="0.05" value="1">
    <span class="lv" id="levVal">1.00x</span>
    <button class="btn ghost" id="riskReset">Reset all</button>`;
  const slider = levRow.querySelector("#levSlider");
  const levVal = levRow.querySelector("#levVal");
  let tmr = null;
  slider.addEventListener("input", () => {
    S.lev = +slider.value;
    levVal.textContent = S.lev.toFixed(2) + "x";
    syncRiskUI();
    clearTimeout(tmr);
    tmr = setTimeout(apply, 160);
  });

  // per-strategy multiplier grid
  const grid = document.getElementById("multGrid");
  grid.innerHTML = "";
  const rows = [];
  for (const n of names) {
    const row = document.createElement("div");
    row.className = "multrow";
    const native = S.nativeBps.get(n);
    row.innerHTML = `<span class="nm" title="${n}">${n}</span>
      <input type="number" min="0" max="5" step="0.1" value="1">
      <span class="bps"></span>`;
    const inp = row.querySelector("input");
    inp.addEventListener("change", () => {
      let v = parseFloat(inp.value);
      if (!isFinite(v) || v < 0) v = 0;
      if (v > 5) v = 5;
      inp.value = v;
      S.mult.set(n, v);
      syncRiskUI();
      apply();
    });
    rows.push({ name: n, inp, bpsEl: row.querySelector(".bps"), native });
    grid.appendChild(row);
  }

  levRow.querySelector("#riskReset").addEventListener("click", () => {
    S.lev = 1; slider.value = "1"; levVal.textContent = "1.00x";
    for (const r of rows) { r.inp.value = "1"; S.mult.set(r.name, 1); }
    syncRiskUI();
    apply();
  });

  function syncRiskUI() {
    for (const r of rows) {
      const m = S.mult.get(r.name);
      r.inp.classList.toggle("tweaked", m !== 1);
      if (r.native != null) {
        const eff = r.native * m * S.lev;
        r.bpsEl.textContent = `${r.native.toFixed(0)} -> ${eff.toFixed(0)} bps`;
      } else r.bpsEl.textContent = "";
    }
    const summary = document.getElementById("riskSummary");
    summary.textContent = anyMultActive()
      ? `(ACTIVE: ${S.lev.toFixed(2)}x leverage` +
        ([...S.mult.values()].some(v => v !== 1) ? ", per-strategy overrides set)" : ")")
      : "(all at native risk)";
  }
  S.syncRiskUI = syncRiskUI;
  syncRiskUI();
}

/* ================= filtering ================= */
function tickerTokens() {
  if (!S.f.tickerQ) return null;
  return S.f.tickerQ.toUpperCase().split(",").map(s => s.trim()).filter(Boolean);
}

function filteredTrades(ignoreDates) {
  const toks = tickerTokens();
  const { strategies, tier, dir, from, to } = S.f;
  return S.trades.filter(t => {
    if (!strategies.has(t.Strategy)) return false;
    if (tier !== "All" && t.Tier !== tier) return false;
    if (dir !== "All" && t.Direction !== dir) return false;
    if (!ignoreDates) {
      const d = t.Entry_Date || t.Signal_Date;
      if (from && d < from) return false;
      if (to && d > to) return false;
    }
    if (toks && !toks.some(tok => (t.Ticker || "").toUpperCase().startsWith(tok))) return false;
    return true;
  });
}

function curveExact() {
  return S.sd && S.f.dir === "All" && !tickerTokens();
}

/* daily pnl array (risk multipliers + leverage applied) for current filters */
function dailySeries(trades, ignoreDates) {
  if (curveExact()) {
    const keys = Object.keys(S.sd.series).filter(k => {
      const [strat, tier] = k.split("||");
      if (!S.f.strategies.has(strat)) return false;
      if (S.f.tier !== "All" && tier !== S.f.tier) return false;
      return true;
    });
    let i0 = 0, i1 = S.dateIdx.length - 1;
    if (!ignoreDates && S.f.from) i0 = lowerBound(S.dateIdx, S.f.from);
    if (!ignoreDates && S.f.to) i1 = upperBound(S.dateIdx, S.f.to) - 1;
    if (i1 < i0) return { dates: [], pnl: [], exact: true };
    const n = i1 - i0 + 1;
    const pnl = new Float64Array(n);
    for (const k of keys) {
      const m = multFor(k.split("||")[0]);
      if (m === 0) continue;
      const arr = S.sd.series[k];
      for (let i = 0; i < n; i++) pnl[i] += arr[i0 + i] * m;
    }
    return { dates: S.dateIdx.slice(i0, i1 + 1), pnl: Array.from(pnl), exact: true };
  }
  // fallback: realized PnL on exit dates
  const map = new Map();
  for (const t of trades) {
    const d = t.Exit_Date;
    if (!d || t.PnL_flat == null) continue;
    map.set(d, (map.get(d) || 0) + t.PnL_flat * multFor(t.Strategy));
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
  // Dollar metrics carry the risk multipliers + leverage; R stats stay raw
  // (R is per unit of risk by definition).
  const pnls = tr.filter(t => t.PnL_flat != null)
                 .map(t => t.PnL_flat * multFor(t.Strategy));
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
  const scaled = S.sizing === "scaled";
  // Arithmetic daily returns on the $750k base drive Sharpe / Sortino / vol
  // in both modes (standard convention). In scaled mode the equity path
  // compounds them geometrically: sizes grow/shrink with equity each day.
  const rets = pnl.map(v => v / START_EQ);
  const sum = a => a.reduce((x, y) => x + y, 0);
  const m = sum(rets) / n;
  const sd = n > 1 ? Math.sqrt(sum(rets.map(v => (v - m) ** 2)) / (n - 1)) : 0;
  const downside = rets.filter(v => v < 0);
  const dsd = downside.length > 1 ?
    Math.sqrt(sum(downside.map(v => v * v)) / downside.length) : 0;
  // equity path + max drawdown vs running peak (per active sizing mode),
  // displayed from a $10k start (scale-invariant in % terms)
  let eq = DISPLAY_EQ, peak = DISPLAY_EQ, maxDD = 0;
  let bestDay = -Infinity, worstDay = Infinity;
  const equity = new Array(n);
  for (let i = 0; i < n; i++) {
    const dayPnl = scaled ? eq * rets[i] : pnl[i] * DSCALE;
    eq += dayPnl;
    if (eq <= 0) eq = 0.01; // leverage blow-up floor; keeps log axis sane
    equity[i] = eq;
    if (dayPnl > bestDay) bestDay = dayPnl;
    if (dayPnl < worstDay) worstDay = dayPnl;
    if (eq > peak) peak = eq;
    const dd = eq / peak - 1;
    if (dd < maxDD) maxDD = dd;
  }
  const years = n / 252;
  const annRet = scaled
    ? (years > 0 ? Math.pow(eq / DISPLAY_EQ, 1 / years) - 1 : null)  // CAGR
    : m * 252;                                                        // arithmetic
  const annVol = sd * Math.sqrt(252);
  return {
    equity, annRet, annVol,
    sharpe: sd ? (m / sd) * Math.sqrt(252) : null,
    sortino: dsd ? (m / dsd) * Math.sqrt(252) : null,
    maxDD,
    mar: (maxDD && annRet != null) ? annRet / Math.abs(maxDD) : null,
    totPnl: eq - DISPLAY_EQ,
    bestDay, worstDay,
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
  // rolling Sharpe needs 252d of warmup, so it computes on the date-unbounded
  // stream (same strategy/tier/dir/ticker filters) and only displays the window
  renderRolling(dailySeries(filteredTrades(true), true));
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
  const scaled = S.sizing === "scaled";
  el.innerHTML = [
    kpiCard("Trades", tm.n.toLocaleString()),
    kpiCard("Win Rate", tm.winRate == null ? "-" : fmt.pct(tm.winRate, 1)),
    kpiCard("Total R", fmt.num(tm.totR, 0), clsSign(tm.totR)),
    kpiCard("Avg R", tm.avgR == null ? "-" : fmt.num(tm.avgR, 3), clsSign(tm.avgR)),
    kpiCard("Profit Factor", tm.pf == null ? "-" : fmt.num(tm.pf, 2)),
    kpiCard("Expectancy", tm.expectancy == null ? "-" : fmt.money(tm.expectancy), clsSign(tm.expectancy), "per trade @ $750k alloc"),
    kpiCard("SQN", tm.sqn == null ? "-" : fmt.num(tm.sqn, 2)),
    kpiCard("Payoff", tm.payoff == null ? "-" : fmt.num(tm.payoff, 2), null, "avg win / avg loss"),
    kpiCard("Total PnL", dm.totPnl == null ? "-" : fmt.money(dm.totPnl), clsSign(dm.totPnl),
            scaled ? "compounded, $10k start" : "flat, $10k base"),
    kpiCard(scaled ? "CAGR" : "Ann Return",
            dm.annRet == null ? "-" : fmt.pct(dm.annRet, 1), clsSign(dm.annRet),
            scaled ? "geometric" : "of $750k"),
    kpiCard("Ann Vol", dm.annVol == null ? "-" : fmt.pct(dm.annVol, 1)),
    kpiCard("Sharpe", dm.sharpe == null ? "-" : fmt.num(dm.sharpe, 2), clsSign(dm.sharpe)),
    kpiCard("Sortino", dm.sortino == null ? "-" : fmt.num(dm.sortino, 2)),
    kpiCard("Max Drawdown", dm.maxDD == null ? "-" : fmt.pct(dm.maxDD, 1), "neg",
            scaled ? "compounded path" : "flat path"),
    kpiCard("MAR", dm.mar == null ? "-" : fmt.num(dm.mar, 2)),
    kpiCard("Tail Ratio", tm.tail == null ? "-" : fmt.num(tm.tail, 2), null, "|p95 / p5| of R"),
    kpiCard("Max Consec Losses", tm.maxConsecL),
    kpiCard("Avg Hold", tm.avgHold == null ? "-" : fmt.num(tm.avgHold, 1), null, "trading days"),
  ].join("");
}

function renderEquity(ds, dm) {
  const eqEl = document.getElementById("eqChart");
  const ddEl = document.getElementById("ddChart");
  const cap = document.getElementById("eqCaption");
  if (!ds.dates.length) { Plotly.purge(eqEl); Plotly.purge(ddEl); cap.textContent = ""; return; }
  const scaled = S.sizing === "scaled";
  cap.textContent = (scaled
    ? "Scaled with portfolio size: daily returns compound geometrically (risk grows/shrinks with equity). Log scale, $10k start."
    : "Flat sizing: every trade risks bps of a fixed allocation. Era-comparable; best for judging raw edge. $10k display base.")
    + (S.rangeTouched ? "" : " Showing trailing 1y by default — pick a Range or double-click to zoom out.");
  const equity = dm.equity;
  const lastDate = ds.dates[ds.dates.length - 1];

  // Default (untouched Range): trailing-1y viewport over full-history data.
  // Once the user picks a Range, show exactly the filtered span (autorange).
  let xRange = null, yRange = null, ddRangeY = null, w0 = 0;
  let peak = -Infinity;
  const dd = equity.map(v => { peak = Math.max(peak, v); return (v / peak - 1) * 100; });
  if (!S.rangeTouched) {
    const cut = new Date(lastDate + "T00:00:00Z");
    cut.setUTCFullYear(cut.getUTCFullYear() - 1);
    const cutStr = cut.toISOString().slice(0, 10);
    w0 = lowerBound(ds.dates, cutStr);
    const winEq = equity.slice(w0);
    if (winEq.length) {
      xRange = [cutStr, lastDate];
      const lo = Math.min(...winEq), hi = Math.max(...winEq);
      yRange = scaled
        ? [Math.log10(Math.max(lo, 0.01) * 0.98), Math.log10(hi * 1.02)]
        : [lo - (hi - lo) * 0.06 - 1, hi + (hi - lo) * 0.06 + 1];
      const winDD = dd.slice(w0);
      ddRangeY = [Math.min(...winDD) * 1.15 - 0.1, 0.5];
    }
  }

  Plotly.react(eqEl, [{
    x: ds.dates, y: equity, mode: "lines", name: "Equity",
    line: { color: "#00d18f", width: 1.8 },
  }], plotLayout({
    height: 330,
    xaxis: xRange ? { range: xRange } : {},
    yaxis: { type: scaled ? "log" : "linear", tickformat: "$,.4~s",
             range: yRange,
             title: { text: scaled ? "Equity (compounded, log, $10k start)" : "Equity (flat, $10k base)", font: { size: 11 } } },
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: DISPLAY_EQ, y1: DISPLAY_EQ,
               line: { color: "#444c5c", width: 1, dash: "dot" } }],
  }), PLOT_CFG);

  Plotly.react(ddEl, [{
    x: ds.dates, y: dd, mode: "lines", name: "Drawdown",
    fill: "tozeroy", line: { color: "#ff5d5d", width: 1 }, fillcolor: "rgba(255,93,93,.18)",
  }], plotLayout({
    height: 150, margin: { t: 8 },
    xaxis: xRange ? { range: xRange } : {},
    yaxis: { ticksuffix: "%", range: ddRangeY },
  }), PLOT_CFG);
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

function renderRolling(dsAll) {
  const el = document.getElementById("rollSharpeChart");
  const W = 252;
  if (dsAll.pnl.length < W + 10) {
    Plotly.purge(el);
    el.innerHTML = '<p class="cap">Fewer than ~262 trading days of history for these filters — rolling 252d Sharpe unavailable.</p>';
    return;
  }
  // Only clear the container when it holds the fallback caption (no live plot).
  // Wiping innerHTML on an element Plotly has already initialized breaks
  // Plotly.react on every subsequent render.
  if (!el._fullLayout) el.innerHTML = "";
  const rets = dsAll.pnl.map(v => v / START_EQ);
  // prefix sums for O(n) rolling mean/std
  const n = rets.length, ps = new Float64Array(n + 1), ps2 = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) { ps[i + 1] = ps[i] + rets[i]; ps2[i + 1] = ps2[i] + rets[i] * rets[i]; }
  const xs = [], ys = [];
  for (let i = W; i <= n; i++) {
    const s = ps[i] - ps[i - W], s2 = ps2[i] - ps2[i - W];
    const m = s / W, varr = (s2 - W * m * m) / (W - 1);
    const sd = varr > 0 ? Math.sqrt(varr) : 0;
    xs.push(dsAll.dates[i - 1]);
    ys.push(sd ? +(m / sd * Math.sqrt(252)).toFixed(3) : null);
  }
  // display only the active date window (computed with full-history warmup)
  let xRange = null;
  if (S.f.from || S.f.to) {
    xRange = [S.f.from || xs[0], S.f.to || xs[xs.length - 1]];
  }
  Plotly.react(el, [{
    x: xs, y: ys, mode: "lines", name: "Sharpe (252d)",
    line: { color: "#4da3ff", width: 1.4 },
  }], plotLayout({
    height: 280,
    xaxis: xRange ? { range: xRange } : {},
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
      TotR: g.r, PnL: tot * DSCALE, Ret: tot / START_EQ,
      MaxDD: maxDD, Sharpe: sd ? m / sd * Math.sqrt(252) : null,
    };
  });
  makeTable(el, {
    columns: [
      { key: "Year", label: "Year", align: "l" },
      { key: "Trades", label: "Trades" },
      { key: "Win", label: "Win %", fmt: v => fmt.pct(v, 1) },
      { key: "TotR", label: "Total R", fmt: v => fmt.num(v, 1), cls: clsSign },
      { key: "PnL", label: "PnL ($10k base)", fmt: v => fmt.money(v), cls: clsSign },
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
  // closed trades only — open positions (time stop not reached) live in the
  // Open Positions section above
  const rows = tr.filter(t => !t.Open)
    .sort((a, b) => (b.Entry_Date || "").localeCompare(a.Entry_Date || ""));
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
