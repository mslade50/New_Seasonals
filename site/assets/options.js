/* options.js — Options Workbench: signal-to-structure pipeline.

   Flow (top to bottom): prefill (query string from a Signals-page "Express"
   button, or manual ticker) -> IV context strip (rank/percentile/RV from
   iv_context.json) -> expiry picker (real listed expiries via the agent) ->
   edge-vs-priced expected-move comparator (implied move over the hold vs the
   strategy's terminal move from strategy_stats.json) -> structure shootout at
   equal risk -> risk-bps sizing (debit-at-risk) -> BAG combo ticket.

   Live chain data comes from POST /exec-workbench (agent -> option_workbench.py,
   one round-trip; expiry clicks re-query mode:"chain"). Orders go through the
   signed /exec-command path as type "option_spread" — DRY-RUN until the user
   arms the type in exec_agent.env. Exit-date P&L columns are client-side BSM
   (bsm.js) with dividend/rate inputs — labeled approximation.

   EM convention (everywhere): 1-sigma move = IV_ATM * sqrt(days/365), calendar
   days. The straddle mid is shown as "market price of the move", never blended. */
"use strict";

document.addEventListener("DOMContentLoaded", initOptions);

const COMM = 0.65;                    // $/contract per leg per side
const state = {
  params: {},                          // prefill from the query string
  ivCtx: null, market: null, stats: null, earn: null, signals: null,
  marketGroup: "All ETFs",
  wb: null,                            // latest workbench result
  wbId: null, wbTimer: null,
  structures: [],                      // shootout rows (assembled client-side)
  selStructure: null,
  manual: { view: "bullish", horizon: 10, risk: 1500, target: null, adverse: null, fraction: 1.0 },
  forecast: {
    active: false, event: "touch", target: null, probability: 0.50,
    cutoff: defaultForecastDate(), touchIvShift: 8, noTouchSpot: null,
    noTouchIvShift: -3, objective: "ev_risk",
  },
  pricing: { ivShiftPts: -3, rate: 0.04, divYield: 0 },
  account: "primary",
  book: null, status: null, commands: [],
  pollTimer: null,
};

/* ---------------- init + prefill ---------------- */
async function initOptions() {
  renderNav("options.html");
  state.params = parseParams();
  state.manual.view = isShort() ? "bearish" : "bullish";
  state.manual.horizon = state.params.hold || 10;
  state.manual.risk = state.params.risk || 1500;
  state.manual.fraction = hasSignal() ? 0.5 : 1.0;
  const [iv, market, stats, earn, sig] = await Promise.all([
    fetchJSONOrNull("data/iv_context.json"),
    fetchJSONOrNull("data/options_market.json"),
    fetchJSONOrNull("data/strategy_stats.json"),
    fetchJSONOrNull("data/earnings_next.json"),
    fetchJSONOrNull("data/signals.json"),
  ]);
  state.ivCtx = iv; state.market = market; state.stats = stats; state.earn = earn; state.signals = sig;
  document.getElementById("content").innerHTML = shell();
  document.getElementById("wbGo").addEventListener("click", () => loadTicker());
  document.getElementById("wbTicker").addEventListener("keydown", (e) => { if (e.key === "Enter") loadTicker(); });
  document.getElementById("wbView").addEventListener("change", (e) => {
    state.manual.view = e.target.value;
    state.params.dir = e.target.value === "bearish" || e.target.value === "hedge" ? "Short" : "Long";
    if (!hasSignal()) { state.manual.target = null; state.manual.adverse = null; }
    if (state.wb) renderAll();
  });
  ["wbHorizon", "wbRisk"].forEach((id) => document.getElementById(id).addEventListener("change", syncToolbar));
  document.querySelectorAll("[data-acct]").forEach((b) =>
    b.addEventListener("click", () => setAccount(b.dataset.acct)));
  renderMarketOverview();
  await pollExec();
  state.pollTimer = setInterval(pollExec, 8000);
  if (state.params.ticker) {
    document.getElementById("wbTicker").value = state.params.ticker;
    loadTicker();
  }
}

function parseParams() {
  const q = new URLSearchParams(location.search);
  const num = (k) => { const v = q.get(k); return v == null || v === "" ? null : Number(v); };
  return {
    ticker: (q.get("ticker") || "").toUpperCase().trim() || null,
    dir: q.get("dir") || "Long",
    entry: num("entry"), stop: num("stop"), target: num("target"), atr: num("atr"),
    risk: num("risk"), hold: num("hold"),
    texit: q.get("texit") || null, fw: num("fw"),
    strategy: q.get("strategy") || null, sig: q.get("sig") || null,
    cond: q.get("cond") || null,
  };
}

function hasSignal() { const p = state.params; return !!(p.entry && p.stop && p.risk); }
function isShort() { return String(state.params.dir).toUpperCase().includes("SHORT"); }
function currentView() {
  const e = document.getElementById("wbView");
  return (e && e.value) || state.manual.view || (isShort() ? "bearish" : "bullish");
}
function tradeHorizon() {
  const e = document.getElementById("wbHorizon");
  const v = e ? Number(e.value) : state.manual.horizon;
  return isFinite(v) && v > 0 ? Math.round(v) : (state.params.hold || 10);
}
function tradeRisk() {
  const e = document.getElementById("wbRisk");
  const v = e ? Number(e.value) : state.manual.risk;
  return isFinite(v) && v > 0 ? v : 0;
}
function syncToolbar() {
  state.manual.horizon = tradeHorizon();
  state.manual.risk = tradeRisk();
  if (state.wb) {
    renderThesis(); renderComparator(); buildStructures(); renderForecastLab(); renderShootout();
    renderScenarioLab(); renderSizing(); renderTicket();
  }
}

function shell() {
  const p = state.params;
  const view = isShort() ? "bearish" : "bullish";
  return `
    <div id="modeBanner"></div>
    <div id="marketOverview"></div>
    <div class="card opt-toolbar">
      <label><span>Ticker</span><input id="wbTicker" placeholder="SPY" style="text-transform:uppercase"></label>
      <label><span>View</span><select id="wbView">
        <option value="bullish"${view === "bullish" ? " selected" : ""}>Bullish</option>
        <option value="bearish"${view === "bearish" ? " selected" : ""}>Bearish</option>
        <option value="big_move">Big move</option><option value="range">Range / short vol</option>
        <option value="hedge">Portfolio hedge</option></select></label>
      <label><span>Horizon (td)</span><input id="wbHorizon" type="number" min="1" max="63" value="${p.hold || 10}"${hasSignal() ? " disabled" : ""}></label>
      <label><span>Risk budget</span><input id="wbRisk" type="number" min="1" step="100" value="${p.risk || 1500}"${hasSignal() ? " disabled" : ""}></label>
      <button class="btn" id="wbGo">Analyze chain</button>
      <span class="exec-tabs"><button class="btn" data-acct="primary">Primary</button>
        <button class="btn ghost" data-acct="pa">PA</button></span>
      <span id="wbMsg" class="cap"></span>
      <span id="connDot" class="cap" style="margin-left:auto"></span>
    </div>
    ${signalContextHtml(p)}
    <div id="thesis"></div>
    <div id="forecastLab"></div>
    <div id="ivStrip"></div>
    <div id="volDashboard"></div>
    <div id="termLab"></div>
    <div id="expiryRow"></div>
    <div id="emCompare"></div>
    <div id="shootout"></div>
    <div id="scenarioLab"></div>
    <div id="sizing"></div>
    <div id="ticket"></div>
    <div id="activity" style="margin-top:18px"></div>`;
}

function signalContextHtml(p) {
  if (!hasSignal()) {
    return `<p class="cap" style="margin:0 0 12px">No signal context — manual mode. Open this page via a
      Signals-page <b>Express</b> button to pre-fill entry/stop/target/risk and unlock the comparator,
      shootout scenario P&amp;L, and sizing.</p>`;
  }
  const dupe = stagedStockRow(p.ticker);
  const dupeWarn = dupe ? `<div class="oc-line" style="color:#ffc14d"><b>&#9888; Stock signal also staged for
      ${esc(p.ticker)}</b> — an options ticket runs ALONGSIDE it (combined delta). Size the fraction below accordingly.</div>` : "";
  return `<div class="card" style="margin-bottom:12px">
    <div style="font:700 14px inherit">${esc(p.ticker || "")} · ${esc(p.dir)} ·
      <span style="color:#4da3ff">${esc(p.strategy || "manual")}</span>
      <span class="cap" style="display:inline;margin-left:8px">signal ${esc(p.sig || "?")}</span></div>
    <div class="kv" style="margin-top:8px">
      <div class="k">Entry / Stop / Target</div><div class="v">${fmt.num(p.entry, 2)} / ${fmt.num(p.stop, 2)} / ${p.target ? fmt.num(p.target, 2) : "—"}</div>
      <div class="k">ATR / Hold / Time exit</div><div class="v">${fmt.num(p.atr, 2)} / ${p.hold || "?"} td / ${esc(p.texit || "?")}</div>
      <div class="k">Stock Risk_Amt</div><div class="v">${fmt.money(p.risk)}</div>
    </div>
    ${p.cond ? `<div class="openconds" style="margin-top:8px"><div class="oc-h">Entry condition (stock-side — submit the combo manually when it triggers)</div>
      <div class="oc-line">${esc(p.cond)}</div></div>` : ""}
    ${dupeWarn}</div>`;
}

function stagedStockRow(ticker) {
  const tabs = (state.signals && state.signals.tabs) || {};
  for (const rows of Object.values(tabs)) {
    const hit = (rows || []).find((r) => String(r.Symbol || "").toUpperCase() === ticker);
    if (hit) return hit;
  }
  return null;
}

/* ---------------- market weather + manual thesis ---------------- */
function renderMarketOverview() {
  const el = document.getElementById("marketOverview");
  const m = state.market;
  if (!el || !m || !m.n) { if (el) el.innerHTML = ""; return; }
  const allRows = m.etfs || [];
  const rows = state.marketGroup === "All ETFs" ? allRows : allRows.filter((r) => r.group === state.marketGroup);
  const medianOf = (values) => {
    const clean = values.filter((x) => x != null && isFinite(Number(x))).map(Number).sort((a, b) => a - b);
    if (!clean.length) return null;
    const mid = Math.floor(clean.length / 2);
    return clean.length % 2 ? clean[mid] : (clean[mid - 1] + clean[mid]) / 2;
  };
  const median = medianOf(rows.map((r) => r.pctile));
  const medianVrp = medianOf(rows.map((r) => r.vrp));
  const asofMs = m.asof ? new Date(m.asof + "T00:00:00").getTime() : NaN;
  const ageDays = isFinite(asofMs) ? Math.floor((Date.now() - asofMs) / 86400000) : null;
  const stale = ageDays != null && ageDays > 5;
  const label = stale ? "ETF volatility map is stale" : median == null ? "No volatility history in this sleeve" : median < 35 ? "ETF vol is broadly low" : median > 65 ? "ETF vol is broadly elevated" : "ETF vol is mixed";
  const tone = stale || median == null ? "#ffc14d" : median < 35 ? "#3ddb8f" : median > 65 ? "#ff6b6b" : "#ffc14d";
  const groups = ["All ETFs", ...(m.groups || []).map((g) => g.name)];
  const groupTabs = groups.map((g) => `<button class="btn xs ${g === state.marketGroup ? "" : "ghost"}" data-opt-group="${esc(g)}">${esc(g)}</button>`).join("");
  const medRv = medianOf(rows.map((r) => r.rv_pctile));
  const disp = m.dispersion || {};
  const corrNow = disp.sector_corr_21d;
  const laneMeta = {
    "Own convexity": ["#3ddb8f", "Low IV, low RV and little premium. Long gamma/vega candidates."],
    "Harvest premium": ["#ff6b6b", "High IV/RV premium. Defined-risk option-sale candidates."],
    "Calendar watch": ["#4da3ff", "Low IV but rich versus RV. Confirm a flat live curve before buying the calendar."],
    "Front-stress watch": ["#ffc14d", "High IV and RV with little premium. Confirm backwardation before fading stress."],
  };
  const lanesHtml = Object.entries(laneMeta).map(([name, meta]) => {
    const picks = rows.slice()
      .sort((a, b) => Number((b.fits || {})[name] || 0) - Number((a.fits || {})[name] || 0))
      .slice(0, 4)
      .map((r) => ({...r, score: (r.fits || {})[name] || 0}));
    return `<div class="opt-lane" style="border-top-color:${meta[0]}"><div class="opt-lane-head">${esc(name)}</div>
      <div class="cap">${meta[1]}</div>
      <div class="opt-lane-picks">${picks.length ? picks.map((r) => `<button class="opt-ticker-link" data-opt-ticker="${r.ticker}"><b>${r.ticker}</b><span>${fmt.num(r.score, 0)}</span></button>`).join("") : '<span class="cap">No covered names in this sleeve.</span>'}</div></div>`;
  }).join("");
  const tableRows = rows.slice().sort((a, b) => (b.setup_score || 0) - (a.setup_score || 0)).map((r) => `<tr>
    <td><button class="opt-ticker-link compact" data-opt-ticker="${r.ticker}"><b>${r.ticker}</b></button></td>
    <td>${esc(r.group || "")}</td><td>${fmt.pctRaw((r.iv || 0) * 100, 1)}</td>
    <td>p${fmt.num(r.pctile, 0)}</td><td>${fmt.pctRaw((r.rv21 || 0) * 100, 1)}</td>
    <td>${r.rv_pctile != null ? "p" + fmt.num(r.rv_pctile, 0) : "—"}</td>
    <td class="${r.vrp > 0 ? "neg" : "pos"}">${fmt.pct(r.vrp, 0)}</td>
    <td>${r.iv_change_1d != null ? fmt.signed(r.iv_change_1d * 100, 1) : "—"}</td>
    <td><span class="opt-setup-pill">${esc(r.setup || "unranked")}</span></td></tr>`).join("");
  el.innerHTML = `<section class="opt-market-shell">
    <div class="card opt-market-hero">
      <div><div class="cap">ETF / index volatility cockpit · ${rows.length} covered in this view</div>
        <div class="opt-hero-value" style="color:${tone}">${label}</div>
        <div class="opt-meter"><i style="margin-left:${Math.max(0, Math.min(100, median))}%"></i></div></div>
      <div class="opt-market-kpis">
        <div><span>IV percentile</span><b>${fmt.num(median, 0)}</b></div>
        <div><span>Median VRP</span><b>${medianVrp != null ? fmt.pct(medianVrp, 0) : "—"}</b></div>
        <div><span>RV percentile</span><b>${medRv != null ? fmt.num(medRv, 0) : "—"}</b></div>
        <div><span>Sector corr 21d</span><b>${corrNow != null ? fmt.num(corrNow, 2) : "—"}</b></div>
      </div>
      <div class="cap">${stale ? `<b style="color:#ffc14d">${ageDays}d old — do not trade from the ranks.</b> · ` : ""}As of ${esc(m.asof || "?")}. IV = IBKR 30d underlying IV; RV = 21d Yang–Zhang. Scanner scores are cross-sectional screens, not signals.</div>
    </div>
    <div class="opt-group-tabs">${groupTabs}</div>
    <div class="opt-market-grid">
      <div class="card"><div class="opt-section-head"><div><b>Cross-asset volatility map</b><div class="cap">RV percentile → x · IV/RV premium → y · color = own-history IV percentile</div></div></div><div id="optMarketScatter" class="opt-market-chart"></div></div>
      <div class="card opt-dispersion-card"><div class="opt-section-head"><div><b>Index / sector dispersion monitor</b><div class="cap">A pricing lens, deliberately not labeled implied correlation</div></div></div>
        <div class="opt-dispersion-read"><div><span>Sector IV minus SPY</span><b>${disp.iv_spread_points != null ? fmt.signed(disp.iv_spread_points, 1) + " pts" : "—"}</b></div>
          <div><span>Sector RV minus SPY</span><b>${disp.rv_spread_points != null ? fmt.signed(disp.rv_spread_points, 1) + " pts" : "—"}</b></div>
          <div><span>Avg sector corr</span><b>${corrNow != null ? fmt.num(corrNow, 2) : "—"}</b></div></div>
        <p class="cap">${esc(disp.basis || "Needs SPY plus sector ETF coverage.")}</p>
        <div class="opt-dispersion-note">Wide sector/index vol spreads plus low realized correlation favor a dispersion lens. A correlation rebound is the central risk.</div></div>
    </div>
    <div class="opt-lanes">${lanesHtml}</div>
    <details class="card opt-etf-tape"><summary><b>ETF volatility tape</b> <span class="cap">· all four static dimensions · click a ticker for the live curve and skew</span></summary>
      <div class="table-wrap"><table><thead><tr><th>Ticker</th><th>Sleeve</th><th>IV30</th><th>IV pct</th><th>RV21</th><th>RV pct</th><th>VRP</th><th>ΔIV 1d</th><th>Nearest screen</th></tr></thead><tbody>${tableRows}</tbody></table></div>
    </details>
  </section>`;
  el.querySelectorAll("[data-opt-group]").forEach((b) => b.addEventListener("click", () => {
    state.marketGroup = b.dataset.optGroup; renderMarketOverview();
  }));
  el.querySelectorAll("[data-opt-ticker]").forEach((b) => b.addEventListener("click", () => {
    const input = document.getElementById("wbTicker"); if (input) input.value = b.dataset.optTicker; loadTicker();
  }));
  renderMarketScatter(rows);
}

function renderMarketScatter(rows) {
  const el = document.getElementById("optMarketScatter");
  if (!el || !window.Plotly || !rows.length) return;
  const compact = window.innerWidth < 650;
  const data = [{
    type: "scatter", mode: compact ? "markers" : "markers+text",
    x: rows.map((r) => r.rv_pctile), y: rows.map((r) => (r.vrp || 0) * 100),
    text: rows.map((r) => r.ticker), textposition: "top center",
    customdata: rows.map((r) => [r.group, r.iv * 100, r.pctile, r.rv21 * 100, r.setup]),
    hovertemplate: "<b>%{text}</b> · %{customdata[0]}<br>IV %{customdata[1]:.1f}% (p%{customdata[2]:.0f})<br>RV21 %{customdata[3]:.1f}% · VRP %{y:.0f}%<br>%{customdata[4]}<extra></extra>",
    marker: { size: rows.map((r) => 8 + Math.max(0, Math.min(100, r.pctile || 0)) / 12),
      color: rows.map((r) => r.pctile), colorscale: [[0, "#3ddb8f"], [.5, "#ffc14d"], [1, "#ff6b6b"]],
      cmin: 0, cmax: 100, showscale: true, colorbar: { title: "IV pct", thickness: 9 }, line: { color: "#0b0f17", width: 1 } },
  }];
  const layout = plotLayout({ height: compact ? 300 : 330, margin: { l: 50, r: compact ? 38 : 55, t: 15, b: 45 }, showlegend: false,
    xaxis: { title: "Realized-vol percentile", range: [-5, 105], gridcolor: "#202838", zeroline: false },
    yaxis: { title: "IV / RV premium", ticksuffix: "%", gridcolor: "#202838", zerolinecolor: "#6f7888" },
    shapes: [{ type: "line", x0: 50, x1: 50, y0: 0, y1: 1, yref: "paper", line: { color: "#303a4c", dash: "dot" } },
             { type: "line", x0: 0, x1: 1, xref: "paper", y0: 0, y1: 0, line: { color: "#6f7888", dash: "dot" } }],
  });
  Plotly.newPlot(el, data, layout, PLOT_CFG);
}

function thesisDefaults() {
  const wb = state.wb;
  if (!wb) return { target: null, adverse: null };
  if (hasSignal()) return { target: state.params.target, adverse: state.params.stop };
  const spot = wb.spot;
  const iv = chainAtmIv() || ((state.ivCtx && state.ivCtx[wb.ticker] || {}).iv) || 0.30;
  const em = spot * iv * Math.sqrt(Math.max(1, tradeHorizon()) / 252);
  const view = currentView();
  if (view === "bearish" || view === "hedge") return { target: spot - em, adverse: spot + em * 0.5 };
  if (view === "range") return { target: spot, adverse: spot + em };
  return { target: spot + em, adverse: view === "big_move" ? spot : spot - em * 0.5 };
}

function scenarioSpots() {
  const d = thesisDefaults();
  return {
    entry: hasSignal() ? state.params.entry : (state.wb && state.wb.spot),
    target: hasSignal() ? state.params.target : (state.manual.target || d.target),
    adverse: hasSignal() ? state.params.stop : (state.manual.adverse || d.adverse),
  };
}

function renderThesis() {
  const el = document.getElementById("thesis");
  if (!el || !state.wb || hasSignal()) { if (el) el.innerHTML = ""; return; }
  const d = thesisDefaults();
  if (state.manual.target == null) state.manual.target = round2(d.target);
  if (state.manual.adverse == null) state.manual.adverse = round2(d.adverse);
  el.innerHTML = `<div class="card" style="margin-bottom:12px">
    <div style="font:700 14px inherit;margin-bottom:7px">Manual thesis
      <span class="cap" style="display:inline;font-weight:400">· used for horizon P&amp;L, ranking, and sizing—not an order condition</span></div>
    <div class="opt-thesis">
      <label><span>Underlying now</span><input value="${fmt.num(state.wb.spot, 2)}" disabled></label>
      <label><span>Thesis price</span><input id="th_target" type="number" min="0.01" step="0.01" value="${fmt.num(state.manual.target, 2)}"></label>
      <label><span>Adverse price</span><input id="th_adverse" type="number" min="0.01" step="0.01" value="${fmt.num(state.manual.adverse, 2)}"></label>
      <div class="cap">Defaults use the selected chain&rsquo;s 1σ move over ${tradeHorizon()} trading days. Override them to match your actual thesis.</div>
    </div></div>`;
  ["th_target", "th_adverse"].forEach((id) => document.getElementById(id).addEventListener("change", () => {
    state.manual.target = Number(document.getElementById("th_target").value) || d.target;
    state.manual.adverse = Number(document.getElementById("th_adverse").value) || d.adverse;
    buildStructures(); renderForecastLab(); renderShootout(); renderScenarioLab(); renderSizing(); renderTicket();
  }));
}

function constantMaturityIv(expiries, targetDte) {
  const exps = (expiries || []).filter((e) => e.atm_iv > 0 && e.dte > 0).slice().sort((a, b) => a.dte - b.dte);
  if (!exps.length) return null;
  const exact = exps.find((e) => e.dte === targetDte);
  if (exact) return exact.atm_iv;
  const lo = exps.filter((e) => e.dte < targetDte).pop();
  const hi = exps.find((e) => e.dte > targetDte);
  if (!lo || !hi) return null; // do not disguise extrapolation as constant maturity
  const totalLo = lo.atm_iv ** 2 * lo.dte;
  const totalHi = hi.atm_iv ** 2 * hi.dte;
  const w = (targetDte - lo.dte) / (hi.dte - lo.dte);
  const total = totalLo + w * (totalHi - totalLo);
  return total > 0 ? Math.sqrt(total / targetDte) : null;
}

function forwardVol(iv1, dte1, iv2, dte2) {
  if (!(iv1 > 0 && iv2 > 0 && dte2 > dte1)) return null;
  const variance = (iv2 ** 2 * dte2 - iv1 ** 2 * dte1) / (dte2 - dte1);
  return variance > 0 ? Math.sqrt(variance) : null;
}

function surfaceSkewMetrics(wb) {
  const rows = (((wb || {}).chain || {}).strikes || []);
  const puts = rows.filter((r) => r.right === "P"), calls = rows.filter((r) => r.right === "C");
  const p25 = byDelta(puts, 0.25), c25 = byDelta(calls, 0.25);
  const atm = chainAtmIv();
  const p = p25 && p25.iv, c = c25 && c25.iv;
  return {
    put25: p || null, call25: c || null,
    putNormPct: p && atm ? (p / atm - 1) * 100 : null,
    callNormPct: c && atm ? (c / atm - 1) * 100 : null,
    rr25Pts: p && c ? (p - c) * 100 : ((((wb || {}).chain || {}).rr25 || 0) * 100 || null),
  };
}

function selectedTermMetrics(wb) {
  const exps = (wb && wb.expiries || []).filter((e) => e.atm_iv != null).slice().sort((a, b) => a.dte - b.dte);
  const iv10 = constantMaturityIv(exps, 10), iv30 = constantMaturityIv(exps, 30);
  const iv60 = constantMaturityIv(exps, 60), iv90 = constantMaturityIv(exps, 90);
  const iv180 = constantMaturityIv(exps, 180);
  const ratio = iv30 && iv90 ? iv30 / iv90 : null;
  const slopePts = iv30 && iv90 ? (iv90 - iv30) * 100 : null;
  const fwd30_90 = forwardVol(iv30, 30, iv90, 90);
  const shape = ratio == null ? "unknown" : ratio > 1.05 ? "backwardation" : ratio < 0.95 ? "contango" : "flat";
  return { iv10, iv30, iv60, iv90, iv180, ratio, slopePts, fwd30_90, shape, expiries: exps };
}

function assessVolRegime(metrics) {
  const scores = [];
  if (metrics.pctile != null) scores.push(Number(metrics.pctile));
  if (metrics.vrp != null) scores.push(Math.max(0, Math.min(100, 50 + 100 * Number(metrics.vrp))));
  const score = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null;
  const label = score == null ? "UNRANKED" : score < 35 ? "CHEAP" : score > 65 ? "RICH" : "FAIR";
  const tone = label === "CHEAP" ? "#3ddb8f" : label === "RICH" ? "#ff6b6b" : label === "FAIR" ? "#ffc14d" : "#9aa3b2";
  return { score, label, tone, confidence: scores.length === 2 ? "high" : scores.length ? "medium" : "low" };
}

function shapeGuidance(view, metrics) {
  const regime = assessVolRegime(metrics).label;
  const richPutSkew = metrics.putSkewPct != null ? metrics.putSkewPct > 12 : metrics.rr25Pts != null && metrics.rr25Pts > 3;
  const frontRich = metrics.termShape === "backwardation";
  if (view === "big_move") {
    if (regime === "CHEAP" && !frontRich) return { shape: "Long straddle or strangle", why: "Convexity is inexpensive and the front end is not carrying a stress premium.", avoid: "Avoid selling the move simply because the straddle looks large in dollars." };
    return { shape: "Calendar or wait", why: "Outright gamma is not cheap; isolate a tenor dislocation instead of paying the whole surface.", avoid: "Avoid naked long premium without a move estimate above the market&rsquo;s." };
  }
  if (view === "range") {
    if (regime === "RICH") return { shape: "Defined-risk credit spread / iron condor", why: "The market is paying a premium to own movement; keep tails capped and sell only liquid wings.", avoid: "Avoid undefined-risk short options." };
    return { shape: "Wait or use stock", why: "Cheap volatility is poor inventory to sell for a range thesis.", avoid: "Avoid forcing a short-vol trade when the premium is not there." };
  }
  if (view === "hedge" || view === "bearish") {
    if (regime === "CHEAP" && !richPutSkew) return { shape: "Long put", why: "Both outright vol and the downside wing are reasonably priced, so keep the convexity.", avoid: "Avoid capping the hedge too early unless the budget requires it." };
    if (regime === "RICH" || richPutSkew) return { shape: "Put spread or bear call spread", why: "Sell an expensive wing to subsidize delta; use the credit spread only when the directional thesis can tolerate assignment risk.", avoid: "Avoid a naked put whose vega bill can swamp a correct direction call." };
    return { shape: "Put spread", why: "Vol is middling; a defined payout keeps the trade tied to the price target.", avoid: "Avoid paying for far-tail convexity you do not need." };
  }
  if (regime === "CHEAP") return { shape: "Long call or wide call spread", why: "Outright delta and convexity are inexpensive; keep more upside if the thesis allows it.", avoid: "Avoid selling a rich-looking strike without checking the full surface." };
  if (regime === "RICH" || frontRich) return { shape: "Call spread or bull put spread", why: "Sell expensive vol to fund the directional exposure and define the loss.", avoid: "Avoid a naked call whose theta/vega drag can erase a correct stock view." };
  return { shape: "Target-anchored call spread", why: "Fair vol makes the stock target—not the vol level—the cleanest short-strike anchor.", avoid: "Avoid extra tenor or width that the thesis does not use." };
}

function volMetrics() {
  const wb = state.wb;
  const rec = wb && state.ivCtx && state.ivCtx[wb.ticker];
  const term = selectedTermMetrics(wb);
  const skew = surfaceSkewMetrics(wb);
  const iv = term.iv30 || chainAtmIv() || (rec && rec.iv);
  const rv21 = rec && rec.rv21;
  const recMs = rec && rec.last ? new Date(rec.last + "T00:00:00").getTime() : NaN;
  return {
    iv, rv21, pctile: rec && rec.pctile, rvPctile: rec && rec.rv21_pctile,
    vrp: iv != null && rv21 ? iv / rv21 - 1 : null,
    rr25Pts: skew.rr25Pts, putSkewPct: skew.putNormPct, callSkewPct: skew.callNormPct,
    termSlopePts: term.slopePts, termShape: term.shape, term,
    historyAgeDays: isFinite(recMs) ? Math.floor((Date.now() - recMs) / 86400000) : null,
  };
}

function renderVolDashboard() {
  const el = document.getElementById("volDashboard");
  if (!el || !state.wb) { if (el) el.innerHTML = ""; return; }
  const m = volMetrics();
  const reg = assessVolRegime(m);
  const guide = shapeGuidance(currentView(), m);
  const stale = m.historyAgeDays != null && m.historyAgeDays > 5;
  const pctTxt = m.pctile != null ? `p${fmt.num(m.pctile, 0)} vs its 1y range` : "history unavailable";
  const vrpTxt = m.vrp != null ? `${fmt.pct(m.vrp, 0)} vs RV21` : "RV comparison unavailable";
  const termTxt = m.term.ratio != null ? `30d / 90d ${fmt.num(m.term.ratio, 2)}${m.term.fwd30_90 != null ? ` · 30→90 fwd ${fmt.pctRaw(m.term.fwd30_90 * 100, 1)}` : ""}` : "need expiries around 30d and 90d";
  const putSkewTxt = m.putSkewPct != null ? `${fmt.signed(m.putSkewPct, 1)}% vs ATM IV` : "not quoted";
  const callSkewTxt = m.callSkewPct != null ? `${fmt.signed(m.callSkewPct, 1)}% vs ATM IV` : "not quoted";
  el.innerHTML = `<div class="card" style="margin-bottom:12px">
    <div style="display:flex;justify-content:space-between;gap:12px;align-items:start;flex-wrap:wrap">
      <div><div class="cap">Volatility compass</div><div class="opt-hero-value" style="color:${reg.tone}">${reg.label} VOL</div></div>
      <div class="cap" style="max-width:560px;text-align:right">The funnel is level → realized → premium → curve → skew. Cheap/rich starts the investigation; it does not finish it.</div>
    </div>
    ${stale ? `<div class="opt-risk-warn">IV rank and IV/RV history are ${m.historyAgeDays} days old. The live curve and skew are current, but do not treat the cheap/rich label as current until the recorder catches up.</div>` : ""}
    <div class="opt-compass six">
      <div class="tile"><div class="eyebrow">30d implied</div><div class="reading">${m.iv != null ? fmt.pctRaw(m.iv * 100, 1) : "—"}</div><div class="detail">${pctTxt}</div></div>
      <div class="tile"><div class="eyebrow">21d realized</div><div class="reading">${m.rv21 != null ? fmt.pctRaw(m.rv21 * 100, 1) : "—"}</div><div class="detail">${m.rvPctile != null ? `p${fmt.num(m.rvPctile, 0)} vs ~3y history` : "history unavailable"}</div></div>
      <div class="tile"><div class="eyebrow">Vol risk premium</div><div class="reading">${m.vrp != null ? fmt.pct(m.vrp, 0) : "—"}</div><div class="detail">${vrpTxt}; IV30 / RV21 − 1</div></div>
      <div class="tile"><div class="eyebrow">Term shape</div><div class="reading">${esc(String(m.termShape || "unknown").toUpperCase())}</div><div class="detail">${termTxt}</div></div>
      <div class="tile"><div class="eyebrow">25Δ put skew</div><div class="reading">${m.putSkewPct != null ? fmt.signed(m.putSkewPct, 1) + "%" : "—"}</div><div class="detail">${putSkewTxt}; normalized to ATM</div></div>
      <div class="tile"><div class="eyebrow">25Δ call skew</div><div class="reading">${m.callSkewPct != null ? fmt.signed(m.callSkewPct, 1) + "%" : "—"}</div><div class="detail">${callSkewTxt}; risk reversal ${m.rr25Pts != null ? fmt.signed(m.rr25Pts, 1) + " pts" : "—"}</div></div>
    </div>
    <div class="opt-playbook">
      <div class="callout"><div class="cap">Best-fit shape for ${esc(currentView().replace("_", " "))}</div>
        <div class="shape">${guide.shape}</div><div>${guide.why}</div></div>
      <div><div class="cap">What this means</div><div style="margin-top:3px">${guide.avoid}</div>
        <div class="cap" style="margin-top:8px">Confidence: ${reg.confidence}. Term/skew readings are descriptive until enough self-recorded history exists for their own percentiles.</div></div>
    </div></div>`;
}

/* ---------------- workbench query ---------------- */
async function loadTicker(expiry) {
  const ticker = (document.getElementById("wbTicker").value || "").toUpperCase().trim();
  if (!ticker) return;
  if (!expiry && (!state.wb || state.wb.ticker !== ticker)) {
    state.manual.target = null;
    state.manual.adverse = null;
  }
  const msg = document.getElementById("wbMsg");
  msg.textContent = expiry ? "re-quoting expiry…" : "fetching chain + term structure… (~15-20s)";
  clearTimeout(state.wbTimer);
  const p = state.params;
  const body = {
    ticker,
    mode: expiry ? "chain" : "full",
    expiry: expiry || null,
    // Enough tenors to bracket 30d/60d/90d on weekly-heavy ETF chains.
    max_expiries: state.forecast.active ? 32 : 20,
    context: state.forecast.active ? {
      direction: state.forecast.target < ((state.wb || {}).spot || Infinity) ? "Short" : "Long",
      target: state.forecast.target,
      hold_days: bdaysUntil(state.forecast.cutoff),
      time_exit_date: state.forecast.cutoff,
      forecast_event: state.forecast.event,
      forecast_probability: state.forecast.probability,
    } : hasSignal() && ticker === p.ticker ? {
      direction: p.dir, entry: p.entry, stop: p.stop, target: p.target,
      atr: p.atr, hold_days: p.hold, time_exit_date: p.texit,
    } : null,
  };
  try {
    const r = await fetch("/exec-workbench", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const d = await r.json();
    if (!d.ok) { msg.textContent = "error: " + (d.error || ("HTTP " + r.status)); return; }
    state.wbId = d.id;
    pollWb(0, expiry);
  } catch (e) { msg.textContent = "error: " + e; }
}

async function pollWb(n, expiryOnly) {
  if (n > 45) { document.getElementById("wbMsg").textContent = "timed out — is the agent online?"; return; }
  const d = (await fetchJSONOrNull(`/exec-workbench?id=${encodeURIComponent(state.wbId)}`)) || {};
  const q = d.query;
  if (q && q.id === state.wbId && q.result) {
    document.getElementById("wbMsg").textContent = "";
    if (q.result.error) {
      document.getElementById("shootout").innerHTML =
        `<div class="card"><span class="neg">${esc(q.result.error)}</span></div>`;
      return;
    }
    if (expiryOnly && state.wb) {           // chain-only re-quote keeps the term pass
      state.wb.chain = q.result.chain;
      state.wb.spot = q.result.spot;
      state.wb.asof = q.result.asof;
      state.wb.market_data_type = q.result.market_data_type;
    } else {
      state.wb = q.result;
    }
    renderAll();
    return;
  }
  state.wbTimer = setTimeout(() => pollWb(n + 1, expiryOnly), 2000);
}

function renderAll() {
  renderThesis();
  renderIvStrip();
  renderVolDashboard();
  renderTermLab();
  renderExpiries();
  renderComparator();
  buildStructures();
  renderForecastLab();
  renderShootout();
  renderScenarioLab();
  renderSizing();
  renderTicket();
}

/* ---------------- IV context strip ---------------- */
function ivRegime(pctile) {
  if (pctile == null) return ["#9aa3b2", "no history"];
  if (pctile < 30) return ["#3ddb8f", "premium CHEAP — debit structures"];
  if (pctile < 50) return ["#9aa3b2", "middling — tiebreak on cost of delta"];
  if (pctile < 80) return ["#ffc14d", "elevated — spreads over singles"];
  return ["#ff6b6b", "RICH — sell premium to buy delta (credit structures)"];
}
function renderIvStrip() {
  const el = document.getElementById("ivStrip");
  const wb = state.wb;
  const rec = state.ivCtx && state.ivCtx[wb.ticker];
  if (!rec) {
    el.innerHTML = `<div class="card" style="margin-bottom:12px"><b>${esc(wb.ticker)}</b>
      <span class="badge warn" style="margin-left:8px">NO IV HISTORY</span>
      <span class="cap" style="display:inline;margin-left:8px">the local IV cache hasn't covered this name —
      rank/percentile unavailable; judge richness off IV vs realized manually.</span></div>`;
    return;
  }
  const [tone, label] = ivRegime(rec.pctile);
  const recMs = rec.last ? new Date(rec.last + "T00:00:00").getTime() : NaN;
  const ageDays = isFinite(recMs) ? Math.floor((Date.now() - recMs) / 86400000) : null;
  const stale = ageDays != null && ageDays > 5;
  const rvTiles = [10, 21, 63].map((n) => {
    const rv = rec[`rv${n}`];
    if (rv == null) return "";
    const sp = (rec.iv - rv) * 100;
    const cls = sp > 4 ? "neg" : sp < 0 ? "pos" : "";
    return `<div class="k">IV − RV${n}</div><div class="v ${cls}">${fmt.signed(sp, 1)} pts</div>`;
  }).join("");
  el.innerHTML = `<div class="card" style="margin-bottom:12px">
    <div style="display:flex;gap:14px;align-items:center;flex-wrap:wrap">
      <span style="font:700 15px inherit">IV30 ${fmt.pctRaw(rec.iv * 100, 1)}</span>
      <span>rank <b>${rec.rank != null ? fmt.num(rec.rank, 0) : "—"}</b> · pctile <b>${fmt.num(rec.pctile, 0)}</b></span>
      <span style="color:${tone};font-weight:700">${label}</span>
      ${stale ? `<span class="badge warn">STALE ${ageDays}D</span>` : ""}
      <span class="cap" style="display:inline;margin-left:auto">as of ${esc(rec.last)}</span>
    </div>
    <div style="display:flex;gap:20px;align-items:center;margin-top:8px;flex-wrap:wrap">
      <div id="ivSpark" style="width:280px;height:52px"></div>
      <div class="kv" style="flex:1;min-width:220px">${rvTiles}</div>
    </div></div>`;
  if (rec.spark && rec.spark.length > 2 && window.Plotly) {
    Plotly.newPlot("ivSpark", [{ y: rec.spark, mode: "lines", line: { color: "#4da3ff", width: 1.5 } }],
      plotLayout({ margin: { l: 2, r: 2, t: 2, b: 2 }, xaxis: { visible: false }, yaxis: { visible: false },
                   height: 52, showlegend: false, hovermode: false }), PLOT_CFG);
  }
}

/* ---------------- term structure + realized-vol cone ---------------- */
function representativeExpiries(expiries) {
  const exps = (expiries || []).filter((e) => e.atm_iv > 0 && e.dte > 0).slice().sort((a, b) => a.dte - b.dte);
  const picked = [];
  for (const target of [7, 14, 30, 60, 90, 180]) {
    if (!exps.length) break;
    const hit = exps.reduce((best, e) => Math.abs(e.dte - target) < Math.abs(best.dte - target) ? e : best, exps[0]);
    if (!picked.includes(hit)) picked.push(hit);
  }
  return picked.sort((a, b) => a.dte - b.dte);
}

function forwardMatrixHtml(expiries) {
  const exps = representativeExpiries(expiries);
  if (exps.length < 2) return '<span class="cap">Not enough quoted expiries.</span>';
  const head = exps.map((e) => `<th>${e.dte}d</th>`).join("");
  const rows = exps.map((from, i) => `<tr><th>${from.dte}d</th>${exps.map((to, j) => {
    if (j <= i) return `<td class="muted">${j === i ? "—" : ""}</td>`;
    const fwd = forwardVol(from.atm_iv, from.dte, to.atm_iv, to.dte);
    return `<td>${fwd ? fmt.pctRaw(fwd * 100, 1) : "—"}</td>`;
  }).join("")}</tr>`).join("");
  return `<div class="table-wrap"><table class="opt-forward-table"><thead><tr><th>from → to</th>${head}</tr></thead><tbody>${rows}</tbody></table></div>`;
}

function renderTermLab() {
  const el = document.getElementById("termLab"), wb = state.wb;
  if (!el || !wb) { if (el) el.innerHTML = ""; return; }
  const exps = (wb.expiries || []).filter((e) => e.atm_iv > 0 && e.dte > 0).slice().sort((a, b) => a.dte - b.dte);
  if (exps.length < 2) { el.innerHTML = ""; return; }
  const term = selectedTermMetrics(wb);
  const rec = state.ivCtx && state.ivCtx[wb.ticker];
  const cmRows = [[10, term.iv10], [30, term.iv30], [60, term.iv60], [90, term.iv90], [180, term.iv180]]
    .filter((x) => x[1] != null).map((x) => `<span><b>${x[0]}d</b> ${fmt.pctRaw(x[1] * 100, 1)}</span>`).join("");
  el.innerHTML = `<div class="card opt-term-lab">
    <div class="opt-section-head"><div><b>Curve, forwards & realized cone</b><div class="cap">Live chain snapshot · total-variance interpolation · no extrapolation</div></div>
      <div class="opt-cm-strip">${cmRows}</div></div>
    <div class="opt-term-grid"><div><div id="optTermCurve" class="opt-term-chart"></div></div><div><div id="optVolCone" class="opt-term-chart"></div></div></div>
    <details class="opt-forward-details"><summary><b>Forward-vol matrix</b> <span class="cap">· volatility implied only between each pair of expiries</span></summary>${forwardMatrixHtml(exps)}</details>
  </div>`;
  if (!window.Plotly) return;
  const sel = wb.chain && wb.chain.expiry;
  Plotly.newPlot("optTermCurve", [{
    x: exps.map((e) => e.dte), y: exps.map((e) => e.atm_iv * 100), type: "scatter", mode: "lines+markers",
    text: exps.map((e) => e.date), hovertemplate: "%{text}<br>%{x} DTE · %{y:.1f}% IV<extra></extra>",
    marker: { size: exps.map((e) => e.date === sel ? 10 : 6), color: exps.map((e) => e.date === sel ? "#ffc14d" : "#4da3ff") },
    line: { color: "#4da3ff", width: 2 },
  }], plotLayout({ height: 285, margin: { l: 48, r: 15, t: 25, b: 42 }, title: { text: "ATM implied-volatility term structure", font: { size: 12 } },
    xaxis: { title: "Days to expiry", gridcolor: "#202838" }, yaxis: { title: "IV", ticksuffix: "%", gridcolor: "#202838" }, showlegend: false }), PLOT_CFG);

  const cone = rec && rec.cone || {};
  const horizons = Object.keys(cone).map(Number).sort((a, b) => a - b);
  if (horizons.length) {
    const ys = (key) => horizons.map((h) => (cone[String(h)][key] || 0) * 100);
    const cm = horizons.map((h) => constantMaturityIv(exps, h));
    const traces = [
      { x: horizons, y: ys("p10"), mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
      { x: horizons, y: ys("p90"), mode: "lines", fill: "tonexty", fillcolor: "rgba(77,163,255,.10)", line: { width: 0 }, name: "P10–P90" },
      { x: horizons, y: ys("p25"), mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
      { x: horizons, y: ys("p75"), mode: "lines", fill: "tonexty", fillcolor: "rgba(77,163,255,.22)", line: { width: 0 }, name: "P25–P75" },
      { x: horizons, y: ys("p50"), mode: "lines+markers", line: { color: "#9aa3b2", dash: "dot" }, name: "Median RV" },
      { x: horizons, y: ys("current"), mode: "lines+markers", line: { color: "#3ddb8f" }, name: "Current RV" },
    ];
    if (cm.some((v) => v != null)) traces.push({ x: horizons, y: cm.map((v) => v == null ? null : v * 100), mode: "lines+markers", line: { color: "#ffc14d", width: 2 }, name: "Live IV" });
    Plotly.newPlot("optVolCone", traces, plotLayout({ height: 285, margin: { l: 48, r: 15, t: 25, b: 42 }, title: { text: "IV over the realized-vol cone", font: { size: 12 } },
      xaxis: { title: "Horizon (trading days)", gridcolor: "#202838" }, yaxis: { title: "Annualized vol", ticksuffix: "%", gridcolor: "#202838" },
      legend: { orientation: "h", y: -0.28, font: { size: 9 } } }), PLOT_CFG);
  } else {
    document.getElementById("optVolCone").innerHTML = '<div class="cap" style="padding:30px">Realized-vol cone unavailable for this ticker.</div>';
  }
}

/* ---------------- expiry picker ---------------- */
function bdaysUntil(iso) {
  if (!iso) return null;
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const end = new Date(iso + "T00:00:00");
  if (isNaN(end) || end <= today) return 0;
  let n = 0; const d = new Date(today);
  while (d < end) { d.setDate(d.getDate() + 1); if (d.getDay() > 0 && d.getDay() < 6) n++; }
  return n;
}
function expLabel(e) { return `${e.slice(4, 6)}/${e.slice(6, 8)}`; }

function renderExpiries() {
  const el = document.getElementById("expiryRow");
  const wb = state.wb;
  const exps = wb.expiries || [];
  if (!exps.length) { el.innerHTML = ""; return; }
  const texitBd = bdaysUntil(state.params.texit);
  const earnDate = state.earn && state.earn[wb.ticker];
  const noEarnData = state.earn && (state.earn._no_data || []).includes(wb.ticker);
  const cells = exps.map((e, i) => {
    const sel = wb.chain && wb.chain.expiry === e.date;
    const dteAtExit = texitBd != null ? Math.round(e.dte * (252 / 365)) - texitBd : null;
    const earnFlag = earnDate && earnDate.replace(/-/g, "") <= e.date ? " &#9888;" : "";
    // forward vol vs the prior expiry (variance additivity)
    let fwd = "";
    if (i > 0 && e.atm_iv && exps[i - 1].atm_iv && e.dte > exps[i - 1].dte) {
      const v = (e.atm_iv ** 2 * e.dte - exps[i - 1].atm_iv ** 2 * exps[i - 1].dte) / (e.dte - exps[i - 1].dte);
      if (v > 0) fwd = `fwd ${(Math.sqrt(v) * 100).toFixed(0)}`;
    }
    const warn = dteAtExit != null && dteAtExit < 7 ? ' style="color:#ffc14d"' : "";
    return `<button class="btn ${sel ? "" : "ghost"}" data-exp="${e.date}" style="flex-direction:column;line-height:1.3">
      <span>${expLabel(e.date)} <span class="cap" style="display:inline">(${e.dte}d)</span>${earnFlag}</span>
      <span class="cap" style="display:inline">${e.atm_iv ? "IV " + (e.atm_iv * 100).toFixed(0) : "IV —"}${fwd ? " · " + fwd : ""}</span>
      ${dteAtExit != null ? `<span class="cap"${warn} style="display:inline">${dteAtExit}d left at exit</span>` : ""}
    </button>`;
  }).join("");
  const earnNote = noEarnData
    ? `<span class="badge warn">NO EARNINGS DATA</span> <span class="cap" style="display:inline">this ticker has no
       earnings coverage — long premium may straddle an unflagged report; verify the date yourself.</span>`
    : earnDate ? `<span class="cap" style="display:inline">next earnings <b>${esc(earnDate)}</b>${
        state.params.texit && earnDate <= state.params.texit ? ' <b style="color:#ff6b6b">INSIDE THE HOLD</b>' : ""}</span>` : "";
  el.innerHTML = `<div class="card" style="margin-bottom:12px">
    <div style="font:700 14px inherit;margin-bottom:6px">Expiry
      <span class="cap" style="display:inline;font-weight:400">· default = first listed &ge; time-exit + 5 td · &#9888; = earnings before expiry</span></div>
    <div style="display:flex;gap:8px;flex-wrap:wrap">${cells}</div>
    <div style="margin-top:6px">${earnNote}</div></div>`;
  el.querySelectorAll("[data-exp]").forEach((b) =>
    b.addEventListener("click", () => loadTicker(b.dataset.exp)));
}

/* ---------------- edge-vs-priced comparator ---------------- */
function holdCalDays() {
  const h = tradeHorizon();
  return h ? Math.round(h * 365 / 252) : null;
}
function defaultForecastDate() {
  const now = new Date();
  let year = now.getFullYear();
  let d = new Date(year, 9, 30, 12, 0, 0);
  if (d <= now) d = new Date(++year, 9, 30, 12, 0, 0);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}
function calDaysUntil(iso) {
  if (!iso) return null;
  const now = new Date(); now.setHours(0, 0, 0, 0);
  const end = new Date(iso + "T00:00:00");
  if (isNaN(end)) return null;
  return Math.max(0, Math.round((end - now) / 86400000));
}
function chainAtmIv() {
  const wb = state.wb;
  const e = (wb.expiries || []).find((x) => wb.chain && x.date === wb.chain.expiry);
  if (e && e.atm_iv) return e.atm_iv;
  // chain-only mode: median IV of the 5 strikes nearest spot
  const rows = ((wb.chain || {}).strikes || []).filter((r) => r.iv != null);
  if (!rows.length) return null;
  rows.sort((a, b) => Math.abs(a.strike - wb.spot) - Math.abs(b.strike - wb.spot));
  const ivs = rows.slice(0, 6).map((r) => r.iv).sort((a, b) => a - b);
  return ivs[Math.floor(ivs.length / 2)];
}
function renderComparator() {
  const el = document.getElementById("emCompare");
  const wb = state.wb, p = state.params;
  const stats = p.strategy && state.stats && state.stats[p.strategy];
  const iv = chainAtmIv();
  const cal = holdCalDays();
  if (!hasSignal() || !stats || !iv || !cal) {
    el.innerHTML = !hasSignal() ? "" : `<div class="card" style="margin-bottom:12px"><span class="cap">
      Edge comparator unavailable (${!stats ? "no strategy stats for " + esc(p.strategy || "?") : "no ATM IV"}).</span></div>`;
    return;
  }
  const implied = iv * Math.sqrt(cal / 365);                       // 1-sigma over the hold
  const tm = stats.terminal_move || {};
  const fc = Math.abs(tm.median_pct != null ? tm.median_pct : tm.mean_pct || 0);
  const ratio = implied > 0 ? fc / implied : null;
  const [tone, verdict] = ratio == null ? ["#9aa3b2", "—"]
    : ratio > 1.2 ? ["#3ddb8f", "your edge exceeds what's priced — long premium is cheap"]
    : ratio >= 0.8 ? ["#ffc14d", "roughly fair — decide on cost of delta + IV regime"]
    : ["#ff6b6b", "market prices a bigger move than your signal delivers — prefer credit structures or stock"];
  const straddle = (wb.expiries || []).find((x) => wb.chain && x.date === wb.chain.expiry);
  const w = (v) => Math.min(100, v / Math.max(implied, fc) * 100);
  el.innerHTML = `<div class="card" style="margin-bottom:12px">
    <div style="font:700 14px inherit;margin-bottom:6px">Edge vs priced
      <span class="cap" style="display:inline;font-weight:400">· 1&sigma; = IV&middot;&radic;(days/365), scaled to the ${p.hold} td hold</span></div>
    <div class="kv">
      <div class="k">Implied 1&sigma; over hold</div>
      <div class="v"><span style="display:inline-block;background:#4da3ff33;border-left:3px solid #4da3ff;padding:1px 6px;width:${w(implied)}%">${fmt.pctRaw(implied * 100, 2)}</span></div>
      <div class="k">${esc(p.strategy)} median terminal move</div>
      <div class="v"><span style="display:inline-block;background:#00d18f33;border-left:3px solid #00d18f;padding:1px 6px;width:${w(fc)}%">${fmt.pctRaw(fc * 100, 2)}</span>
        <span class="cap" style="display:inline">(mean ${fmt.pctRaw(Math.abs(tm.mean_pct || 0) * 100, 2)}, n=${stats.n}, win ${fmt.pct(stats.win_rate, 0)})</span></div>
      <div class="k">Edge ratio</div>
      <div class="v"><b style="color:${tone}">${ratio != null ? fmt.num(ratio, 2) : "—"}</b> — ${verdict}</div>
      ${straddle && straddle.straddle_mid ? `<div class="k">ATM straddle (market price of the move)</div>
        <div class="v">${fmt.num(straddle.straddle_mid, 2)} <span class="cap" style="display:inline">separate quantity — never blended into the ratio</span></div>` : ""}
    </div></div>`;
}

/* ---------------- structure assembly ---------------- */
function chainRows(right) {
  return (((state.wb || {}).chain || {}).strikes || [])
    .filter((r) => r.right === right)
    .sort((a, b) => a.strike - b.strike);
}
function byDelta(rows, absD) {
  let best = null, bd = 1e9;
  for (const r of rows) {
    if (r.delta == null) continue;
    const d = Math.abs(Math.abs(r.delta) - absD);
    if (d < bd) { bd = d; best = r; }
  }
  return best;
}
function byStrike(rows, px) {
  let best = null, bd = 1e9;
  for (const r of rows) {
    const d = Math.abs(r.strike - px);
    if (d < bd) { bd = d; best = r; }
  }
  return best;
}
function legPrice(row, side, kind) {   // kind: "mid" | "nat"
  if (kind === "mid") return row.mid;
  return side === "BUY" ? row.ask : row.bid;   // natural: pay the ask, hit the bid
}
function structureFrom(name, legs, note, meta = {}) {
  if (!legs || !legs.length || legs.length > 4 || legs.some((l) => !l.row)) return null;
  let signedMid = 0, signedNat = 0, delta = 0, gamma = 0, theta = 0, vega = 0, ok = true;
  for (const l of legs) {
    const m = legPrice(l.row, l.side, "mid"), n = legPrice(l.row, l.side, "nat");
    if (m == null) ok = false;
    const s = l.side === "BUY" ? 1 : -1;
    signedMid += s * (m || 0);
    signedNat += s * (n != null ? n : (m || 0));
    if (l.row.delta != null) delta += s * l.row.delta;
    if (l.row.gamma != null) gamma += s * l.row.gamma;
    if (l.row.theta != null) theta += s * l.row.theta;
    if (l.row.vega != null) vega += s * l.row.vega;
  }
  const credit = !!meta.credit;
  const mid = credit ? -signedMid : signedMid;
  const nat = credit ? -signedNat : signedNat;
  if (!ok || mid <= 0) return null;
  return { name, legs, mid: round2(mid), nat: round2(nat), delta: round2(delta, 3),
           gamma: round2(gamma, 4), theta: round2(theta, 3), vega: round2(vega, 3),
           width: meta.width == null ? null : meta.width, credit, category: meta.category || "directional",
           note: note || "", tradeable: legs.every((l) => l.row.con_id) };
}
function spreadFrom(name, longRow, shortRow, right, note) {
  if (!longRow || (shortRow && longRow.strike === shortRow.strike)) return null;
  const legs = [{ side: "BUY", row: longRow }];
  if (shortRow) legs.push({ side: "SELL", row: shortRow });
  return structureFrom(name, legs, note, {
    width: shortRow ? Math.abs(longRow.strike - shortRow.strike) : null,
    category: shortRow ? "debit_vertical" : "single",
  });
}
function round2(v, d = 2) { return v == null ? null : Math.round(v * 10 ** d) / 10 ** d; }

function buildStructures() {
  const wb = state.wb, p = state.params;
  state.structures = [];
  if (!wb || !wb.chain) return;
  const view = currentView();
  const short = view === "bearish" || view === "hedge";
  const right = short ? "P" : "C";
  const rows = chainRows(right);
  if (!rows.length) return;
  const spot = wb.spot;
  const scen = scenarioSpots();
  const entryPx = scen.entry || spot, tgtPx = scen.target;

  // Directional helpers: for calls the short strike sits ABOVE the long; for puts below.
  const anchorLong = hasSignal() ? byStrike(rows, entryPx) : byDelta(rows, 0.45);
  const anchorShort = tgtPx ? byStrike(rows, tgtPx) : byDelta(rows, 0.25);
  const atmLong = byStrike(rows, spot);
  const d50 = byDelta(rows, 0.50), d40 = byDelta(rows, 0.40), d25 = byDelta(rows, 0.25);

  const add = (s) => { if (s && !state.structures.some((x) => sameLegs(x, s))) state.structures.push(s); };

  const creditVertical = (creditRight, label) => {
    const cr = chainRows(creditRight), short30 = byDelta(cr, 0.30), long15 = byDelta(cr, 0.15);
    if (!short30 || !long15 || short30.strike === long15.strike) return null;
    const width = Math.abs(short30.strike - long15.strike);
    return structureFrom(label, [{ side: "SELL", row: short30 }, { side: "BUY", row: long15 }],
      "defined-risk premium sale · short ~30Δ / long ~15Δ", { credit: true, width, category: "credit_vertical" });
  };

  if (view === "big_move") {
    const callAtm = byStrike(chainRows("C"), spot), putAtm = byStrike(chainRows("P"), spot);
    add(structureFrom("Long ATM straddle", [{ side: "BUY", row: callAtm }, { side: "BUY", row: putAtm }],
      "own both tails; highest theta bill", { category: "straddle" }));
    const call25 = byDelta(chainRows("C"), 0.25), put25 = byDelta(chainRows("P"), 0.25);
    add(structureFrom("Long 25Δ strangle", [{ side: "BUY", row: call25 }, { side: "BUY", row: put25 }],
      "cheaper convexity; needs a larger move", { category: "strangle" }));
  } else if (view === "range") {
    const puts = chainRows("P"), calls = chainRows("C");
    const p30 = byDelta(puts, 0.30), p15 = byDelta(puts, 0.15);
    const c30 = byDelta(calls, 0.30), c15 = byDelta(calls, 0.15);
    if (p30 && p15 && c30 && c15) {
      const width = Math.max(Math.abs(p30.strike - p15.strike), Math.abs(c30.strike - c15.strike));
      add(structureFrom("Iron condor 30Δ/15Δ", [
        { side: "BUY", row: p15 }, { side: "SELL", row: p30 },
        { side: "SELL", row: c30 }, { side: "BUY", row: c15 },
      ], "defined tails; short the inside wings", { credit: true, width, category: "iron_condor" }));
    }
    add(creditVertical("P", "Bull put spread 30Δ/15Δ"));
    add(creditVertical("C", "Bear call spread 30Δ/15Δ"));
  } else if (view === "hedge") {
    add(spreadFrom("Hedge put spread 40Δ/25Δ", d40, d25, right, "budgeted downside protection"));
    add(spreadFrom("Long put ~40Δ", d40, null, right, "uncapped crash convexity, full theta"));
  } else {
    add(spreadFrom("Target-anchored vertical", anchorLong, anchorShort, right,
      tgtPx ? "short strike nearest the thesis target" : "long near entry / short ~25Δ"));
    add(spreadFrom("50Δ/25Δ vertical", d50 || atmLong, d25, right, "balanced delta and premium"));
    add(spreadFrom(`Long ${right === "C" ? "call" : "put"} ~40Δ`, d40, null, right,
      "uncapped convexity, full theta and vega"));
    add(creditVertical(short ? "C" : "P", short ? "Bear call spread 30Δ/15Δ" : "Bull put spread 30Δ/15Δ"));
  }
  if (!state.structures.some((s) => s.tradeable)) return;
  if (!state.selStructure || !state.structures.some((s) => s.name === state.selStructure)) {
    state.selStructure = state.structures[0].name;
  }
}
function sameLegs(a, b) {
  const key = (s) => s.legs.map((l) => `${l.side}${l.row.strike}${l.row.right}`).join("|");
  return key(a) === key(b);
}

/* ---------------- exit-date scenario pricing (BSM approximation) ---------------- */
function scenarioInputs() {
  return {
    r: numInput("sh_rate", state.pricing.rate),
    q: numInput("sh_divy", state.pricing.divYield),
    ivShift: numInput("sh_ivshift", state.pricing.ivShiftPts) / 100,
  };
}
function numInput(id, dflt) {
  const e = document.getElementById(id);
  if (!e || e.value === "") return dflt;
  const v = Number(e.value);
  return isFinite(v) ? v : dflt;
}
function valueAtExit(struct, spotAtExit, inp) {
  // sum of leg values at the planned exit date; T from each leg's expiry
  const exitBd = tradeHorizon();
  const exitCal = Math.round(exitBd * 365 / 252);
  return valueAtFuture(struct, spotAtExit, exitCal, inp);
}
function valueAtFuture(struct, spotAtExit, calendarDaysFromNow, inp) {
  // Reprice a structure on a specified future date. This is still a constant-
  // shift BSM approximation; the forecast lab makes that smile assumption explicit.
  let val = 0;
  for (const l of struct.legs) {
    // All current legs share the selected chain expiry.
    const dte = ((state.wb || {}).chain || {}).dte || 0;
    const T = Math.max(0, (dte - calendarDaysFromNow) / 365);
    const sigma = Math.max(0.01, (l.row.iv != null ? l.row.iv : chainAtmIv() || 0.3) + inp.ivShift);
    const v = BSM.price(spotAtExit, l.row.strike, T, sigma, inp.r, inp.q, l.row.right);
    val += (l.side === "BUY" ? 1 : -1) * (v || 0);
  }
  return val;
}
function scenarioPnl(struct) {
  // per-structure $ P&L at the planned horizon for thesis / adverse / flat.
  const p = state.params;
  const scen = scenarioSpots();
  if (!scen.target || !scen.adverse || !scen.entry) return null;
  const inp = scenarioInputs();
  const stats = p.strategy && state.stats && state.stats[p.strategy];
  const short = currentView() === "bearish" || currentView() === "hedge";
  const cost = struct.credit ? -struct.mid : struct.mid;
  const pnlAt = (spot) => (valueAtExit(struct, spot, inp) - cost) * 100;
  const out = {
    target: pnlAt(scen.target),
    stop: pnlAt(scen.adverse),
    flat: pnlAt(scen.entry),
  };
  if (stats && hasSignal()) {
    const lm = (stats.loser_mix || {}).avg_loser_move_pct;
    const loserSpot = lm != null ? p.entry * (1 + (short ? -1 : 1) * lm) : p.stop;
    out.ev = stats.win_rate * out.target + (1 - stats.win_rate) * pnlAt(loserSpot);
  }
  return out;
}

/* ---------------- forecast-to-trade lab ---------------- */
function forecastPnlAt(struct, spot, calendarDays, ivShiftPts, executable = true) {
  const inp = scenarioInputs();
  const opening = executable && struct.nat != null ? struct.nat : struct.mid;
  const openingValue = struct.credit ? -opening : opening;
  const value = valueAtFuture(struct, spot, calendarDays, {
    r: inp.r, q: inp.q, ivShift: ivShiftPts / 100,
  });
  return (value - openingValue) * 100 - commRT(struct);
}

function forecastScore(metrics, objective) {
  if (!metrics) return -Infinity;
  if (objective === "touch_payout") return metrics.touchPnl / Math.max(metrics.risk, 1);
  if (objective === "lowest_cost") return -metrics.risk;
  if (objective === "robust_touch") return metrics.touchWorst / Math.max(metrics.risk, 1);
  return metrics.ev / Math.max(metrics.risk, 1);
}

function forecastMetrics(struct, qty, cfg) {
  const cutoffDays = calDaysUntil(cfg.cutoff);
  if (!cutoffDays && cutoffDays !== 0) return null;
  const p = Math.max(0, Math.min(1, Number(cfg.probability)));
  const touchFractions = cfg.event === "touch" ? [0.25, 0.55, 0.85] : [1];
  const touchWeights = cfg.event === "touch" ? [0.25, 0.50, 0.25] : [1];
  const touchPnls = touchFractions.map((f) =>
    forecastPnlAt(struct, cfg.target, Math.max(1, Math.round(cutoffDays * f)), cfg.touchIvShift) * qty);
  const touchPnl = touchPnls.reduce((sum, x, i) => sum + x * touchWeights[i], 0);
  const noTouchPnl = forecastPnlAt(struct, cfg.noTouchSpot, cutoffDays, cfg.noTouchIvShift) * qty;
  const risk = riskPerUnit(struct) * qty;
  const ev = p * touchPnl + (1 - p) * noTouchPnl;
  return {
    touchPnl, touchWorst: Math.min(...touchPnls), touchBest: Math.max(...touchPnls),
    noTouchPnl, ev, risk, evRisk: risk > 0 ? ev / risk : null,
  };
}

function syncForecastInputs() {
  const f = state.forecast;
  f.event = document.getElementById("fc_event").value;
  f.target = numInput("fc_target", f.target);
  f.probability = Math.max(0, Math.min(1, numInput("fc_prob", f.probability * 100) / 100));
  f.cutoff = document.getElementById("fc_cutoff").value || f.cutoff;
  f.touchIvShift = numInput("fc_touch_iv", f.touchIvShift);
  f.noTouchSpot = numInput("fc_no_touch", f.noTouchSpot);
  f.noTouchIvShift = numInput("fc_no_touch_iv", f.noTouchIvShift);
  f.objective = document.getElementById("fc_objective").value;
}

function renderForecastLab() {
  const el = document.getElementById("forecastLab");
  const wb = state.wb;
  if (!el || !wb) { if (el) el.innerHTML = ""; return; }
  const f = state.forecast;
  if (f.target == null) f.target = round2(wb.spot * 0.95);
  if (f.noTouchSpot == null) f.noTouchSpot = wb.spot;
  const cutoffCompact = String(f.cutoff || "").replace(/-/g, "");
  const expiry = ((wb.chain || {}).expiry || "");
  const covers = !!expiry && expiry >= cutoffCompact;
  const activeRows = f.active && covers ? state.structures.map((s) => {
    const qty = contractsFor(s, sizeAlloc());
    const metrics = qty > 0 ? forecastMetrics(s, qty, f) : null;
    return { s, qty, metrics, score: forecastScore(metrics, f.objective) };
  }).filter((x) => x.metrics && isFinite(x.score)).sort((a, b) => b.score - a.score) : [];

  const objectiveLabel = {
    ev_risk: "expected P&L per dollar at risk",
    touch_payout: "payout if the forecast hits",
    robust_touch: "late/early-touch robustness",
    lowest_cost: "lowest defined risk",
  }[f.objective] || "selected objective";
  const ranking = activeRows.length ? `<div class="opt-forecast-result">
    <div class="callout"><div class="cap">Top expression under these assumptions</div>
      <div class="shape">${esc(activeRows[0].s.name)}</div>
      <div>Ranks first on ${esc(objectiveLabel)} using executable-side entry marks and round-trip commissions.</div></div>
    <div class="tblwrap"><table class="tbl"><thead><tr><th>#</th><th class="l">Structure</th><th>Lots</th>
      <th>Defined risk</th><th>Touch P&amp;L</th><th>No-touch P&amp;L</th><th>Expected P&amp;L</th><th>EV / risk</th><th>Spread tax</th><th></th></tr></thead>
      <tbody>${activeRows.map((x, i) => `<tr${i === 0 ? ' style="background:#4da3ff14"' : ""}><td>${i + 1}</td>
        <td class="l"><b>${esc(x.s.name)}</b><br><span class="cap">${x.s.legs.map((l) => `${l.side[0]} ${l.row.strike}${l.row.right}`).join(" / ")}</span></td>
        <td>${x.qty}</td><td>${fmt.money(x.metrics.risk)}</td>
        <td class="${clsSign(x.metrics.touchPnl)}">${fmt.money(x.metrics.touchPnl)}<br><span class="cap">${fmt.money(x.metrics.touchWorst)} to ${fmt.money(x.metrics.touchBest)}</span></td>
        <td class="${clsSign(x.metrics.noTouchPnl)}">${fmt.money(x.metrics.noTouchPnl)}</td>
        <td class="${clsSign(x.metrics.ev)}"><b>${fmt.money(x.metrics.ev)}</b></td>
        <td class="${clsSign(x.metrics.evRisk)}">${fmt.pct(x.metrics.evRisk, 1)}</td><td>${taxLight(spreadTax(x.s))}</td>
        <td><button class="btn xs ghost" data-fc-struct="${esc(x.s.name)}">use</button></td></tr>`).join("")}</tbody></table></div>
    <div class="cap" style="margin-top:8px">Touch P&amp;L is the weighted result of early (25%), middle (55%), and late (85%) touch dates; the smaller line shows the range. This is a scenario score, not a statistical option-pricing model.</div>
  </div>` : "";

  const coverWarning = f.active && !covers ? `<div class="opt-risk-warn">The selected ${esc(expiry || "?")} expiry ends before ${esc(f.cutoff)}. Click <b>Load forecast expiry</b> to request a chain that remains alive through the forecast window.</div>` : "";
  el.innerHTML = `<div class="card opt-forecast" style="margin-bottom:12px">
    <div style="display:flex;justify-content:space-between;gap:12px;align-items:start;flex-wrap:wrap">
      <div><div class="cap">Forecast lab</div><div style="font:700 17px inherit">Turn a probabilistic path view into a ranked structure</div></div>
      <div class="cap" style="max-width:560px;text-align:right">A touch forecast is incomplete without timing, vol-at-touch, and a no-touch outcome. Those assumptions stay visible and editable.</div>
    </div>
    <div class="opt-forecast-grid">
      <label><span>Event</span><select id="fc_event"><option value="touch"${f.event === "touch" ? " selected" : ""}>Touches level by date</option><option value="terminal"${f.event === "terminal" ? " selected" : ""}>At level on date</option></select></label>
      <label><span>Level</span><input id="fc_target" type="number" step="0.01" value="${f.target}"></label>
      <label><span>Probability</span><div class="opt-input-suffix"><input id="fc_prob" type="number" min="0" max="100" step="1" value="${round2(f.probability * 100, 0)}"><i>%</i></div></label>
      <label><span>By date</span><input id="fc_cutoff" type="date" value="${esc(f.cutoff)}"></label>
      <label><span>IV change if hit</span><div class="opt-input-suffix"><input id="fc_touch_iv" type="number" step="1" value="${f.touchIvShift}"><i>pts</i></div></label>
      <label><span>No-touch spot at date</span><input id="fc_no_touch" type="number" step="0.01" value="${f.noTouchSpot}"></label>
      <label><span>No-touch IV change</span><div class="opt-input-suffix"><input id="fc_no_touch_iv" type="number" step="1" value="${f.noTouchIvShift}"><i>pts</i></div></label>
      <label><span>Optimize for</span><select id="fc_objective">
        <option value="ev_risk"${f.objective === "ev_risk" ? " selected" : ""}>Expected P&amp;L / risk</option>
        <option value="touch_payout"${f.objective === "touch_payout" ? " selected" : ""}>Payout if hit</option>
        <option value="robust_touch"${f.objective === "robust_touch" ? " selected" : ""}>Robust across touch timing</option>
        <option value="lowest_cost"${f.objective === "lowest_cost" ? " selected" : ""}>Lowest defined risk</option></select></label>
    </div>
    <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:10px">
      <button class="btn" id="fc_apply">${covers ? "Run forecast ranking" : "Load forecast expiry"}</button>
      ${f.active ? '<span class="badge ok">FORECAST ACTIVE</span>' : '<span class="cap">Nothing changes until you run it.</span>'}
      <span class="cap">Risk budget ${fmt.money(tradeRisk())} &middot; current complex ${esc(wb.ticker)} &middot; selected expiry ${esc(expiry || "?")}</span>
    </div>
    ${coverWarning}${ranking}
  </div>`;

  document.getElementById("fc_apply").addEventListener("click", () => {
    syncForecastInputs();
    f.active = true;
    const bearish = f.target < wb.spot;
    state.manual.view = bearish ? "bearish" : "bullish";
    state.params.dir = bearish ? "Short" : "Long";
    const viewEl = document.getElementById("wbView");
    if (viewEl) viewEl.value = state.manual.view;
    state.manual.target = f.target;
    state.manual.adverse = f.noTouchSpot;
    const h = bdaysUntil(f.cutoff);
    const hEl = document.getElementById("wbHorizon");
    if (hEl && !hEl.disabled && h > 0) { hEl.value = h; state.manual.horizon = h; }
    const selectedExpiry = ((state.wb || {}).chain || {}).expiry || "";
    if (selectedExpiry < String(f.cutoff).replace(/-/g, "")) loadTicker();
    else renderAll();
  });
  el.querySelectorAll("[data-fc-struct]").forEach((b) => b.addEventListener("click", () => {
    state.selStructure = b.dataset.fcStruct;
    renderForecastLab(); renderShootout(); renderScenarioLab(); renderSizing(); renderTicket();
  }));
}

/* ---------------- shootout ---------------- */
function commRT(struct) {
  const n = struct.legs.length;
  return COMM * n * 2;               // per spread, round trip
}
function spreadTax(struct) {
  if (struct.mid == null || struct.nat == null || struct.mid <= 0) return null;
  return (struct.credit ? struct.mid - struct.nat : struct.nat - struct.mid) / struct.mid;
}
function taxLight(t) {
  if (t == null) return "—";
  const pct = t * 100;
  const c = pct <= 3 ? "#3ddb8f" : pct <= 8 ? "#ffc14d" : "#ff6b6b";
  return `<span style="color:${c};font-weight:700">&#9679;</span> ${pct.toFixed(1)}%`;
}
function breakevenPct(struct) {
  if (!struct.legs.length || struct.category === "straddle" || struct.category === "strangle" || struct.category === "iron_condor") return null;
  const anchor = struct.credit
    ? struct.legs.find((l) => l.side === "SELL")
    : struct.legs.find((l) => l.side === "BUY");
  if (!anchor) return null;
  const be = anchor.row.right === "C"
    ? anchor.row.strike + (struct.credit ? struct.mid : struct.mid)
    : anchor.row.strike - (struct.credit ? struct.mid : struct.mid);
  const spot = state.wb.spot;
  return (anchor.row.right === "C" ? be - spot : spot - be) / spot;
}

function riskPerUnit(struct) {
  if (!struct) return null;
  const premiumRisk = struct.credit ? ((struct.width || 0) - struct.mid) * 100 : struct.mid * 100;
  return premiumRisk > 0 ? premiumRisk + commRT(struct) : null;
}

function maxProfitLoss(struct) {
  if (!struct) return { maxProfit: null, maxLoss: null };
  if (struct.credit) return {
    maxProfit: struct.mid * 100 - commRT(struct),
    maxLoss: struct.width != null ? (struct.width - struct.mid) * 100 + commRT(struct) : null,
  };
  return {
    maxProfit: struct.width != null ? (struct.width - struct.mid) * 100 - commRT(struct) : null,
    maxLoss: struct.mid * 100 + commRT(struct),
  };
}

function renderShootout() {
  const el = document.getElementById("shootout");
  const wb = state.wb, p = state.params;
  if (!wb || !wb.chain) { el.innerHTML = ""; return; }
  if (!state.structures.length) {
    el.innerHTML = `<div class="card"><span class="neg">No quotable structures in the band (missing mids/greeks
      — thin chain or delayed data).</span></div>`;
    return;
  }
  const delayed = wb.market_data_type === 3 || wb.market_data_type === 4;
  const age = wb.asof ? Math.max(0, Math.round(Date.now() / 1000 - wb.asof)) : null;
  const stats = p.strategy && state.stats && state.stats[p.strategy];

  // stock baseline row (only with full signal context)
  let baselineHtml = "";
  if (hasSignal() && p.target) {
    const riskAlloc = sizeAlloc();
    const dist = Math.abs(p.entry - p.stop);
    const sh = dist > 0 ? Math.floor(riskAlloc / dist) : 0;
    const pnlT = sh * Math.abs(p.target - p.entry);
    const ev = stats ? stats.win_rate * pnlT + (1 - stats.win_rate) *
      (-(sh * dist) * Math.abs(((stats.loser_mix || {}).avg_loser_move_pct || (-dist / p.entry)) * p.entry) / dist) : null;
    baselineHtml = `<tr style="border-top:2px solid #2a3242">
      <td class="l"><b>Stock with stop</b> <span class="cap" style="display:inline">(baseline)</span></td>
      <td class="l">${sh} sh @ ${fmt.num(p.entry, 2)}</td>
      <td>—</td><td>—</td><td>${fmt.num(sh, 0)}</td><td>—</td><td>—</td><td>0%</td>
      <td class="pos">${fmt.money(pnlT)}</td>
      <td class="neg">${fmt.money(-riskAlloc)}</td>
      <td>$0</td><td>${fmt.money(pnlT)} / ${fmt.money(-riskAlloc)}</td>
      <td>${ev != null ? `<b class="${clsSign(ev)}">${fmt.money(ev)}</b>` : "—"}</td>
      <td>—</td><td></td></tr>`;
  }

  const rows = state.structures.map((s) => {
    const sc = scenarioPnl(s);
    const be = breakevenPct(s);
    const qty = Math.max(1, contractsFor(s, sizeAlloc()));
    const ml = maxProfitLoss(s);
    const legsTxt = s.legs.map((l) => `${l.side[0]} ${l.row.strike}${l.row.right}${l.row.delta != null ? ` (${Math.abs(l.row.delta).toFixed(2)}Δ)` : ""}`).join(" / ");
    const sel = s.name === state.selStructure;
    return `<tr${sel ? ' style="background:#4da3ff14"' : ""}>
      <td class="l"><b>${esc(s.name)}</b>${s.note ? `<br><span class="cap" style="display:inline">${esc(s.note)}</span>` : ""}</td>
      <td class="l">${esc(legsTxt)}</td>
      <td>${fmt.num(s.mid, 2)}${s.credit ? " cr" : ""}</td>
      <td>${s.nat != null ? fmt.num(s.nat, 2) : "—"}</td>
      <td>${s.delta != null ? fmt.num(s.delta * 100 * qty, 0) : "—"}</td>
      <td>${s.theta != null ? fmt.money(s.theta * 100 * qty) : "—"}</td>
      <td>${s.vega != null ? fmt.money(s.vega * 100 * qty) : "—"}</td>
      <td>${be != null ? fmt.pctRaw(be * 100, 1) : "—"}</td>
      <td class="${sc ? clsSign(sc.target) : ""}">${sc ? fmt.money(sc.target * qty) : "—"}</td>
      <td class="${sc ? clsSign(sc.stop) : ""}">${sc ? fmt.money(sc.stop * qty) : "—"}</td>
      <td class="${sc ? clsSign(sc.flat) : ""}">${sc ? fmt.money(sc.flat * qty) : "—"}</td>
      <td>${ml.maxProfit != null ? fmt.money(ml.maxProfit * qty) : "uncapped"} / ${ml.maxLoss != null ? fmt.money(-ml.maxLoss * qty) : "—"}</td>
      <td>${sc && sc.ev != null ? `<b class="${clsSign(sc.ev)}">${fmt.money(sc.ev * qty)}</b>` : "—"}</td>
      <td>${taxLight(spreadTax(s))}</td>
      <td class="l">${s.tradeable ? `<button class="btn xs${sel ? "" : " ghost"}" data-struct="${esc(s.name)}">${sel ? "selected" : "select"}</button>` : '<span class="cap">n/a</span>'}</td>
    </tr>`;
  }).join("");

  // verdict: best EV among rows that have one, vs stock baseline
  let verdict = "";
  if (hasSignal() && p.target && stats) {
    const evs = state.structures.map((s) => ({ s, sc: scenarioPnl(s), qty: contractsFor(s, sizeAlloc()) }))
      .filter((x) => x.qty > 0 && x.sc && x.sc.ev != null);
    const riskAlloc = sizeAlloc();
    const dist = Math.abs(p.entry - p.stop);
    const sh = dist > 0 ? Math.floor(riskAlloc / dist) : 0;
    const stockEv = stats.win_rate * sh * Math.abs(p.target - p.entry) +
      (1 - stats.win_rate) * -(riskAlloc * 0.8);
    if (evs.length) {
      const best = evs.reduce((a, b) => (b.sc.ev * b.qty > a.sc.ev * a.qty ? b : a));
      const bestEv = best.sc.ev * best.qty;
      verdict = bestEv > stockEv
        ? `<div style="margin-top:8px"><b style="color:#3ddb8f">Verdict:</b> <b>${esc(best.s.name)}</b> wins on modeled EV (${fmt.money(bestEv)} vs stock ${fmt.money(stockEv)}) at the same risk budget.</div>`
        : `<div style="margin-top:8px"><b style="color:#ffc14d">Verdict: USE STOCK</b> — no structure beats the stock baseline after costs (${fmt.money(stockEv)} vs best ${fmt.money(bestEv)}).</div>`;
    }
  }

  el.innerHTML = `<div class="card" style="margin-bottom:12px">
    <div style="font:700 14px inherit;margin-bottom:4px">Structure shootout
      <span class="cap" style="display:inline;font-weight:400">· spot ${fmt.num(wb.spot, 2)} · ${esc(wb.chain.expiry)} (${wb.chain.dte}d)
      ${age != null ? `· quotes ${age}s old` : ""}
      ${delayed ? ' · <b style="color:#ffc14d">DELAYED MARKS — do not anchor limits off these</b>' : ""}</span></div>
    <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:4px 0 8px">
      <label class="cap">Exit-date repricing (BSM approx):</label>
      <label class="cap">IV shift</label><input id="sh_ivshift" value="${state.pricing.ivShiftPts}" style="width:50px"> <span class="cap">pts</span>
      <label class="cap">rate</label><input id="sh_rate" value="${state.pricing.rate}" style="width:56px">
      <label class="cap">div yield</label><input id="sh_divy" value="${state.pricing.divYield}" style="width:50px">
      <span class="cap">P&amp;L is sized to the risk budget at the ${tradeHorizon()} td horizon, not expiry.</span>
    </div>
    <div class="tblwrap"><table class="tbl"><thead><tr>
      <th class="l">Structure</th><th class="l">Legs</th><th>Mid</th><th>Nat</th><th>&Delta; sh-eq</th>
      <th>&Theta;/day</th><th>Vega/pt</th><th>BE move</th><th>P&amp;L thesis</th><th>P&amp;L adverse</th>
      <th>P&amp;L flat</th><th>Max P / L</th><th>EV</th><th>Spread tax</th><th class="l"></th>
    </tr></thead><tbody>${rows}${baselineHtml}</tbody></table></div>
    ${verdict}</div>`;
  el.querySelectorAll("[data-struct]").forEach((b) =>
    b.addEventListener("click", () => { state.selStructure = b.dataset.struct; renderShootout(); renderScenarioLab(); renderSizing(); renderTicket(); }));
  ["sh_ivshift", "sh_rate", "sh_divy"].forEach((id) => {
    const e = document.getElementById(id);
    if (e) e.addEventListener("change", () => {
      state.pricing.ivShiftPts = numInput("sh_ivshift", -3);
      state.pricing.rate = numInput("sh_rate", 0.04);
      state.pricing.divYield = numInput("sh_divy", 0);
      renderForecastLab(); renderShootout(); renderScenarioLab(); renderSizing(); renderTicket();
    });
  });
}

/* ---------------- selected-structure scenario lab ---------------- */
function structurePnlAt(struct, spot, ivShiftPts) {
  const base = scenarioInputs();
  const value = valueAtExit(struct, spot, { r: base.r, q: base.q, ivShift: ivShiftPts / 100 });
  const openingValue = struct.credit ? -struct.mid : struct.mid;
  return (value - openingValue) * 100;
}

function renderScenarioLab() {
  const el = document.getElementById("scenarioLab");
  const s = selStruct(), wb = state.wb;
  if (!el || !s || !wb) { if (el) el.innerHTML = ""; return; }
  const qty = Math.max(1, contractsFor(s, sizeAlloc()));
  const iv = chainAtmIv() || 0.30;
  const span = wb.spot * iv * Math.sqrt(Math.max(1, tradeHorizon()) / 252) * 2.25;
  const lo = Math.max(0.01, wb.spot - span), hi = wb.spot + span;
  const xs = Array.from({ length: 51 }, (_, i) => lo + (hi - lo) * i / 50);
  const baseShift = state.pricing.ivShiftPts;
  const quoteHtml = s.legs.map((l) => {
    const mid = l.row.mid || 0;
    const width = l.row.bid != null && l.row.ask != null ? l.row.ask - l.row.bid : null;
    const wide = width != null && (width > 0.15 || (mid > 0 && width / mid > 0.10));
    return `<div class="opt-quote"><div class="qleg">${l.side} ${l.row.strike}${l.row.right}</div>
      <div class="qpx">${fmt.num(l.row.bid, 2)} / ${fmt.num(l.row.ask, 2)} · mid ${fmt.num(l.row.mid, 2)}</div>
      <div class="qpx">Δ ${fmt.num(l.row.delta, 2)} · IV ${l.row.iv != null ? fmt.pctRaw(l.row.iv * 100, 1) : "—"}
        ${wide ? ' · <b style="color:#ffc14d">WIDE</b>' : ""}</div></div>`;
  }).join("");
  const wideLegs = s.legs.filter((l) => l.row.bid != null && l.row.ask != null &&
    ((l.row.ask - l.row.bid) > 0.15 || (l.row.mid > 0 && (l.row.ask - l.row.bid) / l.row.mid > 0.10))).length;
  el.innerHTML = `<div class="card" style="margin-bottom:12px">
    <div style="font:700 14px inherit">Scenario lab — ${esc(s.name)}
      <span class="cap" style="display:inline;font-weight:400">· ${qty} lot${qty === 1 ? "" : "s"} · ${tradeHorizon()} td horizon · BSM approximation</span></div>
    <div class="opt-quote-grid">${quoteHtml}</div>
    ${wideLegs ? `<div class="opt-risk-warn">${wideLegs} leg${wideLegs === 1 ? " is" : "s are"} wider than $0.15 or 10% of mid. Treat the modeled edge as untradeable until the market tightens.</div>` : ""}
    <div id="optScenarioChart" class="opt-scenario-chart"></div>
    <div class="cap">The lines move only spot and IV; they do not model path, early exercise, dividends beyond the flat yield input, or a changing smile.</div>
  </div>`;
  if (!window.Plotly) return;
  const traces = [
    { shift: baseShift - 5, name: `IV ${fmt.signed(baseShift - 5, 0)} pts`, color: "#4da3ff88", dash: "dot" },
    { shift: baseShift, name: `IV ${fmt.signed(baseShift, 0)} pts`, color: "#f2f5f9", dash: "solid" },
    { shift: baseShift + 5, name: `IV ${fmt.signed(baseShift + 5, 0)} pts`, color: "#ff8c66aa", dash: "dot" },
  ].map((z) => ({ x: xs, y: xs.map((x) => structurePnlAt(s, x, z.shift) * qty), type: "scatter", mode: "lines",
    name: z.name, line: { color: z.color, width: z.dash === "solid" ? 2.4 : 1.5, dash: z.dash } }));
  const scen = scenarioSpots();
  Plotly.newPlot("optScenarioChart", traces, plotLayout({
    height: 265, margin: { l: 55, r: 18, t: 16, b: 40 }, hovermode: "x unified",
    xaxis: { title: "Underlying at horizon", gridcolor: "#1c2230" },
    yaxis: { title: "P&L ($)", gridcolor: "#1c2230", zerolinecolor: "#657080" },
    shapes: [
      { type: "line", x0: wb.spot, x1: wb.spot, y0: 0, y1: 1, yref: "paper", line: { color: "#667080", dash: "dot" } },
      ...(scen.target ? [{ type: "line", x0: scen.target, x1: scen.target, y0: 0, y1: 1, yref: "paper", line: { color: "#3ddb8f55", dash: "dot" } }] : []),
      ...(scen.adverse ? [{ type: "line", x0: scen.adverse, x1: scen.adverse, y0: 0, y1: 1, yref: "paper", line: { color: "#ff6b6b55", dash: "dot" } }] : []),
    ],
  }), PLOT_CFG);
}

/* ---------------- sizing ---------------- */
function selStruct() { return state.structures.find((s) => s.name === state.selStructure) || null; }
function sizeFraction() { return numInput("sz_frac", state.manual.fraction); }
function sizeAlloc() { return tradeRisk() * sizeFraction(); }
function contractsFor(struct, alloc) {
  const per = riskPerUnit(struct);
  return per > 0 ? Math.floor(alloc / per) : 0;
}
function renderSizing() {
  const el = document.getElementById("sizing");
  const s = selStruct();
  if (!s || !tradeRisk()) { el.innerHTML = ""; return; }
  const alloc = sizeAlloc();
  const n = contractsFor(s, alloc);
  const per = riskPerUnit(s);
  const eff = n * per;
  const effUp = (n + 1) * per;
  const warnings = [];
  if (n < 1) warnings.push(`target risk ${fmt.money(alloc)} is below one structure's max-loss basis (${fmt.money(per)}) — no viable size`);
  const quantErr = alloc > 0 ? (alloc - eff) / alloc : 0;
  el.innerHTML = `<div class="card" style="margin-bottom:12px">
    <div style="font:700 14px inherit;margin-bottom:6px">Sizing — defined max loss
      <span class="cap" style="display:inline;font-weight:400">· debit for long premium; width minus credit for credit structures; round-trip commission included</span></div>
    <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:8px">
      <label class="cap">${hasSignal() ? "Fraction of stock Risk_Amt" : "Fraction of manual budget"}</label>
      <input id="sz_frac" value="${state.manual.fraction}" style="width:56px">
      <span class="cap">${hasSignal() ? "(options may run alongside stock — combined delta still matters)" : "(1.0 uses the full manual risk budget)"}</span>
    </div>
    <div class="kv">
      <div class="k">Risk allocation</div><div class="v">${fmt.money(alloc)} <span class="cap" style="display:inline">of ${fmt.money(tradeRisk())}</span></div>
      <div class="k">Contracts</div><div class="v"><b style="font-size:15px">${n}</b> &times; ${esc(s.name)} @ ${fmt.num(s.mid, 2)} ${s.credit ? "credit" : "debit"}</div>
      <div class="k">Effective risk</div><div class="v">${fmt.money(eff)} <span class="cap" style="display:inline">(${fmt.pctRaw(quantErr * 100, 1)} under target; ${n + 1} lots = ${fmt.money(effUp)})</span></div>
      <div class="k">Net delta</div><div class="v">${s.delta != null ? fmt.num(s.delta * 100 * n, 0) + " share-equivalents" : "—"}
        ${stagedStockRow(state.params.ticker) ? ' <b style="color:#ffc14d">+ the staged stock position</b>' : ""}</div>
    </div>
    ${warnings.length ? `<div style="color:#ff6b6b;margin-top:8px">${warnings.map(esc).join("<br>")}</div>` : ""}
  </div>`;
  const fr = document.getElementById("sz_frac");
  if (fr) fr.addEventListener("change", () => {
    state.manual.fraction = Math.max(0, Number(fr.value) || (hasSignal() ? 0.5 : 1));
    renderForecastLab(); renderShootout(); renderScenarioLab(); renderSizing(); renderTicket();
  });
}

/* ---------------- ticket ---------------- */
function snapNetLimit(v, action, tick = 0.05) {
  const n = Number(v) / tick;
  const snapped = String(action).toUpperCase() === "SELL" ? Math.ceil(n - 1e-9) : Math.floor(n + 1e-9);
  return Math.round(snapped * tick * 100) / 100;
}
function renderTicket() {
  const el = document.getElementById("ticket");
  const s = selStruct();
  const wb = state.wb;
  if (!s || !s.tradeable || !wb) { el.innerHTML = ""; return; }
  const n = contractsFor(s, sizeAlloc());
  const action = s.credit ? "SELL" : "BUY";
  const dfltLimit = snapNetLimit(Math.max(0.05, s.mid + (s.credit ? 0.01 : -0.01)), action).toFixed(2);
  const ticketKind = s.legs.length === 1 ? "Single-option ticket" : s.credit ? "Defined-risk credit ticket" : "Combo ticket";
  const legLines = s.legs.map((l) =>
    `${l.side} ${l.row.strike}${l.row.right} ${wb.chain.expiry} (conId ${l.row.con_id || "?"})`).join("<br>");
  el.innerHTML = `<div class="card" style="max-width:860px;margin-bottom:12px">
    <div style="font:700 14px inherit;margin-bottom:4px">${ticketKind} — ${esc(s.name)}</div>
    <p class="cap" style="margin:0 0 8px">${s.legs.length === 1 ? "One SMART-routed option limit order." : "One native SMART BAG limit order (atomic — never legs you in)."}
      The execution agent independently re-validates contract identity, quantity, price, and max-risk caps.</p>
    <div class="exec-legs" style="margin-bottom:8px">${legLines}</div>
    ${state.params.cond ? `<div class="openconds" style="margin-bottom:8px"><div class="oc-h">Entry condition — check before sending</div>
      <div class="oc-line">${esc(state.params.cond)}</div></div>` : ""}
    <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px">
      <label class="cap">Qty</label><input id="tk_qty" value="${n > 0 ? n : ""}" style="width:60px">
      <label class="cap">${s.credit ? "Credit limit" : "Debit limit"}</label><input id="tk_limit" value="${dfltLimit}" style="width:76px">
      <span class="cap">(mid ${fmt.num(s.mid, 2)} / natural ${s.nat != null ? fmt.num(s.nat, 2) : "?"}; 0.05 grid, ${action} snaps ${s.credit ? "up" : "down"} toward your favor)</span>
      <label class="cap">TIF</label><select id="tk_tif"><option>DAY</option><option>GTC</option></select>
      <span class="cap">Account: <b>${esc(state.account)}</b></span>
    </div>
    <button class="btn" id="tk_send">${execMode() === "dry-run" ? "Preview / dry-run" : `Send ${s.legs.length === 1 ? "option" : "spread"}`}</button>
    <span id="tk_msg" class="cap" style="margin-left:10px"></span>
  </div>`;
  document.getElementById("tk_send").addEventListener("click", sendOptionOrder);
}

function execMode() {
  if (state.book && state.book.mode === "live") return "live";
  const online = !!(state.status && state.status.online);
  const fresh = !!(state.book && state.book.at && (Date.now() - state.book.at) <= 90000);
  if (online && fresh && state.book.mode === "dry-run") return "dry-run";
  return "unknown";
}
function actionLead(verb) {
  const m = execMode();
  return m === "dry-run" ? `Dry-run ${verb} (places nothing):` : `⚠️ LIVE — really ${verb}`;
}

const idemState = { id: null, key: null };
function commandId(type, account, payload) {
  const key = JSON.stringify({ type, account, payload });
  if (idemState.key !== key || !idemState.id) {
    idemState.id = crypto.randomUUID();
    idemState.key = key;
  }
  return idemState.id;
}

function canonicalPayloadLegs(struct, expiry, action) {
  return struct.legs.map((l) => ({
    // A SELL parent inverts the canonical BUY combo. The UI keeps the desired
    // trade sides; the BAG definition flips them for a credit parent order.
    side: action === "SELL" ? (l.side === "BUY" ? "SELL" : "BUY") : l.side,
    right: l.row.right, expiry, strike: l.row.strike, ratio: 1, con_id: l.row.con_id || null,
  }));
}

function sendOptionOrder() {
  const s = selStruct();
  const wb = state.wb, p = state.params;
  const msg = document.getElementById("tk_msg");
  if (!s || !wb) return;
  const qty = Math.floor(Number(document.getElementById("tk_qty").value));
  const limit = Number(document.getElementById("tk_limit").value);
  if (!(qty > 0)) { msg.textContent = "BLOCKED: qty must be a positive integer"; return; }
  if (!(limit > 0)) { msg.textContent = "BLOCKED: limit must be > 0"; return; }
  if (s.width != null && limit >= s.width) { msg.textContent = `BLOCKED: net ${s.credit ? "credit" : "debit"} ${limit} >= width ${s.width}`; return; }
  const action = s.credit ? "SELL" : "BUY";
  const snapped = snapNetLimit(limit, action);
  const riskPremium = s.credit ? s.width - snapped : snapped;
  if (!(riskPremium > 0)) { msg.textContent = "BLOCKED: defined max risk must be > 0"; return; }
  const payload = {
    symbol: wb.ticker,
    action,
    quantity: qty,
    limit: snapped,
    tif: document.getElementById("tk_tif").value || "DAY",
    structure: s.name.toLowerCase().replace(/[^a-z0-9]+/g, "_"),
    debit_risk: riskPremium,            // backward-compatible max-risk premium basis
    risk_per_unit: riskPremium,
    credit: !!s.credit,
    legs: canonicalPayloadLegs(s, wb.chain.expiry, action),
    strategy: p.strategy || null, signal_date: p.sig || null,
    entry_condition: p.cond || null,
  };
  const legsTxt = s.legs.map((l) => `${l.side[0]}${l.row.strike}${l.row.right}`).join("/");
  const riskDollars = riskPremium * 100 * qty + commRT(s) * qty;
  if (!confirm(`${actionLead("place")} ${action} ${qty}x ${wb.ticker} ${wb.chain.expiry} [${legsTxt}] LMT ${payload.limit} ${s.credit ? "credit" : "debit"} on ${state.account}?\n\nDefined max risk: about ${fmt.money(riskDollars)}.`)) return;
  sendCommand("option_spread", payload, "tk_msg");
}

async function sendCommand(type, payload, msgId) {
  const msg = document.getElementById(msgId);
  if (msg) msg.textContent = "sending...";
  const id = commandId(type, state.account, payload);
  try {
    const r = await fetch("/exec-command", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, type, account: state.account, payload }),
    });
    const d = await r.json();
    const ok = r.ok && d && d.ok;
    if (ok) { idemState.id = null; idemState.key = null; }
    if (msg) msg.textContent = ok ? `queued ${(d.id || id).slice(0, 8)} — see Activity below` : `error: ${(d && d.error) || ("HTTP " + r.status)}`;
  } catch (e) {
    if (msg) msg.textContent = "error: " + e;
  }
  setTimeout(pollExec, 800);
}

/* ---------------- execution status / activity ---------------- */
function setAccount(acct) {
  state.account = acct;
  document.querySelectorAll("[data-acct]").forEach((b) =>
    b.className = "btn" + (b.dataset.acct === acct ? "" : " ghost"));
  renderTicket();
}
async function pollExec() {
  const [st, bk, cm] = await Promise.all([
    fetchJSONOrNull("/exec-status"),
    fetchJSONOrNull("/exec-book"),
    fetchJSONOrNull("/exec-commands"),
  ]);
  state.status = st || { online: false };
  state.book = (bk && bk.book) || null;
  state.commands = ((cm && cm.commands) || []).filter((c) => c.type === "option_spread" || c.type === "echo");
  const dot = document.getElementById("connDot");
  if (dot) dot.textContent = state.status.online ? "agent online" : "agent offline";
  setAsof(state.status.online ? "execution online" : "execution offline");
  renderModeBanner();
  renderActivity();
}
function renderModeBanner() {
  const el = document.getElementById("modeBanner");
  if (!el) return;
  const mode = execMode();
  if (mode === "live") {
    el.innerHTML = `<div class="card" style="border-color:#a8852f;background:rgba(255,193,77,.10);padding:9px 14px;font:700 13px inherit;color:#ffc14d">
      &#9888;&#65039; LIVE ARMED — an options order sent here transmits to IBKR if option_spread is in LIVE_TYPES.</div>`;
  } else if (mode === "unknown") {
    el.innerHTML = `<div class="card" style="border-color:#a8852f;background:rgba(255,193,77,.10);padding:9px 14px;font:700 13px inherit;color:#ffc14d">
      &#9888;&#65039; MODE UNKNOWN — assume LIVE. No fresh book confirms dry-run.</div>`;
  } else {
    el.innerHTML = `<div class="card" style="border-color:#2c8f63;background:rgba(61,219,143,.08);padding:9px 14px;font:700 13px inherit;color:#3ddb8f">
      &#9679; DRY-RUN MODE — options orders are validated + previewed, nothing transmits.</div>`;
  }
}
function stateBadge(st) {
  const map = { dry_run: ["#3ddb8f", "DRY-RUN OK"], rejected: ["#ff6b6b", "REJECTED"],
                executed: ["#ffc14d", "EXECUTED"], duplicate: ["#9aa3b2", "duplicate"],
                pushed: ["#9aa3b2", "pushed"], error: ["#ffc14d", "ERROR"] };
  const [c, t] = map[st] || ["#9aa3b2", st || ""];
  return `<span style="color:${c};font-weight:600">${esc(t)}</span>`;
}
function renderActivity() {
  const el = document.getElementById("activity");
  if (!el) return;
  const cmds = state.commands || [];
  if (!cmds.length) { el.innerHTML = ""; return; }
  const rows = cmds.map((c) => {
    const res = c.result || {};
    const pv = res.preview || {};
    let cell = `<span class="${res.ok === true ? "pos" : res.ok === false ? "neg" : "neu"}">${esc(res.detail || c.state || "pending")}</span>`;
    if (pv.legs && pv.legs.length) {
      cell += `<div class="exec-legs">${pv.legs.map(esc).join("<br>")}` +
        (pv.summary ? `<br><span style="color:#c7ccd6;font-weight:600">${esc(pv.summary)}</span>` : "") + "</div>";
    }
    const t = c.created_at ? new Date(c.created_at).toLocaleTimeString("en-US", { hour12: false }) : "";
    return `<tr style="vertical-align:top"><td class="l" style="color:#8c95a2">${esc(t)}</td>
      <td class="l">${esc(c.account || "")}</td><td class="l">${stateBadge(c.state)}</td><td class="l">${cell}</td></tr>`;
  }).join("");
  el.innerHTML = `<div style="font:700 14px inherit;margin-bottom:6px">Options activity</div>
    <div class="tblwrap"><table class="tbl"><thead><tr>
    <th class="l">time</th><th class="l">acct</th><th class="l">state</th><th class="l">result / preview</th>
    </tr></thead><tbody>${rows}</tbody></table></div>`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
