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
  seasonalTheses: null, risk: null,
  section: "moontower", marketTicker: null,
  marketGroup: "All ETFs",
  wb: null,                            // latest workbench result
  wbId: null, wbTimer: null,
  structures: [],                      // shootout rows (assembled client-side)
  selStructure: null,
  manual: { view: "bullish", horizon: 10, risk: 1500, target: null, adverse: null, fraction: 1.0 },
  house: { source: "seasonal", appliedSource: null, appliedHorizon: null },
  forecast: {
    active: false, event: "touch", target: null, probability: 0.50,
    cutoff: defaultForecastDate(), touchIvShift: 8, noTouchSpot: null,
    noTouchIvShift: -3, objective: "ev_risk", selectTop: false,
  },
  pricing: { ivShiftPts: -3, rate: 0.04, divYield: 0 },
  calendar: { front: null, back: null, frontChain: null, backChain: null, loading: false, error: null },
  account: "primary",
  book: null, status: null, commands: [],
  pollTimer: null,
};

/* ---------------- init + prefill ---------------- */
async function initOptions() {
  renderNav("options.html");
  state.params = parseParams();
  state.section = state.params.section || (state.params.ticker ? "ideas" : "moontower");
  state.manual.view = isShort() ? "bearish" : "bullish";
  state.manual.horizon = state.params.hold || 10;
  state.manual.risk = state.params.risk || 1500;
  state.manual.fraction = hasSignal() ? 0.5 : 1.0;
  const [iv, market, stats, earn, sig, seasonalTheses, risk] = await Promise.all([
    fetchJSONOrNull("data/iv_context.json"),
    fetchJSONOrNull("data/options_market.json"),
    fetchJSONOrNull("data/strategy_stats.json"),
    fetchJSONOrNull("data/earnings_next.json"),
    fetchJSONOrNull("data/signals.json"),
    fetchJSONOrNull("data/seasonality/theses.json"),
    fetchJSONOrNull("data/risk.json"),
  ]);
  state.ivCtx = iv; state.market = market; state.stats = stats; state.earn = earn; state.signals = sig;
  state.seasonalTheses = seasonalTheses; state.risk = risk;
  document.getElementById("content").innerHTML = shell();
  document.querySelectorAll("[data-opt-section]").forEach((button) =>
    button.addEventListener("click", () => switchOptionsSection(button.dataset.optSection)));
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
  switchOptionsSection(state.section);
  renderMoontowerOverview();
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
    section: ["moontower", "ideas"].includes(q.get("section")) ? q.get("section") : null,
  };
}

function switchOptionsSection(section) {
  state.section = section === "ideas" ? "ideas" : "moontower";
  document.querySelectorAll("[data-opt-section]").forEach((button) => {
    const active = button.dataset.optSection === state.section;
    button.classList.toggle("ghost", !active);
    button.setAttribute("aria-selected", active ? "true" : "false");
  });
  const moon = document.getElementById("moontowerSection");
  const ideas = document.getElementById("ideasSection");
  if (moon) moon.hidden = state.section !== "moontower";
  if (ideas) ideas.hidden = state.section !== "ideas";
  document.querySelectorAll(".opt-trade-only").forEach((el) => { el.hidden = state.section !== "ideas"; });
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
    renderHouseThesis(); renderThesis(); renderComparator(); buildStructures(); renderForecastLab(); renderShootout();
    renderScenarioLab(); renderSizing(); renderTicket();
  }
}

function shell() {
  const p = state.params;
  const view = isShort() ? "bearish" : "bullish";
  return `
    <div class="opt-section-switch" role="tablist" aria-label="Options workspace">
      <button class="btn" role="tab" data-opt-section="moontower">Moontower</button>
      <button class="btn ghost" role="tab" data-opt-section="ideas">Our Trade Ideas</button>
      <span class="cap">Screen the volatility surface first; express a thesis second.</span>
    </div>
    <div id="modeBanner" class="opt-trade-only"></div>
    <div class="card opt-toolbar">
      <label><span>Ticker</span><input id="wbTicker" placeholder="SPY" style="text-transform:uppercase"></label>
      <label class="opt-trade-only"><span>View</span><select id="wbView">
        <option value="bullish"${view === "bullish" ? " selected" : ""}>Bullish</option>
        <option value="bearish"${view === "bearish" ? " selected" : ""}>Bearish</option>
        <option value="big_move">Big move</option><option value="range">Range / short vol</option>
        <option value="hedge">Portfolio hedge</option></select></label>
      <label class="opt-trade-only"><span>Horizon (td)</span><input id="wbHorizon" type="number" min="1" max="63" value="${p.hold || 10}"${hasSignal() ? " disabled" : ""}></label>
      <label class="opt-trade-only"><span>Risk budget</span><input id="wbRisk" type="number" min="1" step="100" value="${p.risk || 1500}"${hasSignal() ? " disabled" : ""}></label>
      <button class="btn" id="wbGo">Analyze chain</button>
      <span class="exec-tabs opt-trade-only"><button class="btn" data-acct="primary">Primary</button>
        <button class="btn ghost" data-acct="pa">PA</button></span>
      <span id="wbMsg" class="cap"></span>
      <span id="connDot" class="cap" style="margin-left:auto"></span>
    </div>
    <section id="moontowerSection" role="tabpanel">
      <div id="marketOverview"></div>
      <div id="ivStrip"></div>
      <div id="volDashboard"></div>
      <div id="termLab"></div>
      <div id="positioningLab"></div>
    </section>
    <section id="ideasSection" role="tabpanel" hidden>
      <div id="ideaBridge"></div>
      ${signalContextHtml(p)}
      <div id="houseThesis"></div>
      <div id="thesis"></div>
      <div id="forecastLab"></div>
      <div id="calendarBuilder"></div>
      <div id="expiryRow"></div>
      <div id="emCompare"></div>
      <div id="shootout"></div>
      <div id="scenarioLab"></div>
      <div id="sizing"></div>
      <div id="ticket"></div>
      <div id="activity" style="margin-top:18px"></div>
    </section>`;
}

function signalContextHtml(p) {
  if (!hasSignal()) {
    return `<p class="cap" style="margin:0 0 12px">No signal context — manual mode. Open this page via a
      Signals-page <b>Express</b> button to pre-fill entry/stop/target/risk and unlock the comparator,
      shootout scenario P&amp;L, and sizing.</p>`;
  }
  const dupe = stagedStockRow(p.ticker);
  const dupeWarn = dupe ? `<div class="oc-line" style="color:#ffc14d"><b>[WARN] Stock signal also staged for
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

function renderMoontowerOverview() {
  const el = document.getElementById("marketOverview"), market = state.market;
  if (!el || !market || !market.n) { if (el) el.innerHTML = ""; return; }
  const allRows = market.etfs || [];
  const rows = state.marketGroup === "All ETFs" ? allRows : allRows.filter((row) => row.group === state.marketGroup);
  const median = (values) => {
    const clean = values.filter((value) => value != null && isFinite(Number(value))).map(Number).sort((a, b) => a - b);
    if (!clean.length) return null;
    const mid = Math.floor(clean.length / 2);
    return clean.length % 2 ? clean[mid] : (clean[mid - 1] + clean[mid]) / 2;
  };
  const ivPct = median(rows.map((row) => row.pctile));
  const rvPct = median(rows.map((row) => row.rv_pctile));
  const asofMs = market.asof ? new Date(market.asof + "T00:00:00").getTime() : NaN;
  const ageDays = isFinite(asofMs) ? Math.floor((Date.now() - asofMs) / 86400000) : null;
  const stale = ageDays != null && ageDays > 5;
  const label = stale ? "ETF volatility map is stale" : ivPct == null ? "No comparable volatility history" : ivPct < 35 ? "ETF volatility is broadly low" : ivPct > 65 ? "ETF volatility is broadly elevated" : "ETF volatility is mixed";
  const tone = stale || ivPct == null ? "#ffc14d" : ivPct < 35 ? "#3ddb8f" : ivPct > 65 ? "#ff6b6b" : "#ffc14d";
  const groups = ["All ETFs", ...(market.groups || []).map((group) => group.name)];
  const groupTabs = groups.map((group) => `<button class="btn xs ${group === state.marketGroup ? "" : "ghost"}" data-opt-group="${esc(group)}">${esc(group)}</button>`).join("");
  const laneMeta = {
    "Buy gamma / vega": ["#3ddb8f", "Low IV, low RV and low VRP. Look for underpriced convexity."],
    "Sell vega": ["#ff6b6b", "High VRP with high RV. Keep the short-vol expression defined risk."],
    "Long calendar": ["#4da3ff", "Low outright IV, high VRP and a flat curve. Buy back-month vega against the front."],
    "Short calendar": ["#ffc14d", "High IV/RV, low VRP and a steep front. Sell the stressed tenor against the back."],
  };
  const lanes = Object.entries(laneMeta).map(([name, meta]) => {
    const picks = rows.slice().sort((a, b) => Number((b.fits || {})[name] || 0) - Number((a.fits || {})[name] || 0)).slice(0, 4);
    return `<div class="opt-lane" style="border-top-color:${meta[0]}"><div class="opt-lane-head">${esc(name)}</div>
      <div class="cap">${meta[1]}</div><div class="opt-lane-picks">${picks.map((row) =>
        `<button class="opt-ticker-link" data-opt-ticker="${row.ticker}" title="${esc(row.first_rejection || "")}"><b>${row.ticker}</b><span>${fmt.num((row.fits || {})[name], 0)} · ${row.coverage_dims || 0}/4</span></button>`).join("") || '<span class="cap">No covered names.</span>'}</div></div>`;
  }).join("");
  const dispersion = market.dispersion || {};
  const tableRows = rows.slice().sort((a, b) => (b.setup_score || 0) - (a.setup_score || 0)).map((row) => `<tr>
    <td><button class="opt-ticker-link compact" data-opt-ticker="${row.ticker}"><b>${row.ticker}</b></button></td>
    <td>${esc(row.group || "")}</td><td>${fmt.pctRaw((row.iv || 0) * 100, 1)}</td><td>p${fmt.num(row.pctile, 0)}</td>
    <td>${fmt.pctRaw((row.rv30 || row.rv21 || 0) * 100, 1)}</td><td>${row.rv_pctile != null ? "p" + fmt.num(row.rv_pctile, 0) : "—"}</td>
    <td class="${row.vrp_log > 0 ? "neg" : "pos"}">${row.vrp_log != null ? fmt.signed(row.vrp_log, 1) : "—"}</td>
    <td>${row.steepness_xs_pct != null ? "p" + fmt.num(row.steepness_xs_pct, 0) : "collecting"}</td><td>${row.coverage_dims || 0}/4</td>
    <td><span class="opt-setup-pill">${esc(row.setup || "unranked")}</span><div class="cap">${esc(row.first_rejection || "")}</div></td></tr>`).join("");
  const shockHtml = (dispersion.corr_shocks || []).map((shock) => `<span>ρ ${fmt.num(shock.rho, 1)} → ${fmt.pctRaw(shock.basket_iv * 100, 1)}</span>`).join("");
  el.innerHTML = `<section class="opt-market-shell">
    <div class="card opt-market-hero"><div><div class="cap">Moontower-style ETF / index cockpit · ${rows.length} covered</div>
      <div class="opt-hero-value" style="color:${tone}">${label}</div><div class="opt-meter"><i style="margin-left:${Math.max(0, Math.min(100, ivPct || 0))}%"></i></div></div>
      <div class="opt-market-kpis"><div><span>IV percentile</span><b>${fmt.num(ivPct, 0)}</b></div>
        <div><span>Median log VRP</span><b>${market.median_vrp_log != null ? fmt.signed(market.median_vrp_log, 1) : "—"}</b></div>
        <div><span>RV percentile</span><b>${fmt.num(rvPct, 0)}</b></div><div><span>Full 4D coverage</span><b>${market.full_4d_n || 0} / ${market.n}</b></div></div>
      <div class="cap">${stale ? `<b style="color:#ffc14d">${ageDays}d old — do not trade from the ranks.</b> · ` : ""}Level/RV as of ${esc(market.asof || "?")}; surface ${esc(market.surface_asof || "not seeded")}. Missing curve data reduces the score instead of being imputed.</div></div>
    <div class="opt-evidence-bar"><b>Workflow:</b> candidate → why now → first rejection → live surface → executable structure. A high screen score advances research; it is not a trade recommendation.</div>
    <div class="opt-group-tabs">${groupTabs}</div>
    <div class="opt-market-grid"><div class="card"><div class="opt-section-head"><div><b>Cross-asset volatility map</b><div class="cap">RV rank → x · 100×ln(IV30/RV30) → y · color = own-history IV rank</div></div></div><div id="optMarketScatter" class="opt-market-chart"></div></div>
      <div class="card opt-dispersion-card"><div class="opt-section-head"><div><b>Index / sector dispersion</b><div class="cap">Equal-weight sector proxy—not constituent SPX implied correlation</div></div></div>
        <div class="opt-dispersion-read"><div><span>Sector IV − SPY</span><b>${dispersion.iv_spread_points != null ? fmt.signed(dispersion.iv_spread_points, 1) + " pts" : "—"}</b></div>
          <div><span>Implied-corr proxy</span><b>${dispersion.implied_corr_proxy != null ? fmt.num(dispersion.implied_corr_proxy, 2) : "—"}</b></div>
          <div><span>Realized corr 21d</span><b>${dispersion.sector_corr_21d != null ? fmt.num(dispersion.sector_corr_21d, 2) : "—"}</b></div></div>
        <div class="opt-corr-shocks">${shockHtml}</div><p class="cap">${esc(dispersion.basis || "Needs SPY plus sector coverage.")}</p>
        <div class="opt-dispersion-note">First rejection: this is a dirty, equal-weight basket. Correlation rebound and sector-basis mismatch can overwhelm the apparent spread.</div></div></div>
    <div class="opt-lanes">${lanes}</div>
    <details class="card opt-etf-tape"><summary><b>ETF volatility tape</b> <span class="cap">· four-factor screen · click a ticker for the live surface</span></summary>
      <div class="table-wrap"><table><thead><tr><th>Ticker</th><th>Sleeve</th><th>IV30</th><th>IV pct</th><th>RV30</th><th>RV pct</th><th>Log VRP</th><th>Curve rank</th><th>Coverage</th><th>Nearest screen / first rejection</th></tr></thead><tbody>${tableRows}</tbody></table></div></details>
  </section>`;
  el.querySelectorAll("[data-opt-group]").forEach((button) => button.addEventListener("click", () => {
    state.marketGroup = button.dataset.optGroup; renderMoontowerOverview();
  }));
  el.querySelectorAll("[data-opt-ticker]").forEach((button) => button.addEventListener("click", () => {
    state.marketTicker = button.dataset.optTicker;
    const input = document.getElementById("wbTicker"); if (input) input.value = state.marketTicker;
    loadTicker();
  }));
  renderMoontowerScatter(rows);
}

function renderMoontowerScatter(rows) {
  const el = document.getElementById("optMarketScatter");
  if (!el || !window.Plotly || !rows.length) return;
  const compact = window.innerWidth < 650;
  Plotly.newPlot(el, [{ type: "scatter", mode: compact ? "markers" : "markers+text",
    x: rows.map((row) => row.rv_pctile), y: rows.map((row) => row.vrp_log), text: rows.map((row) => row.ticker), textposition: "top center",
    customdata: rows.map((row) => [row.group, row.iv * 100, row.pctile, (row.rv30 || row.rv21) * 100, row.setup, row.coverage_dims]),
    hovertemplate: "<b>%{text}</b> · %{customdata[0]}<br>IV %{customdata[1]:.1f}% (p%{customdata[2]:.0f})<br>RV30 %{customdata[3]:.1f}% · log VRP %{y:.1f}<br>%{customdata[4]} · %{customdata[5]}/4 dimensions<extra></extra>",
    marker: { size: rows.map((row) => 8 + Math.max(0, Math.min(100, row.pctile || 0)) / 12), color: rows.map((row) => row.pctile),
      colorscale: [[0, "#3ddb8f"], [.5, "#ffc14d"], [1, "#ff6b6b"]], cmin: 0, cmax: 100, showscale: true,
      colorbar: { title: "IV pct", thickness: 9 }, line: { color: "#0b0f17", width: 1 } },
  }], plotLayout({ height: compact ? 300 : 330, margin: { l: 50, r: compact ? 38 : 55, t: 15, b: 45 }, showlegend: false,
    xaxis: { title: "Realized-vol percentile", range: [-5, 105], gridcolor: "#202838", zeroline: false },
    yaxis: { title: "100 × ln(IV30 / RV30)", gridcolor: "#202838", zerolinecolor: "#6f7888" },
    shapes: [{ type: "line", x0: 50, x1: 50, y0: 0, y1: 1, yref: "paper", line: { color: "#303a4c", dash: "dot" } },
             { type: "line", x0: 0, x1: 1, xref: "paper", y0: 0, y1: 0, line: { color: "#6f7888", dash: "dot" } }],
  }), PLOT_CFG);
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
  const appliedLabel = state.house.appliedSource === "risk" ? "SPY risk-dial thesis"
    : state.house.appliedSource === "seasonal" ? "Seasonal thesis" : "Manual thesis";
  if (state.manual.target == null) state.manual.target = round2(d.target);
  if (state.manual.adverse == null) state.manual.adverse = round2(d.adverse);
  el.innerHTML = `<div class="card" style="margin-bottom:12px">
    <div style="font:700 14px inherit;margin-bottom:7px">${appliedLabel}
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
    state.house.appliedSource = null;
    state.house.appliedHorizon = null;
    state.house.source = "seasonal";
    state.calendar = { front: null, back: null, frontChain: null, backChain: null, loading: false, error: null };
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
  renderTradeIdeaBridge();
  renderHouseThesis();
  renderThesis();
  renderIvStrip();
  renderVolDashboard();
  renderTermLab();
  renderPositioningLab();
  renderCalendarBuilder();
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

function renderTradeIdeaBridge() {
  const el = document.getElementById("ideaBridge"), wb = state.wb;
  if (!el || !wb) { if (el) el.innerHTML = ""; return; }
  const candidate = ((state.market || {}).etfs || []).find((row) => row.ticker === wb.ticker);
  const houseLabel = state.house.appliedSource === "risk"
    ? "SPY risk-dial distribution"
    : state.house.appliedSource === "seasonal" ? "weighted seasonal distribution" : null;
  const source = houseLabel || (hasSignal() ? `${state.params.strategy || "strategy"} signal` : "manual thesis");
  const candidateHtml = candidate ? `<div><span>Screen provenance</span><b>${esc(candidate.setup)} · ${fmt.num(candidate.setup_score, 0)} · ${candidate.coverage_dims}/4 dimensions</b></div>` :
    '<div><span>Screen provenance</span><b>Outside the ranked ETF cockpit</b></div>';
  el.innerHTML = `<div class="card opt-idea-bridge"><div class="opt-section-head"><div><b>How this trade idea is formed</b><div class="cap">Thesis supplies direction, prices, timing, probability, and risk; the chain supplies only feasible structures and executable costs.</div></div></div>
    <div class="opt-bridge-grid"><div><span>Thesis source</span><b>${esc(source)}</b></div>${candidateHtml}
      <div><span>Available set</span><b>${(wb.chain && wb.chain.strikes || []).length} quoted contracts · ${(wb.expiries || []).length} expiries</b></div>
      <div><span>Ranking engine</span><b>Scenario P&amp;L, executable marks, spread tax, Greeks, and defined risk</b></div></div>
    ${candidate ? `<div class="opt-evidence-bar"><b>First rejection:</b> ${esc(candidate.first_rejection || "Live liquidity and catalyst timing still need confirmation.")}</div>` : ""}
    <div class="cap">The volatility screen never invents direction. A selected house distribution can pre-fill an editable candidate thesis; BSM remains a scenario repricer, not a prediction model.</div></div>`;
}

function futureBusinessDate(days) {
  const d = new Date(); d.setHours(12, 0, 0, 0);
  let left = Math.max(0, Math.round(Number(days) || 0));
  while (left > 0) {
    d.setDate(d.getDate() + 1);
    if (d.getDay() > 0 && d.getDay() < 6) left--;
  }
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function nearestHorizon(keys, requested) {
  const values = (keys || []).map(Number).filter((value) => isFinite(value) && value > 0);
  if (!values.length) return null;
  return values.reduce((best, value) => Math.abs(value - requested) < Math.abs(best - requested) ? value : best, values[0]);
}

function shrinkHouseProbability(raw, n, strength = 4) {
  const sample = Math.max(0, Number(n) || 0);
  const p = Math.max(0, Math.min(1, Number(raw)));
  return (p * sample + 0.5 * strength) / (sample + strength);
}

function seasonalHouseThesis(ticker, requestedHorizon) {
  const payload = state.seasonalTheses || {};
  const record = (payload.tickers || {})[String(ticker || "").toUpperCase()];
  if (!record || !record.horizons) return null;
  const horizon = nearestHorizon(Object.keys(record.horizons), requestedHorizon);
  const dist = record.horizons[String(horizon)] || {};
  if (!dist.eligible) return {
    sourceKey: "seasonal", sourceLabel: "Seasonal distribution", horizon,
    eligible: false, asof: record.asof, firstRejection: dist.first_rejection || "Insufficient seasonal history.",
    nSame: dist.n_same_cycle || 0, nOther: dist.n_other_cycle || 0,
  };
  const pUp = Number(dist.p_up);
  const direction = pUp >= 0.5 ? "bullish" : "bearish";
  const side = direction === "bullish" ? dist.bull : dist.bear;
  const spot = Number((state.wb || {}).spot);
  const targetReturn = Number(side && side.target_return);
  const noTouchReturn = Number(side && side.no_touch_return);
  const adverseReturn = direction === "bullish" ? Number(dist.q10) : Number(dist.q90);
  const probability = Number(side && side.touch_probability);
  return {
    sourceKey: "seasonal", sourceLabel: "Seasonal distribution", eligible: true,
    horizon, asof: record.asof, direction, pUp,
    probability: isFinite(probability) ? probability : Number(side.terminal_probability),
    terminalProbability: Number(side.terminal_probability), event: "touch",
    targetReturn, noTouchReturn, adverseReturn,
    target: spot * (1 + targetReturn),
    noTouchSpot: spot * (1 + noTouchReturn),
    adverse: spot * (1 + adverseReturn),
    q10: Number(dist.q10), q25: Number(dist.q25), median: Number(dist.median),
    q75: Number(dist.q75), q90: Number(dist.q90), mean: Number(dist.mean_return),
    forecastRv: dist.forecast_rv == null ? null : Number(dist.forecast_rv),
    effectiveN: Number(dist.effective_n), nSame: dist.n_same_cycle, nOther: dist.n_other_cycle,
    confidence: dist.confidence || "low", firstRejection: dist.first_rejection,
    methodology: "70% same presidential-cycle years + 30% other-cycle years; disjoint cohorts; 20-year recency half-life.",
  };
}

function riskHouseThesis(requestedHorizon) {
  const risk = state.risk || {};
  const all = risk.forward_returns || {};
  const scoreKey = requestedHorizon <= 7 ? "5d" : requestedHorizon <= 31 ? "21d" : "63d";
  const block = all[scoreKey];
  if (!block || !block.returns) return null;
  const horizon = nearestHorizon(Object.keys(block.returns).filter((key) => block.returns[key]), requestedHorizon);
  const dist = block.returns[String(horizon)] || block.returns[horizon];
  if (!dist) return null;
  const n = Number(dist.n || block.n_episodes || 0);
  const rawPUp = dist.p_up != null ? Number(dist.p_up) : 1 - Number(dist.pct_neg);
  const pUp = shrinkHouseProbability(rawPUp, n);
  const direction = pUp >= 0.5 ? "bullish" : "bearish";
  const targetReturn = direction === "bullish"
    ? Number(dist.up_median != null ? dist.up_median : dist.q75 != null ? dist.q75 : dist.median)
    : Number(dist.down_median != null ? dist.down_median : dist.q25 != null ? dist.q25 : dist.median);
  const noTouchReturn = direction === "bullish"
    ? Number(dist.down_median != null ? dist.down_median : dist.q25 != null ? dist.q25 : 0)
    : Number(dist.up_median != null ? dist.up_median : dist.q75 != null ? dist.q75 : 0);
  const adverseReturn = direction === "bullish"
    ? Number(dist.q10 != null ? dist.q10 : dist.worst)
    : Number(dist.q90 != null ? dist.q90 : dist.best);
  const spot = Number((state.wb || {}).spot);
  return {
    sourceKey: "risk", sourceLabel: "SPY risk-dial distribution", eligible: n >= 5,
    horizon, scoreKey, asof: risk.asof, direction, pUp,
    probability: direction === "bullish" ? pUp : 1 - pUp,
    terminalProbability: direction === "bullish" ? pUp : 1 - pUp,
    event: "terminal", targetReturn, noTouchReturn, adverseReturn,
    target: spot * (1 + targetReturn), noTouchSpot: spot * (1 + noTouchReturn),
    adverse: spot * (1 + adverseReturn),
    q10: Number(dist.q10 != null ? dist.q10 : dist.worst),
    q25: Number(dist.q25 != null ? dist.q25 : dist.median), median: Number(dist.median),
    q75: Number(dist.q75 != null ? dist.q75 : dist.median),
    q90: Number(dist.q90 != null ? dist.q90 : dist.best), mean: Number(dist.mean),
    effectiveN: n, nSame: null, nOther: null,
    confidence: n >= 20 ? "moderate" : "low",
    firstRejection: n >= 5
      ? "Fragility analogues are declustered historical episodes, not a causal forecast; current catalysts can dominate."
      : "Fewer than five complete similar-fragility episodes.",
    methodology: `${scoreKey} risk-dial analogues, independently sampled; no seasonal data is included.`,
  };
}

function availableThesisSources(ticker) {
  const symbol = String(ticker || "").toUpperCase();
  const out = [];
  if (seasonalHouseThesis(symbol, tradeHorizon())) out.push("seasonal");
  if (symbol === "SPY" && riskHouseThesis(tradeHorizon())) out.push("risk");
  return out;
}

function activeHouseThesis() {
  const wb = state.wb;
  if (!wb) return null;
  if (state.house.source === "risk" && wb.ticker === "SPY") return riskHouseThesis(tradeHorizon());
  return seasonalHouseThesis(wb.ticker, tradeHorizon());
}

function applyHouseThesis(thesis) {
  if (!thesis || !thesis.eligible || !isFinite(thesis.target) || !isFinite(thesis.probability)) return;
  state.house.appliedSource = thesis.sourceKey;
  state.house.appliedHorizon = thesis.horizon;
  state.manual.view = thesis.direction;
  state.params.dir = thesis.direction === "bearish" ? "Short" : "Long";
  state.manual.target = thesis.target;
  state.manual.adverse = thesis.adverse;
  const view = document.getElementById("wbView"); if (view) view.value = thesis.direction;
  const horizon = document.getElementById("wbHorizon");
  if (horizon && !horizon.disabled) { horizon.value = thesis.horizon; state.manual.horizon = thesis.horizon; }
  state.forecast.active = false;
  state.forecast.event = thesis.event;
  state.forecast.target = round2(thesis.target);
  state.forecast.probability = Math.max(0, Math.min(1, thesis.probability));
  state.forecast.cutoff = futureBusinessDate(thesis.horizon);
  state.forecast.noTouchSpot = round2(thesis.noTouchSpot);
  renderAll();
}

function renderHouseThesis() {
  const el = document.getElementById("houseThesis"), wb = state.wb;
  if (!el || !wb) { if (el) el.innerHTML = ""; return; }
  const isSpy = wb.ticker === "SPY";
  if (!isSpy && state.house.source === "risk") state.house.source = "seasonal";
  const seasonal = seasonalHouseThesis(wb.ticker, tradeHorizon());
  const risk = isSpy ? riskHouseThesis(tradeHorizon()) : null;
  if (state.house.source === "seasonal" && !seasonal && risk) state.house.source = "risk";
  if (state.house.source === "risk" && !risk && seasonal) state.house.source = "seasonal";
  const tabs = [
    `<button class="btn xs ${state.house.source === "seasonal" ? "" : "ghost"}" data-house-source="seasonal"${seasonal ? "" : " disabled"}>Seasonal</button>`,
    isSpy ? `<button class="btn xs ${state.house.source === "risk" ? "" : "ghost"}" data-house-source="risk"${risk ? "" : " disabled"}>Risk dial</button>` : "",
  ].join("");
  const thesis = activeHouseThesis();
  if (!thesis) {
    el.innerHTML = `<div class="card opt-house-thesis"><div class="opt-section-head"><div><b>House forward distribution</b><div class="cap">No compatible ${isSpy ? "seasonal or risk-dial" : "seasonal"} history is available for this ticker.</div></div><div>${tabs}</div></div></div>`;
    return;
  }
  if (!thesis.eligible) {
    el.innerHTML = `<div class="card opt-house-thesis"><div class="opt-section-head"><div><b>${esc(thesis.sourceLabel)}</b><div class="cap">${thesis.horizon || "?"} trading-day candidate prior</div></div><div>${tabs}</div></div>
      <div class="opt-risk-warn">${esc(thesis.firstRejection)}</div><div class="cap">Same-cycle observations ${thesis.nSame || 0} · other-cycle observations ${thesis.nOther || 0}. Manual thesis remains available below.</div></div>`;
  } else {
    const sideProbability = thesis.probability;
    const applied = state.house.appliedSource === thesis.sourceKey && state.house.appliedHorizon === thesis.horizon;
    el.innerHTML = `<div class="card opt-house-thesis"><div class="opt-section-head"><div><div class="cap">House forward distribution · candidate prior</div><b>${esc(thesis.sourceLabel)}</b> <span class="badge ${thesis.confidence === "moderate" ? "ok" : "warn"}">${esc(String(thesis.confidence).toUpperCase())} CONFIDENCE</span></div><div class="opt-house-tabs">${tabs}</div></div>
      ${isSpy ? '<div class="opt-source-independence">SPY sources are independent. Selecting Risk dial does not adjust or blend the seasonal distribution.</div>' : ""}
      <div class="opt-house-kpis">
        <div><span>Bias</span><b class="${thesis.direction === "bullish" ? "pos" : "neg"}">${esc(thesis.direction.toUpperCase())}</b><small>${thesis.horizon} trading days</small></div>
        <div><span>Probability up</span><b>${fmt.pct(thesis.pUp, 1)}</b><small>${thesis.sourceKey === "seasonal" ? "shrunk weighted frequency" : thesis.scoreKey + " analogue sample"}</small></div>
        <div><span>${thesis.event === "touch" ? "Target-touch odds" : "Representative-side odds"}</span><b>${fmt.pct(sideProbability, 1)}</b><small>target ${fmt.money(thesis.target)}</small></div>
        <div><span>Median / mean</span><b>${fmt.pct(thesis.median, 1)} / ${fmt.pct(thesis.mean, 1)}</b><small>${thesis.effectiveN ? `effective n ${fmt.num(thesis.effectiveN, 1)}` : ""}</small></div>
      </div>
      <div class="opt-distribution-band"><span>p10 ${fmt.pct(thesis.q10, 1)}</span><span>p25 ${fmt.pct(thesis.q25, 1)}</span><b>p50 ${fmt.pct(thesis.median, 1)}</b><span>p75 ${fmt.pct(thesis.q75, 1)}</span><span>p90 ${fmt.pct(thesis.q90, 1)}</span></div>
      <div class="opt-house-grid"><div><span>Representative target</span><b>${fmt.money(thesis.target)} (${fmt.pct(thesis.targetReturn, 1)})</b></div><div><span>Adverse tail</span><b>${fmt.money(thesis.adverse)} (${fmt.pct(thesis.adverseReturn, 1)})</b></div>
        <div><span>As of</span><b>${esc(thesis.asof || "?")}</b></div><div><span>Method</span><b>${esc(thesis.methodology)}</b></div></div>
      <div class="opt-evidence-bar"><b>First rejection:</b> ${esc(thesis.firstRejection || "Current catalyst and live option pricing still need confirmation.")}</div>
      <div class="opt-house-actions"><button class="btn" id="house_apply">Apply to Forecast Lab</button>${applied ? '<span class="badge ok">APPLIED · STILL EDITABLE</span>' : '<span class="cap">Preview only until applied. This does not place or stage an order.</span>'}</div></div>`;
  }
  el.querySelectorAll("[data-house-source]").forEach((button) => button.addEventListener("click", () => {
    state.house.source = button.dataset.houseSource;
    renderHouseThesis();
  }));
  const apply = document.getElementById("house_apply");
  if (apply) apply.addEventListener("click", () => applyHouseThesis(activeHouseThesis()));
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

/* ---------------- positioning + true calendar pair ---------------- */
function positioningMetrics(wb, marketRow) {
  const rows = (((wb || {}).chain || {}).strikes || []);
  const spot = Number((wb || {}).spot || 0);
  let callOi = 0, putOi = 0, gammaAbs = 0, gammaProxy = 0, oiSeen = 0, gammaSeen = 0;
  const byStrike = {};
  for (const row of rows) {
    const oi = Number(row.oi), gamma = Number(row.gamma), strike = Number(row.strike);
    if (isFinite(oi) && oi >= 0 && row.oi != null) {
      oiSeen += 1;
      if (row.right === "C") callOi += oi; else putOi += oi;
      byStrike[strike] = byStrike[strike] || { oi: 0, gamma: 0 };
      byStrike[strike].oi += oi;
      if (row.gamma != null && isFinite(gamma) && spot > 0) {
        gammaSeen += 1;
        const dollars = Math.abs(gamma) * spot * spot * 0.01 * 100 * oi;
        gammaAbs += dollars;
        gammaProxy += dollars * (row.right === "C" ? 1 : -1);
        byStrike[strike].gamma += dollars;
      }
    }
  }
  const maxStrike = (key) => Object.entries(byStrike).sort((a, b) => b[1][key] - a[1][key])[0];
  const totalOi = oiSeen ? callOi + putOi : marketRow && marketRow.total_oi;
  return {
    source: oiSeen ? "live selected expiry" : marketRow && marketRow.total_oi != null ? "nightly 30/60/90d snapshot" : "unavailable",
    totalOi, putCall: oiSeen && callOi > 0 ? putOi / callOi : marketRow && marketRow.put_call_oi,
    gammaAbs: gammaSeen ? gammaAbs : marketRow && marketRow.gamma_abs_1pct,
    gammaProxy: gammaSeen ? gammaProxy : marketRow && marketRow.gamma_proxy,
    maxOiStrike: oiSeen && maxStrike("oi") ? Number(maxStrike("oi")[0]) : marketRow && marketRow.max_oi_strike,
    maxGammaStrike: gammaSeen && maxStrike("gamma") ? Number(maxStrike("gamma")[0]) : marketRow && marketRow.max_gamma_strike,
    coverage: rows.length ? oiSeen / rows.length : marketRow && marketRow.oi_coverage,
  };
}

function renderPositioningLab() {
  const el = document.getElementById("positioningLab"), wb = state.wb;
  if (!el || !wb) { if (el) el.innerHTML = ""; return; }
  const marketRow = ((state.market || {}).etfs || []).find((row) => row.ticker === wb.ticker) || null;
  const metrics = positioningMetrics(wb, marketRow);
  const seeded = metrics.totalOi != null;
  const skewN = marketRow && marketRow.skew_history_n || 0;
  el.innerHTML = `<div class="card opt-positioning">
    <div class="opt-section-head"><div><b>Skew history & positioning concentration</b><div class="cap">${esc(wb.ticker)} · ${esc(metrics.source)} · inventory map, not dealer-positioning truth</div></div>
      <span class="badge ${seeded ? "ok" : "warn"}">${seeded ? "SNAPSHOT AVAILABLE" : "COLLECTING"}</span></div>
    <div class="opt-compass six">
      <div class="tile"><div class="eyebrow">25Δ RR history</div><div class="reading">${marketRow && marketRow.rr25_pctile != null ? "p" + fmt.num(marketRow.rr25_pctile, 0) : "—"}</div><div class="detail">${skewN} nightly observations; 20 required</div></div>
      <div class="tile"><div class="eyebrow">Put / call OI</div><div class="reading">${metrics.putCall != null ? fmt.num(metrics.putCall, 2) : "—"}</div><div class="detail">OI coverage ${metrics.coverage != null ? fmt.pct(metrics.coverage, 0) : "—"}</div></div>
      <div class="tile"><div class="eyebrow">Total OI sampled</div><div class="reading">${metrics.totalOi != null ? fmt.num(metrics.totalOi, 0) : "—"}</div><div class="detail">30/60/90d strike bands</div></div>
      <div class="tile"><div class="eyebrow">Absolute γ / 1% move</div><div class="reading">${metrics.gammaAbs != null ? fmt.money(metrics.gammaAbs) : "—"}</div><div class="detail">unsigned concentration, not P&amp;L forecast</div></div>
      <div class="tile"><div class="eyebrow">Max γ strike</div><div class="reading">${metrics.maxGammaStrike != null ? fmt.num(metrics.maxGammaStrike, 2) : "—"}</div><div class="detail">largest sampled OI×model-gamma node</div></div>
      <div class="tile"><div class="eyebrow">Call − put γ proxy</div><div class="reading ${metrics.gammaProxy > 0 ? "pos" : metrics.gammaProxy < 0 ? "neg" : ""}">${metrics.gammaProxy != null ? fmt.money(metrics.gammaProxy) : "—"}</div><div class="detail">heuristic sign only; participant inventory is unknown</div></div>
    </div>
    <div class="opt-evidence-bar"><b>First rejection:</b> open interest has no reliable dealer/customer sign. Use this layer to locate concentration and pin/gap risk; do not call it “dealer gamma” without flow data.</div>
  </div>`;
}

function calendarStructureFrom(frontChain, backChain, right, spot) {
  if (!frontChain || !backChain || !(backChain.dte > frontChain.dte)) return null;
  const front = (frontChain.strikes || []).filter((row) => row.right === right && row.mid != null);
  const backByStrike = new Map((backChain.strikes || []).filter((row) => row.right === right && row.mid != null).map((row) => [Number(row.strike), row]));
  const pairs = front.map((row) => [row, backByStrike.get(Number(row.strike))]).filter((pair) => pair[1]);
  if (!pairs.length) return null;
  const [frontRow, backRow] = pairs.reduce((best, pair) =>
    Math.abs(pair[0].strike - spot) < Math.abs(best[0].strike - spot) ? pair : best, pairs[0]);
  const shortLeg = { ...frontRow, expiry: frontChain.expiry, dte: frontChain.dte };
  const longLeg = { ...backRow, expiry: backChain.expiry, dte: backChain.dte };
  return structureFrom(`Long ${right === "C" ? "call" : "put"} calendar`, [
    { side: "SELL", row: shortLeg }, { side: "BUY", row: longLeg },
  ], `same-strike ${expLabel(frontChain.expiry)} → ${expLabel(backChain.expiry)}; front theta funds back vega`, { category: "calendar" });
}

async function requestWorkbenchChain(ticker, expiry) {
  const response = await fetch("/exec-workbench", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticker, mode: "chain", expiry, max_expiries: 2, context: null }) });
  const accepted = await response.json();
  if (!accepted.ok) throw new Error(accepted.error || `HTTP ${response.status}`);
  for (let attempt = 0; attempt <= 40; attempt++) {
    const status = (await fetchJSONOrNull(`/exec-workbench?id=${encodeURIComponent(accepted.id)}`)) || {};
    const query = status.query;
    if (query && query.id === accepted.id && query.result) {
      if (query.result.error) throw new Error(query.result.error);
      if (!query.result.chain) throw new Error(query.result.chain_error || `no chain for ${expiry}`);
      return query.result.chain;
    }
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
  throw new Error(`timed out quoting ${expiry}`);
}

async function loadCalendarPair() {
  const wb = state.wb, calendar = state.calendar;
  if (!wb || !calendar.front || !calendar.back || calendar.back <= calendar.front) return;
  calendar.loading = true; calendar.error = null; renderCalendarBuilder();
  try {
    calendar.frontChain = wb.chain && wb.chain.expiry === calendar.front ? wb.chain : await requestWorkbenchChain(wb.ticker, calendar.front);
    calendar.backChain = wb.chain && wb.chain.expiry === calendar.back ? wb.chain : await requestWorkbenchChain(wb.ticker, calendar.back);
  } catch (error) {
    calendar.error = String(error.message || error);
  } finally {
    calendar.loading = false;
    buildStructures(); renderCalendarBuilder(); renderForecastLab(); renderShootout(); renderScenarioLab(); renderSizing(); renderTicket();
  }
}

function renderCalendarBuilder() {
  const el = document.getElementById("calendarBuilder"), wb = state.wb;
  if (!el || !wb || !(wb.expiries || []).length) { if (el) el.innerHTML = ""; return; }
  const expiries = (wb.expiries || []).filter((row) => row.atm_iv > 0 && row.dte > 5).slice().sort((a, b) => a.dte - b.dte);
  if (expiries.length < 2) { el.innerHTML = ""; return; }
  const nearest = (target, after=0) => {
    const pool = expiries.filter((row) => row.dte > after);
    return pool.length ? pool.reduce((best, row) => Math.abs(row.dte - target) < Math.abs(best.dte - target) ? row : best, pool[0]) : null;
  };
  if (!state.calendar.front) state.calendar.front = nearest(30).date;
  const frontRow = expiries.find((row) => row.date === state.calendar.front) || nearest(30);
  const defaultBack = nearest(Math.max(60, frontRow.dte + 25), frontRow.dte);
  if ((!state.calendar.back || state.calendar.back <= frontRow.date) && defaultBack) state.calendar.back = defaultBack.date;
  const backRow = expiries.find((row) => row.date === state.calendar.back);
  const fwd = backRow ? forwardVol(frontRow.atm_iv, frontRow.dte, backRow.atm_iv, backRow.dte) : null;
  const options = (selected) => expiries.map((row) => `<option value="${row.date}"${row.date === selected ? " selected" : ""}>${expLabel(row.date)} · ${row.dte}d · IV ${(row.atm_iv * 100).toFixed(1)}%</option>`).join("");
  const structure = calendarStructureFrom(state.calendar.frontChain, state.calendar.backChain,
    ["bearish", "hedge"].includes(currentView()) ? "P" : "C", wb.spot);
  el.innerHTML = `<div class="card opt-calendar-builder"><div class="opt-section-head"><div><b>True calendar builder</b><div class="cap">Quotes two expiries and creates a same-strike diagonal-in-time BAG candidate</div></div>
      <span class="badge ${structure ? "ok" : "warn"}">${structure ? "PAIR READY" : state.calendar.loading ? "QUOTING" : "NEEDS TWO CHAINS"}</span></div>
    <div class="opt-calendar-controls"><label><span>Front expiry</span><select id="cal_front">${options(state.calendar.front)}</select></label>
      <label><span>Back expiry</span><select id="cal_back">${options(state.calendar.back)}</select></label>
      <button class="btn" id="cal_quote"${state.calendar.loading ? " disabled" : ""}>${state.calendar.loading ? "Quoting…" : "Quote both expiries"}</button>
      <div class="cap">${backRow ? `Front/back IV ${fmt.pctRaw(frontRow.atm_iv * 100, 1)} / ${fmt.pctRaw(backRow.atm_iv * 100, 1)} · implied forward ${fwd != null ? fmt.pctRaw(fwd * 100, 1) : "invalid variance"}` : "Back expiry must follow the front."}</div></div>
    ${state.calendar.error ? `<div class="opt-risk-warn">${esc(state.calendar.error)}</div>` : ""}
    ${structure ? `<div class="opt-evidence-bar"><b>Added to Structure Shootout:</b> ${esc(structure.name)} at ${fmt.num(structure.mid, 2)} debit. Each leg retains its own expiry and time-to-expiry in scenario repricing.</div>` :
      '<div class="cap" style="margin-top:8px">Long calendars are supported. Short calendars remain screen-only until the execution layer has an explicit margin and tail-loss model.</div>'}
  </div>`;
  ["cal_front", "cal_back"].forEach((id) => document.getElementById(id).addEventListener("change", () => {
    state.calendar.front = document.getElementById("cal_front").value;
    state.calendar.back = document.getElementById("cal_back").value;
    state.calendar.frontChain = null; state.calendar.backChain = null; state.calendar.error = null;
    renderCalendarBuilder(); buildStructures(); renderShootout(); renderScenarioLab(); renderSizing(); renderTicket();
  }));
  document.getElementById("cal_quote").addEventListener("click", loadCalendarPair);
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
    const earnFlag = earnDate && earnDate.replace(/-/g, "") <= e.date ? " [WARN]" : "";
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
      <span class="cap" style="display:inline;font-weight:400">· default = first listed &ge; time-exit + 5 td · [WARN] = earnings before expiry</span></div>
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
function localStrikeStep(rows, px) {
  const strikes = [...new Set(rows.map((r) => Number(r.strike)).filter((x) => isFinite(x)))].sort((a, b) => a - b);
  if (strikes.length < 2) return null;
  let idx = 0;
  for (let i = 1; i < strikes.length; i++) {
    if (Math.abs(strikes[i] - px) < Math.abs(strikes[idx] - px)) idx = i;
  }
  const gaps = [];
  if (idx > 0) gaps.push(strikes[idx] - strikes[idx - 1]);
  if (idx < strikes.length - 1) gaps.push(strikes[idx + 1] - strikes[idx]);
  return gaps.filter((x) => x > 0).sort((a, b) => a - b)[0] || null;
}
function targetVerticalRows(rows, spot, target, right, moveFraction) {
  const shortRow = byStrike(rows, target);
  if (!shortRow || !isFinite(spot) || !isFinite(target)) return null;
  const step = localStrikeStep(rows, shortRow.strike) || 0;
  const width = Math.max(step, Math.abs(spot - target) * moveFraction);
  const longPx = shortRow.strike + (right === "P" ? width : -width);
  const validLongs = rows.filter((r) => right === "P" ? r.strike > shortRow.strike : r.strike < shortRow.strike);
  const longRow = byStrike(validLongs, longPx);
  return longRow ? { longRow, shortRow } : null;
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
  if (shortRow && right === "P" && longRow.strike <= shortRow.strike) return null;
  if (shortRow && right === "C" && longRow.strike >= shortRow.strike) return null;
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
    if (state.forecast.active && tgtPx) {
      const tight = targetVerticalRows(rows, spot, tgtPx, right, 0.25);
      const balanced = targetVerticalRows(rows, spot, tgtPx, right, 0.50);
      add(tight && spreadFrom("Target-zone tight vertical", tight.longRow, tight.shortRow, right,
        "both strikes near the forecast level; long strike sits 25% of the forecast move back toward spot"));
      add(balanced && spreadFrom("Target-zone balanced vertical", balanced.longRow, balanced.shortRow, right,
        "short strike nearest the target; long strike sits halfway back toward spot"));
      add(spreadFrom("Spot-to-target vertical", atmLong, anchorShort, right,
        "full-move reference: long near spot, short strike nearest the forecast target"));
    } else {
      add(spreadFrom("Target-anchored vertical", anchorLong, anchorShort, right,
        tgtPx ? "short strike nearest the thesis target" : "long near entry / short ~25Δ"));
    }
    add(spreadFrom("50Δ/25Δ reference vertical", d50 || atmLong, d25, right,
      "delta-based baseline for comparison; forecast ranking can choose a tighter target-zone spread"));
    add(spreadFrom(`Long ${right === "C" ? "call" : "put"} ~40Δ`, d40, null, right,
      "uncapped convexity, full theta and vega"));
    add(creditVertical(short ? "C" : "P", short ? "Bear call spread 30Δ/15Δ" : "Bull put spread 30Δ/15Δ"));
  }
  if (state.calendar.frontChain && state.calendar.backChain) {
    add(calendarStructureFrom(state.calendar.frontChain, state.calendar.backChain,
      ["bearish", "hedge"].includes(view) ? "P" : "C", spot));
  }
  if (!state.structures.some((s) => s.tradeable)) return;
  if (!state.selStructure || !state.structures.some((s) => s.name === state.selStructure)) {
    state.selStructure = state.structures[0].name;
  }
}
function sameLegs(a, b) {
  const key = (s) => s.legs.map((l) => `${l.side}${l.row.strike}${l.row.right}${l.row.expiry || ""}`).join("|");
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
    const dte = l.row.dte != null ? Number(l.row.dte) : (((state.wb || {}).chain || {}).dte || 0);
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
  const expiry = [((wb.chain || {}).expiry || ""), state.calendar.back || ""].sort().pop();
  const covers = !!expiry && expiry >= cutoffCompact;
  const activeRows = f.active && covers ? state.structures.map((s) => {
    const qty = contractsFor(s, sizeAlloc());
    const metrics = qty > 0 ? forecastMetrics(s, qty, f) : null;
    return { s, qty, metrics, score: forecastScore(metrics, f.objective) };
  }).filter((x) => x.metrics && isFinite(x.score)).sort((a, b) => b.score - a.score) : [];
  if (activeRows.length && f.selectTop) {
    state.selStructure = activeRows[0].s.name;
    f.selectTop = false;
  }

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
    f.selectTop = true;
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
  return `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${c}"></span> ${pct.toFixed(1)}%`;
}
function breakevenPct(struct) {
  if (!struct.legs.length || ["straddle", "strangle", "iron_condor", "calendar"].includes(struct.category)) return null;
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
    const legsTxt = s.legs.map((l) => `${l.side[0]} ${l.row.strike}${l.row.right}${l.row.expiry ? ` ${expLabel(l.row.expiry)}` : ""}${l.row.delta != null ? ` (${Math.abs(l.row.delta).toFixed(2)}Δ)` : ""}`).join(" / ");
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
    `${l.side} ${l.row.strike}${l.row.right} ${l.row.expiry || wb.chain.expiry} (conId ${l.row.con_id || "?"})`).join("<br>");
  el.innerHTML = `<div class="card" style="max-width:860px;margin-bottom:12px">
    <div style="font:700 14px inherit;margin-bottom:4px">${ticketKind} — ${esc(s.name)}</div>
    <p class="cap" style="margin:0 0 8px">${s.legs.length === 1 ? "One SMART-routed option limit order." : "One native SMART BAG limit order (atomic — never legs you in)."}
      The execution agent independently re-validates contract identity, quantity, price, and defined max loss. Primary has no hard quantity or risk cap; PA options remain disabled.</p>
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
  return m === "dry-run" ? `Dry-run ${verb} (places nothing):` : `[WARN] LIVE — really ${verb}`;
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
    right: l.row.right, expiry: l.row.expiry || expiry, strike: l.row.strike, ratio: 1, con_id: l.row.con_id || null,
  }));
}

function sendOptionOrder() {
  const s = selStruct();
  const wb = state.wb, p = state.params;
  const msg = document.getElementById("tk_msg");
  if (!s || !wb) return;
  if (state.account !== "primary") { msg.textContent = "BLOCKED: options execution remains disabled for PA"; return; }
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
  const expiryLabel = [...new Set(s.legs.map((l) => l.row.expiry || wb.chain.expiry))].join("/");
  const riskDollars = riskPremium * 100 * qty + commRT(s) * qty;
  if (!confirm(`${actionLead("place")} ${action} ${qty}x ${wb.ticker} ${expiryLabel} [${legsTxt}] LMT ${payload.limit} ${s.credit ? "credit" : "debit"} on ${state.account}?\n\nDefined max risk: about ${fmt.money(riskDollars)}.`)) return;
  const accountRow = (((state.book || {}).accounts) || []).find((a) => a.key === state.account);
  const nlv = Number(accountRow && accountRow.nlv);
  if (!(nlv > 0)) {
    if (!confirm(`SECONDARY RISK APPROVAL\n\nThis Primary options order is uncapped and current NLV is unavailable. Defined max loss is about ${fmt.money(riskDollars)}. Really continue?`)) return;
    payload.risk_ack = true;
  } else if (riskDollars > nlv * 0.05) {
    if (!confirm(`SECONDARY RISK APPROVAL\n\nThis Primary options order has defined max loss of about ${fmt.money(riskDollars)}, or ${(riskDollars / nlv * 100).toFixed(1)}% of NLV. There is no hard size cap. Really continue?`)) return;
    payload.risk_ack = true;
  }
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
      [WARN] LIVE ARMED — an options order sent here transmits to IBKR if option_spread is in LIVE_TYPES.</div>`;
  } else if (mode === "unknown") {
    el.innerHTML = `<div class="card" style="border-color:#a8852f;background:rgba(255,193,77,.10);padding:9px 14px;font:700 13px inherit;color:#ffc14d">
      [WARN] MODE UNKNOWN — assume LIVE. No fresh book confirms dry-run.</div>`;
  } else {
    el.innerHTML = `<div class="card" style="border-color:#2c8f63;background:rgba(61,219,143,.08);padding:9px 14px;font:700 13px inherit;color:#3ddb8f">
      [DRY-RUN] Options orders are validated + previewed, nothing transmits.</div>`;
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
