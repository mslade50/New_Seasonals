/* charts.js — per-trade signal-chart gallery.

   Pick a strategy, flip through each trade's chart (6mo pre-signal -> trade ->
   3mo post-exit) with the list, the Prev/Next buttons, or the arrow keys. The
   PNGs stream from R2 via the /charts/ Pages Function; this file only drives
   selection off data/charts.json. */
"use strict";

document.addEventListener("DOMContentLoaded", init);

let ROWS = [];   // all chart records
let VIEW = [];   // current filtered + sorted subset
let IDX = 0;     // index into VIEW
const state = { strat: null, tier: "All", sort: "new", q: "" };

async function init() {
  renderNav("charts.html");
  const el = document.getElementById("content");
  const [meta, charts] = await Promise.all([
    fetchJSONOrNull("data/meta.json"),
    fetchJSONOrNull("data/charts.json"),
  ]);
  if (meta) setAsof(`built ${meta.built_at}`);
  if (!charts) {
    el.innerHTML = '<p class="cap">No charts payload in this build.</p>';
    return;
  }
  ROWS = rowsFromColumnar(charts);
  const strats = [...new Set(ROWS.map(r => r.strategy))].sort();
  state.strat = strats[0];
  el.innerHTML = controlsHtml(strats) +
    '<div class="gallery"><div class="tradelist" id="tradelist"></div>' +
    '<div class="viewer" id="viewer"></div></div>' +
    '<div id="effSection"></div>';
  wireControls();
  refilter();
  document.addEventListener("keydown", onKey);
}

function controlsHtml(strats) {
  const opts = ['<option value="All">All strategies</option>']
    .concat(strats.map(s => `<option value="${esc(s)}">${esc(s)}</option>`)).join("");
  return `<div class="filters">
    <label>Strategy</label>
    <select id="selStrat">${opts}</select>
    <div class="seg" id="segTier">
      <button data-v="All" class="on">All</button>
      <button data-v="Liquid">Liquid</button>
      <button data-v="Overflow">Overflow</button>
    </div>
    <div class="seg" id="segSort">
      <button data-v="new" class="on">Newest</button>
      <button data-v="old">Oldest</button>
      <button data-v="rbest">Best R</button>
      <button data-v="rworst">Worst R</button>
    </div>
    <input type="text" id="qTicker" placeholder="Filter ticker...">
    <div class="info" id="galCount"></div>
  </div>`;
}

function wireControls() {
  const sel = document.getElementById("selStrat");
  sel.value = state.strat;
  sel.addEventListener("change", () => { state.strat = sel.value; refilter(); });
  segWire("segTier", v => { state.tier = v; refilter(); });
  segWire("segSort", v => { state.sort = v; refilter(); });
  const q = document.getElementById("qTicker");
  q.addEventListener("input", () => { state.q = q.value.trim().toUpperCase(); refilter(); });
}

function segWire(id, cb) {
  const seg = document.getElementById(id);
  seg.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
    seg.querySelectorAll("button").forEach(x => x.classList.remove("on"));
    b.classList.add("on");
    cb(b.dataset.v);
  }));
}

function refilter() {
  VIEW = ROWS.filter(r =>
    (state.strat === "All" || r.strategy === state.strat) &&
    (state.tier === "All" || r.tier === state.tier) &&
    (!state.q || String(r.ticker).toUpperCase().includes(state.q)));
  const cmp = {
    new: (a, b) => (a.signal_date < b.signal_date ? 1 : a.signal_date > b.signal_date ? -1 : 0),
    old: (a, b) => (a.signal_date > b.signal_date ? 1 : a.signal_date < b.signal_date ? -1 : 0),
    rbest: (a, b) => (b.r == null ? -1e9 : b.r) - (a.r == null ? -1e9 : a.r),
    rworst: (a, b) => (a.r == null ? 1e9 : a.r) - (b.r == null ? 1e9 : b.r),
  }[state.sort];
  VIEW.sort(cmp);
  IDX = 0;
  renderList();
  renderViewer();
  scheduleEfficiency();
}

// The efficiency section rebuilds two Plotly plots (one WebGL) — debounce so
// per-keystroke ticker filtering stays as light as the pre-existing list render.
let effTimer = null;
function scheduleEfficiency() {
  clearTimeout(effTimer);
  effTimer = setTimeout(renderEfficiency, 250);
}

function renderList() {
  const el = document.getElementById("tradelist");
  document.getElementById("galCount").textContent = `${VIEW.length} trades`;
  if (!VIEW.length) {
    el.innerHTML = '<div class="cap" style="padding:12px">No trades match.</div>';
    return;
  }
  const head = `<div class="tl-row tl-head">
    <span class="tk">Tkr</span><span class="dt">Signal</span>
    <span class="xt" title="Exit type">Exit</span>
    <span class="r" title="Normalized R (1 unit of risk)">R</span>
    <span class="ar" title="Actual return = R x sizing multiplier">Act</span>
    <span class="dd" title="Max drawdown in-trade (MAE), R">DD</span></div>`;
  el.innerHTML = head + VIEW.map((r, i) => {
    const rc = r.r == null ? "neu" : r.r > 0 ? "pos" : "neg";
    const arc = r.actual_r == null ? "neu" : r.actual_r > 0 ? "pos" : "neg";
    const ds = r.size_mult != null && r.size_mult < 0.999;
    const dsTitle = ds ? ` title="sized ${Math.round(r.size_mult * 100)}% of full"` : "";
    return `<div class="tl-row${i === IDX ? " on" : ""}" data-i="${i}">
      <span class="tk">${esc(r.ticker)}</span>
      <span class="dt">${fmt.date(r.signal_date)}</span>
      <span class="xt" title="${esc(r.exit_type)}">${esc(exitAbbr(r.exit_type))}</span>
      <span class="r ${rc}">${r.r == null ? "" : fmt.signed(r.r, 1)}</span>
      <span class="ar ${arc}"${dsTitle}>${r.actual_r == null ? "" : fmt.signed(r.actual_r, 1) + (ds ? "*" : "")}</span>
      <span class="dd neg">${r.mae_r == null ? "" : fmt.signed(r.mae_r, 1)}</span>
    </div>`;
  }).join("");
  el.querySelectorAll(".tl-row[data-i]").forEach(row =>
    row.addEventListener("click", () => {
      IDX = +row.dataset.i; renderList(); renderViewer(); scrollActive();
    }));
}

function exitAbbr(t) {
  return ({ Stop: "Stop", Target: "Tgt", Time: "Time", "EOD-DD": "EOD", SignalDeact: "Sig" })[t]
    || String(t || "").slice(0, 4);
}

function renderViewer() {
  const el = document.getElementById("viewer");
  if (!VIEW.length) {
    el.innerHTML = '<div class="vimg"><div class="ph">No chart selected.</div></div>';
    return;
  }
  const r = VIEW[IDX];
  const rc = r.r == null ? "neu" : r.r > 0 ? "pos" : "neg";
  const arc = r.actual_r == null ? "neu" : r.actual_r > 0 ? "pos" : "neg";
  const ds = r.size_mult != null && r.size_mult < 0.999;
  const trunc = r.post_short ? ' <span class="note">post-window &lt; 3mo</span>' : "";
  el.innerHTML = `
    <div class="vcap">
      <b>${esc(r.ticker)}</b>
      <span class="badge ${r.direction === "Short" ? "dirS" : "dirL"}">${esc(String(r.direction).toUpperCase())}</span>
      <span class="badge ${r.tier === "Overflow" ? "warn" : "conv"}">${esc(r.tier)}</span>
      <span class="muted">signal ${fmt.date(r.signal_date)} &rarr; exit ${fmt.date(r.exit_date)} (${esc(r.exit_type)})</span>
      &nbsp; R <b class="${rc}">${r.r == null ? "?" : fmt.signed(r.r, 2)}</b>
      ${ds ? `&nbsp; actual <b class="${arc}">${fmt.signed(r.actual_r, 2)}R</b> <span class="muted">(${Math.round(r.size_mult * 100)}% size)</span>` : ""}
      &nbsp; ret <b class="${rc}">${fmt.signed(r.ret, 2)}%</b>
      &nbsp; <span class="muted">MFE ${fmt.signed(r.mfe_r, 2)}R / MAE ${fmt.signed(r.mae_r, 2)}R</span>${trunc}
    </div>
    <div class="vimg" id="vimg"><div class="ph">Loading chart...</div></div>
    <div class="vnav">
      <button class="btn" id="btnPrev">&larr; Prev</button>
      <span class="idx">${IDX + 1} / ${VIEW.length}</span>
      <button class="btn" id="btnNext">Next &rarr;</button>
    </div>`;
  const img = new Image();
  img.alt = `${r.ticker} ${r.signal_date}`;
  img.onload = () => {
    const c = document.getElementById("vimg");
    if (c) { c.innerHTML = ""; c.appendChild(img); }
  };
  img.onerror = () => {
    const c = document.getElementById("vimg");
    if (c) c.innerHTML = '<div class="ph">Chart not generated yet (appears after the next site build).</div>';
  };
  img.src = r.path;
  document.getElementById("btnPrev").addEventListener("click", () => step(-1));
  document.getElementById("btnNext").addEventListener("click", () => step(1));
  preloadNeighbors();
}

function step(d) {
  if (!VIEW.length) return;
  IDX = (IDX + d + VIEW.length) % VIEW.length;
  renderList(); renderViewer(); scrollActive();
}

function onKey(e) {
  const t = e.target.tagName;
  if (t === "INPUT" || t === "SELECT" || t === "TEXTAREA") return;
  if (e.key === "ArrowLeft") { step(-1); e.preventDefault(); }
  else if (e.key === "ArrowRight") { step(1); e.preventDefault(); }
}

function scrollActive() {
  const a = document.querySelector(".tl-row.on");
  if (a) a.scrollIntoView({ block: "nearest" });
}

function preloadNeighbors() {
  for (const d of [1, -1]) {
    const r = VIEW[(IDX + d + VIEW.length) % VIEW.length];
    if (r) { const im = new Image(); im.src = r.path; }
  }
}

/* ---------- Exit Efficiency (MFE capture) ----------
   Aggregates the geometry already shipped in charts.json (mfe_r / mae_r /
   exit_type / r) for the CURRENT filter selection. Geometry-only reads:
   MFE/MAE use full-bar highs/lows (entry-day bar included, which flatters MFE
   for limit entries), and the R denominator is the conventional stop_atr x ATR
   unit even for strategies that trade without a live stop. No post-exit or
   stop-survival panels here — charts.json carries no post-exit-direction or
   live-stop fields (post_short is only a chart-window truncation flag). */

const EXIT_COLORS = {
  Time: "#4da3ff", Target: "#00d18f", Stop: "#ff5d5d",
  "EOD-DD": "#ffc14d", SignalDeact: "#b07cff",
};
const CAPTURE_MIN_MFE = 0.25; // R — guard tiny denominators in the capture ratio

function renderEfficiency() {
  const el = document.getElementById("effSection");
  if (!el) return;
  if (typeof Plotly !== "undefined") {
    // release prior WebGL contexts before innerHTML wipes the plot divs
    el.querySelectorAll(".chart").forEach(d => { try { Plotly.purge(d); } catch (e) {} });
  }
  if (typeof Plotly === "undefined") {
    el.innerHTML = '<p class="cap">Plotly failed to load — exit-efficiency charts unavailable.</p>';
    return;
  }
  // Open trades are booked Exit Type=='Time' at the last bar the engine saw
  // (never a future exit_date), so openness comes from the payload's open
  // flag — a date comparison can't detect them.
  const rows = VIEW.filter(r => r.mfe_r != null && r.r != null && !r.open);
  const scope = state.strat === "All" ? "All strategies" : state.strat;

  let html = `<h2 style="margin-top:22px">Exit efficiency — MFE capture (${esc(scope)})</h2>
    <p class="cap">${rows.length.toLocaleString()} closed trades with geometry / ${VIEW.length.toLocaleString()}
      in the current filter. Capture ratio = realized R / MFE R, computed only where
      MFE &ge; ${CAPTURE_MIN_MFE}R. Geometry-only read: full-bar MFE (entry day included)
      is an upper bound on what any exit rule could have banked — a real answer needs an engine sweep.</p>`;

  if (rows.length < 5) {
    el.innerHTML = html + '<p class="cap">Not enough closed trades in this selection.</p>';
    return;
  }

  html += `<div class="grid2">
      <div class="card"><div class="cap" style="margin-top:0">MFE vs realized R, by exit type
        (dotted line = full capture)</div><div class="chart" id="effScatter"></div></div>
      <div class="card"><div class="cap" style="margin-top:0">Capture-ratio distribution
        (MFE &ge; ${CAPTURE_MIN_MFE}R)</div><div class="chart" id="effHist"></div></div>
    </div>
    <div class="card" style="margin-top:14px">
      <div class="cap" style="margin-top:0">Money left on the table, by exit type (R units)</div>
      <div id="effTable"></div>
    </div>`;
  el.innerHTML = html;

  // scatter: one trace per exit type
  const types = [...new Set(rows.map(r => r.exit_type))].sort();
  const scatter = types.map(t => {
    const sub = rows.filter(r => r.exit_type === t);
    return {
      x: sub.map(r => r.mfe_r), y: sub.map(r => r.r),
      name: `${t} (${sub.length})`, mode: "markers", type: "scattergl",
      marker: { color: EXIT_COLORS[t] || "#8fd3ff", size: 5, opacity: 0.55 },
      text: sub.map(r => `${r.ticker} ${fmt.date(r.signal_date)}`),
      hovertemplate: "%{text}<br>MFE %{x:.2f}R &rarr; realized %{y:.2f}R<extra>" + t + "</extra>",
    };
  });
  const maxMfe = Math.max(1, ...rows.map(r => r.mfe_r));
  scatter.push({
    x: [0, maxMfe], y: [0, maxMfe], mode: "lines", name: "Full capture",
    line: { color: "#5a6478", width: 1, dash: "dot" }, hoverinfo: "skip",
  });
  Plotly.newPlot(document.getElementById("effScatter"), scatter, plotLayout({
    height: 320, hovermode: "closest",
    xaxis: { title: { text: "MFE (R)", font: { size: 11 } } },
    yaxis: { title: { text: "Realized R", font: { size: 11 } }, zerolinecolor: "#3a4356" },
  }), PLOT_CFG);

  // capture-ratio histogram
  const captures = rows.filter(r => r.mfe_r >= CAPTURE_MIN_MFE)
    .map(r => Math.max(-3, Math.min(1.05, r.r / r.mfe_r)));
  Plotly.newPlot(document.getElementById("effHist"), [{
    x: captures, type: "histogram", marker: { color: "#4da3ff", opacity: 0.85 },
    xbins: { start: -3, end: 1.1, size: 0.1 },
    hovertemplate: "capture %{x}<br>n=%{y}<extra></extra>",
  }], plotLayout({
    height: 320, bargap: 0.05,
    xaxis: { title: { text: "Capture ratio (clipped to [-3, 1])", font: { size: 11 } } },
    yaxis: { title: { text: "Trades", font: { size: 11 } } },
  }), PLOT_CFG);

  // money-left-on-table summary by exit type
  const groups = types.map(t => [t, rows.filter(r => r.exit_type === t)])
    .concat([["All", rows]]);
  const trs = groups.map(([t, sub]) => {
    if (!sub.length) return "";
    const avg = a => a.reduce((s, v) => s + v, 0) / a.length;
    const med = a => {
      if (!a.length) return null;
      const s = a.slice().sort((x, y) => x - y);
      const m = s.length >> 1;
      return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
    };
    const avgMfe = avg(sub.map(r => r.mfe_r));
    const avgR = avg(sub.map(r => r.r));
    const left = avgMfe - avgR;
    const capMed = med(sub.filter(r => r.mfe_r >= CAPTURE_MIN_MFE).map(r => r.r / r.mfe_r));
    const win = sub.filter(r => r.r > 0).length / sub.length;
    return `<tr${t === "All" ? ' style="font-weight:650"' : ""}>
      <td class="l">${esc(t)}</td><td>${sub.length.toLocaleString()}</td>
      <td>${fmt.signed(avgMfe, 2)}</td>
      <td class="${avgR > 0 ? "pos" : avgR < 0 ? "neg" : ""}">${fmt.signed(avgR, 2)}</td>
      <td class="${left > 0.5 ? "neg" : ""}">${fmt.signed(left, 2)}</td>
      <td>${capMed == null ? "" : fmt.num(capMed, 2)}</td>
      <td>${fmt.pct(win, 0)}</td></tr>`;
  }).join("");
  document.getElementById("effTable").innerHTML = `<div class="tblwrap"><table class="tbl">
    <thead><tr><th class="l">Exit type</th><th>N</th><th>Avg MFE (R)</th><th>Avg realized (R)</th>
      <th>Avg left on table (R)</th><th>Median capture</th><th>Win %</th></tr></thead>
    <tbody>${trs}</tbody></table></div>`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
