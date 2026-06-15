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
    '<div class="viewer" id="viewer"></div></div>';
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
}

function renderList() {
  const el = document.getElementById("tradelist");
  document.getElementById("galCount").textContent = `${VIEW.length} trades`;
  if (!VIEW.length) {
    el.innerHTML = '<div class="cap" style="padding:12px">No trades match.</div>';
    return;
  }
  el.innerHTML = VIEW.map((r, i) => {
    const rc = r.r == null ? "neu" : r.r > 0 ? "pos" : "neg";
    return `<div class="tl-row${i === IDX ? " on" : ""}" data-i="${i}">
      <span class="tk">${esc(r.ticker)}</span>
      <span class="dt">${fmt.date(r.signal_date)}</span>
      <span class="r ${rc}">${r.r == null ? "" : fmt.signed(r.r, 1) + "R"}</span>
    </div>`;
  }).join("");
  el.querySelectorAll(".tl-row").forEach(row =>
    row.addEventListener("click", () => {
      IDX = +row.dataset.i; renderList(); renderViewer(); scrollActive();
    }));
}

function renderViewer() {
  const el = document.getElementById("viewer");
  if (!VIEW.length) {
    el.innerHTML = '<div class="vimg"><div class="ph">No chart selected.</div></div>';
    return;
  }
  const r = VIEW[IDX];
  const rc = r.r == null ? "neu" : r.r > 0 ? "pos" : "neg";
  const trunc = r.post_short ? ' <span class="note">post-window &lt; 3mo</span>' : "";
  el.innerHTML = `
    <div class="vcap">
      <b>${esc(r.ticker)}</b>
      <span class="badge ${r.direction === "Short" ? "dirS" : "dirL"}">${esc(String(r.direction).toUpperCase())}</span>
      <span class="badge ${r.tier === "Overflow" ? "warn" : "conv"}">${esc(r.tier)}</span>
      <span class="muted">signal ${fmt.date(r.signal_date)} &rarr; exit ${fmt.date(r.exit_date)} (${esc(r.exit_type)})</span>
      &nbsp; R <b class="${rc}">${r.r == null ? "?" : fmt.signed(r.r, 2)}</b>
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

function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
