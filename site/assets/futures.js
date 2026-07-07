/* futures.js — Futures Lab: the strategy book replayed on non-equity history.
   Payload: /research/futures_lab.json (one-off, committed static — NOT nightly). */
"use strict";

document.addEventListener("DOMContentLoaded", init);

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function init() {
  renderNav("futures.html");
  const el = document.getElementById("content");
  const d = await fetchJSONOrNull("research/futures_lab.json");
  if (!d) {
    el.innerHTML = '<p class="cap">No futures-lab payload in this build yet ' +
      "(run scratch/futures_lab_build.py and commit site/research/futures_lab.json).</p>";
    return;
  }
  setAsof(`computed ${d.computed_asof}`);

  let html = `
    <div class="card" style="border:1px solid var(--amber,#ffc14d);margin-bottom:14px">
      <strong style="color:var(--amber,#ffc14d)">RESEARCH ONLY — NOT TRADEABLE.</strong>
      <span class="cap">Roll-distorted continuous contracts; liquidity floors removed;
      futures left the tradeable universe 2026-07-06. Computed once as of
      <strong>${esc(d.computed_asof)}</strong> (backtest from ${esc(d.bt_start)},
      ${esc(d.basis)} basis) — not rebuilt nightly.</span>
    </div>`;

  for (const key of ["futures", "fx_crypto"]) {
    const u = d.universes && d.universes[key];
    if (!u) continue;
    html += universeSection(key, u);
  }

  html += `<div class="card"><h3>Caveats</h3><ul class="cap">` +
    (d.caveats || []).map(c => `<li>${esc(c)}</li>`).join("") + "</ul></div>";

  el.innerHTML = html;

  for (const key of ["futures", "fx_crypto"]) {
    const u = d.universes && d.universes[key];
    if (!u) continue;
    renderTables(key, u);
    renderCumR(key, u);
  }
}

function universeSection(key, u) {
  const traded = (u.summary_by_strategy || []).filter(s => s.n > 0);
  const totR = traded.reduce((a, s) => a + (s.tot_r || 0), 0);
  const totPnl = traded.reduce((a, s) => a + (s.pnl_flat || 0), 0);
  const wins = rowsFromColumnar(u.trades).filter(t => (t.pnl_flat || 0) > 0).length;
  const n = u.n_trades || 0;
  return `
    <h2>${esc(u.label)} <span class="cap">(${(u.tickers || []).length} tickers)</span></h2>
    <div class="kpis">
      ${kpi("Signals", (u.n_signals || 0).toLocaleString())}
      ${kpi("Trades", n.toLocaleString())}
      ${kpi("Total R", fmt.signed(totR, 1), clsSign(totR))}
      ${kpi("Avg R / trade", n ? fmt.signed(totR / n, 3) : "—", clsSign(totR))}
      ${kpi("Win %", n ? fmt.pctRaw(100 * wins / n, 1) : "—")}
      ${kpi("PnL (flat $750k)", fmt.money(totPnl), clsSign(totPnl))}
      ${kpi("Zero-share drops", String(u.n_zero_share_dropped || 0))}
    </div>
    <div class="grid2">
      <div class="card"><h3>Summary by strategy</h3><div id="sum_${key}"></div></div>
      <div class="card"><h3>Cumulative R by strategy (top by |total R|)</h3>
        <div id="cum_${key}" style="height:320px"></div></div>
    </div>
    <div class="card"><h3>Trades</h3><div id="trd_${key}"></div></div>`;
}

function kpi(label, value, cls) {
  return `<div class="kpi"><div class="l">${esc(label)}</div>
    <div class="v ${cls || ""}">${value}</div></div>`;
}

function renderTables(key, u) {
  makeTable(document.getElementById(`sum_${key}`), {
    columns: [
      { key: "strategy", label: "Strategy", align: "l" },
      { key: "direction", label: "Dir", align: "l" },
      { key: "n_signals", label: "Signals" },
      { key: "n", label: "Trades" },
      { key: "tot_r", label: "Tot R", fmt: v => fmt.signed(v, 1), cls: clsSign },
      { key: "avg_r", label: "Avg R", fmt: v => v == null ? "—" : fmt.signed(v, 3), cls: clsSign },
      { key: "win_pct", label: "Win %", fmt: v => v == null ? "—" : fmt.pctRaw(v, 1) },
      { key: "pf", label: "PF", fmt: v => v == null ? "—" : fmt.num(v, 2) },
      { key: "pnl_flat", label: "PnL $", fmt: v => fmt.money(v), cls: clsSign },
    ],
    rows: u.summary_by_strategy || [],
    defaultSort: { key: "tot_r", dir: -1 },
  });

  makeTable(document.getElementById(`trd_${key}`), {
    columns: [
      { key: "strategy", label: "Strategy", align: "l" },
      { key: "ticker", label: "Ticker", align: "l" },
      { key: "direction", label: "Dir", align: "l" },
      { key: "signal_date", label: "Signal", fmt: fmt.date },
      { key: "entry_date", label: "Entry", fmt: fmt.date },
      { key: "exit_date", label: "Exit", fmt: fmt.date },
      { key: "exit_type", label: "Exit type", align: "l" },
      { key: "entry_price", label: "In", fmt: v => fmt.num(v, 2) },
      { key: "exit_price", label: "Out", fmt: v => fmt.num(v, 2) },
      { key: "r", label: "R", fmt: v => v == null ? "—" : fmt.signed(v, 2), cls: clsSign },
      { key: "pnl_flat", label: "PnL $", fmt: v => fmt.money(v), cls: clsSign },
    ],
    rows: rowsFromColumnar(u.trades),
    pageSize: 25, search: true, csvName: `futures_lab_${key}.csv`,
    defaultSort: { key: "exit_date", dir: -1 },
  });
}

function renderCumR(key, u) {
  const div = document.getElementById(`cum_${key}`);
  const rows = rowsFromColumnar(u.trades).filter(t => t.r != null);
  if (!rows.length) {
    div.innerHTML = '<p class="cap">No trades to chart.</p>';
    return;
  }
  const top = (u.summary_by_strategy || [])
    .filter(s => s.n > 0)
    .sort((a, b) => Math.abs(b.tot_r || 0) - Math.abs(a.tot_r || 0))
    .slice(0, 6).map(s => s.strategy);
  const traces = top.map((name, i) => {
    const tr = rows.filter(t => t.strategy === name);   // already exit-date sorted
    let c = 0;
    return {
      x: tr.map(t => t.exit_date),
      y: tr.map(t => (c += t.r, +c.toFixed(2))),
      mode: "lines", name, line: { width: 1.6, color: PALETTE[i % PALETTE.length] },
    };
  });
  Plotly.newPlot(div, traces, plotLayout({
    yaxis: { title: "cum R" }, margin: { t: 8 },
  }), PLOT_CFG);
}
