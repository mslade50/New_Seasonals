/* entry.js — Entry Lab: entry-rule what-if sweeps over the full book.
   Payload: /research/entry_lab.json (one-off, committed static — NOT nightly). */
"use strict";

document.addEventListener("DOMContentLoaded", init);

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function init() {
  renderNav("entry.html");
  const el = document.getElementById("content");
  const d = await fetchJSONOrNull("research/entry_lab.json");
  if (!d) {
    el.innerHTML = '<p class="cap">No entry-lab payload in this build yet ' +
      "(run scratch/entry_lab_build.py and commit site/research/entry_lab.json).</p>";
    return;
  }
  setAsof(`computed ${d.computed_asof}`);

  const bb = d.baseline && d.baseline.book || {};
  let html = `
    <div class="card" style="border:1px solid var(--amber,#ffc14d);margin-bottom:14px">
      <strong style="color:var(--amber,#ffc14d)">RESEARCH ONLY.</strong>
      <span class="cap">In-sample entry-rule sweeps — prod parameters were chosen on this
      same history; a better cell here is a lead, not evidence. Computed once as of
      <strong>${esc(d.computed_asof)}</strong> (from ${esc(d.bt_start)}, ${esc(d.basis)}
      basis) — not rebuilt nightly.</span>
    </div>
    <div class="kpis">
      ${kpi("Baseline trades", (bb.n || 0).toLocaleString())}
      ${kpi("Baseline total R", fmt.signed(bb.tot_r, 1), clsSign(bb.tot_r))}
      ${kpi("Baseline avg R", fmt.signed(bb.avg_r, 3), clsSign(bb.avg_r))}
      ${kpi("Baseline win %", bb.win_pct == null ? "—" : fmt.pctRaw(bb.win_pct, 1))}
      ${kpi("Baseline PnL $", fmt.money(bb.pnl_flat), clsSign(bb.pnl_flat))}
    </div>`;

  (d.sweeps || []).forEach((sw, i) => {
    html += `
      <h2>${esc(sw.label || sw.dimension)}</h2>
      <p class="cap">${esc(sw.notes || "")}</p>
      <div class="grid3">
        <div class="card"><h3>Avg R</h3><div id="sw${i}_avg" style="height:260px"></div></div>
        <div class="card"><h3>Total R</h3><div id="sw${i}_tot" style="height:260px"></div></div>
        <div class="card"><h3>Fill rate</h3><div id="sw${i}_fill" style="height:260px"></div></div>
      </div>
      <div class="card"><h3>Detail</h3><div id="sw${i}_tbl"></div></div>`;
  });

  if ((d.dropped_dimensions || []).length) {
    html += `<div class="card"><h3>Dropped dimensions</h3><ul class="cap">` +
      d.dropped_dimensions.map(x => `<li>${esc(x)}</li>`).join("") + "</ul></div>";
  }
  html += `<div class="card"><h3>Caveats</h3><ul class="cap">` +
    (d.caveats || []).map(c => `<li>${esc(c)}</li>`).join("") + "</ul></div>";

  el.innerHTML = html;

  (d.sweeps || []).forEach((sw, i) => renderSweep(sw, i));
}

function kpi(label, value, cls) {
  return `<div class="kpi"><div class="l">${esc(label)}</div>
    <div class="v ${cls || ""}">${value}</div></div>`;
}

function renderSweep(sw, i) {
  const strategies = sw.strategy_scope || [];
  const points = sw.points || [];
  const prod = sw.prod_values || {};

  const metricChart = (divId, key, title, dec) => {
    const traces = [];
    strategies.forEach((name, si) => {
      const color = PALETTE[si % PALETTE.length];
      const xs = [], ys = [];
      points.forEach(p => {
        const row = (p.per_strategy || []).find(r => r.strategy === name);
        if (row && row[key] != null) { xs.push(p.value); ys.push(row[key]); }
      });
      traces.push({ x: xs, y: ys, mode: "lines+markers", name,
        line: { width: 1.6, color }, marker: { size: 6 } });
      // prod marker
      const pv = prod[name];
      const pi = xs.indexOf(pv);
      if (pi >= 0) {
        traces.push({ x: [xs[pi]], y: [ys[pi]], mode: "markers", showlegend: false,
          hoverinfo: "skip",
          marker: { size: 13, symbol: "circle-open", color,
                    line: { width: 2.5, color } } });
      }
    });
    Plotly.newPlot(document.getElementById(divId), traces, plotLayout({
      xaxis: { title: sw.dimension, tickvals: points.map(p => p.value) },
      yaxis: { title },
      margin: { t: 8 },
      showlegend: strategies.length > 1,
    }), PLOT_CFG);
  };

  metricChart(`sw${i}_avg`, "avg_r", "avg R", 3);
  metricChart(`sw${i}_tot`, "tot_r", "total R", 1);
  metricChart(`sw${i}_fill`, "fill_rate", "fills / signals", 3);

  const rows = [];
  points.forEach(p => (p.per_strategy || []).forEach(r => rows.push(Object.assign({
    value: p.value,
    is_prod: prod[r.strategy] === p.value ? "PROD" : "",
  }, r))));
  makeTable(document.getElementById(`sw${i}_tbl`), {
    columns: [
      { key: "strategy", label: "Strategy", align: "l" },
      { key: "value", label: "Value" },
      { key: "is_prod", label: "", align: "l",
        cls: v => v ? "pos" : "" },
      { key: "n_signals", label: "Signals" },
      { key: "n", label: "Trades" },
      { key: "fill_rate", label: "Fill rate", fmt: v => v == null ? "—" : fmt.num(v, 3) },
      { key: "tot_r", label: "Tot R", fmt: v => fmt.signed(v, 1), cls: clsSign },
      { key: "avg_r", label: "Avg R", fmt: v => v == null ? "—" : fmt.signed(v, 3), cls: clsSign },
      { key: "win_pct", label: "Win %", fmt: v => v == null ? "—" : fmt.pctRaw(v, 1) },
      { key: "pf", label: "PF", fmt: v => v == null ? "—" : fmt.num(v, 2) },
      { key: "pnl_flat", label: "PnL $", fmt: v => fmt.money(v), cls: clsSign },
    ],
    rows,
    csvName: `entry_lab_${sw.dimension}.csv`,
    defaultSort: { key: "strategy", dir: 1 },
  });
}
