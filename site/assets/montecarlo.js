/* montecarlo.js — render data/montecarlo.json (book outcome distributions).
   Payload contract: scripts/build_site.py build_monte_carlo(). All dollars are
   on the flat $750k sizing basis; NAV-% figures divide by basis_nav. */
"use strict";

document.addEventListener("DOMContentLoaded", init);

const PCTS = ["5", "25", "50", "75", "95"];

function kpi(label, value, sub) {
  return `<div class="kpi"><div class="l">${label}</div><div class="v">${value}</div>
    ${sub ? `<div class="s">${sub}</div>` : ""}</div>`;
}

function pctOfNav(v, nav, d = 1) {
  return (v >= 0 ? "+" : "") + (v / nav * 100).toFixed(d) + "%";
}

function oncePer(perYr) {
  if (!perYr) return "never in sample";
  if (perYr >= 12) return `${perYr.toFixed(1)}/yr`;
  return `~1 per ${(12 / perYr).toFixed(1)} mo`;
}

function bandCells(bands, nav) {
  return PCTS.map(p => {
    const v = bands[p];
    const cls = v < 0 ? "neg" : v > 0 ? "pos" : "neu";
    return `<td style="text-align:right"><b class="${cls}">${fmt.money(v)}</b>
      <span class="cap" style="display:inline"> ${pctOfNav(v, nav)}</span></td>`;
  }).join("");
}

function bandsTable(mc) {
  const nav = mc.basis_nav;
  const rows = [
    ["Day (all days, empirical)", mc.empirical.day_bands,
     `${(100 - mc.empirical.p_up_all - mc.empirical.p_flat).toFixed(1)}%`],
    ["Month (21td, simulated)", mc.month.bands, `${mc.month.p_neg}%`],
    ["Year (252td, simulated)", mc.year.bands, `${mc.year.p_neg}%`],
  ];
  return `<div class="card"><h2>Outcome bands</h2><div class="tblwrap">
    <table class="tbl"><tr><th>Horizon</th>${PCTS.map(p => `<th style="text-align:right">p${p}</th>`).join("")}
      <th style="text-align:right">P(negative)</th></tr>
    ${rows.map(([label, bands, pneg]) =>
      `<tr><td>${label}</td>${bandCells(bands, nav)}
       <td style="text-align:right">${pneg}</td></tr>`).join("")}
    </table></div>
    <p class="cap">Simulated horizons: stationary block bootstrap, ${mc.n_sims.toLocaleString()} sims,
    mean block ${mc.mean_block_td}td, daily basis ${mc.date_min} → ${mc.asof}.</p></div>`;
}

function thresholdTable(mc) {
  const t = mc.empirical.thresholds;
  return `<div class="card"><h2>Daily loss thresholds</h2><div class="tblwrap">
    <table class="tbl"><tr><th>Down day worse than</th><th style="text-align:right">$</th>
      <th style="text-align:right">Times (23y)</th><th style="text-align:right">Frequency</th></tr>
    ${t.map(r => `<tr><td>${r.label}</td>
      <td style="text-align:right">${fmt.money(-r.dollars)}</td>
      <td style="text-align:right">${r.count}</td>
      <td style="text-align:right">${oncePer(r.per_yr)}</td></tr>`).join("")}
    </table></div>
    <p class="cap">Worst days: ${mc.empirical.worst_days.map(w =>
      `${w.date} <b class="neg">${fmt.money(w.pnl)}</b>`).join(" · ")}</p></div>`;
}

function histCard(title, h, bands, nav, elId) {
  return `<div class="card"><h2>${title}</h2><div class="chart" id="${elId}"></div>
    <p class="cap">p5 ${fmt.money(bands["5"])} · median ${fmt.money(bands["50"])} ·
    p95 ${fmt.money(bands["95"])}</p></div>`;
}

function plotHist(elId, h, bands) {
  const centers = [], counts = h.hist.counts;
  for (let i = 0; i < counts.length; i++)
    centers.push((h.hist.edges[i] + h.hist.edges[i + 1]) / 2);
  const colors = centers.map(c => c < 0 ? "#ff5d5d" : "#4da3ff");
  const shapes = ["5", "50", "95"].map(p => ({
    type: "line", x0: bands[p], x1: bands[p], y0: 0, y1: 1, yref: "paper",
    line: { color: "#c7ccd6", width: 1, dash: p === "50" ? "solid" : "dot" },
  }));
  Plotly.newPlot(elId, [{
    type: "bar", x: centers, y: counts, marker: { color: colors },
    hovertemplate: "%{x:$,.0f}: %{y} sims<extra></extra>",
  }], plotLayout({
    height: 260, bargap: 0.05, shapes,
    xaxis: { tickformat: "$,.0s" }, yaxis: { title: { text: "sims", font: { size: 11 } } },
    hovermode: "closest", showlegend: false,
  }), PLOT_CFG);
}

function intradayCard(mc) {
  const it = mc.intraday;
  if (!it) return "";
  const nav = mc.basis_nav;
  const rows = it.table.filter(r => r.count > 0).map(r =>
    `<tr><td>-${r.pct.toFixed(1)}%</td>
     <td style="text-align:right">${r.count}</td>
     <td style="text-align:right">${r.per_yr}/yr</td>
     <td style="text-align:right"><b class="${r.median_finish < 0 ? "neg" : "pos"}">${fmt.money(r.median_finish)}</b>
       <span class="cap" style="display:inline">${pctOfNav(r.median_finish, nav)}</span></td>
     <td style="text-align:right">${r.p_green}%</td>
     <td style="text-align:right">${r.p_recovered_half}%</td>
     <td style="text-align:right">${r.p_at_or_below}%</td></tr>`).join("");
  return `<div class="card"><h2>Intraday drawdown touches</h2>
    <p class="cap">Book marked at each open position's worst print (Low for longs, High for
    shorts) vs prior close / entry price, summed. Pessimistic bound: per-ticker extremes are
    not simultaneous. Limit entries make entry days near-tight (fills at first touch).
    Finishes reconcile to booked fills. Drawups are not shown — entry-day highs can predate
    the fill, so the favorable side is unknowable from daily bars.</p>
    <div class="tblwrap"><table class="tbl">
      <tr><th>Touched intraday</th><th style="text-align:right">Times (${it.cal_years}y)</th>
        <th style="text-align:right">Freq</th><th style="text-align:right">Median finish</th>
        <th style="text-align:right">Green</th><th style="text-align:right">Recovered &gt; half</th>
        <th style="text-align:right">Closed at/below touch</th></tr>${rows}</table></div>
    <p class="cap">Deepest troughs: ${it.deepest.map(d =>
      `${d.date} <b class="neg">${fmt.money(d.trough)}</b> → <b class="${d.close < 0 ? "neg" : "pos"}">${fmt.money(d.close)}</b>`).join(" · ")}</p>
    <div class="grid2" style="margin-top:10px">
      <div><h2 style="font-size:13px">Trough distribution (days touching &lt; -0.1%)</h2>
        <div class="chart" id="troughHist"></div></div>
      <div><h2 style="font-size:13px">Trough vs finish (days touching &le; -1%)</h2>
        <div class="chart" id="troughScatter"></div></div>
    </div></div>`;
}

function plotIntraday(mc) {
  const it = mc.intraday;
  if (!it) return;
  const h = it.hist, centers = [];
  for (let i = 0; i < h.counts.length; i++)
    centers.push((h.edges[i] + h.edges[i + 1]) / 2);
  Plotly.newPlot("troughHist", [{
    type: "bar", x: centers, y: h.counts, marker: { color: "#ff5d5d" },
    hovertemplate: "%{x:.1f}%: %{y} days<extra></extra>",
  }], plotLayout({
    height: 250, bargap: 0.05, hovermode: "closest", showlegend: false,
    xaxis: { title: { text: "intraday trough, % of basis", font: { size: 11 } } },
    yaxis: { title: { text: "days", font: { size: 11 } }, type: "log" },
  }), PLOT_CFG);

  const s = it.scatter;
  const lo = Math.min(...s.trough_pct) * 1.05;
  Plotly.newPlot("troughScatter", [{
    type: "scatter", mode: "markers", x: s.trough_pct, y: s.finish_pct,
    text: s.dates, marker: { color: "#4da3ff", size: 5, opacity: 0.6 },
    hovertemplate: "%{text}<br>trough %{x:.1f}% → close %{y:.1f}%<extra></extra>",
  }, {
    type: "scatter", mode: "lines", x: [lo, 0], y: [lo, 0], showlegend: false,
    line: { color: "#2a3242", width: 1, dash: "dot" }, hoverinfo: "skip",
  }], plotLayout({
    height: 250, hovermode: "closest", showlegend: false,
    xaxis: { title: { text: "intraday trough, %", font: { size: 11 } } },
    yaxis: { title: { text: "close, %", font: { size: 11 } },
             zerolinecolor: "#3a4356" },
  }), PLOT_CFG);
}

function withinRiskCard(mc) {
  const nav = mc.basis_nav;
  const row = (label, h) =>
    `<tr><td>${label}</td>
     <td style="text-align:right">${h.p_bad_day}%</td>
     <td style="text-align:right">${h.p_lt_2pct}% / ${h.p_lt_5pct}%</td>
     <td style="text-align:right">${pctOfNav(h.dd_p50, nav)}</td>
     <td style="text-align:right">${pctOfNav(h.dd_p95, nav)}</td>
     <td style="text-align:right" class="neg">${pctOfNav(h.dd_worst, nav)}</td></tr>`;
  return `<div class="card"><h2>Within-horizon risk (simulated)</h2><div class="tblwrap">
    <table class="tbl"><tr><th>Horizon</th><th style="text-align:right">P(&ge;1 day &lt; -1.5%)</th>
      <th style="text-align:right">P(total &lt; -2% / -5%)</th>
      <th style="text-align:right">maxDD p50</th><th style="text-align:right">maxDD p95</th>
      <th style="text-align:right">maxDD worst sim</th></tr>
    ${row("Month (21td)", mc.month)}${row("Year (252td)", mc.year)}
    </table></div></div>`;
}

function eraCard(mc) {
  const rows = [["Full 2003+", mc.empirical], ["Modern 2020+", mc.modern]];
  return `<div class="card"><h2>Era check</h2><div class="tblwrap">
    <table class="tbl"><tr><th>Sample</th><th style="text-align:right">P(up | active)</th>
      <th style="text-align:right">Avg day</th><th style="text-align:right">Ann. pace</th>
      <th style="text-align:right">Sharpe</th><th style="text-align:right">VaR99 / CVaR99</th></tr>
    ${rows.map(([l, e]) => `<tr><td>${l}</td>
      <td style="text-align:right">${e.p_up_active}%</td>
      <td style="text-align:right">${fmt.money(e.mean_day)}</td>
      <td style="text-align:right">${fmt.money(e.ann_pnl)}</td>
      <td style="text-align:right">${e.sharpe}</td>
      <td style="text-align:right">${fmt.money(-e.var99)} / ${fmt.money(-e.cvar99)}</td></tr>`).join("")}
    </table></div>
    <p class="cap">Plan on the full-history row; treat 2020+ as the optimistic case, not the base case.</p></div>`;
}

function calendarCard(mc) {
  const c = mc.calendar;
  return `<div class="card"><h2>Calendar actuals (reality check)</h2>
    <p style="margin:6px 0">Real months: <b>${c.months.p_neg}%</b> negative of ${c.months.n},
      median <b class="pos">${fmt.money(c.months.median)}</b>,
      worst <b class="neg">${fmt.money(c.months.worst)}</b> (${c.months.worst_when}).</p>
    <p style="margin:6px 0">Real years: <b>${c.years.p_neg}%</b> negative of ${c.years.n},
      median <b class="pos">${fmt.money(c.years.median)}</b>,
      worst <b class="${c.years.worst < 0 ? "neg" : "pos"}">${fmt.money(c.years.worst)}</b> (${c.years.worst_when}).</p>
    <p class="cap">If the simulated bands and these disagree wildly, distrust the sim first.</p></div>`;
}

async function init() {
  renderNav("montecarlo.html");
  const mc = await fetchJSONOrNull("data/montecarlo.json");
  const el = document.getElementById("content");
  if (!mc) {
    el.innerHTML = '<p class="cap">No montecarlo.json in this build (needs the daily PnL parquet from the ledger build).</p>';
    return;
  }
  setAsof(`daily basis through ${mc.asof}`);
  const e = mc.empirical;
  el.innerHTML = `
    <div class="kpis">
      ${kpi("P(up day | in market)", e.p_up_active + "%", `${e.p_up_all}% all days · ${e.p_flat}% flat`)}
      ${kpi("Avg day", fmt.money(e.mean_day), `σ ${fmt.money(e.std_day)}`)}
      ${kpi("Annual pace", fmt.money(e.ann_pnl), pctOfNav(e.ann_pnl, mc.basis_nav) + " of basis")}
      ${kpi("Sharpe (daily, ann.)", e.sharpe, `${e.n_days.toLocaleString()} days`)}
      ${kpi("Daily VaR99", fmt.money(-e.var99), `CVaR99 ${fmt.money(-e.cvar99)}`)}
      ${kpi("P(year negative)", mc.year.p_neg + "%", "simulated")}
    </div>
    ${bandsTable(mc)}
    <div class="grid2" style="margin-top:14px">
      ${histCard("Month PnL distribution", mc.month, mc.month.bands, mc.basis_nav, "histMonth")}
      ${histCard("Year PnL distribution", mc.year, mc.year.bands, mc.basis_nav, "histYear")}
    </div>
    <div style="margin-top:14px">${thresholdTable(mc)}</div>
    ${mc.intraday ? `<div style="margin-top:14px">${intradayCard(mc)}</div>` : ""}
    <div style="margin-top:14px">${withinRiskCard(mc)}</div>
    <div class="grid2" style="margin-top:14px">
      ${eraCard(mc)}
      ${calendarCard(mc)}
    </div>`;
  plotHist("histMonth", mc.month, mc.month.bands);
  plotHist("histYear", mc.year, mc.year.bands);
  plotIntraday(mc);
}
