/* risk.js — render data/risk.json (condensed risk_dashboard_v2 summary) */
"use strict";

const RISK_SIGNAL_COLORS = {
  "Distribution Dominance": "#e74c3c",
  "VIX Range Compression": "#e67e22",
  "Defensive Leadership": "#2ecc71",
  "Pre-FOMC Rally": "#3498db",
  "Low Absorption Ratio": "#9b59b6",
  "Seasonal Rank Divergence": "#1abc9c",
  "Dispersion": "#f39c12",
};

document.addEventListener("DOMContentLoaded", init);

async function init() {
  renderNav("risk.html");
  const el = document.getElementById("content");
  const d = await fetchJSONOrNull("data/risk.json");
  if (!d) {
    el.innerHTML = '<p class="cap">No risk payload in this build (build_risk_json.py skipped or failed).</p>';
    return;
  }
  setAsof(`as of ${d.asof} · built ${d.built_at}`);

  let html = "";

  // fragility cards
  const frag = d.fragility || {};
  const cards = ["5d", "21d", "63d"].map(h => {
    const v = frag[h];
    if (v == null) return "";
    const lbl = v < 33 ? "Robust" : v < 66 ? "Neutral" : "Fragile";
    const cls = v < 33 ? "pos" : v < 66 ? "" : "neg";
    return `<div class="kpi"><div class="l">Fragility ${h}</div>
      <div class="v ${cls}">${Math.round(v)}</div><div class="s">${lbl} (0-100)</div></div>`;
  }).join("");
  const ctx = d.price_ctx || {};
  html += `<div class="kpis">
    <div class="kpi"><div class="l">SPY</div><div class="v">${fmt.num(d.spy_last, 2)}</div>
      <div class="s">${esc(ctx.regime_label || "")}</div></div>
    ${cards}
    <div class="kpi"><div class="l">Active Signals</div>
      <div class="v ${d.n_active > 0 ? "neg" : "pos"}">${d.n_active} / ${(d.signals || []).length}</div>
      <div class="s">regime mult ${d.regime_mult != null ? Number(d.regime_mult).toFixed(2) + "x" : "-"}</div></div>
  </div>`;

  // price context kv
  const scalars = Object.entries(ctx).filter(([, v]) => typeof v !== "object" || v == null);
  if (scalars.length) {
    html += `<h2>Price context</h2><div class="card"><div class="kv">` +
      scalars.map(([k, v]) => `<div class="k">${esc(k)}</div><div class="v">${fmtCtx(v)}</div>`).join("") +
      `</div></div>`;
  }

  // signals (the richer signal_detail block is optional so older best-effort
  // payloads continue to render the original cards).
  const signalDetail = d.signal_detail && typeof d.signal_detail === "object"
    ? d.signal_detail : null;
  html += `<h2>Signals</h2>`;
  if (signalDetail && d.spy_series) {
    html += `<div class="card risk-overlay-card">
      <div class="cap" style="margin-top:0">Signal activity vs SPY · opens on the latest year; double-click for full history</div>
      <div class="chart" id="signalOverlayChart"></div></div>`;
  }
  for (const [i, s] of (d.signals || []).entries()) {
    const badgeCls = s.on ? "on" : (s.badge || "").startsWith("DECAYING") ? "warn" : "off";
    const sd = signalDetail ? signalDetail[s.name] : null;
    const current = sd && sd.current ? sd.current : null;
    const currentFigure = currentMetricText(sd);
    const hasMetric = !!(sd && sd.metric && Array.isArray(sd.metric.values) &&
      sd.metric.values.some(v => v != null && Number.isFinite(Number(v))));
    const hasPeriods = !!(sd && Array.isArray(sd.periods) && sd.periods.length);
    const hasChart = !!(d.spy_series && sd && (hasMetric || hasPeriods));
    html += `<div class="card idea">
      <div class="head"><span class="tkr">${esc(s.name)}</span>
        <span class="badge ${badgeCls}">${esc(s.badge)}</span>
        ${currentFigure ? `<span class="signal-current">${esc(currentFigure)}</span>` : ""}</div>
      ${s.detail ? `<div class="cap">${esc(detailText(s.detail))}</div>` : ""}
      ${current && current.summary ? `<div class="signal-summary">${esc(current.summary)}</div>` : ""}
      ${hasChart ? `<div class="chart signal-chart" id="signalMetricChart-${i}"></div>` : ""}
    </div>`;
  }

  // chart: SPY + fragility
  if (d.spy_series) {
    html += `<h2>SPY vs fragility (1y, 5d-smoothed)</h2>
      <div class="card"><div class="chart" id="riskChart"></div></div>`;
  }

  // forward returns
  const fwd = d.forward_returns || {};
  const horizons = Object.keys(fwd);
  if (horizons.length) {
    html += `<h2>Forward returns at similar fragility readings</h2>`;
    for (const h of ["5d", "21d", "63d"]) {
      const r = fwd[h];
      if (!r) continue;
      html += fwdTable(h, r);
    }
  }

  // protection cost + hedge recommendation (optional 'hedge' block)
  const hg = d.hedge;
  if (hg) {
    const ratioTxt = hg.term_ratio == null ? "" :
      ` = ${fmt.num(hg.term_ratio, 3)}${hg.term_ratio > 1 ? " (inverted)" : ""}`;
    html += `<h2>Protection cost &amp; hedge posture</h2>
    <div class="grid2">
      <div class="card">
        ${hg.pctile != null ? '<div class="chart" id="hedgeGauge"></div>' :
          '<p class="cap">Protection-cost percentile unavailable (needs ~1y of proxy history).</p>'}
        <div class="kv">
          <div class="k">Proxy (VIX3M x SKEW/130)</div><div class="v">${fmt.num(hg.proxy, 2)}</div>
          <div class="k">VIX / VIX3M</div>
          <div class="v">${hg.vix != null ? fmt.num(hg.vix, 2) : "-"} / ${fmt.num(hg.vix3m, 2)}${esc(ratioTxt)}</div>
          <div class="k">Regime input</div><div class="v">${esc(hg.regime || "-")} (${hg.regime_points != null ? hg.regime_points : "-"} pts)</div>
          <div class="k">As of</div><div class="v">${esc(hg.asof || "-")}</div>
        </div>
      </div>
      <div class="card">
        <div style="font-weight:700;font-size:15px;color:${esc(hg.color || "#c7ccd6")}">${esc(hg.rec || "")}</div>
        <p style="margin:8px 0 0;font-size:13px;line-height:1.5">${esc(hg.detail || "")}</p>
        <p class="cap">Regime basis: ${esc(hg.regime_basis || "")}.
          Advisory text only (legacy Layer 4C decision tree, hit rates are placeholder
          estimates) — not an order instruction.</p>
      </div>
    </div>`;
    if (hg.spark && hg.spark.dates && hg.spark.dates.length) {
      html += `<div class="card" style="margin-top:14px">
        <div class="cap" style="margin-top:0">Protection-cost proxy, 2y weekly, with trailing-5y percentile</div>
        <div class="chart" id="hedgeSpark"></div></div>`;
    }
  }

  el.innerHTML = html;

  if (signalDetail && d.spy_series) {
    renderSignalOverlay(d, signalDetail);
    for (const [i, s] of (d.signals || []).entries()) {
      const sd = signalDetail[s.name];
      if (sd) renderSignalMetricChart(d, s.name, sd, `signalMetricChart-${i}`);
    }
  }

  if (d.spy_series) {
    const chartEl = document.getElementById("riskChart");
    const spyDates = seriesDates(d, d.spy_series);
    const initialRange = latestYearRange(spyDates, d.asof);
    const traces = [{
      x: spyDates, y: d.spy_series.close, name: "SPY",
      mode: "lines", line: { color: "#4da3ff", width: 1.6 },
    }];
    const fs = d.fragility_series;
    if (fs && fs["21d"]) {
      const fragDates = seriesDates(d, fs);
      traces.push({
        x: fragDates, y: fs["21d"], name: "Fragility 21d", yaxis: "y2",
        mode: "lines", line: { color: "#ffc14d", width: 1.2 },
      });
    }
    const spyRange = valuesRange(d.spy_series.close, spyDates, initialRange);
    const fragRange = fs && fs["21d"]
      ? valuesRange(fs["21d"], seriesDates(d, fs), initialRange) : null;
    Plotly.newPlot(chartEl, traces, plotLayout({
      height: 340,
      xaxis: { range: initialRange },
      yaxis: { range: spyRange, title: { text: "SPY", font: { size: 11 } } },
      yaxis2: { overlaying: "y", side: "right", range: fragRange, showgrid: false,
                title: { text: "Fragility", font: { size: 11 } } },
    }), PLOT_CFG);
    enableFullHistoryReset(chartEl, ["xaxis", "yaxis", "yaxis2"]);
  }

  if (hg) renderHedgeCharts(hg);
}

function seriesDates(payload, series) {
  if (series && Array.isArray(series.dates)) return series.dates;
  return Array.isArray(payload.dates) ? payload.dates : [];
}

function latestYearRange(dates, asof) {
  const valid = (dates || []).filter(Boolean);
  const endText = asof || valid[valid.length - 1];
  if (!endText) return null;
  const end = new Date(`${String(endText).slice(0, 10)}T00:00:00Z`);
  if (!Number.isFinite(end.getTime())) return null;
  const start = new Date(end.getTime());
  start.setUTCFullYear(start.getUTCFullYear() - 1);
  return [start.toISOString().slice(0, 10), end.toISOString().slice(0, 10)];
}

function valuesRange(values, dates, dateRange, extras) {
  const loDate = dateRange ? Date.parse(dateRange[0]) : -Infinity;
  const hiDate = dateRange ? Date.parse(dateRange[1]) : Infinity;
  const nums = [];
  for (let i = 0; i < Math.min((values || []).length, (dates || []).length); i += 1) {
    const t = Date.parse(dates[i]);
    const v = Number(values[i]);
    if (Number.isFinite(t) && t >= loDate && t <= hiDate &&
        values[i] != null && Number.isFinite(v)) nums.push(v);
  }
  for (const raw of extras || []) {
    const v = Number(raw);
    if (raw != null && Number.isFinite(v)) nums.push(v);
  }
  if (!nums.length) return null;
  const lo = Math.min(...nums);
  const hi = Math.max(...nums);
  const pad = lo === hi ? Math.max(Math.abs(lo) * 0.05, 1) : (hi - lo) * 0.07;
  return [lo - pad, hi + pad];
}

function currentMetricText(detail) {
  if (!detail || !detail.metric || !detail.current || detail.current.value == null) return "";
  const metric = detail.metric;
  const value = Number(detail.current.value);
  if (!Number.isFinite(value)) return "";
  const decimals = Number.isInteger(metric.decimals) ? metric.decimals : 1;
  if (metric.unit === "pp") return `${value >= 0 ? "+" : ""}${value.toFixed(decimals)}pp`;
  if (metric.unit === "percentile") return `${value.toFixed(decimals)} pctile`;
  return value.toFixed(decimals);
}

function colorWithAlpha(hex, alpha) {
  const m = /^#([0-9a-f]{6})$/i.exec(hex || "");
  if (!m) return `rgba(136,136,136,${alpha})`;
  const n = parseInt(m[1], 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}

function enableFullHistoryReset(el, axes) {
  if (!el || typeof el.on !== "function") return;
  el.on("plotly_doubleclick", () => {
    const update = {};
    for (const axis of axes) update[`${axis}.autorange`] = true;
    Plotly.relayout(el, update);
    return false;
  });
}

function renderSignalOverlay(d, signalDetail) {
  const el = document.getElementById("signalOverlayChart");
  const dates = seriesDates(d, d.spy_series);
  if (!el || !dates.length || !(d.spy_series.close || []).length) return;

  const names = (d.signals || []).map(s => s.name).filter(name => signalDetail[name]);
  if (!names.length) return;
  const initialRange = latestYearRange(dates, d.asof);
  const traces = [{
    x: dates, y: d.spy_series.close, name: "SPY", showlegend: false,
    mode: "lines", line: { color: "rgba(190,196,208,.9)", width: 1.5 },
    xaxis: "x", yaxis: "y",
  }];
  const shapes = [];
  names.forEach((name, i) => {
    const color = RISK_SIGNAL_COLORS[name] || "#888888";
    traces.push({
      x: [null], y: [null], name, xaxis: "x2", yaxis: "y2",
      mode: "markers", marker: { color, size: 8, symbol: "square" },
      hoverinfo: "skip", showlegend: true,
    });
    for (const period of signalDetail[name].periods || []) {
      if (!period || !period[0] || !period[1]) continue;
      shapes.push({
        type: "rect", xref: "x2", yref: "y2", layer: "below",
        x0: period[0], x1: period[1], y0: i - 0.34, y1: i + 0.34,
        line: { width: 0 }, fillcolor: colorWithAlpha(color, 0.72),
      });
    }
  });

  Plotly.newPlot(el, traces, plotLayout({
    height: 500,
    margin: { l: 158, r: 18, t: 42, b: 40 },
    shapes,
    xaxis: {
      type: "date", domain: [0, 1], anchor: "y",
      range: initialRange, showticklabels: false,
    },
    yaxis: {
      domain: [0.34, 1], range: valuesRange(d.spy_series.close, dates, initialRange),
      title: { text: "SPY", font: { size: 11 } },
    },
    xaxis2: {
      type: "date", domain: [0, 1], anchor: "y2",
      range: initialRange, matches: "x",
    },
    yaxis2: {
      domain: [0, 0.25], range: [-0.5, names.length - 0.5],
      tickmode: "array", tickvals: names.map((_, i) => i), ticktext: names,
      fixedrange: true, showgrid: true, gridcolor: "rgba(128,128,128,.10)",
    },
    legend: { orientation: "h", y: 1.11, x: 1, xanchor: "right", font: { size: 9 } },
    hovermode: "x unified",
  }), PLOT_CFG);
  enableFullHistoryReset(el, ["xaxis", "xaxis2", "yaxis"]);
}

function renderSignalMetricChart(d, name, detail, id) {
  const el = document.getElementById(id);
  const dates = seriesDates(d, d.spy_series);
  const spy = d.spy_series && d.spy_series.close;
  if (!el || !dates.length || !Array.isArray(spy)) return;

  const metric = detail.metric;
  const metricValues = metric && Array.isArray(metric.values) ? metric.values : [];
  const hasMetric = metricValues.some(v => v != null && Number.isFinite(Number(v)));
  const periods = Array.isArray(detail.periods) ? detail.periods : [];
  if (!hasMetric && !periods.length) return;

  const color = RISK_SIGNAL_COLORS[name] || "#888888";
  const initialRange = latestYearRange(dates, d.asof);
  const traces = [{
    x: dates, y: spy, name: "SPY", mode: "lines",
    line: { color: "rgba(180,186,198,.58)", width: 1.1 },
  }];
  const thresholds = metric && Array.isArray(metric.thresholds) ? metric.thresholds : [];
  if (hasMetric) {
    traces.push({
      x: dates, y: metricValues, name: metric.label || "Metric", yaxis: "y2",
      mode: "lines", line: { color, width: 1.6 },
    });
    for (const threshold of thresholds) {
      traces.push({
        x: dates.length ? [dates[0], dates[dates.length - 1]] : [],
        y: [threshold.value, threshold.value], yaxis: "y2", mode: "lines",
        name: `${threshold.label || "Threshold"} ${threshold.operator || ""} ${threshold.value}`.trim(),
        line: { color: colorWithAlpha(color, 0.8), width: 1, dash: "dash" },
        hoverinfo: "name+y",
      });
    }
  }
  const shapes = periods.filter(p => p && p[0] && p[1]).map(period => ({
    type: "rect", xref: "x", yref: "paper", layer: "below",
    x0: period[0], x1: period[1], y0: 0, y1: 1,
    line: { width: 0 }, fillcolor: colorWithAlpha(color, 0.15),
  }));
  const thresholdValues = thresholds.map(t => t.value);
  const layout = {
    height: 270,
    margin: { l: 52, r: hasMetric ? 64 : 18, t: 34, b: 34 },
    shapes,
    xaxis: { range: initialRange },
    yaxis: {
      range: valuesRange(spy, dates, initialRange),
      title: { text: "SPY", font: { size: 10 } },
    },
    legend: { orientation: "h", y: 1.12, x: 1, xanchor: "right", font: { size: 9 } },
  };
  if (hasMetric) {
    layout.yaxis2 = {
      overlaying: "y", side: "right", showgrid: false,
      range: valuesRange(metricValues, dates, initialRange, thresholdValues),
      title: { text: metric.label || "Metric", font: { size: 10 } },
    };
  }
  Plotly.newPlot(el, traces, plotLayout(layout), PLOT_CFG);
  enableFullHistoryReset(el, hasMetric ? ["xaxis", "yaxis", "yaxis2"] : ["xaxis", "yaxis"]);
}

/* Layer 4B-style gauge (green/yellow/orange/red bands match the 4C decision
   tree thresholds: <20 cheap, 20-60 fair, 60-85 expensive, >=85 very) plus
   the 2y weekly proxy sparkline with its trailing-5y percentile. */
function renderHedgeCharts(hg) {
  const gaugeEl = document.getElementById("hedgeGauge");
  if (gaugeEl && hg.pctile != null) {
    Plotly.newPlot(gaugeEl, [{
      type: "indicator", mode: "gauge+number", value: hg.pctile,
      number: { suffix: "", font: { size: 30, color: "#e8eaf0" } },
      title: { text: "Protection cost percentile (trailing 5y)",
               font: { size: 12, color: "#c7ccd6" } },
      gauge: {
        axis: { range: [0, 100], tickcolor: "#2a3242",
                tickfont: { color: "#c7ccd6", size: 10 } },
        bar: { color: "#e8eaf0", thickness: 0.22 },
        bgcolor: "rgba(0,0,0,0)", borderwidth: 0,
        steps: [
          { range: [0, 20],   color: "rgba(0,204,0,.45)" },
          { range: [20, 60],  color: "rgba(255,215,0,.40)" },
          { range: [60, 85],  color: "rgba(255,140,0,.45)" },
          { range: [85, 100], color: "rgba(204,0,0,.50)" },
        ],
      },
    }], plotLayout({ height: 210, margin: { l: 28, r: 28, t: 40, b: 6 } }), PLOT_CFG);
  }

  const sparkEl = document.getElementById("hedgeSpark");
  if (sparkEl && hg.spark) {
    const traces = [{
      x: hg.spark.dates, y: hg.spark.proxy, name: "Proxy",
      mode: "lines", line: { color: "#4da3ff", width: 1.6 },
    }];
    if ((hg.spark.pctile || []).some(v => v != null)) {
      traces.push({
        x: hg.spark.dates, y: hg.spark.pctile, name: "Pctile (5y)", yaxis: "y2",
        mode: "lines", line: { color: "#ffc14d", width: 1.1, dash: "dot" },
      });
    }
    Plotly.newPlot(sparkEl, traces, plotLayout({
      height: 220,
      yaxis: { title: { text: "VIX3M x SKEW/130", font: { size: 11 } } },
      yaxis2: { overlaying: "y", side: "right", range: [0, 100], showgrid: false,
                title: { text: "Pctile", font: { size: 11 } } },
    }), PLOT_CFG);
  }
}

function fwdTable(h, r) {
  const head = `<tr>
    <th class="l">Window</th><th>Mean</th><th>Median</th><th>% Neg</th>
    <th>Mean Z</th><th>Median Z</th><th>Baseline</th></tr>`;
  let rows = "";
  for (const [w, st] of Object.entries(r.returns || {})) {
    if (!st) continue;
    const mz = st.mean_z || 0;
    const mCls = mz <= -1 ? "neg" : mz < 0 ? "" : "pos";
    rows += `<tr>
      <td class="l">${esc(w)}d</td>
      <td class="${mCls}">${fmt.pct(st.mean, 2)}</td>
      <td>${fmt.pct(st.median, 2)}</td>
      <td class="${st.pct_neg > 0.5 ? "neg" : ""}">${fmt.pct(st.pct_neg, 0)}</td>
      <td class="${mCls}">${fmt.signed(mz, 2)}</td>
      <td>${fmt.signed(st.median_z || 0, 2)}</td>
      <td class="${st.uncond_mean >= 0 ? "pos" : "neg"}">${fmt.pct(st.uncond_mean, 2)}</td></tr>`;
  }
  if (!rows) return "";
  return `<div class="card" style="margin-bottom:12px">
    <div class="cap" style="margin-top:0">${h.toUpperCase()} fragility = ${Math.round(r.current_score)} ·
      ${r.n_episodes} episodes · band ${Math.round(r.band_low)}-${Math.round(r.band_high)}</div>
    <div class="tblwrap"><table class="tbl"><thead>${head}</thead><tbody>${rows}</tbody></table></div>
  </div>`;
}

function detailText(v) {
  if (typeof v === "string") return v;
  if (typeof v === "object" && v != null)
    return Object.entries(v).map(([k, x]) => `${k}: ${x}`).join(" · ");
  return String(v);
}
function fmtCtx(v) {
  if (typeof v === "number") return Math.abs(v) < 1 ? v.toFixed(4) : v.toFixed(2);
  return esc(String(v));
}
function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
