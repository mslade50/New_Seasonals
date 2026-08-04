/* risk.js — render data/risk.json (condensed risk_dashboard_v2 summary).
   Page order: sizing state (the number that sizes live orders, PIT-sourced),
   KPI strip, price context, nuggets, signals, SPY-vs-dial chart, forward
   returns. Every post-2026-07-16 block is guarded so older payloads render. */
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

  // 1. sizing state hero — the one number that sizes live orders
  const sz = d.sizing_state;
  if (sz && sz.score != null) html += sizingHeroHtml(sz);
  if (d.atr_downside) html += atrDialTableHtml(d.atr_downside);

  // 2. KPI strip: SPY, the 63d dial, context horizons, signals, vol term
  html += kpiRowHtml(d);

  // 3. price context strip (replaces the old raw kv dump)
  html += contextStripHtml(d);

  // 3b. Trade Console — the one decisive block (configuration-conditional,
  // display-only). Replaces the nuggets zone; nuggets render only as a
  // fallback for older payloads (the payload key stays — ideas.js reads it).
  const tc = d.trade_console;
  if (tc && tc.state) {
    html += tradeConsoleHtml(tc);
  } else if (Array.isArray(d.nuggets) && d.nuggets.length) {
    html += nuggetsHtml(d.nuggets);
  }

  // 4. signals: overlay chart + accordion (charts lazy-render on expand)
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
    const headRow = `<span class="tkr">${esc(s.name)}</span>
        <span class="badge ${badgeCls}">${esc(s.badge)}</span>
        ${currentFigure ? `<span class="signal-current">${esc(currentFigure)}</span>` : ""}`;
    if (hasChart) {
      html += `<details class="card idea signal-acc" data-sig="${i}">
        <summary class="head">${headRow}</summary>
        ${s.detail ? `<div class="cap">${esc(detailText(s.detail))}</div>` : ""}
        ${current && current.summary ? `<div class="signal-summary">${esc(current.summary)}</div>` : ""}
        <div class="chart signal-chart" id="signalMetricChart-${i}"></div>
      </details>`;
    } else {
      html += `<div class="card idea">
        <div class="head">${headRow}</div>
        ${s.detail ? `<div class="cap">${esc(detailText(s.detail))}</div>` : ""}
        ${current && current.summary ? `<div class="signal-summary">${esc(current.summary)}</div>` : ""}
      </div>`;
    }
    // downside table under a FIRING signal (always visible, outside the accordion)
    if (s.on) html += atrSignalTableHtml(s.name, d.atr_downside);
  }

  // 5. chart: SPY + the exact PIT sizing statistic.  The recompute-vintage
  // fragility series is retained only as a fallback for old payloads.
  const fs = d.fragility_series || {};
  const fragKey = fs["63d"] ? "63d" : fs["21d"] ? "21d" : null;
  const sizingChart = sz && sz.spark && Array.isArray(sz.spark.dates) &&
    Array.isArray(sz.spark.ma) && sz.spark.dates.length;
  const dailyChart = sizingChart && Array.isArray(sz.spark.daily) &&
    sz.spark.daily.length === sz.spark.dates.length;
  if (d.spy_series) {
    const riskBasis = sizingChart
      ? "production sizing fragility (63d · 10d MA · append-only PIT)"
      : `display recompute ${fragKey || ""} (legacy payload; not a sizing input)`;
    html += `<h2>SPY vs ${riskBasis}</h2>
      <div class="card">
        ${dailyChart ? '<div class="cap" style="margin-top:0">Top: SPY + 10d moving average line &middot; Bottom: each individual daily 63d dial reading</div>' : ""}
        <div class="tbl-controls"><label>Range</label><span class="seg" id="dialRangeSeg"></span>
          <span class="info cap-inline">dial history 2016+ &middot; pre-2026-07 rows are the recompute vintage, not PIT</span></div>
        <div class="chart" id="riskChart"></div></div>`;
  }

  // 6. forward returns — 63d is the sizing horizon; other horizons collapsed
  const fwd = d.forward_returns || {};
  if (fwd["63d"] || fwd["21d"] || fwd["5d"]) {
    html += `<h2>Forward returns at similar fragility readings</h2>`;
    if (fwd["63d"]) html += fwdTable("63d", fwd["63d"]);
    const others = ["5d", "21d"].filter(h => fwd[h]);
    if (others.length) {
      html += `<details class="fwd-others"><summary class="cap">Context horizons (${others.join(", ")}) — not sizing inputs</summary>` +
        others.map(h => fwdTable(h, fwd[h])).join("") + `</details>`;
    }
    html += `<p class="cap">Similar-reading history includes the pre-2026-07-02 recompute
      vintage; the point-in-time series starts 2026-07-02.</p>`;
  }

  // 7. Deliberately last on the page for now: ATR drawdown paths from the
  // exact anchor dates in the primary 63d similar-fragility sample above.
  if (d.drawdown_iv && d.drawdown_iv.rows_by_horizon) {
    html += drawdownIvHtml(d.drawdown_iv);
  }

  el.innerHTML = html;

  if (d.drawdown_iv && d.drawdown_iv.rows_by_horizon) {
    initDrawdownIvTable(d.drawdown_iv);
  }

  if (sz && sz.spark && Array.isArray(sz.spark.dates) && sz.spark.dates.length) {
    renderSizingSpark(sz);
  }

  if (signalDetail && d.spy_series) {
    renderSignalOverlay(d, signalDetail);
    // accordion charts lazy-render on first expand
    for (const acc of document.querySelectorAll("details.signal-acc")) {
      acc.addEventListener("toggle", () => {
        if (!acc.open || acc.dataset.rendered) return;
        const i = Number(acc.dataset.sig);
        const s = (d.signals || [])[i];
        const sd = s ? signalDetail[s.name] : null;
        if (sd) renderSignalMetricChart(d, s.name, sd, `signalMetricChart-${i}`);
        acc.dataset.rendered = "1";
      });
    }
  }

  if (d.spy_series) {
    const chartEl = document.getElementById("riskChart");
    const spyDates = seriesDates(d, d.spy_series);
    const initialRange = latestYearRange(spyDates, d.asof);
    const traces = [];
    let riskDates = null;
    let riskValues = null;
    let riskName = null;
    let dailyDates = null;
    let dailyValues = null;
    if (sizingChart) {
      riskDates = sz.spark.dates;
      riskValues = sz.spark.ma;
      if (dailyChart) {
        dailyDates = sz.spark.dates;
        dailyValues = sz.spark.daily;
      }
      riskName = "Sizing fragility 63d · 10d MA · PIT";
    } else if (fragKey) {
      riskDates = seriesDates(d, fs);
      riskValues = fs[fragKey];
      riskName = `Display recompute ${fragKey}`;
    }
    if (riskValues) {
      traces.push({
        x: riskDates, y: riskValues, name: riskName, yaxis: "y2",
        mode: "lines", line: { color: "#ffc14d", width: 1.8 },
        hovertemplate: "%{y:.1f}<extra>63d 10d MA</extra>",
      });
    }
    // SPY and the moving average remain together in the upper panel.
    traces.push({
      x: spyDates, y: d.spy_series.close, name: "SPY",
      mode: "lines", line: { color: "#4da3ff", width: 1.8 },
    });
    if (dailyValues) {
      traces.push({
        x: dailyDates, y: dailyValues, name: "Daily fragility 63d", yaxis: "y3",
        type: "bar",
        marker: { color: "rgba(255,193,77,.72)", line: { width: 0 } },
        hovertemplate: "%{y:.1f}<extra>Daily 63d</extra>",
      });
    }
    const spyRange = valuesRange(d.spy_series.close, spyDates, initialRange);
    const fragRange = riskValues
      ? valuesRange(riskValues, riskDates, initialRange) : null;
    const dailyExtent = dailyValues
      ? valuesRange(dailyValues, dailyDates, initialRange) : null;
    const dailyRange = dailyExtent ? [0, Math.max(100, dailyExtent[1])] : null;
    const thresholdShapes = [];
    if (riskValues) thresholdShapes.push({
      type: "line", xref: "paper", yref: "y2", x0: 0, x1: 1, y0: 50, y1: 50,
      line: { color: "rgba(255,107,53,.75)", width: 1, dash: "dash" },
    });
    if (dailyValues) thresholdShapes.push({
      type: "line", xref: "paper", yref: "y3", x0: 0, x1: 1, y0: 50, y1: 50,
      line: { color: "rgba(255,107,53,.75)", width: 1, dash: "dash" },
    });
    const layout = {
      height: dailyValues ? 510 : 340,
      xaxis: {
        range: initialRange,
        anchor: dailyValues ? "y3" : "y",
        rangebreaks: nonTradingRangebreaks(dailyDates || riskDates || spyDates),
      },
      yaxis: {
        domain: dailyValues ? [0.38, 1] : [0, 1],
        range: spyRange,
        title: { text: "SPY", font: { size: 11 } },
      },
      yaxis2: { overlaying: "y", side: "right", range: fragRange, showgrid: false,
                title: { text: "63d 10d MA", font: { size: 11 } } },
      shapes: thresholdShapes,
      bargap: 0,
    };
    if (dailyValues) {
      layout.yaxis3 = {
        domain: [0, 0.25], range: dailyRange,
        title: { text: "Daily 63d", font: { size: 11 } },
      };
    }
    Plotly.newPlot(chartEl, traces, plotLayout(layout), PLOT_CFG);
    enableFullHistoryReset(chartEl, dailyValues
      ? ["xaxis", "yaxis", "yaxis2", "yaxis3"]
      : ["xaxis", "yaxis", "yaxis2"]);

    // Range presets (default 1Y) — portfolio-page style. The payload ships
    // full dial history (2016+); presets just re-window the axes.
    const seg = document.getElementById("dialRangeSeg");
    if (seg && typeof seg.querySelectorAll === "function") {
      const allDates = (dailyDates || riskDates || spyDates || []).filter(Boolean);
      const lastDate = String((d.asof || allDates[allDates.length - 1] || "")).slice(0, 10);
      const applyRange = preset => {
        let from = null;
        if (preset === "YTD") from = lastDate.slice(0, 4) + "-01-01";
        else if (preset !== "All") {
          const dt = new Date(`${lastDate}T00:00:00Z`);
          dt.setUTCFullYear(dt.getUTCFullYear() - parseInt(preset, 10));
          from = dt.toISOString().slice(0, 10);
        }
        const xr = [from || allDates[0] || lastDate, lastDate];
        const upd = { "xaxis.range": xr };
        const sr = valuesRange(d.spy_series.close, spyDates, xr);
        if (sr) upd["yaxis.range"] = sr;
        if (riskValues) {
          const fr = valuesRange(riskValues, riskDates, xr);
          if (fr) upd["yaxis2.range"] = fr;
        }
        if (dailyValues) {
          const de = valuesRange(dailyValues, dailyDates, xr);
          if (de) upd["yaxis3.range"] = [0, Math.max(100, de[1])];
        }
        Plotly.relayout(chartEl, upd);
      };
      seg.innerHTML = ["All", "5Y", "3Y", "1Y", "YTD"].map(p =>
        `<button${p === "1Y" ? ' class="on"' : ""}>${p}</button>`).join("");
      for (const b of seg.querySelectorAll("button")) {
        b.addEventListener("click", () => {
          for (const x of seg.querySelectorAll("button")) x.classList.remove("on");
          b.classList.add("on");
          applyRange(b.textContent);
        });
      }
    }
  }
}

function sizingHeroHtml(sz) {
  const on = !!sz.throttle_on;
  const gap = sz.gap_to_threshold;
  const gapTxt = gap == null ? "" :
    on ? `${fmt.num(Math.abs(gap), 1)} above threshold` :
    `${fmt.num(Math.abs(gap), 1)} below threshold ${fmt.num(sz.threshold, 0)}`;
  const throttled = Array.isArray(sz.throttled) ? sz.throttled : [];
  const bandCount = Array.isArray(sz.banded_strategies) ? sz.banded_strategies.length : 0;
  const throttleLine = on && throttled.length
    ? throttled.map(t => `<span class="badge on">${esc(t.strategy)} @ ${fmt.num(t.mult, 2)}x</span>`).join(" ")
    : `<span class="cap-inline">all ${bandCount} banded strategies at full size</span>`;
  const expo = sz.exposure;
  const expoLine = expo && expo.mult != null
    ? `Exposure leg: ${fmt.num(expo.mult, 2)}x (${expo.active_rule ? esc(String(expo.active_rule)) : "no rule active"}), as of ${esc(expo.asof || "-")}`
    : "";
  const sleeve = sz.sleeve;
  const sleeveLine = sleeve && sleeve.position
    ? `Clean-air SPY sleeve (paper): <span class="badge ${sleeve.position === "LONG" ? "off" : "warn"}">${esc(sleeve.position)}</span> since ${esc(sleeve.since || "-")} · ${sleeve.n_transitions} transition${sleeve.n_transitions === 1 ? "" : "s"} · enter dial&lt;20 near highs, exit dial&ge;25 or 2 closes outside band`
    : "";
  return `<div class="card sizing-hero">
    <div class="head"><span class="tkr">Sizing State</span>
      <span class="badge ${on ? "on" : "off"}">${on ? "THROTTLE ON" : "THROTTLE OFF"}</span>
      <span class="signal-current">${fmt.num(sz.score, 1)}</span></div>
    <div class="cap">10d MA of the 63d dial — the number that sizes live orders ·
      ${gapTxt} · ${sz.days_in_state != null ? `${sz.days_in_state}d in state` : ""} ·
      as of ${esc(sz.asof || "-")} (append-only PIT series)</div>
    <div class="sizing-throttle">${throttleLine}</div>
    ${expoLine ? `<div class="cap">${expoLine}</div>` : ""}
    ${sleeveLine ? `<div class="cap">${sleeveLine}</div>` : ""}
    <div class="chart sizing-spark" id="sizingSpark"></div>
  </div>`;
}

function renderSizingSpark(sz) {
  const el = document.getElementById("sizingSpark");
  if (!el) return;
  // Payload now carries full dial history; the hero spark stays trailing-1y.
  const cut = Math.max(0, sz.spark.dates.length - 252);
  const dates = sz.spark.dates.slice(cut);
  const sparkMa = sz.spark.ma.slice(cut);
  const shapes = [{
    type: "line", xref: "paper", yref: "y", x0: 0, x1: 1,
    y0: sz.threshold, y1: sz.threshold,
    line: { color: "rgba(255,107,53,.8)", width: 1, dash: "dash" },
  }];
  for (const ep of sz.episodes || []) {
    if (!ep || !ep[0] || !ep[1]) continue;
    shapes.push({
      type: "rect", xref: "x", yref: "paper", layer: "below",
      x0: ep[0], x1: ep[1], y0: 0, y1: 1,
      line: { width: 0 }, fillcolor: "rgba(255,107,53,.16)",
    });
  }
  Plotly.newPlot(el, [{
    x: dates, y: sparkMa, name: "63d 10d-MA",
    mode: "lines", line: { color: "#ffc14d", width: 1.6 },
  }], plotLayout({
    height: 170,
    margin: { l: 36, r: 12, t: 8, b: 28 },
    shapes,
    showlegend: false,
    yaxis: { rangemode: "tozero" },
  }), PLOT_CFG);
}

function kpiRowHtml(d) {
  const frag = d.fragility || {};
  const sz = d.sizing_state;
  const ctx = d.price_ctx || {};
  let cells = `<div class="kpi"><div class="l">SPY</div><div class="v">${fmt.num(d.spy_last, 2)}</div>
      <div class="s">${esc(ctx.regime_label || "")}</div></div>`;
  if (sz && sz.score != null) {
    const on = !!sz.throttle_on;
    const cls = on === null ? "" : on ? "neg" : "pos";
    cells += `<div class="kpi"><div class="l">Sizing Fragility</div>
      <div class="v ${cls}">${fmt.num(sz.score, 1)}</div>
      <div class="s">63d · 10d MA · PIT as of ${esc(sz.asof || "-")} · threshold ${fmt.num(sz.threshold, 0)}</div></div>`;
  } else if (frag["63d"] != null) {
    cells += `<div class="kpi"><div class="l">Fragility 63d</div>
      <div class="v">${Math.round(frag["63d"])}</div>
      <div class="s">display recompute · legacy payload · not a sizing input</div></div>`;
  }
  const chips = ["5d", "21d"].filter(h => frag[h] != null)
    .map(h => `${h} ${Math.round(frag[h])}`).join(" · ");
  if (chips) {
    cells += `<div class="kpi"><div class="l">Context horizons</div>
      <div class="v" style="font-size:15px;padding-top:6px">${chips}</div>
      <div class="s">not sizing inputs</div></div>`;
  }
  cells += `<div class="kpi"><div class="l">Active Signals</div>
      <div class="v ${d.n_active > 0 ? "neg" : "pos"}">${d.n_active} / ${(d.signals || []).length}</div>
      <div class="s">regime mult ${d.regime_mult != null ? Number(d.regime_mult).toFixed(2) + "x" : "-"}</div></div>`;
  const vk = d.vol_kpi;
  if (vk && vk.vix != null) {
    const inverted = vk.term_ratio != null && vk.term_ratio > 1;
    cells += `<div class="kpi"><div class="l">VIX / VIX3M</div>
      <div class="v ${inverted ? "neg" : ""}">${fmt.num(vk.vix, 1)} / ${fmt.num(vk.vix3m, 1)}</div>
      <div class="s">ratio ${vk.term_ratio != null ? fmt.num(vk.term_ratio, 2) : "-"}${inverted ? " (inverted)" : ""}</div></div>`;
  }
  return `<div class="kpis">${cells}</div>`;
}

function contextStripHtml(d) {
  const ctx = d.price_ctx || {};
  const bits = [];
  if (ctx.regime_label) bits.push(esc(String(ctx.regime_label)));
  if (ctx.extension_200d != null) bits.push(`${fmt.pct(ctx.extension_200d, 1)} vs 200d`);
  if (ctx.drawdown != null) bits.push(`${fmt.pct(ctx.drawdown, 1)} off 52w high`);
  if (ctx.ret_12m != null) bits.push(`${fmt.pct(ctx.ret_12m, 1)} over 12m`);
  if (ctx.days_since_5pct != null) bits.push(`${ctx.days_since_5pct}d since 5% pullback`);
  if (ctx.days_since_10pct != null) bits.push(`${ctx.days_since_10pct}d since 10% drawdown`);
  if (!bits.length) return "";
  return `<div class="card context-strip">${bits.map(b => `<span>${b}</span>`).join('<span class="sep">·</span>')}</div>`;
}

function tradeConsoleHtml(tc) {
  if (tc.state === "silent") {
    return `<div class="card trade-console tc-silent">
      <div class="head"><span class="tkr">Trade Console</span>
        <span class="badge warn">WITHHELD</span></div>
      <div class="cap">${esc(tc.reason || "inputs unavailable")}</div></div>`;
  }
  const badgeCls = tc.class_id === "NONE" ? "off"
    : (tc.headline || "").startsWith("ELEVATED") ? "on" : "warn";
  const degraded = tc.state === "degraded";
  // asof-drift guard: a skipped deploy fossilizes "fired N sessions ago"
  const asofMs = Date.parse(`${String(tc.asof || "").slice(0, 10)}T00:00:00Z`);
  const staleDays = Number.isFinite(asofMs)
    ? Math.floor((Date.now() - asofMs) / 86400000) : 0;
  const staleBadge = staleDays > 4
    ? `<span class="badge warn">AS OF ${esc(tc.asof)}</span>` : "";
  return `<div class="card trade-console">
    <div class="head"><span class="tkr">Trade Console</span>
      <span class="badge ${badgeCls}">${esc(tc.headline || "")}</span>
      ${degraded ? '<span class="badge warn">DEGRADED</span>' : ""}${staleBadge}</div>
    ${degraded && tc.reason ? `<div class="cap">${esc(tc.reason)}</div>` : ""}
    <div class="tc-fired">${esc(tc.fired_line || "")}</div>
    ${tc.dist_line ? `<p class="tc-dist">${esc(tc.dist_line)}</p>` : ""}
    ${tc.structure_line ? `<p class="tc-structure">${esc(tc.structure_line)}</p>` : ""}
    ${tc.extra_line ? `<p class="tc-dist">${esc(tc.extra_line)}</p>` : ""}
    <div class="tc-action">${esc(tc.action_line || "")}</div>
    <div class="tc-foot">${(tc.caveats || []).map(esc).join(" · ")}</div>
  </div>`;
}

function nuggetsHtml(nuggets) {
  const toneCls = { good: "off", warn: "warn", bad: "on", info: "conv" };
  const cards = nuggets.map(n => {
    const lines = (n.lines || []).filter(Boolean)
      .map(l => `<p class="nugget-line">${esc(l)}</p>`).join("");
    return `<div class="card nugget">
      <div class="head"><span class="tkr">${esc(n.title || "")}</span>
        <span class="badge ${toneCls[n.tone] || ""}">${esc((n.tone || "info").toUpperCase())}</span></div>
      ${lines}</div>`;
  }).join("");
  return `<h2>Read</h2><div class="nugget-grid">${cards}</div>`;
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

function nonTradingRangebreaks(dates) {
  const valid = (dates || []).filter(Boolean).map(d => String(d).slice(0, 10));
  if (valid.length < 2) return [];
  const present = new Set(valid);
  const start = new Date(`${valid[0]}T00:00:00Z`);
  const end = new Date(`${valid[valid.length - 1]}T00:00:00Z`);
  if (!Number.isFinite(start.getTime()) || !Number.isFinite(end.getTime())) return [];
  const missingWeekdays = [];
  for (const day = new Date(start); day <= end; day.setUTCDate(day.getUTCDate() + 1)) {
    const weekday = day.getUTCDay();
    const iso = day.toISOString().slice(0, 10);
    if (weekday !== 0 && weekday !== 6 && !present.has(iso)) missingWeekdays.push(iso);
  }
  const breaks = [{ bounds: ["sat", "mon"] }];
  if (missingWeekdays.length) breaks.push({ values: missingWeekdays });
  return breaks;
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

function fwdTable(h, r) {
  const head = `<tr>
    <th class="l">Window</th><th>Mean</th><th>Median</th><th>% Neg</th><th>Mean Z</th></tr>`;
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
      <td class="${mCls}">${fmt.signed(mz, 2)}</td></tr>`;
  }
  if (!rows) return "";
  return `<div class="card" style="margin-bottom:12px">
    <div class="cap" style="margin-top:0">${h.toUpperCase()} fragility = ${Math.round(r.current_score)} ·
      ${r.n_episodes} episodes · band ${Math.round(r.band_low)}-${Math.round(r.band_high)}</div>
    <div class="tblwrap"><table class="tbl"><thead>${head}</thead><tbody>${rows}</tbody></table></div>
  </div>`;
}

function drawdownIvFilteredRows(di, horizon, threshold) {
  return ((di.rows_by_horizon || {})[horizon] || [])
    .filter(e => Number(e.max_drawdown_atr) >= threshold);
}

function numericSummary(values) {
  const nums = (values || []).map(Number).filter(Number.isFinite).sort((a, b) => a - b);
  if (!nums.length) return { n: 0, mean: null, median: null };
  const mid = Math.floor(nums.length / 2);
  const median = nums.length % 2 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2;
  return {
    n: nums.length,
    mean: nums.reduce((total, value) => total + value, 0) / nums.length,
    median,
  };
}

function drawdownIvSummaryHtml(di, horizon, threshold) {
  const rows = drawdownIvFilteredRows(di, horizon, threshold);
  const iv = numericSummary(rows.map(row => row.iv_change_points));
  const atr = numericSummary(rows.map(row => row.max_drawdown_atr));
  const ivMean = iv.mean == null ? "&mdash;" : `${fmt.signed(iv.mean, 1)} pts`;
  const ivMedian = iv.median == null ? "&mdash;" : `${fmt.signed(iv.median, 1)} pts`;
  const atrMean = atr.mean == null ? "&mdash;" : `${fmt.num(atr.mean, 2)} ATR`;
  const atrMedian = atr.median == null ? "&mdash;" : `${fmt.num(atr.median, 2)} ATR`;
  return `<div class="risk-dd-stat">
      <div class="risk-dd-stat-head"><span>IV change</span><span>${iv.n} path${iv.n === 1 ? "" : "s"}</span></div>
      <div class="risk-dd-stat-values">
        <div><span>Average</span><b>${ivMean}</b></div>
        <div><span>Median</span><b>${ivMedian}</b></div>
      </div>
    </div>
    <div class="risk-dd-stat">
      <div class="risk-dd-stat-head"><span>ATR drawdown</span><span>${atr.n} path${atr.n === 1 ? "" : "s"}</span></div>
      <div class="risk-dd-stat-values">
        <div><span>Average</span><b>${atrMean}</b></div>
        <div><span>Median</span><b>${atrMedian}</b></div>
      </div>
    </div>`;
}

function drawdownIvRowsHtml(di, horizon, threshold) {
  const rows = drawdownIvFilteredRows(di, horizon, threshold);
  if (!rows.length) {
    return `<tr><td class="l" colspan="8"><span class="cap">No similar-reading paths reached ${fmt.num(threshold, 0)} ATR within ${esc(horizon)}.</span></td></tr>`;
  }
  return rows.map(e => {
    const ivPath = e.iv_start_close != null && e.iv_peak != null
      ? `${fmt.num(e.iv_start_close, 1)} &rarr; <b>${fmt.num(e.iv_peak, 1)}</b>` : "&mdash;";
    const ivDelta = e.iv_change_points != null
      ? `${fmt.signed(e.iv_change_points, 1)} pts${e.iv_change_pct != null ? `<br><span class="cap">${fmt.signed(e.iv_change_pct * 100, 0)}%</span>` : ""}`
      : "&mdash;";
    return `<tr>
      <td class="l"><b>${esc(e.anchor_date || "")}</b><br><span class="cap">SPY close ${fmt.num(e.anchor_spy_close, 2)} &middot; ATR ${fmt.num(e.anchor_atr, 2)}</span></td>
      <td class="l"><b>${esc(e.worst_low_date || "")}</b><br><span class="cap">SPY low ${fmt.num(e.worst_spy_low, 2)}</span></td>
      <td class="neg"><b>${fmt.num(e.max_drawdown_atr, 2)} ATR</b><br><span class="cap">${fmt.pct(e.max_drawdown_pct, 1)}</span></td>
      <td>${e.sessions_to_low != null ? e.sessions_to_low : "&mdash;"}</td>
      <td>${ivPath}</td>
      <td><b>${e.iv_peak != null ? fmt.num(e.iv_peak, 1) : "&mdash;"}</b></td>
      <td class="l">${esc(e.iv_peak_date || "&mdash;")}</td>
      <td class="${e.iv_change_points > 0 ? "neg" : ""}">${ivDelta}</td>
    </tr>`;
  }).join("");
}

function drawdownIvThresholdOptions(di, horizon, selected) {
  const counts = (di.counts && di.counts[horizon]) || {};
  return (di.thresholds || [1, 2, 3, 5]).map(Number).map(t => {
    const n = counts[String(t)] == null ? 0 : counts[String(t)];
    return `<option value="${t}"${Math.abs(t - selected) < 1e-9 ? " selected" : ""}>&ge;${fmt.num(t, 0)} ATR (${n} path${n === 1 ? "" : "s"})</option>`;
  }).join("");
}

function drawdownIvHtml(di) {
  const horizons = di.horizons || Object.keys(di.rows_by_horizon || {});
  const selectedHorizon = horizons.includes(di.default_horizon) ? di.default_horizon : horizons[0];
  const selectedThreshold = Number(di.default_threshold || 2);
  const horizonOptions = horizons.map(h => {
    const n = (di.eligible_by_horizon || {})[h];
    const suffix = n == null ? "" : ` (${n} complete)`;
    return `<option value="${esc(h)}"${h === selectedHorizon ? " selected" : ""}>${esc(h)}${suffix}</option>`;
  }).join("");
  const thresholdOptions = drawdownIvThresholdOptions(di, selectedHorizon, selectedThreshold);
  return `<h2>Peak IV after similar risk readings</h2>
    <div class="card risk-dd-iv">
      <div class="tbl-controls">
        <label for="ddIvHorizon">Forward window</label>
        <select id="ddIvHorizon">${horizonOptions}</select>
        <label for="ddIvThreshold">Minimum downside</label>
        <select id="ddIvThreshold">${thresholdOptions}</select>
      </div>
      <div class="risk-dd-summary" id="ddIvSummary">${drawdownIvSummaryHtml(di, selectedHorizon, selectedThreshold)}</div>
      <div class="cap">Same ${di.n_episodes == null ? "" : di.n_episodes + " declustered "}historical anchors as the 63d similar-fragility table above: current score ${fmt.num(di.current_score, 1)}, band ${fmt.num(di.band_low, 1)}&ndash;${fmt.num(di.band_high, 1)}. A row appears only when that analog reached the selected ATR downside inside the selected window.</div>
      <div class="tblwrap"><table class="tbl"><thead><tr>
        <th class="l">Similar-reading close</th><th class="l">Worst SPY low</th><th>Max DD</th>
        <th>Sessions</th><th>VIX close &rarr; peak</th><th>Peak IV</th><th class="l">Peak IV date</th><th>IV change</th>
      </tr></thead><tbody id="ddIvRows">${drawdownIvRowsHtml(di, selectedHorizon, selectedThreshold)}</tbody></table></div>
      <div class="cap">Max DD = (analog-date SPY close &minus; lowest subsequent intraday SPY low in-window) / analog-date Wilder ATR(${di.atr_period || 14}). VIX change runs from the analog-date close to the maximum ${esc(di.iv_basis || "VIX reading")} through the worst-low date. Peak IV is descriptive, not a forecast.</div>
    </div>`;
}

function initDrawdownIvTable(di) {
  const horizonSelect = document.getElementById("ddIvHorizon");
  const select = document.getElementById("ddIvThreshold");
  const summary = document.getElementById("ddIvSummary");
  const body = document.getElementById("ddIvRows");
  if (!horizonSelect || !select || !summary || !body || typeof select.addEventListener !== "function") return;
  const renderRows = () => {
    const horizon = horizonSelect.value || di.default_horizon || "63d";
    const threshold = Number(select.value);
    const selectedThreshold = Number.isFinite(threshold) ? threshold : 2;
    summary.innerHTML = drawdownIvSummaryHtml(di, horizon, selectedThreshold);
    body.innerHTML = drawdownIvRowsHtml(di, horizon, selectedThreshold);
  };
  horizonSelect.addEventListener("change", () => {
    const selected = Number(di.default_threshold || 2);
    select.innerHTML = drawdownIvThresholdOptions(di, horizonSelect.value, selected);
    select.value = String(selected);
    renderRows();
  });
  select.addEventListener("change", renderRows);
  renderRows();
}

// ---- ATR downside tables (signal cards + dial band) ----
// Cell = P(SPY intraday low reaches >= k*ATR below the fire/anchor close within
// the window), %. Bold = conditional; grey below = all-market baseline; the cell
// turns red when the conditional runs >= 8pp above baseline (materially more
// downside than a normal day).
function atrCellsHtml(table, baseline, mults, horizons) {
  const head = `<tr><th class="l">Window</th>` +
    mults.map(k => `<th>&ge;${k} ATR</th>`).join("") + `</tr>`;
  let rows = "";
  for (const h of horizons) {
    const row = (table && table[h]) || {};
    const base = (baseline && baseline[h]) || {};
    let tds = `<td class="l">${esc(h)}</td>`;
    for (const k of mults) {
      const v = row[String(k)];
      const b = base[String(k)];
      if (v == null) { tds += `<td>&mdash;</td>`; continue; }
      const cls = (b != null && v - b >= 8) ? "neg" : "";
      const baseHtml = (b == null) ? "" : `<span class="atr-base">${Math.round(b)}</span>`;
      tds += `<td class="${cls}"><b>${Math.round(v)}</b>${baseHtml}</td>`;
    }
    rows += `<tr>${tds}</tr>`;
  }
  return `<div class="tblwrap"><table class="tbl atr-tbl"><thead>${head}</thead><tbody>${rows}</tbody></table></div>`;
}

function atrSignalTableHtml(name, ad) {
  if (!ad || !ad.signals || !ad.signals[name] || !ad.signals[name].episode) return "";
  const s = ad.signals[name];
  const cap = `Downside after a fresh ${esc(name)} trigger &middot; ${s.n_episodes} episodes since ${esc(ad.data_from || "")} &middot; ` +
    `bold = P(low reaches &ge;k&middot;ATR under the close in-window); <span class="atr-base" style="display:inline">grey</span> = all-market baseline`;
  return `<div class="card atr-card">
    <div class="cap" style="margin-top:0">${cap}</div>
    ${atrCellsHtml(s.episode, ad.baseline, ad.mults, ad.horizons)}</div>`;
}

function atrDialTableHtml(ad) {
  if (!ad || !ad.dial || !ad.dial.table) return "";
  const dl = ad.dial;
  const n10 = dl.n_by_h && dl.n_by_h["10d"] != null ? `${dl.n_by_h["10d"]} days` : "";
  const cap = `Downside when the dial sits here &middot; dial-MA within &plusmn;${dl.band} of ${fmt.num(dl.value, 1)} ` +
    `(band ${fmt.num(dl.lo, 1)}&ndash;${fmt.num(dl.hi, 1)})${n10 ? " &middot; " + n10 : ""}${dl.band_from ? " since " + esc(dl.band_from) : ""} &middot; ` +
    `bold = P(low reaches &ge;k&middot;ATR under the close in-window); <span class="atr-base" style="display:inline">grey</span> = all-market baseline`;
  return `<div class="card atr-card atr-dial-card">
    <div class="cap" style="margin-top:0">${cap}</div>
    ${atrCellsHtml(dl.table, ad.baseline, ad.mults, ad.horizons)}</div>`;
}

function detailText(v) {
  if (typeof v === "string") return v;
  if (typeof v === "object" && v != null)
    return Object.entries(v).map(([k, x]) => `${k}: ${x}`).join(" · ");
  return String(v);
}
function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
