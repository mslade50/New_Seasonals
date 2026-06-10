/* risk.js — render data/risk.json (condensed risk_dashboard_v2 summary) */
"use strict";

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
      <div class="s">${esc(ctx.regime || ctx.label || "")}</div></div>
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

  // signals
  html += `<h2>Signals</h2>`;
  for (const s of d.signals || []) {
    const badgeCls = s.on ? "on" : (s.badge || "").startsWith("DECAYING") ? "warn" : "off";
    html += `<div class="card idea">
      <div class="head"><span class="tkr">${esc(s.name)}</span>
        <span class="badge ${badgeCls}">${esc(s.badge)}</span></div>
      ${s.detail ? `<div class="cap">${esc(detailText(s.detail))}</div>` : ""}
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

  el.innerHTML = html;

  if (d.spy_series) {
    const traces = [{
      x: d.spy_series.dates, y: d.spy_series.close, name: "SPY",
      mode: "lines", line: { color: "#4da3ff", width: 1.6 },
    }];
    const fs = d.fragility_series;
    if (fs && fs["21d"]) {
      traces.push({
        x: fs.dates, y: fs["21d"], name: "Fragility 21d", yaxis: "y2",
        mode: "lines", line: { color: "#ffc14d", width: 1.2 },
      });
    }
    Plotly.newPlot(document.getElementById("riskChart"), traces, plotLayout({
      height: 340,
      yaxis: { title: { text: "SPY", font: { size: 11 } } },
      yaxis2: { overlaying: "y", side: "right", range: [0, 100], showgrid: false,
                title: { text: "Fragility", font: { size: 11 } } },
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
