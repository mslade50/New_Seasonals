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

  if (d.curves && d.curves.dates && d.curves.dates.length) {
    html += `
      <h2>Equity-curve what-if</h2>
      <p class="cap">Pick a sweep, a strategy, and an entry value. Left: the strategy's
      cumulative realized PnL, baseline vs variant. Right: the WHOLE BOOK with that single
      strategy swapped to the variant (baseline book − strategy baseline + strategy variant;
      additive flat $750k, realized at exit). Variant trades come from a run where the whole
      sweep scope moved together, so cross-strategy cap interactions are approximate.</p>
      <div class="filters" id="cvControls"></div>
      <div class="kpis" id="cvKpis"></div>
      <div class="grid2">
        <div class="card"><h3>Strategy — baseline vs variant</h3>
          <div id="cvStrat" style="height:330px"></div></div>
        <div class="card"><h3>Book — baseline vs with-variant</h3>
          <div id="cvBook" style="height:330px"></div></div>
      </div>`;
  }

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

  if (d.curves && d.curves.dates && d.curves.dates.length) initCurveLab(d);
  (d.sweeps || []).forEach((sw, i) => renderSweep(sw, i));
}

/* ---------- equity-curve what-if ---------- */
function initCurveLab(d) {
  const cv = d.curves;
  const sweeps = (d.sweeps || []).filter(sw => {
    const dim = cv.variants[sw.dimension];
    return dim && Object.keys(dim).length;
  });
  if (!sweeps.length) return;

  const host = document.getElementById("cvControls");
  const mkSel = label => {
    const box = document.createElement("span");
    const l = document.createElement("label");
    l.textContent = label;
    box.appendChild(l);
    const sel = document.createElement("select");
    sel.className = "btn";
    box.appendChild(sel);
    host.appendChild(box);
    return sel;
  };
  const swSel = mkSel("Sweep"), stSel = mkSel("Strategy"), vSel = mkSel("Value");

  const fill = (sel, opts, cur) => {
    sel.innerHTML = "";
    for (const o of opts) {
      const el2 = document.createElement("option");
      el2.value = o.value; el2.textContent = o.label;
      sel.appendChild(el2);
    }
    if (cur != null && opts.some(o => String(o.value) === String(cur))) sel.value = cur;
  };

  const state = { si: 0, strat: null, val: null };

  function syncStrats() {
    const sw = sweeps[state.si];
    const names = (sw.strategy_scope || []).filter(n => {
      const dim = cv.variants[sw.dimension];
      return Object.values(dim).some(pt => pt[n]) && cv.strategy_base[n];
    });
    fill(stSel, names.map(n => ({ value: n, label: n })), state.strat);
    state.strat = stSel.value;
    syncVals();
  }
  function syncVals() {
    const sw = sweeps[state.si];
    const vals = Object.keys(cv.variants[sw.dimension]);
    vals.sort((a, b) => parseFloat(a) - parseFloat(b));
    const prodV = (sw.prod_values || {})[state.strat];
    fill(vSel, vals.map(v => ({
      value: v,
      label: v + (prodV != null && String(prodV) === v ? " (prod)" : ""),
    })), state.val != null ? state.val : (prodV != null ? String(prodV) : vals[0]));
    state.val = vSel.value;
    render();
  }
  swSel.addEventListener("change", () => { state.si = +swSel.value; state.val = null; syncStrats(); });
  stSel.addEventListener("change", () => { state.strat = stSel.value; state.val = null; syncVals(); });
  vSel.addEventListener("change", () => { state.val = vSel.value; render(); });

  fill(swSel, sweeps.map((sw, i) => ({ value: i, label: sw.label || sw.dimension })));
  syncStrats();

  function cum(arr) {
    let s = 0;
    return arr.map(v => (s += v));
  }
  function maxDD(c) {
    let peak = -Infinity, dd = 0;
    for (const v of c) { if (v > peak) peak = v; if (peak - v > dd) dd = peak - v; }
    return dd;
  }

  function render() {
    const sw = sweeps[state.si];
    const varArr = ((cv.variants[sw.dimension] || {})[state.val] || {})[state.strat];
    const baseArr = cv.strategy_base[state.strat];
    if (!varArr || !baseArr) return;
    const dates = cv.dates;
    const sBase = cum(baseArr), sVar = cum(varArr);
    const bBase = cum(cv.book_base);
    const bVar = cum(cv.book_base.map((v, i) => v - baseArr[i] + varArr[i]));
    const isProd = String((sw.prod_values || {})[state.strat]) === String(state.val);

    const dEnd = sVar[sVar.length - 1] - sBase[sBase.length - 1];
    const ddB = maxDD(bBase), ddV = maxDD(bVar);
    document.getElementById("cvKpis").innerHTML = [
      kpi("Variant", `${state.val}${isProd ? " (prod)" : ""}`),
      kpi("Strategy Δ PnL (= book Δ)", fmt.money(dEnd), clsSign(dEnd)),
      kpi("Strategy total, base → var",
          `${fmt.money(sBase[sBase.length - 1])} → ${fmt.money(sVar[sVar.length - 1])}`),
      kpi("Book max DD, base → var",
          `${fmt.money(-ddB)} → ${fmt.money(-ddV)}`,
          ddV < ddB ? "pos" : ddV > ddB ? "neg" : ""),
    ].join("");

    Plotly.react(document.getElementById("cvStrat"), [
      { x: dates, y: sBase, mode: "lines", name: "baseline (prod)",
        line: { color: "#4da3ff", width: 1.5 } },
      { x: dates, y: sVar, mode: "lines", name: `variant ${state.val}`,
        line: { color: "#ffc14d", width: 1.5 } },
    ], plotLayout({ yaxis: { tickformat: "$,.3~s" }, margin: { t: 8 } }), PLOT_CFG);

    Plotly.react(document.getElementById("cvBook"), [
      { x: dates, y: bBase, mode: "lines", name: "book baseline",
        line: { color: "#00d18f", width: 1.5 } },
      { x: dates, y: bVar, mode: "lines", name: `book w/ ${state.strat} @ ${state.val}`,
        line: { color: "#b07cff", width: 1.5 } },
    ], plotLayout({ yaxis: { tickformat: "$,.3~s" }, margin: { t: 8 } }), PLOT_CFG);
  }
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
