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

  if (d.curves && d.curves.format === "trades" && d.curves.dates && d.curves.dates.length) {
    html += `
      <h2>Equity-curve what-if</h2>
      <p class="cap">Pick a sweep, a strategy, an entry value, and a window. Left: the
      strategy's cumulative realized PnL, prod vs variant. Right: the WHOLE BOOK with that
      single strategy swapped to the variant (baseline book − strategy baseline + strategy
      variant; additive flat $750k, realized at exit). Stats below mirror the portfolio
      page on the same realized basis — daily metrics count every trading day in the
      window (zero-PnL days included); time in market = trading days with ≥1 open position.
      Variant trades come from a run where the whole sweep scope moved together, so
      cross-strategy cap interactions are approximate.</p>
      <div class="filters" id="cvControls"></div>
      <div class="grid2">
        <div class="card"><h3>Strategy — prod vs variant</h3>
          <div id="cvStrat" style="height:330px"></div></div>
        <div class="card"><h3>Book — prod vs with-variant</h3>
          <div id="cvBook" style="height:330px"></div></div>
      </div>
      <div class="card"><h3>Window stats — prod vs variant</h3><div id="cvStats"></div></div>`;
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

  if (d.curves && d.curves.format === "trades" && d.curves.dates && d.curves.dates.length) {
    initCurveLab(d);
  }
  (d.sweeps || []).forEach((sw, i) => renderSweep(sw, i));
}

function kpi(label, value, cls) {
  return `<div class="kpi"><div class="l">${esc(label)}</div>
    <div class="v ${cls || ""}">${value}</div></div>`;
}

/* ---------- equity-curve what-if ----------
   curves.format === "trades": each series is [entry_idx, exit_idx, pnl, r]
   into the shared curves.dates trading-day calendar. Everything below is
   additive, so book-with-variant = book − strategy-baseline + variant holds
   componentwise for trade counts, win counts, gross win/loss, daily PnL and
   open-position counts alike. */
function initCurveLab(d) {
  const cv = d.curves;
  const N = cv.dates.length;
  const EQ = 750000;
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

  const state = { si: 0, strat: null, val: null, range: "All" };

  // range preset segment (mirrors the portfolio page's Range control)
  const rBox = document.createElement("span");
  const rLbl = document.createElement("label");
  rLbl.textContent = "Range";
  rBox.appendChild(rLbl);
  const rSeg = document.createElement("span");
  rSeg.className = "seg";
  for (const v of ["All", "10Y", "5Y", "3Y", "1Y"]) {
    const b = document.createElement("button");
    b.textContent = v;
    if (v === state.range) b.classList.add("on");
    b.addEventListener("click", () => {
      rSeg.querySelectorAll("button").forEach(x => x.classList.remove("on"));
      b.classList.add("on");
      state.range = v;
      render();
    });
    rSeg.appendChild(b);
  }
  rBox.appendChild(rSeg);
  host.appendChild(rBox);

  const fill = (sel, opts, cur) => {
    sel.innerHTML = "";
    for (const o of opts) {
      const el2 = document.createElement("option");
      el2.value = o.value; el2.textContent = o.label;
      sel.appendChild(el2);
    }
    if (cur != null && opts.some(o => String(o.value) === String(cur))) sel.value = cur;
  };

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

  function windowBounds() {
    if (state.range === "All") return [0, N - 1];
    const yrs = parseInt(state.range, 10);
    const c = new Date(cv.dates[N - 1] + "T00:00:00Z");
    c.setUTCFullYear(c.getUTCFullYear() - yrs);
    const cut = c.toISOString().slice(0, 10);
    let lo = 0, hi = N;
    while (lo < hi) { const m = (lo + hi) >> 1; cv.dates[m] < cut ? lo = m + 1 : hi = m; }
    return [Math.min(lo, N - 1), N - 1];
  }

  /* additive window aggregate over a trade list. PnL is realized on the exit
     day (trades whose exit falls inside the window); time-in-market counts any
     overlap of the entry..exit span with the window. */
  function agg(trades, w0, w1) {
    const len = w1 - w0 + 1;
    const pnlByDay = new Float64Array(len);
    const open = new Int32Array(len + 1);   // diff array of open-position count
    let n = 0, wins = 0, grossWin = 0, grossLoss = 0, sumR = 0, nR = 0, totPnl = 0;
    for (const t of trades) {
      const e = t[0], x = t[1], p = t[2], r = t[3];
      if (x >= w0 && x <= w1) {
        n++; totPnl += p;
        if (p > 0) { wins++; grossWin += p; }
        else if (p < 0) grossLoss += -p;
        if (r != null) { sumR += r; nR++; }
        pnlByDay[x - w0] += p;
      }
      const a = Math.max(e, w0), b = Math.min(x, w1);
      if (a <= b) { open[a - w0]++; open[b - w0 + 1]--; }
    }
    return { n, wins, grossWin, grossLoss, sumR, nR, totPnl, pnlByDay, open };
  }

  /* book-with-variant = book − strategyBaseline + variant, componentwise */
  function combine(bk, sb, sv) {
    const len = bk.pnlByDay.length;
    const pnlByDay = new Float64Array(len);
    const open = new Int32Array(bk.open.length);
    for (let i = 0; i < len; i++) pnlByDay[i] = bk.pnlByDay[i] - sb.pnlByDay[i] + sv.pnlByDay[i];
    for (let i = 0; i < bk.open.length; i++) open[i] = bk.open[i] - sb.open[i] + sv.open[i];
    return {
      n: bk.n - sb.n + sv.n,
      wins: bk.wins - sb.wins + sv.wins,
      grossWin: bk.grossWin - sb.grossWin + sv.grossWin,
      grossLoss: bk.grossLoss - sb.grossLoss + sv.grossLoss,
      sumR: bk.sumR - sb.sumR + sv.sumR,
      nR: bk.nR - sb.nR + sv.nR,
      totPnl: bk.totPnl - sb.totPnl + sv.totPnl,
      pnlByDay, open,
    };
  }

  function metrics(a) {
    const len = a.pnlByDay.length;
    let s = 0, s2 = 0, dneg = 0, nneg = 0;
    let cum = 0, peak = 0, dd = 0;
    for (let i = 0; i < len; i++) {
      const r = a.pnlByDay[i] / EQ;
      s += r; s2 += r * r;
      if (r < 0) { dneg += r * r; nneg++; }
      cum += a.pnlByDay[i];
      if (cum > peak) peak = cum;
      if (peak - cum > dd) dd = peak - cum;
    }
    const m = len ? s / len : 0;
    const varr = len > 1 ? (s2 - len * m * m) / (len - 1) : 0;
    const sd = varr > 0 ? Math.sqrt(varr) : 0;
    const dsd = nneg > 1 ? Math.sqrt(dneg / nneg) : 0;
    let inMkt = 0, run = 0;
    for (let i = 0; i < len; i++) { run += a.open[i]; if (run > 0) inMkt++; }
    return {
      n: a.n,
      win: a.n ? a.wins / a.n : null,
      totR: a.sumR,
      avgR: a.nR ? a.sumR / a.nR : null,
      pf: a.grossLoss > 0 ? a.grossWin / a.grossLoss : null,
      expct: a.n ? a.totPnl / a.n : null,
      totPnl: a.totPnl,
      annRet: m * 252,
      annVol: sd * Math.sqrt(252),
      sharpe: sd ? m / sd * Math.sqrt(252) : null,
      sortino: dsd ? m / dsd * Math.sqrt(252) : null,
      maxDD: dd,
      tim: len ? inMkt / len : null,
    };
  }

  function cumArr(a) {
    let s = 0;
    return Array.from(a.pnlByDay, v => (s += v));
  }

  function statCell(v, base, fmtFn, deltaFmt, invert) {
    if (v == null) return "<td>—</td>";
    let out = fmtFn(v);
    if (base != null) {
      const dv = v - base;
      const good = invert ? dv < 0 : dv > 0;
      const cls = Math.abs(dv) < 1e-12 ? "" : good ? "pos" : "neg";
      out += ` <span class="${cls}" style="font-size:11px">(${dv >= 0 ? "+" : ""}${deltaFmt(dv)})</span>`;
    }
    return `<td>${out}</td>`;
  }

  function render() {
    const sw = sweeps[state.si];
    const varTr = ((cv.variants[sw.dimension] || {})[state.val] || {})[state.strat];
    const baseTr = cv.strategy_base[state.strat];
    if (!varTr || !baseTr) return;
    const [w0, w1] = windowBounds();
    const dates = cv.dates.slice(w0, w1 + 1);
    const isProd = String((sw.prod_values || {})[state.strat]) === String(state.val);
    const vLabel = state.val + (isProd ? " (prod)" : "");

    const aSB = agg(baseTr, w0, w1);
    const aSV = agg(varTr, w0, w1);
    const aBB = agg(cv.book_base, w0, w1);
    const aBV = combine(aBB, aSB, aSV);
    const mSB = metrics(aSB), mSV = metrics(aSV), mBB = metrics(aBB), mBV = metrics(aBV);

    // signal-level diff: trades whose underlying SIGNAL (t[4]) fills only in
    // one world. A signal filling at a different price under the variant is
    // the same signal, not a new trade — the sig_id diff excludes it.
    const baseSig = new Set(baseTr.map(t => t[4]));
    const varSig = new Set(varTr.map(t => t[4]));
    const newTr = varTr.filter(t => !baseSig.has(t[4]));
    const lostTr = baseTr.filter(t => !varSig.has(t[4]));
    const aNew = agg(newTr, w0, w1);
    const aLost = agg(lostTr, w0, w1);

    Plotly.react(document.getElementById("cvStrat"), [
      { x: dates, y: cumArr(aSB), mode: "lines", name: "prod",
        line: { color: "#4da3ff", width: 1.5 } },
      { x: dates, y: cumArr(aSV), mode: "lines", name: `variant ${state.val}`,
        line: { color: "#ffc14d", width: 1.5 } },
      { x: dates, y: cumArr(aNew), mode: "lines",
        name: `new trades only (${aNew.n}, ${fmt.money(aNew.totPnl)})`,
        line: { color: "#3ddbd9", width: 1.3, dash: "dot" } },
      { x: dates, y: cumArr(aLost), mode: "lines", visible: "legendonly",
        name: `removed vs prod (${aLost.n}, ${fmt.money(aLost.totPnl)})`,
        line: { color: "#ff5d5d", width: 1.3, dash: "dot" } },
    ], plotLayout({ yaxis: { tickformat: "$,.3~s" }, margin: { t: 8 } }), PLOT_CFG);

    Plotly.react(document.getElementById("cvBook"), [
      { x: dates, y: cumArr(aBB), mode: "lines", name: "book prod",
        line: { color: "#00d18f", width: 1.5 } },
      { x: dates, y: cumArr(aBV), mode: "lines", name: `book w/ ${state.val}`,
        line: { color: "#b07cff", width: 1.5 } },
    ], plotLayout({ yaxis: { tickformat: "$,.3~s" }, margin: { t: 8 } }), PLOT_CFG);

    // metric rows: [label, key, fmt, deltaFmt, lowerIsBetter]
    const money0 = v => fmt.money(v);
    const rows = [
      ["Trades", "n", v => v.toLocaleString(), v => v.toLocaleString(), false],
      ["Win rate", "win", v => fmt.pct(v, 1), v => fmt.pct(v, 1), false],
      ["Total R", "totR", v => fmt.num(v, 1), v => fmt.num(v, 1), false],
      ["Avg R", "avgR", v => fmt.num(v, 3), v => fmt.num(v, 3), false],
      ["Profit factor", "pf", v => fmt.num(v, 2), v => fmt.num(v, 2), false],
      ["Expectancy / trade", "expct", money0, money0, false],
      ["Total PnL", "totPnl", money0, money0, false],
      ["Ann return (of $750k)", "annRet", v => fmt.pct(v, 1), v => fmt.pct(v, 1), false],
      ["Ann vol", "annVol", v => fmt.pct(v, 1), v => fmt.pct(v, 1), true],
      ["Sharpe", "sharpe", v => fmt.num(v, 2), v => fmt.num(v, 2), false],
      ["Sortino", "sortino", v => fmt.num(v, 2), v => fmt.num(v, 2), false],
      ["Max DD (flat $)", "maxDD", v => fmt.money(-v), money0, true],
      ["Time in market", "tim", v => fmt.pct(v, 1), v => fmt.pct(v, 1), false],
    ];
    let t = `<div class="tblwrap"><table class="tbl"><thead><tr>
      <th class="l">Metric</th><th>Strategy prod</th><th>Strategy @ ${esc(vLabel)}</th>
      <th>Book prod</th><th>Book @ ${esc(vLabel)}</th></tr></thead><tbody>`;
    for (const [label, key, f, df, inv] of rows) {
      t += `<tr><td class="l">${esc(label)}</td>` +
        `<td>${mSB[key] == null ? "—" : f(mSB[key])}</td>` +
        statCell(mSV[key], mSB[key], f, df, inv) +
        `<td>${mBB[key] == null ? "—" : f(mBB[key])}</td>` +
        statCell(mBV[key], mBB[key], f, df, inv) +
        `</tr>`;
    }
    t += "</tbody></table></div>";
    t += `<p class="cap">Window: ${dates[0]} → ${dates[dates.length - 1]} (${dates.length}
      trading days). Trades counted by exit date inside the window; open spans clipped to it
      for time-in-market. Ann vol / Max DD deltas are colored green when LOWER.
      Signal diff in this window: ${aNew.n} new fills under the variant
      (${fmt.money(aNew.totPnl)}), ${aLost.n} prod fills removed (${fmt.money(aLost.totPnl)})
      — the dotted teal trace charts the new fills; the removed-fills trace is in the
      legend (click to show).</p>`;
    document.getElementById("cvStats").innerHTML = t;
  }
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
