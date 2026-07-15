/* macro_seasonal.js — browser-side port of pages/macro_seasonality.py.

   The table comes pre-computed from data/seasonality/macro.json (built by
   scripts/macro_site_data.py). Charts are computed here from the same
   per-ticker binaries the Seasonality Lab uses, via the shared helpers in
   seasonality.js. Read-only: no yfinance, no write path. */
"use strict";

const MACRO_TABLE_COLUMNS = [
  ["price", "Price", "price"],
  ["r5", "5d_r", "ret"], ["s5", "Sznl_5", "sznl"], ["s10", "Sznl_10", "sznl"],
  ["r20", "20d_r", "ret"], ["s21", "Sznl_21", "sznl"],
  ["r50", "50d_r", "ret"], ["s63", "Sznl_63", "sznl"], ["s126", "Sznl_126", "sznl"],
  ["r200", "200d_r", "ret"], ["s252", "Sznl_252", "sznl"],
];

document.addEventListener("DOMContentLoaded", initMacroSeasonality);

function macroReturnClass(value) {
  if (!isFiniteNumber(value)) return "";
  return value > 90 ? "hi-red" : value < 15 ? "hi-green" : "";
}

function macroSznlClass(value) {
  if (!isFiniteNumber(value)) return "";
  return value > 85 ? "hi-green" : value < 15 ? "hi-red" : "";
}

function macroCell(value, kind) {
  if (!isFiniteNumber(value)) return "<td>—</td>";
  if (kind === "price") return `<td>${value.toFixed(2)}</td>`;
  const cls = kind === "ret" ? macroReturnClass(value) : macroSznlClass(value);
  return `<td${cls ? ` class="${cls}"` : ""}>${value.toFixed(1)}</td>`;
}

async function initMacroSeasonality() {
  const root = document.getElementById("macro-seasonality");
  if (!root) return;
  const payload = await fetchJSONOrNull("data/seasonality/macro.json");
  if (!payload || !Array.isArray(payload.rows) || !payload.rows.length) {
    root.innerHTML = '<div class="fetchfail">The macro seasonality payload is not present in this build. ' +
      "Rebuild the site with the master-price snapshot and atr_seasonal_ranks.parquet to enable this tab.</div>";
    return;
  }
  renderMacroSeasonality(root, payload);
}

function renderMacroSeasonality(root, payload) {
  const cycle = defaultCycle();
  const warning = payload.sznl_available ? "" :
    '<div class="fetchfail">Seasonal ranks (Sznl_*) are empty — atr_seasonal_ranks.parquet was not ' +
    "available when this build ran. MA-extension ranks and charts are unaffected.</div>";

  const head = "<tr><th class=\"l\">Ticker</th><th class=\"l\">Name</th><th>IBKR</th>" +
    MACRO_TABLE_COLUMNS.map(col => `<th>${col[1]}</th>`).join("") + "</tr>";
  const body = payload.rows.map((row, i) => {
    const clickable = row.file ? ` class="macro-row" data-chart="macro-chart-${i}"` : "";
    return `<tr${clickable}><td class="l strong">${labEsc(row.ticker)}</td>` +
      `<td class="l">${labEsc(row.name || "")}</td><td>${labEsc(row.ibkr || "")}</td>` +
      MACRO_TABLE_COLUMNS.map(col => macroCell(row[col[0]], col[2])).join("") + "</tr>";
  }).join("");

  const noPrices = payload.rows.filter(row => !row.file).map(row => row.ticker);
  const chartCards = payload.rows.map((row, i) => {
    if (!row.file) return "";
    const label = row.name ? `${row.ticker} — ${row.name}` : row.ticker;
    return `<div class="card seasonality-chart-card macro-chart-card" id="macro-chart-${i}"
      data-file="${labEsc(row.file)}" data-ticker="${labEsc(row.ticker)}" data-label="${labEsc(label)}">
      <div class="chart macro-chart"><div class="spin">…</div></div></div>`;
  }).join("");

  root.innerHTML = `${warning}
    <div class="cap" style="margin:4px 2px 8px">Prices through ${labEsc(payload.asof || "?")} ·
      seasonal ranks as-of ${labEsc(payload.sznl_asof || "?")} · returns rank vs their trailing 2y
      distribution (red = extended, green = washed out) · Sznl above 85 = seasonally strong window
      (green), below 15 = weak (red). Click a row to jump to its chart.</div>
    <div class="card"><div class="tblwrap"><table class="tbl macro-tbl"><thead>${head}</thead>
      <tbody>${body}</tbody></table></div></div>
    <h2 style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">Seasonal charts
      <span class="cap" style="display:inline">${labEsc(cycle)} cycle avg + All Years + realized, ATR units, 20yr half-life</span>
      <label class="cap" style="display:inline-flex;align-items:center;gap:6px;cursor:pointer">
        <input type="checkbox" id="macro-show-pct" style="accent-color:var(--accent)">Overlay % return</label>
    </h2>
    ${noPrices.length ? `<p class="cap">No price history in the snapshot (table ranks only): ${noPrices.map(labEsc).join(", ")}.</p>` : ""}
    <div class="macro-chart-grid">${chartCards}</div>`;

  root.querySelectorAll(".macro-row").forEach(row => row.addEventListener("click", () => {
    const target = document.getElementById(row.dataset.chart);
    if (target) target.scrollIntoView({ behavior: "smooth", block: "center" });
  }));

  const state = { cache: new Map(), rendered: new Set(), showPct: false };
  const cards = Array.from(root.querySelectorAll(".macro-chart-card"));
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      observer.unobserve(entry.target);
      renderMacroChart(entry.target, state, cycle);
    }
  }, { rootMargin: "600px 0px" });
  cards.forEach(card => observer.observe(card));

  const pctToggle = root.querySelector("#macro-show-pct");
  pctToggle.addEventListener("change", () => {
    state.showPct = pctToggle.checked;
    for (const card of state.rendered) renderMacroChart(card, state, cycle);
  });
}

async function macroRows(state, file) {
  let promise = state.cache.get(file);
  if (!promise) {
    promise = fetch(`data/seasonality/${file}`, { cache: "no-store" })
      .then(response => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.arrayBuffer();
      })
      .then(buffer => enrichSeasonalityRows(decodeSeasonalityBuffer(buffer)));
    state.cache.set(file, promise);
  }
  return promise;
}

function macroTrace(path, name, axis, dateFor, opts) {
  const hover = axis === "y2"
    ? "<b>%{customdata[0]}</b><br>Day: %{x}<br>Cumulative ATR: %{y:.2f}<extra></extra>"
    : "<b>%{customdata[0]}</b><br>Day: %{x}<br>Return: %{y:.2%}<extra></extra>";
  return { x: path.map(p => p.day), y: path.map(p => p.value), type: "scatter", mode: "lines",
    name, yaxis: axis, customdata: path.map(p => [dateFor(p)]), hovertemplate: hover,
    line: { color: opts.color, width: opts.width || 2, dash: opts.dash || "solid" } };
}

async function renderMacroChart(card, state, cycle) {
  const target = card.querySelector(".macro-chart");
  try {
    const rows = await macroRows(state, card.dataset.file);
    const spin = target.querySelector(".spin");
    if (spin) spin.remove();
    const currentYear = new Date().getFullYear();
    const history = rows.filter(row => row.year < currentYear);
    const current = rows.filter(row => row.year === currentYear);
    const dateMap = buildDateMap(rows, currentYear);
    const markerDay = markerDayFor(rows, currentYear, new Date(), null);
    const mapped = point => dateMap.get(point.day) || `Day ${point.day}`;
    const actual = point => point.date
      ? new Date(`${point.date}T00:00:00Z`).toLocaleString("en-US", { month: "short", day: "2-digit", timeZone: "UTC" })
      : mapped(point);

    const atrCycle = calculateSeasonalPath(history, cycle, "atrReturn", 20, currentYear);
    const traces = [
      macroTrace(atrCycle, `${cycle} ATR`, "y2", mapped, { color: "#ff8c00" }),
      macroTrace(calculateSeasonalPath(history, "All Years", "atrReturn", 20, currentYear),
        "All Years ATR", "y2", mapped, { color: "#87cefa", width: 1, dash: "dot" }),
      macroTrace(cumulativeActual(current, "atrReturn"), `${currentYear} ATR`, "y2", actual,
        { color: "#39ff14" }),
    ];
    let pctCycle = [];
    if (state.showPct) {
      pctCycle = calculateSeasonalPath(history, cycle, "logReturn", 20, currentYear);
      traces.push(
        macroTrace(pctCycle, `${cycle} %`, "y", mapped, { color: "#ff8c00", width: 1, dash: "dash" }),
        macroTrace(calculateSeasonalPath(history, "All Years", "logReturn", 20, currentYear),
          "All Years %", "y", mapped, { color: "#87cefa", width: 1, dash: "dot" }),
        macroTrace(cumulativeActual(current, "logReturn"), `${currentYear} %`, "y", actual,
          { color: "#39ff14", width: 1, dash: "dash" }));
    }
    if (markerDay != null) {
      const atrMarker = pathValue(atrCycle, markerDay);
      if (atrMarker != null) traces.push({ x: [markerDay], y: [atrMarker], yaxis: "y2", type: "scatter",
        mode: "markers", showlegend: false, hoverinfo: "skip",
        marker: { color: "#fff", size: 8, symbol: "diamond", line: { color: "#000", width: 1 } } });
      const pctMarker = pathValue(pctCycle, markerDay);
      if (pctMarker != null) traces.push({ x: [markerDay], y: [pctMarker], yaxis: "y", type: "scatter",
        mode: "markers", showlegend: false, hoverinfo: "skip",
        marker: { color: "#fff", size: 7, line: { color: "#000", width: 1 } } });
    }
    const layout = plotLayout({
      title: { text: card.dataset.label, x: 0.02, xanchor: "left", font: { size: 13 } },
      height: 400, margin: { l: 46, r: 52, t: 40, b: 30 },
      xaxis: { gridcolor: "#1c2230" },
      yaxis: { tickformat: ".1%", showgrid: false, visible: state.showPct },
      yaxis2: { title: "Cum ATR", overlaying: "y", side: "right", tickformat: ".1f", gridcolor: "#1c2230" },
      legend: { orientation: "h", y: -0.14, x: 0, xanchor: "left", font: { size: 10 } },
    });
    Plotly.newPlot(target, traces, layout, PLOT_CFG);
    state.rendered.add(card);
  } catch (error) {
    target.innerHTML = `<div class="fetchfail">Chart failed for ${labEsc(card.dataset.ticker)}: ${labEsc(error.message)}</div>`;
  }
}

globalThis.MacroSeasonalMath = { macroReturnClass, macroSznlClass, macroCell };
