/* heatmaps.js — corrected browser-side market and cross-asset state maps.

   The legacy "Correlation Heatmaps" page was a conditional forward-outcome
   surface, not a correlation calculation. This port keeps that useful surface
   and adds actual rolling daily-return correlations. All feature ranks are
   expanding and point-in-time; unsupported cells stay blank rather than being
   fabricated by display smoothing. */
"use strict";

(() => {
  const FEATURES = {
    ret5Rank: "5d return rank",
    ret10Rank: "10d return rank",
    ret21Rank: "21d return rank",
    ret63Rank: "63d return rank",
    ret252Rank: "252d return rank",
    rv21Rank: "21d realized-vol rank",
    rv63Rank: "63d realized-vol rank",
    volume10Rank: "10d relative-volume rank",
    volume21Rank: "21d relative-volume rank",
  };
  const SPY_FEATURES = {
    spyRet5Rank: "SPY 5d return rank",
    spyRet10Rank: "SPY 10d return rank",
    spyRet21Rank: "SPY 21d return rank",
  };
  const TARGETS = {
    fwd1: "1d forward return",
    fwd5: "5d forward return",
    fwd10: "10d forward return",
    fwd21: "21d forward return",
    fwd63: "63d forward return",
  };
  const CACHE = new Map();
  const FEATURE_CACHE = new Map();
  const COLORS = [[0, "#ff5d5d"], [0.5, "#171d29"], [1, "#00d18f"]];

  document.addEventListener("DOMContentLoaded", initHeatmapPage);

  const finite = value => typeof value === "number" && Number.isFinite(value);
  const esc = value => String(value).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  async function initHeatmapPage() {
    const root = document.getElementById("heatmap-app");
    if (!root) return;
    const manifest = await fetchJSONOrNull("data/seasonality/manifest.json");
    if (!manifest || !manifest.tickers || !globalThis.SeasonalityMath) {
      root.innerHTML = '<div class="fetchfail">Heatmap price inputs are unavailable in this build.</div>';
      return;
    }
    const tickers = Object.keys(manifest.tickers).sort();
    const mode = document.body.dataset.page;
    root.innerHTML = mode === "shared-correlations"
      ? correlationShell(tickers.length, manifest.asof)
      : inspectorShell(tickers.length, manifest.asof);
    wireTickerAutocomplete(root.querySelector("#hm-ticker1"), root.querySelector("#hm-ticker1-options"), tickers);
    if (mode === "shared-correlations") {
      wireTickerAutocomplete(root.querySelector("#hm-ticker2"), root.querySelector("#hm-ticker2-options"), tickers);
      root.querySelector("#hm-run").addEventListener("click", () => runCorrelations(root, manifest));
    } else {
      root.querySelector("#hm-run").addEventListener("click", () => runInspector(root, manifest));
    }
  }

  function optionMarkup(options, selected) {
    return Object.entries(options).map(([key, label]) =>
      `<option value="${key}"${key === selected ? " selected" : ""}>${esc(label)}</option>`).join("");
  }

  function tickerField(id, value, label) {
    return `<div class="seasonality-ticker-combobox hm-ticker-field"><label><span>${esc(label)}</span>
      <input id="${id}" value="${esc(value)}" autocomplete="off" spellcheck="false" role="combobox"
        aria-autocomplete="list" aria-expanded="false" aria-controls="${id}-options"></label>
      <div id="${id}-options" class="seasonality-ticker-options" role="listbox" hidden></div></div>`;
  }

  function sharedSettings() {
    return `<label><span>History begins</span><select id="hm-start">
        <option value="2000">All available</option><option value="2005" selected>2005</option>
        <option value="2010">2010</option><option value="2015">2015</option><option value="2020">2020</option>
      </select></label>
      <label><span>Grid bins</span><input id="hm-bins" type="number" min="8" max="30" value="15"></label>
      <label><span>Minimum observations / cell</span><input id="hm-min-count" type="number" min="2" max="30" value="8"></label>
      <label><span>Count-weighted smoothing</span><input id="hm-sigma" type="number" min="0" max="2" step="0.1" value="0.8"></label>`;
  }

  function inspectorShell(count, asof) {
    return `<div class="card hm-method-note"><b>Corrected methodology</b><span>Point-in-time expanding ranks; adjusted closes;
      maximum-edge observations retained; low-support cells hidden; smoothing weighted by cell counts.</span></div>
    <div class="card hm-controls"><div class="hm-control-grid">
      ${tickerField("hm-ticker1", "SMH", "Ticker")}
      <label><span>X-axis state</span><select id="hm-x">${optionMarkup({...FEATURES, ...SPY_FEATURES}, "ret21Rank")}</select></label>
      <label><span>Y-axis state</span><select id="hm-y">${optionMarkup({...FEATURES, ...SPY_FEATURES}, "rv21Rank")}</select></label>
      <label><span>Forward outcome</span><select id="hm-z">${optionMarkup(TARGETS, "fwd5")}</select></label>
      ${sharedSettings()}<button class="btn primary" id="hm-run">Generate heatmap</button>
    </div><div class="cap">${count.toLocaleString()} tickers · adjusted prices and volume through ${esc(asof)} · browser-only analysis</div></div>
    <div id="hm-status" class="seasonality-status cap">Choose a ticker and generate the map.</div>
    <div id="hm-results"></div>`;
  }

  function correlationShell(count, asof) {
    return `<div class="card hm-method-note"><b>What changed</b><span>The rolling chart is genuine return correlation.
      The lower chart preserves the original page's analysis under its accurate name: a conditional outcome map.</span></div>
    <div class="card hm-controls"><div class="hm-control-grid">
      ${tickerField("hm-ticker1", "SPY", "Target ticker")}
      ${tickerField("hm-ticker2", "TLT", "Signal ticker")}
      <label><span>Target state (X)</span><select id="hm-x">${optionMarkup(FEATURES, "ret21Rank")}</select></label>
      <label><span>Signal state (Y)</span><select id="hm-y">${optionMarkup(FEATURES, "ret21Rank")}</select></label>
      <label><span>Target outcome</span><select id="hm-z">${optionMarkup(TARGETS, "fwd21")}</select></label>
      ${sharedSettings()}<button class="btn primary" id="hm-run">Run cross-asset analysis</button>
    </div><div class="cap">${count.toLocaleString()} tickers · adjusted prices and volume through ${esc(asof)} · exact-date alignment</div></div>
    <div id="hm-status" class="seasonality-status cap">Choose two tickers and run the analysis.</div>
    <div id="hm-results"></div>`;
  }

  function wireTickerAutocomplete(input, menu, tickers) {
    let active = -1;
    const close = () => {
      menu.hidden = true; menu.innerHTML = ""; active = -1;
      input.setAttribute("aria-expanded", "false");
    };
    const choose = ticker => { input.value = ticker; close(); input.focus(); };
    const show = () => {
      const query = input.value.trim().toUpperCase();
      if (!query) { close(); return; }
      const prefix = tickers.filter(ticker => ticker.startsWith(query));
      const rest = prefix.length >= 12 ? [] : tickers.filter(ticker => !ticker.startsWith(query) && ticker.includes(query));
      const matches = prefix.concat(rest).slice(0, 12);
      if (!matches.length || (matches.length === 1 && matches[0] === query)) { close(); return; }
      menu.innerHTML = matches.map(ticker => `<button type="button" role="option" data-ticker="${esc(ticker)}">${esc(ticker)}</button>`).join("");
      menu.hidden = false; input.setAttribute("aria-expanded", "true"); active = -1;
      menu.querySelectorAll("button").forEach(button => {
        button.addEventListener("mousedown", event => event.preventDefault());
        button.addEventListener("click", () => choose(button.dataset.ticker));
      });
    };
    const move = direction => {
      const buttons = Array.from(menu.querySelectorAll("button"));
      if (!buttons.length) return;
      active = (active + direction + buttons.length) % buttons.length;
      buttons.forEach((button, i) => button.classList.toggle("on", i === active));
      buttons[active].scrollIntoView({ block: "nearest" });
    };
    input.addEventListener("input", () => { input.value = input.value.toUpperCase(); show(); });
    input.addEventListener("focus", show);
    input.addEventListener("blur", () => window.setTimeout(close, 100));
    input.addEventListener("keydown", event => {
      if (["ArrowDown", "ArrowUp"].includes(event.key)) {
        event.preventDefault(); if (menu.hidden) show(); move(event.key === "ArrowDown" ? 1 : -1);
      } else if (event.key === "Enter" && !menu.hidden && active >= 0) {
        event.preventDefault(); choose(menu.querySelectorAll("button")[active].dataset.ticker);
      } else if (event.key === "Escape") close();
    });
  }

  async function loadTicker(ticker, manifest) {
    if (CACHE.has(ticker)) return CACHE.get(ticker);
    const meta = manifest.tickers[ticker];
    if (!meta) throw new Error(`${ticker} is not in the price snapshot`);
    const response = await fetch(`data/seasonality/${meta.file}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`${ticker} price snapshot returned HTTP ${response.status}`);
    const rows = SeasonalityMath.enrichSeasonalityRows(SeasonalityMath.decodeSeasonalityBuffer(await response.arrayBuffer()));
    CACHE.set(ticker, rows);
    return rows;
  }

  function lowerBound(array, value) {
    let low = 0, high = array.length;
    while (low < high) { const mid = (low + high) >> 1; if (array[mid] < value) low = mid + 1; else high = mid; }
    return low;
  }

  function expandingRank(values, minimum = 252) {
    const unique = Array.from(new Set(values.filter(finite))).sort((a, b) => a - b);
    const tree = new Int32Array(unique.length + 1);
    const add = index => { for (let i = index + 1; i < tree.length; i += i & -i) tree[i]++; };
    const sum = index => { let out = 0; for (let i = index + 1; i > 0; i -= i & -i) out += tree[i]; return out; };
    let seen = 0;
    return values.map(value => {
      if (!finite(value)) return null;
      const index = lowerBound(unique, value);
      add(index); seen++;
      const less = index ? sum(index - 1) : 0;
      const equal = sum(index) - less;
      return seen >= minimum ? (less + (equal + 1) / 2) / seen * 100 : null;
    });
  }

  function trailingReturn(closes, window) {
    return closes.map((close, i) => i >= window && closes[i - window] > 0 ? close / closes[i - window] - 1 : null);
  }

  function rollingMean(values, window) {
    let total = 0, valid = 0;
    return values.map((value, i) => {
      if (finite(value)) { total += value; valid++; }
      if (i >= window && finite(values[i - window])) { total -= values[i - window]; valid--; }
      return i >= window - 1 && valid === window ? total / window : null;
    });
  }

  function rollingStd(values, window) {
    return values.map((_, i) => {
      if (i < window - 1) return null;
      const sample = values.slice(i - window + 1, i + 1);
      if (!sample.every(finite)) return null;
      const mean = sample.reduce((sum, value) => sum + value, 0) / window;
      return Math.sqrt(sample.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (window - 1));
    });
  }

  function forwardReturn(closes, window) {
    return closes.map((close, i) => i + window < closes.length && close > 0 ? (closes[i + window] / close - 1) * 100 : null);
  }

  function buildFeatures(ticker, rows) {
    if (FEATURE_CACHE.has(ticker)) return FEATURE_CACHE.get(ticker);
    const closes = rows.map(row => row.close);
    const volumes = rows.map(row => finite(row.volume) && row.volume > 0 ? row.volume : null);
    const logReturns = closes.map((close, i) => i && closes[i - 1] > 0 ? Math.log(close / closes[i - 1]) : null);
    const columns = {};
    for (const window of [5, 10, 21, 63, 252]) columns[`ret${window}Rank`] = expandingRank(trailingReturn(closes, window));
    for (const window of [21, 63]) columns[`rv${window}Rank`] = expandingRank(rollingStd(logReturns, window));
    const volume63 = rollingMean(volumes, 63);
    for (const window of [10, 21]) {
      const short = rollingMean(volumes, window);
      columns[`volume${window}Rank`] = expandingRank(short.map((value, i) => finite(value) && finite(volume63[i]) && volume63[i] > 0 ? value / volume63[i] : null));
    }
    for (const window of [1, 5, 10, 21, 63]) columns[`fwd${window}`] = forwardReturn(closes, window);
    const output = rows.map((row, i) => {
      const feature = { date: row.date, close: row.close };
      for (const [key, values] of Object.entries(columns)) feature[key] = values[i];
      return feature;
    });
    FEATURE_CACHE.set(ticker, output);
    return output;
  }

  function quantile(sorted, probability) {
    if (!sorted.length) return null;
    const position = (sorted.length - 1) * probability;
    const low = Math.floor(position), high = Math.ceil(position), fraction = position - low;
    return sorted[low] * (1 - fraction) + sorted[high] * fraction;
  }

  function quantileEdges(values, bins) {
    const sorted = values.filter(finite).sort((a, b) => a - b);
    const edges = [];
    for (let i = 0; i <= bins; i++) {
      const value = quantile(sorted, i / bins);
      if (!edges.length || value > edges[edges.length - 1]) edges.push(value);
    }
    return edges;
  }

  function binIndex(value, edges) {
    if (!finite(value) || edges.length < 2 || value < edges[0] || value > edges[edges.length - 1]) return -1;
    if (value === edges[edges.length - 1]) return edges.length - 2;
    return Math.max(0, Math.min(edges.length - 2, lowerBound(edges, value) - 1));
  }

  function smoothGrid(sums, counts, sigma, minimum) {
    const height = counts.length, width = counts[0].length;
    const output = Array.from({ length: height }, () => Array(width).fill(null));
    const radius = sigma > 0 ? Math.max(1, Math.ceil(sigma * 2)) : 0;
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      if (counts[y][x] < minimum) continue;
      if (!radius) { output[y][x] = sums[y][x] / counts[y][x]; continue; }
      let weightedSum = 0, weightedCount = 0;
      for (let yy = Math.max(0, y - radius); yy <= Math.min(height - 1, y + radius); yy++) {
        for (let xx = Math.max(0, x - radius); xx <= Math.min(width - 1, x + radius); xx++) {
          if (!counts[yy][xx]) continue;
          const distance = (xx - x) ** 2 + (yy - y) ** 2;
          const weight = Math.exp(-distance / (2 * sigma ** 2));
          weightedSum += sums[yy][xx] * weight;
          weightedCount += counts[yy][xx] * weight;
        }
      }
      output[y][x] = weightedCount ? weightedSum / weightedCount : null;
    }
    return output;
  }

  function buildGrid(records, xKey, yKey, zKey, bins, minimum, sigma) {
    const clean = records.filter(row => finite(row[xKey]) && finite(row[yKey]) && finite(row[zKey]));
    const xEdges = quantileEdges(clean.map(row => row[xKey]), bins);
    const yEdges = quantileEdges(clean.map(row => row[yKey]), bins);
    if (xEdges.length < 3 || yEdges.length < 3) throw new Error("The selected features do not have enough distinct values");
    const width = xEdges.length - 1, height = yEdges.length - 1;
    const sums = Array.from({ length: height }, () => Array(width).fill(0));
    const counts = Array.from({ length: height }, () => Array(width).fill(0));
    for (const row of clean) {
      const x = binIndex(row[xKey], xEdges), y = binIndex(row[yKey], yEdges);
      if (x < 0 || y < 0) continue;
      sums[y][x] += row[zKey]; counts[y][x]++;
    }
    const z = smoothGrid(sums, counts, sigma, minimum);
    const visible = z.flat().filter(finite);
    const scale = Math.max(quantile(visible.map(Math.abs).sort((a, b) => a - b), 0.95) || 0, 0.05);
    return {
      clean, xEdges, yEdges, counts, z, scale,
      x: xEdges.slice(0, -1).map((edge, i) => (edge + xEdges[i + 1]) / 2),
      y: yEdges.slice(0, -1).map((edge, i) => (edge + yEdges[i + 1]) / 2),
    };
  }

  function settings(root) {
    return {
      ticker1: root.querySelector("#hm-ticker1").value.trim().toUpperCase(),
      ticker2: root.querySelector("#hm-ticker2")?.value.trim().toUpperCase(),
      x: root.querySelector("#hm-x").value, y: root.querySelector("#hm-y").value,
      z: root.querySelector("#hm-z").value, start: Number(root.querySelector("#hm-start").value),
      bins: Number(root.querySelector("#hm-bins").value), minimum: Number(root.querySelector("#hm-min-count").value),
      sigma: Number(root.querySelector("#hm-sigma").value),
    };
  }

  async function runInspector(root, manifest) {
    const cfg = settings(root), status = root.querySelector("#hm-status"), results = root.querySelector("#hm-results");
    status.textContent = `Loading ${cfg.ticker1}…`; results.innerHTML = "";
    try {
      const [rows, spyRows] = await Promise.all([loadTicker(cfg.ticker1, manifest), loadTicker("SPY", manifest)]);
      const records = buildFeatures(cfg.ticker1, rows).filter(row => Number(row.date.slice(0, 4)) >= cfg.start);
      const spy = new Map(buildFeatures("SPY", spyRows).map(row => [row.date, row]));
      for (const row of records) {
        const match = spy.get(row.date);
        row.spyRet5Rank = match?.ret5Rank ?? null; row.spyRet10Rank = match?.ret10Rank ?? null;
        row.spyRet21Rank = match?.ret21Rank ?? null;
      }
      const grid = buildGrid(records, cfg.x, cfg.y, cfg.z, cfg.bins, cfg.minimum, cfg.sigma);
      results.innerHTML = resultShell("Conditional forward-outcome heatmap", "hm-main-chart", grid, cfg);
      renderGrid("hm-main-chart", grid, cfg, cfg.ticker1, cfg.ticker1, records);
      status.textContent = `${cfg.ticker1}: ${grid.clean.length.toLocaleString()} completed daily anchors since ${cfg.start}.`;
    } catch (error) { status.innerHTML = `<span class="neg">${esc(error.message)}</span>`; }
  }

  function resultShell(title, chartId, grid, cfg) {
    const supported = grid.z.flat().filter(finite).length;
    return `<div class="kpis"><div class="kpi"><div class="l">Completed anchors</div><div class="v">${grid.clean.length.toLocaleString()}</div></div>
      <div class="kpi"><div class="l">Supported cells</div><div class="v">${supported.toLocaleString()}</div><div class="s">N ≥ ${cfg.minimum}</div></div>
      <div class="kpi"><div class="l">Displayed scale</div><div class="v">±${grid.scale.toFixed(2)}%</div><div class="s">robust 95th percentile</div></div></div>
      <div class="card hm-chart-card"><h2>${esc(title)}</h2><div id="${chartId}" class="chart hm-main-chart"></div></div>
      <p class="cap">Daily forward windows overlap, so anchor count is not an independent-sample count. This is an exploratory conditional average, not a causal estimate.</p>`;
  }

  function renderGrid(id, grid, cfg, target, signal, records) {
    const xLabel = cfg.xLabel || {...FEATURES, ...SPY_FEATURES}[cfg.x];
    const yLabel = cfg.yLabel || FEATURES[cfg.y] || SPY_FEATURES[cfg.y];
    const current = records.slice().reverse().find(row => finite(row[cfg.x]) && finite(row[cfg.y]));
    const traces = [{
      type: "heatmap", x: grid.x, y: grid.y, z: grid.z,
      customdata: grid.counts.map(row => row.map(count => [count])), colorscale: COLORS,
      zmin: -grid.scale, zmax: grid.scale, zmid: 0, colorbar: { title: "Mean %" },
      hovertemplate: `${esc(xLabel)}: %{x:.1f}<br>${esc(yLabel)}: %{y:.1f}<br>Mean outcome: %{z:.2f}%<br>Raw N: %{customdata[0]}<extra></extra>`,
    }];
    if (current) traces.push({ type: "scatter", mode: "markers", x: [current[cfg.x]], y: [current[cfg.y]],
      name: `Current · ${current.date}`, marker: { color: "#fff", size: 10, symbol: "diamond", line: { color: "#111", width: 1.5 } },
      hovertemplate: `<b>Current ${current.date}</b><br>${target}: %{x:.1f}<br>${signal}: %{y:.1f}<extra></extra>` });
    Plotly.newPlot(id, traces, plotLayout({
      height: 610, margin: { l: 65, r: 25, t: 25, b: 55 },
      xaxis: { title: `${target}: ${xLabel}`, gridcolor: "#1c2230" },
      yaxis: { title: `${signal}: ${yLabel}`, gridcolor: "#1c2230" },
      legend: { orientation: "h", y: 1.08, x: 0 },
    }), PLOT_CFG);
  }

  function alignedReturns(rows1, rows2, start) {
    const returns = rows => new Map(rows.map((row, i) => [row.date,
      i && rows[i - 1].close > 0 ? Math.log(row.close / rows[i - 1].close) : null]));
    const left = returns(rows1), right = returns(rows2), output = [];
    for (const [date, a] of left) if (Number(date.slice(0, 4)) >= start && right.has(date) && finite(a) && finite(right.get(date))) {
      output.push({ date, a, b: right.get(date) });
    }
    return output.sort((a, b) => a.date.localeCompare(b.date));
  }

  function rollingCorrelation(pairs, window) {
    return pairs.map((pair, i) => {
      if (i < window - 1) return { date: pair.date, value: null };
      const sample = pairs.slice(i - window + 1, i + 1);
      const meanA = sample.reduce((sum, row) => sum + row.a, 0) / window;
      const meanB = sample.reduce((sum, row) => sum + row.b, 0) / window;
      let cov = 0, varA = 0, varB = 0;
      for (const row of sample) { cov += (row.a - meanA) * (row.b - meanB); varA += (row.a - meanA) ** 2; varB += (row.b - meanB) ** 2; }
      return { date: pair.date, value: varA > 0 && varB > 0 ? cov / Math.sqrt(varA * varB) : null };
    });
  }

  async function runCorrelations(root, manifest) {
    const cfg = settings(root), status = root.querySelector("#hm-status"), results = root.querySelector("#hm-results");
    if (!cfg.ticker1 || !cfg.ticker2 || cfg.ticker1 === cfg.ticker2) {
      status.innerHTML = '<span class="neg">Choose two different tickers.</span>'; return;
    }
    status.textContent = `Aligning ${cfg.ticker1} and ${cfg.ticker2}…`; results.innerHTML = "";
    try {
      const [rows1, rows2] = await Promise.all([loadTicker(cfg.ticker1, manifest), loadTicker(cfg.ticker2, manifest)]);
      const pairs = alignedReturns(rows1, rows2, cfg.start);
      const windows = [21, 63, 126], correlations = Object.fromEntries(windows.map(w => [w, rollingCorrelation(pairs, w)]));
      const t1 = new Map(buildFeatures(cfg.ticker1, rows1).map(row => [row.date, row]));
      const t2 = new Map(buildFeatures(cfg.ticker2, rows2).map(row => [row.date, row]));
      const records = [];
      for (const [date, left] of t1) {
        const right = t2.get(date);
        if (!right || Number(date.slice(0, 4)) < cfg.start) continue;
        records.push({ date, close: left.close, xState: left[cfg.x], yState: right[cfg.y], outcome: left[cfg.z] });
      }
      const surfaceCfg = {
        ...cfg, x: "xState", y: "yState", z: "outcome",
        xLabel: FEATURES[cfg.x], yLabel: FEATURES[cfg.y],
      };
      const grid = buildGrid(records, surfaceCfg.x, surfaceCfg.y, surfaceCfg.z,
        surfaceCfg.bins, surfaceCfg.minimum, surfaceCfg.sigma);
      const latest = window => correlations[window].slice().reverse().find(row => finite(row.value));
      results.innerHTML = `<div class="kpis">${windows.map(window => {
        const row = latest(window); return `<div class="kpi"><div class="l">${window}d correlation</div><div class="v">${row ? row.value.toFixed(2) : "—"}</div><div class="s">${row?.date || ""}</div></div>`;
      }).join("")}</div>
        <div class="card hm-chart-card"><h2>Rolling adjusted-close return correlation</h2><div id="hm-corr-chart" class="chart hm-corr-chart"></div></div>
        ${resultShell("Cross-asset conditional outcome map", "hm-main-chart", grid, surfaceCfg)}`;
      Plotly.newPlot("hm-corr-chart", windows.map((window, i) => ({
        type: "scatter", mode: "lines", name: `${window}d`,
        x: correlations[window].map(row => row.date), y: correlations[window].map(row => row.value),
        line: { color: ["#4da3ff", "#ffbf4d", "#00d18f"][i], width: window === 63 ? 2.4 : 1.5 },
        hovertemplate: `%{x}<br>${window}d correlation: %{y:.2f}<extra></extra>`,
      })), plotLayout({
        height: 390, margin: { l: 55, r: 20, t: 25, b: 40 },
        yaxis: { title: "Pearson correlation", range: [-1, 1], gridcolor: "#1c2230", zeroline: true },
        xaxis: { gridcolor: "#1c2230" }, legend: { orientation: "h", y: 1.08, x: 0 },
      }), PLOT_CFG);
      renderGrid("hm-main-chart", grid, surfaceCfg, cfg.ticker1, cfg.ticker2, records);
      status.textContent = `${cfg.ticker1} vs ${cfg.ticker2}: ${pairs.length.toLocaleString()} aligned return sessions since ${cfg.start}.`;
    } catch (error) { status.innerHTML = `<span class="neg">${esc(error.message)}</span>`; }
  }

  globalThis.HeatmapMath = { expandingRank, quantileEdges, binIndex, buildGrid, rollingCorrelation };
})();
