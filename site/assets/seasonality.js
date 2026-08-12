/* seasonality.js — browser-side port of pages/user_input.py.

   The only network reads are static files generated from master_prices.parquet:
     data/seasonality/manifest.json
     data/seasonality/t/<ticker-id>.bin

   All paths, ranks, forward statistics, analogs, and similarity calculations
   happen in the browser. There is no yfinance call and no write path. */
"use strict";

const SEASONAL_CYCLES = {
  "Election": 1952,
  "Pre-Election": 1951,
  "Post-Election": 1953,
  "Midterm": 1950,
};
const SEASONAL_HORIZONS = [5, 10, 21];
const SEASONAL_COLORS = ["#ff8c00", "#00ffff", "#fcd12a", "#ff4dff", "#87cefa",
  "#ff6347", "#adff2f", "#ff69b4", "#dda0dd"];

document.addEventListener("DOMContentLoaded", initSeasonalityLab);

function labEsc(value) {
  return String(value).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function compactDate(value) {
  const s = String(value || "");
  return /^\d{8}$/.test(s) ? `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)}` : s.slice(0, 10);
}

function normalizeSeasonalityPayload(payload) {
  if (!payload || !Array.isArray(payload.d) || !Array.isArray(payload.c) || !Array.isArray(payload.a)) {
    throw new Error("Malformed seasonality price payload");
  }
  if (payload.d.length !== payload.c.length || payload.d.length !== payload.a.length) {
    throw new Error("Seasonality payload columns have different lengths");
  }
  const rows = [];
  for (let i = 0; i < payload.d.length; i++) {
    const close = Number(payload.c[i]);
    if (!Number.isFinite(close) || close <= 0) continue;
    const atrValue = payload.a[i] == null ? null : Number(payload.a[i]);
    rows.push({
      date: compactDate(payload.d[i]),
      close,
      atr: Number.isFinite(atrValue) && atrValue > 0 ? atrValue : null,
    });
  }
  rows.sort((a, b) => a.date.localeCompare(b.date));
  return rows;
}

function decodeSeasonalityBuffer(buffer) {
  const view = new DataView(buffer);
  if (view.byteLength < 8 || String.fromCharCode(...new Uint8Array(buffer, 0, 4)) !== "SLB1") {
    throw new Error("Malformed seasonality binary payload");
  }
  const count = view.getUint32(4, true);
  const expected = 8 + count * 12;
  if (view.byteLength !== expected) throw new Error("Seasonality binary payload has an invalid length");
  const dayOffset = 8;
  const closeOffset = dayOffset + count * 4;
  const atrOffset = closeOffset + count * 4;
  const rows = new Array(count);
  for (let i = 0; i < count; i++) {
    const epochDay = view.getInt32(dayOffset + i * 4, true);
    const close = view.getFloat32(closeOffset + i * 4, true);
    const atr = view.getFloat32(atrOffset + i * 4, true);
    rows[i] = {
      date: new Date(epochDay * 86400000).toISOString().slice(0, 10),
      close,
      atr: Number.isFinite(atr) && atr > 0 ? atr : null,
    };
  }
  return rows;
}

function enrichSeasonalityRows(input) {
  let previousClose = null;
  let previousYear = null;
  let day = 0;
  return input.map(row => {
    const year = Number(row.date.slice(0, 4));
    day = year === previousYear ? day + 1 : 1;
    const logReturn = previousClose == null ? null : Math.log(row.close / previousClose);
    const atrReturn = previousClose == null || !isFiniteNumber(row.atr) || row.atr <= 0
      ? null : (row.close - previousClose) / row.atr;
    const dailyPct = previousClose == null ? null : row.close / previousClose - 1;
    previousClose = row.close;
    previousYear = year;
    return Object.assign({}, row, { year, day, logReturn, atrReturn, dailyPct });
  });
}

function recencyWeight(year, anchorYear, halfLife) {
  if (!halfLife || halfLife <= 0) return 1;
  return Math.pow(0.5, (anchorYear - year) / halfLife);
}

function weightedMean(values, weights) {
  let numerator = 0;
  let denominator = 0;
  for (let i = 0; i < values.length; i++) {
    if (!isFiniteNumber(values[i]) || !isFiniteNumber(weights[i])) continue;
    numerator += values[i] * weights[i];
    denominator += weights[i];
  }
  return denominator > 0 ? numerator / denominator : null;
}

function weightedMedian(values, weights) {
  const pairs = [];
  for (let i = 0; i < values.length; i++) {
    if (isFiniteNumber(values[i]) && isFiniteNumber(weights[i])) pairs.push([values[i], weights[i]]);
  }
  if (!pairs.length) return null;
  pairs.sort((a, b) => a[0] - b[0]);
  const total = pairs.reduce((sum, pair) => sum + pair[1], 0);
  if (total <= 0) return null;
  let cumulative = 0;
  for (const pair of pairs) {
    cumulative += pair[1];
    if (cumulative >= total / 2) return pair[0];
  }
  return pairs[pairs.length - 1][0];
}

function isCycleYear(year, label) {
  if (label === "All Years") return true;
  const start = SEASONAL_CYCLES[label];
  return start != null && ((year - start) % 4 + 4) % 4 === 0;
}

function calculateSeasonalPath(rows, cycleLabel, returnKey, halfLife, anchorYear) {
  const grouped = new Map();
  for (const row of rows) {
    if (!isCycleYear(row.year, cycleLabel) || !isFiniteNumber(row[returnKey])) continue;
    const weight = recencyWeight(row.year, anchorYear, halfLife);
    const agg = grouped.get(row.day) || { numerator: 0, denominator: 0 };
    agg.numerator += row[returnKey] * weight;
    agg.denominator += weight;
    grouped.set(row.day, agg);
  }
  let cumulative = 0;
  return Array.from(grouped.keys()).sort((a, b) => a - b).map(day => {
    const agg = grouped.get(day);
    cumulative += agg.numerator / agg.denominator;
    return { day, value: returnKey === "logReturn" ? Math.exp(cumulative) - 1 : cumulative };
  });
}

function percentileRanks(dayValues) {
  const pairs = Array.from(dayValues.entries()).filter(pair => isFiniteNumber(pair[1]));
  pairs.sort((a, b) => a[1] - b[1]);
  const ranks = new Map();
  let i = 0;
  while (i < pairs.length) {
    let j = i + 1;
    while (j < pairs.length && pairs[j][1] === pairs[i][1]) j++;
    const averageOneBasedRank = ((i + 1) + j) / 2;
    const percentile = pairs.length ? averageOneBasedRank / pairs.length * 100 : null;
    for (let k = i; k < j; k++) ranks.set(pairs[k][0], percentile);
    i = j;
  }
  return ranks;
}

function nearestFilled(profile, maxDay) {
  const keys = Array.from(profile.keys()).sort((a, b) => a - b);
  const out = new Map();
  if (!keys.length) {
    for (let day = 1; day <= maxDay; day++) out.set(day, 50);
    return out;
  }
  const low = keys[0], high = keys[keys.length - 1];
  for (let day = 1; day <= maxDay; day++) {
    if (profile.has(day)) {
      out.set(day, profile.get(day));
      continue;
    }
    if (day < low || day > high) {
      out.set(day, 50);
      continue;
    }
    let nearest = keys[0];
    for (const key of keys) {
      if (Math.abs(key - day) < Math.abs(nearest - day)) nearest = key;
      if (key > day && Math.abs(key - day) > Math.abs(nearest - day)) break;
    }
    out.set(day, profile.get(nearest));
  }
  return out;
}

function rankProfileFor(rows, rowSelector, cutoffYear, halfLife, maxDay) {
  const profiles = [];
  for (const horizon of SEASONAL_HORIZONS) {
    const grouped = new Map();
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      if (!rowSelector(row)) continue;
      const future = rows[i + horizon];
      if (!future || !isFiniteNumber(row.close) || !isFiniteNumber(future.close)) continue;
      const value = Math.log(future.close / row.close);
      const weight = recencyWeight(row.year, cutoffYear, halfLife);
      const agg = grouped.get(row.day) || { numerator: 0, denominator: 0 };
      agg.numerator += value * weight;
      agg.denominator += weight;
      grouped.set(row.day, agg);
    }
    const means = new Map();
    for (const [day, agg] of grouped) means.set(day, agg.numerator / agg.denominator);
    profiles.push(nearestFilled(percentileRanks(means), maxDay));
  }
  const average = new Map();
  for (let day = 1; day <= maxDay; day++) {
    average.set(day, profiles.reduce((sum, profile) => sum + profile.get(day), 0) / profiles.length);
  }
  return average;
}

function calculateSeasonalRank(rows, cycleLabel, cutoffYear, halfLife) {
  const history = rows.filter(row => row.year < cutoffYear);
  if (!history.length) return new Map();
  const maxDay = Math.max(253, ...history.map(row => row.day));
  const all = rankProfileFor(history, () => true, cutoffYear, halfLife, maxDay);
  const hasCycleHistory = history.some(row => isCycleYear(row.year, cycleLabel));
  const cycle = cycleLabel === "All Years" || !hasCycleHistory ? all : rankProfileFor(
    history, row => isCycleYear(row.year, cycleLabel), cutoffYear, halfLife, maxDay);
  const combined = new Map();
  for (let day = 1; day <= maxDay; day++) combined.set(day, (all.get(day) + 3 * cycle.get(day)) / 4);

  const smoothed = new Map();
  for (let day = 1; day <= maxDay; day++) {
    const values = [];
    for (let d = Math.max(1, day - 2); d <= Math.min(maxDay, day + 2); d++) values.push(combined.get(d));
    smoothed.set(day, values.reduce((sum, value) => sum + value, 0) / values.length);
  }
  return smoothed;
}

function cumulativeActual(rows, returnKey) {
  let cumulative = 0;
  const out = [];
  for (const row of rows) {
    if (!isFiniteNumber(row[returnKey])) continue;
    cumulative += row[returnKey];
    out.push({ day: row.day, value: returnKey === "logReturn" ? Math.exp(cumulative) - 1 : cumulative,
      date: row.date });
  }
  return out;
}

function markerDayFor(rows, year, now, fastForwardDay) {
  if (fastForwardDay != null) return Math.max(1, Math.min(253, Number(fastForwardDay)));
  const context = rows.filter(row => row.year === year);
  if (!context.length) return null;
  const monthDay = `${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
  const found = context.find(row => row.date.slice(5) >= monthDay);
  return (found || context[context.length - 1]).day;
}

function markerMonthDayFor(rows, year, now, fastForwardDay) {
  const today = `${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
  if (fastForwardDay == null) return today;
  const context = rows.filter(row => row.year === year && row.date);
  if (!context.length) return today;
  const targetDay = Math.max(1, Math.min(253, Number(fastForwardDay)));
  const marker = context.reduce((best, row) =>
    Math.abs(row.day - targetDay) < Math.abs(best.day - targetDay) ? row : best);
  return marker.date.slice(5);
}

function closestCalendarPoint(path, monthDay) {
  const dated = path.filter(point => /^\d{4}-\d{2}-\d{2}$/.test(point.date || ""));
  if (!dated.length || !/^\d{2}-\d{2}$/.test(monthDay || "")) return null;
  const year = dated[0].date.slice(0, 4);
  const target = Date.parse(`${year}-${monthDay}T00:00:00Z`);
  if (!Number.isFinite(target)) return null;
  return dated.reduce((best, point) => {
    const distance = Math.abs(Date.parse(`${point.date}T00:00:00Z`) - target);
    const bestDistance = Math.abs(Date.parse(`${best.date}T00:00:00Z`) - target);
    return distance < bestDistance || (distance === bestDistance && point.date > best.date) ? point : best;
  });
}

function buildDateMap(rows, year) {
  const out = new Map();
  const context = rows.filter(row => row.year === year);
  const format = date => date.toLocaleString("en-US", { month: "short", day: "2-digit", timeZone: "UTC" });
  for (const row of context) out.set(row.day, format(new Date(`${row.date}T00:00:00Z`)));
  if (!context.length) {
    let day = 0;
    for (let date = new Date(Date.UTC(year, 0, 1)); date.getUTCFullYear() === year;
      date.setUTCDate(date.getUTCDate() + 1)) {
      if (date.getUTCDay() === 0 || date.getUTCDay() === 6) continue;
      out.set(++day, format(date));
    }
    return out;
  }
  let day = context[context.length - 1].day;
  const last = new Date(`${context[context.length - 1].date}T00:00:00Z`);
  for (last.setUTCDate(last.getUTCDate() + 1); last.getUTCFullYear() === year;
    last.setUTCDate(last.getUTCDate() + 1)) {
    if (last.getUTCDay() === 0 || last.getUTCDay() === 6) continue;
    out.set(++day, format(last));
  }
  return out;
}

function sampleStd(values) {
  const finite = values.filter(isFiniteNumber);
  if (finite.length < 2) return null;
  const mean = finite.reduce((sum, value) => sum + value, 0) / finite.length;
  const variance = finite.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / (finite.length - 1);
  return Math.sqrt(variance);
}

function forwardSnapshots(rows, markerDay, cutoffYear) {
  const records = [];
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    if (row.day !== markerDay || row.year >= cutoffYear || !isFiniteNumber(row.atr) || row.atr <= 0) continue;
    const record = { year: row.year, entryDate: row.date, entry: row.close };
    for (const horizon of SEASONAL_HORIZONS) {
      const future = rows[i + horizon];
      record[`date${horizon}`] = future ? future.date : "";
      record[`ret${horizon}`] = future ? (future.close - row.close) / row.atr : null;
      const daily = rows.slice(i + 1, i + horizon + 1).map(item => item.dailyPct);
      const std = daily.length === horizon ? sampleStd(daily) : null;
      record[`rv${horizon}`] = std == null ? null : std * Math.sqrt(252) * 100;
    }
    records.push(record);
  }
  return records;
}

function calculateStats(records, cutoffYear, halfLife) {
  const weights = records.map(row => recencyWeight(row.year, cutoffYear, halfLife));
  const out = { n: records.length };
  for (const horizon of SEASONAL_HORIZONS) {
    const returns = records.map(row => row[`ret${horizon}`]);
    out[`median${horizon}`] = weightedMedian(returns, weights);
    out[`mean${horizon}`] = weightedMean(returns, weights);
    const valid = returns.map((value, i) => [value, weights[i]]).filter(pair => isFiniteNumber(pair[0]));
    const totalWeight = valid.reduce((sum, pair) => sum + pair[1], 0);
    out[`positive${horizon}`] = totalWeight > 0
      ? valid.filter(pair => pair[0] > 0).reduce((sum, pair) => sum + pair[1], 0) / totalWeight * 100
      : null;
    out[`rv${horizon}`] = weightedMean(records.map(row => row[`rv${horizon}`]), weights);
  }
  return out;
}

function pathsByYear(rows, cycleLabel, minimumYear) {
  const grouped = new Map();
  for (const row of rows) {
    if (row.year < minimumYear || !isCycleYear(row.year, cycleLabel) || !isFiniteNumber(row.atrReturn)) continue;
    if (!grouped.has(row.year)) grouped.set(row.year, []);
    grouped.get(row.year).push(row);
  }
  const out = [];
  for (const [year, yearRows] of grouped) out.push({ year, path: cumulativeActual(yearRows, "atrReturn") });
  return out.sort((a, b) => a.year - b.year);
}

function pathSimilarity(yearPaths, currentYear, markerDay) {
  const current = yearPaths.find(item => item.year === currentYear);
  if (!current || markerDay == null) return [];
  const currentMap = new Map(current.path.filter(p => p.day <= markerDay).map(p => [p.day, p.value]));
  const out = [];
  for (const item of yearPaths) {
    if (item.year === currentYear) continue;
    const other = new Map(item.path.map(p => [p.day, p.value]));
    const common = Array.from(currentMap.keys()).filter(day => other.has(day));
    if (!common.length) continue;
    const mae = common.reduce((sum, day) => sum + Math.abs(currentMap.get(day) - other.get(day)), 0) / common.length;
    out.push({ year: item.year, mae, n: common.length });
  }
  return out.sort((a, b) => a.mae - b.mae);
}

function analyzeSeasonality(rows, options) {
  const now = options.now || new Date();
  const currentYear = now.getFullYear();
  const cutoffYear = options.timeTravel ? options.referenceYear : currentYear;
  const referenceYear = options.referenceYear || currentYear;
  const halfLife = options.halfLife || 20;
  const history = rows.filter(row => row.year < currentYear);
  const referenceHistory = rows.filter(row => row.year < referenceYear);
  const currentRows = rows.filter(row => row.year === currentYear);
  const referenceRows = rows.filter(row => row.year === referenceYear);
  const markerYear = options.timeTravel ? referenceYear : currentYear;
  const markerDay = markerDayFor(rows, markerYear, now, options.fastForwardDay);
  const markerMonthDay = markerMonthDayFor(rows, markerYear, now, options.fastForwardDay);
  const rankCurrent = calculateSeasonalRank(rows, options.cycle, currentYear, halfLife);
  const rankAll = options.overlayAll ? calculateSeasonalRank(rows, "All Years", currentYear, halfLife) : new Map();
  const rankHistorical = options.timeTravel
    ? calculateSeasonalRank(rows, options.cycle, referenceYear, halfLife) : new Map();

  const paths = {
    pctCycle: options.showPct ? calculateSeasonalPath(history, options.cycle, "logReturn", halfLife, currentYear) : [],
    pctAll: options.showPct && options.overlayAll
      ? calculateSeasonalPath(history, "All Years", "logReturn", halfLife, currentYear) : [],
    pctRealized: options.showPct ? cumulativeActual(currentRows, "logReturn") : [],
    atrCycle: calculateSeasonalPath(history, options.cycle, "atrReturn", halfLife, currentYear),
    atrAll: options.overlayAll
      ? calculateSeasonalPath(history, "All Years", "atrReturn", halfLife, currentYear) : [],
    atrRealized: cumulativeActual(currentRows, "atrReturn"),
    pctHistorical: options.timeTravel && options.showPct
      ? calculateSeasonalPath(referenceHistory, options.cycle, "logReturn", halfLife, referenceYear) : [],
    pctReference: options.timeTravel && options.showPct ? cumulativeActual(referenceRows, "logReturn") : [],
    atrHistorical: options.timeTravel
      ? calculateSeasonalPath(referenceHistory, options.cycle, "atrReturn", halfLife, referenceYear) : [],
    atrReference: options.timeTravel ? cumulativeActual(referenceRows, "atrReturn") : [],
  };
  const snapshots = markerDay == null ? [] : forwardSnapshots(rows, markerDay, cutoffYear);
  const cycleSnapshots = options.cycle === "All Years" ? snapshots
    : snapshots.filter(row => isCycleYear(row.year, options.cycle));
  const yearPaths = options.cycle === "All Years" ? [] : pathsByYear(rows, options.cycle, 2000);
  const last = rows[rows.length - 1];
  const lastAtrRow = rows.slice().reverse().find(row => isFiniteNumber(row.atr));
  return {
    currentYear, cutoffYear, referenceYear, markerYear, markerDay, markerMonthDay,
    currentPrice: last ? last.close : null,
    currentAtr: lastAtrRow ? lastAtrRow.atr : null,
    currentAtrPct: lastAtrRow ? lastAtrRow.atr / lastAtrRow.close * 100 : null,
    lastDate: last ? last.date : null,
    rankCurrent, rankAll, rankHistorical, paths,
    dateMap: buildDateMap(rows, markerYear),
    statsAll: calculateStats(snapshots, cutoffYear, halfLife),
    statsCycle: calculateStats(cycleSnapshots, cutoffYear, halfLife),
    snapshots: cycleSnapshots.slice().sort((a, b) => b.year - a.year),
    yearPaths,
    similarity: pathSimilarity(yearPaths, currentYear, markerDay),
  };
}

function defaultCycle() {
  const remainder = new Date().getFullYear() % 4;
  return remainder === 0 ? "Election" : remainder === 1 ? "Post-Election"
    : remainder === 2 ? "Midterm" : "Pre-Election";
}

function setSeasonalView(view) {
  const buttons = Array.from(document.querySelectorAll("[data-seasonal-view]"));
  if (!buttons.some(button => button.dataset.seasonalView === view)) view = "signals";
  for (const button of buttons) {
    const active = button.dataset.seasonalView === view;
    button.classList.toggle("on", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
    const panel = document.getElementById(`seasonal-panel-${button.dataset.seasonalView}`);
    if (panel) panel.hidden = !active;
  }
  if (window.history && window.history.replaceState) {
    window.history.replaceState(null, "", view === "signals"
      ? window.location.pathname + window.location.search : `#${view}`);
  }
  document.dispatchEvent(new CustomEvent("seasonal-view", { detail: { view } }));
}

async function initSeasonalityLab() {
  document.querySelectorAll("[data-seasonal-view]").forEach(button =>
    button.addEventListener("click", () => setSeasonalView(button.dataset.seasonalView)));
  if (/^#(lab|macro)$/.test(window.location.hash)) setSeasonalView(window.location.hash.slice(1));

  const root = document.getElementById("seasonality-lab");
  if (!root) return;
  const manifest = await fetchJSONOrNull("data/seasonality/manifest.json");
  if (!manifest || !manifest.tickers) {
    root.innerHTML = '<div class="fetchfail">Seasonality inputs are not present in this build. ' +
      "Try again after the next data refresh.</div>";
    return;
  }
  const tickers = Object.keys(manifest.tickers).sort();
  const earliest = Math.min(...Object.values(manifest.tickers).map(item => Number(String(item.start).slice(0, 4))));
  const currentYear = new Date().getFullYear();
  const referenceMin = Math.max(earliest + 1, 2001);
  root.innerHTML = seasonalityControls(manifest, tickers, referenceMin, currentYear);
  const cache = new Map();
  wireSeasonalityControls(root, manifest, cache);
}

function seasonalityControls(manifest, tickers, referenceMin, currentYear) {
  const options = tickers.map(ticker => `<option value="${labEsc(ticker)}"></option>`).join("");
  const cycle = defaultCycle();
  return `<div class="seasonality-controls card">
    <div class="seasonality-control-grid">
      <label><span>Ticker</span><input id="sl-ticker" list="sl-tickers" value="SPY" autocomplete="off"
        spellcheck="false" aria-describedby="sl-ticker-help"><datalist id="sl-tickers">${options}</datalist></label>
      <label><span>Cycle type</span><select id="sl-cycle">
        ${["All Years", "Election", "Pre-Election", "Post-Election", "Midterm"].map(value =>
          `<option${value === cycle ? " selected" : ""}>${value}</option>`).join("")}
      </select></label>
      <label class="check-control"><input id="sl-time" type="checkbox"><span>Enable time travel</span></label>
      <label><span>Compare vs year</span><input id="sl-reference" type="number" min="${referenceMin}"
        max="${currentYear}" value="${currentYear - 1}" disabled></label>
      <label class="check-control"><input id="sl-fast" type="checkbox"><span>Fast forward</span></label>
      <label><span>Trading day (1–253)</span><input id="sl-day" type="number" min="1" max="253" value="1" disabled></label>
      <label class="check-control"><input id="sl-overlay" type="checkbox"><span>Overlay All Years avg</span></label>
      <label class="check-control"><input id="sl-pct" type="checkbox"><span>Show % return</span></label>
      <label class="range-control"><span>Recency half-life: <b id="sl-half-value">20</b> yrs</span>
        <input id="sl-half" type="range" min="5" max="100" step="1" value="20"></label>
      <button class="btn primary" id="sl-run">Run analysis</button>
    </div>
    <div class="cap" id="sl-ticker-help">${manifest.ticker_count.toLocaleString()} tickers ·
      ${manifest.history_floor} onward · prices through ${labEsc(manifest.asof)} · adjusted basis ·
      simple ${manifest.atr.window}-session ATR</div>
  </div>
  <div id="sl-status" class="seasonality-status cap">Choose a ticker and run the analysis.</div>
  <div id="sl-results"></div>`;
}

function wireSeasonalityControls(root, manifest, cache) {
  const byId = id => root.querySelector(`#${id}`);
  byId("sl-time").addEventListener("change", event => { byId("sl-reference").disabled = !event.target.checked; });
  byId("sl-fast").addEventListener("change", event => { byId("sl-day").disabled = !event.target.checked; });
  byId("sl-half").addEventListener("input", event => { byId("sl-half-value").textContent = event.target.value; });
  byId("sl-ticker").addEventListener("keydown", event => {
    if (event.key === "Enter") { event.preventDefault(); byId("sl-run").click(); }
  });
  byId("sl-run").addEventListener("click", async () => {
    const ticker = byId("sl-ticker").value.trim().toUpperCase();
    const meta = manifest.tickers[ticker];
    const status = byId("sl-status");
    const results = byId("sl-results");
    if (!meta) {
      status.innerHTML = `<span class="neg">${labEsc(ticker || "Ticker")} is not in the read-only price snapshot.</span>`;
      results.innerHTML = "";
      return;
    }
    const button = byId("sl-run");
    button.disabled = true;
    button.textContent = "Calculating…";
    status.textContent = `Loading ${ticker}…`;
    try {
      let rows = cache.get(ticker);
      if (!rows) {
        const response = await fetch(`data/seasonality/${meta.file}`, { cache: "no-store" });
        if (!response.ok) throw new Error(`price snapshot: HTTP ${response.status}`);
        rows = enrichSeasonalityRows(decodeSeasonalityBuffer(await response.arrayBuffer()));
        cache.set(ticker, rows);
      }
      const options = {
        cycle: byId("sl-cycle").value,
        timeTravel: byId("sl-time").checked,
        referenceYear: Number(byId("sl-reference").value),
        fastForwardDay: byId("sl-fast").checked ? Number(byId("sl-day").value) : null,
        overlayAll: byId("sl-overlay").checked,
        showPct: byId("sl-pct").checked,
        halfLife: Number(byId("sl-half").value),
      };
      const analysis = analyzeSeasonality(rows, options);
      renderSeasonalityResults(results, ticker, options, analysis);
      status.textContent = `${ticker}: ${rows.length.toLocaleString()} sessions (${meta.start} → ${meta.end}). ` +
        `Analysis is local to this browser.`;
    } catch (error) {
      results.innerHTML = "";
      status.innerHTML = `<span class="neg">Could not analyze ${labEsc(ticker)}: ${labEsc(error.message)}</span>`;
    } finally {
      button.disabled = false;
      button.textContent = "Run analysis";
    }
  });
}

function renderSeasonalityResults(root, ticker, options, analysis) {
  const cycleLabel = options.cycle === "All Years" ? "All Years" : `${options.cycle} Cycle`;
  root.innerHTML = `<div class="kpis seasonality-kpis">
      ${labKpi("Last close", analysis.currentPrice == null ? "—" : `$${analysis.currentPrice.toFixed(2)}`, analysis.lastDate || "")}
      ${labKpi("ATR (14)", analysis.currentAtr == null ? "—" : `$${analysis.currentAtr.toFixed(2)}`,
        analysis.currentAtrPct == null ? "" : `${analysis.currentAtrPct.toFixed(2)}% of price`)}
      ${labKpi("Anchor day", analysis.markerDay == null ? "—" : analysis.markerDay,
        `${analysis.markerYear}${options.fastForwardDay ? " · fast forward" : ""}`)}
      ${labKpi("Model", cycleLabel, `${options.halfLife}yr half-life`)}
    </div>
    <div class="card seasonality-chart-card"><div id="sl-main-chart" class="chart seasonality-main-chart"></div></div>
    <h2>Forward return statistics <span class="cap" style="display:inline">pre-${analysis.cutoffYear}, ATR units</span></h2>
    <div class="card" id="sl-stats-table">${statsTable(options, analysis)}</div>
    ${options.cycle === "All Years" ? "" : `<h2>${labEsc(options.cycle)} years since 2000</h2>
      <div class="card seasonality-chart-card"><div id="sl-cycle-chart" class="chart seasonality-cycle-chart"></div></div>
      <h2>Path similarity <span class="cap" style="display:inline">MAE vs ${analysis.currentYear} YTD; lower is closer</span></h2>
      <div class="card">${similarityTable(analysis.similarity)}</div>`}
    <h2>Cycle-year analogs <span class="cap" style="display:inline">actual dates &amp; forward ATR performance</span></h2>
    <div class="card">${analogTable(options, analysis)}</div>`;

  if (typeof Plotly === "undefined") {
    root.querySelector("#sl-main-chart").innerHTML = '<div class="fetchfail">Plotly did not load, so charts are unavailable.</div>';
    return;
  }
  renderMainSeasonalityChart(ticker, options, analysis);
  if (options.cycle !== "All Years") renderCycleChart(options, analysis);
}

function labKpi(label, value, sub) {
  return `<div class="kpi"><div class="l">${labEsc(label)}</div><div class="v">${labEsc(value)}</div>` +
    `<div class="s">${labEsc(sub)}</div></div>`;
}

function pathValue(path, day) {
  const point = path.find(item => item.day === day);
  return point ? point.value : null;
}

function pathTrace(path, name, color, axis, rank, analysis, opts) {
  const anchor = analysis.markerDay == null ? null : pathValue(path, analysis.markerDay);
  const customdata = path.map(point => {
    const theoretical = opts.projection && anchor != null && analysis.currentAtr != null
      ? analysis.currentPrice + (point.value - anchor) * analysis.currentAtr : null;
    const actualDate = point.date
      ? new Date(`${point.date}T00:00:00Z`).toLocaleString("en-US", { month: "short", day: "2-digit", timeZone: "UTC" })
      : null;
    return [opts.actual && actualDate ? actualDate : analysis.dateMap.get(point.day) || `Day ${point.day}`,
      rank.get(point.day) == null ? null : rank.get(point.day), theoretical];
  });
  const hover = axis === "y2"
    ? "<b>%{customdata[0]}</b><br>Day: %{x}<br>Cumulative ATR: %{y:.2f}<br>" +
      (opts.projection ? "Theoretical: $%{customdata[2]:,.2f}<br>" : "") +
      "Seasonal rank: %{customdata[1]:.0f}<extra></extra>"
    : "<b>%{customdata[0]}</b><br>Day: %{x}<br>Return: %{y:.2%}<br>" +
      "Seasonal rank: %{customdata[1]:.0f}<extra></extra>";
  return { x: path.map(p => p.day), y: path.map(p => p.value), type: "scatter", mode: "lines", name,
    yaxis: axis, customdata, hovertemplate: hover,
    line: { color, width: opts.width || 2, dash: opts.dash || "solid" } };
}

function renderMainSeasonalityChart(ticker, options, analysis) {
  const traces = [];
  const p = analysis.paths;
  if (p.pctCycle.length) traces.push(pathTrace(p.pctCycle, `${options.cycle} %`, "#ff8c00", "y",
    analysis.rankCurrent, analysis, { width: 3 }));
  if (p.pctAll.length) traces.push(pathTrace(p.pctAll, "All Years %", "#87cefa", "y",
    analysis.rankAll, analysis, { width: 1, dash: "dot" }));
  if (p.pctHistorical.length) traces.push(pathTrace(p.pctHistorical, `Model ${analysis.referenceYear} %`, "#fcd12a", "y",
    analysis.rankHistorical, analysis, { width: 2 }));
  if (p.pctRealized.length) traces.push(pathTrace(p.pctRealized, `${analysis.currentYear} %`, "#39ff14", "y",
    analysis.rankCurrent, analysis, { width: 2, actual: true }));
  if (p.pctReference.length) traces.push(pathTrace(p.pctReference, `${analysis.referenceYear} %`, "#00ffff", "y",
    analysis.rankHistorical, analysis, { width: 2, actual: true }));
  if (p.atrCycle.length) traces.push(pathTrace(p.atrCycle, `${options.cycle} ATR`, "#ff8c00", "y2",
    analysis.rankCurrent, analysis, { width: 2, projection: true }));
  if (p.atrAll.length) traces.push(pathTrace(p.atrAll, "All Years ATR", "#87cefa", "y2",
    analysis.rankAll, analysis, { width: 1, dash: "dot", projection: true }));
  if (p.atrHistorical.length) traces.push(pathTrace(p.atrHistorical, `Model ${analysis.referenceYear} ATR`, "#fcd12a", "y2",
    analysis.rankHistorical, analysis, { width: 2, projection: true }));
  if (p.atrRealized.length) traces.push(pathTrace(p.atrRealized, `${analysis.currentYear} ATR`, "#39ff14", "y2",
    analysis.rankCurrent, analysis, { width: 2, actual: true }));
  if (p.atrReference.length) traces.push(pathTrace(p.atrReference, `${analysis.referenceYear} ATR`, "#00ffff", "y2",
    analysis.rankHistorical, analysis, { width: 2, actual: true }));
  if (analysis.markerDay != null) {
    const pctMarker = pathValue(p.pctCycle, analysis.markerDay);
    if (pctMarker != null) traces.push({ x: [analysis.markerDay], y: [pctMarker], yaxis: "y", type: "scatter",
      mode: "markers", showlegend: false, hoverinfo: "skip",
      marker: { color: options.timeTravel ? "#fcd12a" : "#fff", size: 8,
        line: { color: "#000", width: 1 } } });
    const marker = pathValue(p.atrCycle, analysis.markerDay);
    if (marker != null) traces.push({ x: [analysis.markerDay], y: [marker], yaxis: "y2", type: "scatter", mode: "markers",
      showlegend: false, hoverinfo: "skip", marker: { color: options.timeTravel ? "#fcd12a" : "#fff", size: 9,
        symbol: "diamond", line: { color: "#000", width: 1 } } });
  }
  const layout = plotLayout({
    title: { text: `Seasonal Analysis: ${ticker}${options.timeTravel ? ` vs ${analysis.referenceYear}` : ""}`,
      x: 0.01, xanchor: "left", font: { size: 15 } },
    height: 650, margin: { l: 58, r: 58, t: 48, b: 48 },
    yaxis: { title: options.showPct ? "Return" : "", tickformat: ".1%", gridcolor: "#1c2230" },
    yaxis2: { title: "Cumulative ATR", overlaying: "y", side: "right", tickformat: ".1f", showgrid: false },
    xaxis: { title: "Trading day of year", gridcolor: "#1c2230" },
    legend: { orientation: "h", y: -0.12, x: 0, xanchor: "left" },
  });
  Plotly.newPlot("sl-main-chart", traces, layout, PLOT_CFG);
}

function renderCycleChart(options, analysis) {
  const traces = [];
  const tradingMarkers = { x: [], y: [], colors: [], customdata: [] };
  const calendarMarkers = { x: [], y: [], colors: [], customdata: [] };
  for (let i = 0; i < analysis.yearPaths.length; i++) {
    const item = analysis.yearPaths[i];
    const current = item.year === analysis.currentYear;
    const color = current ? "#39ff14" : SEASONAL_COLORS[i % SEASONAL_COLORS.length];
    traces.push({ x: item.path.map(p => p.day), y: item.path.map(p => p.value), type: "scatter", mode: "lines",
      name: `${item.year}${current ? " (YTD)" : ""}`,
      line: { color, width: current ? 3 : 1.5 },
      hovertemplate: `<b>${item.year}</b><br>Day: %{x}<br>Cum ATR: %{y:.2f}<extra></extra>` });
    if (analysis.markerDay != null && item.path.length) {
      const marker = item.path.reduce((best, point) =>
        Math.abs(point.day - analysis.markerDay) < Math.abs(best.day - analysis.markerDay) ? point : best);
      tradingMarkers.x.push(marker.day);
      tradingMarkers.y.push(marker.value);
      tradingMarkers.colors.push(color);
      tradingMarkers.customdata.push([item.year, marker.date]);
    }
    const calendarMarker = closestCalendarPoint(item.path, analysis.markerMonthDay);
    if (calendarMarker) {
      calendarMarkers.x.push(calendarMarker.day);
      calendarMarkers.y.push(calendarMarker.value);
      calendarMarkers.colors.push(color);
      calendarMarkers.customdata.push([item.year, calendarMarker.date, analysis.markerMonthDay]);
    }
  }
  if (analysis.paths.atrCycle.length) traces.push({
    x: analysis.paths.atrCycle.map(p => p.day), y: analysis.paths.atrCycle.map(p => p.value),
    type: "scatter", mode: "lines", name: `${options.cycle} Avg (weighted)`,
    line: { color: "#fff", width: 2.5, dash: "dot" },
    hovertemplate: "Day: %{x}<br>Avg Cum ATR: %{y:.2f}<extra></extra>",
  });
  if (tradingMarkers.x.length) traces.push({
    x: tradingMarkers.x, y: tradingMarkers.y, customdata: tradingMarkers.customdata,
    type: "scatter", mode: "markers", name: `Trading day ${analysis.markerDay}`,
    marker: { color: tradingMarkers.colors, size: 5, symbol: "circle", line: { color: "#fff", width: 0.75 } },
    hovertemplate: "<b>%{customdata[0]} — same trading day</b><br>Date: %{customdata[1]}" +
      "<br>Day: %{x}<br>Cum ATR: %{y:.2f}<extra></extra>",
  });
  if (calendarMarkers.x.length) {
    const calendarLabel = new Date(`2000-${analysis.markerMonthDay}T00:00:00Z`).toLocaleString("en-US",
      { month: "short", day: "numeric", timeZone: "UTC" });
    traces.push({
      x: calendarMarkers.x, y: calendarMarkers.y, customdata: calendarMarkers.customdata,
      type: "scatter", mode: "markers", name: `Calendar date · ${calendarLabel}`, visible: "legendonly",
      marker: { color: calendarMarkers.colors, size: 8, symbol: "triangle-up", line: { color: "#fff", width: 0.75 } },
      hovertemplate: "<b>%{customdata[0]} — calendar-date match</b><br>Nearest session: %{customdata[1]}" +
        "<br>Target: %{customdata[2]}<br>Day: %{x}<br>Cum ATR: %{y:.2f}<extra></extra>",
    });
  }
  Plotly.newPlot("sl-cycle-chart", traces, plotLayout({
    height: 470, margin: { l: 58, r: 18, t: 24, b: 58 },
    xaxis: { title: "Trading day of year", gridcolor: "#1c2230" },
    yaxis: { title: "Cumulative ATR", tickformat: ".1f", gridcolor: "#1c2230" },
    legend: { orientation: "h", y: -0.16, x: 0, xanchor: "left" },
  }), PLOT_CFG);
}

function formatStat(value, type) {
  if (!isFiniteNumber(value)) return "—";
  if (type === "pct") return `${value.toFixed(1)}%`;
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

function statsTable(options, analysis) {
  const rows = [
    [`All History (<${analysis.cutoffYear})`, analysis.statsAll],
    [options.cycle === "All Years" ? "All Years" : `${options.cycle} Cycle`, analysis.statsCycle],
  ];
  return `<div class="tblwrap"><table class="tbl seasonality-stats"><thead><tr><th class="l" rowspan="2">Scope</th><th rowspan="2">N</th>` +
    SEASONAL_HORIZONS.map(h => `<th colspan="4">${h} sessions</th>`).join("") +
    `</tr><tr>${SEASONAL_HORIZONS.map(() => "<th>Median</th><th>Mean</th><th>Positive</th><th>RV</th>").join("")}</tr></thead><tbody>` +
    rows.map(([label, stats]) => `<tr><td class="l">${labEsc(label)}</td><td>${stats.n}</td>` +
      SEASONAL_HORIZONS.map(h => `<td class="${signClass(stats[`median${h}`])}">${formatStat(stats[`median${h}`])}</td>` +
        `<td class="${signClass(stats[`mean${h}`])}">${formatStat(stats[`mean${h}`])}</td>` +
        `<td class="${positiveClass(stats[`positive${h}`])}">${formatStat(stats[`positive${h}`], "pct")}</td>` +
        `<td>${formatStat(stats[`rv${h}`], "pct")}</td>`).join("") + `</tr>`).join("") +
    `</tbody></table></div><p class="cap">Weighted by a ${options.halfLife}-year half-life. RV is annualized realized volatility.</p>`;
}

function signClass(value) { return !isFiniteNumber(value) ? "" : value > 0 ? "pos" : value < 0 ? "neg" : ""; }
function positiveClass(value) { return !isFiniteNumber(value) ? "" : value > 80 ? "pos strong" : value < 25 ? "neg strong" : ""; }

function similarityTable(rows) {
  if (!rows.length) return '<p class="cap">Path similarity needs a current-year path and completed comparison years.</p>';
  return `<div class="tblwrap"><table class="tbl"><thead><tr><th>Year</th><th>MAE (ATR)</th><th>N</th></tr></thead><tbody>` +
    rows.map(row => `<tr><td>${row.year}</td><td>${row.mae.toFixed(3)}</td><td>${row.n}</td></tr>`).join("") +
    "</tbody></table></div>";
}

function analogTable(options, analysis) {
  if (!analysis.snapshots.length) return '<p class="cap">No completed analog years are available for this anchor day.</p>';
  const cells = analysis.snapshots.map(row => `<tr><td>${row.year}</td><td>${row.entryDate}</td><td>${row.entry.toFixed(2)}</td>` +
    SEASONAL_HORIZONS.map(h => `<td>${row[`date${h}`] || ""}</td><td class="${signClass(row[`ret${h}`])}">` +
      `${formatStat(row[`ret${h}`])}</td>`).join("") + `</tr>`).join("");
  return `<div class="tblwrap"><table class="tbl"><thead><tr><th>Year</th><th>Entry date</th><th>Entry</th>` +
    SEASONAL_HORIZONS.map(h => `<th>+${h}d date</th><th>+${h}d ATR</th>`).join("") +
    `</tr></thead><tbody>${cells}</tbody></table></div><p class="cap">Entry = trading day ${analysis.markerDay} of the year, ` +
    `${options.cycle === "All Years" ? "all years" : `${labEsc(options.cycle)} cycle years`}, pre-${analysis.cutoffYear}. ` +
    "ATR performance = (future close − entry close) / entry ATR.</p>";
}

globalThis.SeasonalityMath = {
  normalizeSeasonalityPayload,
  decodeSeasonalityBuffer,
  enrichSeasonalityRows,
  recencyWeight,
  weightedMean,
  weightedMedian,
  isCycleYear,
  calculateSeasonalPath,
  calculateSeasonalRank,
  markerDayFor,
  markerMonthDayFor,
  closestCalendarPoint,
  forwardSnapshots,
  calculateStats,
  pathSimilarity,
  analyzeSeasonality,
};
