/* strategies.js: the strategy catalog, one summary table plus one card per
 * strategy, grouped by family.
 *
 * Reads data/strategies.json (optional; the page shows an empty state when a
 * build did not produce it). Every string in the payload is treated as
 * external text and escaped. Stats shown are `stats || frozen_stats`, always
 * labelled with their source, because a ledger replay (today's config over
 * all history) and a frozen prereg transcription are different evidence.
 */
"use strict";

const STRAT_FAMILIES = [
  "Systematic equity book", "Futures", "Intraday ETF", "Sleeves", "Agent products",
];
const STRAT_STATUSES = ["live", "pilot", "manual", "shadow", "paper", "retired"];
const STRAT_SOURCE_LABEL = {
  ledger_replay: "ledger replay of today's config over full history",
  research_backtest: "research backtest",
  frozen_evidence: "frozen evidence",
  live_journal: "live journal",
  none: "no stats",
};
const stratView = { family: "all", status: "all", table: null, payload: null };

const stratEsc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function stratAnchor(id) {
  return "strat-" + String(id == null ? "" : id).replace(/[^A-Za-z0-9_-]/g, "-");
}

function stratList(value) {
  if (value == null) return "";
  if (Array.isArray(value)) return value.map(String).join(", ");
  return String(value);
}

function stratDirection(value) {
  return String(value || "").replace(/_/g, "/");
}

function stratNum(value) {
  if (value == null || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

/* The stats block to show plus a human label for where it came from. */
function stratStats(s) {
  const declared = (s && s.stats_source) || "none";
  const declaredLabel = STRAT_SOURCE_LABEL[declared] || String(declared);
  const computed = s && (s.stats || s.ledger_stats);
  if (computed) return { stats: computed, source: declared, label: declaredLabel };
  if (s && s.frozen_stats) {
    // A ledger strategy whose replay was unavailable this build falls back to
    // its frozen numbers; say so rather than calling them a replay.
    const label = declared === "ledger_replay"
      ? "frozen evidence (ledger replay unavailable in this build)" : declaredLabel;
    return { stats: s.frozen_stats, source: declared === "ledger_replay" ? "frozen_evidence" : declared, label };
  }
  return { stats: null, source: declared, label: declaredLabel };
}

function stratLiveFills(live) {
  if (!live) return null;
  return stratNum(live.fills != null ? live.fills : live.n_fills);
}

/* Account labels only (primary / pa). A raw broker account id must never
   render, even if a future build keys by_account on it. */
const STRAT_ACCOUNT_ID = /(?<![A-Za-z0-9])(?:DU|DF|U|F)\d{5,}(?!\d)/g;
function stratMaskAccount(value) {
  return String(value == null ? "" : value).replace(STRAT_ACCOUNT_ID, "[account]");
}

function stratLiveAccounts(live) {
  if (!live) return "";
  const names = live.accounts != null ? (Array.isArray(live.accounts) ? live.accounts : [live.accounts])
    : live.by_account && typeof live.by_account === "object" ? Object.keys(live.by_account).sort() : [];
  return names.map(stratMaskAccount).join(", ");
}

/* One line on what the fills store covers, including the untagged share
   that no catalog entry can claim. */
function stratCoverageText(payload) {
  const fills = payload && payload.sources && payload.sources.fills;
  if (!fills || fills.available !== true) {
    return "Live fills: the fills store was not available to this build, so live attribution is blank.";
  }
  const rows = stratNum(fills.rows);
  const untagged = stratNum(fills.untagged_rows);
  const span = fills.first_session && fills.last_session
    ? ` from ${fmt.date(fills.first_session)} to ${fmt.date(fills.last_session)}` : "";
  const share = rows > 0 && untagged != null
    ? `; ${fmt.num(untagged, 0)} (${fmt.num(100 * untagged / rows, 0)}%) carry no orderRef strategy and are not attributed to any entry` : "";
  return rows == null ? "" : `Live fills store: ${fmt.num(rows, 0)} executions${span}${share}.`;
}

function stratHold(stats) {
  if (!stats) return null;
  const days = stratNum(stats.median_hold_days);
  if (days != null) return days;
  return stats.median_hold == null ? null : stats.median_hold;
}

function stratWinPct(stats) {
  const w = stratNum(stats && stats.win_rate);
  if (w == null) return null;
  return w <= 1 ? w * 100 : w;
}

const stratFmt = {
  count: (v, d = 1) => v == null ? "" : fmt.num(v, d),
  pct: (v) => v == null ? "" : fmt.num(v, 1) + "%",
  r: (v) => v == null ? "" : fmt.signed(v, 2) + "R",
  hold: (v) => v == null ? "" : (typeof v === "number" ? fmt.num(v, 1) + "d" : String(v)),
};

function stratSummaryRows(list) {
  return (list || []).map((s) => {
    const { stats, label } = stratStats(s);
    return {
      id: s.id,
      name: s.name || s.id || "",
      family: s.family || "",
      status: s.status || "",
      direction: stratDirection(s.direction),
      instruments: stratList(s.instruments),
      tpy: stratNum(stats && stats.trades_per_year),
      tpm: stratNum(stats && stats.trades_per_month),
      win: stratWinPct(stats),
      avg_r: stratNum(stats && stats.avg_r),
      hold: stratHold(stats),
      live_since: s.live_since || "",
      source: label,
    };
  });
}

function stratFilter(list, family, status) {
  return (list || []).filter((s) =>
    (!family || family === "all" || s.family === family)
    && (!status || status === "all" || s.status === status));
}

function stratStatusBadge(status) {
  const key = STRAT_STATUSES.includes(status) ? status : "unknown";
  return `<span class="strat-badge strat-${key}">${stratEsc(String(status || "unknown").toUpperCase())}</span>`;
}

function stratChip(label, value) {
  return `<div class="strat-chip"><div class="l">${stratEsc(label)}</div><div class="v">${stratEsc(value)}</div></div>`;
}

function stratStatsBlock(s) {
  const { stats, label } = stratStats(s);
  if (!stats) return `<div class="cap">No trade statistics for this strategy (source: ${stratEsc(label)}).</div>`;
  const chips = [];
  const add = (name, value) => { if (value != null && value !== "") chips.push(stratChip(name, value)); };
  add("Trades / yr", stratFmt.count(stratNum(stats.trades_per_year)));
  add("Trades / mo", stratFmt.count(stratNum(stats.trades_per_month)));
  add("Trades", stats.n_trades == null ? null : fmt.num(stats.n_trades, 0));
  add("Win rate", stratFmt.pct(stratWinPct(stats)));
  add("Avg R", stratFmt.r(stratNum(stats.avg_r)));
  add("Avg bps", stratNum(stats.avg_bps) == null ? null : fmt.signed(stats.avg_bps, 1));
  add("PF", stratNum(stats.profit_factor) == null ? null : fmt.num(stats.profit_factor, 2));
  add("Sharpe", stratNum(stats.sharpe) == null ? null : fmt.num(stats.sharpe, 2));
  add("Median hold", stratFmt.hold(stratHold(stats)));
  add("Span", stats.span);
  const notes = stats.notes ? `<div class="cap">${stratEsc(stats.notes)}</div>` : "";
  return `<div class="strat-chips">${chips.join("")}</div>
    <div class="cap">source: ${stratEsc(label)}</div>${notes}${stratCasesTable(stats.cases)}`;
}

function stratCasesTable(cases) {
  if (!Array.isArray(cases) || !cases.length) return "";
  const cell = (v, f) => stratNum(v) == null ? "" : stratEsc(f(stratNum(v)));
  const rows = cases.map((c) => `<tr>
    <td class="l">${stratEsc(c.case)}</td>
    <td>${cell(c.n_trades, (v) => fmt.num(v, 0))}</td>
    <td>${cell(c.trades_per_year, (v) => fmt.num(v, 1))}</td>
    <td>${cell(c.win_rate, (v) => fmt.num(v <= 1 ? v * 100 : v, 1) + "%")}</td>
    <td>${stratNum(c.avg_r) != null ? cell(c.avg_r, (v) => fmt.signed(v, 2) + "R") : cell(c.avg_bps, (v) => fmt.signed(v, 1) + " bps")}</td>
    <td>${cell(c.profit_factor, (v) => fmt.num(v, 2))}</td>
    <td>${stratNum(c.sharpe) != null ? cell(c.sharpe, (v) => fmt.num(v, 2)) : cell(c.t_stat, (v) => "t " + fmt.num(v, 2))}</td></tr>`).join("");
  return `<div class="tblwrap strat-components"><table class="tbl">
    <thead><tr><th class="l">Case</th><th>N</th><th>Trades/yr</th><th>Win</th><th>Avg</th><th>PF</th><th>Sharpe / t</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

function stratComponentsTable(components) {
  if (!Array.isArray(components) || !components.length) return "";
  const cell = (v, f) => v == null ? "" : stratEsc(f(v));
  const rows = components.map((c) => `<tr>
    <td class="l">${stratEsc(c.name || c.id)}${c.ticker ? ` <span class="cap">${stratEsc(c.ticker)}${c.side ? " " + stratEsc(c.side) : ""}</span>` : ""}</td>
    <td>${cell(stratNum(c.n), (v) => fmt.num(v, 0))}</td>
    <td>${cell(stratNum(c.avg_bps), (v) => fmt.signed(v, 0))}</td>
    <td>${cell(stratNum(c.t), (v) => fmt.num(v, 2))}</td>
    <td>${cell(stratNum(c.hit), (v) => fmt.num(v <= 1 ? v * 100 : v, 0) + "%")}</td>
    <td class="l">${stratEsc(c.span)}</td></tr>`).join("");
  return `<div class="tblwrap strat-components"><table class="tbl">
    <thead><tr><th class="l">Component</th><th>N</th><th>Avg bps</th><th>t</th><th>Hit</th><th class="l">Span</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

function stratLiveLine(live, untagged = false) {
  if (!live) return `<div class="strat-live">Live fills: not available in this build.</div>`;
  const fills = stratLiveFills(live);
  const since = live.first_fill ? ` since ${stratEsc(fmt.date(live.first_fill))}` : "";
  const symbols = stratList(live.symbols);
  const sym = symbols ? ` (${stratEsc(symbols)})` : "";
  const last = live.last_fill ? `, last ${stratEsc(fmt.date(live.last_fill))}` : "";
  const pnl = stratNum(live.realized_pnl);
  const realized = pnl == null ? "" : `, broker realized <span class="${clsSign(pnl)}">${stratEsc(fmt.money(pnl))}</span>`;
  const accountList = stratLiveAccounts(live);
  const accounts = accountList ? `, accounts: ${stratEsc(accountList)}` : "";
  if (!(fills > 0)) {
    return `<div class="strat-live">Live fills: ${untagged
      ? "none attributable (no orderRef tag)." : "none in the fills store."}</div>`;
  }
  const notes = live.notes ? `<div class="cap">${stratEsc(live.notes)}</div>` : "";
  return `<div class="strat-live">Live fills: ${stratEsc(fmt.num(fills, 0))}${since}${sym}${last}${realized}${accounts}</div>${notes}`;
}

function stratCardHtml(s) {
  const def = (label, value) => {
    const text = stratList(value);
    return text ? `<dt>${label}</dt><dd>${stratEsc(text)}</dd>` : "";
  };
  const universe = stratNum(s.universe_size);
  const meta = [
    s.direction ? stratEsc(stratDirection(s.direction)) : "",
    stratList(s.instruments) ? stratEsc(stratList(s.instruments)) : "",
    universe != null ? `universe ${stratEsc(fmt.num(universe, 0))}` : "",
    s.live_since ? `live since ${stratEsc(s.live_since)}` : "",
  ].filter(Boolean).join(" &middot; ");
  const defs = [
    def("Entry", s.entry), def("Exit", s.exit), def("Sizing", s.sizing),
    def("Risk controls", s.risk_controls),
  ].join("");
  const status = STRAT_STATUSES.includes(s.status) ? s.status : "unknown";
  return `<div class="card strat-card strat-card-${status}" id="${stratEsc(stratAnchor(s.id))}">
    <div class="strat-head"><b>${stratEsc(s.name || s.id)}</b>${stratStatusBadge(s.status)}</div>
    <div class="cap">${meta}</div>
    ${s.captures ? `<div class="strat-sec">What it captures</div><p class="strat-captures">${stratEsc(s.captures)}</p>` : ""}
    ${defs ? `<dl class="strat-defs">${defs}</dl>` : ""}
    ${stratStatsBlock(s)}
    ${stratComponentsTable(s.components)}
    ${stratLiveLine(s.live, Array.isArray(s.order_ref_tags) && !s.order_ref_tags.length)}
    ${s.notes ? `<div class="cap">${stratEsc(s.notes)}</div>` : ""}
    ${s.doc ? `<div class="cap">Doc: <code>${stratEsc(s.doc)}</code></div>` : ""}
  </div>`;
}

function stratFamilyOrder(list) {
  const seen = new Set((list || []).map((s) => s.family || "Other"));
  const order = STRAT_FAMILIES.filter((f) => seen.has(f));
  for (const f of seen) if (!order.includes(f)) order.push(f);
  return order;
}

function stratCardsHtml(list) {
  if (!list || !list.length) return '<p class="cap">No strategies match these filters.</p>';
  return stratFamilyOrder(list).map((family) => {
    const members = list.filter((s) => (s.family || "Other") === family);
    return `<h2 class="strat-family">${stratEsc(family)} <span class="cap" style="display:inline">${members.length}</span></h2>
      <div class="grid2 strat-grid">${members.map(stratCardHtml).join("")}</div>`;
  }).join("");
}

function stratEmptyHtml() {
  return `<div class="card strat-empty"><b>Strategies payload not built</b>
    <div class="cap">This deploy has no <code>data/strategies.json</code>. The catalog is best effort;
      check the build log for the strategies step.</div></div>`;
}

function stratFiltersHtml(list) {
  const families = stratFamilyOrder(list);
  const statuses = STRAT_STATUSES.filter((st) => (list || []).some((s) => s.status === st));
  const opt = (value, label, current) =>
    `<option value="${stratEsc(value)}"${value === current ? " selected" : ""}>${stratEsc(label)}</option>`;
  return `<label for="stratFamily">Family</label>
    <select id="stratFamily">${opt("all", "All families", stratView.family)}${families.map((f) => opt(f, f, stratView.family)).join("")}</select>
    <label for="stratStatus">Status</label>
    <select id="stratStatus">${opt("all", "All statuses", stratView.status)}${statuses.map((st) => opt(st, st, stratView.status)).join("")}</select>`;
}

const STRAT_COLUMNS = [
  { key: "name", label: "Strategy", align: "l" },
  { key: "family", label: "Family", align: "l" },
  { key: "status", label: "Status", align: "l" },
  { key: "direction", label: "Direction", align: "l" },
  { key: "instruments", label: "Instruments", align: "l" },
  { key: "tpy", label: "Trades/yr", fmt: (v) => stratFmt.count(v) },
  { key: "tpm", label: "Trades/mo", fmt: (v) => stratFmt.count(v) },
  { key: "win", label: "Win rate", fmt: stratFmt.pct },
  { key: "avg_r", label: "Avg R", fmt: stratFmt.r, cls: clsSign },
  { key: "hold", label: "Median hold", fmt: stratFmt.hold },
  { key: "live_since", label: "Live since" },
  { key: "source", label: "Stats source", align: "l" },
];

function stratApplyFilters() {
  const list = (stratView.payload && stratView.payload.strategies) || [];
  const shown = stratFilter(list, stratView.family, stratView.status);
  if (stratView.table) stratView.table.setRows(stratSummaryRows(shown));
  const cards = document.getElementById("stratCards");
  if (cards) cards.innerHTML = stratCardsHtml(shown);
  return shown;
}

function stratSetFilter(kind, value) {
  stratView[kind] = value || "all";
  return stratApplyFilters();
}

function renderStrategiesPage(payload) {
  const summary = document.getElementById("stratSummary");
  const filters = document.getElementById("stratFilters");
  const cards = document.getElementById("stratCards");
  const coverage = document.getElementById("stratCoverage");
  stratView.payload = payload;
  stratView.table = null;
  const list = payload && Array.isArray(payload.strategies) ? payload.strategies : null;
  if (coverage) coverage.textContent = list ? stratCoverageText(payload) : "";
  if (!list) {
    if (filters) filters.innerHTML = "";
    if (cards) cards.innerHTML = "";
    if (summary) summary.innerHTML = stratEmptyHtml();
    return false;
  }
  if (filters) {
    filters.innerHTML = stratFiltersHtml(list);
    const fam = document.getElementById("stratFamily");
    const st = document.getElementById("stratStatus");
    if (fam && fam.addEventListener) fam.addEventListener("change", () => stratSetFilter("family", fam.value));
    if (st && st.addEventListener) st.addEventListener("change", () => stratSetFilter("status", st.value));
  }
  if (summary) {
    stratView.table = makeTable(summary, {
      columns: STRAT_COLUMNS, rows: [], textOnly: true,
      defaultSort: { key: "tpy", dir: -1 },
    });
  }
  stratApplyFilters();
  return true;
}

/* Prefer the coherent-build loader; fall back to a plain fetch so the page
   still works on a local dist/ without a matching meta.json. Missing or
   unreadable payload resolves to null (empty state), never a thrown page. */
async function loadStrategiesPayload() {
  try {
    const snap = await loadSiteSnapshot(async (meta) => {
      const flags = meta.payloads || {};
      if (flags.strategies === false) return { strategies: null };
      try { return { strategies: await fetchSitePayload(meta, "data/strategies.json") }; }
      catch (e) { if (e.siteSnapshotMismatch && flags.strategies === true) throw e; return { strategies: null }; }
    }, 2);
    if (snap.strategies) return { payload: snap.strategies, meta: snap.meta };
    return { payload: null, meta: snap.meta };
  } catch (e) {
    return { payload: await fetchJSONOrNull("data/strategies.json"), meta: null };
  }
}

async function initStrategies() {
  renderNav("strategies.html");
  const { payload, meta } = await loadStrategiesPayload();
  const generated = payload && payload.generated_at;
  setAsof(generated ? `catalog built ${fmt.date(generated)}` : (meta && meta.built_at ? `built ${meta.built_at}` : ""));
  renderStrategiesPage(payload);
  const hash = typeof location !== "undefined" ? String(location.hash || "") : "";
  if (hash.startsWith("#strat-")) {
    const target = document.getElementById(decodeURIComponent(hash.slice(1)));
    if (target && target.scrollIntoView) target.scrollIntoView();
  }
}

if (typeof document !== "undefined" && document.addEventListener)
  document.addEventListener("DOMContentLoaded", initStrategies);
