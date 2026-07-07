/* pipeline.js — pipeline freshness + health (data/health.json).

   Renders: (1) a build-age banner computed against the BROWSER clock — a
   frozen pipeline leaves every baked age green, so the client must judge
   built_at itself; (2) a status strip of per-artifact cards from health.json;
   (3) the ledger provenance card; (4) a static "what runs when" reference
   table transcribed from CLAUDE.md's Automated Pipeline section (ET times). */
"use strict";

document.addEventListener("DOMContentLoaded", init);

const ART_ORDER = [
  ["master_prices",     "Master prices",     "R2 OHLCV cache — feeds scans, ledger, site"],
  ["earnings_calendar", "Earnings calendar", "FMP backfill — OVS blackout filter"],
  ["fragility",         "rd2 fragility",     "Append-only dial series — SIZES LIVE ORDERS"],
  ["exposure_state",    "Exposure state",    "Committed by the AM scan"],
  ["signals",           "Staged signals",    "Order_Staging + Overflow Sheets fetch"],
];

/* Weekday deploys run 2x/day; the Fri-PM -> Mon-AM gap is ~60h. */
const BUILD_WARN_H = 26, BUILD_STALE_H = 75;

const SCHEDULE = [
  ["update_master_prices", "Weekdays 4:17 AM ET (local dispatch; GHA cron fallback 5:30 AM) + 4:30 PM ET",
   "Incremental yfinance bar append to master_prices.parquet on R2. AM runs exclude today's placeholder bar."],
  ["daily_screener", "Weekdays 4:47 AM ET (local dispatch; fallback 6:30 AM) + 6:00 PM ET",
   "Unified scan, both runs --scope=all (liquid + overflow). AM run commits exposure_state.json."],
  ["deploy_site (this site)", "Same run as each successful daily_screener scan (2x/trading day)",
   "Ledger rebuild -> new trade charts -> seasonal ideas -> risk JSON -> site payloads -> Cloudflare Pages deploy. A skipped/failed scan skips the deploy; the prior deploy stays up."],
  ["update_intraday_prices", "Weekdays 4:45 PM ET",
   "15-min OHLCV cache refresh on R2 (liquid universe)."],
  ["risk_report", "Weekdays 5:15 PM ET",
   "Daily risk email; APPENDS today's row to rd2_fragility.parquet (sizes live orders via frag_risk_bands)."],
  ["verify_fills", "Weekdays 5:15 PM ET",
   "Post-close fill verification -> Trade_Signals_Log sheet."],
  ["build_earnings_calendar", "Weekdays 5:30 PM ET (+ local Task Scheduler mirror)",
   "FMP earnings pull -> earnings_calendar.parquet -> R2."],
  ["portfolio_report", "Weekdays 5:30 PM ET",
   "Portfolio health email + Portfolio Sheets tab (open-position counts for ladder sizing)."],
  ["trend_sleeve", "Weekdays 5:35 PM ET (no-op except month's last trading day)",
   "Monthly trend-following ballast rebalance -> Trend Sheets tab + state on R2."],
  ["radar_weekly_summary (local)", "Sundays 8:30 AM ET",
   "Radar digest distilled by Claude, committed to the repo."],
  ["weekly_rundown", "Sundays 9:00 AM ET",
   "Tabloid PDF rundown email incorporating the radar digest."],
  ["bootstrap_caches", "Manual only (workflow_dispatch)",
   "One-shot full master_prices rebuild to seed R2."],
];

async function init() {
  renderNav("pipeline.html");
  const el = document.getElementById("content");
  const [meta, health] = await Promise.all([
    fetchJSONOrNull("data/meta.json"),
    fetchJSONOrNull("data/health.json"),
  ]);
  if (meta) setAsof(`built ${meta.built_at}`);

  let html = "";
  if (!health) {
    html += '<p class="cap">No health payload in this build (meta.payloads.health off or builder failed). Schedule reference below.</p>';
  } else {
    html += buildBanner(health, meta);
    html += `<h2>Artifact freshness</h2>
      <p class="cap">Judged server-side against the expected last trading day
        (${esc(health.expected_last_td || "?")}); "fresh" tolerates one trading day
        (&ge; ${esc(health.prev_td || "?")}) to absorb the AM deploy's checkout lag on
        committed inputs like exposure_state.</p>
      <div class="sigcards">${(ART_ORDER.map(([k, name, desc]) =>
        artifactCard(k, name, desc, (health.artifacts || {})[k]))).join("")}</div>`;
    html += ledgerCard((health.artifacts || {}).ledger);
  }

  html += `<h2>What runs when</h2>
    <p class="cap">Static reference (ET), transcribed from the repo's Automated Pipeline
      docs. AM runs fire from the local machine via workflow_dispatch to dodge GitHub's
      shared-cron queue lag; each has a later GHA cron fallback that auto-skips if the
      dispatch already succeeded.</p>
    <div class="card"><div class="tblwrap"><table class="tbl">
      <thead><tr><th class="l">Workflow</th><th class="l">Schedule</th><th class="l">What it does</th></tr></thead>
      <tbody>${SCHEDULE.map(([w, s, d]) =>
        `<tr><td class="l" style="white-space:nowrap">${esc(w)}</td>
         <td class="l">${esc(s)}</td><td class="l">${esc(d)}</td></tr>`).join("")}</tbody>
    </table></div></div>`;

  el.innerHTML = html;
}

/* -------- build-age banner (browser-clock check) -------- */
function buildBanner(health, meta) {
  const stamp = health.built_at || (meta && meta.built_at);
  const t = parseUTC(stamp);
  if (t == null) {
    return `<div class="card" style="margin-bottom:14px">
      <span class="badge warn">UNKNOWN</span>
      <span style="margin-left:8px">Build timestamp unparseable (${esc(String(stamp))}).</span></div>`;
  }
  const hrs = (Date.now() - t) / 36e5;
  const badge = hrs < BUILD_WARN_H ? '<span class="badge off">CURRENT</span>'
    : hrs < BUILD_STALE_H ? '<span class="badge warn">AGING</span>'
    : '<span class="badge on">FROZEN?</span>';
  const note = hrs >= BUILD_STALE_H
    ? " No deploy in over three days — the deploy chain may not be firing (every per-artifact age below is baked at build time and goes stale WITH the build)."
    : hrs >= BUILD_WARN_H
      ? " More than a day since the last deploy; normal over a weekend/holiday, otherwise check the daily_screener run."
      : "";
  return `<div class="card" style="margin-bottom:14px">${badge}
    <span style="margin-left:8px">This site build is <b>${hrs < 48 ? Math.round(hrs) + "h" : (hrs / 24).toFixed(1) + "d"}</b>
    old (built ${esc(String(stamp))}, vs your clock).${esc(note)}</span></div>`;
}

/* -------- per-artifact cards -------- */
function stBadge(status) {
  if (status === "fresh") return '<span class="badge off">FRESH</span>';
  if (status === "stale") return '<span class="badge warn">STALE</span>';
  return '<span class="badge on">MISSING</span>';
}

function artifactCard(key, name, desc, a) {
  const kv = [];
  const add = (k, v) => { if (v != null && v !== "") kv.push([k, String(v)]); };
  if (!a) a = { status: "missing" };
  if (key === "master_prices") {
    add("SPY last bar", a.spy_last_date);
    add("Overall last bar", a.last_date);
    add("Age (td)", a.age_td);
  } else if (key === "earnings_calendar") {
    add("Last updated", a.last_updated);
    add("Max fwd date", a.max_date);
    add("Rows", a.rows == null ? null : Number(a.rows).toLocaleString());
    add("Age (td)", a.age_td);
  } else if (key === "fragility") {
    add("Last date", a.last_date);
    add("63d dial", a.last_63d);
    add("Age (td)", a.age_td);
  } else if (key === "exposure_state") {
    add("As of", a.asof);
    add("Age (td)", a.age_td);
  } else if (key === "signals") {
    add("Fetched", a.fetched_at);
    add("Source", a.source === "previous_build" ? "previous build (this run's fetch failed)" : a.source);
    if ((a.tabs_failed || []).length) add("Tabs FAILED", a.tabs_failed.join(", "));
  }
  if (a.note) add("Note", a.note);
  return `<div class="card">
    <div style="display:flex;align-items:baseline;gap:10px;flex-wrap:wrap">
      <b>${esc(name)}</b>${stBadge(a.status)}</div>
    <div class="cap" style="margin:4px 0 0">${esc(desc)}</div>
    <div class="kv">${kv.map(([k, v]) =>
      `<div class="k">${esc(k)}</div><div class="v">${esc(v)}</div>`).join("")}</div>
  </div>`;
}

/* -------- ledger provenance -------- */
function ledgerCard(led) {
  if (!led) return "";
  const kv = [
    ["Build (UTC)", led.build_utc],
    ["Source", led.source],
    ["Git SHA", led.git_sha ? String(led.git_sha).slice(0, 12) : null],
    ["Rows", led.rows],
    ["Age (days)", led.age_days],
    ["Note", led.note],
  ].filter(([, v]) => v != null && v !== "");
  return `<h2>Trade ledger provenance</h2>
    <div class="card">
      <div style="display:flex;align-items:baseline;gap:10px;flex-wrap:wrap">
        <b>backtest_trades_full.parquet</b>${stBadge(led.status)}</div>
      <div class="cap" style="margin:4px 0 0">Full backtest REBUILD, not a fill record —
        marginal fills flicker between vintages. This vintage gates the OLV sector-loss
        check and feeds every payload on this site. Non-GHA or &gt;4d-old vintages are flagged.</div>
      <div class="kv">${kv.map(([k, v]) =>
        `<div class="k">${esc(k)}</div><div class="v">${esc(String(v))}</div>`).join("")}</div>
    </div>`;
}

/* "YYYY-MM-DD HH:MM UTC" | ISO -> epoch ms (null if unparseable) */
function parseUTC(s) {
  if (!s) return null;
  let iso = String(s).trim().replace(" UTC", "Z").replace(" ", "T");
  if (!/Z|[+-]\d\d:?\d\d$/.test(iso)) iso += "Z";
  const t = Date.parse(iso);
  return Number.isNaN(t) ? null : t;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
