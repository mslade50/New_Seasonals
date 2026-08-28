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
  ["exposure_state",    "Exposure state",    "AM scan snapshot — canonical on R2"],
  ["signals",           "Staged signals",    "Order_Staging + Overflow Sheets fetch"],
];

/* Weekday deploys run 2x/day; the Fri-PM -> Mon-AM gap is ~60h. */
const BUILD_WARN_H = 26, BUILD_STALE_H = 75;

const SCHEDULE = [
  ["premarket pipeline", "Weekdays 4:10 AM ET — local Task Scheduler primary",
   "Serial CBOE, settled prices, risk correction, Event sleeve and full scan; then hands both sites to their required cloud-only deploy workflows. Publishes canonical state to R2."],
  ["Daily Pitch (agent)", "Weekdays 5:10 AM ET — existing local Task Scheduler task",
   "Pulls the completed premarket state from R2, grades prior ideas, builds today's state and sends the research pitch."],
  ["health pipeline", "Weekdays 7:30 AM ET — local Task Scheduler primary",
   "Checks receipts, state freshness, delivery evidence and the targeted automation test battery."],
  ["discretionary pipeline", "Weekdays 8:35 AM ET — local Task Scheduler primary",
   "Research-only 0-2 name attention list; no capital allocation, staging or broker actions."],
  ["execution pipeline", "Weekdays 4:30 PM ET — local Task Scheduler primary",
   "Sends the live-account execution status report once."],
  ["postclose pipeline", "Weekdays 5:10 PM ET — local Task Scheduler primary",
   "Serial close prices, risk/fills/earnings/portfolio/CBOE/sleeves/intraday/scan/macro jobs; then hands both sites to cloud-only deploy workflows."],
  ["indicator pipeline", "Mondays 3:00 AM ET — local Task Scheduler primary",
   "Builds and uploads the liquid and overflow strategy-indicator caches."],
  ["weekly-rundown pipeline", "Sundays 8:00 AM ET — local Task Scheduler primary",
   "Renders and emails the tabloid weekly market rundown."],
  ["GitHub guarded backup", "Hourly at :47 UTC plus DST-safe 8:50 AM ET probes",
   "Reads R2 receipts and dispatches only missing jobs inside bounded recovery windows. Private and shared sites always build and deploy in GitHub Actions from R2."],
];

async function init() {
  renderNav("pipeline.html");
  const el = document.getElementById("content");
  const meta = await fetchSiteMeta().catch(() => null);
  const health = meta && meta.freshness;
  if (meta) setAsof(`built ${meta.built_at}`);

  let html = "";
  if (!health) {
    html += '<p class="cap">No health payload in this build (meta.payloads.health off or builder failed). Schedule reference below.</p>';
  } else {
    html += buildBanner(health, meta);
    html += `<h2>Artifact freshness</h2>
      <p class="cap">Judged server-side against the expected last trading day
        (${esc(health.expected_last_td || "?")}); "fresh" tolerates one trading day
        (&ge; ${esc(health.prev_td || "?")}) for session-lagged state and
        weekend/holiday continuity.</p>
      <div class="sigcards">${(ART_ORDER.map(([k, name, desc]) =>
        artifactCard(k, name, desc, (health.artifacts || {})[k]))).join("")}</div>`;
    html += ledgerCard((health.artifacts || {}).ledger);
  }

  html += `<h2>What runs when</h2>
    <p class="cap">Static reference (ET), transcribed from the repo's Automated Pipeline
      docs. Primary production pipelines execute on this machine from a pinned Task
      Scheduler runtime. GitHub is the receipt-guarded backup and remains the mandatory
      cloud builder/deployer for both sites.</p>
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
