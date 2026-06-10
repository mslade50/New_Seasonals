/* signals.js — render data/signals.json (Order_Staging + Overflow snapshots) */
"use strict";

document.addEventListener("DOMContentLoaded", init);

// Columns we surface if present, in this order; everything else is appended.
const PREFERRED = ["Strategy", "Ticker", "Action", "Entry Type", "Entry_Type",
  "Limit Price", "Limit_Price", "Signal Close", "Signal_Close", "ATR", "Shares",
  "Risk $", "Risk_USD", "Risk bps", "Stop", "Target", "Signal Date", "Signal_Date",
  "Scan_Source", "Entry Criteria", "Entry_Criteria"];

async function init() {
  renderNav("signals.html");
  const el = document.getElementById("content");
  const data = await fetchJSONOrNull("data/signals.json");
  if (!data) {
    el.innerHTML = '<p class="cap">No signals payload in this build (Sheets fetch skipped or failed).</p>';
    return;
  }
  setAsof(`fetched ${data.fetched_at}`);
  el.innerHTML = "";
  const tabs = data.tabs || {};
  for (const [tab, label] of [["Order_Staging", "Liquid — Order_Staging"], ["Overflow", "Overflow"]]) {
    const rows = tabs[tab] || [];
    const h = document.createElement("h2");
    h.textContent = `${label} (${rows.length})`;
    el.appendChild(h);
    const div = document.createElement("div");
    el.appendChild(div);
    if (!rows.length) {
      div.innerHTML = '<p class="cap">No staged rows.</p>';
      continue;
    }
    const keys = orderedKeys(rows);
    makeTable(div, {
      columns: keys.map(k => ({
        key: k, label: k.replace(/_/g, " "),
        align: isNumericCol(rows, k) ? "r" : "l",
        fmt: v => v == null ? "" : escCell(v),
      })),
      rows, search: rows.length > 10, csvName: `${tab.toLowerCase()}.csv`,
      pageSize: rows.length > 30 ? 25 : 0,
    });
  }
}

function orderedKeys(rows) {
  const all = new Set();
  for (const r of rows) Object.keys(r).forEach(k => { if (k) all.add(k); });
  const pref = PREFERRED.filter(k => all.has(k));
  const rest = [...all].filter(k => !pref.includes(k));
  return [...pref, ...rest];
}

function isNumericCol(rows, k) {
  let seen = 0;
  for (const r of rows) {
    const v = r[k];
    if (v === "" || v == null) continue;
    if (typeof v !== "number" && isNaN(parseFloat(v))) return false;
    if (++seen >= 5) break;
  }
  return seen > 0;
}

function escCell(v) {
  return String(v).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
