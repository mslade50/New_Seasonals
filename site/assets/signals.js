/* signals.js — staged scan signals as cards (Order_Staging + Overflow tabs).

   One card per staged order: ticker, direction, strategy, entry mechanics,
   estimated bracket levels (computed from Limit/Signal Close +/- ATR mults —
   actual prices resolve off the live fill in order_staging), risk, and any
   special stamps (OVS 2-path, Monday-gap kill, T+1 open filters). The raw
   rows stay available in a collapsed table per tab. */
"use strict";

document.addEventListener("DOMContentLoaded", init);

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
  for (const [tab, label] of [["Order_Staging", "Liquid"], ["Overflow", "Overflow"]]) {
    const rows = (tabs[tab] || []).slice().sort((a, b) => num(b.Risk_Amt) - num(a.Risk_Amt));
    const h = document.createElement("h2");
    h.textContent = `${label} (${rows.length})`;
    el.appendChild(h);
    if (!rows.length) {
      const p = document.createElement("p");
      p.className = "cap";
      p.textContent = "No staged orders on this tab.";
      el.appendChild(p);
      continue;
    }
    const grid = document.createElement("div");
    grid.className = "sigcards";
    grid.innerHTML = rows.map(signalCard).join("");
    el.appendChild(grid);

    // raw table fallback for the long tail of columns
    const det = document.createElement("details");
    det.className = "card riskpanel";
    det.innerHTML = `<summary>Raw staging rows (${rows.length})</summary><div class="rawtbl"></div>`;
    el.appendChild(det);
    const keys = [...new Set(rows.flatMap(r => Object.keys(r)))].filter(Boolean);
    makeTable(det.querySelector(".rawtbl"), {
      columns: keys.map(k => ({
        key: k, label: k.replace(/_/g, " "), align: isNum(rows, k) ? "r" : "l",
        fmt: v => v == null ? "" : esc(String(v)),
      })),
      rows, search: rows.length > 8, csvName: `${tab.toLowerCase()}.csv`,
      pageSize: rows.length > 25 ? 25 : 0,
    });
  }
}

function signalCard(r) {
  const sym = esc(r.Symbol || r.Ticker || "?");
  const strat = esc(r.Strategy_Ref || r.Strategy || "");
  const isShort = String(r.Trade_Direction || r.Action || "").toUpperCase().includes("SHORT")
    || String(r.Action || "").toUpperCase() === "SELL";
  const dirBadge = isShort ? '<span class="badge dirS">SHORT</span>'
                           : '<span class="badge dirL">LONG</span>';
  const tier = esc(r.Scan_Source || "");

  const atr = num(r.Frozen_ATR);
  const sigClose = num(r.Signal_Close);
  const lmt = num(r.Limit_Price);
  const qty = num(r.Quantity);
  const risk = num(r.Risk_Amt);
  const bps = num(r.Risk_Bps);

  // entry mechanics line
  const ot = String(r.Order_Type || "");
  const offs = num(r.Offset_ATR_Mult);
  let entry;
  if (ot === "MOO") entry = "Market on open (OPG)";
  else if (ot === "LOC") entry = `Limit-on-close ${money(lmt)}`;
  else {
    const offTxt = offs ? ` (open ${isShort ? "+" : "-"}${offs} ATR)` : "";
    entry = `LMT ${money(lmt)}${offTxt}, ${esc(r.TIF || "DAY")}`;
  }

  // estimated bracket levels off the staged reference price
  const base = lmt || sigClose;
  const sgn = isShort ? -1 : 1;
  const sMult = num(r.Stop_ATR_Mult), tMult = num(r.Tgt_ATR_Mult);
  const useStop = String(r.Use_Stop).toLowerCase() === "true" && sMult > 0;
  const useTgt = String(r.Use_Target).toLowerCase() === "true" && tMult > 0;
  const stopPx = useStop && base && atr ? base - sgn * sMult * atr : null;
  const tgtPx = useTgt && base && atr ? base + sgn * tMult * atr : null;
  const hold = num(r.Hold_Days);

  const levels = [
    `<div class="lv"><span>Stop</span><b class="${useStop ? "neg" : ""}">${
       useStop ? `${money(stopPx)} (${sMult} ATR, arms day 2)` : "none"}</b></div>`,
    `<div class="lv"><span>Target</span><b class="${useTgt ? "pos" : ""}">${
       useTgt ? `${money(tgtPx)} (${tMult} ATR)` : "none"}</b></div>`,
    `<div class="lv"><span>Time exit</span><b>${esc(String(r.Time_Exit_Date || "").slice(0, 10))}${
       hold ? ` (${hold}td)` : ""}</b></div>`,
  ].join("");

  // special stamps
  const chips = [];
  if (num(r.Path1_Bps)) chips.push(`OVS 2-path: P1 ${num(r.Path1_Bps)} bps / P2 ${num(r.Path2_Bps)} bps`);
  if (num(r.MonGapKill_ATR)) chips.push(`Gap-kill ${num(r.MonGapKill_ATR)} ATR ${esc(r.MonGapKill_Dir || "up")}`);
  if (r.T1_Open_Filters && String(r.T1_Open_Filters).length > 2) chips.push("T+1 open filter");
  const chipHtml = chips.length
    ? `<div class="chips">${chips.map(c => `<span class="badge conv">${c}</span>`).join(" ")}</div>` : "";

  return `<div class="card sigcard">
    <div class="head">
      <span class="tkr">${sym}</span>
      ${dirBadge}
      ${tier ? `<span class="badge ${tier === "Overflow" ? "warn" : "conv"}">${tier}</span>` : ""}
      <span class="chan">${strat}</span>
    </div>
    <div class="entryline">${entry}
      <span class="muted">| signal close ${money(sigClose)} | ATR ${atr ? atr.toFixed(2) : "?"}</span></div>
    <div class="levels">${levels}</div>
    <div class="riskline">Risk ${money(risk)}${bps ? ` (${bps} bps)` : ""}${
      qty ? ` | ${qty.toLocaleString()} sh` : ""}${
      lmt && qty ? ` | ~${money(lmt * qty)} notional` : ""}</div>
    ${chipHtml}
  </div>`;
}

/* ---------- helpers ---------- */
function num(v) {
  if (v == null || v === "") return 0;
  const f = parseFloat(v);
  return isFinite(f) ? f : 0;
}
function money(v) {
  return v ? fmt.money(v, v < 10 ? 2 : 2) : "?";
}
function isNum(rows, k) {
  let seen = 0;
  for (const r of rows) {
    const v = r[k];
    if (v === "" || v == null) continue;
    if (typeof v !== "number" && isNaN(parseFloat(v))) return false;
    if (++seen >= 5) break;
  }
  return seen > 0;
}
function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
