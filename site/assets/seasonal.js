/* seasonal.js — seasonal-ideas staged trades as cards.

   Reuses signalCard() + its helpers (num/money/esc, buildStratContext, the
   shared S_CTX binding) from signals.js, which is loaded first and shares the
   classic-script top-level scope. signals.js's own init bails on this page
   (body[data-page=seasonal]); we run our own renderer here.

   Renders the `Seasonal` tab as cards (the staged seasonal trades) and
   `sznl_nostage` (equity shorts + non-tradeable need-proxy signals) as a
   compact reference table. */
"use strict";

document.addEventListener("DOMContentLoaded", initSeasonal);

const NOSTAGE_TEXT_COLS = new Set(
  ["Symbol", "Trade_Direction", "SecType", "Order_Type", "Time_Exit_Date", "Strategy_Ref"]);

async function initSeasonal() {
  renderNav("seasonal.html");
  const el = document.getElementById("content");
  const [data, notes, ideas] = await Promise.all([
    fetchJSONOrNull("data/signals.json"),
    fetchJSONOrNull("data/strat_notes.json"),
    fetchJSONOrNull("data/ideas.json"),
  ]);
  S_CTX = buildStratContext(notes, ideas);   // shared binding declared in signals.js
  if (!data) {
    el.innerHTML = '<p class="cap">No signals payload in this build (Sheets fetch skipped or failed).</p>';
    return;
  }
  setAsof(`fetched ${data.fetched_at}`);
  el.innerHTML = "";
  const tabs = data.tabs || {};

  // --- staged seasonal trades (cards) ---
  const staged = (tabs["Seasonal"] || []).slice()
    .sort((a, b) => num(b.Risk_Amt) - num(a.Risk_Amt));
  const h = document.createElement("h2");
  h.textContent = `Staged (${staged.length})`;
  el.appendChild(h);
  if (!staged.length) {
    const p = document.createElement("p");
    p.className = "cap";
    p.textContent = "No staged seasonal trades in this snapshot.";
    el.appendChild(p);
  } else {
    const totRisk = staged.reduce((s, r) => s + num(r.Risk_Amt), 0);
    const cap = document.createElement("p");
    cap.className = "cap";
    cap.textContent = `Total staged risk ${money(totRisk)} across ${staged.length} trades.`;
    el.appendChild(cap);
    const grid = document.createElement("div");
    grid.className = "sigcards";
    grid.innerHTML = staged.map(signalCard).join("");
    el.appendChild(grid);
  }

  // --- not auto-staged (eq shorts + need-proxy) as a compact table ---
  const nostage = (tabs["sznl_nostage"] || []).slice();
  const h2 = document.createElement("h2");
  h2.textContent = `Not auto-staged (${nostage.length})`;
  el.appendChild(h2);
  const note = document.createElement("p");
  note.className = "cap";
  note.textContent = "Single-stock equity shorts and non-tradeable signals "
    + "(futures / index / FX needing a proxy ETF). Reference only — order_staging does not submit these.";
  el.appendChild(note);
  if (!nostage.length) {
    const p = document.createElement("p");
    p.className = "cap";
    p.textContent = "None.";
    el.appendChild(p);
  } else {
    const wrap = document.createElement("div");
    wrap.className = "card";
    el.appendChild(wrap);
    const cols = ["Symbol", "Trade_Direction", "SecType", "Order_Type", "Quantity",
                  "Signal_Close", "Stop_ATR_Mult", "Tgt_ATR_Mult", "Hold_Days",
                  "Time_Exit_Date", "Strategy_Ref"];
    makeTable(wrap, {
      columns: cols.map(k => ({
        key: k, label: k.replace(/_/g, " "),
        align: NOSTAGE_TEXT_COLS.has(k) ? "l" : "r",
        fmt: v => v == null ? "" : esc(String(v)),
      })),
      rows: nostage, search: nostage.length > 8,
    });
  }
}
