/* seasonal.js — seasonal-ideas staged trades as thesis-first cards.

   The execution fill (T+1 open limit, or market-on-open) is unknown ahead of
   time, so the stop/target anchor to it and absolute prices can't be shown.
   Instead each card leads with the trade idea + why it flagged + hit rate
   (joined from the ideas payload by ticker), and shows execution as ATR
   distances / mechanism / size only. `sznl_nostage` rows (equity shorts +
   non-tradeable need-proxy signals) go in a compact reference table. */
"use strict";

document.addEventListener("DOMContentLoaded", initSeasonal);

const NOSTAGE_TEXT_COLS = new Set(
  ["Symbol", "Trade_Direction", "SecType", "Order_Type", "Time_Exit_Date", "Strategy_Ref"]);

function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function num(v) {
  if (v == null || v === "") return 0;
  const f = parseFloat(v);
  return isFinite(f) ? f : 0;
}

/* ticker (UPPER) -> tradeable idea candidate (the one carrying a TICKET), for the
   headline / evidence / conviction / p-value join. */
function ideaIndex(ideas) {
  const m = new Map();
  for (const c of (ideas && ideas.candidates) || []) {
    const ev = c.evidence || {};
    if (!ev.TICKET) continue;                 // only tradeable tickets become staged trades
    const k = String(c.ticker || "").toUpperCase();
    if (k && !m.has(k)) m.set(k, c);
  }
  return m;
}

async function initSeasonal() {
  renderNav("seasonal.html");
  const el = document.getElementById("content");
  const [data, ideas] = await Promise.all([
    fetchJSONOrNull("data/signals.json"),
    fetchJSONOrNull("data/ideas.json"),
  ]);
  if (!data) {
    el.innerHTML = '<p class="cap">No signals payload in this build (Sheets fetch skipped or failed).</p>';
    return;
  }
  setAsof(`fetched ${data.fetched_at}`);
  const IDEAS = ideaIndex(ideas);
  el.innerHTML = "";
  const tabs = data.tabs || {};

  // --- staged seasonal trades (thesis cards) ---
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
    cap.textContent = `Total staged risk ${fmt.money(totRisk)} across ${staged.length} trades. `
      + "Entry / stop / target prices resolve off the live fill, so cards show the thesis and ATR distances, not fixed levels.";
    el.appendChild(cap);
    const grid = document.createElement("div");
    grid.className = "sigcards";
    grid.innerHTML = staged.map(r => seasonalCard(r, IDEAS.get(String(r.Symbol).toUpperCase()))).join("");
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
                  "Stop_ATR_Mult", "Tgt_ATR_Mult", "Hold_Days", "Time_Exit_Date", "Strategy_Ref"];
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

/* One staged seasonal trade. `idea` (joined by ticker) supplies the thesis;
   `r` (the staged row) supplies direction + execution mechanics. No prices. */
function seasonalCard(r, idea) {
  const sym = esc(r.Symbol || (idea && idea.ticker) || "?");
  const isShort = String(r.Trade_Direction || r.Action || "").toUpperCase().includes("SHORT")
    || String(r.Action || "").toUpperCase() === "SELL";
  const dirBadge = isShort ? '<span class="badge dirS">SHORT</span>'
                           : '<span class="badge dirL">LONG</span>';
  const conv = idea && idea.conviction
    ? `<span class="badge conv">Conviction ${esc(idea.conviction)}</span>` : "";
  const horizon = esc((idea && idea.horizon) || String(r.Strategy_Ref || "").split("/")[2] || "");
  const pv = idea && idea.p_value != null
    ? `<span class="cap" style="display:inline">p=${Number(idea.p_value).toFixed(3)}</span>` : "";

  const headline = idea && idea.headline ? `<div class="headline">${esc(idea.headline)}</div>` : "";

  // evidence = "why it flagged" + hit rates (all-years / midterm). Drop the raw
  // TICKET line — it's the price plan, which we deliberately don't surface.
  const ev = (idea && idea.evidence) || {};
  const kv = Object.entries(ev).filter(([k]) => k !== "TICKET").map(([k, v]) =>
    `<div class="k">${esc(k)}</div><div class="v">${esc(String(v))}</div>`).join("");
  const evHtml = kv ? `<div class="kv">${kv}</div>`
    : (idea ? "" : '<p class="cap">Thesis unavailable for this ticker in the ideas payload.</p>');

  // execution — mechanism + ATR distances + size, never absolute prices
  const ot = String(r.Order_Type || "");
  const offs = num(r.Offset_ATR_Mult) || 0.25;
  const entry = ot === "MOO" ? "Market-on-open"
    : `T+1 open ${isShort ? "+" : "-"} ${offs} ATR limit, ${esc(r.TIF || "DAY")}`;
  const sMult = num(r.Stop_ATR_Mult), tMult = num(r.Tgt_ATR_Mult);
  const useStop = String(r.Use_Stop).toLowerCase() === "true" && sMult > 0;
  const useTgt = String(r.Use_Target).toLowerCase() === "true" && tMult > 0;
  const exitBits = [
    useStop ? `<b class="neg">${sMult} ATR</b> stop` : "",
    useTgt ? `<b class="pos">${tMult} ATR</b> target` : "",
  ].filter(Boolean).join(" &middot; ");
  const hold = num(r.Hold_Days);
  const texit = esc(String(r.Time_Exit_Date || "").slice(0, 10));
  const risk = num(r.Risk_Amt), bps = num(r.Risk_Bps), qty = num(r.Quantity);

  const execKv = [
    `<div class="k">Enter</div><div class="v">${esc(entry)}</div>`,
    `<div class="k">Exit</div><div class="v">${exitBits || "time only"} off the fill; hold ${hold}td${texit ? ` &rarr; by ${texit}` : ""}</div>`,
    `<div class="k">Size</div><div class="v">${fmt.money(risk)} risk${bps ? ` (${bps} bps)` : ""}${qty ? ` &middot; ${qty.toLocaleString()} sh` : ""}</div>`,
  ].join("");

  return `<div class="card idea">
    <div class="head">
      <span class="tkr">${sym}</span>
      ${dirBadge} ${conv}
      ${horizon ? `<span class="chan">${horizon}</span>` : ""}
      ${pv}
    </div>
    ${headline}
    ${evHtml}
    <div class="cap" style="margin:8px 0 2px">Execution</div>
    <div class="kv">${execKv}</div>
  </div>`;
}
