/* seasonal.js — seasonal signal board (display-only).

   Sources every flagged ticket from the ideas payload (all channels, including
   the non-tradeable macro names), grouped by channel. No order staging. Stripped
   cards: ticker / direction / conviction / horizon + the trade idea + thesis
   evidence — no execution, no binomial-p, no enter/exit/size. The screener only
   surfaces signals whose ATR path bottoms the very next session, so every card
   here is an imminent nadir (long) / peak (short) setup. */
"use strict";

document.addEventListener("DOMContentLoaded", initSeasonal);

// evidence rows intentionally hidden on the board
const HIDE_EV = new Set(["TICKET", "binomial p (all-yrs)", "entry timing"]);

function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function initSeasonal() {
  renderNav("seasonal.html");
  const el = document.getElementById("content");
  const data = await fetchJSONOrNull("data/ideas.json");
  if (!data) {
    el.innerHTML = '<p class="cap">No ideas payload in this build.</p>';
    return;
  }
  const meta = data.meta || {};
  setAsof(`as of ${meta.asof || "?"}`);

  // every candidate carrying a trade ticket (incl. non-tradeable macro symbols)
  const cands = (data.candidates || []).filter(c => (c.evidence || {}).TICKET);
  if (!cands.length) {
    el.innerHTML = '<p class="cap">No seasonal setups bottoming next session today.</p>';
    return;
  }
  const byChan = new Map();
  for (const c of cands) {
    if (!byChan.has(c.channel)) byChan.set(c.channel, []);
    byChan.get(c.channel).push(c);
  }
  let html = "";
  for (const [chan, arr] of byChan) {
    html += `<h2>${esc(chan)} <span class="cap" style="display:inline">(${arr.length})</span></h2>`;
    html += '<div class="sigcards">' + arr.map(card).join("") + "</div>";
  }
  el.innerHTML = html;
}

function card(c) {
  const dir = String(c.direction || "").toLowerCase();
  const dirBadge = dir === "short" ? '<span class="badge dirS">SHORT</span>'
                                   : '<span class="badge dirL">LONG</span>';
  const conv = c.conviction ? `<span class="badge conv">Conviction ${esc(c.conviction)}</span>` : "";
  const horizon = c.horizon ? `<span class="chan">${esc(c.horizon)}</span>` : "";
  const ev = c.evidence || {};
  const kv = Object.entries(ev).filter(([k]) => !HIDE_EV.has(k))
    .map(([k, v]) => `<div class="k">${esc(k)}</div><div class="v">${esc(String(v))}</div>`).join("");
  return `<div class="card idea">
    <div class="head">
      <span class="tkr">${esc(c.ticker || "")}</span>
      ${dirBadge} ${conv} ${horizon}
    </div>
    ${c.headline ? `<div class="headline">${esc(c.headline)}</div>` : ""}
    ${kv ? `<div class="kv">${kv}</div>` : ""}
  </div>`;
}
