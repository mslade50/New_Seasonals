/* ideas.js — render data/ideas.json (daily_seasonal_ideas output) */
"use strict";

document.addEventListener("DOMContentLoaded", init);

async function init() {
  renderNav("ideas.html");
  const el = document.getElementById("content");
  const data = await fetchJSONOrNull("data/ideas.json");
  if (!data) {
    el.innerHTML = '<p class="cap">No ideas payload in this build (daily_seasonal_ideas.py did not run).</p>';
    return;
  }
  const meta = data.meta || {};
  setAsof(`ideas as of ${meta.asof || "?"}`);

  // regime banner
  const reg = document.getElementById("regime");
  reg.innerHTML = `<div class="card" style="margin-bottom:14px">
    <div class="cap" style="margin-top:0">REGIME</div>
    <div>${esc(meta.regime || "")}</div>
    <div class="cap">${mdLite(meta.summary || "")}</div>
    ${(meta.stale_notes || []).map(n => `<div class="note">${esc(n)}</div>`).join(" ")}
  </div>`;

  const cands = data.candidates || [];
  if (!cands.length) {
    el.innerHTML = '<p class="cap">No candidates surfaced today.</p>';
  } else {
    // group by channel, preserve order of appearance
    const byChan = new Map();
    for (const c of cands) {
      if (!byChan.has(c.channel)) byChan.set(c.channel, []);
      byChan.get(c.channel).push(c);
    }
    let html = "";
    for (const [chan, arr] of byChan) {
      html += `<h2>${esc(chan)} <span class="cap" style="display:inline">(${arr.length})</span></h2>`;
      for (const c of arr) html += ideaCard(c);
    }
    el.innerHTML = html;
  }

  const foot = document.getElementById("footer");
  if (meta.footer) foot.textContent = meta.footer;
}

function ideaCard(c) {
  const dir = (c.direction || "").toLowerCase();
  const dirBadge = dir === "long" ? '<span class="badge dirL">LONG</span>'
    : dir === "short" ? '<span class="badge dirS">SHORT</span>'
    : `<span class="badge conv">${esc((c.direction || "").toUpperCase())}</span>`;
  const conv = c.conviction ? `<span class="badge conv">Conviction ${esc(c.conviction)}</span>` : "";
  const ev = c.evidence || {};
  const kv = Object.entries(ev).map(([k, v]) =>
    `<div class="k">${esc(k)}</div><div class="v">${esc(String(v))}</div>`).join("");
  return `<div class="card idea">
    <div class="head">
      <span class="tkr">${esc(c.ticker || "")}</span>
      ${dirBadge} ${conv}
      <span class="chan">${esc(c.horizon || "")}</span>
      ${c.p_value != null ? `<span class="cap" style="display:inline">p=${Number(c.p_value).toFixed(3)}</span>` : ""}
    </div>
    <div class="headline">${esc(c.headline || "")}</div>
    <div class="kv">${kv}</div>
    ${c.notes ? `<p class="cap">${esc(c.notes)}</p>` : ""}
  </div>`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function mdLite(s) {
  return esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
}
