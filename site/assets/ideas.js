/* ideas.js — render data/ideas.json (daily_seasonal_ideas output) */
"use strict";

document.addEventListener("DOMContentLoaded", init);

const NOTE_BADGE = {
  size_up:   { cls: "dirL", label: "SIZE-UP TILT" },
  size_down: { cls: "dirS", label: "SIZE-DOWN TILT" },
  hold:      { cls: "conv", label: "HOLD NATIVE" },
  thin:      { cls: "warn", label: "THIN SAMPLE" },
  neutral:   { cls: "conv", label: "NEUTRAL" },
};

function renderStratNotes(payload) {
  if (!payload || !payload.notes || !payload.notes.length) return;
  document.getElementById("stratNotesSection").style.display = "";
  const W = payload.window;
  document.getElementById("stratNotes").innerHTML = payload.notes.map(n => {
    const b = NOTE_BADGE[n.action] || NOTE_BADGE.neutral;
    const pctCls = n.bucket === "cold" ? "neg" : n.bucket === "hot" ? "pos" : "neu";
    const r3Cls = clsSign(n.trail_3mo_r);
    return `<div class="card idea">
      <div class="head">
        <span class="tkr">${esc(n.strategy)}</span>
        <span class="badge ${b.cls}">${b.label}</span>
        <span class="chan">${n.n_trades.toLocaleString()} trades all-time</span>
      </div>
      <div class="headline">Trailing ${W}-trade avg R <b class="${pctCls}">${fmt.signed(n.trail_avg_r, 2)}</b>
        — <b class="${pctCls}">${Math.round(n.trail_pct)}th %ile</b> of its own history.
        Trailing 3mo: <b class="${r3Cls}">${fmt.signed(n.trail_3mo_r, 1)}R</b> over ${n.n_3mo_trades} trades.</div>
      <p class="cap" style="margin:4px 0 0">${esc(n.verdict)}</p>
    </div>`;
  }).join("");
}

async function init() {
  renderNav("ideas.html");
  const el = document.getElementById("content");
  const [data, notes] = await Promise.all([
    fetchJSONOrNull("data/ideas.json"),
    fetchJSONOrNull("data/strat_notes.json"),
  ]);
  renderStratNotes(notes);
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
