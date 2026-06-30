/* options.js — on-demand 40Δ/20Δ debit-spread R/R viewer (read-only).

   Type a ticker -> POST /exec-option -> the agent fetches that chain read-only,
   finds the ~40-delta and ~20-delta strikes, and prices the debit CALL and PUT
   verticals at the mid. We poll /exec-option for the result and render the R/R.
   Pick another expiry from the dropdown to re-quote. */
"use strict";

document.addEventListener("DOMContentLoaded", initOptions);

const state = { id: null, ticker: null, timer: null };

async function initOptions() {
  renderNav("options.html");
  const el = document.getElementById("content");
  el.innerHTML = shell();
  document.getElementById("optGo").addEventListener("click", () => quote());
  document.getElementById("optTicker").addEventListener("keydown", (e) => { if (e.key === "Enter") quote(); });
}

function shell() {
  return `
    <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:6px">
      <input id="optTicker" placeholder="ticker (e.g. AAPL)" style="width:170px;text-transform:uppercase">
      <select id="optExpiry" style="min-width:150px"><option value="">nearest weekly+</option></select>
      <button class="btn" id="optGo">Quote 40&Delta;/20&Delta; spreads</button>
      <span id="optMsg" class="cap"></span>
    </div>
    <p class="cap" style="margin:0 0 14px">Debit verticals at the mid &mdash; buy the ~40&Delta;, sell the ~20&Delta;.
      Strikes are the nearest available to those deltas, so they're approximate.</p>
    <div id="optResult"></div>`;
}

async function quote(expiry) {
  const ticker = (document.getElementById("optTicker").value || "").toUpperCase().trim();
  if (!ticker) return;
  const msg = document.getElementById("optMsg");
  msg.textContent = "fetching chain… (a few seconds)";
  clearTimeout(state.timer);
  try {
    const r = await fetch("/exec-option", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker, expiry: expiry || document.getElementById("optExpiry").value || null }),
    });
    const d = await r.json();
    if (!d.ok) { msg.textContent = "error: " + (d.error || ("HTTP " + r.status)); return; }
    state.id = d.id; state.ticker = ticker;
    pollResult(0);
  } catch (e) { msg.textContent = "error: " + e; }
}

async function pollResult(n) {
  if (n > 40) { document.getElementById("optMsg").textContent = "timed out — is the agent online?"; return; }
  const d = await fetchJSONOrNull("/exec-option") || {};
  const q = d.query;
  if (q && q.id === state.id && q.result) {
    document.getElementById("optMsg").textContent = "";
    render(q.result);
    return;
  }
  state.timer = setTimeout(() => pollResult(n + 1), 2000);
}

function render(data) {
  const el = document.getElementById("optResult");
  if (data.error) { el.innerHTML = `<div class="card"><span class="neg">${esc(data.error)}</span></div>`; return; }
  const sel = document.getElementById("optExpiry");
  if (data.expiries && data.expiries.length) {
    sel.innerHTML = data.expiries.map((e) =>
      `<option value="${e.date}" ${e.date === data.expiry ? "selected" : ""}>${e.date} (${e.dte}d)</option>`).join("");
    sel.onchange = () => quote(sel.value);
  }
  el.innerHTML = `
    <div class="card" style="margin-bottom:14px">
      <span style="font:700 16px inherit">${esc(data.ticker)}</span>
      <span class="cap" style="display:inline;margin-left:8px">spot ${fmt.num(data.spot, 2)} &middot; ${esc(data.expiry)} (${data.dte}d)</span>
    </div>
    <div class="sigcards">${spreadCard("Debit CALL spread", data.call_spread)}${spreadCard("Debit PUT spread", data.put_spread)}</div>`;
}

function spreadCard(title, s) {
  if (!s || s.error) {
    return `<div class="card"><div style="font:700 14px inherit;margin-bottom:6px">${title}</div>
      <span class="cap neg">${esc((s && s.error) || "n/a")}</span></div>`;
  }
  const L = s.long, S = s.short;
  return `<div class="card">
    <div style="font:700 14px inherit;margin-bottom:6px">${title}
      <span style="float:right;color:#4da3ff;font-weight:700">R/R ${fmt.num(s.rr, 2)} : 1</span></div>
    <div class="exec-legs" style="color:#c7ccd6">BUY  ${L.strike}  (&Delta;${L.delta})  @ ${L.mid}
SELL ${S.strike}  (&Delta;${S.delta})  @ ${S.mid}</div>
    <div class="kv" style="margin-top:10px">
      <div class="k">Cost (debit)</div><div class="v">${fmt.num(s.debit, 2)}</div>
      <div class="k">Max profit</div><div class="v pos">${fmt.num(s.max_profit, 2)}</div>
      <div class="k">Max loss</div><div class="v neg">${fmt.num(s.max_loss, 2)}</div>
      <div class="k">Breakeven</div><div class="v">${fmt.num(s.breakeven, 2)}</div>
      <div class="k">Width</div><div class="v">${fmt.num(s.width, 2)}</div>
      <div class="k">Net &Delta;</div><div class="v">${s.net_delta != null ? fmt.num(s.net_delta, 3) : "—"}</div>
    </div></div>`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
