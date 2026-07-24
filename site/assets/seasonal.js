/* seasonal.js — seasonal signal board (display-only) + manual trade sizer.

   Sources every flagged ticket from the ideas payload (all channels, including
   the non-tradeable macro names), grouped by channel. No order staging happens
   HERE — but each card (and the free-form sizer at the top) can prefill the
   Execution tab's entry-bracket ticket via a deep link (execution.html?stage=…).

   Sizing conventions (manual seasonal trades, 2026-07-09):
     risk 30 bps of NLV per trade; stop distance by hold window
     5d -> 1.0 ATR, 10d -> 1.3 ATR, 21d -> 1.6 ATR; target 2:1; time stop at
     the window's end. Close/ATR come from data/sizer.json (adjusted
     master_prices, Wilder ATR14); NLVs from the live execution book when the
     agent is online, else sizes are shown per $100k NLV. */
"use strict";

document.addEventListener("DOMContentLoaded", initSeasonal);

// evidence rows intentionally hidden on the board ("entry timing" unhidden
// 2026-07-24: with the nadir surface filter retired, the expected path
// peak/nadir day is context the card should show, not a gate)
const HIDE_EV = new Set(["TICKET", "binomial p (all-yrs)"]);

const WINDOWS = [[5, 1.0], [10, 1.3], [21, 1.6]];   // [hold td, stop ATR mult]
const RISK_BPS = 30;
const TGT_RR = 2;

const SZ = { sizer: null, nlvs: null };   // sizer.json tickers + {primary, pa} NLVs

function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function initSeasonal() {
  renderNav("seasonal.html");
  const el = document.getElementById("content");
  const [data, sizer, book] = await Promise.all([
    fetchJSONOrNull("data/ideas.json"),
    fetchJSONOrNull("data/sizer.json"),
    fetchJSONOrNull("/exec-book"),
  ]);
  SZ.sizer = sizer;
  const accs = (book && book.book && book.book.accounts) || [];
  const nlvs = {};
  for (const a of accs) if (a.nlv != null) nlvs[a.key] = a.nlv;
  SZ.nlvs = Object.keys(nlvs).length ? nlvs : null;

  if (!data) {
    el.innerHTML = '<p class="cap">No ideas payload in this build.</p>';
    return;
  }
  const meta = data.meta || {};
  setAsof(`as of ${meta.asof || "?"}`);

  let html = sizerCard();

  // every candidate carrying a trade ticket (incl. non-tradeable macro symbols)
  const cands = (data.candidates || []).filter(c => (c.evidence || {}).TICKET);
  if (!cands.length) {
    html += '<p class="cap">No qualifying seasonal setups today (rank + dual-cohort win% + stretch gates, cycle-path turn within T+5).</p>';
    el.innerHTML = html;
    wireSizer();
    return;
  }
  const byChan = new Map();
  for (const c of cands) {
    if (!byChan.has(c.channel)) byChan.set(c.channel, []);
    byChan.get(c.channel).push(c);
  }
  for (const [chan, arr] of byChan) {
    html += `<h2>${esc(chan)} <span class="cap" style="display:inline">(${arr.length})</span></h2>`;
    html += '<div class="sigcards">' + arr.map(card).join("") + "</div>";
  }
  el.innerHTML = html;
  wireSizer();
}

/* ---------- sizing helpers ---------- */
function winFor(horizon) {
  const m = String(horizon || "").match(/(\d+)/);
  const d = m ? +m[1] : 10;
  let best = WINDOWS[0];
  for (const w of WINDOWS) if (Math.abs(w[0] - d) < Math.abs(best[0] - d)) best = w;
  return best;   // [days, atrMult]
}
function sizerRec(ticker) {
  const t = (SZ.sizer && SZ.sizer.tickers) || {};
  return t[String(ticker || "").toUpperCase().trim()] || null;
}
function sharesAt(nlv, stopDist) {
  return Math.floor((nlv * RISK_BPS / 10000) / stopDist);
}
/* size line html for a ticker at [days, mult]; null when no sizer data */
function sizeLine(ticker, win) {
  const rec = sizerRec(ticker);
  if (!rec) return null;
  const [days, mult] = win;
  const dist = mult * rec.atr;
  const bits = [`stop ${mult.toFixed(1)} ATR = ${fmt.num(dist, 2)}`];
  if (SZ.nlvs) {
    for (const k of ["primary", "pa"]) {
      if (SZ.nlvs[k] != null) bits.push(`${k} <b>${sharesAt(SZ.nlvs[k], dist).toLocaleString()}</b> sh`);
    }
  } else {
    bits.push(`<b>${sharesAt(100000, dist).toLocaleString()}</b> sh per $100k NLV (agent offline)`);
  }
  return `${days}d @ ${RISK_BPS} bps: ${bits.join(" · ")}`;
}
function stageUrl(ticker, direction, win, entryPx) {
  const rec = sizerRec(ticker);
  if (!rec) return null;
  const px = entryPx > 0 ? entryPx : rec.close;
  const side = String(direction || "").toLowerCase() === "short" ? "SELL" : "BUY";
  return `execution.html?stage=1&sym=${encodeURIComponent(String(ticker).toUpperCase())}` +
         `&side=${side}&win=${win[0]}&atr=${rec.atr}&px=${px}`;
}

/* ---------- free-form sizer card ---------- */
function sizerCard() {
  const asof = SZ.sizer ? SZ.sizer.asof : null;
  if (!SZ.sizer) {
    return `<div class="card" style="margin-bottom:14px"><span class="cap">No sizer.json in this
      build — per-card trade sizes and the manual sizer need the next site deploy.</span></div>`;
  }
  const nlvNote = SZ.nlvs
    ? Object.entries(SZ.nlvs).map(([k, v]) => `${k} NLV ${fmt.money(v)}`).join(" · ")
    : "execution agent offline — sizes shown per $100k NLV";
  return `<div class="card" style="margin-bottom:14px">
    <div style="font:700 14px inherit;margin-bottom:4px">Manual seasonal sizer
      <span class="cap" style="display:inline;font-weight:400">· ${RISK_BPS} bps risk · stop 1.0/1.3/1.6 ATR
      by window · target ${TGT_RR}:1 · prices asof ${esc(asof || "?")}</span></div>
    <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:6px">
      <label class="cap">Ticker</label><input id="sz_tkr" placeholder="SMH" style="width:80px;text-transform:uppercase">
      <label class="cap">Entry</label><input id="sz_entry" placeholder="last" style="width:80px" title="Optional — defaults to the last close; stop/target/stage anchor to what's here">
      <label class="cap">Window</label><span class="seg" id="sz_win">
        ${WINDOWS.map(([d], i) => `<button ${i === 1 ? 'class="on"' : ""} data-d="${d}">${d}d</button>`).join("")}</span>
      <label class="cap">Direction</label><span class="seg" id="sz_dir">
        <button class="on" data-v="long">Long</button><button data-v="short">Short</button></span>
      <button class="btn" id="sz_stage" disabled>Stage in Execution</button>
    </div>
    <div id="sz_out" class="cap">${esc(nlvNote)}</div>
  </div>`;
}

function wireSizer() {
  const tkr = document.getElementById("sz_tkr");
  if (!tkr) return;
  const state = { win: WINDOWS[1], dir: "long" };
  const segWire = (id, cb) => {
    const seg = document.getElementById(id);
    seg.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
      seg.querySelectorAll("button").forEach(x => x.classList.remove("on"));
      b.classList.add("on");
      cb(b);
      render();
    }));
  };
  const entryEl = document.getElementById("sz_entry");
  const entryVal = () => {
    const v = Number(String(entryEl.value).trim());
    return isFinite(v) && v > 0 ? v : null;
  };
  segWire("sz_win", b => { state.win = WINDOWS.find(w => w[0] === +b.dataset.d) || WINDOWS[1]; });
  segWire("sz_dir", b => { state.dir = b.dataset.v; });
  tkr.addEventListener("input", render);
  entryEl.addEventListener("input", render);
  document.getElementById("sz_stage").addEventListener("click", () => {
    const url = stageUrl(tkr.value, state.dir, state.win, entryVal());
    if (url) window.location.href = url;
  });

  function render() {
    const out = document.getElementById("sz_out");
    const btn = document.getElementById("sz_stage");
    const t = String(tkr.value || "").toUpperCase().trim();
    if (!t) {
      out.innerHTML = SZ.nlvs
        ? Object.entries(SZ.nlvs).map(([k, v]) => `${k} NLV ${fmt.money(v)}`).join(" · ")
        : "execution agent offline — sizes shown per $100k NLV";
      btn.disabled = true;
      return;
    }
    const rec = sizerRec(t);
    if (!rec) {
      out.innerHTML = `<span style="color:#ffc14d">${esc(t)} not in the price cache — no size.</span>`;
      btn.disabled = true;
      return;
    }
    entryEl.placeholder = fmt.num(rec.close, 2);   // "last" becomes the actual number once known
    const [days, mult] = state.win;
    const dist = mult * rec.atr;
    const long = state.dir !== "short";
    const custom = entryVal();
    const px = custom != null ? custom : rec.close;
    const stop = long ? px - dist : px + dist;
    const tgt = long ? px + TGT_RR * dist : px - TGT_RR * dist;
    const rows = [
      `entry <b>${fmt.num(px, 2)}</b> ${custom != null ? `(custom · last ${fmt.num(rec.close, 2)})` : "(last)"} · ` +
      `ATR <b>${fmt.num(rec.atr, 2)}</b> · ` +
      `stop <b>${fmt.num(stop, 2)}</b> (${mult.toFixed(1)} ATR) · target <b>${fmt.num(tgt, 2)}</b> (${TGT_RR}:1) · ` +
      `time stop <b>${days}td</b>`,
    ];
    if (SZ.nlvs) {
      rows.push(Object.entries(SZ.nlvs)
        .map(([k, v]) => `${k}: <b>${sharesAt(v, dist).toLocaleString()}</b> sh (risk ${fmt.money(v * RISK_BPS / 10000)})`)
        .join(" · "));
    } else {
      rows.push(`<b>${sharesAt(100000, dist).toLocaleString()}</b> sh per $100k NLV (agent offline — ` +
                "the Execution ticket fills qty from the live NLV)");
    }
    out.innerHTML = rows.join("<br>");
    btn.disabled = false;
  }
}

/* ---------- idea cards ---------- */
function card(c) {
  const dir = String(c.direction || "").toLowerCase();
  const dirBadge = dir === "short" ? '<span class="badge dirS">SHORT</span>'
                                   : '<span class="badge dirL">LONG</span>';
  const conv = c.conviction ? `<span class="badge conv">Conviction ${esc(c.conviction)}</span>` : "";
  const horizon = c.horizon ? `<span class="chan">${esc(c.horizon)}</span>` : "";
  const ev = c.evidence || {};
  const kv = Object.entries(ev).filter(([k]) => !HIDE_EV.has(k))
    .map(([k, v]) => `<div class="k">${esc(k)}</div><div class="v">${esc(String(v))}</div>`).join("");
  // manual-trade sizing line + stage link (only when the ticker prices)
  const win = winFor(c.horizon);
  const sz = sizeLine(c.ticker, win);
  const url = sz ? stageUrl(c.ticker, c.direction, win) : null;
  const sizeRow = sz
    ? `<div class="cap" style="margin-top:8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
        <span>${sz}</span>
        ${url ? `<a class="btn xs ghost" href="${url}">Stage&hellip;</a>` : ""}</div>`
    : "";
  return `<div class="card idea">
    <div class="head">
      <span class="tkr">${esc(c.ticker || "")}</span>
      ${dirBadge} ${conv} ${horizon}
    </div>
    ${c.headline ? `<div class="headline">${esc(c.headline)}</div>` : ""}
    ${kv ? `<div class="kv">${kv}</div>` : ""}
    ${sizeRow}
  </div>`;
}
