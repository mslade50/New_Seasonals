/* signals.js — staged scan signals as cards (Order_Staging + Overflow tabs).

   One card per staged order: ticker, direction, strategy, entry mechanics,
   estimated bracket levels (computed from Limit/Signal Close +/- ATR mults —
   actual prices resolve off the live fill in order_staging), risk, and any
   special stamps (OVS 2-path, Monday-gap kill, T+1 open filters). The raw
   rows stay available in a collapsed table per tab. */
"use strict";

document.addEventListener("DOMContentLoaded", init);

// strategy (UPPERCASE) -> {note, cycle} context joined from the ideas-page
// payloads; rendered as a "Strategy context" block on each card
let S_CTX = new Map();

const NOTE_LABEL = {
  size_up: ["dirL", "SIZE-UP TILT"], size_down: ["dirS", "SIZE-DOWN TILT"],
  hold: ["conv", "HOLD NATIVE"], thin: ["warn", "THIN SAMPLE"], neutral: ["conv", "NEUTRAL"],
};

function buildStratContext(notes, ideas) {
  const m = new Map();
  for (const n of (notes && notes.notes) || []) {
    m.set(n.strategy.toUpperCase(), { note: n, cycle: null });
  }
  // Regime / Sleeve Tilt candidates use the strategy name (uppercased) as
  // their ticker; "BOOK" is book-level and stays on the ideas page.
  for (const c of (ideas && ideas.candidates) || []) {
    if (!/sleeve|regime/i.test(c.channel || "") || !c.ticker || c.ticker === "BOOK") continue;
    const k = String(c.ticker).toUpperCase();
    if (!m.has(k)) m.set(k, { note: null, cycle: null });
    m.get(k).cycle = c;
  }
  return m;
}

async function init() {
  renderNav("signals.html");
  const el = document.getElementById("content");
  const [data, notes, ideas] = await Promise.all([
    fetchJSONOrNull("data/signals.json"),
    fetchJSONOrNull("data/strat_notes.json"),
    fetchJSONOrNull("data/ideas.json"),
  ]);
  S_CTX = buildStratContext(notes, ideas);
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

  // open-dependent execution rules, with concrete computed levels
  const openLines = openConditions(r, sigClose, atr, isShort);
  const openHtml = openLines.length
    ? `<div class="openconds"><div class="oc-h">At the open</div>${
        openLines.map(l => `<div class="oc-line">${l}</div>`).join("")}</div>` : "";

  // why it fired — live filter readings stamped at signal time
  let critHtml = "";
  try {
    const lf = typeof r.Live_Filters === "string" && r.Live_Filters.length > 2
      ? JSON.parse(r.Live_Filters) : null;
    if (lf && lf.length) {
      critHtml = `<div class="openconds"><div class="oc-h">Why it fired</div>${
        lf.map(([d, v]) =>
          `<div class="oc-line">${esc(d)} <b>${esc(String(v))}</b></div>`).join("")}</div>`;
    }
  } catch (e) { /* malformed stamp — skip */ }

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
    ${openHtml}
    ${critHtml}
    ${stratContextHtml(strat)}
  </div>`;
}

/* "Strategy context" block — joined from strat_notes.json (trailing-percentile
   regime note) and ideas.json (cycle/sleeve tilt). Same data as the Ideas page,
   surfaced where the order decision happens. */
function stratContextHtml(stratName) {
  const ctx = S_CTX.get(String(stratName).toUpperCase());
  if (!ctx || (!ctx.note && !ctx.cycle)) return "";
  const lines = [];
  if (ctx.note) {
    const n = ctx.note;
    const [cls, label] = NOTE_LABEL[n.action] || NOTE_LABEL.neutral;
    const pctCls = n.bucket === "cold" ? "neg" : n.bucket === "hot" ? "pos" : "";
    lines.push(`<div class="oc-line"><span class="badge ${cls}">${label}</span> ` +
      `trailing 20-trade avg R <b class="${pctCls}">${fmt.signed(n.trail_avg_r, 2)}</b> ` +
      `(<b class="${pctCls}">${Math.round(n.trail_pct)}th %ile</b> of its history); ` +
      `3mo ${fmt.signed(n.trail_3mo_r, 1)}R.</div>`);
    if (n.verdict) lines.push(`<div class="oc-line">${esc(n.verdict)}</div>`);
  }
  if (ctx.cycle && ctx.cycle.headline) {
    const h = String(ctx.cycle.headline);
    const tone = /fade|de-risk|under/i.test(h) ? "dirS" : /lean into|favorable|outperform/i.test(h) ? "dirL" : "conv";
    lines.push(`<div class="oc-line"><span class="badge ${tone}">CYCLE</span> ${esc(h)}</div>`);
  }
  return `<div class="openconds"><div class="oc-h">Strategy context</div>${lines.join("")}</div>`;
}

/* Concrete price levels for the rules order_staging enforces at the T+1 open */
function openConditions(r, sigClose, atr, isShort) {
  const lines = [];
  if (!sigClose || !atr) return lines;

  // OVS 2-path gap tier (threshold = close + 0.25 ATR, shorts enter into strength)
  if (num(r.Path1_Bps)) {
    const thr = sigClose + 0.25 * atr;
    lines.push(`<b class="pos">Path 1</b> (${num(r.Path1_Bps)} bps full) if open &gt; ${money(thr)}`);
    lines.push(`<b>Path 2</b> (${num(r.Path2_Bps)} bps, 1% daily cap) if open ${money(sigClose)}&ndash;${money(thr)}`);
    lines.push(`<b class="neg">Skip</b> if open &le; ${money(sigClose)}`);
  }

  // Monday-gap kill (weekday-gated by SIGNAL date)
  if (num(r.MonGapKill_ATR)) {
    const mult = num(r.MonGapKill_ATR);
    const dir = String(r.MonGapKill_Dir || "up");
    const level = dir === "down" ? sigClose - mult * atr : sigClose + mult * atr;
    let active = true;
    try {
      const wds = JSON.parse(r.MonGapKill_Weekdays || "[]");
      const sd = new Date(String(r.Scan_Date) + "T00:00:00Z");
      // JS getUTCDay: Sun=0..Sat=6 -> python Mon=0..Fri=4
      active = wds.length === 0 || wds.includes((sd.getUTCDay() + 6) % 7);
    } catch (e) { /* keep active=true */ }
    if (active) {
      lines.push(`<b class="neg">Killed</b> if open ${dir === "down" ? "&lt;" : "&gt;"} ${money(level)} ` +
                 `(${mult} ATR gap-${dir} rule)`);
    }
  }

  // generic T+1 open filters: [{logic, reference, atr_offset}]
  try {
    const tf = typeof r.T1_Open_Filters === "string" && r.T1_Open_Filters.length > 2
      ? JSON.parse(r.T1_Open_Filters) : null;
    if (tf) {
      for (const f of tf) {
        const refVal = /high/i.test(f.reference || "") ? num(r.Signal_High) : sigClose;
        if (!refVal) continue;
        const lvl = refVal + (parseFloat(f.atr_offset) || 0) * atr;
        const off = f.atr_offset ? ` (${f.reference} ${f.atr_offset > 0 ? "+" : ""}${f.atr_offset} ATR)` : ` (${f.reference})`;
        lines.push(`Requires open ${esc(f.logic)} ${money(lvl)}${esc(off)}`);
      }
    }
  } catch (e) { /* malformed spec — skip */ }

  return lines;
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
