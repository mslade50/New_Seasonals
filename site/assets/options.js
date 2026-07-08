/* options.js — Options Workbench: signal-to-structure pipeline.

   Flow (top to bottom): prefill (query string from a Signals-page "Express"
   button, or manual ticker) -> IV context strip (rank/percentile/RV from
   iv_context.json) -> expiry picker (real listed expiries via the agent) ->
   edge-vs-priced expected-move comparator (implied move over the hold vs the
   strategy's terminal move from strategy_stats.json) -> structure shootout at
   equal risk -> risk-bps sizing (debit-at-risk) -> BAG combo ticket.

   Live chain data comes from POST /exec-workbench (agent -> option_workbench.py,
   one round-trip; expiry clicks re-query mode:"chain"). Orders go through the
   signed /exec-command path as type "option_spread" — DRY-RUN until the user
   arms the type in exec_agent.env. Exit-date P&L columns are client-side BSM
   (bsm.js) with dividend/rate inputs — labeled approximation.

   EM convention (everywhere): 1-sigma move = IV_ATM * sqrt(days/365), calendar
   days. The straddle mid is shown as "market price of the move", never blended. */
"use strict";

document.addEventListener("DOMContentLoaded", initOptions);

const COMM = 0.65;                    // $/contract per leg per side
const state = {
  params: {},                          // prefill from the query string
  ivCtx: null, stats: null, earn: null, signals: null,
  wb: null,                            // latest workbench result
  wbId: null, wbTimer: null,
  structures: [],                      // shootout rows (assembled client-side)
  selStructure: null,
  account: "primary",
  book: null, status: null, commands: [],
  pollTimer: null,
};

/* ---------------- init + prefill ---------------- */
async function initOptions() {
  renderNav("options.html");
  state.params = parseParams();
  const [iv, stats, earn, sig] = await Promise.all([
    fetchJSONOrNull("data/iv_context.json"),
    fetchJSONOrNull("data/strategy_stats.json"),
    fetchJSONOrNull("data/earnings_next.json"),
    fetchJSONOrNull("data/signals.json"),
  ]);
  state.ivCtx = iv; state.stats = stats; state.earn = earn; state.signals = sig;
  document.getElementById("content").innerHTML = shell();
  document.getElementById("wbGo").addEventListener("click", () => loadTicker());
  document.getElementById("wbTicker").addEventListener("keydown", (e) => { if (e.key === "Enter") loadTicker(); });
  document.querySelectorAll("[data-acct]").forEach((b) =>
    b.addEventListener("click", () => setAccount(b.dataset.acct)));
  await pollExec();
  state.pollTimer = setInterval(pollExec, 8000);
  if (state.params.ticker) {
    document.getElementById("wbTicker").value = state.params.ticker;
    loadTicker();
  }
}

function parseParams() {
  const q = new URLSearchParams(location.search);
  const num = (k) => { const v = q.get(k); return v == null || v === "" ? null : Number(v); };
  return {
    ticker: (q.get("ticker") || "").toUpperCase().trim() || null,
    dir: q.get("dir") || "Long",
    entry: num("entry"), stop: num("stop"), target: num("target"), atr: num("atr"),
    risk: num("risk"), hold: num("hold"),
    texit: q.get("texit") || null, fw: num("fw"),
    strategy: q.get("strategy") || null, sig: q.get("sig") || null,
    cond: q.get("cond") || null,
  };
}

function hasSignal() { const p = state.params; return !!(p.entry && p.stop && p.risk); }
function isShort() { return String(state.params.dir).toUpperCase().includes("SHORT"); }

function shell() {
  const p = state.params;
  return `
    <div id="modeBanner"></div>
    <div class="card" style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:10px 14px;margin-bottom:12px">
      <input id="wbTicker" placeholder="ticker" style="width:110px;text-transform:uppercase">
      <button class="btn" id="wbGo">Load workbench</button>
      <span class="exec-tabs"><button class="btn" data-acct="primary">Primary</button>
        <button class="btn ghost" data-acct="pa">PA</button></span>
      <span id="wbMsg" class="cap"></span>
      <span id="connDot" class="cap" style="margin-left:auto"></span>
    </div>
    ${signalContextHtml(p)}
    <div id="ivStrip"></div>
    <div id="expiryRow"></div>
    <div id="emCompare"></div>
    <div id="shootout"></div>
    <div id="sizing"></div>
    <div id="ticket"></div>
    <div id="activity" style="margin-top:18px"></div>`;
}

function signalContextHtml(p) {
  if (!hasSignal()) {
    return `<p class="cap" style="margin:0 0 12px">No signal context — manual mode. Open this page via a
      Signals-page <b>Express</b> button to pre-fill entry/stop/target/risk and unlock the comparator,
      shootout scenario P&amp;L, and sizing.</p>`;
  }
  const dupe = stagedStockRow(p.ticker);
  const dupeWarn = dupe ? `<div class="oc-line" style="color:#ffc14d"><b>&#9888; Stock signal also staged for
      ${esc(p.ticker)}</b> — an options ticket runs ALONGSIDE it (combined delta). Size the fraction below accordingly.</div>` : "";
  return `<div class="card" style="margin-bottom:12px">
    <div style="font:700 14px inherit">${esc(p.ticker || "")} · ${esc(p.dir)} ·
      <span style="color:#4da3ff">${esc(p.strategy || "manual")}</span>
      <span class="cap" style="display:inline;margin-left:8px">signal ${esc(p.sig || "?")}</span></div>
    <div class="kv" style="margin-top:8px">
      <div class="k">Entry / Stop / Target</div><div class="v">${fmt.num(p.entry, 2)} / ${fmt.num(p.stop, 2)} / ${p.target ? fmt.num(p.target, 2) : "—"}</div>
      <div class="k">ATR / Hold / Time exit</div><div class="v">${fmt.num(p.atr, 2)} / ${p.hold || "?"} td / ${esc(p.texit || "?")}</div>
      <div class="k">Stock Risk_Amt</div><div class="v">${fmt.money(p.risk)}</div>
    </div>
    ${p.cond ? `<div class="openconds" style="margin-top:8px"><div class="oc-h">Entry condition (stock-side — submit the combo manually when it triggers)</div>
      <div class="oc-line">${esc(p.cond)}</div></div>` : ""}
    ${dupeWarn}</div>`;
}

function stagedStockRow(ticker) {
  const tabs = (state.signals && state.signals.tabs) || {};
  for (const rows of Object.values(tabs)) {
    const hit = (rows || []).find((r) => String(r.Symbol || "").toUpperCase() === ticker);
    if (hit) return hit;
  }
  return null;
}

/* ---------------- workbench query ---------------- */
async function loadTicker(expiry) {
  const ticker = (document.getElementById("wbTicker").value || "").toUpperCase().trim();
  if (!ticker) return;
  const msg = document.getElementById("wbMsg");
  msg.textContent = expiry ? "re-quoting expiry…" : "fetching chain + term structure… (~15-20s)";
  clearTimeout(state.wbTimer);
  const p = state.params;
  const body = {
    ticker,
    mode: expiry ? "chain" : "full",
    expiry: expiry || null,
    context: hasSignal() && ticker === p.ticker ? {
      direction: p.dir, entry: p.entry, stop: p.stop, target: p.target,
      atr: p.atr, hold_days: p.hold, time_exit_date: p.texit,
    } : null,
  };
  try {
    const r = await fetch("/exec-workbench", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const d = await r.json();
    if (!d.ok) { msg.textContent = "error: " + (d.error || ("HTTP " + r.status)); return; }
    state.wbId = d.id;
    pollWb(0, expiry);
  } catch (e) { msg.textContent = "error: " + e; }
}

async function pollWb(n, expiryOnly) {
  if (n > 45) { document.getElementById("wbMsg").textContent = "timed out — is the agent online?"; return; }
  const d = (await fetchJSONOrNull(`/exec-workbench?id=${encodeURIComponent(state.wbId)}`)) || {};
  const q = d.query;
  if (q && q.id === state.wbId && q.result) {
    document.getElementById("wbMsg").textContent = "";
    if (q.result.error) {
      document.getElementById("shootout").innerHTML =
        `<div class="card"><span class="neg">${esc(q.result.error)}</span></div>`;
      return;
    }
    if (expiryOnly && state.wb) {           // chain-only re-quote keeps the term pass
      state.wb.chain = q.result.chain;
      state.wb.spot = q.result.spot;
      state.wb.asof = q.result.asof;
      state.wb.market_data_type = q.result.market_data_type;
    } else {
      state.wb = q.result;
    }
    renderAll();
    return;
  }
  state.wbTimer = setTimeout(() => pollWb(n + 1, expiryOnly), 2000);
}

function renderAll() {
  renderIvStrip();
  renderExpiries();
  renderComparator();
  buildStructures();
  renderShootout();
  renderSizing();
  renderTicket();
}

/* ---------------- IV context strip ---------------- */
function ivRegime(pctile) {
  if (pctile == null) return ["#9aa3b2", "no history"];
  if (pctile < 30) return ["#3ddb8f", "premium CHEAP — debit structures"];
  if (pctile < 50) return ["#9aa3b2", "middling — tiebreak on cost of delta"];
  if (pctile < 80) return ["#ffc14d", "elevated — spreads over singles"];
  return ["#ff6b6b", "RICH — sell premium to buy delta (credit structures)"];
}
function renderIvStrip() {
  const el = document.getElementById("ivStrip");
  const wb = state.wb;
  const rec = state.ivCtx && state.ivCtx[wb.ticker];
  if (!rec) {
    el.innerHTML = `<div class="card" style="margin-bottom:12px"><b>${esc(wb.ticker)}</b>
      <span class="badge warn" style="margin-left:8px">NO IV HISTORY</span>
      <span class="cap" style="display:inline;margin-left:8px">the local IV cache hasn't covered this name —
      rank/percentile unavailable; judge richness off IV vs realized manually.</span></div>`;
    return;
  }
  const [tone, label] = ivRegime(rec.pctile);
  const rvTiles = [10, 21, 63].map((n) => {
    const rv = rec[`rv${n}`];
    if (rv == null) return "";
    const sp = (rec.iv - rv) * 100;
    const cls = sp > 4 ? "neg" : sp < 0 ? "pos" : "";
    return `<div class="k">IV − RV${n}</div><div class="v ${cls}">${fmt.signed(sp, 1)} pts</div>`;
  }).join("");
  el.innerHTML = `<div class="card" style="margin-bottom:12px">
    <div style="display:flex;gap:14px;align-items:center;flex-wrap:wrap">
      <span style="font:700 15px inherit">IV30 ${fmt.pctRaw(rec.iv * 100, 1)}</span>
      <span>rank <b>${rec.rank != null ? fmt.num(rec.rank, 0) : "—"}</b> · pctile <b>${fmt.num(rec.pctile, 0)}</b></span>
      <span style="color:${tone};font-weight:700">${label}</span>
      <span class="cap" style="display:inline;margin-left:auto">as of ${esc(rec.last)}</span>
    </div>
    <div style="display:flex;gap:20px;align-items:center;margin-top:8px;flex-wrap:wrap">
      <div id="ivSpark" style="width:280px;height:52px"></div>
      <div class="kv" style="flex:1;min-width:220px">${rvTiles}</div>
    </div></div>`;
  if (rec.spark && rec.spark.length > 2 && window.Plotly) {
    Plotly.newPlot("ivSpark", [{ y: rec.spark, mode: "lines", line: { color: "#4da3ff", width: 1.5 } }],
      plotLayout({ margin: { l: 2, r: 2, t: 2, b: 2 }, xaxis: { visible: false }, yaxis: { visible: false },
                   height: 52, showlegend: false, hovermode: false }), PLOT_CFG);
  }
}

/* ---------------- expiry picker ---------------- */
function bdaysUntil(iso) {
  if (!iso) return null;
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const end = new Date(iso + "T00:00:00");
  if (isNaN(end) || end <= today) return 0;
  let n = 0; const d = new Date(today);
  while (d < end) { d.setDate(d.getDate() + 1); if (d.getDay() > 0 && d.getDay() < 6) n++; }
  return n;
}
function expLabel(e) { return `${e.slice(4, 6)}/${e.slice(6, 8)}`; }

function renderExpiries() {
  const el = document.getElementById("expiryRow");
  const wb = state.wb;
  const exps = wb.expiries || [];
  if (!exps.length) { el.innerHTML = ""; return; }
  const texitBd = bdaysUntil(state.params.texit);
  const earnDate = state.earn && state.earn[wb.ticker];
  const noEarnData = state.earn && (state.earn._no_data || []).includes(wb.ticker);
  const cells = exps.map((e, i) => {
    const sel = wb.chain && wb.chain.expiry === e.date;
    const dteAtExit = texitBd != null ? Math.round(e.dte * (252 / 365)) - texitBd : null;
    const earnFlag = earnDate && earnDate.replace(/-/g, "") <= e.date ? " &#9888;" : "";
    // forward vol vs the prior expiry (variance additivity)
    let fwd = "";
    if (i > 0 && e.atm_iv && exps[i - 1].atm_iv && e.dte > exps[i - 1].dte) {
      const v = (e.atm_iv ** 2 * e.dte - exps[i - 1].atm_iv ** 2 * exps[i - 1].dte) / (e.dte - exps[i - 1].dte);
      if (v > 0) fwd = `fwd ${(Math.sqrt(v) * 100).toFixed(0)}`;
    }
    const warn = dteAtExit != null && dteAtExit < 7 ? ' style="color:#ffc14d"' : "";
    return `<button class="btn ${sel ? "" : "ghost"}" data-exp="${e.date}" style="flex-direction:column;line-height:1.3">
      <span>${expLabel(e.date)} <span class="cap" style="display:inline">(${e.dte}d)</span>${earnFlag}</span>
      <span class="cap" style="display:inline">${e.atm_iv ? "IV " + (e.atm_iv * 100).toFixed(0) : "IV —"}${fwd ? " · " + fwd : ""}</span>
      ${dteAtExit != null ? `<span class="cap"${warn} style="display:inline">${dteAtExit}d left at exit</span>` : ""}
    </button>`;
  }).join("");
  const earnNote = noEarnData
    ? `<span class="badge warn">NO EARNINGS DATA</span> <span class="cap" style="display:inline">this ticker has no
       earnings coverage — long premium may straddle an unflagged report; verify the date yourself.</span>`
    : earnDate ? `<span class="cap" style="display:inline">next earnings <b>${esc(earnDate)}</b>${
        state.params.texit && earnDate <= state.params.texit ? ' <b style="color:#ff6b6b">INSIDE THE HOLD</b>' : ""}</span>` : "";
  el.innerHTML = `<div class="card" style="margin-bottom:12px">
    <div style="font:700 14px inherit;margin-bottom:6px">Expiry
      <span class="cap" style="display:inline;font-weight:400">· default = first listed &ge; time-exit + 5 td · &#9888; = earnings before expiry</span></div>
    <div style="display:flex;gap:8px;flex-wrap:wrap">${cells}</div>
    <div style="margin-top:6px">${earnNote}</div></div>`;
  el.querySelectorAll("[data-exp]").forEach((b) =>
    b.addEventListener("click", () => loadTicker(b.dataset.exp)));
}

/* ---------------- edge-vs-priced comparator ---------------- */
function holdCalDays() {
  const h = state.params.hold;
  return h ? Math.round(h * 365 / 252) : null;
}
function chainAtmIv() {
  const wb = state.wb;
  const e = (wb.expiries || []).find((x) => wb.chain && x.date === wb.chain.expiry);
  if (e && e.atm_iv) return e.atm_iv;
  // chain-only mode: median IV of the 5 strikes nearest spot
  const rows = ((wb.chain || {}).strikes || []).filter((r) => r.iv != null);
  if (!rows.length) return null;
  rows.sort((a, b) => Math.abs(a.strike - wb.spot) - Math.abs(b.strike - wb.spot));
  const ivs = rows.slice(0, 6).map((r) => r.iv).sort((a, b) => a - b);
  return ivs[Math.floor(ivs.length / 2)];
}
function renderComparator() {
  const el = document.getElementById("emCompare");
  const wb = state.wb, p = state.params;
  const stats = p.strategy && state.stats && state.stats[p.strategy];
  const iv = chainAtmIv();
  const cal = holdCalDays();
  if (!hasSignal() || !stats || !iv || !cal) {
    el.innerHTML = !hasSignal() ? "" : `<div class="card" style="margin-bottom:12px"><span class="cap">
      Edge comparator unavailable (${!stats ? "no strategy stats for " + esc(p.strategy || "?") : "no ATM IV"}).</span></div>`;
    return;
  }
  const implied = iv * Math.sqrt(cal / 365);                       // 1-sigma over the hold
  const tm = stats.terminal_move || {};
  const fc = Math.abs(tm.median_pct != null ? tm.median_pct : tm.mean_pct || 0);
  const ratio = implied > 0 ? fc / implied : null;
  const [tone, verdict] = ratio == null ? ["#9aa3b2", "—"]
    : ratio > 1.2 ? ["#3ddb8f", "your edge exceeds what's priced — long premium is cheap"]
    : ratio >= 0.8 ? ["#ffc14d", "roughly fair — decide on cost of delta + IV regime"]
    : ["#ff6b6b", "market prices a bigger move than your signal delivers — prefer credit structures or stock"];
  const straddle = (wb.expiries || []).find((x) => wb.chain && x.date === wb.chain.expiry);
  const w = (v) => Math.min(100, v / Math.max(implied, fc) * 100);
  el.innerHTML = `<div class="card" style="margin-bottom:12px">
    <div style="font:700 14px inherit;margin-bottom:6px">Edge vs priced
      <span class="cap" style="display:inline;font-weight:400">· 1&sigma; = IV&middot;&radic;(days/365), scaled to the ${p.hold} td hold</span></div>
    <div class="kv">
      <div class="k">Implied 1&sigma; over hold</div>
      <div class="v"><span style="display:inline-block;background:#4da3ff33;border-left:3px solid #4da3ff;padding:1px 6px;width:${w(implied)}%">${fmt.pctRaw(implied * 100, 2)}</span></div>
      <div class="k">${esc(p.strategy)} median terminal move</div>
      <div class="v"><span style="display:inline-block;background:#00d18f33;border-left:3px solid #00d18f;padding:1px 6px;width:${w(fc)}%">${fmt.pctRaw(fc * 100, 2)}</span>
        <span class="cap" style="display:inline">(mean ${fmt.pctRaw(Math.abs(tm.mean_pct || 0) * 100, 2)}, n=${stats.n}, win ${fmt.pct(stats.win_rate, 0)})</span></div>
      <div class="k">Edge ratio</div>
      <div class="v"><b style="color:${tone}">${ratio != null ? fmt.num(ratio, 2) : "—"}</b> — ${verdict}</div>
      ${straddle && straddle.straddle_mid ? `<div class="k">ATM straddle (market price of the move)</div>
        <div class="v">${fmt.num(straddle.straddle_mid, 2)} <span class="cap" style="display:inline">separate quantity — never blended into the ratio</span></div>` : ""}
    </div></div>`;
}

/* ---------------- structure assembly ---------------- */
function chainRows(right) {
  return (((state.wb || {}).chain || {}).strikes || [])
    .filter((r) => r.right === right)
    .sort((a, b) => a.strike - b.strike);
}
function byDelta(rows, absD) {
  let best = null, bd = 1e9;
  for (const r of rows) {
    if (r.delta == null) continue;
    const d = Math.abs(Math.abs(r.delta) - absD);
    if (d < bd) { bd = d; best = r; }
  }
  return best;
}
function byStrike(rows, px) {
  let best = null, bd = 1e9;
  for (const r of rows) {
    const d = Math.abs(r.strike - px);
    if (d < bd) { bd = d; best = r; }
  }
  return best;
}
function legPrice(row, side, kind) {   // kind: "mid" | "nat"
  if (kind === "mid") return row.mid;
  return side === "BUY" ? row.ask : row.bid;   // natural: pay the ask, hit the bid
}
function spreadFrom(name, longRow, shortRow, right, note) {
  if (!longRow || (shortRow && longRow.strike === shortRow.strike)) return null;
  const legs = [{ side: "BUY", row: longRow }];
  if (shortRow) legs.push({ side: "SELL", row: shortRow });
  let mid = 0, nat = 0, delta = 0, ok = true;
  for (const l of legs) {
    const m = legPrice(l.row, l.side, "mid"), n = legPrice(l.row, l.side, "nat");
    if (m == null) ok = false;
    const s = l.side === "BUY" ? 1 : -1;
    mid += s * (m || 0);
    nat += s * (n != null ? n : (m || 0));
    if (l.row.delta != null) delta += s * l.row.delta;
  }
  if (!ok || mid <= 0) return null;
  return { name, right, legs, mid: round2(mid), nat: round2(nat), delta: round2(delta, 3),
           width: shortRow ? Math.abs(longRow.strike - shortRow.strike) : null, note: note || "", tradeable: true };
}
function round2(v, d = 2) { return v == null ? null : Math.round(v * 10 ** d) / 10 ** d; }

function buildStructures() {
  const wb = state.wb, p = state.params;
  state.structures = [];
  if (!wb || !wb.chain) return;
  const short = isShort();
  const right = short ? "P" : "C";
  const rows = chainRows(right);
  if (!rows.length) return;
  const spot = wb.spot;
  const entryPx = p.entry || spot, tgtPx = p.target;

  // Directional helpers: for calls the short strike sits ABOVE the long; for puts below.
  const anchorLong = hasSignal() ? byStrike(rows, entryPx) : byDelta(rows, 0.40);
  const anchorShort = tgtPx ? byStrike(rows, tgtPx) : byDelta(rows, 0.20);
  const atmLong = byStrike(rows, spot);
  const d40 = byDelta(rows, 0.40), d20 = byDelta(rows, 0.20);

  const add = (s) => { if (s && !state.structures.some((x) => sameLegs(x, s))) state.structures.push(s); };
  add(spreadFrom("Target-anchored vertical", anchorLong, anchorShort, right,
    tgtPx ? "short strike at the stock target" : "long ~entry / short ~20Δ"));
  add(spreadFrom("ATM vertical", atmLong, anchorShort, right, "long ATM"));
  add(spreadFrom("40Δ/20Δ vertical", d40, d20, right, "the O1 preset"));
  const single = spreadFrom(`Long ${right === "C" ? "call" : "put"} ~40Δ`, d40, null, right, "unlimited upside, full theta");
  add(single);

  // Credit vertical on the OPPOSITE side (sell the move NOT happening): analytics
  // only in phase 1 (ticket is debit-only). Screen: credit >= 25% of width.
  const oppRows = chainRows(short ? "C" : "P");
  const s30 = byDelta(oppRows, 0.30), l15 = byDelta(oppRows, 0.15);
  if (s30 && l15 && s30.strike !== l15.strike && s30.mid != null && l15.mid != null) {
    const credit = round2(s30.mid - l15.mid);
    const width = Math.abs(s30.strike - l15.strike);
    if (credit > 0 && credit >= 0.25 * width) {
      state.structures.push({
        name: `Credit ${short ? "call" : "put"} spread 30Δ/15Δ`, right: short ? "C" : "P",
        legs: [{ side: "SELL", row: s30 }, { side: "BUY", row: l15 }],
        mid: credit, nat: round2((s30.bid ?? s30.mid) - (l15.ask ?? l15.mid)),
        delta: round2(-(s30.delta || 0) + (l15.delta || 0), 3),
        width, credit: true, note: `collect ${credit} on ${width} width — not tradeable from this ticket yet`,
        tradeable: false,
      });
    }
  }
  if (!state.structures.some((s) => s.tradeable)) return;
  if (!state.selStructure || !state.structures.some((s) => s.name === state.selStructure)) {
    state.selStructure = state.structures[0].name;
  }
}
function sameLegs(a, b) {
  const key = (s) => s.legs.map((l) => `${l.side}${l.row.strike}${l.row.right}`).join("|");
  return key(a) === key(b);
}

/* ---------------- exit-date scenario pricing (BSM approximation) ---------------- */
function scenarioInputs() {
  return {
    r: numInput("sh_rate", 0.04),
    q: numInput("sh_divy", 0),
    ivShift: numInput("sh_ivshift", -3) / 100,
  };
}
function numInput(id, dflt) {
  const e = document.getElementById(id);
  if (!e || e.value === "") return dflt;
  const v = Number(e.value);
  return isFinite(v) ? v : dflt;
}
function valueAtExit(struct, spotAtExit, inp) {
  // sum of leg values at the planned exit date; T from each leg's expiry
  const exitBd = state.params.hold || 0;
  const exitCal = Math.round(exitBd * 365 / 252);
  let val = 0;
  for (const l of struct.legs) {
    // all phase-1 legs share the chain expiry; per-leg DTE comes with calendars later
    const dte = ((state.wb || {}).chain || {}).dte || 0;
    const T = Math.max(0, (dte - exitCal) / 365);
    const sigma = Math.max(0.01, (l.row.iv != null ? l.row.iv : chainAtmIv() || 0.3) + inp.ivShift);
    const v = BSM.price(spotAtExit, l.row.strike, T, sigma, inp.r, inp.q, l.row.right);
    val += (l.side === "BUY" ? 1 : -1) * (v || 0);
  }
  return val;
}
function scenarioPnl(struct) {
  // per-spread $ P&L at the planned exit for target / stop / flat + EV weights
  const p = state.params;
  if (!hasSignal() || !p.target) return null;
  const inp = scenarioInputs();
  const stats = p.strategy && state.stats && state.stats[p.strategy];
  const short = isShort();
  const cost = struct.credit ? -struct.mid : struct.mid;
  const pnlAt = (spot) => (valueAtExit(struct, spot, inp) - cost) * 100;
  const out = {
    target: pnlAt(p.target),
    stop: pnlAt(p.stop),
    flat: pnlAt(p.entry),
  };
  if (stats) {
    const lm = (stats.loser_mix || {}).avg_loser_move_pct;
    const loserSpot = lm != null ? p.entry * (1 + (short ? -1 : 1) * lm) : p.stop;
    out.ev = stats.win_rate * out.target + (1 - stats.win_rate) * pnlAt(loserSpot);
  }
  return out;
}

/* ---------------- shootout ---------------- */
function commRT(struct) {
  const n = struct.legs.length;
  return COMM * n * 2;               // per spread, round trip
}
function spreadTax(struct) {
  if (struct.mid == null || struct.nat == null || struct.mid <= 0) return null;
  return (struct.nat - struct.mid) / struct.mid;
}
function taxLight(t) {
  if (t == null) return "—";
  const pct = t * 100;
  const c = pct <= 3 ? "#3ddb8f" : pct <= 8 ? "#ffc14d" : "#ff6b6b";
  return `<span style="color:${c};font-weight:700">&#9679;</span> ${pct.toFixed(1)}%`;
}
function breakevenPct(struct) {
  if (struct.credit || !struct.legs.length) return null;
  const long = struct.legs[0].row;
  const be = long.right === "C" ? long.strike + struct.mid : long.strike - struct.mid;
  const spot = state.wb.spot;
  return (long.right === "C" ? be - spot : spot - be) / spot;
}

function renderShootout() {
  const el = document.getElementById("shootout");
  const wb = state.wb, p = state.params;
  if (!wb || !wb.chain) { el.innerHTML = ""; return; }
  if (!state.structures.length) {
    el.innerHTML = `<div class="card"><span class="neg">No quotable structures in the band (missing mids/greeks
      — thin chain or delayed data).</span></div>`;
    return;
  }
  const delayed = wb.market_data_type === 3 || wb.market_data_type === 4;
  const age = wb.asof ? Math.max(0, Math.round(Date.now() / 1000 - wb.asof)) : null;
  const stats = p.strategy && state.stats && state.stats[p.strategy];

  // stock baseline row (only with full signal context)
  let baselineHtml = "";
  if (hasSignal() && p.target) {
    const riskAlloc = sizeAlloc();
    const dist = Math.abs(p.entry - p.stop);
    const sh = dist > 0 ? Math.floor(riskAlloc / dist) : 0;
    const pnlT = sh * Math.abs(p.target - p.entry);
    const ev = stats ? stats.win_rate * pnlT + (1 - stats.win_rate) *
      (-(sh * dist) * Math.abs(((stats.loser_mix || {}).avg_loser_move_pct || (-dist / p.entry)) * p.entry) / dist) : null;
    baselineHtml = `<tr style="border-top:2px solid #2a3242">
      <td class="l"><b>Stock with stop</b> <span class="cap" style="display:inline">(baseline)</span></td>
      <td class="l">${sh} sh @ ${fmt.num(p.entry, 2)}</td>
      <td>—</td><td>—</td>
      <td>${fmt.num(sh, 0)}</td><td>—</td><td>0%</td>
      <td class="pos">${fmt.money(pnlT)}</td>
      <td class="neg">${fmt.money(-riskAlloc)}</td>
      <td>~$1</td>
      <td>${ev != null ? `<b class="${clsSign(ev)}">${fmt.money(ev)}</b>` : "—"}</td>
      <td>—</td><td></td></tr>`;
  }

  const rows = state.structures.map((s) => {
    const sc = scenarioPnl(s);
    const be = breakevenPct(s);
    const maxP = s.width != null ? (s.credit ? s.mid : s.width - s.mid) * 100 : null;
    const maxL = s.credit ? (s.width - s.mid) * 100 : s.mid * 100;
    const premPerDelta = !s.credit && s.delta ? s.mid / Math.abs(s.delta) : null;
    const legsTxt = s.legs.map((l) => `${l.side[0]} ${l.row.strike}${l.row.right}${l.row.delta != null ? ` (${Math.abs(l.row.delta).toFixed(2)}Δ)` : ""}`).join(" / ");
    const sel = s.name === state.selStructure;
    return `<tr${sel ? ' style="background:#4da3ff14"' : ""}>
      <td class="l"><b>${esc(s.name)}</b>${s.note ? `<br><span class="cap" style="display:inline">${esc(s.note)}</span>` : ""}</td>
      <td class="l">${esc(legsTxt)}</td>
      <td>${fmt.num(s.mid, 2)}${s.credit ? " cr" : ""}</td>
      <td>${s.nat != null ? fmt.num(s.nat, 2) : "—"}</td>
      <td>${s.delta != null ? fmt.num(s.delta * 100, 0) : "—"}</td>
      <td>${premPerDelta != null ? fmt.num(premPerDelta, 2) : "—"}</td>
      <td>${be != null ? fmt.pctRaw(be * 100, 1) : "—"}</td>
      <td class="${sc ? clsSign(sc.target) : ""}">${sc ? fmt.money(sc.target) : "—"}</td>
      <td class="${sc ? clsSign(sc.stop) : ""}">${sc ? fmt.money(sc.stop) : "—"}</td>
      <td>${fmt.money(commRT(s))}</td>
      <td>${sc && sc.ev != null ? `<b class="${clsSign(sc.ev)}">${fmt.money(sc.ev)}</b>` : "—"}</td>
      <td>${taxLight(spreadTax(s))}</td>
      <td class="l">${s.tradeable ? `<button class="btn xs${sel ? "" : " ghost"}" data-struct="${esc(s.name)}">${sel ? "selected" : "select"}</button>` : '<span class="cap">n/a</span>'}</td>
    </tr>`;
  }).join("");

  // verdict: best EV among rows that have one, vs stock baseline
  let verdict = "";
  if (hasSignal() && p.target && stats) {
    const evs = state.structures.map((s) => ({ s, sc: scenarioPnl(s) })).filter((x) => x.sc && x.sc.ev != null);
    const riskAlloc = sizeAlloc();
    const dist = Math.abs(p.entry - p.stop);
    const sh = dist > 0 ? Math.floor(riskAlloc / dist) : 0;
    const stockEv = stats.win_rate * sh * Math.abs(p.target - p.entry) +
      (1 - stats.win_rate) * -(riskAlloc * 0.8);
    if (evs.length) {
      const best = evs.reduce((a, b) => (b.sc.ev > a.sc.ev ? b : a));
      verdict = best.sc.ev > stockEv
        ? `<div style="margin-top:8px"><b style="color:#3ddb8f">Verdict:</b> <b>${esc(best.s.name)}</b> wins on EV (${fmt.money(best.sc.ev)} vs stock ${fmt.money(stockEv)}) — per-spread, at the planned exit.</div>`
        : `<div style="margin-top:8px"><b style="color:#ffc14d">Verdict: USE STOCK</b> — no structure beats the stock baseline after costs (${fmt.money(stockEv)} vs best ${fmt.money(best.sc.ev)}).</div>`;
    }
  }

  el.innerHTML = `<div class="card" style="margin-bottom:12px">
    <div style="font:700 14px inherit;margin-bottom:4px">Structure shootout
      <span class="cap" style="display:inline;font-weight:400">· spot ${fmt.num(wb.spot, 2)} · ${esc(wb.chain.expiry)} (${wb.chain.dte}d)
      ${age != null ? `· quotes ${age}s old` : ""}
      ${delayed ? ' · <b style="color:#ffc14d">DELAYED MARKS — do not anchor limits off these</b>' : ""}</span></div>
    <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:4px 0 8px">
      <label class="cap">Exit-date repricing (BSM approx):</label>
      <label class="cap">IV shift</label><input id="sh_ivshift" value="-3" style="width:50px"> <span class="cap">pts</span>
      <label class="cap">rate</label><input id="sh_rate" value="0.04" style="width:56px">
      <label class="cap">div yield</label><input id="sh_divy" value="0" style="width:50px">
      <span class="cap">P&amp;L columns are per spread at the ${state.params.hold || "?"} td exit, not expiry.</span>
    </div>
    <div class="tblwrap"><table class="tbl"><thead><tr>
      <th class="l">Structure</th><th class="l">Legs</th><th>Mid</th><th>Nat</th><th>&Delta; sh-eq</th>
      <th>$/&Delta;</th><th>BE move</th><th>P&amp;L @tgt</th><th>P&amp;L @stop</th><th>Comm RT</th>
      <th>EV</th><th>Spread tax</th><th class="l"></th>
    </tr></thead><tbody>${rows}${baselineHtml}</tbody></table></div>
    ${verdict}</div>`;
  el.querySelectorAll("[data-struct]").forEach((b) =>
    b.addEventListener("click", () => { state.selStructure = b.dataset.struct; renderShootout(); renderSizing(); renderTicket(); }));
  ["sh_ivshift", "sh_rate", "sh_divy"].forEach((id) => {
    const e = document.getElementById(id);
    if (e) e.addEventListener("change", () => { renderShootout(); renderSizing(); renderTicket(); });
  });
}

/* ---------------- sizing ---------------- */
function selStruct() { return state.structures.find((s) => s.name === state.selStructure) || null; }
function sizeFraction() { return numInput("sz_frac", 0.5); }
function sizeAlloc() { return (state.params.risk || 0) * sizeFraction(); }
function contractsFor(struct, alloc) {
  const per = struct.mid * 100 + commRT(struct);
  return per > 0 ? Math.floor(alloc / per) : 0;
}
function renderSizing() {
  const el = document.getElementById("sizing");
  const s = selStruct();
  if (!s || !hasSignal()) { el.innerHTML = ""; return; }
  const alloc = sizeAlloc();
  const n = contractsFor(s, alloc);
  const per = s.mid * 100 + commRT(s);
  const eff = n * per;
  const effUp = (n + 1) * per;
  const warnings = [];
  if (n < 1) warnings.push(`target risk ${fmt.money(alloc)} is below one spread's cost (${fmt.money(per)}) — no viable size`);
  const quantErr = alloc > 0 ? (alloc - eff) / alloc : 0;
  el.innerHTML = `<div class="card" style="margin-bottom:12px">
    <div style="font:700 14px inherit;margin-bottom:6px">Sizing — debit at risk
      <span class="cap" style="display:inline;font-weight:400">· contracts = &lfloor;fraction &middot; Risk_Amt / (debit&times;100 + comm)&rfloor;</span></div>
    <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:8px">
      <label class="cap">Fraction of stock Risk_Amt</label>
      <input id="sz_frac" value="0.5" style="width:56px">
      <span class="cap">(options run ALONGSIDE a smaller stock position — 0.5 default)</span>
    </div>
    <div class="kv">
      <div class="k">Risk allocation</div><div class="v">${fmt.money(alloc)} <span class="cap" style="display:inline">of ${fmt.money(state.params.risk)} staged</span></div>
      <div class="k">Contracts</div><div class="v"><b style="font-size:15px">${n}</b> &times; ${esc(s.name)} @ ${fmt.num(s.mid, 2)}</div>
      <div class="k">Effective risk</div><div class="v">${fmt.money(eff)} <span class="cap" style="display:inline">(${fmt.pctRaw(quantErr * 100, 1)} under target; ${n + 1} lots = ${fmt.money(effUp)})</span></div>
      <div class="k">Net delta</div><div class="v">${s.delta != null ? fmt.num(s.delta * 100 * n, 0) + " share-equivalents" : "—"}
        ${stagedStockRow(state.params.ticker) ? ' <b style="color:#ffc14d">+ the staged stock position</b>' : ""}</div>
    </div>
    ${warnings.length ? `<div style="color:#ff6b6b;margin-top:8px">${warnings.map(esc).join("<br>")}</div>` : ""}
  </div>`;
  const fr = document.getElementById("sz_frac");
  if (fr) fr.addEventListener("change", () => { renderSizing(); renderTicket(); });
}

/* ---------------- ticket ---------------- */
function snapLimit(v) { return Math.round(Math.floor((v + 1e-9) / 0.05) * 5) / 100; }
function renderTicket() {
  const el = document.getElementById("ticket");
  const s = selStruct();
  const wb = state.wb;
  if (!s || !s.tradeable || !wb) { el.innerHTML = ""; return; }
  const n = hasSignal() ? contractsFor(s, sizeAlloc()) : 1;
  const dfltLimit = snapLimit(Math.max(0.05, s.mid - 0.01)).toFixed(2);
  const legLines = s.legs.map((l) =>
    `${l.side} ${l.row.strike}${l.row.right} ${wb.chain.expiry} (conId ${l.row.con_id || "?"})`).join("<br>");
  el.innerHTML = `<div class="card" style="max-width:760px;margin-bottom:12px">
    <div style="font:700 14px inherit;margin-bottom:4px">Combo ticket — ${esc(s.name)}</div>
    <p class="cap" style="margin:0 0 8px">One native BAG limit order (SMART, atomic — never legs you in). Submits per the
      mode banner: <b>option_spread is unarmed</b>, so this dry-runs until you add it to LIVE_TYPES.</p>
    <div class="exec-legs" style="margin-bottom:8px">${legLines}</div>
    ${state.params.cond ? `<div class="openconds" style="margin-bottom:8px"><div class="oc-h">Entry condition — check before sending</div>
      <div class="oc-line">${esc(state.params.cond)}</div></div>` : ""}
    <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px">
      <label class="cap">Qty</label><input id="tk_qty" value="${n > 0 ? n : ""}" style="width:60px">
      <label class="cap">Net limit</label><input id="tk_limit" value="${dfltLimit}" style="width:76px">
      <span class="cap">(mid ${fmt.num(s.mid, 2)} / natural ${s.nat != null ? fmt.num(s.nat, 2) : "?"}; 0.05 grid, BUY snaps down)</span>
      <label class="cap">TIF</label><select id="tk_tif"><option>DAY</option><option>GTC</option></select>
      <span class="cap">Account: <b>${esc(state.account)}</b></span>
    </div>
    <button class="btn" id="tk_send">Send combo</button>
    <span id="tk_msg" class="cap" style="margin-left:10px"></span>
  </div>`;
  document.getElementById("tk_send").addEventListener("click", sendSpread);
}

function execMode() {
  if (state.book && state.book.mode === "live") return "live";
  const online = !!(state.status && state.status.online);
  const fresh = !!(state.book && state.book.at && (Date.now() - state.book.at) <= 90000);
  if (online && fresh && state.book.mode === "dry-run") return "dry-run";
  return "unknown";
}
function actionLead(verb) {
  const m = execMode();
  return m === "dry-run" ? `Dry-run ${verb} (places nothing):` : `⚠️ LIVE — really ${verb}`;
}

const idemState = { id: null, key: null };
function commandId(type, account, payload) {
  const key = JSON.stringify({ type, account, payload });
  if (idemState.key !== key || !idemState.id) {
    idemState.id = crypto.randomUUID();
    idemState.key = key;
  }
  return idemState.id;
}

function sendSpread() {
  const s = selStruct();
  const wb = state.wb, p = state.params;
  const msg = document.getElementById("tk_msg");
  if (!s || !wb) return;
  const qty = Math.floor(Number(document.getElementById("tk_qty").value));
  const limit = Number(document.getElementById("tk_limit").value);
  if (!(qty > 0)) { msg.textContent = "BLOCKED: qty must be a positive integer"; return; }
  if (!(limit > 0)) { msg.textContent = "BLOCKED: limit must be > 0"; return; }
  if (s.width != null && limit >= s.width) { msg.textContent = `BLOCKED: net debit ${limit} >= width ${s.width}`; return; }
  const snapped = snapLimit(limit);
  const payload = {
    symbol: wb.ticker,
    action: "BUY",
    quantity: qty,
    limit: snapped,
    tif: document.getElementById("tk_tif").value || "DAY",
    structure: s.name.toLowerCase().replace(/[^a-z0-9]+/g, "_"),
    debit_risk: snapped,               // debit-at-risk basis = what the order can pay
    legs: s.legs.map((l) => ({
      side: l.side, right: l.row.right, expiry: wb.chain.expiry,
      strike: l.row.strike, ratio: 1, con_id: l.row.con_id || null,
    })),
    strategy: p.strategy || null, signal_date: p.sig || null,
    entry_condition: p.cond || null,
  };
  const legsTxt = payload.legs.map((l) => `${l.side[0]}${l.strike}${l.right}`).join("/");
  if (!confirm(`${actionLead("place")} BUY ${qty}x ${wb.ticker} ${wb.chain.expiry} [${legsTxt}] LMT ${payload.limit} net on ${state.account}?`)) return;
  sendCommand("option_spread", payload, "tk_msg");
}

async function sendCommand(type, payload, msgId) {
  const msg = document.getElementById(msgId);
  if (msg) msg.textContent = "sending...";
  const id = commandId(type, state.account, payload);
  try {
    const r = await fetch("/exec-command", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, type, account: state.account, payload }),
    });
    const d = await r.json();
    const ok = r.ok && d && d.ok;
    if (ok) { idemState.id = null; idemState.key = null; }
    if (msg) msg.textContent = ok ? `queued ${(d.id || id).slice(0, 8)} — see Activity below` : `error: ${(d && d.error) || ("HTTP " + r.status)}`;
  } catch (e) {
    if (msg) msg.textContent = "error: " + e;
  }
  setTimeout(pollExec, 800);
}

/* ---------------- execution status / activity ---------------- */
function setAccount(acct) {
  state.account = acct;
  document.querySelectorAll("[data-acct]").forEach((b) =>
    b.className = "btn" + (b.dataset.acct === acct ? "" : " ghost"));
  renderTicket();
}
async function pollExec() {
  const [st, bk, cm] = await Promise.all([
    fetchJSONOrNull("/exec-status"),
    fetchJSONOrNull("/exec-book"),
    fetchJSONOrNull("/exec-commands"),
  ]);
  state.status = st || { online: false };
  state.book = (bk && bk.book) || null;
  state.commands = ((cm && cm.commands) || []).filter((c) => c.type === "option_spread" || c.type === "echo");
  const dot = document.getElementById("connDot");
  if (dot) dot.textContent = state.status.online ? "agent online" : "agent offline";
  setAsof(state.status.online ? "execution online" : "execution offline");
  renderModeBanner();
  renderActivity();
}
function renderModeBanner() {
  const el = document.getElementById("modeBanner");
  if (!el) return;
  const mode = execMode();
  if (mode === "live") {
    el.innerHTML = `<div class="card" style="border-color:#a8852f;background:rgba(255,193,77,.10);padding:9px 14px;font:700 13px inherit;color:#ffc14d">
      &#9888;&#65039; LIVE ARMED — a combo sent here transmits to IBKR if option_spread is in LIVE_TYPES.</div>`;
  } else if (mode === "unknown") {
    el.innerHTML = `<div class="card" style="border-color:#a8852f;background:rgba(255,193,77,.10);padding:9px 14px;font:700 13px inherit;color:#ffc14d">
      &#9888;&#65039; MODE UNKNOWN — assume LIVE. No fresh book confirms dry-run.</div>`;
  } else {
    el.innerHTML = `<div class="card" style="border-color:#2c8f63;background:rgba(61,219,143,.08);padding:9px 14px;font:700 13px inherit;color:#3ddb8f">
      &#9679; DRY-RUN MODE — combos are validated + previewed, nothing transmits.</div>`;
  }
}
function stateBadge(st) {
  const map = { dry_run: ["#3ddb8f", "DRY-RUN OK"], rejected: ["#ff6b6b", "REJECTED"],
                executed: ["#ffc14d", "EXECUTED"], duplicate: ["#9aa3b2", "duplicate"],
                pushed: ["#9aa3b2", "pushed"], error: ["#ffc14d", "ERROR"] };
  const [c, t] = map[st] || ["#9aa3b2", st || ""];
  return `<span style="color:${c};font-weight:600">${esc(t)}</span>`;
}
function renderActivity() {
  const el = document.getElementById("activity");
  if (!el) return;
  const cmds = state.commands || [];
  if (!cmds.length) { el.innerHTML = ""; return; }
  const rows = cmds.map((c) => {
    const res = c.result || {};
    const pv = res.preview || {};
    let cell = `<span class="${res.ok === true ? "pos" : res.ok === false ? "neg" : "neu"}">${esc(res.detail || c.state || "pending")}</span>`;
    if (pv.legs && pv.legs.length) {
      cell += `<div class="exec-legs">${pv.legs.map(esc).join("<br>")}` +
        (pv.summary ? `<br><span style="color:#c7ccd6;font-weight:600">${esc(pv.summary)}</span>` : "") + "</div>";
    }
    const t = c.created_at ? new Date(c.created_at).toLocaleTimeString("en-US", { hour12: false }) : "";
    return `<tr style="vertical-align:top"><td class="l" style="color:#8c95a2">${esc(t)}</td>
      <td class="l">${esc(c.account || "")}</td><td class="l">${stateBadge(c.state)}</td><td class="l">${cell}</td></tr>`;
  }).join("");
  el.innerHTML = `<div style="font:700 14px inherit;margin-bottom:6px">Option-spread activity</div>
    <div class="tblwrap"><table class="tbl"><thead><tr>
    <th class="l">time</th><th class="l">acct</th><th class="l">state</th><th class="l">result / preview</th>
    </tr></thead><tbody>${rows}</tbody></table></div>`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
