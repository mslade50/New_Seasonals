/* option-positions.js - Close / Roll for HELD option positions (Execution tab).

   Loaded after execution.js and shares its globals (state, esc, fmt, sendCommand,
   execMode, actionLead, acctBook, set). Pure helpers at the top are unit-tested
   (tests/js/test_option_positions.js).

   * Structures: option legs that share account + underlying + orderRef (the
     orderRef comes from the trailing fills window, /exec-fills, by conId) are
     grouped into one structure; legs without a known orderRef stand alone. The
     owner can tick legs of one underlying to group them by hand.
   * Close...  opens a ticket prefilled with the reverse legs, the held quantity
     and a limit at mid (quotes via the existing /exec-workbench chain path),
     shows natural/mid, P&L vs average cost and quote age, and sends ONE
     `option_close` command (plain OPT LMT for one leg, one SMART BAG otherwise).
   * Roll...   opens the Options builder in roll mode (options.html?section=custom
     &roll=...), closing legs locked, new legs picked from the ladder.
   Both appear only when the running agent advertises the command type in
   book.capabilities (an agent without the list = not supported).
   The executor re-checks everything against the live broker positions. */
"use strict";

const OPT_POS = { picked: new Set(), fills: null, fillsAt: 0, ticket: null, seq: 0 };
const OPT_QUOTE_STALE_S = 120;

/* ---------------- pure helpers (unit-tested) ---------------- */

function optCapabilities(book) {
  const caps = book && Array.isArray(book.capabilities) ? book.capabilities : [];
  return new Set(caps.map(String));
}
function optSupports(book, type) { return optCapabilities(book).has(type); }

function optLegKey(p) { return `${p.account_key || ""}|${Number(p.con_id) || 0}`; }

/* conId -> most recent orderRef among fills of that account. */
function optOrderRefByConId(fills, accountKey) {
  const out = new Map(), at = new Map();
  for (const f of fills || []) {
    if (!f || (accountKey && f.account_key && f.account_key !== accountKey)) continue;
    const cid = Number(f.con_id) || 0, ref = f.order_ref ? String(f.order_ref) : "";
    if (!cid || !ref || String(f.sec_type || "OPT").toUpperCase() === "BAG") continue;
    const t = Date.parse(f.time) || 0;
    if (!out.has(cid) || t >= at.get(cid)) { out.set(cid, ref); at.set(cid, t); }
  }
  return out;
}

/* Group option positions into structures. picked = Set of conIds ticked by the
   owner: ticked legs of ONE underlying form one manual group (overriding the
   orderRef grouping). Returns [{id, symbol, legs:[pos], source}] where source is
   "manual" | "order_ref" | "single". */
function optGroupPositions(positions, accountKey, refByConId, picked) {
  const opts = (positions || []).filter((p) => String(p.sec_type).toUpperCase() === "OPT" && Number(p.position) && Number(p.con_id));
  const groups = [];
  const used = new Set();
  const pickedLegs = opts.filter((p) => picked && picked.has(Number(p.con_id)));
  const pickedSyms = new Set(pickedLegs.map((p) => String(p.symbol).toUpperCase()));
  if (pickedLegs.length >= 1 && pickedSyms.size === 1) {
    groups.push({ id: `manual:${accountKey}:${[...pickedSyms][0]}`, symbol: [...pickedSyms][0], legs: pickedLegs, source: "manual" });
    pickedLegs.forEach((p) => used.add(Number(p.con_id)));
  }
  const byRef = new Map();
  for (const p of opts) {
    const cid = Number(p.con_id);
    if (used.has(cid)) continue;
    const ref = refByConId ? refByConId.get(cid) : null;
    if (!ref) { groups.push({ id: `single:${accountKey}:${cid}`, symbol: String(p.symbol).toUpperCase(), legs: [p], source: "single" }); continue; }
    const key = `${accountKey}|${String(p.symbol).toUpperCase()}|${ref}`;
    if (!byRef.has(key)) byRef.set(key, []);
    byRef.get(key).push(p);
  }
  for (const [key, legs] of byRef) {
    groups.push({ id: `ref:${key}`, symbol: String(legs[0].symbol).toUpperCase(), legs, source: legs.length > 1 ? "order_ref" : "single" });
  }
  return groups;
}

function optGcd(a, b) { a = Math.abs(a); b = Math.abs(b); while (b) [a, b] = [b, a % b]; return a; }

/* Reverse legs of a held structure: [{con_id, action, ratio, held, ...identity}] and
   the number of whole structure units held. Ratios are reduced by the gcd. */
function optCloseLegs(legs) {
  const held = legs.map((p) => Math.round(Math.abs(Number(p.position))));
  const units = held.reduce((g, h) => optGcd(g, h), 0) || 1;
  return {
    units,
    legs: legs.map((p, i) => ({
      con_id: Number(p.con_id), action: Number(p.position) > 0 ? "SELL" : "BUY", ratio: held[i] / units,
      held: Number(p.position), right: String(p.right || "").toUpperCase(), strike: Number(p.strike),
      expiry: String(p.expiry_full || p.expiry || "").replace(/-/g, "").slice(0, 8),
      symbol: String(p.symbol || "").toUpperCase(), avg_cost: p.avg_cost == null ? null : Number(p.avg_cost),
      multiplier: Number(p.multiplier) || 100,
    })),
  };
}

/* Net price of closing legs from quote rows {bid, ask, mid}. Signed: debit +.
   natural = BUY at ask / SELL at bid. Returns {mid, nat, action} or nulls. */
function optNetQuote(legs, quoteFor) {
  let mid = 0, nat = 0, midOk = true, natOk = true;
  for (const l of legs) {
    const q = quoteFor(l) || {};
    const s = l.action === "BUY" ? 1 : -1;
    const m = q.mid != null ? Number(q.mid) : (q.bid != null && q.ask != null ? (Number(q.bid) + Number(q.ask)) / 2 : null);
    const n = l.action === "BUY" ? q.ask : q.bid;
    if (m == null || !isFinite(m)) midOk = false; else mid += s * l.ratio * m;
    if (n == null || !isFinite(Number(n))) natOk = false; else nat += s * l.ratio * Number(n);
  }
  const r = (v) => Math.round(v * 10000) / 10000;
  return { mid: midOk ? r(mid) : null, nat: natOk ? r(nat) : null };
}

/* Snap toward safe on the 0.05 grid: debit (BUY) down, credit (SELL) up. */
function optSnap(v, action, tick = 0.05) {
  const n = Number(v) / tick;
  const s = action === "SELL" ? Math.ceil(n - 1e-9) : Math.floor(n + 1e-9);
  return Math.round(s * tick * 100) / 100;
}

/* Realized P&L vs average cost if `units` close at signed net L (debit +).
   IBKR avg_cost for options is per contract (premium x multiplier), positive for
   longs (paid) and shorts (received). */
function optClosePnl(legs, units, signedNet) {
  if (legs.some((l) => l.avg_cost == null || !isFinite(l.avg_cost))) return null;
  const cost = legs.reduce((a, l) => a + Math.sign(l.held) * units * l.ratio * l.avg_cost, 0);
  return Math.round((-signedNet * 100 * units - cost) * 100) / 100;
}

/* ET regular-session test for the quote staleness rule. */
function optInRth(now = new Date()) {
  const parts = Object.fromEntries(new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York",
    weekday: "short", hour: "2-digit", minute: "2-digit", hour12: false }).formatToParts(now).map((p) => [p.type, p.value]));
  if (["Sat", "Sun"].includes(parts.weekday)) return false;
  const m = Number(parts.hour) % 24 * 60 + Number(parts.minute);
  return m >= 570 && m < 960;
}

/* Freshness of a workbench result: {label: "LIVE"|"FROZEN"|"DELAYED"|"STALE"|"UNKNOWN", ageS, stale, text}.
   market_data_type 1 live, 2 frozen, 3/4 delayed. asof is epoch seconds. */
function optQuoteFreshness(result, now = Date.now(), rth = optInRth(new Date(now))) {
  const mdt = Number(result && result.market_data_type);
  const asof = Number(result && result.asof);
  const ageS = isFinite(asof) && asof > 0 ? Math.max(0, Math.round(now / 1000 - (asof > 1e12 ? asof / 1000 : asof))) : null;
  let label = "LIVE";
  if (mdt === 2) label = "FROZEN";
  else if (mdt === 3 || mdt === 4) label = "DELAYED";
  else if (mdt !== 1) label = "UNKNOWN";
  if (label === "LIVE" && rth && ageS != null && ageS > OPT_QUOTE_STALE_S) label = "STALE";
  const stale = label !== "LIVE";
  const age = ageS == null ? "age unknown" : ageS < 90 ? `${ageS}s old` : `${Math.round(ageS / 60)}m old`;
  return { label, ageS, stale, text: `${label} quotes, ${age}` };
}

/* option_close payload (contract: OPTION_COMBO_SPEC.md section 6). */
function optBuildClosePayload({ symbol, legs, units, limit, action, tif }) {
  const snapped = optSnap(limit, action);
  if (!(snapped > 0)) return { error: "limit must be > 0 after snapping to the 0.05 grid" };
  if (!(units > 0) || !Number.isInteger(units)) return { error: "units must be a positive whole number" };
  const sides = new Set(legs.map((l) => l.action));
  if (sides.size === 1 && [...sides][0] !== action) {
    return { error: `every leg ${[...sides][0] === "SELL" ? "sells: a net CREDIT" : "buys: a net DEBIT"}; the price side is wrong` };
  }
  if (legs.length === 1 && legs[0].action !== action) return { error: "single-leg close: the order side must be the leg's closing side" };
  for (const l of legs) {
    if (units * l.ratio > Math.abs(l.held) + 1e-9) return { error: `closing ${units * l.ratio} exceeds held ${Math.abs(l.held)} (conId ${l.con_id})` };
  }
  return { payload: {
    symbol, action, quantity: units, limit: snapped, tif: tif || "DAY",
    legs: legs.map((l) => ({ con_id: l.con_id, action: l.action, ratio: l.ratio, right: l.right, strike: l.strike, expiry: l.expiry })),
  } };
}

/* Exactly what is sent, in words (confirm text). */
function optCloseConfirmText({ account, symbol, legs, units, payload, freshness, pnl }) {
  const legTxt = legs.map((l) => `${l.action} ${units * l.ratio} ${symbol} ${l.expiry} ${l.strike}${l.right} (conId ${l.con_id}, held ${l.held})`).join("\n  ");
  const kind = legs.length === 1 ? "one OPT limit order" : "one SMART BAG combo limit order (fills all legs together or not at all)";
  return [
    `CLOSE on ${account}: ${kind}`,
    `  ${legTxt}`,
    `Order: ${payload.action} ${payload.quantity}x LMT ${payload.limit.toFixed(2)} net ${payload.action === "BUY" ? "DEBIT (you pay)" : "CREDIT (you receive)"}, ${payload.tif}.`,
    pnl != null ? `P&L vs average cost if filled at this limit: about ${pnl >= 0 ? "+" : ""}$${pnl.toFixed(2)} (before commissions).` : "P&L vs average cost: unavailable (average cost missing).",
    freshness && freshness.stale ? `[WARN] Quotes are ${freshness.text}: the default limit came from that mid, check it against TWS.` : (freshness ? `Quotes: ${freshness.text}.` : ""),
    "The executor refuses if any leg would open or increase a position.",
  ].filter(Boolean).join("\n");
}

/* Roll deep link into the Options builder. */
function optRollHref(group, accountKey) {
  const cl = optCloseLegs(group.legs);
  const enc = cl.legs.map((l) => [l.con_id, l.action, l.ratio, l.right, l.strike, l.expiry, l.held].join(":")).join(",");
  const q = new URLSearchParams({ section: "custom", ticker: group.symbol, roll: enc, qty: String(cl.units), acct: accountKey });
  return `options.html?${q.toString()}`;
}

/* ---------------- quotes (existing /exec-workbench chain path) ---------------- */

async function optWorkbenchChain(ticker, expiry, center) {
  const body = { ticker, mode: "chain", expiry, max_expiries: 2, context: null };
  if (center > 0) body.strike_center = center;
  const r = await fetch("/exec-workbench", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const acc = await r.json();
  if (!acc.ok) throw new Error(acc.error || `HTTP ${r.status}`);
  for (let i = 0; i <= 45; i++) {
    const d = (await fetchJSONOrNull(`/exec-workbench?id=${encodeURIComponent(acc.id)}`)) || {};
    const q = d.query;
    if (q && q.id === acc.id && q.result) {
      if (q.result.error) throw new Error(q.result.error);
      return q.result;
    }
    await new Promise((res) => setTimeout(res, 2000));
  }
  throw new Error("timed out - is the agent online?");
}

async function optLoadFills() {
  if (OPT_POS.fills && Date.now() - OPT_POS.fillsAt < 60000) return OPT_POS.fills;
  const d = await fetchJSONOrNull("/exec-fills");
  OPT_POS.fills = (d && Array.isArray(d.fills)) ? d.fills : [];
  OPT_POS.fillsAt = Date.now();
  return OPT_POS.fills;
}

/* ---------------- Execution-tab integration ---------------- */

function optCurrentGroups() {
  const ab = acctBook();
  const refs = optOrderRefByConId(OPT_POS.fills || [], state.account);
  const positions = ((ab && ab.positions) || []).map((p) => ({ ...p, account_key: state.account }));
  return optGroupPositions(positions, state.account, refs, OPT_POS.picked);
}
function optGroupFor(conId) {
  return optCurrentGroups().find((g) => g.legs.some((l) => Number(l.con_id) === Number(conId))) || null;
}

/* Row actions for one OPT position row (called from execution.js renderPositions). */
function optPositionActions(p) {
  if (!OPT_POS.fills) optLoadFills().then(() => { if (typeof set === "function") set("positions", renderPositions()); });
  const book = state.book;
  const canClose = optSupports(book, "option_close"), canRoll = optSupports(book, "option_roll");
  const cid = Number(p.con_id) || 0;
  const g = optGroupFor(cid);
  const n = g ? g.legs.length : 1;
  const tag = g && n > 1 ? `<span class="cap" style="display:inline" title="${g.source === "manual" ? "grouped by your ticks" : "legs share account + underlying + orderRef"}">${n}-leg ${g.source === "manual" ? "group" : "structure"}</span> ` : "";
  const tick = `<label class="cap" style="display:inline" title="Tick legs of one underlying to close/roll them together"><input type="checkbox" ${OPT_POS.picked.has(cid) ? "checked" : ""} onchange="optTogglePick(${cid})"> group</label>`;
  if (!canClose && !canRoll) {
    return `${tag}<span class="cap" title="The running execution agent does not advertise option_close/option_roll">close/roll needs a newer agent - use TWS</span>`;
  }
  return `${tag}${tick}
    ${canClose ? `<button class="btn xs" data-mutation onclick="optCloseTicket(${cid})" title="Close this ${n > 1 ? "structure" : "leg"} with one limit order">Close&hellip;</button>` : ""}
    ${canRoll ? `<button class="btn xs ghost" onclick="optRollTicket(${cid})" title="Open the Options builder with these legs locked as the closing side">Roll&hellip;</button>` : ""}`;
}
function optTogglePick(cid) {
  if (OPT_POS.picked.has(cid)) OPT_POS.picked.delete(cid); else OPT_POS.picked.add(cid);
  set("positions", renderPositions());
}

function optRollTicket(cid) {
  const g = optGroupFor(cid);
  if (!g) return;
  window.location.href = optRollHref(g, state.account);
}

async function optCloseTicket(cid) {
  const g = optGroupFor(cid);
  const host = document.getElementById("optTicket");
  if (!g || !host) return;
  const cl = optCloseLegs(g.legs);
  const seq = ++OPT_POS.seq;
  OPT_POS.ticket = { group: g, close: cl, account: state.account, quotes: new Map(), freshness: null, units: cl.units,
    limit: null, limitTouched: false, tif: "DAY", error: null, loading: true };
  optRenderTicket();
  host.scrollIntoView && host.scrollIntoView({ behavior: "smooth", block: "nearest" });
  try {
    const expiries = [...new Set(cl.legs.map((l) => l.expiry))];
    let worst = null;
    for (const exp of expiries) {
      const ks = cl.legs.filter((l) => l.expiry === exp).map((l) => l.strike);
      const res = await optWorkbenchChain(g.symbol, exp, ks.reduce((a, b) => a + b, 0) / ks.length);
      if (seq !== OPT_POS.seq) return;
      for (const row of (res.chain && res.chain.strikes) || []) {
        if (row.con_id) OPT_POS.ticket.quotes.set(Number(row.con_id), row);
      }
      const f = optQuoteFreshness(res);
      if (!worst || (f.stale && !worst.stale) || (f.ageS || 0) > (worst.ageS || 0)) worst = f;
    }
    OPT_POS.ticket.freshness = worst;
  } catch (e) {
    if (seq === OPT_POS.seq) OPT_POS.ticket.error = String(e.message || e);
  }
  if (seq !== OPT_POS.seq) return;
  OPT_POS.ticket.loading = false;
  optRenderTicket();
}

function optTicketNumbers(t) {
  const legs = t.close.legs;
  const net = optNetQuote(legs, (l) => t.quotes.get(l.con_id));
  const sides = new Set(legs.map((l) => l.action));
  let action = sides.size === 1 ? [...sides][0] : (net.mid != null ? (net.mid >= 0 ? "BUY" : "SELL") : "BUY");
  if (legs.length === 1) action = legs[0].action;
  if (!t.limitTouched) t.limit = net.mid != null ? Math.max(0.05, optSnap(Math.abs(net.mid), action)) : null;
  const limit = Number(t.limit);
  const signed = action === "BUY" ? limit : -limit;
  const pnl = limit > 0 ? optClosePnl(legs, Number(t.units), signed) : null;
  return { net, action, limit, pnl };
}

function optRenderTicket() {
  const host = document.getElementById("optTicket");
  const t = OPT_POS.ticket;
  if (!host) return;
  if (!t) { host.innerHTML = ""; return; }
  const { net, action, pnl } = optTicketNumbers(t);
  const f = t.freshness;
  const badge = f ? `<span style="font-weight:700;color:${f.stale ? "#ff6b6b" : "#3ddb8f"}">${esc(f.label)}</span> <span class="cap" style="display:inline">${esc(f.text)}</span>` : "";
  const rows = t.close.legs.map((l) => {
    const q = t.quotes.get(l.con_id) || {};
    return `<tr><td class="l"><b>${l.action}</b> ${l.ratio > 1 ? l.ratio + "x " : ""}${esc(l.symbol)} ${esc(l.expiry)} ${l.strike}${esc(l.right)}</td>
      <td>held ${fmt.num(l.held, 0)}</td><td>avg ${l.avg_cost != null ? fmt.num(l.avg_cost / (l.multiplier || 100), 2) : "-"}</td>
      <td>${q.bid != null ? fmt.num(q.bid, 2) : "-"} / ${q.ask != null ? fmt.num(q.ask, 2) : "-"}</td></tr>`;
  }).join("");
  const effect = action === "BUY" ? "debit" : "credit";
  host.innerHTML = `<div class="card" style="margin:10px 0">
    <div style="font:700 14px inherit;margin-bottom:6px">Close ${t.close.legs.length > 1 ? "structure" : "option"} &mdash; ${esc(t.group.symbol)} (${esc(t.account)})
      <button class="btn xs ghost" onclick="optCloseTicketDismiss()" style="margin-left:8px">dismiss</button></div>
    ${t.loading ? '<div class="cap">fetching quotes via the workbench (~10-20s)...</div>' : ""}
    ${t.error ? `<div class="cap neg">quotes unavailable: ${esc(t.error)} (you can still enter a limit)</div>` : ""}
    <div class="tblwrap"><table class="tbl"><tbody>${rows}</tbody></table></div>
    <div class="kv" style="margin-top:8px">
      <div class="k">Net mid / natural</div><div class="v">${net.mid != null ? fmt.num(Math.abs(net.mid), 2) : "-"} / ${net.nat != null ? fmt.num(Math.abs(net.nat), 2) : "-"} ${effect} per unit ${badge}</div>
      <div class="k">P&amp;L vs avg cost at limit</div><div class="v ${pnl != null ? (pnl >= 0 ? "pos" : "neg") : ""}">${pnl != null ? fmt.money(pnl) : "-"} <span class="cap" style="display:inline">before commissions</span></div>
    </div>
    <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:8px">
      <label class="cap">Units</label><input id="optc_units" value="${esc(t.units)}" style="width:60px" inputmode="numeric">
      <span class="cap">of ${t.close.units} held</span>
      <label class="cap">${effect === "debit" ? "Debit" : "Credit"} limit</label><input id="optc_limit" value="${t.limit != null ? Number(t.limit).toFixed(2) : ""}" style="width:76px">
      <label class="cap">TIF</label><select id="optc_tif"><option${t.tif === "DAY" ? " selected" : ""}>DAY</option><option${t.tif === "GTC" ? " selected" : ""}>GTC</option></select>
      <button class="btn" data-mutation id="optc_send">${execMode() === "dry-run" ? "Preview / dry-run" : "Send close"}</button>
      <span id="optc_msg" class="cap"></span>
    </div>
    <div class="cap" style="margin-top:6px">${t.close.legs.length === 1 ? "One SMART option limit order." : "One SMART BAG limit order: all legs fill together or not at all."} The executor re-reads the live positions and refuses anything that would open or increase a leg.</div>
  </div>`;
  const $ = (id) => document.getElementById(id);
  $("optc_units").addEventListener("change", (e) => { t.units = Number(e.target.value); optRenderTicket(); });
  $("optc_limit").addEventListener("change", (e) => { t.limitTouched = true; t.limit = e.target.value; optRenderTicket(); });
  $("optc_tif").addEventListener("change", (e) => { t.tif = e.target.value; });
  $("optc_send").addEventListener("click", optSendClose);
}
function optCloseTicketDismiss() { OPT_POS.ticket = null; OPT_POS.seq++; optRenderTicket(); }

function optSendClose() {
  const t = OPT_POS.ticket;
  const msg = document.getElementById("optc_msg");
  if (!t) return;
  if (!optSupports(state.book, "option_close")) { if (msg) msg.textContent = "the running agent does not support option_close"; return; }
  const { action, limit, pnl } = optTicketNumbers(t);
  const built = optBuildClosePayload({ symbol: t.group.symbol, legs: t.close.legs, units: Number(t.units), limit, action, tif: t.tif });
  if (built.error) { if (msg) msg.textContent = "BLOCKED: " + built.error; return; }
  const text = optCloseConfirmText({ account: t.account, symbol: t.group.symbol, legs: t.close.legs, units: Number(t.units),
    payload: built.payload, freshness: t.freshness, pnl });
  if (!confirm(`${actionLead("close options")}\n\n${text}`)) return;
  sendCommand("option_close", built.payload, "optc_msg", { account: t.account });
}

window.optPositionActions = optPositionActions;
window.optTogglePick = optTogglePick;
window.optCloseTicket = optCloseTicket;
window.optRollTicket = optRollTicket;
window.optCloseTicketDismiss = optCloseTicketDismiss;
