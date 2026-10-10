/* options-custom.js - Custom spread builder for the Options tab.

   Any ticker, any listed expiry (0DTE and weeklies included), any strike, any
   quantity: pick legs off the live ladder, see the expiry payoff, set a limit
   and stage through the SAME signed /exec-command option_spread path as the
   shootout ticket (buildOptionSpreadPayload in options.js is the single payload
   builder, so the contract the desktop executor validates is unchanged).

   There is NO client-side quantity or risk cap here. The only gates are the
   desktop executor's. The structures it accepts (OPTION_COMBO_SPEC.md: same-expiry
   1-4 legs with integer ratios, and covered debit calendars/diagonals) are mirrored
   by OC_EXEC_CAPS plus comboCheckStructure/comboEvaluate in options.js, which port
   the executor's structure rules and risk math. Anything the spec rejects is
   refused here with the reason.

   Loaded after options.js; shares its globals (state, esc, fmt, snapNetLimit,
   canonicalPayloadLegs, sendCommand, execMode, commRT, ...). */
"use strict";

/* What the desktop executor accepts (option_combo_risk.py / OPTION_COMBO_SPEC.md).
   Structure rules themselves live in comboCheckStructure (options.js). */
const OC_EXEC_CAPS = { maxLegs: 4, allowRatio: true, allowShortSingle: true, allowCalendar: true };
const OC_MAX_LEGS = 4;

const ocState = {
  ticker: null, result: null, expiries: [], chain: null, spot: null, expiry: null,
  legs: [], qty: 1, limit: null, limitTouched: false, tif: "DAY",
  center: null, centerRequested: null, note: "", loading: false, seq: 0,
  prefill: null, recentredOnce: false, netMode: null,
  fresh: null,          // quote freshness of the last workbench result (optQuoteFreshness)
  blocked: null,        // blocking banner: nothing can be sent while set
  cols: null,           // ladder columns (ocLoadCols)
  roll: null,           // roll mode (options-roll.js): closing legs locked
};

/* Ladder columns (TWS-style): default bid, ask, delta, IV, OI. bid/ask cells are
   the click targets (click bid = SELL, click ask = BUY). */
const OC_COLUMNS = {
  bid: { label: "bid", d: 2 }, ask: { label: "ask", d: 2 }, mid: { label: "mid", d: 2 },
  last: { label: "last", d: 2 }, iv: { label: "IV", pct: true }, delta: { label: "delta", d: 2 },
  gamma: { label: "gamma", d: 4 }, theta: { label: "theta", d: 3 }, vega: { label: "vega", d: 3 },
  oi: { label: "OI", int: true }, volume: { label: "vol", int: true },
};
const OC_DEFAULT_COLS = ["bid", "ask", "delta", "iv", "oi"];
function ocNormCols(cols) {
  const want = (Array.isArray(cols) ? cols : []).filter((c) => OC_COLUMNS[c]);
  for (const c of ["bid", "ask"]) if (!want.includes(c)) want.unshift(c);   // the click targets never disappear
  return Object.keys(OC_COLUMNS).filter((c) => want.includes(c));
}
function ocLoadCols() {
  try { const v = JSON.parse(localStorage.getItem("oc_cols") || "null"); if (Array.isArray(v)) return ocNormCols(v); } catch (_) { /* default */ }
  return OC_DEFAULT_COLS.slice();
}
function ocSaveCols(cols) { try { localStorage.setItem("oc_cols", JSON.stringify(cols)); } catch (_) { /* per-viewer only */ } }

/* Quote freshness; option-positions.js holds the shared rule when loaded. */
function ocFreshness(res) {
  return typeof optQuoteFreshness === "function" ? optQuoteFreshness(res) : null;
}
const OC_NO_EXITS_TEXT = "No automatic stop, target or time exit is attached. You must close or roll this position yourself: Execution tab > the position row > Close... / Roll... (execution.html#positions).";
function ocStaleText(fresh) {
  return fresh && fresh.stale ? `\n\n[WARN] Quotes are ${fresh.text}: the default limit was taken from that mid and may not be tradeable. Check the price against TWS.` : "";
}

/* ---------------- pure helpers (unit-tested) ---------------- */

function ocNormExpiry(s) {
  const d = String(s == null ? "" : s).replace(/-/g, "").trim();
  return /^\d{8}$/.test(d) ? d : null;
}
function ocIsoExpiry(s) {
  const d = ocNormExpiry(s);
  return d ? `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}` : String(s || "");
}

/* "P:748:2026-11-30:+1,P:720:2026-11-30:-1" -> legs. A bare "+" in a query
   string decodes to a space, so a leading blank counts as plus. */
function ocParseLegs(str) {
  const out = [];
  if (!str) return out;
  for (const part of String(str).split(",")) {
    const f = part.split(":");
    if (f.length !== 4) continue;
    const right = f[0].trim().toUpperCase();
    const strike = Number(f[1]);
    const expiry = ocNormExpiry(f[2]);
    const q = f[3].replace(/%2B/gi, "+");
    const signed = Number(q.trim().replace(/^\+/, ""));
    if (!["C", "P"].includes(right) || !(strike > 0) || !expiry || !Number.isInteger(signed) || signed === 0) continue;
    out.push({ right, strike, expiry, side: signed < 0 ? "SELL" : "BUY", ratio: Math.abs(signed) });
  }
  return out;
}

/* Inverse of ocParseLegs; legs [{right, strike, expiry, qty (signed)}] or
   [{right, strike, expiry, side, ratio}]. Plus is percent-encoded. */
function ocLegsParam(legs) {
  return (legs || []).map((l) => {
    const signed = l.qty != null ? Number(l.qty)
      : (String(l.side).toUpperCase() === "SELL" ? -1 : 1) * Number(l.ratio || 1);
    const q = signed < 0 ? String(signed) : "%2B" + signed;
    return `${String(l.right).toUpperCase()}:${l.strike}:${ocIsoExpiry(l.expiry)}:${q}`;
  }).join(",");
}

/* Expiry payoff of a leg set. net = signed premium PER SHARE (debit positive,
   credit negative). Returns per-share numbers; multiply by 100 for dollars.
   P&L is piecewise linear with kinks at the strikes, so extremes sit at 0 and
   the strikes; the upper tail is governed by the net call slope. */
function ocPayoff(legs, net) {
  const L = (legs || []).map((l) => ({
    s: String(l.side).toUpperCase() === "SELL" ? -1 : 1, r: Number(l.ratio || 1),
    right: l.right, k: Number(l.strike),
  }));
  if (!L.length) return null;
  const value = (S) => L.reduce((a, l) =>
    a + l.s * l.r * (l.right === "C" ? Math.max(0, S - l.k) : Math.max(0, l.k - S)), 0);
  const pnl = (S) => value(S) - net;
  const pts = [0, ...new Set(L.map((l) => l.k))].sort((a, b) => a - b);
  const vals = pts.map(pnl);
  const tailSlope = L.reduce((a, l) => a + (l.right === "C" ? l.s * l.r : 0), 0);
  const lossUnbounded = tailSlope < 0, gainUnbounded = tailSlope > 0;
  const be = [];
  const push = (x) => { if (isFinite(x) && x >= 0 && !be.some((b) => Math.abs(b - x) < 1e-9)) be.push(x); };
  for (let i = 0; i < pts.length; i++) {
    if (Math.abs(vals[i]) < 1e-9) push(pts[i]);
    if (i + 1 < pts.length && vals[i] * vals[i + 1] < 0) {
      push(pts[i] + (0 - vals[i]) * (pts[i + 1] - pts[i]) / (vals[i + 1] - vals[i]));
    }
  }
  const last = pts.length - 1;
  if (tailSlope !== 0 && vals[last] * tailSlope < 0) push(pts[last] - vals[last] / tailSlope);
  const r6 = (x) => Math.round(x * 1e6) / 1e6;
  return {
    maxLoss: lossUnbounded ? null : r6(-Math.min(...vals)),
    maxGain: gainUnbounded ? null : r6(Math.max(...vals)),
    lossUnbounded, gainUnbounded, tailSlope,
    breakevens: be.sort((a, b) => a - b).map(r6),
  };
}

/* Can the executor place these resolved legs [{side,row,ratio}]? Returns a reason
   or null. `credit` = the structure is entered as a SELL parent (legs flipped). */
function ocExecIssue(legs, expiry, caps, credit) {
  const c = caps || OC_EXEC_CAPS;
  if (!legs || !legs.length) return "Add at least one leg";
  if (legs.some((l) => !l.row)) return "A selected strike is not in the returned chain; recentre or re-quote";
  if (legs.some((l) => !l.row.con_id)) return "Contract qualification required (no conId on a leg)";
  if (legs.length > c.maxLegs) return `Rejected by the executor: more than ${c.maxLegs} legs (${legs.length} selected)`;
  if (!c.allowRatio && legs.some((l) => Number(l.ratio || 1) !== 1)) return "Rejected: ratios are not enabled";
  const action = credit ? "SELL" : "BUY";
  const wire = legs.map((l) => ({
    side: credit ? (l.side === "BUY" ? "SELL" : "BUY") : l.side,
    right: l.row.right, expiry: String(l.row.expiry || expiry).replace(/-/g, ""),
    strike: l.row.strike, ratio: l.ratio || 1,
  }));
  const chk = comboCheckStructure(action, wire);
  if (chk.error) return `Rejected by the executor: ${chk.error}`;
  if (chk.shape === "calendar" && !c.allowCalendar) return "Rejected: calendars are not enabled";
  return null;
}

function ocFindRow(chain, right, strike) {
  const rows = (chain && chain.strikes) || [];
  return rows.find((r) => r.right === right && Math.abs(Number(r.strike) - Number(strike)) < 1e-9) || null;
}
function ocMid(row) {
  if (!row) return null;
  if (row.mid != null) return Number(row.mid);
  return row.bid != null && row.ask != null ? (Number(row.bid) + Number(row.ask)) / 2 : null;
}

/* Row for a leg: legs carry their own expiry (and a row snapshot taken when the
   leg was added) so a calendar can span two expiries although the ladder shows one. */
function ocResolveRow(l, chain) {
  const e = l.expiry ? ocNormExpiry(l.expiry) : null;
  const ce = ocNormExpiry(chain.expiry);
  if (e && e !== ce) return l.row ? { ...l.row, expiry: e } : null;
  const r = ocFindRow(chain, l.right, l.strike);
  return r && e ? { ...r, expiry: e } : r;
}

/* Struct in the same shape structureFrom emits, so canonicalPayloadLegs,
   commRT and buildOptionSpreadPayload treat it exactly like a shootout row.
   netMode "debit"/"credit" overrides the sign of the mid (default: from the mid). */
function ocBuildStruct(legs, chain, netMode) {
  if (!legs || !legs.length || !chain) return null;
  const rl = legs.map((l) => ({ side: l.side, ratio: l.ratio || 1, row: ocResolveRow(l, chain), want: l }));
  let signedMid = 0, signedNat = 0, midOk = true, natOk = true;
  for (const l of rl) {
    if (!l.row) { midOk = false; natOk = false; continue; }
    const s = l.side === "BUY" ? 1 : -1;
    const m = ocMid(l.row);
    const n = l.side === "BUY" ? l.row.ask : l.row.bid;
    if (m == null) midOk = false; else signedMid += s * l.ratio * m;
    if (n == null) natOk = false; else signedNat += s * l.ratio * Number(n);
  }
  const credit = netMode === "credit" ? true : netMode === "debit" ? false : midOk && signedMid < 0;
  const sameExp = rl.every((l) => l.row) && new Set(rl.map((l) => l.row.expiry || chain.expiry)).size === 1;
  const vertical = rl.length === 2 && sameExp && rl[0].row.right === rl[1].row.right &&
    rl[0].side !== rl[1].side && rl[0].row.strike !== rl[1].row.strike &&
    rl[0].ratio === 1 && rl[1].ratio === 1;
  const width = vertical ? Math.abs(rl[0].row.strike - rl[1].row.strike) : null;
  const rnd = (v) => Math.round(v * 100) / 100;
  const struct = {
    name: "Custom", legs: rl.map(({ side, ratio, row }) => ({ side, ratio, row })),
    mid: midOk ? rnd(Math.abs(signedMid)) : null,
    nat: natOk ? rnd(credit ? -signedNat : signedNat) : null,
    credit, width, category: "custom", note: "",
  };
  struct.execution_issue = ocExecIssue(rl.map(({ side, ratio, row }) => ({ side, ratio, row })), chain.expiry, null, credit);
  struct.tradeable = !struct.execution_issue;
  struct.missing = rl.filter((l) => !l.row).map((l) => l.want);
  return struct;
}

/* Conservative underlying for the unbounded stress: the max of every live
   underlying figure we hold (mid, last, chain spot). Overstating is allowed by
   the executor for claims; understating is not. */
function ocSpotUsed(result, chain, spot) {
  const cands = [spot, result && result.spot, result && result.last, result && result.mid,
    result && result.underlying_last, result && result.underlying_mid,
    chain && chain.spot, chain && chain.last, chain && chain.underlying_price]
    .map(Number).filter((x) => Number.isFinite(x) && x > 0);
  return cands.length ? Math.max(...cands) : null;
}

function ocExpiryList(result) {
  if (!result) return [];
  const all = Array.isArray(result.all_expiries) && result.all_expiries.length ? result.all_expiries : null;
  const src = all || (result.expiries || []);
  return src.map((e) => ({ date: ocNormExpiry(e.expiry || e.date), dte: e.dte }))
    .filter((e) => e.date).sort((a, b) => a.date.localeCompare(b.date));
}

/* True when a requested strike_center falls inside the returned strike span. */
function ocCenterHonored(chain, center) {
  if (center == null) return true;
  const ks = ((chain && chain.strikes) || []).map((r) => Number(r.strike)).filter((x) => isFinite(x));
  return ks.length > 0 && Math.min(...ks) <= center && center <= Math.max(...ks);
}

function ocDefaultLimit(struct) {
  if (!struct || struct.mid == null) return null;
  const action = struct.credit ? "SELL" : "BUY";
  return snapNetLimit(Math.max(0.05, struct.mid + (struct.credit ? 0.01 : -0.01)), action);
}

/* Add or merge a leg: same contract (right, strike, expiry) + same side bumps the
   ratio, opposite side flips it. expiry/row are optional (row = quote snapshot so a
   leg on another expiry than the ladder still prices). False at the leg limit. */
function ocAddLeg(legs, right, strike, side, expiry, row) {
  const ex = legs.find((l) => l.right === right && l.strike === strike && (l.expiry || null) === (expiry || null));
  if (ex) {
    if (ex.side === side) ex.ratio += 1; else { ex.side = side; ex.ratio = 1; }
    return true;
  }
  if (legs.length >= OC_MAX_LEGS) return false;
  const leg = { right, strike, side, ratio: 1 };
  if (expiry) { leg.expiry = expiry; leg.row = row || null; }
  legs.push(leg);
  return true;
}

/* ---------------- query ---------------- */

async function ocRequest(body) {
  const r = await fetch("/exec-workbench", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body) });
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

/* Pin every leg to the expiry it was picked on, with a quote snapshot, so the
   ladder can move to another expiry without losing it. */
function ocFreezeLegs() {
  if (!ocState.chain) return;
  for (const l of ocState.legs) {
    if (l.expiry) continue;
    const row = ocFindRow(ocState.chain, l.right, l.strike);
    l.expiry = ocNormExpiry(ocState.chain.expiry);
    l.row = row ? { ...row, expiry: l.expiry } : null;
  }
}

function ocSetMsg(t) { const m = document.getElementById("ocMsg"); if (m) m.textContent = t || ""; }

async function ocFetch(mode, expiry, center) {
  const seq = ++ocState.seq;
  ocState.loading = true;
  ocSetMsg(mode === "full" ? "fetching chain + expiries... (~15-20s)" : "re-quoting...");
  const body = { ticker: ocState.ticker, mode, expiry: expiry || null, max_expiries: mode === "full" ? 40 : 2, context: null };
  // Always the contiguous strike window around a centre (default: spot), never
  // the agent's thinned wide band, so no nearby strike is skipped and the
  // tails are ignored. "full" runs before spot is known; ocLoadTicker
  // re-quotes centred on spot right after it.
  if (center == null && mode === "chain" && ocState.spot > 0) center = Math.round(ocState.spot);
  if (center != null) body.strike_center = center;
  try {
    const res = await ocRequest(body);
    if (seq !== ocState.seq) return false;
    if (mode === "full") ocState.result = res;
    if (res.chain) {
      ocState.chain = res.chain; ocState.spot = res.spot; ocState.expiry = ocNormExpiry(res.chain.expiry);
      ocState.fresh = ocFreshness(res);
    } else {
      ocSetMsg(res.chain_error || "no chain returned for that expiry");
      ocState.loading = false; renderCustom(); return false;
    }
    const exps = ocExpiryList(res);
    if (exps.length) ocState.expiries = exps;
    ocState.centerRequested = center == null ? null : center;
    ocState.note = center != null && !ocCenterHonored(ocState.chain, center)
      ? `Requested strike centre ${center} is outside the returned strikes: the desktop agent returned its default band (strike_center is not supported by the running agent yet).` : "";
    ocState.loading = false;
    ocSetMsg("");
    renderCustom();
    return true;
  } catch (e) {
    if (seq === ocState.seq) { ocState.loading = false; ocSetMsg("error: " + (e.message || e)); }
    return false;
  }
}

async function ocLoadTicker(ticker) {
  const t = String(ticker || "").toUpperCase().trim();
  if (!t) return;
  if (t !== ocState.ticker) {
    ocState.ticker = t; ocState.expiries = []; ocState.chain = null; ocState.legs = [];
    ocState.limitTouched = false; ocState.limit = null; ocState.recentredOnce = false;
    if (!ocState.prefill) ocState.blocked = null;
    if (ocState.roll && ocState.roll.symbol !== t) ocState.roll = null;     // roll mode is bound to its underlying
  }
  const ok = await ocFetch("full", null, null);
  if (!ok) return;
  const pf = ocState.prefill;
  if (!(pf && pf.legs.length)) {
    await ocFetch("chain", ocState.expiry, null);          // centred on spot (see ocFetch)
    return;
  }
  if (pf && pf.legs.length) {
    ocState.prefill = null;
    const plan = ocPrefillPlan(pf.legs);
    if (plan.error) {
      // Never drop a leg silently: a calendar missing its long leg is a naked short.
      ocState.legs = [];
      ocState.blocked = `Prefill refused: ${plan.error}. Nothing was loaded; build the structure from the ladder.`;
      renderCustom();
      return;
    }
    ocState.blocked = null;
    if (pf.qty) ocState.qty = pf.qty;
    if (pf.limit) { ocState.limit = pf.limit; ocState.limitTouched = true; }
    // One chain per expiry, centred on that expiry's strikes; every leg is pinned
    // to its own expiry with a quote snapshot. A leg missing from the returned
    // strikes stays listed (shown as missing, blocks sending); none is dropped.
    const legs = [];
    for (const exp of plan.expiries) {
      const mine = pf.legs.filter((l) => l.expiry === exp);
      await ocFetch("chain", exp, mine.reduce((a, l) => a + l.strike, 0) / mine.length);
      const here = ocState.chain && ocNormExpiry(ocState.chain.expiry) === exp;
      for (const l of mine) {
        const row = here ? ocFindRow(ocState.chain, l.right, l.strike) : null;
        legs.push({ right: l.right, strike: l.strike, side: l.side, ratio: l.ratio, expiry: exp, row: row ? { ...row, expiry: exp } : null });
      }
    }
    ocState.legs = legs;
    ocState.note = plan.expiries.length > 1
      ? `Calendar/diagonal prefill: legs on ${plan.expiries.map(ocIsoExpiry).join(" and ")} (the ladder shows the last expiry).` : "";
    renderCustom();
  }
}

/* Prefill legs -> {expiries} or {error}: at most 4 legs on at most 2 expiries,
   and two expiries only as the covered 2-leg debit calendar/diagonal the
   executor accepts (OPTION_COMBO_SPEC.md). Anything else is refused whole,
   never trimmed to the first expiry. */
function ocPrefillPlan(legs) {
  if (!legs || !legs.length) return { error: "no legs" };
  if (legs.length > OC_MAX_LEGS) return { error: `${legs.length} legs (the executor accepts at most ${OC_MAX_LEGS})` };
  const expiries = [...new Set(legs.map((l) => l.expiry))].sort();
  if (expiries.length > 2) return { error: `legs span ${expiries.length} expiries (at most 2: a covered calendar/diagonal)` };
  if (expiries.length === 2) {
    const chk = comboCheckStructure("BUY", legs.map((l) => ({ side: l.side, right: l.right, expiry: l.expiry, strike: l.strike, ratio: l.ratio })));
    if (chk.error) return { error: `the two-expiry legs are not an accepted calendar/diagonal (${chk.error})` };
  }
  return { expiries };
}

/* ---------------- render ---------------- */

const ocNum = (v, d) => (v == null || !isFinite(Number(v)) ? "-" : Number(v).toFixed(d == null ? 2 : d));

function ocShell() {
  return `<div class="card" style="margin-bottom:12px">
    <div style="font:700 14px inherit;margin-bottom:6px">Custom spread
      <span class="cap" style="display:inline;font-weight:400">- any ticker, any listed expiry, any strikes, up to 4 legs; no size cap here (the execution agent is the only gate)</span></div>
    <div style="display:flex;gap:10px;align-items:end;flex-wrap:wrap">
      <label><span class="cap">Ticker</span><br><input id="ocTicker" placeholder="SPY" style="text-transform:uppercase;width:90px"></label>
      <button class="btn" id="ocGo">Load chain</button>
      <label><span class="cap">Expiry</span><br><select id="ocExpiry" style="min-width:190px"><option value="">load a ticker first</option></select></label>
      <label><span class="cap">Strike centre</span><br><input id="ocCenter" placeholder="spot" style="width:80px"></label>
      <button class="btn ghost" id="ocRecentre">Recentre</button>
      <span id="ocMsg" class="cap"></span>
    </div>
    <div id="ocNote" class="cap" style="margin-top:6px;color:#ffc14d"></div>
    <div id="ocBlocked" style="margin-top:6px"></div>
  </div>
  <div id="ocRoll"></div>
  <div id="ocLadder"></div>
  <div id="ocLegs"></div>
  <div class="card" style="margin-bottom:12px" id="ocOrderCard">
    <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px">
      <label class="cap">Qty (spreads)</label><input id="oc_qty" value="${ocState.qty}" style="width:80px" inputmode="numeric">
      <label class="cap" id="oc_limit_label">Limit (per spread)</label><input id="oc_limit" value="" style="width:80px">
      <label class="cap">TIF</label><select id="oc_tif"><option>DAY</option><option>GTC</option></select>
      <label class="cap">Net</label><select id="oc_net"><option value="">auto (from mid)</option><option value="debit">debit</option><option value="credit">credit</option></select>
      <span class="cap">Account: <b>primary</b> (PA options remain disabled)</span>
    </div>
    <div class="cap" style="margin-bottom:8px">No automatic stop, target or time exit is attached to an option order: close or roll it yourself from the <a href="execution.html#positions">Execution tab</a> (position row: Close&hellip; / Roll&hellip;). Net-short-call structures are UNBOUNDED and need an extra confirmation.</div>
    <div id="ocStats"></div>
    <button class="btn" id="oc_send" style="margin-top:8px">Preview / stage</button>
    <span id="oc_msg" class="cap" style="margin-left:10px"></span>
  </div>
  <div id="customActivity" style="margin-top:18px"></div>`;
}

function ocWireShell() {
  const $ = (id) => document.getElementById(id);
  $("ocGo").addEventListener("click", () => ocLoadTicker($("ocTicker").value));
  $("ocTicker").addEventListener("keydown", (e) => { if (e.key === "Enter") ocLoadTicker($("ocTicker").value); });
  $("ocExpiry").addEventListener("change", (e) => {
    if (!e.target.value) return;
    ocFreezeLegs();                                         // legs keep their own expiry (calendars)
    ocFetch("chain", e.target.value, ocState.centerRequested);
  });
  $("ocRecentre").addEventListener("click", () => {
    const raw = $("ocCenter").value.trim();
    const c = raw === "" ? null : Number(raw);
    if (c != null && !(c > 0)) { ocSetMsg("strike centre must be a positive number"); return; }
    ocFetch("chain", ocState.expiry, c);
  });
  $("oc_qty").addEventListener("input", () => { ocState.qty = $("oc_qty").value; renderCustomStats(); });
  $("oc_limit").addEventListener("input", () => { ocState.limitTouched = true; ocState.limit = $("oc_limit").value; renderCustomStats(); });
  $("oc_tif").addEventListener("change", () => { ocState.tif = $("oc_tif").value; });
  $("oc_net").addEventListener("change", () => { ocState.netMode = $("oc_net").value || null; ocState.limitTouched = false; renderCustom(); });
  $("oc_send").addEventListener("click", sendCustomOrder);
  $("ocLadder").addEventListener("change", (e) => {
    const c = e.target.dataset && e.target.dataset.ocCol;
    if (!c) return;
    const cur = new Set(ocState.cols || ocLoadCols());
    if (e.target.checked) cur.add(c); else cur.delete(c);
    ocState.cols = ocNormCols([...cur]);
    ocSaveCols(ocState.cols);
    renderCustomLadder();
  });
  $("ocLadder").addEventListener("click", (e) => {
    const b = e.target.closest("[data-oc-add]");
    if (!b) return;
    const [right, strike, side] = b.dataset.ocAdd.split("|");
    ocFreezeLegs();
    const k = Number(strike), ex = ocNormExpiry(ocState.chain.expiry);
    const row = ocFindRow(ocState.chain, right, k);
    if (!ocAddLeg(ocState.legs, right, k, side, ex, row ? { ...row, expiry: ex } : null)) ocSetMsg(`at most ${OC_MAX_LEGS} legs`);
    ocState.limitTouched = false;
    renderCustom();
  });
  $("ocLegs").addEventListener("change", (e) => {
    const i = Number(e.target.dataset.ocRatio);
    if (e.target.dataset.ocRatio != null) {
      const v = Number(e.target.value);
      if (Number.isInteger(v) && v >= 1) ocState.legs[i].ratio = v;
      ocState.limitTouched = false; renderCustom();
    }
  });
  $("ocLegs").addEventListener("click", (e) => {
    if (e.target.closest("[data-oc-clear]")) { ocState.legs = []; ocState.limitTouched = false; renderCustom(); return; }
    const b = e.target.closest("[data-oc-del]");
    if (!b) return;
    ocState.legs.splice(Number(b.dataset.ocDel), 1);
    ocState.limitTouched = false; renderCustom();
  });
}

/* Pure ladder HTML: calls | strike | puts with the selected columns, the ATM
   row highlighted, bid cells = SELL buttons and ask cells = BUY buttons. A
   value IBKR did not supply renders as "-". */
function ocLadderHtml({ chain, spot, legs, cols, ticker, fresh }) {
  if (!chain) return "";
  const columns = ocNormCols(cols || OC_DEFAULT_COLS);
  const by = new Map();
  for (const r of chain.strikes || []) {
    const k = Number(r.strike);
    if (!by.has(k)) by.set(k, {});
    by.get(k)[r.right] = r;
  }
  const strikes = [...by.keys()].sort((a, b) => a - b);
  const sp = Number(spot);
  let atm = null;
  strikes.forEach((k) => { if (atm == null || Math.abs(k - sp) < Math.abs(atm - sp)) atm = k; });
  const curExp = ocNormExpiry(chain.expiry);
  const sel = (right, k, side) => (legs || []).some((l) => l.right === right && l.strike === k && l.side === side && (!l.expiry || l.expiry === curExp));
  const fmtv = (c, v) => {
    if (v == null || v === "" || !isFinite(Number(v))) return "-";
    const d = OC_COLUMNS[c];
    if (d.pct) return (Number(v) * 100).toFixed(1);
    if (d.int) return String(Math.round(Number(v)));
    return Number(v).toFixed(d.d);
  };
  const cell = (r, right, k, c) => {
    const v = c === "mid" ? ocMid(r) : r[c];
    if ((c === "bid" || c === "ask") && v != null) {
      const side = c === "bid" ? "SELL" : "BUY";
      const on = sel(right, k, side);
      return `<td class="r"><button class="btn xs${on ? "" : " ghost"}" data-oc-add="${right}|${k}|${side}" title="${side === "BUY" ? "buy at the ask" : "sell at the bid"}">${fmtv(c, v)}</button></td>`;
    }
    return `<td class="r">${fmtv(c, v)}</td>`;
  };
  const half = (right, k) => {
    const r = (by.get(k) || {})[right];
    if (!r) return `<td class="r" colspan="${columns.length}">-</td>`;
    return columns.map((c) => cell(r, right, k, c)).join("");
  };
  const head = columns.map((c) => `<th class="r">${OC_COLUMNS[c].label}</th>`).join("");
  const rows = strikes.map((k) => `<tr${k === atm ? ' class="oc-atm" style="background:rgba(77,163,255,.18);font-weight:600"' : ""}>${half("C", k)}<td class="c"><b>${k}</b>${k === atm ? ' <span class="cap" style="display:inline">ATM</span>' : ""}</td>${half("P", k)}</tr>`).join("");
  const badge = fresh ? ` <span style="font-weight:700;color:${fresh.stale ? "#ff6b6b" : "#3ddb8f"}">${esc(fresh.label)}</span> <span class="cap" style="display:inline">${esc(fresh.text)}</span>` : "";
  const picks = Object.keys(OC_COLUMNS).filter((c) => c !== "bid" && c !== "ask").map((c) =>
    `<label class="cap" style="display:inline;margin-right:8px"><input type="checkbox" data-oc-col="${c}"${columns.includes(c) ? " checked" : ""}> ${OC_COLUMNS[c].label}</label>`).join("");
  return `<div class="card" style="margin-bottom:12px">
    <div style="font:700 14px inherit;margin-bottom:6px">Strike ladder - ${esc(ticker || "")} ${esc(ocIsoExpiry(chain.expiry))}${badge}
      <span class="cap" style="display:inline;font-weight:400">- spot ${ocNum(spot)} - ${chain.dte != null ? chain.dte + "d" : ""} - click a bid to SELL, an ask to BUY (calls left, puts right)</span></div>
    <div style="margin-bottom:6px"><span class="cap" style="display:inline;margin-right:8px">Columns:</span>${picks}</div>
    <div class="tblwrap" style="max-height:420px;overflow:auto"><table class="tbl"><thead>
      <tr><th colspan="${columns.length}" class="c">CALLS</th><th></th><th colspan="${columns.length}" class="c">PUTS</th></tr>
      <tr>${head}<th class="c">strike</th>${head}</tr></thead><tbody>${rows}</tbody></table></div></div>`;
}

function renderCustomLadder() {
  const el = document.getElementById("ocLadder");
  if (!el) return;
  if (!ocState.cols) ocState.cols = ocLoadCols();
  el.innerHTML = ocLadderHtml({ chain: ocState.chain, spot: ocState.spot, legs: ocState.legs, cols: ocState.cols,
    ticker: ocState.ticker, fresh: ocState.fresh });
}

function renderCustomLegs() {
  const el = document.getElementById("ocLegs");
  if (!el) return;
  if (!ocState.legs.length) {
    el.innerHTML = '<div class="card" style="margin-bottom:12px"><span class="cap">No legs yet. Click a bid or ask in the ladder (up to 4 legs; one expiry, or two expiries for a covered debit calendar/diagonal - switch the expiry between clicks).</span></div>';
    return;
  }
  const rows = ocState.legs.map((l, i) => {
    const row = ocResolveRow(l, ocState.chain);
    return `<tr><td class="l"><b>${l.side}</b> ${l.right === "C" ? "Call" : "Put"} ${l.strike} ${esc(ocIsoExpiry(l.expiry || ocState.expiry))}</td>
      <td class="r">ratio <input data-oc-ratio="${i}" value="${l.ratio}" style="width:44px"></td>
      <td class="r">${row ? ocNum(row.bid) + " / " + ocNum(row.ask) : '<span class="neg">not in returned strikes</span>'}</td>
      <td class="r"><button class="btn xs ghost" data-oc-del="${i}">remove</button></td></tr>`;
  }).join("");
  el.innerHTML = `<div class="card" style="margin-bottom:12px"><div style="font:700 14px inherit;margin-bottom:6px">Legs <button class="btn xs ghost" data-oc-clear="1" style="margin-left:8px">clear all</button></div>
    <div class="tblwrap"><table class="tbl"><tbody>${rows}</tbody></table></div></div>`;
}

function ocMoney(v) { return v == null ? "-" : fmt.money(v); }

/* Numbers for the stats block and the confirm text; pure given the state.
   `ev` is the executor-rule risk (comboEvaluate) for the limit snapped to the
   0.05 grid; unbounded structures are evaluated with ack=true FOR DISPLAY ONLY
   (the payload's unbounded_ack is set only after the extra confirm). */
function ocComputeStats(legs, chain, qtyRaw, limitRaw, netMode, spot) {
  const struct = ocBuildStruct(legs, chain, netMode);
  if (!struct) return null;
  const qty = Number(qtyRaw);
  const limit = Number(limitRaw);
  const sgn = struct.credit ? -1 : 1;
  const action = struct.credit ? "SELL" : "BUY";
  const payLegs = legs.map((l) => ({ side: l.side, right: l.right, strike: l.strike, ratio: l.ratio }));
  const expiries = new Set(struct.legs.filter((l) => l.row).map((l) => String(l.row.expiry || chain.expiry).replace(/-/g, "")));
  const multiExpiry = expiries.size > 1;
  const atLimit = !multiExpiry && limit > 0 ? ocPayoff(payLegs, sgn * limit) : null;
  const atMid = !multiExpiry && struct.mid != null ? ocPayoff(payLegs, sgn * struct.mid) : null;
  const comm = COMM * legs.reduce((a, l) => a + (l.ratio || 1), 0) * 2;      // round trip per spread
  let ev = null;
  if (!struct.missing.length && limit > 0 && qty > 0 && Number.isInteger(qty)) {
    const snapped = snapNetLimit(limit, action);
    if (snapped > 0) {
      const wire = canonicalPayloadLegs(struct, chain.expiry, action).map((l) => ({ ...l, expiry: String(l.expiry).replace(/-/g, "") }));
      ev = comboEvaluate(action, snapped, qty, wire, { spot, unboundedAck: true });
    }
  }
  const info = ev && ev.info ? ev.info : null;
  const unbounded = !!(info && info.unbounded) || !!(atLimit && atLimit.lossUnbounded);
  const totalMaxLoss = info && !unbounded ? info.riskUsd
    : (atLimit && atLimit.maxLoss != null && qty > 0 ? (atLimit.maxLoss * 100 + comm) * qty : null);
  const totalStress = info && info.unbounded ? info.riskUsd : null;
  return { struct, qty, limit, atLimit, atMid, comm, totalMaxLoss, totalStress, ev, info, unbounded, multiExpiry, spot };
}

function ocCurrentSpot() { return ocSpotUsed(ocState.result, ocState.chain, ocState.spot); }

function renderCustomStats() {
  const el = document.getElementById("ocStats");
  if (!el) return;
  if (ocState.roll && typeof renderRollStats === "function") { renderRollStats(el); return; }
  const st = ocComputeStats(ocState.legs, ocState.chain, ocState.qty, ocState.limit, ocState.netMode, ocCurrentSpot());
  const lab = document.getElementById("oc_limit_label");
  if (!st) { el.innerHTML = ""; return; }
  const s = st.struct;
  if (lab) lab.textContent = s.credit ? "Credit limit (per spread)" : "Debit limit (per spread)";
  const limitEl = document.getElementById("oc_limit");
  if (limitEl && !ocState.limitTouched) {
    const d = ocDefaultLimit(s);
    ocState.limit = d == null ? "" : d.toFixed(2);
  }
  if (limitEl && document.activeElement !== limitEl && ocState.limit != null) limitEl.value = ocState.limit;
  const st2 = ocComputeStats(ocState.legs, ocState.chain, ocState.qty, ocState.limit, ocState.netMode, ocCurrentSpot());
  const info = st2.info, p = st2.atLimit || st2.atMid;
  const lim = Number(ocState.limit);
  const q = Number(ocState.qty);
  const qOk = Number.isInteger(q) && q > 0;
  let lossTxt = "-", gainTxt = "-", bes = "-", totalTxt = "-";
  if (st2.multiExpiry) {
    if (info) {
      lossTxt = ocMoney(info.unitLoss + st2.comm) + " (the debit)";
      gainTxt = "not defined at one expiry (depends on the later leg's value when the front leg expires)";
      bes = "n/a (calendar/diagonal)";
      totalTxt = ocMoney(info.riskUsd);
    }
  } else if (p) {
    bes = p.breakevens.length ? p.breakevens.map((b) => ocNum(b)).join(", ") : "-";
    gainTxt = p.gainUnbounded ? "unbounded (net long call tail)" : ocMoney(p.maxGain * 100 - st2.comm);
    if (p.lossUnbounded) {
      const sp = info && info.stressSpot;
      lossTxt = `<b class="neg">UNBOUNDED</b>; stress loss at +30%${sp ? " (underlying " + ocNum(st2.spot) + " -> " + ocNum(sp) + ")" : ""}: ${info ? ocMoney(info.unitLoss + st2.comm) : "-"}`;
      totalTxt = st2.totalStress != null ? `UNBOUNDED; stress loss at +30%: ${ocMoney(st2.totalStress)}` : "unbounded";
    } else {
      lossTxt = ocMoney(p.maxLoss * 100 + st2.comm);
      totalTxt = st2.totalMaxLoss != null ? ocMoney(st2.totalMaxLoss) : "-";
    }
  }
  const risk = lim > 0 && qOk && st2.ev && st2.ev.error && !s.missing.length && !s.execution_issue
    ? `<div class="k">Risk check</div><div class="v neg">${esc(st2.ev.error)}</div>` : "";
  el.innerHTML = `<div class="kv">
    <div class="k">Net at mid / natural</div><div class="v">${s.mid != null ? ocNum(s.mid) : "-"} / ${s.nat != null ? ocNum(s.nat) : "-"} ${s.credit ? "credit" : "debit"} per spread</div>
    <div class="k">Max loss / max gain at expiry</div><div class="v">${lossTxt} / ${gainTxt} <span class="cap" style="display:inline">per spread, at your limit, incl. ~${fmt.money(st2.comm)} round-trip commission</span></div>
    <div class="k">Breakevens</div><div class="v">${bes}</div>
    <div class="k">Total risk, ${qOk ? q : "?"} spread${q === 1 ? "" : "s"}</div><div class="v"><b>${totalTxt}</b> <span class="cap" style="display:inline">information only, not a cap</span></div>
    ${risk}
    ${s.missing.length ? `<div class="k">Warning</div><div class="v neg">${s.missing.length} leg(s) not in the returned strikes; recentre on them</div>` : ""}
    ${ocState.fresh && ocState.fresh.stale ? `<div class="k">Quotes</div><div class="v neg">${esc(ocState.fresh.text)}: the default limit uses that mid; check it against TWS before sending</div>` : ""}
    <div class="k">Execution</div><div class="v">${s.execution_issue ? `<span class="neg">${esc(s.execution_issue)}</span>` : "ready: one SMART " + (ocState.legs.length === 1 ? "option" : "BAG") + " limit order"}</div>
  </div>`;
}

function renderCustom() {
  const sel = document.getElementById("ocExpiry");
  if (!sel) return;
  const opts = ocState.expiries.map((e) =>
    `<option value="${e.date}"${e.date === ocState.expiry ? " selected" : ""}>${esc(ocIsoExpiry(e.date))}${e.dte != null ? " (" + e.dte + "d" + (e.dte === 0 ? ", 0DTE" : "") + ")" : ""}</option>`).join("");
  if (opts) sel.innerHTML = opts;
  const tk = document.getElementById("ocTicker");
  if (tk && ocState.ticker && !tk.value) tk.value = ocState.ticker;
  const note = document.getElementById("ocNote");
  if (note) note.textContent = ocState.note || "";
  const blocked = document.getElementById("ocBlocked");
  if (blocked) blocked.innerHTML = ocState.blocked
    ? `<div class="card" style="border-color:#ff6b6b;background:rgba(255,107,107,.10);padding:8px 12px;color:#ff6b6b;font-weight:700">[BLOCKED] ${esc(ocState.blocked)}</div>` : "";
  if (typeof renderRoll === "function") renderRoll();
  renderCustomLadder();
  renderCustomLegs();
  renderCustomStats();
  if (typeof renderActivity === "function") renderActivity();
}

function sendCustomOrder() {
  const msg = document.getElementById("oc_msg");
  const say = (t) => { if (msg) msg.textContent = t; };
  if (!ocState.chain) { say("Load a chain first"); return; }
  if (ocState.blocked) { say(ocState.blocked); return; }
  if (ocState.roll) { sendRollOrder(); return; }
  const spot = ocCurrentSpot();
  const st = ocComputeStats(ocState.legs, ocState.chain, ocState.qty, ocState.limit, ocState.netMode, spot);
  if (!st) { say("Add at least one leg"); return; }
  const s = st.struct;
  if (s.execution_issue) { say(s.execution_issue); return; }
  if (!(st.qty > 0) || !Number.isInteger(st.qty)) { say("BLOCKED: qty must be a positive integer"); return; }
  if (!(st.limit > 0)) { say("BLOCKED: limit must be > 0"); return; }
  const args = { struct: s, symbol: ocState.ticker, expiry: ocState.chain.expiry, qty: st.qty, limit: st.limit,
    tif: ocState.tif || "DAY", params: {}, spot, unboundedAck: false };
  let built = buildOptionSpreadPayload(args);
  const unbounded = !!built.needsAck;
  if (built.error && !unbounded) { say(built.error); return; }
  const legsTxt = ocState.legs.map((l) => `${l.side[0]}${l.ratio > 1 ? l.ratio + "x" : ""}${l.strike}${l.right}${l.expiry ? "@" + l.expiry.slice(2) : ""}`).join("/");
  const action = s.credit ? "SELL" : "BUY";
  const accountRow = (((state.book || {}).accounts) || []).find((a) => a.key === "primary");
  const nlv = Number(accountRow && accountRow.nlv);
  const closeNote = "\n\n" + OC_NO_EXITS_TEXT + ocStaleText(ocState.fresh);
  let totalLoss, riskTxt;
  if (unbounded) {
    if (!st.info || !st.info.unbounded) { say(built.error); return; }
    totalLoss = st.info.riskUsd;
    riskTxt = `UNBOUNDED structure (net short calls). Stress loss at underlying +30% for ${st.qty} spread${st.qty === 1 ? "" : "s"}: about ${fmt.money(totalLoss)}. Loss beyond that is unlimited.`;
  } else {
    const info = built.info;
    const p = ocPayoff(ocState.legs, (s.credit ? -1 : 1) * built.payload.limit);
    totalLoss = info ? info.riskUsd
      : (p && p.maxLoss != null ? (p.maxLoss * 100 + st.comm) * st.qty : built.riskPremium * 100 * st.qty + st.comm * st.qty);
    const pct = nlv > 0 ? ` (${(totalLoss / nlv * 100).toFixed(1)}% of NLV)` : " (NLV unavailable)";
    riskTxt = `Total defined max loss for ${st.qty} spread${st.qty === 1 ? "" : "s"}: about ${fmt.money(totalLoss)}${pct}. There is no size cap on this ticket; the execution agent validates the order.`;
  }
  const limitTxt = built.payload ? built.payload.limit : snapNetLimit(st.limit, action);
  if (!confirm(`${actionLead("place")} ${action} ${st.qty}x ${ocState.ticker} [${legsTxt}] LMT ${limitTxt} ${s.credit ? "credit" : "debit"} on primary?\n\n${riskTxt}${closeNote}`)) return;
  if (unbounded) {
    if (!confirm(`UNBOUNDED RISK - SECOND CONFIRMATION\n\n${ocState.ticker} [${legsTxt}] is an UNBOUNDED structure: it is net short calls and can lose more than any figure shown.\n\nStress loss at +30% (underlying ${ocNum(spot)} -> ${ocNum(st.info.stressSpot)}) for ${st.qty} spread${st.qty === 1 ? "" : "s"}: about ${fmt.money(st.info.riskUsd)}.\n\nSending sets unbounded_ack. Really continue?`)) return;
    built = buildOptionSpreadPayload({ ...args, unboundedAck: true });
    if (built.error) { say(built.error); return; }
  }
  const payload = built.payload;
  if (!(nlv > 0) || totalLoss > nlv * 0.05) payload.risk_ack = true;     // acknowledged in the confirm above
  sendCommand("option_spread", payload, "oc_msg", { account: "primary" });
}

function initCustom(prefill) {
  const box = document.getElementById("customSection");
  if (!box) return;
  box.innerHTML = ocShell();
  ocWireShell();
  if (prefill && prefill.roll) {
    document.getElementById("ocTicker").value = prefill.ticker;
    if (prefill.qty) { ocState.qty = prefill.qty; document.getElementById("oc_qty").value = prefill.qty; }
    if (!prefill.roll.legs.length) {
      ocState.blocked = "Roll link refused: the closing legs in the link are malformed or not opposite to the held positions.";
      renderCustom();
      return;
    }
    ocStartRoll(prefill.roll);
    return;
  }
  if (prefill && prefill.ticker) {
    document.getElementById("ocTicker").value = prefill.ticker;
    ocState.prefill = prefill;
    if (prefill.qty) { ocState.qty = prefill.qty; document.getElementById("oc_qty").value = prefill.qty; }
    ocLoadTicker(prefill.ticker);
  }
}
