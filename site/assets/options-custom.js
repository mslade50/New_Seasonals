/* options-custom.js - Custom spread builder for the Options tab.

   Any ticker, any listed expiry (0DTE and weeklies included), any strike, any
   quantity: pick legs off the live ladder, see the expiry payoff, set a limit
   and stage through the SAME signed /exec-command option_spread path as the
   shootout ticket (buildOptionSpreadPayload in options.js is the single payload
   builder, so the contract the desktop executor validates is unchanged).

   There is NO client-side quantity or risk cap here. The only gates are the
   desktop executor's. What the executor can place today is mirrored in
   OC_EXEC_CAPS; structures beyond it are analysed but the send button refuses
   with a clear message. Flip the caps when the executor learns more shapes.

   Loaded after options.js; shares its globals (state, esc, fmt, snapNetLimit,
   canonicalPayloadLegs, sendCommand, execMode, commRT, ...). */
"use strict";

/* What the desktop executor accepts today (exec_agent_core: one long option or
   one 1:1 same-expiry vertical). maxLegs up to 4 is the builder's own limit. */
const OC_EXEC_CAPS = { maxLegs: 2, allowRatio: false, allowShortSingle: false };
const OC_MAX_LEGS = 4;

const ocState = {
  ticker: null, result: null, expiries: [], chain: null, spot: null, expiry: null,
  legs: [], qty: 1, limit: null, limitTouched: false, tif: "DAY",
  center: null, centerRequested: null, note: "", loading: false, seq: 0,
  prefill: null, recentredOnce: false,
};

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

/* What the executor can place today, given resolved legs [{side,row,ratio}]. */
function ocExecIssue(legs, expiry, caps) {
  const c = caps || OC_EXEC_CAPS;
  if (!legs || !legs.length) return "Add at least one leg";
  if (legs.some((l) => !l.row)) return "A selected strike is not in the returned chain; recentre or re-quote";
  if (legs.some((l) => !l.row.con_id)) return "Contract qualification required (no conId on a leg)";
  if (legs.length > c.maxLegs) return `Analysis only: the execution agent currently takes ${c.maxLegs === 1 ? "single options" : "single options and verticals"} (${legs.length} legs selected)`;
  if (new Set(legs.map((l) => l.row.expiry || expiry)).size > 1) return "Analysis only: execution requires one expiry";
  if (!c.allowRatio && legs.some((l) => Number(l.ratio || 1) !== 1)) return "Analysis only: the execution agent does not take ratios yet (use ratio 1)";
  if (legs.length === 1) return legs[0].side === "BUY" || c.allowShortSingle ? null : "Analysis only: only long single options are executable";
  if (legs.length === 2) {
    const [a, b] = legs;
    if (a.side === b.side || a.row.right !== b.row.right || a.row.strike === b.row.strike) {
      return "Analysis only: execution requires a same-expiry, same-right vertical (one buy, one sell)";
    }
    return null;
  }
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

/* Struct in the same shape structureFrom emits, so canonicalPayloadLegs,
   commRT and buildOptionSpreadPayload treat it exactly like a shootout row. */
function ocBuildStruct(legs, chain) {
  if (!legs || !legs.length || !chain) return null;
  const rl = legs.map((l) => ({ side: l.side, ratio: l.ratio || 1, row: ocFindRow(chain, l.right, l.strike), want: l }));
  let signedMid = 0, signedNat = 0, midOk = true, natOk = true;
  for (const l of rl) {
    if (!l.row) { midOk = false; natOk = false; continue; }
    const s = l.side === "BUY" ? 1 : -1;
    const m = ocMid(l.row);
    const n = l.side === "BUY" ? l.row.ask : l.row.bid;
    if (m == null) midOk = false; else signedMid += s * l.ratio * m;
    if (n == null) natOk = false; else signedNat += s * l.ratio * Number(n);
  }
  const credit = midOk && signedMid < 0;
  const vertical = rl.length === 2 && rl.every((l) => l.row) && rl[0].row.right === rl[1].row.right &&
    rl[0].side !== rl[1].side && rl[0].row.strike !== rl[1].row.strike;
  const width = vertical ? Math.abs(rl[0].row.strike - rl[1].row.strike) : null;
  const rnd = (v) => Math.round(v * 100) / 100;
  const struct = {
    name: "Custom", legs: rl.map(({ side, ratio, row }) => ({ side, ratio, row })),
    mid: midOk ? rnd(Math.abs(signedMid)) : null,
    nat: natOk ? rnd(credit ? -signedNat : signedNat) : null,
    credit, width, category: "custom", note: "",
  };
  struct.execution_issue = ocExecIssue(rl.map(({ side, ratio, row }) => ({ side, ratio, row })), chain.expiry);
  struct.tradeable = !struct.execution_issue;
  struct.missing = rl.filter((l) => !l.row).map((l) => l.want);
  return struct;
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

/* Add or merge a leg: same contract + same side bumps the ratio, opposite side
   flips it. Returns false at the leg limit. */
function ocAddLeg(legs, right, strike, side) {
  const ex = legs.find((l) => l.right === right && l.strike === strike);
  if (ex) {
    if (ex.side === side) ex.ratio += 1; else { ex.side = side; ex.ratio = 1; }
    return true;
  }
  if (legs.length >= OC_MAX_LEGS) return false;
  legs.push({ right, strike, side, ratio: 1 });
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

function ocSetMsg(t) { const m = document.getElementById("ocMsg"); if (m) m.textContent = t || ""; }

async function ocFetch(mode, expiry, center) {
  const seq = ++ocState.seq;
  ocState.loading = true;
  ocSetMsg(mode === "full" ? "fetching chain + expiries... (~15-20s)" : "re-quoting...");
  const body = { ticker: ocState.ticker, mode, expiry: expiry || null, max_expiries: mode === "full" ? 40 : 2, context: null };
  if (center != null) body.strike_center = center;
  try {
    const res = await ocRequest(body);
    if (seq !== ocState.seq) return false;
    if (mode === "full") ocState.result = res;
    if (res.chain) {
      ocState.chain = res.chain; ocState.spot = res.spot; ocState.expiry = ocNormExpiry(res.chain.expiry);
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
  }
  const ok = await ocFetch("full", null, null);
  if (!ok) return;
  const pf = ocState.prefill;
  if (pf && pf.legs.length) {
    ocState.prefill = null;
    const want = pf.legs[0].expiry;
    ocState.legs = pf.legs.filter((l) => l.expiry === want).map((l) => ({ right: l.right, strike: l.strike, side: l.side, ratio: l.ratio }));
    ocState.note = pf.legs.length !== ocState.legs.length ? "Legs on other expiries were dropped: execution takes one expiry." : "";
    if (pf.qty) ocState.qty = pf.qty;
    if (pf.limit) { ocState.limit = pf.limit; ocState.limitTouched = true; }
    if (ocState.expiry !== want) await ocFetch("chain", want, null);
    const missing = ocState.legs.filter((l) => !ocFindRow(ocState.chain, l.right, l.strike));
    if (missing.length && !ocState.recentredOnce) {
      ocState.recentredOnce = true;
      const c = ocState.legs.reduce((a, l) => a + l.strike, 0) / ocState.legs.length;
      await ocFetch("chain", want, c);
    }
    renderCustom();
  }
}

/* ---------------- render ---------------- */

const ocNum = (v, d) => (v == null || !isFinite(Number(v)) ? "-" : Number(v).toFixed(d == null ? 2 : d));

function ocShell() {
  return `<div class="card" style="margin-bottom:12px">
    <div style="font:700 14px inherit;margin-bottom:6px">Custom spread
      <span class="cap" style="display:inline;font-weight:400">- any ticker, any listed expiry, any strikes; no size cap here (the execution agent is the only gate)</span></div>
    <div style="display:flex;gap:10px;align-items:end;flex-wrap:wrap">
      <label><span class="cap">Ticker</span><br><input id="ocTicker" placeholder="SPY" style="text-transform:uppercase;width:90px"></label>
      <button class="btn" id="ocGo">Load chain</button>
      <label><span class="cap">Expiry</span><br><select id="ocExpiry" style="min-width:190px"><option value="">load a ticker first</option></select></label>
      <label><span class="cap">Strike centre</span><br><input id="ocCenter" placeholder="spot" style="width:80px"></label>
      <button class="btn ghost" id="ocRecentre">Recentre</button>
      <span id="ocMsg" class="cap"></span>
    </div>
    <div id="ocNote" class="cap" style="margin-top:6px;color:#ffc14d"></div>
  </div>
  <div id="ocLadder"></div>
  <div id="ocLegs"></div>
  <div class="card" style="margin-bottom:12px" id="ocOrderCard">
    <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px">
      <label class="cap">Qty (spreads)</label><input id="oc_qty" value="${ocState.qty}" style="width:80px" inputmode="numeric">
      <label class="cap" id="oc_limit_label">Limit (per spread)</label><input id="oc_limit" value="" style="width:80px">
      <label class="cap">TIF</label><select id="oc_tif"><option>DAY</option><option>GTC</option></select>
      <span class="cap">Account: <b>primary</b> (PA options remain disabled)</span>
    </div>
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
    ocState.legs = [];                                      // strikes belong to one expiry
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
  $("oc_send").addEventListener("click", sendCustomOrder);
  $("ocLadder").addEventListener("click", (e) => {
    const b = e.target.closest("[data-oc-add]");
    if (!b) return;
    const [right, strike, side] = b.dataset.ocAdd.split("|");
    if (!ocAddLeg(ocState.legs, right, Number(strike), side)) ocSetMsg(`at most ${OC_MAX_LEGS} legs`);
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
    const b = e.target.closest("[data-oc-del]");
    if (!b) return;
    ocState.legs.splice(Number(b.dataset.ocDel), 1);
    ocState.limitTouched = false; renderCustom();
  });
}

function renderCustomLadder() {
  const el = document.getElementById("ocLadder");
  if (!el) return;
  const chain = ocState.chain;
  if (!chain) { el.innerHTML = ""; return; }
  const by = new Map();
  for (const r of chain.strikes || []) {
    const k = Number(r.strike);
    if (!by.has(k)) by.set(k, {});
    by.get(k)[r.right] = r;
  }
  const strikes = [...by.keys()].sort((a, b) => a - b);
  const spot = Number(ocState.spot);
  let atm = null;
  strikes.forEach((k) => { if (atm == null || Math.abs(k - spot) < Math.abs(atm - spot)) atm = k; });
  const sel = (right, k, side) => ocState.legs.some((l) => l.right === right && l.strike === k && l.side === side);
  const cell = (right, k, side, v) => {
    if (v == null) return '<td class="r">-</td>';
    const on = sel(right, k, side);
    return `<td class="r"><button class="btn xs${on ? "" : " ghost"}" data-oc-add="${right}|${k}|${side}" title="${side === "BUY" ? "buy at the ask" : "sell at the bid"}">${ocNum(v)}</button></td>`;
  };
  const side = (right, k) => {
    const r = (by.get(k) || {})[right];
    if (!r) return '<td class="r" colspan="5">-</td>';
    return `${cell(right, k, "SELL", r.bid)}${cell(right, k, "BUY", r.ask)}<td class="r">${ocNum(ocMid(r))}</td>` +
      `<td class="r">${r.iv != null ? (Number(r.iv) * 100).toFixed(1) : "-"}</td><td class="r">${ocNum(r.delta, 2)}</td>`;
  };
  const head = ["bid", "ask", "mid", "iv", "delta"].map((h) => `<th class="r">${h}</th>`).join("");
  const rows = strikes.map((k) => `<tr${k === atm ? ' style="background:rgba(77,163,255,.10)"' : ""}>${side("C", k)}<td class="c"><b>${k}</b></td>${side("P", k)}</tr>`).join("");
  el.innerHTML = `<div class="card" style="margin-bottom:12px">
    <div style="font:700 14px inherit;margin-bottom:6px">Strike ladder - ${esc(ocState.ticker)} ${esc(ocIsoExpiry(chain.expiry))}
      <span class="cap" style="display:inline;font-weight:400">- spot ${ocNum(ocState.spot)} - ${chain.dte != null ? chain.dte + "d" : ""} - click a bid to SELL, an ask to BUY (calls left, puts right)</span></div>
    <div class="tblwrap" style="max-height:420px;overflow:auto"><table class="tbl"><thead>
      <tr><th colspan="5" class="c">CALLS</th><th></th><th colspan="5" class="c">PUTS</th></tr>
      <tr>${head}<th class="c">strike</th>${head}</tr></thead><tbody>${rows}</tbody></table></div></div>`;
}

function renderCustomLegs() {
  const el = document.getElementById("ocLegs");
  if (!el) return;
  if (!ocState.legs.length) {
    el.innerHTML = '<div class="card" style="margin-bottom:12px"><span class="cap">No legs yet. Click a bid or ask in the ladder (up to 4 legs, one expiry).</span></div>';
    return;
  }
  const rows = ocState.legs.map((l, i) => {
    const row = ocFindRow(ocState.chain, l.right, l.strike);
    return `<tr><td class="l"><b>${l.side}</b> ${l.right === "C" ? "Call" : "Put"} ${l.strike} ${esc(ocIsoExpiry(ocState.expiry))}</td>
      <td class="r">ratio <input data-oc-ratio="${i}" value="${l.ratio}" style="width:44px"></td>
      <td class="r">${row ? ocNum(row.bid) + " / " + ocNum(row.ask) : '<span class="neg">not in returned strikes</span>'}</td>
      <td class="r"><button class="btn xs ghost" data-oc-del="${i}">remove</button></td></tr>`;
  }).join("");
  el.innerHTML = `<div class="card" style="margin-bottom:12px"><div style="font:700 14px inherit;margin-bottom:6px">Legs</div>
    <div class="tblwrap"><table class="tbl"><tbody>${rows}</tbody></table></div></div>`;
}

function ocMoney(v) { return v == null ? "-" : fmt.money(v); }

/* Numbers for the stats block and the confirm text; pure given the state. */
function ocComputeStats(legs, chain, qtyRaw, limitRaw) {
  const struct = ocBuildStruct(legs, chain);
  if (!struct) return null;
  const qty = Number(qtyRaw);
  const limit = Number(limitRaw);
  const sgn = struct.credit ? -1 : 1;
  const payLegs = legs.map((l) => ({ side: l.side, right: l.right, strike: l.strike, ratio: l.ratio }));
  const atLimit = limit > 0 ? ocPayoff(payLegs, sgn * limit) : null;
  const atMid = struct.mid != null ? ocPayoff(payLegs, sgn * struct.mid) : null;
  const comm = COMM * legs.reduce((a, l) => a + (l.ratio || 1), 0) * 2;      // round trip per spread
  const totalMaxLoss = atLimit && atLimit.maxLoss != null && qty > 0 ? (atLimit.maxLoss * 100 + comm) * qty : null;
  return { struct, qty, limit, atLimit, atMid, comm, totalMaxLoss };
}

function renderCustomStats() {
  const el = document.getElementById("ocStats");
  if (!el) return;
  const st = ocComputeStats(ocState.legs, ocState.chain, ocState.qty, ocState.limit);
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
  const lim = Number(ocState.limit);
  const p = lim > 0 ? ocPayoff(ocState.legs, (s.credit ? -1 : 1) * lim) : null;
  const pay = p || st.atMid;
  const lossTxt = pay ? (pay.lossUnbounded ? '<b class="neg">UNBOUNDED (net short call tail)</b>' : ocMoney(pay.maxLoss * 100 + st.comm)) : "-";
  const gainTxt = pay ? (pay.gainUnbounded ? "unbounded (net long call tail)" : ocMoney(pay.maxGain * 100 - st.comm)) : "-";
  const bes = pay && pay.breakevens.length ? pay.breakevens.map((b) => ocNum(b)).join(", ") : "-";
  const q = Number(ocState.qty);
  const totalTxt = pay && pay.lossUnbounded ? "unbounded"
    : (st.totalMaxLoss != null ? ocMoney(st.totalMaxLoss) : "-");
  el.innerHTML = `<div class="kv">
    <div class="k">Net at mid / natural</div><div class="v">${s.mid != null ? ocNum(s.mid) : "-"} / ${s.nat != null ? ocNum(s.nat) : "-"} ${s.credit ? "credit" : "debit"} per spread</div>
    <div class="k">Max loss / max gain at expiry</div><div class="v">${lossTxt} / ${gainTxt} <span class="cap" style="display:inline">per spread, at your limit, incl. ~${fmt.money(st.comm)} round-trip commission</span></div>
    <div class="k">Breakevens</div><div class="v">${bes}</div>
    <div class="k">Total max loss, ${Number.isInteger(q) && q > 0 ? q : "?"} spread${q === 1 ? "" : "s"}</div><div class="v"><b>${totalTxt}</b> <span class="cap" style="display:inline">information only, not a cap</span></div>
    ${s.missing.length ? `<div class="k">Warning</div><div class="v neg">${s.missing.length} leg(s) not in the returned strikes; recentre on them</div>` : ""}
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
  renderCustomLadder();
  renderCustomLegs();
  renderCustomStats();
  if (typeof renderActivity === "function") renderActivity();
}

function sendCustomOrder() {
  const msg = document.getElementById("oc_msg");
  const say = (t) => { if (msg) msg.textContent = t; };
  if (!ocState.chain) { say("Load a chain first"); return; }
  const st = ocComputeStats(ocState.legs, ocState.chain, ocState.qty, ocState.limit);
  if (!st) { say("Add at least one leg"); return; }
  const s = st.struct;
  if (s.execution_issue) { say(s.execution_issue); return; }
  if (!(st.qty > 0) || !Number.isInteger(st.qty)) { say("BLOCKED: qty must be a positive integer"); return; }
  if (!(st.limit > 0)) { say("BLOCKED: limit must be > 0"); return; }
  const built = buildOptionSpreadPayload({
    struct: s, symbol: ocState.ticker, expiry: ocState.chain.expiry, qty: st.qty, limit: st.limit,
    tif: ocState.tif || "DAY", params: {},
  });
  if (built.error) { say(built.error); return; }
  const { payload, riskPremium, action } = built;
  const p = ocPayoff(ocState.legs, (s.credit ? -1 : 1) * payload.limit);
  const totalLoss = p && p.maxLoss != null ? (p.maxLoss * 100 + st.comm) * st.qty : riskPremium * 100 * st.qty + st.comm * st.qty;
  const legsTxt = ocState.legs.map((l) => `${l.side[0]}${l.ratio > 1 ? l.ratio + "x" : ""}${l.strike}${l.right}`).join("/");
  const accountRow = (((state.book || {}).accounts) || []).find((a) => a.key === "primary");
  const nlv = Number(accountRow && accountRow.nlv);
  const pct = nlv > 0 ? ` (${(totalLoss / nlv * 100).toFixed(1)}% of NLV)` : " (NLV unavailable)";
  if (!confirm(`${actionLead("place")} ${action} ${st.qty}x ${ocState.ticker} ${ocIsoExpiry(ocState.chain.expiry)} [${legsTxt}] LMT ${payload.limit} ${s.credit ? "credit" : "debit"} on primary?\n\nTotal defined max loss for ${st.qty} spread${st.qty === 1 ? "" : "s"}: about ${fmt.money(totalLoss)}${pct}. There is no size cap on this ticket; the execution agent validates the order.`)) return;
  if (!(nlv > 0) || totalLoss > nlv * 0.05) payload.risk_ack = true;     // acknowledged in the confirm above
  sendCommand("option_spread", payload, "oc_msg", { account: "primary" });
}

function initCustom(prefill) {
  const box = document.getElementById("customSection");
  if (!box) return;
  box.innerHTML = ocShell();
  ocWireShell();
  if (prefill && prefill.ticker) {
    document.getElementById("ocTicker").value = prefill.ticker;
    ocState.prefill = prefill;
    if (prefill.qty) { ocState.qty = prefill.qty; document.getElementById("oc_qty").value = prefill.qty; }
    ocLoadTicker(prefill.ticker);
  }
}
