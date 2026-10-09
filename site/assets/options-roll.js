/* options-roll.js - Roll mode for the Options tab Custom builder.

   Entered from the Execution tab (position row > Roll...), which links to
   options.html?section=custom&ticker=XYZ&roll=<legs>&qty=<units>&acct=<key>.
   The held legs arrive LOCKED as the closing side; the owner picks the new legs
   off the ladder. One `option_roll` command is sent: closing + new legs as ONE
   SMART BAG with one net limit, so the roll fills atomically
   (OPTION_COMBO_SPEC.md section 6).

   Risk of the resulting structure = the NEW legs valued with the option_spread
   rule at the roll's signed net price (ocRollRisk is the JS port of
   option_position_orders.open_structure_risk; the executor recomputes it and
   rejects a mismatching debit_risk). The payoff shown is the new legs at that
   net price.

   Loaded after options-custom.js; shares its globals (ocState, ocRequest, ...). */
"use strict";

/* "cid:ACTION:ratio:right:strike:expiry:held,..." -> closing legs. */
function ocParseRollLegs(str) {
  const out = [];
  if (!str) return out;
  for (const part of String(str).split(",")) {
    const f = part.split(":");
    if (f.length !== 7) return [];
    const [cid, action, ratio, right, strike, expiry, held] = f;
    const leg = { con_id: Number(cid), action: String(action).toUpperCase(), ratio: Number(ratio),
      right: String(right).toUpperCase(), strike: Number(strike), expiry: ocNormExpiry(expiry), held: Number(held) };
    if (!(Number.isInteger(leg.con_id) && leg.con_id > 0) || !["BUY", "SELL"].includes(leg.action) ||
        !(Number.isInteger(leg.ratio) && leg.ratio >= 1) || !["C", "P"].includes(leg.right) ||
        !(leg.strike > 0) || !leg.expiry || !Number.isFinite(leg.held) || leg.held === 0) return [];
    if ((leg.held > 0) !== (leg.action === "SELL")) return [];          // a closing leg opposes the held sign
    out.push(leg);
  }
  return out.length <= 4 ? out : [];
}

/* JS port of option_position_orders.open_structure_risk. open = REAL-side legs
   [{side,right,strike,expiry,ratio}]. Returns {info} | {error, needsAck}. */
function ocRollRisk(action, limit, qty, open, closeRatioSum, { spot = null, unboundedAck = false } = {}) {
  limit = Number(limit); qty = Number(qty);
  if (!(Number.isFinite(limit) && limit >= 0) || !(qty > 0)) return { error: "quantity must be > 0 and limit >= 0" };
  if (!open || !open.length) return { error: "pick at least one new leg from the ladder" };
  if (open.length > 4) return { error: "at most 4 new legs" };
  const keys = new Set();
  for (const l of open) {
    const k = `${l.right}|${l.expiry}|${Number(l.strike).toFixed(6)}`;
    if (keys.has(k)) return { error: "duplicate new legs (same right, expiry and strike)" };
    keys.add(k);
  }
  const L = action === "BUY" ? limit : -limit;
  const eff = open.map((l) => ({ sign: l.side === "BUY" ? 1 : -1, ratio: Number(l.ratio || 1), right: l.right,
    strike: Number(l.strike), expiry: String(l.expiry) }));
  let unitLoss, unbounded = false, stress = null;
  if (new Set(eff.map((e) => e.expiry)).size === 1) {
    const slope = eff.reduce((a, e) => a + (e.right === "C" ? e.sign * e.ratio : 0), 0);
    unbounded = slope < 0;
    const strikes = [...new Set(eff.map((e) => e.strike))].sort((a, b) => a - b);
    let grid;
    if (unbounded) {
      if (unboundedAck !== true) return { error: "UNBOUNDED_ACK_REQUIRED: the new legs are net short calls (unbounded loss)", needsAck: true };
      const sp = Number(spot);
      if (!Number.isFinite(sp) || sp <= 0) return { error: "unbounded roll needs the underlying price (unavailable)" };
      stress = sp * COMBO_STRESS;
      grid = [0, ...strikes.filter((k) => k <= stress), stress];
    } else {
      grid = [0, ...strikes];
    }
    unitLoss = Math.max(0, COMBO_MULT * L - Math.min(...grid.map((S) => comboPayoff(eff, S))));
  } else {
    const chk = comboCheckStructure("BUY", open.map((l) => ({ side: l.side, right: l.right, expiry: l.expiry, strike: l.strike, ratio: l.ratio })));
    if (chk.error || chk.shape !== "calendar") {
      return { error: `new legs on different expiries are accepted only as a covered 2-leg calendar/diagonal (${chk.error || "not a calendar"})` };
    }
    unitLoss = Math.max(0, COMBO_MULT * L);
  }
  const raw = unitLoss / COMBO_MULT;
  const debitRisk = unbounded ? Math.ceil(raw * 100 - 1e-9) / 100 : Math.round(raw * 1e6) / 1e6;
  const openRatio = open.reduce((a, l) => a + Number(l.ratio || 1), 0);
  const comm = COMM * qty * (Number(closeRatioSum || 0) + 2 * openRatio);
  const unit = unbounded ? Math.max(debitRisk, raw) : raw;
  return { info: { unitLoss, unitRisk: raw, debitRisk, unbounded, stressSpot: stress, comm, riskUsd: unit * COMBO_MULT * qty + comm } };
}

/* Signed nets (debit +) of the locked closing legs and the new legs. */
function ocRollNets(roll, open) {
  const close = optNetQuote(roll.legs, (l) => roll.rows.get(l.con_id));
  let mid = 0, nat = 0, midOk = true, natOk = true;
  for (const l of open) {
    const s = l.side === "BUY" ? 1 : -1, r = l.row;
    const m = r ? ocMid(r) : null, n = r ? (l.side === "BUY" ? r.ask : r.bid) : null;
    if (m == null) midOk = false; else mid += s * l.ratio * m;
    if (n == null) natOk = false; else nat += s * l.ratio * Number(n);
  }
  const add = (a, b) => (a == null || b == null ? null : Math.round((a + b) * 10000) / 10000);
  return { close, open: { mid: midOk ? mid : null, nat: natOk ? nat : null },
    mid: add(close.mid, midOk ? mid : null), nat: add(close.nat, natOk ? nat : null) };
}

/* option_roll payload (OPTION_COMBO_SPEC.md section 6) or {error, needsAck}. */
function ocBuildRollPayload({ symbol, roll, open, qty, limit, action, tif, spot, unboundedAck }) {
  if (!(Number.isInteger(qty) && qty > 0)) return { error: "units must be a positive whole number" };
  for (const l of roll.legs) {
    if (qty * l.ratio > Math.abs(l.held)) return { error: `closing ${qty * l.ratio} exceeds held ${Math.abs(l.held)} (${l.strike}${l.right})` };
  }
  if (roll.legs.length + open.length > 6) return { error: "a roll has at most 6 legs in total" };
  if (open.some((l) => !l.row)) return { error: "a new leg is not in the returned strikes; recentre on it" };
  const closeIds = new Set(roll.legs.map((l) => l.con_id));
  if (open.some((l) => l.row && closeIds.has(Number(l.row.con_id)))) return { error: "a new leg is the same contract as a closing leg" };
  const sides = new Set([...roll.legs.map((l) => l.action), ...open.map((l) => l.side)]);
  if (sides.size === 1 && [...sides][0] !== action) return { error: `every leg ${[...sides][0] === "SELL" ? "sells: a net CREDIT" : "buys: a net DEBIT"}; switch Net` };
  const snapped = snapNetLimit(limit, action);
  if (!(snapped >= 0)) return { error: "limit must be >= 0" };
  const ev = ocRollRisk(action, snapped, qty, open, roll.legs.reduce((a, l) => a + l.ratio, 0), { spot, unboundedAck });
  if (ev.error) return { error: ev.error, needsAck: !!ev.needsAck };
  const payload = {
    symbol, action, quantity: qty, limit: snapped, tif: tif || "DAY",
    close_legs: roll.legs.map((l) => ({ con_id: l.con_id, action: l.action, ratio: l.ratio, right: l.right, strike: l.strike, expiry: l.expiry })),
    open_legs: open.map((l) => ({ side: l.side, right: l.right, expiry: String(l.expiry).replace(/-/g, ""), strike: l.strike,
      ratio: l.ratio, con_id: l.row && l.row.con_id ? Number(l.row.con_id) : null })),
    debit_risk: ev.info.debitRisk,
  };
  if (ev.info.unbounded) { payload.unbounded_ack = true; payload.underlying_spot = Number(spot); }
  return { payload, info: ev.info, snapped };
}

/* ---------------- state / quotes ---------------- */

function ocRollOpenLegs() {
  const chain = ocState.chain;
  return ocState.legs.map((l) => {
    const row = chain ? ocResolveRow(l, chain) : (l.row || null);
    return { side: l.side, right: l.right, strike: l.strike, ratio: l.ratio || 1,
      expiry: ocNormExpiry(l.expiry || (row && row.expiry) || (chain && chain.expiry)), row };
  });
}

async function ocStartRoll(roll) {
  ocState.roll = roll;
  await ocLoadTicker(roll.symbol);
  const expiries = [...new Set(roll.legs.map((l) => l.expiry))];
  let worst = null;
  for (const exp of expiries) {
    const mine = roll.legs.filter((l) => l.expiry === exp);
    try {
      const res = await ocRequest({ ticker: roll.symbol, mode: "chain", expiry: exp, max_expiries: 2, context: null,
        strike_center: mine.reduce((a, l) => a + l.strike, 0) / mine.length });
      for (const l of mine) {
        const row = ((res.chain && res.chain.strikes) || []).find((r) => Number(r.con_id) === l.con_id) ||
          ocFindRow(res.chain, l.right, l.strike);
        if (row) roll.rows.set(l.con_id, row);
      }
      const f = ocFreshness(res);
      if (f && (!worst || (f.stale && !worst.stale))) worst = f;
    } catch (e) {
      roll.error = `closing-leg quotes unavailable: ${e.message || e}`;
    }
  }
  roll.fresh = worst;
  renderCustom();
}

/* ---------------- render ---------------- */

function renderRoll() {
  const el = document.getElementById("ocRoll");
  if (!el) return;
  const roll = ocState.roll;
  if (!roll) { el.innerHTML = ""; return; }
  const supported = typeof optSupports === "function" && optSupports(state.book, "option_roll");
  const rows = roll.legs.map((l) => {
    const q = roll.rows.get(l.con_id) || {};
    return `<tr><td class="l"><b>${l.action}</b> ${l.ratio > 1 ? l.ratio + "x " : ""}${esc(roll.symbol)} ${esc(ocIsoExpiry(l.expiry))} ${l.strike}${l.right}</td>
      <td class="r">held ${l.held}</td><td class="r">${ocNum(q.bid)} / ${ocNum(q.ask)}</td><td class="r cap">locked (conId ${l.con_id})</td></tr>`;
  }).join("");
  el.innerHTML = `<div class="card" style="margin-bottom:12px;border-color:#4da3ff">
    <div style="font:700 14px inherit;margin-bottom:6px">ROLL MODE &mdash; closing legs (locked), account <b>${esc(roll.account)}</b>
      <span class="cap" style="display:inline;font-weight:400">- pick the new legs from the ladder; all legs go out as ONE BAG with one net limit (fills together or not at all)</span></div>
    ${supported ? "" : '<div class="neg" style="font-weight:700;margin-bottom:6px">[BLOCKED] The running execution agent does not advertise option_roll; nothing can be sent.</div>'}
    ${roll.error ? `<div class="cap neg">${esc(roll.error)}</div>` : ""}
    ${roll.fresh && roll.fresh.stale ? `<div class="cap neg">Closing-leg quotes: ${esc(roll.fresh.text)}</div>` : ""}
    <div class="tblwrap"><table class="tbl"><tbody>${rows}</tbody></table></div></div>`;
}

/* Called from renderCustomStats in roll mode. */
function renderRollStats(el) {
  const roll = ocState.roll;
  const open = ocRollOpenLegs();
  const nets = ocRollNets(roll, open);
  const auto = nets.mid != null && nets.mid < 0 ? "SELL" : "BUY";
  const action = ocState.netMode === "credit" ? "SELL" : ocState.netMode === "debit" ? "BUY" : auto;
  const lab = document.getElementById("oc_limit_label");
  if (lab) lab.textContent = action === "SELL" ? "Net credit limit (per roll)" : "Net debit limit (per roll)";
  const limitEl = document.getElementById("oc_limit");
  if (!ocState.limitTouched) ocState.limit = nets.mid == null ? "" : snapNetLimit(Math.abs(nets.mid), action).toFixed(2);
  if (limitEl && document.activeElement !== limitEl && ocState.limit != null) limitEl.value = ocState.limit;
  const qty = Number(ocState.qty), lim = Number(ocState.limit);
  const ev = open.length && lim >= 0 && qty > 0
    ? ocRollRisk(action, snapNetLimit(lim, action), qty, open, roll.legs.reduce((a, l) => a + l.ratio, 0), { spot: ocCurrentSpot(), unboundedAck: true }) : null;
  const sameExp = new Set(open.map((l) => l.expiry)).size === 1;
  const p = open.length && sameExp && lim >= 0 ? ocPayoff(open, (action === "BUY" ? 1 : -1) * lim) : null;
  const sgn = (v) => (v == null ? "-" : `${ocNum(Math.abs(v))} ${v < 0 ? "credit" : "debit"}`);
  let payoff = "-";
  if (p) {
    payoff = `${p.lossUnbounded ? '<b class="neg">UNBOUNDED</b>' : ocMoney(p.maxLoss * 100)} / ${p.gainUnbounded ? "unbounded" : ocMoney(p.maxGain * 100)}; breakevens ${p.breakevens.length ? p.breakevens.map((b) => ocNum(b)).join(", ") : "-"}`;
  } else if (open.length && !sameExp) payoff = "calendar/diagonal: max loss = the net debit; gain not defined at one expiry";
  el.innerHTML = `<div class="kv">
    <div class="k">Close legs mid / natural</div><div class="v">${sgn(nets.close.mid)} / ${sgn(nets.close.nat)}</div>
    <div class="k">New legs mid / natural</div><div class="v">${sgn(nets.open.mid)} / ${sgn(nets.open.nat)}</div>
    <div class="k">Roll net mid / natural</div><div class="v"><b>${sgn(nets.mid)}</b> / ${sgn(nets.nat)} per unit</div>
    <div class="k">Resulting structure at the roll net</div><div class="v">max loss / max gain ${payoff} <span class="cap" style="display:inline">(new legs only, per unit, at expiry)</span></div>
    <div class="k">Risk check (executor rule)</div><div class="v">${ev ? (ev.error ? `<span class="neg">${esc(ev.error)}</span>` : `${ev.info.unbounded ? "UNBOUNDED; stress loss at +30%: " : ""}${ocMoney(ev.info.riskUsd)} for ${qty} unit${qty === 1 ? "" : "s"} incl. ~${ocMoney(ev.info.comm)} commissions`) : "-"}</div>
    ${(ocState.fresh && ocState.fresh.stale) || (roll.fresh && roll.fresh.stale) ? '<div class="k">Quotes</div><div class="v neg">stale/frozen/delayed quotes: the default limit uses those mids; check against TWS</div>' : ""}
  </div>`;
}

function sendRollOrder() {
  const msg = document.getElementById("oc_msg");
  const say = (t) => { if (msg) msg.textContent = t; };
  const roll = ocState.roll;
  if (!(typeof optSupports === "function" && optSupports(state.book, "option_roll"))) { say("BLOCKED: the running agent does not support option_roll"); return; }
  const open = ocRollOpenLegs();
  const nets = ocRollNets(roll, open);
  const auto = nets.mid != null && nets.mid < 0 ? "SELL" : "BUY";
  const action = ocState.netMode === "credit" ? "SELL" : ocState.netMode === "debit" ? "BUY" : auto;
  const qty = Number(ocState.qty), limit = Number(ocState.limit);
  if (!(limit >= 0) || ocState.limit === "" || ocState.limit == null) { say("BLOCKED: enter a net limit"); return; }
  const spot = ocCurrentSpot();
  const args = { symbol: roll.symbol, roll, open, qty, limit, action, tif: ocState.tif || "DAY", spot, unboundedAck: false };
  let built = ocBuildRollPayload(args);
  if (built.error && !built.needsAck) { say("BLOCKED: " + built.error); return; }
  const unbounded = !!built.needsAck;
  if (unbounded) {
    built = ocBuildRollPayload({ ...args, unboundedAck: true });
    if (built.error) { say("BLOCKED: " + built.error); return; }
  }
  const p = built.payload, info = built.info;
  const closeTxt = roll.legs.map((l) => `  CLOSE ${l.action} ${qty * l.ratio} ${roll.symbol} ${l.expiry} ${l.strike}${l.right} (conId ${l.con_id}, held ${l.held})`).join("\n");
  const openTxt = open.map((l) => `  OPEN  ${l.side} ${qty * l.ratio} ${roll.symbol} ${l.expiry} ${l.strike}${l.right}`).join("\n");
  const accountRow = (((state.book || {}).accounts) || []).find((a) => a.key === roll.account);
  const nlv = Number(accountRow && accountRow.nlv);
  const riskTxt = info.unbounded
    ? `UNBOUNDED new structure (net short calls). Stress loss at underlying +30%: about ${fmt.money(info.riskUsd)}; loss beyond that is unlimited.`
    : `Risk of the resulting structure at this net: about ${fmt.money(info.riskUsd)}${nlv > 0 ? ` (${(info.riskUsd / nlv * 100).toFixed(1)}% of NLV)` : " (NLV unavailable)"}.`;
  const fresh = (roll.fresh && roll.fresh.stale) ? roll.fresh : ocState.fresh;
  const text = `ROLL on ${roll.account}: one SMART BAG combo limit order (all legs fill together or not at all)\n${closeTxt}\n${openTxt}\n` +
    `Order: ${p.action} ${p.quantity}x LMT ${p.limit.toFixed(2)} net ${p.action === "BUY" ? "DEBIT (you pay)" : "CREDIT (you receive)"}, ${p.tif}.\n\n${riskTxt}\n\n${OC_NO_EXITS_TEXT}${ocStaleText(fresh)}`;
  if (!confirm(`${actionLead("roll")}\n\n${text}`)) return;
  if (info.unbounded && !confirm(`UNBOUNDED RISK - SECOND CONFIRMATION\n\nThe new legs are net short calls and can lose more than any figure shown. Sending sets unbounded_ack. Really continue?`)) return;
  if (!(nlv > 0) || info.riskUsd > nlv * 0.05) p.risk_ack = true;      // acknowledged in the confirm above
  sendCommand("option_roll", p, "oc_msg", { account: roll.account });
}
