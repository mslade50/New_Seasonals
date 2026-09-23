/* pitch.js — today's Daily Pitch ideas, and a deep link that stages one leg.
 *
 * Reads /pitch-today (Pages Function -> R2 key pitch_today.json, published by
 * daily_pitch.py right after the morning's email). RULE (McKinley, 2026-09-23):
 * stage EXACTLY what the pitch says, or block. No substitutes, no derived
 * levels. Quantities, limits, stops, targets, order types and dates come from
 * the publisher's order rows unchanged. The one price this page computes is an
 * OPEN-anchored limit, which the pitch itself prices off the session open; the
 * math and rounding mirror pitch_moo.price_open_row (trading_ibkr) exactly.
 * Anything the pitch marks Manual_Only (futures, MOO/MOC with a price stop or
 * target) or routes through a proxy is blocked, never approximated.
 *
 * The ticket carries the pitch's two execution conventions: stops arm the next
 * session (stop_arm) and a MOO time exit fires at the open (time_stop_at).
 *
 * Clock (pitch_moo's): a leg is stageable only AFTER its pitch_moo pass has run
 * (auction 09:05, open 09:32), so the site can never double a runner placement;
 * MOO entries close at the 09:25 OPG cutoff, MOC entries at 15:30. The same
 * gates re-run at prefill in execution.js. The pass wait applies only while
 * PITCH_MOO_ARMED; the stage link carries armed=0|1 and execution.js enforces
 * the wait unless armed=0 (a missing param fails closed).
 *
 * Stage flow: execution.html?stage=pitch&... -> execution.js applyPitchPrefill()
 * fills the entry-bracket ticket. Nothing is sent.
 */
"use strict";

const PITCH_ENDPOINT = "/pitch-today";
// PRIMARY only: the pitch sizes off the fixed ACCOUNT_VALUE basis, and
// pitch_moo.py places on the primary account.
const PITCH_ACCOUNT = "primary";
// pitch_moo.OPG_CUTOFF / MOC_CUTOFF: the runner's own refusal times.
const PITCH_OPG_CUTOFF = "09:25";
const PITCH_MOC_CUTOFF = "15:30";
const PITCH_SESSION_OPEN = "09:30";
// When each pitch_moo pass has run. Staging before it risks a second copy of
// the order: the runner's dedupe cannot see site-staged orders.
const PITCH_PASS_AFTER = { auction: "09:05", open: "09:32" };
// Set true only when the pitch_moo tasks and pitch_moo_enabled.flag are armed.
const PITCH_MOO_ARMED = false;
// ctx.armed (tests) overrides the constant; the stage link carries the result.
const pitchArmed = (c) => (c && typeof c.armed === "boolean" ? c.armed : PITCH_MOO_ARMED);
const PITCH_RERENDER_MS = 60000;

const pnum = (v) => {
  if (typeof v === "number") return isFinite(v) ? v : null;
  if (typeof v === "string" && v.trim() !== "" && isFinite(Number(v))) return Number(v);
  return null;
};
const ppx = (v) => (pnum(v) == null ? "-" : pnum(v).toFixed(2));
const pesc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pup = (v) => String(v == null ? "" : v).toUpperCase().trim();
const truthy = (v) => v === true || ["TRUE", "1", "Y", "YES"].includes(pup(v));

/* Python's round(x, 2): the correctly rounded value of the exact binary x, ties
   to even. toFixed already rounds the exact binary value; it differs from
   Python only on an EXACT tie, and a positive double sits exactly on a
   2-decimal tie only when 8x is an odd integer (.125 / .375 / .625 / .875). */
function pyRound2(x) {
  const e = x * 8;
  if (Number.isInteger(e) && Math.abs(e % 2) === 1) {
    const lo = Math.floor(x * 100);
    return (lo % 2 === 0 ? lo : lo + 1) / 100;
  }
  return Number(x.toFixed(2));
}

function etNow(now) {
  const d = now || new Date();
  const parts = {};
  new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York", year: "numeric",
    month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23" })
    .formatToParts(d).forEach((p) => { parts[p.type] = p.value; });
  return { date: `${parts.year}-${parts.month}-${parts.day}`, hm: `${parts.hour}:${parts.minute}` };
}

const pitchSide = (leg) => {
  const a = pup(leg.Action);
  return a === "BUY" ? "BUY" : a === "SELL_SHORT" ? "SELL" : null;
};
const legDir = (leg) => (pitchSide(leg) === "BUY" ? 1 : -1);

// LIMIT_CLOSE | LIMIT_OPEN | MOO | MOC | "" (unmappable)
function legKind(leg) {
  const kind = pup(leg.Entry_Type);
  if (kind === "LIMIT") {
    const anchor = pup(leg.Entry_Anchor);
    return anchor === "CLOSE" || anchor === "OPEN" ? `LIMIT_${anchor}` : "";
  }
  return kind === "MOO" || kind === "MOC" ? kind : "";
}

/* pitch_moo.price_open_row, line for line. */
function priceOpenLeg(leg, sessionOpen) {
  const atr = pnum(leg.ATR), off = pnum(leg.Entry_Offset_ATR);
  if (!(sessionOpen > 0) || atr == null || off == null) return null;
  const dir = legDir(leg);
  const limit = pyRound2(sessionOpen + off * atr);
  const stopAtr = pnum(leg.Stop_ATR), tgtAtr = pnum(leg.Target_ATR);
  return {
    limit,
    stop: stopAtr ? pyRound2(limit - dir * stopAtr * atr) : null,
    target: tgtAtr ? pyRound2(limit + dir * tgtAtr * atr) : null,
  };
}

const timeStopAt = (leg) => (pup(leg.Time_Exit_Order) === "MOO" ? "open" : "close");
const legPass = (leg, c) => String(leg.Place_Pass || (c && c.placePass) || "").toLowerCase();

/* What the Execution ticket receives for one leg. `ctx` = { now: Date, open }. */
function stagePlan(leg, ctx) {
  const c = ctx || {};
  const kind = legKind(leg);
  const notes = [];
  const plan = { kind, type: null, entry: null, stop: null, target: null, exp: "",
                 ts: String(leg.Time_Exit_Date || ""), tsat: timeStopAt(leg), stopArm: null,
                 pass: legPass(leg, c), notes, derived: false };
  const gtd = pup(leg.TIF) === "GTD";
  if (kind === "LIMIT_CLOSE") {
    plan.type = pup(leg.Order_Type);
    plan.entry = pnum(leg.Limit_Price);
    plan.stop = pnum(leg.Stop_Price);
    plan.target = pnum(leg.Target_Price);
    if (gtd) plan.exp = String(leg.Entry_Expire_Date || "");
  } else if (kind === "LIMIT_OPEN") {
    plan.type = pup(leg.Order_Type);
    const priced = priceOpenLeg(leg, pnum(c.open));
    if (priced) {
      plan.entry = priced.limit; plan.stop = priced.stop; plan.target = priced.target;
      plan.derived = true;
      notes.push(`limit/stop/target priced off the open you entered (${pnum(c.open)}) with the ` +
        `pitch's own rule (pitch_moo.price_open_row)`);
    }
    if (gtd) plan.exp = String(leg.Entry_Expire_Date || "");
  } else if (kind === "MOO" || kind === "MOC") {
    plan.type = kind;
    plan.entry = pnum(leg.Ref_Close);
    notes.push(`${kind}: entry ${ppx(plan.entry)} is the reference close, sent only as the ` +
      `ticket's risk reference; the order itself is an unpriced ${kind}`);
  }
  if (plan.stop != null) {
    plan.stopArm = "next_session";
    notes.push("stop arms the next session, as pitched");
  }
  if (plan.ts) notes.push(`time exit ${plan.ts} at the ${plan.tsat === "open" ? "OPEN (MOO)" : "close (MOC)"}`);
  return plan;
}

/* A leg is stageable only if the ticket can express it exactly. Anything else
   is shown with the reason, never a half-filled ticket.
   `ctx` = { date, standDown, now: Date, open, placePass }. */
function stageBlockers(leg, ctx) {
  const c = ctx || {};
  const et = etNow(c.now);
  const out = [];
  const kind = legKind(leg);
  const otype = pup(leg.Order_Type), tif = pup(leg.TIF);
  if (c.standDown) out.push("stand-down day: nothing was pitched");
  const dated = String(leg.Execute_On || c.date || "");
  if (dated !== et.date) out.push(`pitch is for ${dated || "?"}, today is ${et.date}`);
  if (leg.Time_Exit_Date && String(leg.Time_Exit_Date) <= et.date)
    out.push(`time exit ${leg.Time_Exit_Date} is today or past`);
  if (truthy(leg.Manual_Only))
    out.push(`manual per pitch${leg.Place_Note ? ` (${leg.Place_Note})` : ""}`);
  if (pup(leg.Proxy_Ticker))
    out.push(`proxy leg (${pup(leg.Proxy_Ticker)} for ${pup(leg.Ticker)}): the ticket would trade a different instrument`);
  if (pup(leg.Sec_Type || "STK") !== "STK")
    out.push(`${pup(leg.Sec_Type)} leg${leg.Contract ? ` (${leg.Contract})` : ""}: enter by hand`);
  if (!pitchSide(leg)) out.push(`action ${leg.Action || "?"} has no ticket equivalent`);
  if (!(pnum(leg.Quantity) > 0)) out.push("no share count");
  const exitOrder = pup(leg.Time_Exit_Order);
  if (exitOrder !== "MOO" && exitOrder !== "MOC")
    out.push(`time exit order ${exitOrder || "?"} has no ticket equivalent`);
  const pass = legPass(leg, c);
  const after = PITCH_PASS_AFTER[pass];
  if (!after && !truthy(leg.Manual_Only)) out.push(`no pitch_moo pass (${pass || "none"})`);
  else if (after && pitchArmed(c) && et.hm < after)
    out.push(`wait until pitch_moo's ${pass} pass has run (${after} ET)`);
  if (kind === "LIMIT_CLOSE" || kind === "LIMIT_OPEN") {
    if (otype !== "LMT") out.push(`order type ${otype || "?"} is not a limit`);
    if (tif !== "DAY" && tif !== "GTD") out.push(`TIF ${tif || "?"} has no ticket equivalent`);
    if (tif === "GTD" && !leg.Entry_Expire_Date) out.push("GTD limit with no expiry date");
  }
  if (kind === "LIMIT_CLOSE") {
    if (!(pnum(leg.Limit_Price) > 0)) out.push("no limit price");
    if (pnum(leg.Stop_ATR) && pnum(leg.Stop_Price) == null) out.push("pitched stop has no price");
    if (pnum(leg.Target_ATR) && pnum(leg.Target_Price) == null) out.push("pitched target has no price");
  } else if (kind === "LIMIT_OPEN") {
    if (et.hm < PITCH_SESSION_OPEN) out.push(`open-anchored: the session opens ${PITCH_SESSION_OPEN} ET`);
    if (!(pnum(c.open) > 0)) out.push("open-anchored: enter today's session open");
    if (pnum(leg.ATR) == null || pnum(leg.Entry_Offset_ATR) == null) out.push("no ATR / offset to price from");
  } else if (kind === "MOO" || kind === "MOC") {
    const want = kind === "MOO" ? "OPG" : "MOC";
    if (otype !== "MKT" || tif !== want) out.push(`${kind} row is ${otype || "?"}/${tif || "?"}, not MKT/${want}`);
    if (pnum(leg.Stop_ATR) || pnum(leg.Target_ATR) || pnum(leg.Stop_Price) != null || pnum(leg.Target_Price) != null)
      out.push(`${kind} with a price stop/target: the pitch leaves it unpriced`);
    if (!(pnum(leg.Ref_Close) > 0)) out.push("no reference close for the ticket's risk math");
    if (kind === "MOO" && et.hm >= PITCH_OPG_CUTOFF) out.push(`MOO past the ${PITCH_OPG_CUTOFF} ET auction cutoff`);
    if (kind === "MOC" && et.hm >= PITCH_MOC_CUTOFF) out.push(`MOC past the ${PITCH_MOC_CUTOFF} ET cutoff`);
  } else {
    out.push(`entry type ${leg.Entry_Type || "?"} has no ticket equivalent`);
  }
  return out;
}

function stageHref(leg, idea, pitchDate, ctx) {
  const plan = stagePlan(leg, ctx);
  const q = new URLSearchParams({
    stage: "pitch", sym: pup(leg.Ticker), side: pitchSide(leg) || "",
    type: plan.type || "", entry: plan.entry == null ? "" : String(plan.entry),
    strat: `Pitch-${idea.idea_id}`, refdate: pitchDate || "", acct: PITCH_ACCOUNT,
    kind: plan.kind, tsat: plan.tsat, pass: plan.pass, armed: pitchArmed(ctx) ? "1" : "0",
  });
  if (plan.stop != null) q.set("stop", String(plan.stop));
  if (plan.target != null) q.set("target", String(plan.target));
  if (pnum(leg.Quantity) != null) q.set("qty", String(leg.Quantity));
  if (plan.exp) q.set("exp", plan.exp);
  if (plan.ts) q.set("ts", plan.ts);
  return `execution.html?${q.toString()}`;
}

function pitchedEntry(leg) {
  const kind = pup(leg.Entry_Type);
  if (kind !== "LIMIT") return pesc(kind);
  const off = pnum(leg.Entry_Offset_ATR);
  const k = off == null ? "?" : `${off >= 0 ? "+" : "-"}${Math.abs(off)}`;
  const px = pnum(leg.Limit_Price) != null ? ` = ${ppx(leg.Limit_Price)}` : "";
  return `LMT ${pesc(leg.Entry_Anchor)} ${k} ATR${px} <span class="cap" style="display:inline">${pesc(leg.TIF)}${
    pup(leg.TIF) === "GTD" ? " " + pesc(leg.Entry_Expire_Date) : ""}</span>`;
}

function legKey(idea, leg) { return `${idea.idea_id}|${leg.Leg}`; }

function legCells(idea, leg, pitchDate, ctx) {
  const plan = stagePlan(leg, ctx);
  const blockers = stageBlockers(leg, ctx);
  const staged = plan.type && !blockers.length
    ? `${plan.type} ${ppx(plan.entry)}${plan.exp ? ` <span class="cap" style="display:inline">GTD ${pesc(plan.exp)}</span>` : ""}`
    : "-";
  const action = blockers.length
    ? `<span class="cap" style="color:#ffc14d">cannot stage: ${blockers.map(pesc).join("; ")}</span>`
    : `<a class="btn" href="${pesc(stageHref(leg, idea, pitchDate, ctx))}">Stage &rarr;</a>`;
  return { plan, html: `<td>${pesc(leg.Leg)}</td><td class="l"><b>${pesc(leg.Ticker)}</b></td>
    <td class="l">${pesc(leg.Action)}</td><td class="l">${pitchedEntry(leg)}</td>
    <td class="l">${staged}</td><td>${pesc(leg.Quantity)}</td>
    <td>${ppx(plan.stop)}</td><td>${ppx(plan.target)}</td>
    <td class="l">${pesc(leg.Time_Exit_Date)} ${pesc(leg.Time_Exit_Order)}</td>
    <td class="l">${action}</td>` };
}

function legRows(idea, pitchDate, opens, now) {
  return (idea.orders || []).map((leg) => {
    const key = legKey(idea, leg);
    const ctx = { date: pitchDate, now, open: opens[key], placePass: idea.place_pass };
    const { plan, html } = legCells(idea, leg, pitchDate, ctx);
    const openInput = plan.kind === "LIMIT_OPEN" ? `<label class="cap" style="display:inline">Today's open
      <input data-open-key="${pesc(key)}" value="${opens[key] == null ? "" : pesc(opens[key])}"
      style="width:80px" inputmode="decimal"></label>` : "";
    const notes = plan.notes.length || openInput
      ? `<tr><td></td><td class="l" colspan="9" style="white-space:normal">${openInput}
          ${plan.notes.map((n) => `<span class="cap" style="display:block">${pesc(n)}</span>`).join("")}</td></tr>`
      : "";
    return `<tr data-leg-key="${pesc(key)}">${html}</tr>${notes}`;
  }).join("");
}

function ideaCard(idea, pitchDate, opens, now) {
  const ev = idea.evidence || {};
  const legs = idea.orders || [];
  const multi = legs.length > 1
    ? `<div class="radar-warn">${legs.length}-leg idea: stage EVERY leg, one ticket each. A single staged
       leg is a different trade from the one pitched.</div>` : "";
  const after = PITCH_PASS_AFTER[String(idea.place_pass || "").toLowerCase()];
  const never = `Stage here OR approve Y in the Sheet, never both: pitch_moo cannot see a site-staged order.`;
  const passNote = !PITCH_MOO_ARMED
    ? `<p class="cap">${never}<br>pitch_moo runner is off — the Sheet Y places nothing.</p>`
    : after
      ? `<p class="cap">Staging opens at ${after} ET, after pitch_moo's ${pesc(idea.place_pass)} pass has run.
         ${never}</p>` : "";
  return `<div class="card radar-card" data-idea="${pesc(idea.idea_id)}">
    <div class="radar-head">
      <b>#${pesc(idea.rank)} ${pesc(idea.title)}</b>
      <span class="radar-pill">${pesc(idea.grade || "?")}</span>
      <span class="cap">${pesc(idea.horizon_td)} td</span>
      <span class="cap">evidence N=${pesc(ev.n == null ? "?" : ev.n)}</span>
      <span class="cap">pitch_moo pass: ${pesc(idea.place_pass || "-")}</span>
      <span class="cap">tag Pitch-${pesc(idea.idea_id)}</span>
    </div>
    <p>${pesc(idea.thesis)}</p>
    ${ev.summary ? `<p class="cap">Evidence: ${pesc(ev.summary)}</p>` : ""}
    ${idea.survived ? `<p class="cap">Survived: ${pesc(idea.survived)}</p>` : ""}
    <p class="cap">What kills it: ${pesc(idea.what_kills_it)}</p>
    ${passNote}
    ${multi}
    <div style="overflow-x:auto"><table class="tbl"><thead><tr>
      <th>Leg</th><th class="l">Ticker</th><th class="l">Action</th><th class="l">Pitched entry</th>
      <th class="l">Staged entry</th><th>Qty</th><th>Stop</th><th>Target</th>
      <th class="l">Time exit</th><th class="l">Stage</th>
    </tr></thead><tbody>${legRows(idea, pitchDate, opens, now)}</tbody></table></div>
  </div>`;
}

const pitchOpens = {};   // leg key -> session open typed on the card (this view only)
let pitchPayload = null;

function render(payload) {
  const el = document.getElementById("content");
  const now = new Date();
  const et = etNow(now);
  const av = pnum(payload.account_value);
  const bits = [`pitch <b>${pesc(payload.date)}</b>`, `published ${pesc(payload.generated_at)}`,
    `checked ${et.hm} ET`];
  const banners = [];
  if (payload.date !== et.date)
    banners.push(`<div class="radar-warn">This pitch is dated ${pesc(payload.date)}; today is ${et.date}.
      Nothing on it can be staged. The next pitch publishes ~05:30 ET on a trading morning.</div>`);
  if (payload.stand_down)
    banners.push(`<div class="radar-warn"><b>Stand-down:</b> nothing survived this morning.
      ${pesc(payload.stand_down_reason)}</div>`);
  const ideas = payload.ideas || [];
  el.innerHTML = `
    <p class="cap">${bits.join(" &nbsp;&middot;&nbsp; ")}</p>
    <p class="cap">Share counts are sized off the fixed <b>ACCOUNT_VALUE</b> basis${
      av ? ` (<b>$${av.toLocaleString()}</b>)` : ""}, not live NLV, and stage to the <b>primary</b>
      account only. Each staged ticket is tagged <code>Pitch-&lt;idea_id&gt;</code>, the same
      strategy tag <code>pitch_moo.py</code> stamps. Legs stage exactly as pitched or not at all.</p>
    ${banners.join("")}
    ${ideas.length ? ideas.map((i) => ideaCard(i, payload.date, pitchOpens, now)).join("")
      : payload.stand_down ? "" : `<p class="cap">No ideas in this payload.</p>`}`;
  el.querySelectorAll("[data-open-key]").forEach((inp) => inp.addEventListener("change", () => {
    const v = pnum(inp.value);
    if (v > 0) pitchOpens[inp.dataset.openKey] = v; else delete pitchOpens[inp.dataset.openKey];
    render(payload);
  }));
}

async function main() {
  renderNav("pitch.html");
  const el = document.getElementById("content");
  try {
    const payload = await fetchJSON(PITCH_ENDPOINT);
    if (payload && payload.error) throw new Error(payload.error);
    pitchPayload = payload;
    render(payload);
  } catch (e) {
    el.innerHTML = `<div class="radar-warn">Could not load today's pitch: ${pesc(e.message || e)}.
      It is published by <code>daily_pitch.py</code> after the morning email.</div>`;
    return;
  }
  // Every gate is clock-dependent: re-render so a card left open never shows a
  // live Stage button past its window.
  const typing = () => { const a = document.activeElement; return !!(a && a.dataset && a.dataset.openKey); };
  setInterval(() => { if (pitchPayload && !typing()) render(pitchPayload); }, PITCH_RERENDER_MS);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && pitchPayload) render(pitchPayload);
  });
}

if (typeof document !== "undefined") document.addEventListener("DOMContentLoaded", main);
if (typeof module !== "undefined") module.exports = {
  stageHref, stageBlockers, stagePlan, priceOpenLeg, pyRound2, etNow, pitchSide, legKind,
  PITCH_PASS_AFTER, PITCH_OPG_CUTOFF, PITCH_MOC_CUTOFF, PITCH_MOO_ARMED,
};
