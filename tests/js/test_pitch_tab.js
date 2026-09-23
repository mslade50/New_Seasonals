"use strict";

/* Pitch tab: stage EXACTLY what the pitch says or block. Covers every entry
   mapping, every clock gate at its boundary, Manual_Only / proxy blocks,
   pitch_moo pass gating, open-anchored pricing parity with
   pitch_moo.price_open_row, the stage link round-trip, the prefill-time
   re-checks in execution.js, and the pitch-only payload fields (radar payloads
   byte-identical). */

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const ASSETS = path.join(__dirname, "..", "..", "site", "assets");
// 2026-09-23 is EDT (UTC-4).
const AT = (hm, day = "23") => new Date(`2026-09-${day}T${hm}:00-04:00`);

function loadPitch() {
  const source = fs.readFileSync(path.join(ASSETS, "pitch.js"), "utf8");
  const context = { console, Date, Intl, document: { addEventListener() {} }, window: {},
                    location: { search: "" }, URLSearchParams, module: { exports: {} } };
  vm.createContext(context);
  vm.runInContext(source, context, { filename: "pitch.js" });
  return context.module.exports;
}

function loadExecution(search, fields) {
  const source = fs.readFileSync(path.join(ASSETS, "execution.js"), "utf8");
  const f = fields || {};
  const context = { console, Date, Intl, window: {}, location: { search }, URLSearchParams,
                    document: { addEventListener() {}, querySelectorAll: () => [],
                                getElementById: (id) => (id in f ? f[id] : null) },
                    setTimeout, clearTimeout, setInterval, clearInterval };
  vm.createContext(context);
  vm.runInContext(source, context, { filename: "execution.js" });
  context.__fields = f;
  return context;
}
const q = (href) => href.slice("execution.html".length);
const runJSON = (ctx, code) => JSON.parse(vm.runInContext(`JSON.stringify(${code})`, ctx));
const stagedFrom = (href, name = "pitchStage") => runJSON(loadExecution(q(href)), name);

const P = loadPitch();
const IDEA = { idea_id: "2026-09-23-1", rank: 1, title: "t", grade: "B" };
const base = { Idea_Id: "2026-09-23-1", Leg: 1, Sec_Type: "STK", Contract: "", Proxy_Ticker: "",
  Execute_On: "2026-09-23", Time_Exit_Date: "2026-09-30", Time_Exit_Order: "MOC",
  Entry_Expire_Date: "2026-09-23", Stop_ATR: "", Target_ATR: "", Stop_Price: "",
  Target_Price: "", Limit_Price: "", Entry_Anchor: "", Entry_Offset_ATR: "",
  Manual_Only: false, Place_Pass: "auction" };
const CLOSE_LMT = { ...base, Ticker: "XLE", Action: "BUY", Entry_Type: "LIMIT",
  Entry_Anchor: "CLOSE", Entry_Offset_ATR: -0.5, Order_Type: "LMT", TIF: "GTD",
  Entry_Expire_Date: "2026-09-24", Limit_Price: 98.75, Stop_Price: 95.75,
  Target_Price: 104.75, Stop_ATR: 1.2, Target_ATR: 2.4, Quantity: 250, Ref_Close: 100.0, ATR: 2.5 };
const OPEN_LMT = { ...base, Ticker: "GLD", Action: "SELL_SHORT", Entry_Type: "LIMIT",
  Entry_Anchor: "OPEN", Entry_Offset_ATR: 0.25, Order_Type: "LMT", TIF: "DAY", Place_Pass: "open",
  Stop_ATR: 1.0, Target_ATR: 2.0, Quantity: 300, Ref_Close: 40.5, ATR: 0.5 };
const MOO = { ...base, Ticker: "TLT", Action: "BUY", Entry_Type: "MOO", Order_Type: "MKT",
  TIF: "OPG", Quantity: 120, Ref_Close: 200.0, ATR: 4.0, Time_Exit_Order: "MOO" };
const MOC = { ...base, Ticker: "IWM", Action: "SELL_SHORT", Entry_Type: "MOC", Order_Type: "MKT",
  TIF: "MOC", Quantity: 80, Ref_Close: 220.0, ATR: 3.0 };
const MANUAL_MOO = { ...MOO, Stop_ATR: 1.5, Place_Pass: "manual", Manual_Only: true,
  Place_Note: "MOO entry with a price stop/target" };
const FUT = { ...MOO, Ticker: "DX", Sec_Type: "FUT", Contract: "202612", Place_Pass: "manual",
  Manual_Only: true, Place_Note: "futures leg" };
const ctx = (hm, extra) => ({ date: "2026-09-23", now: AT(hm), ...(extra || {}) });
const blockers = (leg, c) => P.stageBlockers(leg, c);
const has = (leg, c, s) => blockers(leg, c).some((b) => b.includes(s));
const clear = (leg, c, why) => assert.strictEqual(blockers(leg, c).length, 0, why || JSON.stringify(blockers(leg, c)));

// --- Python round() parity: values and expectations from CPython 3 round(x, 2) --
const PY = [[40.125, 40.12], [40.375, 40.38], [100.625, 100.62], [12.875, 12.88], [0.125, 0.12],
  [98.765, 98.77], [101.2345, 101.23], [57.005, 57.01], [33.3349999, 33.33], [1.005, 1.0],
  [2.675, 2.67], [149.995, 150.0], [76.4449, 76.44], [3.14159, 3.14], [250.0, 250.0]];
for (const [x, want] of PY) assert.strictEqual(P.pyRound2(x), want, `round(${x}, 2)`);

// --- ET clock ----------------------------------------------------------------------
assert.strictEqual(P.etNow(AT("09:27")).hm, "09:27");
assert.strictEqual(P.etNow(new Date("2026-09-24T02:30:00Z")).date, "2026-09-23",
  "22:30 ET is still the 23rd");

// --- open-anchored pricing == pitch_moo.price_open_row (values from CPython) --------
let o = P.priceOpenLeg(OPEN_LMT, 40.0);
assert.deepStrictEqual([o.limit, o.stop, o.target], [40.12, 40.62, 39.12]);
o = P.priceOpenLeg({ ...OPEN_LMT, Action: "BUY", ATR: 1.37, Entry_Offset_ATR: -0.3,
  Stop_ATR: 1.2, Target_ATR: 2.5 }, 87.41);
assert.deepStrictEqual([o.limit, o.stop, o.target], [87.0, 85.36, 90.42]);
o = P.priceOpenLeg({ ...OPEN_LMT, Stop_ATR: "", Target_ATR: "" }, 40.0);
assert.strictEqual(o.stop, null); assert.strictEqual(o.target, null);

// --- generic blockers ----------------------------------------------------------------
clear(CLOSE_LMT, ctx("10:00"));
assert.ok(has(CLOSE_LMT, { ...ctx("10:00"), now: AT("10:00", "24") }, "pitch is for 2026-09-23"),
  "a stale pitch blocks");
assert.ok(has({ ...CLOSE_LMT, Execute_On: undefined }, { ...ctx("10:00"), date: "2026-09-22" }, "pitch is for 2026-09-22"));
assert.ok(has({ ...CLOSE_LMT, Time_Exit_Date: "2026-09-23" }, ctx("10:00"), "today or past"));
assert.ok(has({ ...CLOSE_LMT, Quantity: 0 }, ctx("10:00"), "no share count"));
assert.ok(has({ ...CLOSE_LMT, Quantity: "" }, ctx("10:00"), "no share count"));
assert.ok(has(CLOSE_LMT, ctx("10:00", { standDown: true }), "stand-down"));
assert.ok(has({ ...CLOSE_LMT, Action: "SELL" }, ctx("10:00"), "action SELL has no ticket equivalent"));
assert.ok(has({ ...CLOSE_LMT, Entry_Type: "VWAP" }, ctx("10:00"), "no ticket equivalent"));
assert.ok(has({ ...CLOSE_LMT, Time_Exit_Order: "LOC" }, ctx("10:00"), "time exit order LOC"));

// Manual_Only: every form the publisher / Sheet can produce, and the reason is shown.
assert.ok(has(MANUAL_MOO, ctx("09:10"), "manual per pitch (MOO entry with a price stop/target)"));
assert.ok(has(FUT, ctx("09:10"), "manual per pitch (futures leg)"));
assert.ok(has(FUT, ctx("09:10"), "FUT leg (202612)"));
for (const flag of [true, "TRUE", "True", 1, "1"])
  assert.ok(has({ ...CLOSE_LMT, Manual_Only: flag }, ctx("10:00"), "manual per pitch"), `Manual_Only=${flag}`);
for (const flag of [false, "", "FALSE", 0])
  assert.ok(!has({ ...CLOSE_LMT, Manual_Only: flag }, ctx("10:00"), "manual"), `Manual_Only=${flag}`);
// A MOO/MOC that somehow carries a price stop but no Manual_Only flag still blocks.
assert.ok(has({ ...MOO, Stop_ATR: 1.5 }, ctx("09:10"), "the pitch leaves it unpriced"));
assert.ok(has({ ...MOC, Target_Price: 210 }, ctx("12:00"), "the pitch leaves it unpriced"));

// Proxy: the ticket would trade a different instrument.
assert.ok(has({ ...CLOSE_LMT, Proxy_Ticker: "uso" }, ctx("10:00"), "proxy leg (USO for XLE)"));
assert.ok(has({ ...MOC, Proxy_Ticker: "SPY" }, ctx("12:00"), "proxy leg"));

// Order type / TIF verbatim.
assert.ok(has({ ...CLOSE_LMT, Order_Type: "MKT" }, ctx("10:00"), "not a limit"));
assert.ok(has({ ...CLOSE_LMT, TIF: "GTC" }, ctx("10:00"), "TIF GTC"));
assert.ok(has({ ...CLOSE_LMT, Entry_Expire_Date: "" }, ctx("10:00"), "GTD limit with no expiry"));
assert.ok(has({ ...MOO, TIF: "DAY" }, ctx("09:10"), "not MKT/OPG"));
assert.ok(has({ ...MOC, TIF: "DAY" }, ctx("12:00"), "not MKT/MOC"));
assert.ok(has({ ...CLOSE_LMT, Limit_Price: "" }, ctx("10:00"), "no limit price"));
assert.ok(has({ ...CLOSE_LMT, Stop_Price: "" }, ctx("10:00"), "pitched stop has no price"));
assert.ok(has({ ...MOO, Ref_Close: "" }, ctx("09:10"), "no reference close"));

// --- clock gates, each at its boundary --------------------------------------------------
assert.strictEqual(P.PITCH_MOO_ARMED, false, "runner is not armed: tasks unregistered, no flag");
const armed = (hm, extra) => ctx(hm, { armed: true, ...(extra || {}) });
// ARMED: auction pass (09:05): a close-anchored limit is stageable only after it.
assert.ok(has(CLOSE_LMT, armed("09:04"), "wait until pitch_moo's auction pass has run (09:05 ET)"));
clear(CLOSE_LMT, armed("09:05"));
// ARMED MOO: auction pass 09:05 .. OPG cutoff 09:25 is the whole window.
assert.ok(has(MOO, armed("09:04"), "auction pass"));
clear(MOO, armed("09:05"));
clear(MOO, armed("09:24"));
assert.ok(has(MOO, armed("09:25"), "MOO past the 09:25 ET auction cutoff"));
assert.ok(has(MOO, armed("11:00"), "MOO past the 09:25"), "no MKT substitute after the cutoff");
// ARMED MOC: 09:05 .. 15:30.
assert.ok(has(MOC, armed("09:04"), "auction pass"));
clear(MOC, armed("09:05"));
clear(MOC, armed("15:29"));
assert.ok(has(MOC, armed("15:30"), "MOC past the 15:30 ET cutoff"));
// ARMED open-anchored: open pass at 09:32, and it needs the typed open.
assert.ok(has(OPEN_LMT, armed("09:31", { open: 40 }), "wait until pitch_moo's open pass has run (09:32 ET)"));
assert.ok(has(OPEN_LMT, armed("09:10", { open: 40 }), "session opens 09:30"));
clear(OPEN_LMT, armed("09:32", { open: 40 }));
assert.ok(has(OPEN_LMT, armed("10:00"), "enter today's session open"));
// UNARMED (the default): no pass wait; every other clock gate stays.
clear(MOO, ctx("08:00"), "unarmed auction MOO is stageable at 08:00");
assert.ok(has(MOO, ctx("09:26"), "MOO past the 09:25 ET auction cutoff"));
assert.ok(!has(MOO, ctx("08:00"), "wait until"));
clear(CLOSE_LMT, ctx("08:00"));
clear(MOC, ctx("08:00"));
assert.ok(has(MOC, ctx("15:30"), "MOC past the 15:30 ET cutoff"));
clear(OPEN_LMT, ctx("09:30", { open: 40 }), "unarmed open-anchored needs only the 09:30 open");
assert.ok(has(OPEN_LMT, ctx("09:29", { open: 40 }), "session opens 09:30"));
assert.ok(has(MANUAL_MOO, ctx("08:00"), "manual per pitch"), "Manual_Only still blocks unarmed");
assert.ok(has({ ...MOO, Proxy_Ticker: "SPY" }, ctx("08:00"), "proxy leg"), "proxy still blocks unarmed");
assert.ok(has(MOO, { ...ctx("08:00"), now: AT("08:00", "24") }, "pitch is for 2026-09-23"));
// A leg with no pass (and not manual) is refused; the idea's pass is the fallback.
assert.ok(has({ ...CLOSE_LMT, Place_Pass: "" }, ctx("10:00"), "no pitch_moo pass"));
clear({ ...CLOSE_LMT, Place_Pass: "" }, ctx("10:00", { placePass: "auction" }));

// --- entry mapping ---------------------------------------------------------------------
let plan = P.stagePlan(CLOSE_LMT, ctx("10:00"));
assert.deepStrictEqual([plan.kind, plan.type, plan.entry, plan.stop, plan.target, plan.exp, plan.ts],
  ["LIMIT_CLOSE", "LMT", 98.75, 95.75, 104.75, "2026-09-24", "2026-09-30"]);
assert.strictEqual(plan.derived, false, "a close-anchored limit is verbatim");
assert.deepStrictEqual([plan.stopArm, plan.tsat, plan.pass], ["next_session", "close", "auction"]);
assert.strictEqual(P.stagePlan({ ...CLOSE_LMT, TIF: "DAY" }, ctx("10:00")).exp, "",
  "a DAY limit carries no expiry");
assert.strictEqual(P.stagePlan({ ...CLOSE_LMT, Stop_Price: "", Stop_ATR: "" }, ctx("10:00")).stopArm, null,
  "no stop, no stop_arm");

plan = P.stagePlan(OPEN_LMT, ctx("09:45", { open: 40 }));
assert.deepStrictEqual([plan.type, plan.entry, plan.stop, plan.target], ["LMT", 40.12, 40.62, 39.12]);
assert.ok(plan.derived);
assert.strictEqual(P.stagePlan(OPEN_LMT, ctx("09:45")).entry, null, "no open typed, nothing priced");

plan = P.stagePlan(MOO, ctx("09:10"));
assert.deepStrictEqual([plan.type, plan.entry, plan.stop, plan.target, plan.tsat, plan.stopArm],
  ["MOO", 200.0, null, null, "open", null]);
assert.strictEqual(P.stagePlan(MOO, ctx("11:00")).type, "MOO", "never re-typed MKT");
assert.ok(plan.notes.some((n) => n.includes("risk reference")));
plan = P.stagePlan(MANUAL_MOO, ctx("09:10"));
assert.deepStrictEqual([plan.stop, plan.target], [null, null], "no reference-close levels, ever");

plan = P.stagePlan(MOC, ctx("12:00"));
assert.deepStrictEqual([plan.type, plan.entry, plan.stop, plan.tsat], ["MOC", 220.0, null, "close"]);

// --- round trip through execution.js ---------------------------------------------------
let s = stagedFrom(P.stageHref(CLOSE_LMT, IDEA, "2026-09-23", ctx("10:00")));
assert.ok(s, "execution.js must recognise the pitch stage link");
assert.strictEqual(s.sym, "XLE"); assert.strictEqual(s.side, "BUY"); assert.strictEqual(s.type, "LMT");
assert.strictEqual(s.entry, 98.75); assert.strictEqual(s.stop, 95.75); assert.strictEqual(s.target, 104.75);
assert.strictEqual(s.qty, 250); assert.strictEqual(s.exp, "2026-09-24"); assert.strictEqual(s.ts, "2026-09-30");
assert.strictEqual(s.strat, "Pitch-2026-09-23-1", "same tag pitch_moo stamps");
assert.ok(/^[A-Za-z0-9 _.-]{1,32}$/.test(s.strat), "tag passes the ticket's strategy rule");
assert.deepStrictEqual([s.refdate, s.acct, s.kind, s.tsat, s.pass],
  ["2026-09-23", "primary", "LIMIT_CLOSE", "close", "auction"]);
assert.strictEqual(s.cap, null); assert.strictEqual(s.soFrac, null);

s = stagedFrom(P.stageHref(MOO, IDEA, "2026-09-23", ctx("09:10")));
assert.deepStrictEqual([s.type, s.entry, s.stop, s.target, s.exp, s.tsat], ["MOO", 200, null, null, "", "open"]);
s = stagedFrom(P.stageHref(OPEN_LMT, IDEA, "2026-09-23", ctx("09:45", { open: 40 })));
assert.deepStrictEqual([s.side, s.type, s.entry, s.stop, s.target, s.kind, s.pass],
  ["SELL", "LMT", 40.12, 40.62, 39.12, "LIMIT_OPEN", "open"]);
s = stagedFrom(P.stageHref(MOC, IDEA, "2026-09-23", ctx("12:00")));
assert.deepStrictEqual([s.side, s.type, s.qty, s.kind], ["SELL", "MOC", 80, "MOC"]);

// The pitch and radar parsers never pick up each other's links; a type the pitch
// cannot produce (incl. the retired MKT substitute) or a kind/type mismatch is refused.
const pitchLink = P.stageHref(CLOSE_LMT, IDEA, "2026-09-23", ctx("10:00"));
assert.strictEqual(stagedFrom(pitchLink, "radarStage"), null);
assert.strictEqual(stagedFrom("execution.html?stage=radar&sym=AMG&entry=383.12&type=STP_LMT"), null);
assert.strictEqual(stagedFrom(pitchLink.replace("type=LMT", "type=STP_LMT")), null);
assert.strictEqual(stagedFrom(pitchLink.replace("type=LMT", "type=MKT")), null, "no MKT substitute");
assert.strictEqual(stagedFrom(pitchLink.replace("type=LMT", "type=MOO")), null, "kind/type mismatch");
assert.strictEqual(stagedFrom(pitchLink.replace("tsat=close", "tsat=")), null, "time_stop_at required");

// --- prefill-time re-checks -------------------------------------------------------------
function refusal(href, hm, day) {
  const ex = loadExecution(q(href));
  return vm.runInContext("(now) => pitchPrefillRefusal(pitchStage, now)", ex)(AT(hm, day));
}
const mooLink = P.stageHref(MOO, IDEA, "2026-09-23", ctx("09:10"));
const mocLink = P.stageHref(MOC, IDEA, "2026-09-23", ctx("12:00"));
const openLink = P.stageHref(OPEN_LMT, IDEA, "2026-09-23", ctx("09:45", { open: 40 }));
const armedLink = P.stageHref(CLOSE_LMT, IDEA, "2026-09-23", armed("10:00"));
const armedMoo = P.stageHref(MOO, IDEA, "2026-09-23", armed("09:10"));
const armedOpen = P.stageHref(OPEN_LMT, IDEA, "2026-09-23", armed("09:45", { open: 40 }));
assert.ok(pitchLink.includes("armed=0") && armedLink.includes("armed=1"), "link carries the armed state");
assert.strictEqual(refusal(pitchLink, "10:00"), null);
assert.ok(refusal(pitchLink, "10:00", "24").includes("pitch link is for 2026-09-23, today is 2026-09-24"),
  "a stale refdate is refused at prefill, not just render");
assert.ok(refusal(pitchLink.replace("refdate=2026-09-23", "refdate="), "10:00").includes("pitch link is for ?"));
// ARMED: current pass gating.
assert.ok(refusal(armedLink, "09:04").includes("auction pass has run (09:05 ET)"));
assert.ok(refusal(armedLink, "09:04").includes("don't also approve Y"));
assert.strictEqual(refusal(armedMoo, "09:05"), null);
assert.strictEqual(refusal(armedMoo, "09:24"), null);
assert.ok(refusal(armedMoo, "09:25").includes("MOO past the 09:25"));
assert.ok(refusal(armedMoo, "09:04").includes("auction pass"));
assert.ok(refusal(armedOpen, "09:31").includes("open pass has run (09:32 ET)"));
assert.strictEqual(refusal(armedOpen, "09:32"), null);
// UNARMED (armed=0): no pass wait, every other gate stays.
assert.strictEqual(refusal(pitchLink, "09:04"), null);
assert.strictEqual(refusal(mooLink, "08:00"), null, "unarmed auction MOO prefills at 08:00");
assert.ok(refusal(mooLink, "09:26").includes("MOO past the 09:25"));
assert.strictEqual(refusal(mocLink, "15:29"), null);
assert.ok(refusal(mocLink, "15:30").includes("MOC past the 15:30"));
assert.strictEqual(refusal(openLink, "09:30"), null);
assert.ok(refusal(openLink, "09:29").includes("before the 09:30 ET open"));
assert.ok(refusal(pitchLink.replace("pass=auction", "pass=manual"), "10:00").includes("no pitch_moo pass"));
// Absent or unrecognised armed param fails CLOSED to the pass gate.
for (const bad of ["", "&armed=", "&armed=yes"]) {
  const link = mooLink.replace("&armed=0", bad);
  assert.ok(refusal(link, "08:00").includes("auction pass has run (09:05 ET)"), `armed param ${bad || "absent"}`);
  assert.strictEqual(refusal(link, "09:05"), null);
}

// A refused prefill fills NOTHING and says why.
{
  const msg = { textContent: "" }, cmdType = { value: "echo" };
  const ex = loadExecution(q(mooLink), { cmdMsg: msg, cmdType });
  vm.runInContext("(now) => applyPitchPrefill(now)", ex)(AT("09:30"));
  assert.ok(msg.textContent.startsWith("NOT prefilled from Daily Pitch: MOO past the 09:25"), msg.textContent);
  assert.strictEqual(cmdType.value, "echo", "the ticket is untouched");
  assert.strictEqual(runJSON(ex, "pitchTicket"), null);
}
// An accepted prefill arms the pitch conventions for that symbol.
{
  const ex = loadExecution(q(mooLink), { cmdMsg: { textContent: "" } });
  const seen = [];
  vm.runInContext("(f) => { applyVerbatimPrefill = f; }", ex)((r, m) => seen.push(m));
  vm.runInContext("(now) => applyPitchPrefill(now)", ex)(AT("09:10"));
  assert.deepStrictEqual(runJSON(ex, "pitchTicket"), { sym: "TLT", timeStopAt: "open" });
  assert.ok(seen[0].includes("exactly as pitched") && seen[0].includes("time exit at the OPEN"), seen[0]);
}

// --- payload fields: pitch-only, radar byte-identical ------------------------------------
const TICKET = (sym, extra) => {
  const f = { f_sectype: "STK", f_symbol: sym, f_action: "BUY", f_qty: "10", f_entry_type: "LMT",
    f_entry: "50", f_stop: "48", f_target: "54", f_timestop: "2026-09-30", f_expiry: "",
    f_strategy: "", f_so_frac: "", f_so_target: "", ...(extra || {}) };
  return Object.fromEntries(Object.entries(f).map(([k, v]) => [k, { value: v }]));
};
{
  const ex = loadExecution(q(mooLink), TICKET("TLT"));
  vm.runInContext("pitchTicket = { sym: 'TLT', timeStopAt: 'open' }", ex);
  let p = runJSON(ex, 'ticketPayload("entry_bracket")');
  assert.strictEqual(p.stop_arm, "next_session");
  assert.strictEqual(p.time_stop_at, "open");
  ex.__fields.f_stop.value = "";
  p = runJSON(ex, 'ticketPayload("entry_bracket")');
  assert.ok(!("stop_arm" in p), "no stop, no stop_arm");
  assert.strictEqual(p.time_stop_at, "open");
  ex.__fields.f_timestop.value = "";
  assert.ok(!("time_stop_at" in runJSON(ex, 'ticketPayload("entry_bracket")')), "no time stop, no time_stop_at");
  ex.__fields.f_symbol.value = "SPY";
  ex.__fields.f_stop.value = "48"; ex.__fields.f_timestop.value = "2026-09-30";
  p = runJSON(ex, 'ticketPayload("entry_bracket")');
  assert.ok(!("stop_arm" in p) && !("time_stop_at" in p), "a retyped symbol drops the pitch conventions");
}
{
  const RADAR = "?stage=radar&sym=AMG&side=BUY&type=STP_LMT&entry=383.12&cap=400.17&stop=355.84" +
    "&qty=33&exp=2026-08-28&ts=2026-11-13&strat=Momentum_Radar&refdate=2026-08-16&acct=primary";
  const ex = loadExecution(RADAR, TICKET("AMG", { f_entry_type: "STP_LMT", f_entry: "383.12",
    f_entry_cap: "400.17", f_stop: "355.84", f_target: "", f_qty: "33", f_expiry: "2026-08-28",
    f_timestop: "2026-11-13", f_strategy: "Momentum_Radar" }));
  assert.strictEqual(runJSON(ex, "pitchTicket"), null);
  const want = { symbol: "AMG", sec_type: "STK", currency: "USD", fut_expiry: null, exchange: null,
    fut_ib_symbol: null, fut_trading_class: null, fut_multiplier: null, fut_min_tick: null,
    action: "BUY", quantity: 33, entry_type: "STP_LMT", entry: 383.12, stop: 355.84, target: null,
    entry_cap: 400.17, strategy: "Momentum_Radar", scaleout: null, time_stop: "2026-11-13",
    expiry: "2026-08-28" };
  assert.strictEqual(JSON.stringify(runJSON(ex, 'ticketPayload("entry_bracket")')), JSON.stringify(want),
    "radar payload is byte-identical: no stop_arm / time_stop_at");
}

console.log("PASS pitch tab: exact-or-block mapping, clock gates, prefill re-checks, payload fields");
