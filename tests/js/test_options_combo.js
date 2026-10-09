"use strict";

/* Options Custom spread builder: multi-structure option_spread payloads and risk.
   Expected numbers are HARD-CODED from the Python reference
   (trading_ibkr option_combo_risk.evaluate; OPTION_COMBO_SPEC.md) so this test
   does not need Python at runtime. Regenerate by calling evaluate() with the
   cases below (spot 103.7 for the 1x2 call ratio, 100 for the short call/strangle). */

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const context = {
  console, URLSearchParams, setTimeout, clearTimeout, setInterval, clearInterval,
  document: { addEventListener() {}, getElementById() { return null; } },
  window: {}, location: { search: "" },
};
vm.createContext(context);
for (const file of ["bsm.js", "options.js", "options-custom.js"]) {
  const src = fs.readFileSync(path.join(__dirname, "..", "..", "site", "assets", file), "utf8");
  vm.runInContext(src, context, { filename: file });
}
const C = context;
const plain = (x) => JSON.parse(JSON.stringify(x));
const E = "20261120", F = "20261218";
const W = (side, right, strike, ratio = 1, expiry = E) => ({ side, right, strike, ratio, expiry });
const close = (a, b, msg) => assert.ok(Math.abs(a - b) < 1e-6, `${msg}: ${a} vs ${b}`);

// ---- risk numbers vs the Python reference:
// [action, limit, qty, wire legs, spot, unit_loss_usd, unit_risk, risk_usd, unbounded, stress_spot]
const REF = {
  iron_condor: ["SELL", 1.5, 3, [W("SELL", "P", 90), W("BUY", "P", 95), W("BUY", "C", 105), W("SELL", "C", 110)], null, 350.0, 3.5, 1065.6, false, null],
  put_ratio_1x2: ["BUY", 1.0, 2, [W("BUY", "P", 100), W("SELL", "P", 90, 2)], null, 8100.0, 81.0, 16207.8, false, null],
  call_ratio_1x2: ["BUY", 1.0, 2, [W("BUY", "C", 100), W("SELL", "C", 110, 2)], 103.7, 1581.0, 15.81, 3169.8, true, 134.81],
  short_put: ["SELL", 2.0, 5, [W("BUY", "P", 100)], null, 9800.0, 98.0, 49006.5, false, null],
  short_call: ["SELL", 1.0, 1, [W("BUY", "C", 105)], 100.0, 2400.0, 24.0, 2401.3, true, 130.0],
  short_strangle: ["SELL", 3.0, 1, [W("BUY", "P", 95), W("BUY", "C", 105)], 100.0, 9200.0, 92.0, 9202.6, true, 130.0],
  long_straddle: ["BUY", 5.0, 4, [W("BUY", "C", 100), W("BUY", "P", 100)], null, 500.0, 5.0, 2010.4, false, null],
  butterfly: ["BUY", 1.5, 10, [W("BUY", "C", 95), W("SELL", "C", 100, 2), W("BUY", "C", 105)], null, 150.0, 1.5, 1552.0, false, null],
  call_calendar: ["BUY", 1.2, 3, [W("BUY", "C", 100, 1, F), W("SELL", "C", 100, 1, E)], null, 120.0, 1.2, 367.8, false, null],
  put_diagonal_ratio: ["BUY", 2.0, 2, [W("BUY", "P", 105, 2, F), W("SELL", "P", 100, 1, E)], null, 200.0, 2.0, 407.8, false, null],
};
for (const [name, [action, limit, qty, legs, spot, unitLoss, unitRisk, riskUsd, unb, stress]] of Object.entries(REF)) {
  if (unb) {
    const noAck = C.comboEvaluate(action, limit, qty, legs, { spot });
    assert.ok(noAck.needsAck && /UNBOUNDED_ACK_REQUIRED/.test(noAck.error), name + " needs ack");
  }
  const r = C.comboEvaluate(action, limit, qty, legs, { spot, unboundedAck: true });
  assert.ok(!r.error, name + ": " + r.error);
  const i = r.info;
  close(i.unitLoss, unitLoss, name + " unit_loss");
  close(i.unitRisk, unitRisk, name + " unit_risk");
  close(i.riskUsd, riskUsd, name + " risk_usd");
  assert.strictEqual(i.unbounded, unb, name + " unbounded");
  if (stress != null) close(i.stressSpot, stress, name + " stress");
  assert.strictEqual(i.debitRisk, Math.ceil(unitRisk * 100 - 1e-9) / 100, name + " debit_risk");
}

// ---- rejections the spec lists (each with its reason)
const rej = (action, limit, legs, re, opts) => {
  const r = C.comboEvaluate(action, limit, 1, legs, opts || {});
  assert.ok(r.error && re.test(r.error), `expected ${re}: ${JSON.stringify(plain(r))}`);
};
rej("BUY", 1.2, [W("BUY", "C", 105, 1, F), W("SELL", "C", 100, 1, E)], /cover the short/);          // uncovered calendar
rej("SELL", 1.2, [W("BUY", "C", 100, 1, F), W("SELL", "C", 100, 1, E)], /later expiry/);            // credit calendar
rej("BUY", 1.2, [W("BUY", "C", 100, 1, E), W("SELL", "C", 100, 1, F)], /later expiry/);             // reverse calendar
rej("BUY", 1.2, [W("BUY", "C", 100, 1, F), W("SELL", "P", 100, 1, E)], /share one option right/);   // mixed-right calendar
rej("BUY", 1, [W("BUY", "C", 90), W("SELL", "C", 95), W("SELL", "C", 100), W("BUY", "C", 105), W("BUY", "C", 110)], /1 to 4 legs/);
rej("BUY", 1, [W("BUY", "C", 100), W("SELL", "C", 100)], /duplicate/);
rej("BUY", 1, [W("BUY", "C", 100, 0), W("SELL", "C", 110)], /ratio/);
rej("BUY", 1, [W("BUY", "C", 100, 1.5), W("SELL", "C", 110)], /ratio/);
rej("BUY", 1, [W("SELL", "C", 100)], /single option leg/);
rej("BUY", 1, [W("BUY", "C", 100, 2)], /single option leg/);
rej("BUY", 1, [W("BUY", "C", 100), W("SELL", "C", 110, 1, E), W("SELL", "C", 120, 1, F)], /only as a 2-leg calendar/);
rej("SELL", 1, [W("BUY", "C", 105)], /underlying price/, { unboundedAck: true });                    // unbounded without a spot
rej("SELL", 100, [W("BUY", "C", 100), W("SELL", "C", 110)], /(never profit|no downside)/);

// ---- builder: chain fixture covering both expiries
const row = (right, strike, bid, ask, con_id, expiry) => ({ right, strike, bid, ask, mid: (bid + ask) / 2, con_id, expiry });
const chain = { expiry: E, dte: 30, strikes: [
  row("P", 90, 0.45, 0.55, 90, E), row("P", 95, 1.45, 1.55, 95, E),
  row("C", 100, 3.9, 4.1, 100, E), row("C", 105, 1.55, 1.65, 105, E), row("C", 110, 0.35, 0.45, 110, E)] };
const farRow = row("C", 100, 5.1, 5.3, 1100, F);
const L = (side, right, strike, ratio = 1, extra = {}) => ({ side, right, strike, ratio, ...extra });
const P = (struct, opts) => C.buildOptionSpreadPayload({ struct, symbol: "SPY", expiry: E, qty: 3, limit: 1.5, tif: "DAY", params: {}, ...opts });

// credit iron condor: SELL parent, legs written as the BUY-ticket view, debit_risk = unit_loss/100
const condor = C.ocBuildStruct([L("BUY", "P", 90), L("SELL", "P", 95), L("SELL", "C", 105), L("BUY", "C", 110)], chain);
assert.ok(condor.credit && condor.tradeable, condor.execution_issue);
const cb = P(condor);
assert.ok(!cb.error, cb.error);
assert.deepStrictEqual(plain(cb.payload), {
  symbol: "SPY", action: "SELL", quantity: 3, limit: 1.5, tif: "DAY", structure: "custom",
  debit_risk: 3.5, risk_per_unit: 3.5, credit: true,
  legs: [
    { side: "SELL", right: "P", expiry: E, strike: 90, ratio: 1, con_id: 90 },
    { side: "BUY", right: "P", expiry: E, strike: 95, ratio: 1, con_id: 95 },
    { side: "BUY", right: "C", expiry: E, strike: 105, ratio: 1, con_id: 105 },
    { side: "SELL", right: "C", expiry: E, strike: 110, ratio: 1, con_id: 110 }],
  strategy: null, signal_date: null, entry_condition: null,
});
assert.ok(!("unbounded_ack" in cb.payload) && !("underlying_spot" in cb.payload));
for (const qty of [1, 500, 1000000]) assert.ok(!P(condor, { qty }).error);          // no quantity cap

// covered debit calendar: BUY parent, real legs, two expiries, debit_risk = debit
const cal = C.ocBuildStruct([L("BUY", "C", 100, 1, { expiry: F, row: farRow }), L("SELL", "C", 100, 1, { expiry: E })], chain);
assert.ok(!cal.credit && cal.tradeable, cal.execution_issue);
assert.ok(cal.width == null);
const cp = P(cal, { limit: 1.2 });
assert.ok(!cp.error, cp.error);
assert.strictEqual(cp.payload.action, "BUY"); assert.strictEqual(cp.payload.debit_risk, 1.2);
assert.deepStrictEqual(plain(cp.payload.legs), [
  { side: "BUY", right: "C", expiry: F, strike: 100, ratio: 1, con_id: 1100 },
  { side: "SELL", right: "C", expiry: E, strike: 100, ratio: 1, con_id: 100 }]);
// uncovered calendar is refused with the reason, in the struct and in the builder
const unc = C.ocBuildStruct([L("BUY", "C", 105, 1, { expiry: F, row: { ...farRow, strike: 105, con_id: 1105 } }), L("SELL", "C", 100, 1, { expiry: E })], chain);
assert.ok(unc.execution_issue && /cover the short/.test(unc.execution_issue));
assert.ok(P(unc, { limit: 1.2 }).error);

// unbounded 1x2 call ratio: needs the ack, spot recorded, debit_risk rounded up to the cent
const ratio = C.ocBuildStruct([L("BUY", "C", 100), L("SELL", "C", 110, 2)], chain);
assert.ok(ratio.tradeable && !ratio.credit && ratio.width == null);
const noAck = P(ratio, { limit: 1.0, qty: 2, spot: 103.7 });
assert.ok(noAck.needsAck && /UNBOUNDED_ACK_REQUIRED/.test(noAck.error) && !noAck.payload);
const ub = P(ratio, { limit: 1.0, qty: 2, spot: 103.7, unboundedAck: true });
assert.ok(!ub.error, ub.error);
assert.strictEqual(ub.payload.unbounded_ack, true); assert.strictEqual(ub.payload.underlying_spot, 103.7);
assert.strictEqual(ub.payload.debit_risk, 15.81); assert.strictEqual(ub.payload.action, "BUY");
assert.deepStrictEqual(plain(ub.payload.legs.map((l) => [l.side, l.ratio])), [["BUY", 1], ["SELL", 2]]);
// a higher (conservative) spot only raises the claim
assert.ok(P(ratio, { limit: 1.0, qty: 2, spot: 110, unboundedAck: true }).payload.debit_risk > 15.81);
// short single call (naked, unbounded)
const shortC = C.ocBuildStruct([L("SELL", "C", 105)], chain);
assert.ok(shortC.credit && shortC.tradeable);
const sc = P(shortC, { limit: 1.0, qty: 1, spot: 100, unboundedAck: true }).payload;
assert.strictEqual(sc.action, "SELL"); assert.strictEqual(sc.debit_risk, 24);
assert.deepStrictEqual(plain(sc.legs.map((l) => l.side)), ["BUY"]);

// legacy shapes unchanged: long single and 1:1 vertical keep the original debit_risk basis
const lsing = P(C.ocBuildStruct([L("BUY", "C", 100)], chain), { limit: 3.95 });
assert.strictEqual(lsing.payload.debit_risk, 3.95);
const vert = P(C.ocBuildStruct([L("BUY", "C", 100), L("SELL", "C", 105)], chain), { limit: 2.3 });
assert.strictEqual(vert.payload.debit_risk, 2.3); assert.ok(!("unbounded_ack" in vert.payload));

// stats helper: stress loss + total for the full quantity
const stats = C.ocComputeStats([L("BUY", "C", 100), L("SELL", "C", 110, 2)], chain, 2, 1.0, null, 103.7);
assert.ok(stats.unbounded && stats.totalMaxLoss == null);
close(stats.info.unitLoss, 1581, "stats unit loss"); close(stats.totalStress, 3169.8, "stats total stress");
const calStats = C.ocComputeStats([L("BUY", "C", 100, 1, { expiry: F, row: farRow }), L("SELL", "C", 100, 1, { expiry: E })], chain, 3, 1.2, null, 100);
assert.ok(calStats.multiExpiry); close(calStats.totalMaxLoss, 367.8, "calendar total risk");

// conservative spot = max of every underlying figure held
assert.strictEqual(C.ocSpotUsed({ spot: 100, last: 101.2, mid: 100.5 }, { spot: 99 }, 100.1), 101.2);
assert.strictEqual(C.ocSpotUsed(null, null, null), null);

console.log("test_options_combo ok");
