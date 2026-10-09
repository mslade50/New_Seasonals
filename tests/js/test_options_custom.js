"use strict";

/* Options tab Custom spread builder: payoff math, legs URL contract, no size
   cap, and the option_spread payload staying identical to the shootout ticket. */

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
const plain = (x) => JSON.parse(JSON.stringify(x));
const C = context;
const run = (code) => vm.runInContext(code, context);

const L = (side, right, strike, ratio = 1) => ({ side, right, strike, ratio });

// ---- payoff math
let p = C.ocPayoff([L("BUY", "C", 100), L("SELL", "C", 110)], 4);          // bull call vertical, debit 4
assert.deepStrictEqual(plain(p), { maxLoss: 4, maxGain: 6, lossUnbounded: false, gainUnbounded: false, tailSlope: 0, breakevens: [104] });
p = C.ocPayoff([L("BUY", "P", 90), L("SELL", "P", 95), L("SELL", "C", 105), L("BUY", "C", 110)], -2);   // iron condor, credit 2
assert.strictEqual(p.maxLoss, 3); assert.strictEqual(p.maxGain, 2);
assert.deepStrictEqual(plain(p.breakevens), [93, 107]);
assert.ok(!p.lossUnbounded && !p.gainUnbounded);
p = C.ocPayoff([L("BUY", "C", 100), L("SELL", "C", 110, 2)], 1);           // 1x2 call ratio, unbounded short tail
assert.ok(p.lossUnbounded && p.maxLoss === null && !p.gainUnbounded);
assert.strictEqual(p.maxGain, 9);
assert.deepStrictEqual(plain(p.breakevens), [101, 119]);
p = C.ocPayoff([L("BUY", "C", 100)], 3);                                    // long call: unbounded gain
assert.ok(p.gainUnbounded && p.maxGain === null && p.maxLoss === 3);
p = C.ocPayoff([L("BUY", "P", 100)], 3);                                    // long put: bounded both ways
assert.strictEqual(p.maxGain, 97); assert.deepStrictEqual(plain(p.breakevens), [97]);

// ---- legs URL parsing (a literal plus survives as a space in a decoded query string)
const legs = plain(C.ocParseLegs("P:748:2026-11-30: 1,P:720:20261130:-1,C:5:2026-11-30:+2,X:1:2026-11-30:1,P:1:bad:1"));
assert.deepStrictEqual(legs.map((l) => [l.right, l.strike, l.expiry, l.side, l.ratio]), [
  ["P", 748, "20261130", "BUY", 1], ["P", 720, "20261130", "SELL", 1], ["C", 5, "20261130", "BUY", 2]]);
const q = new URLSearchParams("ticker=SPY&legs=P:748:2026-11-30:%2B1,P:720:2026-11-30:-1&qty=3&limit=3.20");
assert.strictEqual(C.ocParseLegs(q.get("legs")).length, 2);
const q2 = new URLSearchParams("legs=P:748:2026-11-30:+1,P:720:2026-11-30:-1");
assert.strictEqual(C.ocParseLegs(q2.get("legs")).length, 2);
assert.strictEqual(C.ocLegsParam([{ right: "P", strike: 748, expiry: "20261130", qty: 1 }, { right: "P", strike: 720, expiry: "2026-11-30", qty: -1 }]),
  "P:748:2026-11-30:%2B1,P:720:2026-11-30:-1");
const pp = plain(C.parseParams.call(null));
assert.ok("legs" in pp && "qty" in pp && "limit" in pp);

// ---- chain fixture, struct, no qty cap, payload identical to the shootout ticket
const row = (right, strike, bid, ask, con_id) => ({ right, strike, bid, ask, mid: (bid + ask) / 2, con_id, delta: 0.3 });
const chain = { expiry: "20261130", dte: 52, strikes: [
  row("P", 748, 11.0, 11.4, 1748), row("P", 720, 7.6, 8.0, 1720), row("C", 800, 3.0, 3.2, 2800)] };
const st = C.ocBuildStruct([L("BUY", "P", 748), L("SELL", "P", 720)], chain);
assert.strictEqual(st.mid, 3.4); assert.strictEqual(st.width, 28); assert.ok(!st.credit && st.tradeable);

const shootStruct = { name: "Custom", legs: [{ side: "BUY", row: chain.strikes[0] }, { side: "SELL", row: chain.strikes[1] }],
  mid: 3.4, nat: 3.8, credit: false, width: 28, category: "debit_vertical" };
for (const qty of [1, 3, 500, 100000]) {
  const a = C.buildOptionSpreadPayload({ struct: st, symbol: "SPY", expiry: chain.expiry, qty, limit: 3.2, tif: "DAY", params: {} });
  const b = C.buildOptionSpreadPayload({ struct: shootStruct, symbol: "SPY", expiry: chain.expiry, qty, limit: 3.2, tif: "DAY", params: {} });
  assert.ok(!a.error, a.error);
  assert.deepStrictEqual(plain(a.payload), plain(b.payload));
  assert.strictEqual(a.payload.quantity, qty);
}
const pay = C.buildOptionSpreadPayload({ struct: st, symbol: "SPY", expiry: "20261130", qty: 500, limit: 3.2, tif: "GTC", params: {} }).payload;
assert.deepStrictEqual(Object.keys(pay), ["symbol", "action", "quantity", "limit", "tif", "structure", "debit_risk", "risk_per_unit",
  "credit", "legs", "strategy", "signal_date", "entry_condition"]);
assert.strictEqual(pay.debit_risk, 3.2); assert.strictEqual(pay.action, "BUY");
assert.deepStrictEqual(plain(pay.legs), [
  { side: "BUY", right: "P", expiry: "20261130", strike: 748, ratio: 1, con_id: 1748 },
  { side: "SELL", right: "P", expiry: "20261130", strike: 720, ratio: 1, con_id: 1720 }]);

// credit vertical: sell 748 / buy 720 -> action SELL, legs flipped, debit_risk = width - credit
const cr = C.ocBuildStruct([L("SELL", "P", 748), L("BUY", "P", 720)], chain);
assert.ok(cr.credit);
const cp = C.buildOptionSpreadPayload({ struct: cr, symbol: "SPY", expiry: "20261130", qty: 2, limit: 3.4, tif: "DAY", params: {} }).payload;
assert.strictEqual(cp.action, "SELL"); assert.strictEqual(cp.debit_risk, 24.6);
assert.deepStrictEqual(plain(cp.legs.map((l) => l.side)), ["BUY", "SELL"]);

// total max loss for the full quantity is information, not a cap
const stats = C.ocComputeStats([L("BUY", "P", 748), L("SELL", "P", 720)], chain, 500, 3.2);
assert.ok(Math.abs(stats.totalMaxLoss - (3.2 * 100 + stats.comm) * 500) < 1e-6);

// executor caps (OPTION_COMBO_SPEC.md): ratios, short singles, 3-4 legs are sendable; the spec's rejections still show a reason
assert.strictEqual(C.ocBuildStruct([L("SELL", "C", 800)], chain).execution_issue, null);
assert.strictEqual(C.ocBuildStruct([L("BUY", "P", 748, 2), L("SELL", "P", 720)], chain).execution_issue, null);
assert.strictEqual(C.ocBuildStruct([L("BUY", "P", 748), L("SELL", "P", 720), L("SELL", "C", 800)], chain).execution_issue, null);
assert.strictEqual(C.ocBuildStruct([L("BUY", "P", 748), L("SELL", "P", 700)], chain).missing.length, 1);

// expiry list prefers all_expiries; centre honoured check
assert.deepStrictEqual(plain(C.ocExpiryList({ all_expiries: [{ expiry: "20261201", dte: 3 }, { expiry: "20261130", dte: 2 }], expiries: [{ date: "20261130", dte: 2 }] })),
  [{ date: "20261130", dte: 2 }, { date: "20261201", dte: 3 }]);
assert.deepStrictEqual(plain(C.ocExpiryList({ expiries: [{ date: "20261130", dte: 2 }] })), [{ date: "20261130", dte: 2 }]);
assert.ok(C.ocCenterHonored(chain, 730) && !C.ocCenterHonored(chain, 400) && C.ocCenterHonored(chain, null));

// leg add semantics
const ll = [];
C.ocAddLeg(ll, "C", 100, "BUY"); C.ocAddLeg(ll, "C", 100, "BUY"); assert.strictEqual(ll[0].ratio, 2);
C.ocAddLeg(ll, "C", 100, "SELL"); assert.strictEqual(ll[0].side, "SELL"); assert.strictEqual(ll[0].ratio, 1);
for (const k of [101, 102, 103]) assert.ok(C.ocAddLeg(ll, "C", k, "BUY"));
assert.ok(!C.ocAddLeg(ll, "C", 104, "BUY"));

// server side: workbench forwards strike_center, exec-command carries no qty/risk cap
const wbSrc = fs.readFileSync(path.join(__dirname, "..", "..", "functions", "exec-workbench.js"), "utf8");
assert.ok(wbSrc.includes("strike_center"));
const cmdSrc = fs.readFileSync(path.join(__dirname, "..", "..", "functions", "exec-command.js"), "utf8");
assert.ok(!/quantity|\bqty\b|risk_cap|max_risk/i.test(cmdSrc.replace(/\/\*[\s\S]*?\*\//, "")), "exec-command adds no qty/risk caps");
// page wiring
const html = fs.readFileSync(path.join(__dirname, "..", "..", "site", "options.html"), "utf8");
assert.ok(html.indexOf("assets/options.js") < html.indexOf("assets/options-custom.js"));
console.log("test_options_custom ok");
