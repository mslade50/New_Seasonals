"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const context = {
  console,
  document: { addEventListener() {}, getElementById() { return null; } },
  window: {},
  location: { search: "" },
  URLSearchParams,
  setTimeout,
  clearTimeout,
  setInterval,
  clearInterval,
};
for (const file of ["bsm.js", "options.js"]) {
  const source = fs.readFileSync(path.join(__dirname, "..", "..", "site", "assets", file), "utf8");
  vm.runInNewContext(source, context, { filename: file });
}

assert.deepStrictEqual(
  JSON.parse(JSON.stringify(context.assessVolRegime({ pctile: 12, vrp: -0.18 }))).label,
  "CHEAP",
);
assert.strictEqual(context.assessVolRegime({ pctile: 92, vrp: 0.45 }).label, "RICH");
assert.match(context.shapeGuidance("big_move", {
  pctile: 12, vrp: -0.18, termShape: "contango", rr25Pts: 1,
}).shape, /straddle|strangle/i);
assert.match(context.shapeGuidance("bullish", {
  pctile: 90, vrp: 0.35, termShape: "backwardation", rr25Pts: 5,
}).shape, /spread/i);

assert.strictEqual(context.snapNetLimit(1.53, "BUY"), 1.50);
assert.strictEqual(context.snapNetLimit(1.53, "SELL"), 1.55);

const shortPut = { strike: 100, right: "P", con_id: 1, bid: 1.90, ask: 2.10, mid: 2.00,
  delta: -0.30, gamma: 0.02, theta: -0.04, vega: 0.08 };
const longPut = { strike: 95, right: "P", con_id: 2, bid: 0.70, ask: 0.90, mid: 0.80,
  delta: -0.15, gamma: 0.01, theta: -0.02, vega: 0.04 };
const credit = context.structureFrom("bull put", [
  { side: "SELL", row: shortPut }, { side: "BUY", row: longPut },
], "", { credit: true, width: 5, category: "credit_vertical" });
assert.strictEqual(credit.mid, 1.20);
assert.strictEqual(credit.nat, 1.00);
assert.strictEqual(context.riskPerUnit(credit), 382.60);
assert.deepStrictEqual(
  JSON.parse(JSON.stringify(context.canonicalPayloadLegs(credit, "20260918", "SELL").map((l) => l.side))),
  ["BUY", "SELL"],
);

const single = context.structureFrom("long put", [{ side: "BUY", row: shortPut }], "", { category: "single" });
assert.strictEqual(single.mid, 2.00);
assert.strictEqual(context.riskPerUnit(single), 201.30);

const cutoff = new Date();
cutoff.setDate(cutoff.getDate() + 30);
const cutoffIso = `${cutoff.getFullYear()}-${String(cutoff.getMonth() + 1).padStart(2, "0")}-${String(cutoff.getDate()).padStart(2, "0")}`;
vm.runInNewContext("state.wb = {spot: 100, chain: {dte: 60}}; state.pricing = {rate: 0.04, divYield: 0, ivShiftPts: 0};", context);
const forecast = context.forecastMetrics(single, 2, {
  event: "touch", target: 90, probability: 0.50, cutoff: cutoffIso,
  touchIvShift: 10, noTouchSpot: 100, noTouchIvShift: -3,
});
assert.ok(Number.isFinite(forecast.ev));
assert.strictEqual(forecast.risk, context.riskPerUnit(single) * 2);
assert.strictEqual(
  context.forecastScore(forecast, "ev_risk"),
  forecast.ev / forecast.risk,
);
assert.ok(forecast.touchWorst <= forecast.touchBest);

console.log("PASS options lab regime, guidance, pricing, defined-risk sizing, credit orientation, and forecast scoring");
