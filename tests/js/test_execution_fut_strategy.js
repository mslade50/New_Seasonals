"use strict";
// Hedge panel attribution: OpenBreakout MES contracts are strategy exposure,
// not a hedge; untagged MES still counts; Legend_EMA stock attributes through
// the generic STK path. Display-only model, no commands.
const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const context = { console, window: {}, location: { search: "" }, URLSearchParams,
  document: { addEventListener() {}, getElementById() { return null; } } };
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(__dirname, "../../site/assets/execution.js"), "utf8"), context);

const specs = { MES: { multiplier: 5 } };
const betas = { tickers: { SPY: { beta252: 1, beta63: 1 }, QQQ: { beta252: 1.2, beta63: 1.2 } } };
const ob = (id, type, extra) => Object.assign({
  order_id: id, symbol: "MES", sec_type: "FUT", expiry: "20261218", action: "SELL", qty: 1,
  status: "Submitted", order_type: type, oca_group: "ob-1", parent_id: 0,
  order_ref: "MES|BUY|OpenBreakout|2026-09-28|ES-1-STOP",
}, extra || {});

function book(positions, orders) {
  return { key: "primary", positions, orders };
}

// 1. One tagged MES (stop + timed exit legs) and one untagged MES: two contracts
//    held in one position. One is OpenBreakout, the other still counts as hedge.
{
  const orders = [
    ob(11, "STP", { aux: 7700 }),
    ob(12, "MKT", { good_after: "20260928 15:55:00 America/New_York",
      order_ref: "MES|BUY|OpenBreakout|2026-09-28|ES-1-TIME" }),
  ];
  const model = context.attributeBook(book([
    { symbol: "MES", sec_type: "FUT", expiry: "20261218", position: 2, market_price: 7800 },
  ], orders), betas, specs, { today: "20260928" });
  const strategyRow = model.futures.find(f => f.strategy === "OpenBreakout");
  const hedgeRow = model.futures.find(f => f.counted);
  assert.ok(strategyRow, "tagged MES must be attributed to OpenBreakout");
  assert.strictEqual(strategyRow.position, 1);
  assert.strictEqual(strategyRow.counted, false);
  assert.ok(hedgeRow, "untagged remainder must still count as hedge");
  assert.strictEqual(hedgeRow.position, 1);
  assert.strictEqual(model.futuresSpyEquiv, 1 * 5 * 7800);
  const ob1 = model.byStrategy.find(r => r.strategy === "OpenBreakout");
  assert.ok(ob1, "OpenBreakout appears under its strategy");
  assert.strictEqual(ob1.legs, 1);
  assert.strictEqual(ob1.spyEquiv, 5 * 7800);
  assert.strictEqual(ob1.legDetails[0].exitDate, "20260928");
  assert.strictEqual(ob1.legDetails[0].secType, "FUT");
  // Default hedge scope (OLV only) excludes the OpenBreakout contract.
  const target = context.hedgeTarget(model, 0, { multiplier: 5, indexLevel: 7800 });
  assert.strictEqual(target.currentDollars, 5 * 7800);
}

// 2. Fully tagged position: nothing counts as hedge.
{
  const model = context.attributeBook(book([
    { symbol: "MES", sec_type: "FUT", expiry: "20261218", position: 1, market_price: 7800 },
  ], [ob(21, "STP")]), betas, specs, { today: "20260928" });
  assert.strictEqual(model.futuresSpyEquiv, 0);
  assert.strictEqual(model.futures.filter(f => f.counted).length, 0);
  assert.strictEqual(model.futures.length, 1);
  assert.strictEqual(model.futures[0].strategy, "OpenBreakout");
}

// 3. Untagged MES position (and a tagged leg on a different expiry) stays hedge.
{
  const model = context.attributeBook(book([
    { symbol: "MES", sec_type: "FUT", expiry: "20261218", position: -3, market_price: 7800 },
  ], [ob(31, "STP", { action: "BUY", expiry: "20270319",
    order_ref: "MES|SELL|OpenBreakout|2026-09-28|ES-1-STOP" })]), betas, specs, { today: "20260928" });
  assert.strictEqual(model.futures.length, 1);
  assert.strictEqual(model.futures[0].counted, true);
  assert.strictEqual(model.futuresSpyEquiv, -3 * 5 * 7800);
  assert.strictEqual(model.byStrategy.length, 0);
}

// 4. Unknown tag on a futures leg does not carve anything out; a catalog tag
//    passed through opts.strategyTags does.
{
  const orders = [ob(41, "STP", { order_ref: "MES|BUY|ManualHedge|2026-09-28" })];
  const pos = [{ symbol: "MES", sec_type: "FUT", expiry: "20261218", position: 1, market_price: 7800 }];
  const plain = context.attributeBook(book(pos, orders), betas, specs, { today: "20260928" });
  assert.strictEqual(plain.futuresSpyEquiv, 5 * 7800);
  const tagged = context.attributeBook(book(pos, orders), betas, specs,
    { today: "20260928", strategyTags: ["ManualHedge"] });
  assert.strictEqual(tagged.futuresSpyEquiv, 0);
  assert.strictEqual(tagged.futures[0].strategy, "ManualHedge");
}

// 5. Legend_EMA stock attributes through the generic STK path (timed exit leg
//    carries the entry-side ref, as legend_ema.py places it).
{
  const ref = "SPY|BUY|Legend_EMA|2026-09-24|TIME";
  const model = context.attributeBook(book([
    { symbol: "SPY", sec_type: "STK", position: 1, market_price: 700 },
  ], [
    { order_id: 51, symbol: "SPY", sec_type: "STK", action: "SELL", qty: 1, status: "Submitted",
      order_type: "MKT", good_after: "20260929 15:59:00 America/New_York", order_ref: ref, oca_group: "leg-1" },
    { order_id: 52, symbol: "SPY", sec_type: "STK", action: "SELL", qty: 1, status: "Submitted",
      order_type: "STP", aux: 680, order_ref: "SPY|BUY|Legend_EMA|2026-09-24|STOP", oca_group: "leg-1" },
  ]), betas, specs, { today: "20260928" });
  const legend = model.byStrategy.find(r => r.strategy === "Legend_EMA");
  assert.ok(legend, "Legend_EMA stock must attribute");
  assert.strictEqual(legend.legs, 1);
  assert.strictEqual(legend.notionalLong, 700);
  assert.ok(!model.byStrategy.some(r => r.strategy === "Unattributed"));
}

// 6. Wrong side: a short MES hedge plus a working OpenBreakout LONG entry
//    (BUY order, ref side BUY). The BUY is the closing side of the short, but
//    the ref says the strategy position is long, so nothing is carved out.
{
  const model = context.attributeBook(book([
    { symbol: "MES", sec_type: "FUT", expiry: "20261218", position: -2, market_price: 7800 },
  ], [ob(61, "STP", { action: "BUY", aux: 7850,
    order_ref: "MES|BUY|OpenBreakout|2026-09-28|ES-1-ENTRY" })]), betas, specs, { today: "20260928" });
  assert.strictEqual(model.futures.length, 1);
  assert.strictEqual(model.futures[0].counted, true);
  assert.strictEqual(model.futures[0].position, -2);
  assert.strictEqual(model.byStrategy.length, 0);
}

// 7. Wrong expiry: an order with no expiry cannot claim a dated position, and
//    conId decides when both sides carry one.
{
  const pos = [{ symbol: "MES", sec_type: "FUT", expiry: "20261218", con_id: 111, position: 1, market_price: 7800 }];
  const noExpiry = context.attributeBook(book(pos, [ob(71, "STP", { expiry: "" })]),
    betas, specs, { today: "20260928" });
  assert.strictEqual(noExpiry.futuresSpyEquiv, 5 * 7800);
  assert.ok(!noExpiry.futures.some(f => f.strategy));
  const otherCon = context.attributeBook(book(pos, [ob(72, "STP", { con_id: 222 })]),
    betas, specs, { today: "20260928" });
  assert.strictEqual(otherCon.futuresSpyEquiv, 5 * 7800);
  const sameCon = context.attributeBook(book(pos, [ob(73, "STP", { con_id: 111 })]),
    betas, specs, { today: "20260928" });
  assert.strictEqual(sameCon.futuresSpyEquiv, 0);
  assert.strictEqual(sameCon.futures[0].strategy, "OpenBreakout");
}

// 8. No mark: a claimed contract with no price renders zero notional, not NaN.
{
  const model = context.attributeBook(book([
    { symbol: "MES", sec_type: "FUT", expiry: "20261218", position: 1 },
  ], [ob(81, "STP")]), betas, specs, { today: "20260928" });
  const row = model.byStrategy.find(r => r.strategy === "OpenBreakout");
  assert.ok(row);
  assert.ok(Number.isFinite(row.spyEquiv) && Number.isFinite(row.notionalLong));
  assert.ok(Number.isFinite(model.netSpyEquiv));
}

// 9. Legend_EMA futures (cutover 2026-09-26): long 1 MES with two independent
//    exits (GTD target LMT and a 10:30 goodAfterTime MKT, no OCA group), both
//    SELL orders carrying the entry ref plus |TARGET / |TIME, as
//    legend_ema_fut.build_exit_orders places them. One strategy contract, no hedge.
const legend = (id, type, ref, extra) => Object.assign({
  order_id: id, symbol: "MES", sec_type: "FUT", expiry: "20261218", action: "SELL", qty: 1,
  status: "Submitted", order_type: type, oca_group: "", parent_id: 0, order_ref: ref,
}, extra || {});
{
  const sig = "MES|BUY|Legend_EMA|2026-10-01";
  const model = context.attributeBook(book([
    { symbol: "MES", sec_type: "FUT", expiry: "20261218", position: 1, market_price: 7800 },
  ], [
    legend(91, "LMT", `${sig}|TARGET`, { lmt: 7830, tif: "GTD" }),
    legend(92, "MKT", `${sig}|TIME`, { good_after: "20261001 10:30:00 America/New_York" }),
  ]), betas, specs, { today: "20261001" });
  assert.strictEqual(model.futuresSpyEquiv, 0);
  assert.strictEqual(model.futures.filter(f => f.counted).length, 0);
  const row = model.byStrategy.find(r => r.strategy === "Legend_EMA");
  assert.ok(row, "Legend MES must attribute to Legend_EMA");
  assert.strictEqual(row.legs, 1);
  assert.strictEqual(row.spyEquiv, 5 * 7800);
  assert.strictEqual(row.legDetails[0].exitDate, "20261001");
}

// 10. Legend and OpenBreakout both long 1 MES in one 2-lot position: Legend's
//     two exit legs count once, so each strategy claims one contract.
{
  const sig = "MES|BUY|Legend_EMA|2026-10-01";
  const model = context.attributeBook(book([
    { symbol: "MES", sec_type: "FUT", expiry: "20261218", position: 2, market_price: 7800 },
  ], [
    legend(101, "LMT", `${sig}|TARGET`, { lmt: 7830 }),
    legend(102, "MKT", `${sig}|TIME`, { good_after: "20261001 10:30:00 America/New_York" }),
    ob(103, "STP", { aux: 7700 }),
  ]), betas, specs, { today: "20261001" });
  assert.strictEqual(model.futuresSpyEquiv, 0);
  assert.strictEqual(model.byStrategy.find(r => r.strategy === "Legend_EMA").legs, 1);
  assert.strictEqual(model.byStrategy.find(r => r.strategy === "OpenBreakout").legs, 1);
  const legendFut = model.futures.find(f => f.strategy === "Legend_EMA");
  const obFut = model.futures.find(f => f.strategy === "OpenBreakout");
  assert.strictEqual(legendFut.position, 1);
  assert.strictEqual(obFut.position, 1);
}

// 11. Mirror short (shorts are disabled today): short 1 MES with BUY exit legs.
//     The runner writes SELL_SHORT in the 2nd field; a plain SELL ref works too.
for (const action of ["SELL_SHORT", "SELL"]) {
  const sig = `MES|${action}|Legend_EMA|2026-10-01`;
  const model = context.attributeBook(book([
    { symbol: "MES", sec_type: "FUT", expiry: "20261218", position: -1, market_price: 7800 },
  ], [
    legend(111, "LMT", `${sig}|TARGET`, { action: "BUY", lmt: 7770 }),
    legend(112, "MKT", `${sig}|TIME`, { action: "BUY", good_after: "20261001 10:30:00 America/New_York" }),
  ]), betas, specs, { today: "20261001" });
  assert.strictEqual(model.futuresSpyEquiv, 0, `${action} short must not count as hedge`);
  const row = model.byStrategy.find(r => r.strategy === "Legend_EMA");
  assert.ok(row, `${action} short must attribute to Legend_EMA`);
  assert.strictEqual(row.legs, 1);
}

// 12. A Legend long's SELL exits do not claim a short hedge (strict side rule).
{
  const sig = "MES|BUY|Legend_EMA|2026-10-01";
  const model = context.attributeBook(book([
    { symbol: "MES", sec_type: "FUT", expiry: "20261218", position: -1, market_price: 7800 },
  ], [legend(121, "LMT", `${sig}|TARGET`, { lmt: 7830 })]), betas, specs, { today: "20261001" });
  assert.strictEqual(model.futuresSpyEquiv, -5 * 7800);
  assert.strictEqual(model.byStrategy.length, 0);
}

console.log("PASS hedge carve-out: OpenBreakout MES excluded from hedge, untagged MES counted, wrong side/expiry not claimed, Legend_EMA stock and MES futures attributed");
