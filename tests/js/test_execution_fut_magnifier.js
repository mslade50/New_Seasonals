"use strict";

// IBKR quotes grains/meats in cents (priceMagnifier 100); the ticket's dollar
// math must use multiplier / price_magnifier, and a missing field must read 1.
const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(
  path.join(__dirname, "..", "..", "site", "assets", "execution.js"),
  "utf8",
);
const context = {
  console,
  document: { addEventListener() {} },
  window: {},
  location: { search: "" },
  URLSearchParams,
  setTimeout,
  clearTimeout,
  setInterval,
  clearInterval,
};
vm.createContext(context);
vm.runInContext(source, context, { filename: "execution.js" });

function jsonExpr(expr) {
  return JSON.parse(vm.runInContext(`JSON.stringify(${expr})`, context));
}

const XK = '{exchange:"CBOT",multiplier:1000,min_tick:0.00125,price_magnifier:100}';
const MES = '{exchange:"CME",multiplier:5,min_tick:0.25}';

const xk = jsonExpr(`futTicketMetrics(${XK}, 2, 1280, 1260)`);
assert.strictEqual(xk.mult, 10);
assert.strictEqual(xk.risk, 400);
assert.strictEqual(xk.notional, 25600);
assert.strictEqual(jsonExpr(`futPriceTick(${XK})`), 0.125);
assert.strictEqual(jsonExpr(`futUnitHint(${XK})`), "quoted in cents");

const mes = jsonExpr(`futTicketMetrics(${MES}, 2, 5000, 4980)`);
assert.deepStrictEqual(mes, { mult: 5, risk: 200, notional: 50000 });
assert.strictEqual(jsonExpr(`futPriceTick(${MES})`), 0.25);
assert.strictEqual(jsonExpr(`futUnitHint(${MES})`), "");

// Missing or invalid magnifier reads 1 (over-states risk, never under-states).
for (const bad of ["undefined", "null", "0", "-100", "1.5", '"x"']) {
  assert.strictEqual(jsonExpr(`futMagnifier({multiplier:1000,price_magnifier:${bad}})`), 1);
}
assert.strictEqual(jsonExpr(`futMagnifier(null)`), 1);

// Average cost from IBKR is in dollars: XK 12,800/contract -> 1280 quoted.
vm.runInContext(`FUT_SPECS = {XK: ${XK}, MES: ${MES}}`, context);
assert.strictEqual(
  jsonExpr('quotedAverageCost({symbol:"XK",sec_type:"FUT",multiplier:1000,avg_cost:12800})'),
  1280,
);
assert.strictEqual(
  jsonExpr('quotedAverageCost({symbol:"MES",sec_type:"FUT",multiplier:5,avg_cost:25000})'),
  5000,
);

console.log("PASS execution futures price-magnifier ticket math");
