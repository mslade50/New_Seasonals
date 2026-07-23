"use strict";

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

assert.deepStrictEqual(jsonExpr('fxPairWarnings("NZD", "USD")'), []);
assert.deepStrictEqual(jsonExpr('fxPairWarnings("USD", "JPY")'), []);
assert.deepStrictEqual(
  jsonExpr('fxPairWarnings("EUR", "GBP")'),
  ["FX entry supports USD pairs only initially"],
);
assert.deepStrictEqual(
  jsonExpr('fxPairWarnings("NZD", "NZD")'),
  ["FX quote must be a different 3-letter currency"],
);
assert.deepStrictEqual(
  jsonExpr('positionIdentity({symbol:"NZD",currency:"USD",sec_type:"CASH",expiry:"",con_id:39453441,position:-250000})'),
  {
    symbol: "NZD", currency: "USD", sec_type: "CASH", expiry: null,
    con_id: 39453441, expected_position: -250000,
  },
);

const nzd = jsonExpr('fxUsdMetrics("NZD", "USD", 100000, 0.575, 0.565)');
assert.ok(Math.abs(nzd.notional - 57500) < 1e-8);
assert.ok(Math.abs(nzd.risk - 1000) < 1e-8);

const jpy = jsonExpr('fxUsdMetrics("USD", "JPY", 100000, 160, 158)');
assert.strictEqual(jpy.notional, 100000);
assert.strictEqual(jpy.risk, 1250);

console.log("PASS execution FX pair validation and USD exposure math");
