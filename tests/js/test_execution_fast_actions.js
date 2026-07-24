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

const position = { symbol: "SMH", sec_type: "STK", expiry: "", con_id: 12345,
  position: 100, avg_cost: 250.25 };
const trim = JSON.parse(vm.runInContext(`JSON.stringify(trimReaddPayload(${JSON.stringify(position)}, 0.5))`, context));
assert.deepStrictEqual(trim, {
  symbol: "SMH", sec_type: "STK", expiry: null, con_id: 12345,
  expected_position: 100, fraction: 0.5, close_order_type: "MKT",
  readd: true, readd_tif: "DAY",
});

const add = JSON.parse(vm.runInContext(`JSON.stringify(addPositionPayload(${JSON.stringify(position)}, 1))`, context));
assert.deepStrictEqual(add, {
  symbol: "SMH", sec_type: "STK", expiry: null, con_id: 12345,
  expected_position: 100, fraction: 1, order_type: "MKT",
});

vm.runInContext(`state.book = { accounts: [{ key: "primary", orders: [{
  symbol: "SMH", sec_type: "STK", con_id: 12345, action: "SELL",
  order_type: "MKT", qty: 100, good_after: "20260731 15:59:00 US/Eastern"
}] }] }`, context);
assert.strictEqual(vm.runInContext(`hasVisibleProtectiveExit(${JSON.stringify(position)})`, context), true);
vm.runInContext(`state.book.accounts[0].orders[0] = {
  symbol: "SMH", sec_type: "STK", con_id: 12345, action: "SELL",
  order_type: "LMT", qty: 100, lmt: 300
}`, context);
assert.strictEqual(vm.runInContext(`hasVisibleProtectiveExit(${JSON.stringify(position)})`, context), false);

assert.strictEqual(vm.runInContext("fastActionQty(101, 0.5)", context), 51);
assert.strictEqual(vm.runInContext("mutationBlocked('trim_readd')", context), true);
assert.strictEqual(vm.runInContext("mutationBlocked('add_to_position')", context), true);

console.log("PASS execution fast-action payloads, rounding, and unknown-mode block");
