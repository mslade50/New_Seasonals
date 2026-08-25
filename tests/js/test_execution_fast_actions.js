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

const quarterTrim = JSON.parse(vm.runInContext(`JSON.stringify(trimReaddPayload(${JSON.stringify(position)}, 0.25))`, context));
assert.deepStrictEqual(quarterTrim, {
  symbol: "SMH", sec_type: "STK", expiry: null, con_id: 12345,
  expected_position: 100, fraction: 0.25, close_order_type: "MKT",
  readd: true, readd_tif: "DAY",
});

const add = JSON.parse(vm.runInContext(`JSON.stringify(addPositionPayload(${JSON.stringify(position)}, 1))`, context));
assert.deepStrictEqual(add, {
  symbol: "SMH", sec_type: "STK", expiry: null, con_id: 12345,
  expected_position: 100, fraction: 1, order_type: "MKT",
});

context.document.getElementById = () => null;
vm.runInContext(`
  state.account = "primary";
  state.book = { accounts: [{ key: "primary", positions: [${JSON.stringify(position)}], orders: [] }] };
  ticketDraft.fl_position = { account: "primary", ...positionIdentity(${JSON.stringify(position)}) };
  val = (id) => ({ f_symbol: "SMH", fl_qty: "", fl_pct: "40", fl_type: "MKT", fl_tif: "DAY" })[id];
`, context);
const closeOnly = JSON.parse(vm.runInContext(`JSON.stringify(ticketPayload("close_only"))`, context));
assert.deepStrictEqual(closeOnly, {
  symbol: "SMH", order_type: "MKT", tif: "DAY", outside_rth: false,
  sec_type: "STK", expiry: null, expected_position: 100, con_id: 12345,
  fraction: 0.4, action: "SELL",
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
assert.strictEqual(vm.runInContext("fastActionQty(101, 0.25)", context), 25);
assert.strictEqual(vm.runInContext("fastActionQty(102, 0.25)", context), 26);

vm.runInContext(`
  state.account = "primary";
  state.status = { online: true };
  state.book = {
    at: Date.now(), mode: "live",
    accounts: [{ key: "primary", positions: [${JSON.stringify(position)}], orders: [{
      symbol: "SMH", sec_type: "STK", con_id: 12345, action: "SELL",
      order_type: "STP", qty: 100, aux: 220
    }] }]
  };
  confirm = (message) => { lastConfirm = message; return true; };
  alert = () => {};
  sendCommand = (type, payload) => { lastCommand = { type, payload }; };
  fmt = { num: (v) => String(v), money: (v) => String(v), pct: (v) => String(v) };
  esc = (v) => String(v);
  clsSign = () => "";
  panelNote = (v) => String(v);
  lastConfirm = "";
  lastCommand = null;
`, context);

const renderedPositions = vm.runInContext("renderPositions()", context);
assert.match(renderedPositions, /execTrim\([^)]*\},0\.25\)'[^>]*>Trim&frac14;/);
assert.match(renderedPositions, /execTrim\([^)]*\},0\.5\)'[^>]*>Trim&frac12;/);

vm.runInContext(`
  readdRows.set(positionKey(${JSON.stringify(position)}), true);
  execTrim(${JSON.stringify(position)}, 0.25);
`, context);
const readdQuarter = JSON.parse(vm.runInContext("JSON.stringify({ lastConfirm, lastCommand })", context));
assert.match(readdQuarter.lastConfirm, /SELL 25 SMH MKT/);
assert.match(readdQuarter.lastConfirm, /expected post-trim position 75/);
assert.deepStrictEqual(readdQuarter.lastCommand, {
  type: "trim_readd",
  payload: {
    symbol: "SMH", sec_type: "STK", expiry: null, con_id: 12345,
    expected_position: 100, fraction: 0.25, close_order_type: "MKT",
    readd: true, readd_tif: "DAY",
  },
});

vm.runInContext(`
  readdRows.set(positionKey(${JSON.stringify(position)}), false);
  lastConfirm = "";
  lastCommand = null;
  execTrim(${JSON.stringify(position)}, 0.25);
`, context);
const plainQuarter = JSON.parse(vm.runInContext("JSON.stringify({ lastConfirm, lastCommand })", context));
assert.match(plainQuarter.lastConfirm, /flatten 25% of SMH/);
assert.strictEqual(plainQuarter.lastCommand.type, "flatten");
assert.strictEqual(plainQuarter.lastCommand.payload.fraction, 0.25);

vm.runInContext("state.book = null; state.status = { online: false };", context);
assert.strictEqual(vm.runInContext("mutationBlocked('trim_readd')", context), true);
assert.strictEqual(vm.runInContext("mutationBlocked('add_to_position')", context), true);
assert.strictEqual(vm.runInContext("mutationBlocked('close_only')", context), true);

console.log("PASS execution fast-action payloads, quarter/half trims, close-only gate, rounding, and unknown-mode block");
