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
assert.match(renderedPositions, /execSellTicket\(/);
assert.match(renderedPositions, /execAddTicket\(/);
assert.match(renderedPositions, /aria-pressed="false"[^>]*>Re-add<\/button>/);
assert.doesNotMatch(renderedPositions, />Trim|>Flatten|>Add&frac|>Re-add (on|off)/);
// The compact layout is Primary-only; PA retains its existing controls.
vm.runInContext('state.account = "pa"; state.book.accounts[0].key = "pa";', context);
const paPositions = vm.runInContext("renderPositions()", context);
assert.match(paPositions, />Trim&frac14;/);
assert.match(paPositions, />Trim&frac12;/);
vm.runInContext('state.account = "primary"; state.book.accounts[0].key = "primary";', context);

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
// Re-add off: the trim buttons take the SAFE partial close, not flatten.
// This position carries working exits, so it routes to close_resize, which
// shrinks them to the remainder before selling and cancels nothing. (Before
// 2026-09-03 this sent `flatten`, whose cancel-first order leaves the
// remainder unprotected between the cancel and the fill.)
const plainQuarter = JSON.parse(vm.runInContext("JSON.stringify({ lastConfirm, lastCommand })", context));
assert.match(plainQuarter.lastConfirm, /SELL 25 of 100 SMH MKT/);
assert.match(plainQuarter.lastConfirm, /shrink to 75 BEFORE the close/);
assert.strictEqual(plainQuarter.lastCommand.type, "close_resize");
assert.strictEqual(plainQuarter.lastCommand.payload.qty, 25);
assert.strictEqual(plainQuarter.lastCommand.payload.action, "SELL");
assert.strictEqual(plainQuarter.lastCommand.payload.con_id, 12345);
assert.strictEqual(plainQuarter.lastCommand.payload.fraction, undefined);

vm.runInContext("state.book = null; state.status = { online: false };", context);
assert.strictEqual(vm.runInContext("mutationBlocked('trim_readd')", context), false);
assert.strictEqual(vm.runInContext("mutationBlocked('add_to_position')", context), false);
assert.strictEqual(vm.runInContext("mutationBlocked('close_only')", context), false);

// Compact toggle is local state only; Add tickets preserve exact identity and
// require a positive whole quantity (or a percentage that rounds above zero).
vm.runInContext(`
  state.account = "primary";
  state.book = {mode:"live", accounts:[{key:"primary",positions:[${JSON.stringify(position)}],orders:[]}]};
  updateReadout = () => {};
  set = () => {};
  syncMutationControls = () => {};
  readdRows.clear();
  lastCommand = null;
  execToggleReadd(${JSON.stringify(position)});
`, context);
assert.match(vm.runInContext("renderPositions()", context), /aria-pressed="true"[^>]*>Re-add<\/button>/);
assert.strictEqual(vm.runInContext("lastCommand", context), null);
vm.runInContext(`
  execToggleReadd(${JSON.stringify(position)});
  fields = {f_symbol:"SMH",fl_qty:"",fl_pct:"150"};
  val = id => fields[id];
  ticketDraft.fl_position = {account:"primary", ...positionIdentity(${JSON.stringify(position)})};
  document.getElementById = id => id === "cmdType" ? {value:"add_to_position"} : null;
`, context);
assert.match(vm.runInContext("renderPositions()", context), /aria-pressed="false"[^>]*>Re-add<\/button>/);
assert.strictEqual(vm.runInContext("JSON.stringify(addWarnings())", context), "[]");
assert.strictEqual(vm.runInContext("ticketPayload('add_to_position').fraction", context), 1.5);
assert.strictEqual(vm.runInContext("ticketPayload('add_to_position').con_id", context), 12345);
vm.runInContext('confirm = () => false; sendTicket();', context);
assert.strictEqual(vm.runInContext("lastCommand", context), null);
vm.runInContext('confirm = () => true; sendTicket();', context);
assert.strictEqual(vm.runInContext("lastCommand.type", context), "add_to_position");
vm.runInContext('fields.fl_pct = "0.01";', context);
assert.ok(vm.runInContext("addWarnings()", context).some(w=>/zero/.test(w)));
vm.runInContext('fields.fl_qty = "1.5";', context);
assert.ok(vm.runInContext("addWarnings()", context).some(w=>/whole/.test(w)));
vm.runInContext('fields.fl_qty = "25";', context);
assert.strictEqual(vm.runInContext("JSON.stringify(addWarnings())", context), "[]");
assert.strictEqual(vm.runInContext("ticketPayload('add_to_position').qty", context), 25);
console.log("PASS compact Primary controls, Re-add toggle, Add tickets and legacy fast-action contracts");
