"use strict";
const assert = require("assert");
const fs = require("fs");
const vm = require("vm");
const path = require("path");
const nodes = new Map();
let redraws = 0, focused = "", scrolled = "";
const node = id => ({tagName: "DIV", focus() { focused = id; },
  scrollIntoView() { scrolled = id; }, set innerHTML(value) { redraws++; }});
for (const id of ["orders", "exec-orders-SPY", "exec-orders-EUR%2FUSD", "me_qty"])
  nodes.set(id, node(id));
const context = {console, window: {}, location: {search: ""}, URLSearchParams,
  document: {addEventListener() {}, getElementById: id => nodes.get(id)},
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(__dirname, "../../site/assets/execution.js"), "utf8"), context);
vm.runInContext(`renderOrders = () => "orders"; sendCommand = () => { throw Error("Navigation sent an order"); };`, context);
context.execShowOrders({symbol: "SPY", sec_type: "STK"});
assert.equal(focused, "exec-orders-SPY");
assert.equal(scrolled, "exec-orders-SPY");
assert.equal(vm.runInContext('expandedTickers.has("SPY")', context), true);
context.execShowOrders({symbol: "EUR", sec_type: "CASH", currency: "USD"});
assert.equal(focused, "exec-orders-EUR%2FUSD");
context.execShowOrders({symbol: "NO_ORDERS", sec_type: "STK"});
assert.equal(scrolled, "orders");
const before = redraws;
vm.runInContext('orderEdit.key = "pending-edit";', context);
context.execShowOrders({symbol: "SPY", sec_type: "STK"});
assert.equal(redraws, before, "Navigation must preserve an unfinished Modify");
assert.equal(focused, "me_qty");
const details = {tagName: "DETAILS", open: false, scrollIntoView() {}};
nodes.set("hedge-tools", details);
context.execJump("hedge-tools");
assert.equal(details.open, true);
console.log("PASS position-to-orders navigation, FX groups, empty orders, edit preservation, tool disclosure; no commands sent");
