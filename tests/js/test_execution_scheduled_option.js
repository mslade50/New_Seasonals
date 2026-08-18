"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(
  path.join(__dirname, "..", "..", "site", "assets", "execution.js"),
  "utf8",
);
const fields = {};
const context = {
  console,
  document: {
    addEventListener() {},
    getElementById: (id) => (id in fields ? fields[id] : null),
    querySelectorAll: () => [],
  },
  window: {},
  location: { search: "" },
  URLSearchParams,
  setTimeout,
  clearTimeout,
  setInterval,
  clearInterval,
  fmt: { money: (v) => `$${Number(v).toFixed(0)}`, num: (v) => String(v), pct: (v) => String(v) },
  confirm: () => true,
};
vm.createContext(context);
vm.runInContext(source, context, { filename: "execution.js" });

function setFields(map) {
  Object.keys(fields).forEach((key) => delete fields[key]);
  for (const [key, value] of Object.entries(map)) fields[key] = { value: String(value) };
}
const run = (code) => vm.runInContext(code, context);
const json = (code) => JSON.parse(run(`JSON.stringify(${code})`));

run(`state.account = "primary"`);
setFields({
  so_symbol: "spy", so_right: "P", so_delta: "0.15", so_budget: "1000",
  so_date: "2099-08-21", so_time: "15:45", so_expiry_mode: "min_dte",
  so_min_dte: "30", so_expiry: "",
});
assert.deepStrictEqual(json("scheduledOptionWarnings()"), []);
assert.deepStrictEqual(json('ticketPayload("scheduled_option")'), {
  symbol: "SPY", right: "P", target_delta: 0.15, delta_tolerance: 0.03,
  premium_budget: 1000, order_type: "MKT", tif: "DAY",
  execute_date: "2099-08-21", execute_time: "15:45", timezone: "America/New_York",
  grace_minutes: 5, expiry_mode: "min_dte", min_dte: 30, expiry: null,
});

fields.so_expiry_mode.value = "specific";
fields.so_expiry.value = "2099-09-18";
assert.deepStrictEqual(json("scheduledOptionWarnings()"), []);
const specific = json('ticketPayload("scheduled_option")');
assert.strictEqual(specific.expiry_mode, "specific");
assert.strictEqual(specific.min_dte, null);
assert.strictEqual(specific.expiry, "2099-09-18");

fields.so_delta.value = "0.75";
assert.ok(json("scheduledOptionWarnings()").some((x) => x.includes("delta")));
fields.so_delta.value = "0.15";
run(`state.account = "pa"`);
assert.ok(json("scheduledOptionWarnings()").some((x) => x.includes("Primary")));
run(`state.account = "primary"`);

assert.ok(run(`MUTATING_COMMANDS.has("scheduled_option")`));
assert.ok(run(`MUTATING_COMMANDS.has("scheduled_option_cancel")`));
const scheduledCell = run(`resultCell({id:"abc-123", type:"scheduled_option", state:"scheduled", result:{detail:"ready"}})`);
assert.ok(scheduledCell.includes("Cancel schedule"));
assert.ok(scheduledCell.includes("abc-123"));

console.log("PASS scheduled option UI: min-DTE/specific expiry, target-delta MKT payload, guards, and cancel control");
