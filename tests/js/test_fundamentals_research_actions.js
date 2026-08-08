"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(
  path.join(__dirname, "..", "..", "site", "assets", "fundamentals.js"),
  "utf8",
);

const storage = new Map();
const context = {
  console,
  URL,
  window: { location: { href: "http://localhost/fundamentals.html" } },
  document: { addEventListener() {} },
  localStorage: {
    getItem(key) { return storage.has(key) ? storage.get(key) : null; },
    setItem(key, value) { storage.set(key, String(value)); },
  },
};
vm.createContext(context);
vm.runInContext(source, context, { filename: "fundamentals.js" });

vm.runInContext("state = {version: 1, updated_at: null, actions: {}}", context);
vm.runInContext("state = updateLocalAction(state, 'YELP', 'DEEPEN')", context);
assert.strictEqual(vm.runInContext("actionName(state, 'YELP')", context), "DEEPEN");

vm.runInContext("state = updateLocalAction(state, 'YELP', 'PASS')", context);
assert.strictEqual(vm.runInContext("actionName(state, 'YELP')", context), "PASS");

vm.runInContext("state = updateLocalAction(state, 'YELP', 'CLEAR')", context);
assert.strictEqual(vm.runInContext("actionName(state, 'YELP')", context), "");
assert.deepStrictEqual(JSON.parse(storage.get("fundamentalResearchState.v1")).actions, {});

context.audit = {
  pass_summary: {
    background_count: 1146,
    reason_method: "One primary reason per company.",
    reasons: [{ label: "Valuation / expectations unproven", count: 335, pct: 29.2,
      explanation: "The screen has not proved a mispricing." }],
    trend_overlay: { without_full_confirmation: 714, amber: 426, red: 288,
      method: "Trend is an overlapping lens." },
  },
};
const passHTML = vm.runInContext("passesHTML(audit)", context);
assert.match(passHTML, /Why 1,146 companies are not in front of you/);
assert.match(passHTML, /Valuation \/ expectations unproven/);
assert.match(passHTML, /714 lack full confirmation/);

console.log("PASS fundamental research actions and aggregate pass summary");
