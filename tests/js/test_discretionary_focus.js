"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(
  path.join(__dirname, "..", "..", "site", "assets", "focus.js"), "utf8");

const context = {
  console,
  URL,
  window: { location: { href: "https://private.example/focus.html" } },
  document: { addEventListener() {} },
};
vm.createContext(context);
vm.runInContext(source, context, { filename: "focus.js" });

const NOW = "2026-08-26T10:00:00-04:00";

function card(rank, ticker) {
  return {
    rank,
    ticker,
    company_name: `Company ${ticker}`,
    why_now: `${ticker} why now`,
    setup: `${ticker} setup`,
    trigger: `${ticker} trigger`,
    invalidation: `${ticker} invalidation`,
    catalyst: `${ticker} catalyst`,
    priced_in: `${ticker} priced in`,
    next_proof: `${ticker} next proof`,
    event_date: "2026-09-10",
    event_label: "Earnings",
    sources: [{ label: "Company release", url: "https://example.com/release" }],
  };
}

function payload(status, focus) {
  return {
    schema_version: "discretionary-focus.v1",
    research_only: true,
    status,
    as_of: "2026-08-25",
    valid_for: "2026-08-26",
    generated_at: "2026-08-26T08:47:00-04:00",
    expires_at: "2026-08-26T16:15:00-04:00",
    focus,
  };
}

function validate(value, now = NOW) {
  context.value = value;
  context.now = now;
  return JSON.parse(vm.runInContext(
    "JSON.stringify(validateFocusPayload(value, new Date(now)))", context));
}

function render(result) {
  context.result = result;
  return vm.runInContext("focusStateHTML(result)", context);
}

for (const names of [[card(1, "AMPL")], [card(1, "AMPL"), card(2, "EAT")]]) {
  const result = validate(payload("READY", names));
  assert.strictEqual(result.state, "READY");
  const html = render(result);
  assert.match(html, /deserve attention/);
  assert.match(html, /2026-08-26T08:47:00-04:00/);
  for (const row of names) assert.match(html, new RegExp(row.ticker));
}

const noSetup = validate(payload("NO_QUALIFIED_SETUP", []));
assert.strictEqual(noSetup.state, "NO_QUALIFIED_SETUP");
assert.match(render(noSetup), /No qualified setup today/);
assert.match(render(noSetup), /Evidence through 2026-08-25/);

const oversized = validate(payload("READY", [card(1, "AAA"), card(2, "BBB"), card(3, "CCC")]));
assert.strictEqual(oversized.state, "UNAVAILABLE");
assert.match(oversized.reason, /more than two/i);
assert.doesNotMatch(render(oversized), /AAA|BBB|CCC/);

assert.strictEqual(validate(payload("READY", [])).state, "UNAVAILABLE");
assert.strictEqual(validate(payload("NO_QUALIFIED_SETUP", [card(1, "AMPL")])).state, "UNAVAILABLE");

const unsafeMode = payload("READY", [card(1, "AMPL")]);
unsafeMode.research_only = false;
assert.strictEqual(validate(unsafeMode).state, "UNAVAILABLE");

const wrongSchema = payload("READY", [card(1, "AMPL")]);
wrongSchema.schema_version = "discretionary-focus.v2";
assert.strictEqual(validate(wrongSchema).state, "UNAVAILABLE");

const noTimezone = payload("READY", [card(1, "AMPL")]);
noTimezone.generated_at = "2026-08-26T08:47:00";
assert.strictEqual(validate(noTimezone).state, "UNAVAILABLE");

const expired = validate(payload("READY", [card(1, "AMPL")]), "2026-08-26T16:15:00-04:00");
assert.strictEqual(expired.state, "EXPIRED");
const expiredHTML = render(expired);
assert.match(expiredHTML, /Focus list expired/);
assert.doesNotMatch(expiredHTML, /AMPL/);

const malicious = payload("READY", [card(1, "AMPL")]);
malicious.focus[0].why_now = '<img src=x onerror="alert(1)">';
malicious.focus[0].sources = [{ label: "Bad <source>", url: "javascript:alert(1)" }];
const maliciousHTML = render(validate(malicious));
assert.match(maliciousHTML, /&lt;img/);
assert.doesNotMatch(maliciousHTML, /<img/);
assert.doesNotMatch(maliciousHTML, /javascript:/i);
assert.match(maliciousHTML, /Bad &lt;source&gt;/);

const invalidCard = payload("READY", [card(2, "AMPL")]);
assert.strictEqual(validate(invalidCard).state, "UNAVAILABLE");

console.log("PASS discretionary focus payload and rendering contract");
