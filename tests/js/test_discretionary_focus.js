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
    trigger: { condition: `${ticker} trigger` },
    invalidation: {
      technical: `${ticker} technical invalidation`,
      thesis_kill: `${ticker} thesis invalidation`,
    },
    catalyst: `${ticker} catalyst`,
    priced_in: `${ticker} priced in`,
    next_proof: `${ticker} next proof`,
    event_date: "2026-09-10",
    earnings_td: 10,
    technical: {
      observed_at: "2026-08-26T08:45:00-04:00",
      setup_gate: "PASS",
      liquidity_gate: "PASS",
      setup_quality: 75,
    },
    sources: [{
      source_id: `${ticker.toLowerCase()}-release`,
      label: "Company release",
      url: "https://example.com/release",
      as_of: "2026-08-25",
      primary: true,
    }],
  };
}

function payload(status, focus) {
  return {
    schema_version: "discretionary-focus.v1",
    research_only: true,
    quick_review_created: false,
    live_actions_enabled: false,
    order_staging_enabled: false,
    phase: "FINAL",
    status,
    as_of: "2026-08-25",
    valid_for: "2026-08-26",
    generated_at: "2026-08-26T08:47:00-04:00",
    expires_at: "2026-08-26T16:15:00-04:00",
    focus,
    screen_summary: {
      input_count: 4,
      technical_pass_count: 3,
      research_pass_count: focus.length,
      selected_count: focus.length,
      rejected_counts: { not_selected: 4 - focus.length },
    },
    provenance: {
      screen_snapshot_id: "screen-20260826",
      screen_captured_at: "2026-08-26T08:45:00-04:00",
      research_snapshot_id: "research-20260826",
      research_as_of: "2026-08-26T08:46:00-04:00",
      policy_version: "discretionary-focus-policy.v1",
      tradingview_armed_url: "https://www.tradingview.com/screener/FzMHioHX/",
      tradingview_live_url: "https://www.tradingview.com/screener/60i0utaT/",
    },
    ...(status === "NO_QUALIFIED_SETUP" ? {
      no_setup_reason: "No candidate cleared every required gate.",
    } : {}),
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
  assert.doesNotMatch(html, /\[object Object\]/);
  assert.doesNotMatch(html, /condition:/i);
  assert.match(html, /Open Live RVOL screen/);
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

const unsafeAction = payload("READY", [card(1, "AMPL")]);
unsafeAction.focus[0].technical.order = "BUY";
assert.strictEqual(validate(unsafeAction).state, "UNAVAILABLE");

const nearEarnings = payload("READY", [card(1, "AMPL")]);
nearEarnings.focus[0].earnings_td = 5;
assert.strictEqual(validate(nearEarnings).state, "UNAVAILABLE");

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
malicious.focus[0].sources = [{
  source_id: "bad-source",
  label: "Bad <source>",
  url: "https://example.com/release",
  as_of: "2026-08-25",
  primary: true,
}];
const maliciousHTML = render(validate(malicious));
assert.match(maliciousHTML, /&lt;img/);
assert.doesNotMatch(maliciousHTML, /<img/);
assert.doesNotMatch(maliciousHTML, /javascript:/i);
assert.match(maliciousHTML, /Bad &lt;source&gt;/);

const unsafeURL = payload("READY", [card(1, "AMPL")]);
unsafeURL.focus[0].sources[0].url = "javascript:alert(1)";
const unsafeURLResult = validate(unsafeURL);
assert.strictEqual(unsafeURLResult.state, "UNAVAILABLE");
assert.doesNotMatch(render(unsafeURLResult), /javascript:/i);

const relativeURL = payload("READY", [card(1, "AMPL")]);
relativeURL.focus[0].sources[0].url = "/not-an-absolute-source";
assert.strictEqual(validate(relativeURL).state, "UNAVAILABLE");

const duplicate = payload("READY", [card(1, "AMPL"), card(2, "AMPL")]);
assert.strictEqual(validate(duplicate).state, "UNAVAILABLE");

const impossibleCounts = payload("READY", [card(1, "AMPL")]);
impossibleCounts.screen_summary.rejected_counts.not_selected = 1;
assert.strictEqual(validate(impossibleCounts).state, "UNAVAILABLE");

const staleProvenance = payload("READY", [card(1, "AMPL")]);
staleProvenance.provenance.screen_captured_at = "2026-08-20T16:00:00-04:00";
assert.strictEqual(validate(staleProvenance).state, "UNAVAILABLE");

const badDigest = payload("READY", [card(1, "AMPL")]);
badDigest.provenance.screen_digest = "not-a-digest";
assert.strictEqual(validate(badDigest).state, "UNAVAILABLE");

const invalidCard = payload("READY", [card(2, "AMPL")]);
assert.strictEqual(validate(invalidCard).state, "UNAVAILABLE");

context.historyPayload = {
  schema_version: "discretionary-focus-history.v1",
  items: [
    {
      valid_for: "2026-08-25",
      status: "READY",
      phase: "FINAL",
      generated_at: "2026-08-25T08:40:00-04:00",
      focus: [{ ticker: "AMPL", company_name: "Amplitude" }],
    },
    {
      valid_for: "2026-08-24",
      status: "NO_QUALIFIED_SETUP",
      phase: "FINAL",
      generated_at: "2026-08-24T08:40:00-04:00",
      focus: [],
    },
  ],
};
const historyRows = JSON.parse(vm.runInContext(
  "JSON.stringify(validateFocusHistory(historyPayload))", context));
context.historyRows = historyRows;
const historyHTML = vm.runInContext("focusHistoryHTML(historyRows)", context);
assert.match(historyHTML, /Recent history/);
assert.match(historyHTML, /AMPL/);
assert.match(historyHTML, /No qualified setup/);

console.log("PASS discretionary focus payload and rendering contract");
