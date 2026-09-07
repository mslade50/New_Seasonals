"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const root = path.resolve(__dirname, "../..");

function load(relative, context, transform = source => source) {
  vm.createContext(context);
  vm.runInContext(transform(fs.readFileSync(path.join(process.env.AUDIT_SOURCE_ROOT || root, relative), "utf8")), context);
  return context;
}

class Bucket {
  constructor(body = null) { this.body = body; this.revision = body == null ? 0 : 1; this.puts = 0; }
  async get() {
    const body = this.body, etag = String(this.revision);
    return body == null ? null : { etag, text: async () => body };
  }
  async put(key, body, options = {}) {
    this.puts++;
    const condition = options.onlyIf || {};
    if (condition.etagMatches && condition.etagMatches !== String(this.revision)) return null;
    if (condition.etagDoesNotMatch === "*" && this.body != null) return null;
    this.body = body; this.revision++;
    return { etag: String(this.revision) };
  }
}

async function testState() {
  const context = load("functions/fundamental-state.js", { Response, Request, requireAccess: async () => null },
    source => source.replace(/^import .*;\s*$/m, "").replaceAll("export async function", "async function"));
  const post = (bucket, ticker) => context.onRequestPost({
    env: { CHARTS: bucket }, request: new Request("https://example.com/fundamental-state", {
      method: "POST", body: JSON.stringify({ ticker, action: "WATCH" }),
    }),
  });
  for (const malformed of ['{"actions":', '{"version":1,"actions":[]}', '{"version":1,"actions":{"AAA":{"action":"ORDER"}}}']) {
    const bucket = new Bucket(malformed);
    const response = await post(bucket, "AAA");
    assert.equal(response.status, 503, "corrupt history must fail closed");
    assert.equal(bucket.body, malformed);
    assert.equal(bucket.puts, 0);
  }
  for (const original of [null, JSON.stringify({ version: 1, actions: { OLD: { action: "PASS" } } })]) {
    const bucket = new Bucket(original);
    const responses = await Promise.all([post(bucket, "AAA"), post(bucket, "BBB")]);
    assert.deepEqual(responses.map(r => r.status), [200, 200]);
    const result = JSON.parse(bucket.body);
    assert.ok(result.actions.AAA && result.actions.BBB, "racing choices must both survive");
    if (original) assert.equal(result.actions.OLD.action, "PASS");
    assert.equal(result.history.length, 2);
  }
  const rejected = new Bucket();
  rejected.put = async () => null;
  assert.equal((await post(rejected, "AAA")).status, 409);
}

async function testTradeLog() {
  const context = load("site/assets/tradelog.js", { document: { addEventListener() {} }, console, clsSign: () => "" });
  vm.runInContext('renderData = () => {}; tlState.fills = [{exec_id: "old"}]; tlState.lastSuccessfulAt = new Date("2026-08-05T12:00:00Z");', context);
  let stamp;
  context.setAsof = value => { stamp = value; };
  context.fetchJSONOrNull = async () => null;
  await context.tlLoad();
  assert.equal(vm.runInContext("tlState.fills.length", context), 1);
  assert.match(vm.runInContext("tlState.error", context), /Refresh failed/);
  assert.match(stamp, /Stale.*08:00:00/);
  context.fetchJSONOrNull = async () => ({ fills: [] });
  await context.tlLoad();
  assert.equal(vm.runInContext("tlState.error", context), null);
  assert.doesNotMatch(stamp, /Stale/);
}

async function testInbox() {
  const context = load("site/assets/fundamentals.js", { document: { addEventListener() {} } });
  for (const action of ["PASS", "WATCH"]) {
    const state = { actions: { AAA: { action, updated_at: "2026-08-05T12:00:00Z" } } };
    assert.equal(context.inboxSuppressed(state, { ticker: "AAA" }), true);
    assert.equal(context.inboxSuppressed(state, { ticker: "AAA", control_disposition: "REOPENED_BY_TRIGGER",
      control_updated_at: "2026-08-05T12:00:00Z" }), false);
    assert.equal(context.inboxSuppressed(state, { ticker: "AAA", control_disposition: "REOPENED_BY_TRIGGER",
      control_updated_at: "2026-08-04T12:00:00Z" }), true);
  }
}

(async () => {
  const cases = { state: testState, tradelog: testTradeLog, inbox: testInbox };
  if (process.env.AUDIT_CASE) await cases[process.env.AUDIT_CASE]();
  else { await testState(); await testTradeLog(); await testInbox(); }
  console.log("PASS research state corruption/CAS, Trade Log stale refresh, PASS/WATCH current inbox");
})().catch(error => { console.error(error); process.exitCode = 1; });
