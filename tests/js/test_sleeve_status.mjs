import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
const read = path => fs.readFileSync(new URL(`../../${path}`, import.meta.url), "utf8");
const moduleUrl = source => `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
const helperUrl = moduleUrl(read("functions/_sleeve-status.js"));
const { buildStatus, readInputs, INPUTS, text } = await import(helperUrl);
assert.equal(text("rejected for U16584234 and acct_DU1234567; order 12345678 kept"), "rejected for [account] and acct_[account]; order 12345678 kept");
const now = Date.parse("2026-09-06T15:00:00Z");
const checked_at = "2026-09-06T14:30:00Z";
const tasks = Object.fromEntries(["event", "trend", "chain", "legend", "legend_verify"].map(k => [k, { state: "Missing", last_result: null }]));
tasks.event = tasks.chain = { state: "Ready", last_result: 0 };
const runtime = { schema: "sleeve-runtime.v2", checked_at, tasks, event_enabled: true, trend_moo_enabled: false, legend_enabled: false };
const report = data => ({ data, uploaded: checked_at });
const inputs = { runtime: report(runtime), event: report({ asof: "2026-09-04", rows: [], positions: {} }), trend: report({ asof: "2026-08-31", positions: {}, fragility_gate: { state: "CASH", reason: "Fragility above threshold" } }), dial: report({ last_evaluated: "2026-09-04", position: "FLAT", transitions: [] }) };
const rows = x => Object.fromEntries(buildStatus(x, now).sleeves.map(row => [row.id, row]));
let s = rows(inputs);
assert.equal(s.event.deployment, "Scheduled");
assert.equal(s.event.summary, "0 actions staged · 0 intended positions");
assert.equal(s.event.health, "reported");
assert.equal(s.trend.deployment, "Legacy 9:31 execution");
assert.match(s.trend.summary, /Cash per rule/);
assert.equal(s.legend.deployment, "Runner task missing");
assert.equal(s.open_breakout.summary, "unavailable: not reported");
assert.equal(rows({ ...inputs, runtime: report({ ...runtime, schema: "sleeve-runtime.v1" }) }).event.deployment, "Not verified", "retired schema is rejected");
assert.equal(s.dial.deployment, "Paper only");
assert.equal(s.hedge.deployment, "Manual analysis only");
assert.equal(rows({}).event.health, "unavailable");
assert.equal(rows({}).event.deployment, "Not verified");
for (const checked of ["2026-09-06T12:00:00Z", "2026-09-07T00:00:00Z", "invalid"]) {
  assert.equal(rows({ ...inputs, runtime: report({ ...runtime, checked_at: checked }) }).legend.deployment, "Not verified");
}
assert.equal(rows({ ...inputs, runtime: report({ ...runtime, tasks: {} }) }).event.deployment, "Not verified");
assert.equal(rows({ ...inputs, event: report({ asof: "2026-08-01", rows: [], positions: {} }) }).event.health, "aging", "fresh upload cannot hide stale report");
for (const asof of ["2026-02-30", "2026-09-08", "invalid"]) assert.equal(rows({ ...inputs, event: report({ asof, rows: [], positions: {} }) }).event.health, "unavailable");
assert.equal(rows({ ...inputs, event: report({ asof: "2026-09-04", positions: {} }) }).event.health, "unavailable", "missing actions cannot appear as zero");
const moo = { ...runtime, trend_moo_enabled: true, tasks: { ...tasks, trend: { state: "Ready", last_result: 0 } } };
assert.equal(rows({ ...inputs, runtime: report(moo) }).trend.deployment, "Opening-auction runner");
assert.equal(rows({ ...inputs, runtime: report({ ...runtime, trend_moo_enabled: true }) }).trend.deployment, "Runner missing or disabled");
assert.equal(rows({ ...inputs, runtime: report({ ...moo, tasks: { ...moo.tasks, trend: { state: "Ready", last_run_at: checked_at, last_result: 1 } } }) }).trend.deployment, "Last runner failed");
assert.equal(rows({ ...inputs, runtime: report({ ...moo, tasks: { ...moo.tasks, trend: { state: "Running", last_run_at: checked_at, last_result: 267009 } } }) }).trend.deployment, "Opening-auction runner", "running is not a completed failure");
// Legend EMA: the single "IBKR Legend EMA" runner plus its verify task, and the runner's own session evidence.
const legendTasks = { ...tasks, legend: { state: "Ready", last_run_at: "2026-09-04T13:29:29Z", next_run_at: "2026-09-08T13:29:29Z", last_result: 0 },
  legend_verify: { state: "Ready", last_run_at: "2026-09-04T14:40:40Z", next_run_at: "2026-09-08T14:40:40Z", last_result: 0 } };
const legendSession = { available: true, session_date: "2026-09-04", result_error: "", notes: [],
  trades: [{ symbol: "SPY", side: "BUY", qty: 1, status: "SENT", outcome: "target hit" }],
  skips: [{ symbol: "QQQ", status: "NO_SETUP", note: "daily body ratio 0.645 < 0.75" }] };
const legendRow = extra => rows({ ...inputs, runtime: report({ ...runtime, tasks: legendTasks, legend: legendSession, ...extra }) }).legend;
let lg = legendRow({});
assert.equal(lg.deployment, "Dry run (activation flag absent)");
assert.equal(lg.summary, "last: 2026-09-04 SPY long 1 sh, target hit · skipped QQQ no setup (daily body ratio 0.645 < 0.75)");
assert.equal(lg.health, "reported");
assert.match(lg.details[0], /^Runner: last ran Sep 4, 9:29 AM ET · result 0 · next Sep 8, 9:29 AM ET$/);
assert.equal(legendRow({ legend_enabled: true }).deployment, "Armed for live orders");
lg = legendRow({ tasks: { ...legendTasks, legend: { ...legendTasks.legend, last_result: 1 } } });
assert.equal(lg.deployment, "Last run failed (code 1)");
assert.equal(lg.attention, true);
assert.equal(legendRow({ tasks: { ...legendTasks, legend_verify: { ...legendTasks.legend_verify, last_result: 2 } } }).deployment, "Verify failed (code 2)");
lg = legendRow({ legend: { available: false, reason: "journal and last-result file missing" } });
assert.equal(lg.summary, "unavailable: journal and last-result file missing");
assert.equal(lg.health, "unavailable");
assert.equal(legendRow({ legend: { ...legendSession, session_date: "2026-08-20" } }).health, "aging", "old Legend session ages");
assert.equal(legendRow({ legend: { ...legendSession, trades: [], skips: [] } }).summary, "last: 2026-09-04 no entry");
assert.ok(!JSON.stringify(buildStatus(inputs, now)).includes("LegendETF"));

// Open Breakout: newest shadow and live session, heartbeat age at the machine check.
const nowOB = Date.parse("2026-09-28T14:00:00Z"); // 10:00 ET Monday
const checkedOB = "2026-09-28T13:59:00Z";
const session = (mode, extra = {}) => ({ available: true, run_dir: `2026-09-28-${mode}`, session: "2026-09-28", mode, pid: 1, phase: mode === "live" ? "RUNNING_LIVE" : "RUNNING_SHADOW",
  heartbeat_at: "2026-09-28T13:58:58Z", events: 10, finished_at: null, last_error: null,
  prior_range: { NQ: { status: "OK", ratio: 1.4, skip: true, half: false, reason: "ratio >= 1.25" }, ES: { status: "OK", ratio: 1.0, skip: false, half: false, reason: null } }, ...extra });
const ob = open_breakout => Object.fromEntries(buildStatus({ ...inputs, runtime: report({ ...runtime, checked_at: checkedOB, open_breakout }) }, nowOB).sleeves.map(r => [r.id, r])).open_breakout;
let o = ob({ available: true, shadow: session("shadow"), live: session("live") });
assert.equal(o.deployment, "Live RUNNING_LIVE · Shadow RUNNING_SHADOW");
assert.equal(o.sessions.live.heartbeat_age_s, 2);
assert.equal(o.sessions.live.stale, false);
assert.deepEqual(o.sessions.live.skipped, ["NQ"]);
assert.match(o.details[0], /^Live 2026-09-28 · RUNNING_LIVE · heartbeat 2s old at check · prior-range skip: NQ$/);
assert.equal(o.attention, false);
o = ob({ available: true, shadow: session("shadow"), live: session("live", { heartbeat_at: "2026-09-28T13:55:00Z" }) });
assert.equal(o.deployment, "Heartbeat stale");
assert.equal(o.sessions.live.stale, true);
assert.equal(o.health, "aging");
assert.match(o.details[0], /STALE heartbeat 240s old/);
o = ob({ available: true, shadow: session("shadow", { phase: "SESSION_COMPLETE", finished_at: "2026-09-28T20:01:00Z", heartbeat_at: "2026-09-28T13:00:00Z" }), live: null });
assert.equal(o.sessions.shadow.stale, false, "a finished session is never stale");
assert.equal(o.details[0], "Live: unavailable: no session found");
const old = s => ({ ...s, session: "2026-09-25", run_dir: `2026-09-25-${s.mode}`, phase: "FAILED", finished_at: "2026-09-25T13:30:01Z", heartbeat_at: "2026-09-25T13:29:51Z" });
o = ob({ available: true, shadow: old(session("shadow")), live: null });
assert.equal(o.deployment, "No session today");
assert.equal(o.summary, "No session today · last 2026-09-25");
o = ob({ available: true, shadow: { available: false, reason: "sqlite locked", run_dir: "2026-09-28-shadow" }, live: { available: false, reason: "runtime.sqlite missing", run_dir: "2026-09-28-live-2" } });
assert.equal(o.deployment, "Session today · state unavailable");
assert.equal(o.details[1], "Shadow: unavailable: sqlite locked");
assert.equal(o.attention, true);
assert.equal(ob({ available: false, reason: "runs directory missing" }).summary, "unavailable: runs directory missing");
o = ob({ available: true, shadow: null, live: session("live", { session: "2026-09-29", run_dir: "2026-09-29-live", phase: "WAITING_FOR_PREOPEN", heartbeat_at: "2026-09-28T13:50:00Z" }) });
assert.equal(o.sessions.live.stale, true, "an upcoming session launched early must keep its heartbeat");
assert.ok(!JSON.stringify(o).match(/U\d{5,}/));
const fetched = await readInputs({ get: async key => {
  if (key === INPUTS.event) throw new Error("private provider details");
  if (key === INPUTS.trend) return { size: 999999, json: async () => { throw new Error("must not parse"); } };
  if (key === INPUTS.runtime) return null;
  return { size: 100, uploaded: new Date(checked_at), json: async () => inputs.dial.data };
} });
assert.equal(fetched.dial.data.position, "FLAT");
assert.equal(fetched.event.error, "Report unavailable");
assert.equal(fetched.trend.error, "Report exceeds the size limit");
assert.ok(!JSON.stringify(buildStatus(fetched, now)).includes("private provider"));
const routeSource = read("functions/sleeve-status.js").replace('"./_access.js"', JSON.stringify(moduleUrl(read("functions/_access.js")))).replace('"./_sleeve-status.js"', JSON.stringify(helperUrl));
const { onRequestGet } = await import(moduleUrl(routeSource));
let reads = 0;
const denied = await onRequestGet({ request: new Request("https://example.test/sleeve-status"), env: { ACCESS_TEAM_DOMAIN: "team.cloudflareaccess.com", ACCESS_AUD: "aud", CHARTS: { get: async () => { reads++; } } } });
assert.equal(denied.status, 401);
assert.equal(reads, 0, "auth must precede all object reads");
assert.equal(denied.headers.get("Cache-Control"), "no-store");

// Exercise the browser's renderer and refresh error without touching execution controls.
const callbacks = {};
const elements = Object.fromEntries(["sleeve-cards", "sleeve-updated", "sleeve-refresh"].map(id => [id, { children: [], classList: { add() {}, remove() {} }, addEventListener(name, fn) { callbacks[name] = fn; } }]));
let fail = false;
const body = buildStatus(inputs, now);
body.sleeves[0].summary = '<script>alert("x")</script>';
body.sleeves[2].details = ["<b>detail</b>"];
const context = { document: { getElementById: id => elements[id], addEventListener(name, fn) { callbacks[name] = fn; }, hidden: false }, fetch: async () => { if (fail) throw Error("offline"); return { ok: true, json: async () => body }; }, AbortSignal, setInterval() {}, Date, Set };
vm.runInNewContext(read("site/assets/sleeve-status.js"), context);
callbacks.DOMContentLoaded();
await new Promise(resolve => setImmediate(resolve));
assert.match(elements["sleeve-cards"].innerHTML, /&lt;script&gt;/);
assert.ok(!elements["sleeve-cards"].innerHTML.includes("<script>"));
assert.match(elements["sleeve-cards"].innerHTML, /<p class="sleeve-detail">&lt;b&gt;detail&lt;\/b&gt;<\/p>/);
assert.match(elements["sleeve-cards"].innerHTML, /Open Breakout/);
assert.equal(elements["sleeve-refresh"].disabled, false);
const before = elements["sleeve-cards"].innerHTML;
elements["sleeve-cards"].children = [{}];
fail = true;
await callbacks.click();
assert.equal(elements["sleeve-cards"].innerHTML, before);
assert.match(elements["sleeve-updated"].textContent, /previous fetch/);
assert.equal(elements["sleeve-refresh"].disabled, false);
console.log("PASS sleeve status: deployment, freshness, partial failures, auth, escaping, refresh recovery");
