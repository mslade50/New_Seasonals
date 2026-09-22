"use strict";

/* Execution tab — structured position-action lock rejection, the clear-lock
 * control, and the RISK_ACK confirmation wording.
 *
 * Audit: artifacts/recon_2026-09-17/site_execution_audit.md
 *   §4.2 / C2 — the lock accounted for 13 of 22 ring failures and the site
 *               could create it but not clear it.
 *   cause table row 1 — all 17 RISK_ACK bounces were confirmed and filled, so
 *               the prompt has to read as a confirmation, not an error.
 */

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(
  path.join(__dirname, "..", "..", "site", "assets", "execution.js"),
  "utf8",
);
const elements = new Map();
function stubElement(id) {
  if (!elements.has(id)) elements.set(id, { id, value: "", textContent: "", innerHTML: "", focus() {} });
  return elements.get(id);
}
const context = {
  console,
  document: {
    addEventListener() {},
    getElementById: (id) => (elements.has(id) ? elements.get(id) : null),
    querySelectorAll: () => [],
  },
  window: {},
  location: { search: "" },
  URLSearchParams,
  setTimeout,
  clearTimeout,
  setInterval,
  clearInterval,
  structuredClone: (v) => JSON.parse(JSON.stringify(v)),
};
vm.createContext(context);
vm.runInContext(source, context, { filename: "execution.js" });
const run = (expr) => vm.runInContext(expr, context);

run(`
  state.account = "pa";
  state.status = { online: true };
  state.book = { at: Date.now(), mode: "live", accounts: [{ key: "pa", positions: [], orders: [] }] };
  lastConfirm = ""; lastCommand = null; confirmAnswer = true;
  confirm = (m) => { lastConfirm = m; return confirmAnswer; };
  alert = (m) => { lastAlert = m; };
  lastAlert = "";
  sendCommand = (type, payload, msgId, ctx) => { lastCommand = { type, payload, msgId, ctx }; return Promise.resolve("id"); };
  fmt = { num: (v) => String(v), money: (v) => "$" + String(v), pct: (v) => String(v) };
`);

/* ---- 1. the structured lock reason parses, and only with a full identity ---- */
const LOCK = {
  symbol: "UNH", action_type: "close_resize", action_id: "act-0001",
  created_at: "2026-09-16T10:26:00-04:00",
  discrepancy: "current exits cover 0 units but position holds 625",
};
const blocked = { id: "c1", type: "close_resize", account: "pa", state: "rejected",
  result: { ok: false, detail: "live gate: cancellation needs reconciliation", lock: LOCK } };
run(`state.commands = [${JSON.stringify(blocked)}];`);

const parsed = JSON.parse(run(`JSON.stringify(lockRejection(state.commands[0]))`));
assert.strictEqual(parsed.symbol, "UNH");
assert.strictEqual(parsed.action_id, "act-0001");
assert.strictEqual(parsed.account, "pa");

// A refusal with no resolvable identity renders as plain detail, never a guess.
assert.strictEqual(run(`lockRejection({state:"rejected", result:{lock:{symbol:"UNH"}}})`), null);
assert.strictEqual(run(`lockRejection({state:"rejected", result:{detail:"nope"}})`), null);
assert.strictEqual(run(`lockRejection({state:"working", result:{lock:${JSON.stringify(LOCK)}}})`), null);
// `reason` as an object is accepted too (the agent may nest it either way).
assert.strictEqual(
  JSON.parse(run(`JSON.stringify(lockRejection({state:"rejected", account:"pa", result:{reason:${JSON.stringify(LOCK)}}}))`)).symbol,
  "UNH");

/* ---- 2. Activity renders the named blocker + a Clear lock control ---- */
let activity = run("renderActivity()");
assert.match(activity, /Blocked by unresolved close_resize on UNH from 2026-09-16T10:26:00-04:00: current exits cover 0 units but position holds 625/);
assert.match(activity, /Clear lock&hellip;/);
assert.match(activity, /openLockResolve\(/);
// The underlying detail string is still shown.
assert.match(activity, /cancellation needs reconciliation/);

/* ---- 3. the inline form requires an operator note ---- */
run(`openLockResolve(${JSON.stringify(LOCK)})`);
assert.strictEqual(run("lockResolve.id"), "act-0001");
activity = run("renderActivity()");
assert.match(activity, /id="lock_note"/);
assert.match(activity, /Operator note \(required, 8\+ characters\)/);
assert.match(activity, /submitLockResolve\(/);

// Too short: nothing is sent and the form stays open with a complaint.
stubElement("lock_note").value = "ok";
run(`lastCommand = null; submitLockResolve(${JSON.stringify(LOCK)})`);
assert.strictEqual(run("lastCommand"), null, "a short note must not send a command");
assert.strictEqual(run("lockResolve.id"), "act-0001", "the form stays open on a rejected note");
assert.match(run("lockResolve.msg"), /at least 8 characters/);
// Whitespace does not count as a note.
stubElement("lock_note").value = "          ";
run(`lastCommand = null; submitLockResolve(${JSON.stringify(LOCK)})`);
assert.strictEqual(run("lastCommand"), null);

/* ---- 4. resolve payload shape ---- */
stubElement("lock_note").value = "  checked TWS: exits cover the full 625, nothing half-placed  ";
run(`lastCommand = null; confirmAnswer = false; submitLockResolve(${JSON.stringify(LOCK)})`);
assert.strictEqual(run("lastCommand"), null, "a declined confirm sends nothing");
run(`confirmAnswer = true; submitLockResolve(${JSON.stringify(LOCK)})`);
const sent = JSON.parse(run("JSON.stringify(lastCommand)"));
assert.strictEqual(sent.type, "position_action_resolve");
assert.deepStrictEqual(sent.payload, {
  account: "pa", symbol: "UNH", action_id: "act-0001",
  operator_note: "checked TWS: exits cover the full 625, nothing half-placed",
});
assert.strictEqual(sent.ctx.account, "pa", "the command routes to the locked account, not the tab default");
assert.strictEqual(run("lockResolve.id"), null, "the form closes once the command is away");
// It is a vocabulary the site owns, and it places no order.
assert.strictEqual(run(`MUTATING_COMMANDS.has("position_action_resolve")`), true);
assert.match(run("lastConfirm"), /No order is placed/);

/* ---- 5. the resolve response snapshot is summarised ---- */
run(`state.commands = [{ id: "c2", type: "position_action_resolve", account: "pa", state: "resolved",
  result: { ok: true, detail: "resolved", snapshot: { positions: [1, 2], open_orders: [1, 2, 3, 4, 5] } } }];`);
const resolved = run("renderActivity()");
assert.match(resolved, /snapshot recorded: 2 positions · 5 open orders/);
run(`state.commands[0].result.snapshot = { positions: [1], open_orders: [] };`);
assert.match(run("renderActivity()"), /snapshot recorded: 1 position · 0 open orders/);
// No snapshot, no claim.
run(`state.commands[0].result = { ok: true, detail: "resolved" };`);
assert.doesNotMatch(run("renderActivity()"), /snapshot recorded/);

/* ---- 5b. "resolved" is a first-class badge, not grey raw text ---- */
// The agent renames its success state executed -> resolved for this command, so
// without a map entry the Activity row shows a lowercase grey "resolved" that
// reads like an unknown state.
const resolvedBadge = run(`stateBadge("resolved")`);
assert.match(resolvedBadge, /RESOLVED/, "resolved must render an uppercase label");
assert.match(resolvedBadge, /#3ddb8f/, "a cleared lock placed no order — it is a success tone");
assert.doesNotMatch(resolvedBadge, />resolved</, "the raw state string must not be the label");
// It is styled the same way every other known state is.
assert.match(resolvedBadge, /font-weight:600/);
// Unknown states still fall through to the grey raw-text default.
assert.match(run(`stateBadge("some_new_state")`), /#9aa3b2[^]*some_new_state/);
// The badge reaches the Activity table for a real resolve row.
run(`state.commands = [{ id: "c3", type: "position_action_resolve", account: "pa", state: "resolved",
  result: { ok: true, detail: "cleared", snapshot: { positions: [], open_orders: [] } } }];`);
assert.match(run("renderActivity()"), /RESOLVED/);

/* ---- 5c. `reason` is the detail fallback when the agent omits detail ---- */
// The documented resolve response is {state, reason, snapshot}; `detail` is an
// extra the agent happens to send today. Do not depend on it.
run(`state.commands = [{ id: "c4", type: "position_action_resolve", account: "pa", state: "resolved",
  result: { ok: true, reason: "cleared act-0001 after verifying TWS", snapshot: { positions: [1], open_orders: [] } } }];`);
let reasonOnly = run("renderActivity()");
assert.match(reasonOnly, /cleared act-0001 after verifying TWS/, "reason must be shown when detail is absent");
assert.match(reasonOnly, /snapshot recorded: 1 position · 0 open orders/);
// An empty-string detail is still no detail.
run(`state.commands[0].result.detail = "";`);
assert.match(run("renderActivity()"), /cleared act-0001 after verifying TWS/);
// When both are present, detail wins and reason is not duplicated.
run(`state.commands[0].result.detail = "resolved: live book now holds 0 units";`);
reasonOnly = run("renderActivity()");
assert.match(reasonOnly, /resolved: live book now holds 0 units/);
assert.doesNotMatch(reasonOnly, /cleared act-0001 after verifying TWS/);
// A rejected resolve with only a reason still says why, and never renders
// "pending" or the bare state.
run(`state.commands = [{ id: "c5", type: "position_action_resolve", account: "pa", state: "rejected",
  result: { ok: false, reason: "2 working order(s) still tied to act-0001: 901, 902" } }];`);
const rejectedReason = run("renderActivity()");
assert.match(rejectedReason, /2 working order\(s\) still tied to act-0001: 901, 902/);
assert.doesNotMatch(rejectedReason, /pending/);

/* ---- 6. RISK_ACK reads as a confirmation step, same mechanics ---- */
const entry = { action: "BUY", quantity: 80, symbol: "RTX", entry: 141.5, stop: null,
  sec_type: "STK", entry_type: "LMT" };
run(`
  lastConfirm = ""; lastCommand = null; confirmAnswer = true;
  riskAckPending.set("c9", { type: "entry_bracket", payload: ${JSON.stringify(entry)}, account: "pa", dryRun: false });
  state.commands = [{ id: "c9", type: "entry_bracket", account: "pa", state: "rejected",
    result: { ok: false, detail: "RISK_ACK_REQUIRED", fill: { needs_risk_ack: true, est_risk: 1303, est_bps: 109 } } }];
  checkRiskAck();
`);
const prompt = run("lastConfirm");
assert.match(prompt, /This entry has no stop and risks \$1303 = 109 bps of NLV \(2xATR basis\)\./);
assert.match(prompt, /Confirm to place BUY 80 RTX @ 141\.5 with NO STOP on pa\./);
assert.doesNotMatch(prompt, /SECONDARY RISK APPROVAL/);
assert.doesNotMatch(prompt, /\[WARN\]/);
assert.doesNotMatch(prompt, /Approve and resend/);
// Same mechanics: the identical payload is resent with risk_ack.
const resent = JSON.parse(run("JSON.stringify(lastCommand)"));
assert.strictEqual(resent.type, "entry_bracket");
assert.strictEqual(resent.payload.risk_ack, true);
assert.strictEqual(resent.payload.quantity, 80);
assert.strictEqual(resent.ctx.account, "pa");

// Declining still sends nothing.
stubElement("cmdMsg");
run(`
  lastCommand = null; confirmAnswer = false;
  riskAckPending.set("c9", { type: "entry_bracket", payload: ${JSON.stringify(entry)}, account: "pa", dryRun: false });
  checkRiskAck();
`);
assert.strictEqual(run("lastCommand"), null);
assert.match(stubElement("cmdMsg").textContent, /confirmation declined/);

// A stopped entry (the uncapped-futures gate) keeps the same confirmation shape.
run(`
  lastConfirm = ""; confirmAnswer = true;
  riskAckPending.set("c8", { type: "entry_bracket", payload: { action: "BUY", quantity: 1, symbol: "MES", entry: 6500, stop: 6400, sec_type: "FUT" }, account: "primary", dryRun: false });
  state.commands = [{ id: "c8", type: "entry_bracket", account: "primary", state: "rejected",
    result: { ok: false, fill: { needs_risk_ack: true, est_risk: 500, est_bps: 80 } } }];
  checkRiskAck();
`);
assert.match(run("lastConfirm"), /This entry risks \$500 = 80 bps of NLV \(defined stop basis\)\./);
assert.match(run("lastConfirm"), /Confirm to place BUY 1 MES @ 6500 with stop 6400 on primary\./);

// A rejection for some OTHER reason never becomes a risk prompt.
run(`
  lastConfirm = ""; lastCommand = null;
  riskAckPending.set("c7", { type: "entry_bracket", payload: ${JSON.stringify(entry)}, account: "pa", dryRun: false });
  state.commands = [{ id: "c7", type: "entry_bracket", account: "pa", state: "rejected", result: { ok: false, detail: "no short availability" } }];
  checkRiskAck();
`);
assert.strictEqual(run("lastConfirm"), "");
assert.strictEqual(run("lastCommand"), null);

console.log("PASS execution lock resolve: structured rejection, clear-lock note gate, resolve payload, snapshot summary, RISK_ACK confirmation wording");
