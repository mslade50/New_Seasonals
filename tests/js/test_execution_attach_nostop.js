"use strict";
/* Stop-optional entry ticket + exit_attach ticket + RISK_ACK secondary-approval
   resend (2026-07-27). Runs execution.js in a vm with a minimal DOM stub. */

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(
  path.join(__dirname, "..", "..", "site", "assets", "execution.js"),
  "utf8",
);
const fields = {};   // id -> value; getElementById returns a live-ish stub
const confirms = [];
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
  fmt: { money: (v) => `$${v}`, num: (v) => String(v), pct: (v) => String(v) },
  confirm: (msg) => { confirms.push(msg); return context.__confirmAnswer; },
  __confirmAnswer: true,
};
vm.createContext(context);
vm.runInContext(source, context, { filename: "execution.js" });

function setFields(map) {
  Object.keys(fields).forEach((k) => delete fields[k]);
  for (const [k, v] of Object.entries(map)) {
    fields[k] = typeof v === "object" ? v : { value: String(v) };
  }
}
const run = (code) => vm.runInContext(code, context);
const runJSON = (code) => JSON.parse(run(`JSON.stringify(${code})`));

// ---- entry ticket: stop is optional ------------------------------------------
run(`state.book = { accounts: [{ key: "primary", positions: [], orders: [] }] }`);
setFields({
  f_sectype: "STK", f_symbol: "USO", f_action: "BUY", f_qty: "10",
  f_entry_type: "LMT", f_entry: "50", f_stop: "", f_target: "", f_timestop: "", f_expiry: "",
});
assert.deepStrictEqual(runJSON("bracketWarnings()"), [],
  "blank stop must not block the entry ticket");
const noStop = runJSON(`ticketPayload("entry_bracket")`);
assert.strictEqual(noStop.stop, null, "blank stop serializes as null");
assert.strictEqual(noStop.target, null);
assert.strictEqual(noStop.entry_type, "LMT");

// MKT/MOO/MOC carry a required risk reference, but never a limit-entry expiry.
setFields({ f_sectype: "STK", f_symbol: "USO", f_action: "BUY", f_qty: "10",
  f_entry_type: "MKT", f_entry: "50", f_stop: "48", f_target: "55",
  f_timestop: "", f_expiry: "" });
assert.deepStrictEqual(runJSON("bracketWarnings()"), []);
const marketEntry = runJSON(`ticketPayload("entry_bracket")`);
assert.strictEqual(marketEntry.entry_type, "MKT");
assert.strictEqual(marketEntry.entry, 50);
assert.strictEqual(marketEntry.expiry, null);

setFields({ f_sectype: "STK", f_symbol: "USO", f_action: "BUY", f_qty: "10",
  f_entry_type: "MOO", f_entry: "50", f_stop: "48", f_target: "55",
  f_timestop: "", f_expiry: "" });
assert.deepStrictEqual(runJSON("bracketWarnings()"), []);
const mooEntry = runJSON(`ticketPayload("entry_bracket")`);
assert.strictEqual(mooEntry.entry_type, "MOO");
assert.strictEqual(mooEntry.entry, 50);
assert.strictEqual(mooEntry.expiry, null);

setFields({ f_sectype: "STK", f_symbol: "USO", f_action: "BUY", f_qty: "10",
  f_entry_type: "MOC", f_entry: "50", f_stop: "48", f_target: "55",
  f_timestop: "", f_expiry: "" });
assert.deepStrictEqual(runJSON("bracketWarnings()"), []);
assert.strictEqual(runJSON(`ticketPayload("entry_bracket")`).entry_type, "MOC");

setFields({ f_sectype: "FUT", f_symbol: "ES", f_action: "BUY", f_qty: "1",
  f_entry_type: "MOC", f_entry: "5000", f_stop: "4990", f_target: "5020",
  f_timestop: "", f_expiry: "", f_futexp: "202609", f_futexch: "CME" });
assert.ok(runJSON("bracketWarnings()").some((w) => w.includes("MOC entry supports stocks only")));

setFields({ f_sectype: "CASH", f_symbol: "EUR", f_currency: "USD", f_action: "BUY", f_qty: "1000",
  f_entry_type: "MOO", f_entry: "1.15", f_stop: "1.14", f_target: "1.17",
  f_timestop: "", f_expiry: "" });
assert.ok(runJSON("bracketWarnings()").some((w) => w.includes("MOO entry does not support FX")));

setFields({ f_sectype: "STK", f_symbol: "USO", f_action: "BUY", f_qty: "10",
  f_entry_type: "MKT", f_entry: "", f_stop: "", f_target: "",
  f_timestop: "", f_expiry: "" });
assert.ok(runJSON("bracketWarnings()").some((w) => w.includes("reference price required")));

// explicit bad stop still blocks
setFields({ f_sectype: "STK", f_symbol: "USO", f_action: "BUY", f_qty: "10",
  f_entry: "50", f_stop: "0", f_target: "", f_timestop: "", f_expiry: "" });
assert.ok(runJSON("bracketWarnings()").some((w) => w.includes("stop must be > 0")));

// no stop + wrong-side target is still caught client-side
setFields({ f_sectype: "STK", f_symbol: "USO", f_action: "BUY", f_qty: "10",
  f_entry: "50", f_stop: "", f_target: "45", f_timestop: "", f_expiry: "" });
assert.ok(runJSON("bracketWarnings()").some((w) => w.includes("BUY needs entry < target")));

// stopped entries keep the full ordering gate (regression)
setFields({ f_sectype: "STK", f_symbol: "USO", f_action: "BUY", f_qty: "10",
  f_entry: "50", f_stop: "55", f_target: "", f_timestop: "", f_expiry: "" });
assert.ok(runJSON("bracketWarnings()").some((w) => w.includes("BUY needs stop < entry")));

// ---- exit_attach ticket ------------------------------------------------------
run(`state.book = { accounts: [{ key: "primary", positions: [
  { symbol: "AAPL", sec_type: "STK", con_id: 111, position: 100, avg_cost: 200,
    market_price: 205 }], orders: [] }] }`);
setFields({ f_symbol: "AAPL", f_stop: "190", f_target: "230", f_timestop: "",
  ea_rth: { checked: false } });
assert.deepStrictEqual(runJSON("attachWarnings()"), [], "valid attach is sendable");
const attach = runJSON(`ticketPayload("exit_attach")`);
assert.strictEqual(attach.symbol, "AAPL");
assert.strictEqual(attach.con_id, 111);
assert.strictEqual(attach.stop, 190);
assert.strictEqual(attach.target, 230);
assert.strictEqual(attach.time_stop, null);
assert.ok(!("expected_position" in attach), "attach sizes to the live held qty");

// at least one leg required
setFields({ f_symbol: "AAPL", f_stop: "", f_target: "", f_timestop: "",
  ea_rth: { checked: false } });
assert.ok(runJSON("attachWarnings()").some((w) => w.includes("at least one")));

// wrong-side stop for a long is blocked
setFields({ f_symbol: "AAPL", f_stop: "210", f_target: "", f_timestop: "",
  ea_rth: { checked: false } });
assert.ok(runJSON("attachWarnings()").some((w) => w.includes("wrong side")));

// short position: ordering inverts (stop above, target below)
run(`state.book.accounts[0].positions[0].position = -100`);
setFields({ f_symbol: "AAPL", f_stop: "220", f_target: "180", f_timestop: "",
  ea_rth: { checked: false } });
assert.deepStrictEqual(runJSON("attachWarnings()"), []);
setFields({ f_symbol: "AAPL", f_stop: "180", f_target: "220", f_timestop: "",
  ea_rth: { checked: false } });
assert.ok(runJSON("attachWarnings()").length >= 1, "inverted short legs blocked");
run(`state.book.accounts[0].positions[0].position = 100`);

// working closing exit blocks attach
run(`state.book.accounts[0].orders = [{ symbol: "AAPL", sec_type: "STK",
  con_id: 111, action: "SELL", order_type: "STP", qty: 100, aux: 190 }]`);
setFields({ f_symbol: "AAPL", f_stop: "190", f_target: "", f_timestop: "",
  ea_rth: { checked: false } });
assert.ok(runJSON("attachWarnings()").some((w) => w.includes("already working")));
run(`state.book.accounts[0].orders = []`);

// ---- mutation gate covers the new type ---------------------------------------
assert.ok(run(`MUTATING_COMMANDS.has("exit_attach")`));
run(`state.book = null; state.status = { online: false }`);   // unknown mode
assert.strictEqual(run(`mutationBlocked("exit_attach")`), true);

// ---- RISK_ACK secondary approval resend --------------------------------------
run(`state.book = { accounts: [] }; state.status = { online: true }`);
const sent = [];
context.__capture = (type, payload) => { sent.push({ type, payload }); return Promise.resolve("new-id"); };
run(`sendCommand = (type, payload, msgId) => __capture(type, payload)`);
run(`riskAckPending.set("cmd-1", { type: "entry_bracket",
  payload: { symbol: "USO", action: "BUY", quantity: 10, entry: 50, stop: null } })`);
run(`state.commands = [{ id: "cmd-1", state: "rejected",
  result: { fill: { needs_risk_ack: true, est_risk: 5000, est_bps: 80 } } }]`);
confirms.length = 0;
context.__confirmAnswer = true;
run("checkRiskAck()");
assert.strictEqual(confirms.length, 1, "secondary-approval prompt shown once");
assert.ok(confirms[0].includes("80 bps"), "prompt carries the machine's estimate");
assert.strictEqual(sent.length, 1, "approved intent resent");
assert.strictEqual(sent[0].payload.risk_ack, true, "resend carries risk_ack");
assert.strictEqual(sent[0].payload.stop, null, "payload otherwise identical");
assert.strictEqual(run(`riskAckPending.size`), 0, "pending cleared after prompt");

// declined approval sends nothing
run(`riskAckPending.set("cmd-2", { type: "entry_bracket",
  payload: { symbol: "USO", action: "BUY", quantity: 10, entry: 50, stop: null } })`);
run(`state.commands = [{ id: "cmd-2", state: "rejected",
  result: { fill: { needs_risk_ack: true } } }]`);
fields.cmdMsg = { textContent: "" };
sent.length = 0;
context.__confirmAnswer = false;
run("checkRiskAck()");
assert.strictEqual(sent.length, 0, "declined approval sends nothing");

// a rejection WITHOUT needs_risk_ack never prompts
run(`riskAckPending.set("cmd-3", { type: "entry_bracket", payload: { stop: null } })`);
run(`state.commands = [{ id: "cmd-3", state: "rejected",
  result: { fill: null } }]`);
confirms.length = 0;
context.__confirmAnswer = true;
run("checkRiskAck()");
assert.strictEqual(confirms.length, 0, "plain rejection never prompts for an ack");
assert.strictEqual(run(`riskAckPending.size`), 0);

// non-rejected terminal states just clear the pending entry
run(`riskAckPending.set("cmd-4", { type: "entry_bracket", payload: { stop: null } })`);
run(`state.commands = [{ id: "cmd-4", state: "dry_run", result: {} }]`);
run("checkRiskAck()");
assert.strictEqual(run(`riskAckPending.size`), 0);

console.log("test_execution_attach_nostop.js OK");
