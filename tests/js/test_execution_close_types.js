"use strict";

/* Primary Close adjusts exits for partial and full closes. Legacy close types
   remain available, and PA retains its partial-only resize behavior. */

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const ASSETS = path.join(__dirname, "..", "..", "site", "assets");
const source = fs.readFileSync(path.join(ASSETS, "execution.js"), "utf8");

function freshContext() {
  const context = {
    console,
    document: { addEventListener() {}, getElementById: () => null },
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
  return context;
}

const position = { symbol: "UNH", sec_type: "STK", expiry: "", con_id: 42,
  position: 139, avg_cost: 400 };

/* ---- 1. all three close types are offered and recognised as mutations ---- */
{
  const ctx = freshContext();
  for (const t of ["close_only", "close_resize", "flatten"]) {
    assert.strictEqual(vm.runInContext(`CLOSE_COMMANDS.has(${JSON.stringify(t)})`, ctx), true,
      `${t} must share the close ticket fields`);
    assert.strictEqual(vm.runInContext(`MUTATING_COMMANDS.has(${JSON.stringify(t)})`, ctx), true,
      `${t} must be gated as a mutation`);
    // Owner policy: missing snapshots do not disable manual controls.
    vm.runInContext("state.book = null; state.status = { online: false };", ctx);
    assert.strictEqual(vm.runInContext(`mutationBlocked(${JSON.stringify(t)})`, ctx), false);
  }
  assert.match(source, /<option value="close_resize">/, "close_resize must be in the type dropdown");
}

/* ---- 2. close_resize builds a close_only-shaped payload, with the action ---- */
{
  const ctx = freshContext();
  vm.runInContext(`
    state.account = "primary";
    state.book = { accounts: [{ key: "primary", positions: [${JSON.stringify(position)}], orders: [] }] };
    ticketDraft.fl_position = { account: "primary", ...positionIdentity(${JSON.stringify(position)}) };
    val = (id) => ({ f_symbol: "UNH", fl_qty: "70", fl_pct: "", fl_type: "MKT", fl_tif: "DAY" })[id];
  `, ctx);
  const payload = JSON.parse(vm.runInContext('JSON.stringify(ticketPayload("close_resize"))', ctx));
  assert.deepStrictEqual(payload, {
    symbol: "UNH", order_type: "MKT", tif: "DAY", outside_rth: false,
    sec_type: "STK", expiry: null, expected_position: 139, con_id: 42,
    qty: 70, action: "SELL", readd: false,
  });
}

/* ---- 3. a typed symbol still resolves an exact con_id ----
   close_resize/close_only bind to a conId agent-side, so a symbol typed
   straight into the ticket (no "Close…" prefill) has to pick one up from the
   book, and must REFUSE to guess when the symbol names two contracts. */
{
  const ctx = freshContext();
  vm.runInContext(`
    state.account = "primary";
    state.book = { accounts: [{ key: "primary", positions: [${JSON.stringify(position)}], orders: [] }] };
    val = (id) => ({ f_symbol: "UNH", fl_qty: "70", fl_pct: "", fl_type: "MKT", fl_tif: "DAY" })[id];
  `, ctx);
  const resolved = JSON.parse(vm.runInContext('JSON.stringify(ticketPayload("close_resize"))', ctx));
  assert.strictEqual(resolved.con_id, 42, "a unique symbol match supplies the con_id");
  assert.strictEqual(resolved.action, "SELL");

  // Two contract months under one symbol: never guess (futures-roll hazard).
  vm.runInContext(`
    state.book = { accounts: [{ key: "primary", positions: [
      { symbol: "MES", sec_type: "FUT", expiry: "202609", con_id: 1, position: 2 },
      { symbol: "MES", sec_type: "FUT", expiry: "202612", con_id: 2, position: 1 }
    ], orders: [] }] };
    val = (id) => ({ f_symbol: "MES", fl_qty: "1", fl_pct: "", fl_type: "MKT", fl_tif: "DAY" })[id];
  `, ctx);
  const ambiguous = JSON.parse(vm.runInContext('JSON.stringify(ticketPayload("close_resize"))', ctx));
  assert.strictEqual(ambiguous.con_id, undefined, "an ambiguous symbol must not be resolved");
}

/* ---- 4. Primary Close accepts 100%; PA retains its legacy guard ---- */
{
  const ctx = freshContext();
  const fields = { f_symbol: "UNH", fl_qty: "", fl_pct: "100", fl_type: "MKT", fl_tif: "DAY" };
  vm.runInContext(`
    state.account = "primary";
    state.book = { accounts: [{ key: "primary", positions: [${JSON.stringify(position)}], orders: [] }] };
    cmdTypeValue = "close_resize";
    document.getElementById = (id) => (id === "cmdType" ? { value: cmdTypeValue } : null);
    val = (id) => (${JSON.stringify(fields)})[id];
  `, ctx);
  const full = vm.runInContext("flattenWarnings()", ctx);
  assert.strictEqual(JSON.stringify(full), "[]");
  vm.runInContext('state.account = "pa"; state.book.accounts[0].key = "pa";', ctx);
  assert.ok(vm.runInContext("flattenWarnings()", ctx).some((w) => /PARTIAL close/.test(w)));
  vm.runInContext('state.account = "primary"; state.book.accounts[0].key = "primary";', ctx);

  // 50% of the same position is fine. (Arrays cross the vm realm boundary, so
  // compare through JSON rather than deepStrictEqual.)
  const warnsJson = (c) => vm.runInContext("JSON.stringify(flattenWarnings())", c);
  vm.runInContext('val = (id) => ({ f_symbol: "UNH", fl_qty: "", fl_pct: "50", fl_type: "MKT", fl_tif: "DAY" })[id];', ctx);
  assert.strictEqual(warnsJson(ctx), "[]");

  // ...and the same 100% is fine for flatten, which is how you close in full.
  vm.runInContext('cmdTypeValue = "flatten"; val = (id) => (' + JSON.stringify(fields) + ")[id];", ctx);
  assert.strictEqual(warnsJson(ctx), "[]");
}

/* ---- 5. the fast Trim buttons pick the safe command for the position ---- */
{
  const ctx = freshContext();
  const withStop = `{ symbol: "UNH", sec_type: "STK", con_id: 42, action: "SELL",
    order_type: "STP", qty: 139, aux: 380 }`;
  vm.runInContext(`
    state.account = "primary";
    state.status = { online: true };
    state.book = { at: Date.now(), mode: "live",
      accounts: [{ key: "primary", positions: [${JSON.stringify(position)}], orders: [${withStop}] }] };
    confirm = (m) => { lastConfirm = m; return true; };
    alert = (m) => { lastAlert = m; };
    sendCommand = (type, payload) => { lastCommand = { type, payload }; };
    fmt = { num: (v) => String(v), money: (v) => String(v), pct: (v) => String(v) };
    lastConfirm = ""; lastAlert = ""; lastCommand = null;
    execPartialClose(${JSON.stringify(position)}, 0.5);
  `, ctx);
  const guarded = JSON.parse(vm.runInContext("JSON.stringify({ lastConfirm, lastCommand })", ctx));
  assert.strictEqual(guarded.lastCommand.type, "close_resize",
    "a position with working exits trims via close_resize, never flatten");
  assert.strictEqual(guarded.lastCommand.payload.qty, 70);
  assert.match(guarded.lastConfirm, /shrink to 69 BEFORE the close/);
  assert.match(guarded.lastConfirm, /nothing is cancelled/);

  // Bare position: nothing to shrink, so close_only is the honest command.
  vm.runInContext(`
    state.book.accounts[0].orders = [];
    lastConfirm = ""; lastCommand = null;
    execPartialClose(${JSON.stringify(position)}, 0.5);
  `, ctx);
  const bare = JSON.parse(vm.runInContext("JSON.stringify({ lastConfirm, lastCommand })", ctx));
  assert.strictEqual(bare.lastCommand.type, "close_only");
  assert.match(bare.lastConfirm, /no working orders to touch/);

  // Too small to split: refuse rather than send a full close as a "trim".
  vm.runInContext(`
    lastAlert = ""; lastCommand = null;
    execPartialClose({ symbol: "UNH", sec_type: "STK", con_id: 42, position: 1 }, 0.5);
  `, ctx);
  const tiny = JSON.parse(vm.runInContext("JSON.stringify({ lastAlert, lastCommand })", ctx));
  assert.strictEqual(tiny.lastCommand, null);
  assert.match(tiny.lastAlert, /too small/);
}

console.log("PASS execution close types: close_only / close_resize / flatten tickets, con_id resolution, partial-only gate, safe trim routing");
