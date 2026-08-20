"use strict";

const assert = require("assert");
const path = require("path");
const { pathToFileURL } = require("url");

(async () => {
  const moduleUrl = pathToFileURL(
    path.join(__dirname, "..", "..", "functions", "_execution_policy.mjs"),
  ).href;
  const { validateCommandRequest } = await import(moduleUrl);

  const now = 2_000_000_000_000;
  const status = { online: true, session_id: "session-new" };
  const book = {
    mode: "live",
    at: now,
    _broker_session_id: "session-new",
    accounts: [
      {
        key: "primary",
        nlv: 1_000_000,
        positions: [{
          symbol: "SPY", sec_type: "STK", currency: "USD", con_id: 756733,
          position: 100, market_price: 100, avg_cost: 99,
        }],
        orders: [
          { symbol: "SPY", con_id: 756733, order_id: 91, perm_id: 901, qty: 100 },
          { symbol: "SPY", con_id: 756733, action: "SELL", order_type: "STP", aux: 98, qty: 100 },
        ],
      },
      { key: "pa", nlv: 100_000, positions: [], orders: [] },
    ],
  };
  const id = "11111111-2222-4333-8444-555555555555";
  const entry = {
    id,
    type: "entry_bracket",
    account: "primary",
    payload: {
      symbol: "SPY", sec_type: "STK", currency: "USD", action: "BUY", quantity: 100,
      entry_type: "LMT", entry: 100, stop: 98, target: 104,
    },
  };
  const liveEnv = {
    EXEC_LIVE_ENABLED: "1",
    EXEC_LIVE_TYPES: "entry_bracket,close_only,cancel,option_spread",
    EXEC_LIVE_ACCOUNTS: "primary",
  };
  const check = (body, overrides = {}) => validateCommandRequest(body, {
    env: overrides.env || {},
    status: Object.prototype.hasOwnProperty.call(overrides, "status") ? overrides.status : status,
    book: Object.prototype.hasOwnProperty.call(overrides, "book") ? overrides.book : book,
    now,
  });

  // The absence of dry_run must never mean live.
  let result = check(entry);
  assert.strictEqual(result.ok, true);
  assert.strictEqual(result.command.dry_run, true);

  result = check({ ...entry, dry_run: false });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /kill switch/i);

  result = check({ ...entry, dry_run: false }, { env: liveEnv });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /atomic aggregate risk reservation/i);

  const futureEntry = {
    ...entry,
    dry_run: false,
    payload: {
      ...entry.payload,
      symbol: "ES", sec_type: "FUT", quantity: 1, entry: 5000, stop: 4990,
      target: 5020, fut_multiplier: 50, fut_min_tick: 0.25,
      fut_expiry: "202609", exchange: "CME", fut_ib_symbol: "ES", fut_trading_class: "ES",
    },
  };
  result = check(futureEntry, { env: liveEnv });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /instrument FUT is not armed/i);
  result = check(futureEntry, { env: { ...liveEnv, EXEC_LIVE_INSTRUMENTS: "STK,FUT" } });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /atomic aggregate risk reservation/i);

  const marketEntry = {
    ...entry,
    dry_run: false,
    payload: { ...entry.payload, entry_type: "MKT" },
  };
  result = check(marketEntry, { env: liveEnv });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /entry type MKT is not armed/i);
  result = check(marketEntry, { env: { ...liveEnv, EXEC_LIVE_ENTRY_TYPES: "LMT,MKT" } });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /atomic aggregate risk reservation/i);

  result = check({ ...entry, dry_run: false }, {
    env: liveEnv,
    book: { ...book, at: now - 90_001 },
  });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /stale/i);

  result = check({ ...entry, dry_run: false }, { env: liveEnv, status: { online: false } });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /offline/i);

  result = check({ ...entry, dry_run: false }, {
    env: liveEnv,
    status: { online: true, session_id: "session-new" },
    book: { ...book, _broker_session_id: "session-old" },
  });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /same agent session/i);

  result = check({ ...entry, dry_run: false }, {
    env: liveEnv,
    book: { ...book, mode: "dry-run" },
  });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /does not confirm live/i);

  result = check({ ...entry, type: "arbitrary_python", dry_run: true });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /type is not allowed/i);

  result = check({ ...entry, payload: { ...entry.payload, shell: "do something" } });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /unsupported field/i);

  result = check({ ...entry, payload: { ...entry.payload, quantity: "100" } });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /positive integer/i);

  result = check({ ...entry, payload: { ...entry.payload, stop: true } });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /stop must be > 0/i);

  result = check({ ...entry, dry_run: false, payload: { ...entry.payload, stop: null } }, { env: liveEnv });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /live entries require a defined stop/i);

  result = check({
    ...entry,
    dry_run: false,
    payload: { ...entry.payload, quantity: 30_000 },
  }, { env: liveEnv });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /(risk|notional) exceeds server cap/i);

  result = check({
    id,
    type: "close_only",
    account: "primary",
    dry_run: false,
    payload: {
      symbol: "SPY", sec_type: "STK", currency: "USD", con_id: 756733,
      expected_position: 100, action: "SELL",
      fraction: 0.5, order_type: "MKT", tif: "DAY", outside_rth: false,
    },
  }, { env: liveEnv });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /protective-order validation/i);

  result = check({
    id,
    type: "exit_attach",
    account: "primary",
    dry_run: false,
    payload: {
      symbol: "SPY", sec_type: "STK", currency: "USD", con_id: 756733,
      expected_position: 100, stop: 98, outside_rth: false,
    },
  }, {
    env: { ...liveEnv, EXEC_LIVE_TYPES: `${liveEnv.EXEC_LIVE_TYPES},exit_attach` },
  });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /protective-order validation/i);

  const mutationEnv = {
    ...liveEnv,
    EXEC_LIVE_TYPES: `${liveEnv.EXEC_LIVE_TYPES},add_to_position,modify`,
  };
  const add = {
    id,
    type: "add_to_position",
    account: "primary",
    dry_run: false,
    payload: {
      symbol: "SPY", sec_type: "STK", currency: "USD", con_id: 756733,
      expected_position: 100, fraction: 0.5, order_type: "MKT",
    },
  };
  result = check(add, { env: mutationEnv });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /atomic aggregate risk reservation/i);
  result = check(add, {
    env: mutationEnv,
    book: {
      ...book,
      accounts: [{ ...book.accounts[0], orders: [] }, book.accounts[1]],
    },
  });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /protective price stop/i);

  const modifyPrice = {
    id,
    type: "modify",
    account: "primary",
    dry_run: false,
    payload: { symbol: "SPY", perm_id: 901, order_id: 91, new_limit: 101 },
  };
  result = check(modifyPrice, { env: mutationEnv });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /price\/stop modification is disabled/i);
  result = check({
    ...modifyPrice,
    payload: { symbol: "SPY", perm_id: 901, order_id: 91, new_qty: 50 },
  }, { env: mutationEnv });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /protective-order validation/i);

  // The fast row-action flatten omits tif/outside_rth and defaults agent-side.
  result = check({
    id,
    type: "flatten",
    account: "primary",
    payload: {
      symbol: "SPY", con_id: 756733, expected_position: 100,
      fraction: 0.5, order_type: "MKT",
    },
  });
  assert.strictEqual(result.ok, true);

  result = check({
    id,
    type: "cancel",
    account: "primary",
    dry_run: false,
    payload: { scope: "symbol", symbol: "SPY" },
  }, { env: liveEnv });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /symbol-wide cancel is disabled/i);

  result = check({
    id,
    type: "cancel",
    account: "primary",
    dry_run: false,
    payload: { scope: "order", symbol: "SPY", perm_id: 901, order_id: 91 },
  }, { env: liveEnv });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /protective-order validation/i);

  const option = {
    id,
    type: "option_spread",
    account: "primary",
    dry_run: false,
    payload: {
      symbol: "SPY", action: "BUY", quantity: 4, limit: 3, tif: "DAY",
      structure: "call_spread", debit_risk: 3, risk_per_unit: 3, credit: false,
      legs: [
        { side: "BUY", right: "C", expiry: "20260918", strike: 600, ratio: 1, con_id: 1001 },
        { side: "SELL", right: "C", expiry: "20260918", strike: 605, ratio: 1, con_id: 1002 },
      ],
    },
  };
  result = check(option, { env: { ...liveEnv, EXEC_MAX_NEW_RISK_BPS: "10" } });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /risk exceeds server cap/i);

  result = check({
    id,
    type: "close_only",
    account: "primary",
    dry_run: false,
    payload: {
      symbol: "SPY", sec_type: "STK", currency: "USD", expected_position: 100,
      fraction: 1, order_type: "MKT",
    },
  }, { env: liveEnv });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /require con_id/i);

  // Python seconds and JavaScript milliseconds are both accepted.
  result = check(entry, { book: { ...book, at: now / 1000 - 30 } });
  assert.strictEqual(result.ok, true);

  console.log("PASS exec command policy: explicit live intent, freshness, allowlists, schemas, and risk caps");
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
