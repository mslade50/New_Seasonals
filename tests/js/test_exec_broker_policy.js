"use strict";

const assert = require("assert");
const path = require("path");
const { pathToFileURL } = require("url");

(async () => {
  const moduleUrl = pathToFileURL(
    path.join(__dirname, "..", "..", "execution-broker", "src", "command_policy.mjs"),
  ).href;
  const { validHmacHex, validateBrokerCommand } = await import(moduleUrl);

  const now = 2_000_000_000_000;
  const command = {
    id: "11111111-2222-4333-8444-555555555555",
    type: "entry_bracket",
    account: "primary",
    dry_run: true,
    payload: { symbol: "SPY", sec_type: "STK", entry_type: "LMT" },
    policy_version: "2026-08-20.1",
    created_at: now - 1_000,
    expires_at: now + 59_000,
  };
  const state = {
    now,
    lastSeen: now - 1_000,
    socketCount: 1,
    socketSession: "session-new",
    book: { mode: "live", at: now - 1_000, _broker_session_id: "session-new" },
  };

  let result = validateBrokerCommand(command, state);
  assert.strictEqual(result.ok, true);

  result = validateBrokerCommand({ ...command, dry_run: false }, state);
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /kill switch/i);

  const armed = {
    EXEC_LIVE_ENABLED: "1",
    EXEC_LIVE_TYPES: "entry_bracket",
    EXEC_LIVE_ACCOUNTS: "primary",
  };
  result = validateBrokerCommand({ ...command, dry_run: false }, { ...state, env: armed });
  assert.strictEqual(result.ok, true);

  result = validateBrokerCommand({
    ...command,
    dry_run: false,
    payload: { symbol: "ES", sec_type: "FUT", entry_type: "LMT" },
  }, { ...state, env: armed });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /instrument FUT is not armed/i);

  result = validateBrokerCommand({ ...command, dry_run: false }, {
    ...state,
    env: armed,
    book: { mode: "dry-run", at: now - 1_000, _broker_session_id: "session-new" },
  });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /does not confirm live/i);

  result = validateBrokerCommand(command, { ...state, lastSeen: now - 30_000 });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /heartbeat stale/i);

  result = validateBrokerCommand(command, {
    ...state,
    book: { mode: "live", at: now / 1_000 - 91, _broker_session_id: "session-new" },
  });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /book missing, stale/i);

  result = validateBrokerCommand(command, {
    ...state,
    book: { mode: "live", at: now - 1_000, _broker_session_id: "session-old" },
  });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /active agent session/i);

  result = validateBrokerCommand(command, { ...state, socketCount: 2 });
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /session ambiguous/i);

  result = validateBrokerCommand({ ...command, id: "not-a-uuid" }, state);
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /envelope rejected/i);

  result = validateBrokerCommand({ ...command, policy_version: "legacy" }, state);
  assert.strictEqual(result.ok, false);
  assert.match(result.error, /envelope rejected/i);

  const secret = "dedicated-command-secret";
  const message = JSON.stringify(command);
  const key = await crypto.subtle.importKey(
    "raw", new TextEncoder().encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"],
  );
  const rawSig = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(message));
  const signature = [...new Uint8Array(rawSig)].map((b) => b.toString(16).padStart(2, "0")).join("");
  assert.strictEqual(await validHmacHex(secret, message, signature), true);
  assert.strictEqual(await validHmacHex(secret, message + " ", signature), false);
  assert.strictEqual(await validHmacHex(secret, message, "not-hex"), false);

  console.log("PASS broker command policy: fresh heartbeat/book and independent live arming required");
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
