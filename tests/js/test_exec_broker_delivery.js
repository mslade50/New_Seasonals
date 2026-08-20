"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { webcrypto } = require("crypto");

class MemoryStorage {
  constructor(entries = {}) {
    this.rows = new Map(Object.entries(entries));
    this.alarmAt = null;
  }
  async get(key) { return this.rows.get(key); }
  async put(key, value) { this.rows.set(key, structuredClone(value)); }
  async delete(key) { this.rows.delete(key); }
  async getAlarm() { return this.alarmAt; }
  async setAlarm(value) { this.alarmAt = Number(value); }
  async list({ prefix } = {}) {
    return new Map([...this.rows].filter(([key]) => !prefix || key.startsWith(prefix)));
  }
}

class FakeDurableObject {
  constructor(ctx, env) { this.ctx = ctx; this.env = env; }
}

function loadBroker() {
  const filename = path.join(__dirname, "..", "..", "execution-broker", "src", "index.js");
  let source = fs.readFileSync(filename, "utf8");
  source = source
    .replace('import { DurableObject } from "cloudflare:workers";', "const DurableObject = FakeDurableObject;")
    .replace(
      'import { HEARTBEAT_STALE_MS, validHmacHex, validateBrokerCommand } from "./command_policy.mjs";',
      "const HEARTBEAT_STALE_MS = 25000; " +
      "const validHmacHex = async () => true; " +
      "const validateBrokerCommand = () => ({ok:true});",
    )
    .replace("export class ExecBroker", "globalThis.ExecBroker = class ExecBroker")
    .replace("export default {", "globalThis.worker = {");
  const context = {
    FakeDurableObject, URL, Request, Response, TextEncoder, TextDecoder,
    crypto: webcrypto, structuredClone, console,
  };
  vm.runInNewContext(source, context, { filename });
  return context.ExecBroker;
}

(async () => {
  const ExecBroker = loadBroker();
  const storage = new MemoryStorage({
    recent_commands: [{ id: "cmd-1", state: "queued", result: null }],
    scheduled_commands: [],
  });
  const ctx = { storage, getWebSockets: () => [] };
  const broker = new ExecBroker(ctx, {});
  const pending = {
    id: "cmd-1", signed: "{}", sig: "sig", attempts: 0,
    expires_at: Date.now() + 60_000, last_session_id: null,
  };
  await storage.put("pending_command:cmd-1", pending);

  const failed = await broker._deliverPending(pending, {
    send() { throw new Error("closed"); },
  }, "session-1");
  assert.strictEqual(failed.ok, false);
  assert.strictEqual(failed.state, "queued");
  assert.strictEqual((await storage.get("recent_commands"))[0].state, "queued");
  assert.strictEqual((await storage.get("pending_command:cmd-1")).attempts, 1);

  const sentFrames = [];
  const delivered = await broker._deliverPending(
    await storage.get("pending_command:cmd-1"),
    { send(frame) { sentFrames.push(JSON.parse(frame)); } },
    "session-1",
  );
  assert.strictEqual(delivered.ok, true);
  assert.strictEqual(delivered.state, "sent");
  assert.strictEqual(sentFrames[0].type, "command");
  assert.strictEqual((await storage.get("recent_commands"))[0].state, "sent");
  assert.ok(await storage.get("pending_command:cmd-1"));

  await broker.webSocketMessage({}, JSON.stringify({
    type: "command_receipt", id: "cmd-1", policy_version: "2026-08-20.4",
  }));
  assert.strictEqual((await storage.get("recent_commands"))[0].state, "received");
  assert.ok(await storage.get("pending_command:cmd-1"));

  await broker.webSocketMessage({}, JSON.stringify({
    type: "result", id: "cmd-1", state: "dry_run", ok: true, detail: "previewed",
  }));
  const final = (await storage.get("recent_commands"))[0];
  assert.strictEqual(final.state, "dry_run");
  assert.strictEqual(final.result.ok, true);
  assert.strictEqual(await storage.get("pending_command:cmd-1"), undefined);

  // End-to-end command endpoint: a synchronous socket failure is a 503, but
  // the same command id remains retryable and is sent once the socket recovers.
  const command = {
    id: "11111111-2222-4333-8444-555555555555",
    type: "echo", account: "primary", dry_run: true, payload: {},
    policy_version: "2026-08-20.4", created_at: Date.now(), expires_at: Date.now() + 60_000,
  };
  const signed = JSON.stringify(command);
  const deliveryFrames = [];
  const socket = {
    fail: true,
    deserializeAttachment() { return { sessionId: "session-live", lastSeenAt: Date.now() }; },
    send(frame) {
      if (this.fail) throw new Error("socket closed");
      deliveryFrames.push(JSON.parse(frame));
    },
  };
  const endpointStorage = new MemoryStorage({
    last_seen: Date.now(),
    book: { mode: "dry-run", at: Date.now(), _broker_session_id: "session-live" },
  });
  const endpointCtx = { storage: endpointStorage, getWebSockets: () => [socket] };
  const endpoint = new ExecBroker(endpointCtx, { COMMAND_SECRET: "secret" });
  const makeRequest = (signedBody = signed) => new Request("https://broker.test/command", {
    method: "POST",
    headers: { Authorization: "Bearer secret", "Content-Type": "application/json" },
    body: JSON.stringify({ signed: signedBody, sig: "valid" }),
  });
  let response = await endpoint.fetch(makeRequest());
  assert.strictEqual(response.status, 503);
  assert.strictEqual((await response.json()).state, "queued");
  assert.ok(await endpointStorage.get(`pending_command:${command.id}`));
  assert.strictEqual((await endpointStorage.get("recent_commands"))[0].state, "queued");
  assert.ok(endpointStorage.alarmAt > command.expires_at);

  socket.fail = false;
  const refreshedCommand = {
    ...command, created_at: Date.now() + 61_000, expires_at: Date.now() + 121_000,
    payload: {}, // a new object/order proves canonical intent comparison
  };
  const refreshedSigned = JSON.stringify(refreshedCommand);
  response = await endpoint.fetch(makeRequest(refreshedSigned));
  const retry = await response.json();
  assert.strictEqual(response.status, 200);
  assert.strictEqual(retry.retried, true);
  assert.strictEqual(retry.state, "sent");
  assert.strictEqual(deliveryFrames.length, 1);
  assert.strictEqual(deliveryFrames[0].signed, refreshedSigned);

  response = await endpoint.fetch(makeRequest(refreshedSigned));
  const acceptedRetry = await response.json();
  assert.strictEqual(acceptedRetry.deduped, true);
  assert.strictEqual(acceptedRetry.retried, false);
  assert.strictEqual(acceptedRetry.state, "sent");
  assert.strictEqual(deliveryFrames.length, 1, "an accepted send must never be repeated");

  const collision = { ...refreshedCommand, payload: { note: "different intent" } };
  response = await endpoint.fetch(makeRequest(JSON.stringify(collision)));
  assert.strictEqual(response.status, 409);
  assert.match((await response.json()).error, /different intent/i);

  const reconnectFrames = [];
  await endpoint._redeliverPending(
    { send(frame) { reconnectFrames.push(frame); } },
    "replacement-session",
    { mode: "dry-run", at: Date.now(), _broker_session_id: "replacement-session" },
  );
  assert.strictEqual(reconnectFrames.length, 0);
  assert.strictEqual((await endpointStorage.get("recent_commands"))[0].state, "unknown");
  assert.match((await endpointStorage.get("recent_commands"))[0].result.detail, /verify TWS/i);
  assert.strictEqual(await endpointStorage.get(`pending_command:${command.id}`), undefined);

  // If delivery certainty is lost and no reconnect/result arrives, expiry is
  // fail-loud UNKNOWN rather than an indefinitely reassuring SENT badge.
  const uncertainStorage = new MemoryStorage({
    recent_commands: [{ id: "uncertain", state: "received", result: null }],
    scheduled_commands: [],
    "pending_command:uncertain": {
      id: "uncertain", signed: "{}", sig: "sig", expires_at: Date.now() - 1,
      attempts: 1, last_session_id: "old-session",
    },
  });
  const uncertain = new ExecBroker({ storage: uncertainStorage, getWebSockets: () => [] }, {});
  await uncertain.alarm();
  assert.strictEqual((await uncertainStorage.get("recent_commands"))[0].state, "unknown");
  assert.match((await uncertainStorage.get("recent_commands"))[0].result.detail, /verify TWS/i);
  assert.strictEqual(await uncertainStorage.get("pending_command:uncertain"), undefined);

  console.log("PASS broker delivery: failed sends remain retryable and results close the outbox");
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
