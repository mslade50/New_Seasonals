"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(
  path.join(__dirname, "..", "..", "site", "assets", "options.js"),
  "utf8",
);
const context = {
  console,
  document: { addEventListener() {} },
  window: {},
  location: { search: "" },
  URLSearchParams,
  setTimeout,
  clearTimeout,
  setInterval,
  clearInterval,
};
vm.runInNewContext(source, context, { filename: "options.js" });
vm.runInNewContext(
  "globalThis.setExecState = (status, book) => { state.status = status; state.book = book; };",
  context,
);

const now = 2_000_000_000_000;
const status = { online: true, session_id: "session-new" };
const book = (mode, at, session = "session-new") => ({ mode, at, _broker_session_id: session });
context.setExecState(status, book("live", now - 1_000));
assert.strictEqual(context.execMode(now), "live");

context.setExecState(status, book("live", now - 100_000));
assert.strictEqual(context.execMode(now), "unknown");

context.setExecState(status, book("live", now / 1_000 - 1));
assert.strictEqual(context.execMode(now), "live");

context.setExecState({ ...status, online: false }, book("live", now - 1_000));
assert.strictEqual(context.execMode(now), "unknown");

context.setExecState(status, book("dry-run", now - 1_000));
assert.strictEqual(context.execMode(now), "dry-run");

context.setExecState(status, book("live", now - 1_000, "session-old"));
assert.strictEqual(context.execMode(now), "unknown");

console.log("PASS options execution mode: live and dry-run both require an online, fresh book");
