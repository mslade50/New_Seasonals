"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(
  path.join(__dirname, "..", "..", "site", "assets", "execution.js"),
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
vm.runInNewContext(source, context, { filename: "execution.js" });

const now = 2_000_000_000_000;
assert.strictEqual(context.deriveExecMode({ mode: "live", at: 1 }, { online: false }, now), "live");
assert.strictEqual(context.deriveExecMode({ mode: "dry-run", at: now - 1_000 }, { online: true }, now), "dry-run");
assert.strictEqual(context.deriveExecMode({ mode: "dry-run", at: now - 100_000 }, { online: true }, now), "unknown");
assert.strictEqual(context.deriveExecMode({ mode: "dry-run", at: now - 1_000 }, { online: false }, now), "unknown");
assert.strictEqual(context.deriveExecMode(null, { online: true }, now), "unknown");

// Agent snapshots may carry Python epoch seconds; broker fallbacks carry JS
// epoch milliseconds. Both must render and gate against the same real age.
assert.strictEqual(context.epochMs(now / 1_000), now);
assert.strictEqual(context.epochMs(now), now);
assert.strictEqual(context.bookAgeMs({ at: now / 1_000 - 30 }, now), 30_000);
assert.strictEqual(context.deriveExecMode(
  { mode: "dry-run", at: now / 1_000 - 30 }, { online: true }, now,
), "dry-run");
assert.strictEqual(context.deriveExecMode(
  { mode: "dry-run", at: now / 1_000 - 100 }, { online: true }, now,
), "unknown");

console.log("PASS execution mode derivation: seconds/ms timestamps, live, fresh dry-run, stale, offline, and missing book");
