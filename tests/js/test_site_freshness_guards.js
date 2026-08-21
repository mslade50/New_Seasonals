"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

function load(name) {
  const source = fs.readFileSync(
    path.join(__dirname, "..", "..", "site", "assets", name), "utf8");
  const context = { console, document: { addEventListener() {} }, window: {} };
  vm.createContext(context);
  vm.runInContext(source, context, { filename: name });
  return context;
}

const portfolio = load("portfolio.js");
const portfolioMeta = { build_id: "build-1", built_at: "2026-08-06 12:00 UTC", payloads: {
  strategy_daily: true, positions: true, exposure: true, trade_mtm: true, health: true,
} };
const portfolioHealth = {
  build_id: "build-1",
  built_at: "2026-08-06 12:00 UTC",
  prev_td: "2026-08-05",
  artifacts: { ledger: { status: "fresh" }, master_prices: { status: "fresh" } },
};
const positions = {
  asof: "2026-08-06", positions: [{ Ticker: "SPY", Days_To_Time_Stop: 2 }],
};
assert.strictEqual(
  portfolio.portfolioFreshnessError(portfolioMeta, portfolioHealth, positions), null);
portfolioHealth.build_id = "build-2";
assert.match(
  portfolio.portfolioFreshnessError(portfolioMeta, portfolioHealth, positions), /match/i);
portfolioHealth.build_id = "build-1";
portfolioHealth.artifacts.ledger.status = "stale";
assert.match(
  portfolio.portfolioFreshnessError(portfolioMeta, portfolioHealth, positions), /ledger/i);
portfolioHealth.artifacts.ledger.status = "fresh";
positions.positions[0].Days_To_Time_Stop = -1;
assert.match(
  portfolio.portfolioFreshnessError(portfolioMeta, portfolioHealth, positions), /time stop/i);

const seasonal = load("seasonal.js");
const seasonalMeta = { build_id: "build-1", built_at: "2026-08-06 12:00 UTC", payloads: { ideas: true, health: true } };
const seasonalHealth = {
  build_id: "build-1",
  built_at: "2026-08-06 12:00 UTC",
  prev_td: "2026-08-05", artifacts: { ideas: { status: "fresh" } },
};
const ideas = { meta: { asof: "2026-08-05" }, candidates: [] };
assert.strictEqual(
  seasonal.seasonalFreshnessError(seasonalMeta, seasonalHealth, ideas), null);
seasonalHealth.artifacts.ideas.status = "stale";
assert.match(
  seasonal.seasonalFreshnessError(seasonalMeta, seasonalHealth, ideas), /stale/i);

const signals = load("signals.js");
const signalMeta = { build_id: "build-1", built_at: "2026-08-06 12:00 UTC", payloads: { signals: true, health: true } };
const signalHealth = { build_id: "build-1", built_at: "2026-08-06 12:00 UTC", artifacts: { signals: { status: "fresh" } } };
const staged = { fetched_at: "2026-08-06 12:00 UTC", tabs: {} };
assert.strictEqual(signals.signalFreshnessError(signalMeta, signalHealth, staged), null);
signalMeta.payloads.signals = false;
assert.match(signals.signalFreshnessError(signalMeta, signalHealth, staged), /unavailable/i);

for (const name of ["portfolio.js", "seasonal.js", "signals.js", "pipeline.js"]) {
  const source = fs.readFileSync(
    path.join(__dirname, "..", "..", "site", "assets", name), "utf8");
  assert.ok(!source.includes('fetchJSONOrNull("data/health.json")'),
    `${name} must consume the health record embedded in meta.json`);
}
for (const name of ["portfolio.js", "seasonal.js", "signals.js"]) {
  const source = fs.readFileSync(
    path.join(__dirname, "..", "..", "site", "assets", name), "utf8");
  assert.ok(!source.includes("const buildGap"),
    `${name} must use exact build identity, not browser-specific timestamp parsing`);
  assert.ok(source.includes("health.build_id"), `${name} must compare exact build IDs`);
}

console.log("PASS site freshness guards");
