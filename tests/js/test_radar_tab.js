"use strict";

/* Radar tab: what may be staged, and that the stage link round-trips into the
   Execution ticket. The round-trip is the fragile part -- radar.js writes query
   params and execution.js parses them, and a rename on either side silently
   produces an empty ticket rather than an error. */

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const ASSETS = path.join(__dirname, "..", "..", "site", "assets");

function loadRadar() {
  const source = fs.readFileSync(path.join(ASSETS, "radar.js"), "utf8");
  const context = { console, document: { addEventListener() {} }, window: {},
                    location: { search: "" }, URLSearchParams, module: { exports: {} } };
  vm.createContext(context);
  vm.runInContext(source, context, { filename: "radar.js" });
  return context;
}

function loadExecution(search) {
  const source = fs.readFileSync(path.join(ASSETS, "execution.js"), "utf8");
  const context = { console, document: { addEventListener() {} }, window: {},
                    location: { search }, URLSearchParams,
                    setTimeout, clearTimeout, setInterval, clearInterval };
  vm.createContext(context);
  vm.runInContext(source, context, { filename: "execution.js" });
  return context;
}

// The real AMG plan from radar-briefings data/recs/2026-08-16.json.
const AMG = {
  ticker: "AMG", setup_grade: "A", plan_type: "breakout_stop", sector: "Financial Services",
  entry: { order: "BUY_STOP_LIMIT", trigger: 383.12, limit_cap: 400.17 },
  stop: { price: 355.84, atr_mult: 2.0 },
  targets: { t1: 437.68, t1_frac: 0.3333, trail_atr: 2.5, trail_step: "weekly" },
  time: { valid_from: "2026-08-17", valid_through: "2026-08-28", time_exit_date: "2026-11-13" },
  sizing: { shares: 33, risk_bps: 36.01, caps_hit: ["prorata_trim"] },
  ticket: "BUY STOP 383.12 cap 400.17 | ...",
};
const BNY = {
  ticker: "BNY", setup_grade: "A", plan_type: "pullback_limit",
  entry: { order: "BUY_LIMIT", limit: 98.10 },
  stop: { price: 92.00, atr_mult: 2.0 }, targets: { t1: 110.0, t1_frac: 0.3333 },
  time: { valid_from: "2026-08-17", valid_through: "2026-08-21", time_exit_date: "2026-11-13" },
  sizing: { shares: 175 },
};

const radar = loadRadar();
const { stageHref, stageBlockers, entryTypeOf, sideOf } = radar.module.exports;

// --- entry-type mapping ---------------------------------------------------------
assert.strictEqual(entryTypeOf(AMG), "STP_LMT");
assert.strictEqual(entryTypeOf(BNY), "LMT");
assert.strictEqual(entryTypeOf({ entry: { order: "BUY_MOO" } }), null,
  "an unmapped engine order must not silently become a limit");
assert.strictEqual(sideOf(AMG), "BUY");
assert.strictEqual(sideOf({ entry: { order: "SELL_LIMIT" } }), "SELL");

// --- what may be staged ---------------------------------------------------------
// Length, not deepStrictEqual: the array comes from the vm realm, so its
// prototype differs from the host's and deep-equality fails on identical values.
assert.strictEqual(stageBlockers(AMG, 2).length, 0,
  `the live AMG plan is stageable, got ${JSON.stringify(stageBlockers(AMG, 2))}`);
assert.ok(stageBlockers({ ...AMG, entry: { order: "BUY_STOP_LIMIT", trigger: 383.12 } }, 2)
  .some((b) => b.includes("no limit cap")), "a stop-limit without its cap must be refused");
assert.ok(stageBlockers({ ...AMG, sizing: { shares: 0 } }, 2)
  .some((b) => b.includes("no share count")));
assert.ok(stageBlockers({ ...AMG, stop: {} }, 2).some((b) => b.includes("no stop price")));
assert.ok(stageBlockers({ ...AMG, entry: { order: "BUY_MOO" } }, 2)
  .some((b) => b.includes("no ticket equivalent")));
assert.ok(stageBlockers(AMG, 40).some((b) => b.includes("40d old")),
  "a stale recs vintage blocks staging");
// An expired fill window is the one that would quietly place a dead plan.
assert.ok(stageBlockers({ ...AMG, time: { ...AMG.time, valid_through: "2020-01-01" } }, 2)
  .some((b) => b.includes("fill window closed")));
assert.ok(stageBlockers({ ...AMG, time: { ...AMG.time, valid_from: "2099-01-01" } }, 2)
  .some((b) => b.includes("not live until")));

// --- the round-trip -------------------------------------------------------------
const href = stageHref(AMG, "2026-08-16");
assert.ok(href.startsWith("execution.html?"), href);
const exec = loadExecution(href.slice("execution.html".length));
const staged = JSON.parse(vm.runInContext("JSON.stringify(radarStage)", exec));
assert.ok(staged, "execution.js must recognise the radar stage link");
assert.strictEqual(staged.sym, "AMG");
assert.strictEqual(staged.side, "BUY");
assert.strictEqual(staged.type, "STP_LMT");
// Every level arrives byte-identical to the engine's plan -- nothing derived.
assert.strictEqual(staged.entry, 383.12);
assert.strictEqual(staged.cap, 400.17);
assert.strictEqual(staged.stop, 355.84);
assert.strictEqual(staged.target, 437.68);
assert.strictEqual(staged.qty, 33);
assert.strictEqual(staged.exp, "2026-08-28");
assert.strictEqual(staged.ts, "2026-11-13");
assert.strictEqual(staged.strat, "Momentum_Radar");
assert.strictEqual(staged.refdate, "2026-08-16");

// A limit plan carries no cap, and the entry comes from `limit` not `trigger`.
const bnyExec = loadExecution(stageHref(BNY, "2026-08-16").slice("execution.html".length));
const bnyStaged = JSON.parse(vm.runInContext("JSON.stringify(radarStage)", bnyExec));
assert.strictEqual(bnyStaged.type, "LMT");
assert.strictEqual(bnyStaged.entry, 98.10);
assert.strictEqual(bnyStaged.cap, null);

// The seasonal deep link must NOT be picked up by the radar parser, or vice versa.
const seasonal = loadExecution("?stage=1&sym=USO&side=BUY&win=10&atr=2&px=50");
assert.strictEqual(JSON.parse(vm.runInContext("JSON.stringify(radarStage)", seasonal)), null);
assert.strictEqual(JSON.parse(vm.runInContext("JSON.stringify(stage)", exec)), null);

console.log("PASS radar tab: entry mapping, stage blockers, and the verbatim stage round-trip");
