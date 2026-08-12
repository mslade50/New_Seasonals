"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const context = {
  console,
  document: { addEventListener() {} },
  window: { setTimeout },
};
context.globalThis = context;
vm.runInNewContext(fs.readFileSync(
  path.join(__dirname, "..", "..", "site", "assets", "heatmaps.js"), "utf8"),
context, { filename: "heatmaps.js" });

const m = context.HeatmapMath;

const ranked = m.expandingRank([1, 2, 3, 2], 1);
assert.deepStrictEqual(Array.from(ranked, value => Number(value.toFixed(6))), [100, 100, 100, 62.5]);

const edges = [0, 50, 100];
assert.strictEqual(m.binIndex(0, edges), 0);
assert.strictEqual(m.binIndex(100, edges), 1); // maximum edge must not be discarded

const rows = [];
for (let i = 0; i < 40; i++) rows.push({ x: i % 10, y: Math.floor(i / 10), z: i % 2 ? 1 : -1 });
const grid = m.buildGrid(rows, "x", "y", "z", 4, 2, 0);
assert.strictEqual(grid.clean.length, 40);
assert(grid.z.flat().some(value => typeof value === "number"));

const pairs = Array.from({ length: 30 }, (_, i) => ({
  date: `2026-01-${String(i + 1).padStart(2, "0")}`,
  a: i / 100,
  b: i / 50,
}));
const corr = m.rollingCorrelation(pairs, 21).filter(row => typeof row.value === "number");
assert(corr.length > 0);
assert(Math.abs(corr[corr.length - 1].value - 1) < 1e-12);

console.log("PASS heatmap math: causal ranks, max-edge bins, support grid, and rolling correlation");
