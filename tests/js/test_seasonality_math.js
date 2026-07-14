"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(
  path.join(__dirname, "..", "..", "site", "assets", "seasonality.js"),
  "utf8",
);
const context = {
  console,
  document: { addEventListener() {} },
  window: { location: { hash: "", pathname: "/seasonal.html", search: "" }, history: { replaceState() {} } },
};
vm.runInNewContext(source, context, { filename: "seasonality.js" });
const m = context.SeasonalityMath;

const normalized = m.normalizeSeasonalityPayload({
  d: ["20240102", "20240103", "20250102"],
  c: [100, 102, 103],
  a: [null, 2, 2],
});
assert.deepStrictEqual(Array.from(normalized, r => r.date), ["2024-01-02", "2024-01-03", "2025-01-02"]);
const enriched = m.enrichSeasonalityRows(normalized);
assert.deepStrictEqual(Array.from(enriched, r => r.day), [1, 2, 1]);
assert(Math.abs(enriched[1].atrReturn - 1) < 1e-12);

const binary = new ArrayBuffer(8 + 3 * 12);
const binaryView = new DataView(binary);
for (const [i, char] of Array.from("SLB1").entries()) binaryView.setUint8(i, char.charCodeAt(0));
binaryView.setUint32(4, 3, true);
for (let i = 0; i < 3; i++) {
  binaryView.setInt32(8 + i * 4, 19724 + i, true);
  binaryView.setFloat32(8 + 12 + i * 4, 100 + i, true);
  binaryView.setFloat32(8 + 24 + i * 4, i === 0 ? NaN : 2, true);
}
const decoded = m.decodeSeasonalityBuffer(binary);
assert.strictEqual(decoded[0].date, "2024-01-02");
assert.strictEqual(decoded[0].atr, null);
assert.strictEqual(decoded[2].close, 102);

assert.strictEqual(m.recencyWeight(2004, 2024, 20), 0.5);
assert.strictEqual(m.weightedMean([1, 3], [1, 1]), 2);
assert.strictEqual(m.weightedMedian([1, 10, 20], [1, 5, 1]), 10);
assert.strictEqual(m.isCycleYear(2024, "Election"), true);
assert.strictEqual(m.isCycleYear(2022, "Midterm"), true);
assert.strictEqual(m.isCycleYear(2023, "Election"), false);

const pathRows = [
  { year: 2020, day: 1, atrReturn: 1 }, { year: 2020, day: 2, atrReturn: 0.5 },
  { year: 2021, day: 1, atrReturn: 1 }, { year: 2021, day: 2, atrReturn: 0.5 },
];
const pathOut = m.calculateSeasonalPath(pathRows, "All Years", "atrReturn", 0, 2022);
assert.strictEqual(pathOut.length, 2);
assert.strictEqual(pathOut[0].value, 1);
assert.strictEqual(pathOut[1].value, 1.5);

const markerRows = [
  { year: 2024, day: 1, date: "2024-07-12" },
  { year: 2024, day: 2, date: "2024-07-15" },
];
assert.strictEqual(m.markerDayFor(markerRows, 2024, new Date(2024, 6, 14), null), 2);
assert.strictEqual(m.markerDayFor(markerRows, 2024, new Date(2024, 6, 14), 151), 151);

const snapshotRows = [];
for (const year of [2020, 2021]) {
  for (let day = 1; day <= 25; day++) {
    snapshotRows.push({
      year, day, date: `${year}-01-${String(day).padStart(2, "0")}`,
      close: 100 + day, atr: 2, dailyPct: 0.01,
    });
  }
}
const snapshots = m.forwardSnapshots(snapshotRows, 2, 2022);
assert.strictEqual(snapshots.length, 2);
assert.strictEqual(snapshots[0].ret5, 2.5);
assert.strictEqual(snapshots[0].date21, "2020-01-23");
const stats = m.calculateStats(snapshots, 2022, 20);
assert.strictEqual(stats.n, 2);
assert.strictEqual(stats.mean5, 2.5);
assert.strictEqual(stats.positive5, 100);

const rankRows = [];
for (const year of [2020, 2021, 2022, 2023]) {
  for (let day = 1; day <= 35; day++) {
    rankRows.push({ year, day, close: 100 + day + (year - 2020), logReturn: 0.001, atrReturn: 0.1,
      date: `${year}-02-${String(Math.min(day, 28)).padStart(2, "0")}`, atr: 2, dailyPct: 0.001 });
  }
}
const rank = m.calculateSeasonalRank(rankRows, "All Years", 2024, 20);
assert.strictEqual(rank.size >= 253, true);
assert.strictEqual(Array.from(rank.values()).every(v => Number.isFinite(v) && v >= 0 && v <= 100), true);

console.log("PASS seasonality math: payload, weighting, cycle paths, marker, forward stats, and rank profile");
