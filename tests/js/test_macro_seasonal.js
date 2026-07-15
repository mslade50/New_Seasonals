"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const assets = path.join(__dirname, "..", "..", "site", "assets");
const context = {
  console,
  document: { addEventListener() {} },
  window: { location: { hash: "", pathname: "/seasonal.html", search: "" }, history: { replaceState() {} } },
};
// Both scripts share one global scope on the page; loading them into a single
// context catches top-level declaration collisions.
for (const name of ["seasonality.js", "macro_seasonal.js"]) {
  vm.runInNewContext(fs.readFileSync(path.join(assets, name), "utf8"), context, { filename: name });
}
const m = context.MacroSeasonalMath;

assert.strictEqual(m.macroReturnClass(95), "hi-red");     // extended
assert.strictEqual(m.macroReturnClass(10), "hi-green");   // washed out
assert.strictEqual(m.macroReturnClass(50), "");
assert.strictEqual(m.macroReturnClass(null), "");
assert.strictEqual(m.macroSznlClass(90), "hi-green");     // seasonally strong
assert.strictEqual(m.macroSznlClass(10), "hi-red");
assert.strictEqual(m.macroSznlClass(50), "");

assert.strictEqual(m.macroCell(null, "ret"), "<td>—</td>");
assert.strictEqual(m.macroCell(1234.5, "price"), "<td>1234.50</td>");
assert.strictEqual(m.macroCell(92.34, "ret"), '<td class="hi-red">92.3</td>');
assert.strictEqual(m.macroCell(92.34, "sznl"), '<td class="hi-green">92.3</td>');
assert.strictEqual(m.macroCell(50, "sznl"), "<td>50.0</td>");

console.log("PASS macro seasonal: no global collisions, highlight rules, cell formatting");
