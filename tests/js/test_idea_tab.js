"use strict";

/* Idea Check tab: badge logic, in-flight polling rule, and that the render
   puts idea/result text in via textContent only (never innerHTML). */

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(path.join(__dirname, "..", "..", "site", "assets", "idea.js"), "utf8");
assert.ok(!/innerHTML/.test(source.replace(/\/\*[\s\S]*?\*\//, "")), "idea.js must not use innerHTML");
const context = { console, document: { addEventListener() {} }, module: { exports: {} }, setTimeout, clearTimeout };
vm.createContext(context);
vm.runInContext(source, context, { filename: "idea.js" });
const { ideaBadge, anyInflight, renderIdeas } = context.module.exports;

const done = (verdict) => ({ result: { status: "done", verdict } });
assert.deepStrictEqual(ideaBadge({ result: null }).label, "queued");
assert.strictEqual(ideaBadge({ result: null }).inflight, true);
assert.strictEqual(ideaBadge({ result: { status: "running", verdict: null } }).label, "running");
assert.strictEqual(ideaBadge({ result: { status: "running" } }).inflight, true);
for (const [v, cls] of [["KILL", "kill"], ["SURVIVES", "survives"], ["NEAR-MISS", "near"], ["NEEDS-INFO", "info"]]) {
  const b = ideaBadge(done(v));
  assert.strictEqual(b.label, v);
  assert.strictEqual(b.cls, cls);
  assert.strictEqual(b.inflight, false);
}
assert.strictEqual(ideaBadge({ result: { status: "error" } }).label, "error");
assert.strictEqual(ideaBadge({ result: { status: "done", verdict: null } }).label, "error");
assert.strictEqual(anyInflight([done("KILL"), { result: { status: "error" } }]), false);
assert.strictEqual(anyInflight([done("KILL"), { result: null }]), true);
assert.strictEqual(anyInflight([]), false);

// Minimal DOM: records children and textContent; innerHTML assignment throws.
function node(tag) {
  const n = { tag, className: "", children: [], _t: "" };
  Object.defineProperty(n, "innerHTML", { set() { throw new Error("innerHTML used"); } });
  Object.defineProperty(n, "textContent", {
    get() { return n._t + n.children.map((c) => c.textContent).join(""); },
    set(v) { n._t = v; if (v === "") n.children = []; },
  });
  n.appendChild = (c) => { n.children.push(c); return c; };
  return n;
}
const doc = { createElement: node };
const box = node("div");
const evil = "<img src=x onerror=alert(1)>";
renderIdeas(doc, box, [
  { id: "a", text: evil, submitted_at: "2026-10-02T14:00:00Z", result: {
    status: "done", verdict: "KILL", headline: "Dead on arrival", numbers: ["n=12", "t=0.4"],
    tweaks: ["hold 5d"], body_md: "# Detail\nline", error: null } },
  { id: "b", text: "queued one", submitted_at: "2026-10-02T14:05:00Z", result: null },
]);
const text = box.textContent;
for (const s of [evil, "KILL", "Dead on arrival", "n=12", "t=0.4", "hold 5d", "# Detail\nline", "queued", "queued one"])
  assert.ok(text.includes(s), `missing ${s}`);
assert.strictEqual(box.children.length, 2);
const empty = node("div");
renderIdeas(doc, empty, []);
assert.ok(empty.textContent.includes("No checks yet"));
console.log("test_idea_tab.js passed");
