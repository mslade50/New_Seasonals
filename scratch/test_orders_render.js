/* Headless render smoke test for orders.js — shims a minimal DOM + fetch,
   loads common.js + orders.js, fires initOrders against the sample payload,
   and asserts the rendered HTML contains the expected states. */
const fs = require("fs");
const vm = require("vm");

function mkEl() {
  return { innerHTML: "", textContent: "", addEventListener() {},
           querySelector() { return mkEl(); }, querySelectorAll() { return []; },
           appendChild() {}, dataset: {}, style: {} };
}
const document = {
  _els: {},
  getElementById(id) { return (this._els[id] ||= mkEl()); },
  addEventListener(ev, fn) { this._ready = fn; },
  createElement() { return mkEl(); },
};
const sample = JSON.parse(fs.readFileSync("scratch/morning_orders.sample.json", "utf8"));
const fetch = async (path) => {
  const hit = String(path).includes("morning-orders");
  return { ok: hit, status: hit ? 200 : 404, json: async () => sample };
};
const sandbox = { document, fetch, console, setTimeout };
sandbox.window = sandbox;
vm.createContext(sandbox);
const src = fs.readFileSync("site/assets/common.js", "utf8") + "\n" +
            fs.readFileSync("site/assets/orders.js", "utf8");
vm.runInContext(src, sandbox);

(async () => {
  await document._ready();          // initOrders
  const html = document._els["content"].innerHTML;
  const want = ["Primary (TWS)", "PA / small", "USO", "AAPL", "GS",
                "ENTRY NOT WORKING", "all 3 legs", "FILLED", "Risk by strategy",
                "$1,045", "13.9"];
  let ok = true;
  for (const w of want) { const hit = html.includes(w); if (!hit) ok = false;
    console.log((hit ? "OK   " : "MISS ") + w); }
  console.log(`\nrendered ${html.length} chars · ${ok ? "ALL PASS" : "FAILURES"}`);
  process.exit(ok ? 0 : 1);
})();
