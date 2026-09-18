/* Headless render test for the execution dashboard — mock status/book/commands,
   fire initExecution, assert the Positions / Orders / Activity panels render. */
const fs = require("fs"), vm = require("vm");

const NOW = 1782800000000;
const BOOK = {
  at: NOW - 2000,
  accounts: [
    { key: "primary", label: "Primary (TWS)", error: null, nlv: 758420.0,
      positions: [
        { symbol: "USO", sec_type: "STK", expiry: "", position: 692, avg_cost: 104.8, market_price: 108.1, market_value: 74805.2, unrealized_pnl: 2283.6, currency: "USD" },
        { symbol: "MBT", sec_type: "FUT", expiry: "202605", position: -3, avg_cost: 420.5, market_price: 410.2, market_value: -6153.0, unrealized_pnl: 154.5, currency: "USD" },
      ],
      orders: [
        { symbol: "USO", action: "SELL", qty: 692, order_type: "LMT", lmt: 123.21, aux: null, tif: "GTD", status: "PreSubmitted", parent_id: 0, order_id: 101 },
        { symbol: "USO", action: "SELL", qty: 692, order_type: "STP", lmt: null, aux: 103.29, tif: "GTC", status: "PreSubmitted", parent_id: 99, order_id: 102 },
      ] },
    { key: "pa", label: "PA (Gateway)", error: "not connected (ConnectionRefusedError)", nlv: null, positions: [], orders: [] },
  ],
};
const MOCK = {
  "/exec-status": { online: true, configured: true, sockets: 1, heartbeat_age_ms: 500 },
  "/exec-book": { book: BOOK },
  "/exec-commands": { commands: [
    { id: "abcd1234-ee", type: "flatten", account: "primary", state: "dry_run", created_at: NOW - 3000,
      result: { ok: true, detail: "[DRY-RUN] would flatten 100% of USO (primary) via MKT" } } ] },
};

function mkEl() {
  return { innerHTML: "", textContent: "", value: "", dataset: {}, className: "",
    addEventListener() {}, querySelector() { return mkEl(); }, querySelectorAll() { return []; },
    appendChild() {} };
}
const document = {
  _els: {},
  getElementById(id) { return (this._els[id] ||= mkEl()); },
  querySelectorAll() { return []; },
  addEventListener(ev, fn) { this._ready = fn; },
  createElement() { return mkEl(); },
};
const Dnow = () => NOW;
const sandbox = { console, document, setInterval() {}, setTimeout() {},
  Promise, Date: Object.assign(function () {}, { now: Dnow }),
  fetch: async (path) => ({ ok: true, status: 200, json: async () => MOCK[path] || {} }) };
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync("site/assets/common.js", "utf8") + "\n" +
                fs.readFileSync("site/assets/execution.js", "utf8"), sandbox);

(async () => {
  await document._ready();
  await new Promise((r) => setTimeout(r, 50));   // let poll() resolve
  const pos = document._els["positions"].innerHTML;
  const ord = document._els["orders"].innerHTML;
  const act = document._els["activity"].innerHTML;
  const conn = document._els["connBar"].innerHTML;
  const want = [
    [conn, "Execution online"], [conn, "$758,420"],
    [pos, "USO"], [pos, "692"], [pos, "MBT"], [pos, "202605"], [pos, "Flatten"], [pos, "Trim"], [pos, "$2,28"],
    [ord, "123.21"], [ord, "PreSubmitted"], [ord, "Cancel"], [ord, "STP"],
    [act, "would flatten"], [act, "DRY-RUN OK"],
  ];
  let ok = true;
  for (const [hay, w] of want) { const hit = (hay || "").includes(w); if (!hit) ok = false; console.log((hit ? "OK   " : "MISS ") + w); }
  console.log(`\n${ok ? "ALL PASS" : "FAILURES"}  (pos ${pos.length}b, ord ${ord.length}b)`);
  process.exit(ok ? 0 : 1);
})();
