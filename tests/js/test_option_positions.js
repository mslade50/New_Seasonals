"use strict";

/* Held-option Close / Roll (Execution tab + Options builder roll mode), quote
   freshness, multi-expiry prefill, TWS-style ladder columns, and the
   exec-command shape check for option_close / option_roll. No network. */

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const ASSETS = path.join(__dirname, "..", "..", "site", "assets");
const plain = (x) => JSON.parse(JSON.stringify(x));

function ctxWith(files, extra = {}) {
  const context = {
    console, URLSearchParams, setTimeout, clearTimeout, setInterval, clearInterval, Intl, Date,
    document: { addEventListener() {}, getElementById() { return null; }, querySelectorAll() { return []; } },
    window: {}, location: { search: "" }, ...extra,
  };
  vm.createContext(context);
  for (const f of files) vm.runInContext(fs.readFileSync(path.join(ASSETS, f), "utf8"), context, { filename: f });
  return context;
}

/* ---------------- Execution tab helpers ---------------- */
const X = ctxWith(["option-positions.js"]);

// capabilities: an old agent (no list) supports nothing new
assert.strictEqual(X.optSupports({ mode: "live" }, "option_close"), false);
assert.strictEqual(X.optSupports({ capabilities: ["option_spread", "option_close"] }, "option_close"), true);
assert.strictEqual(X.optSupports({ capabilities: ["option_spread", "option_close"] }, "option_roll"), false);

const pos = (con_id, symbol, right, strike, position, avg_cost, expiry = "20261016") =>
  ({ symbol, sec_type: "OPT", con_id, right, strike, position, avg_cost, expiry_full: expiry, multiplier: 100 });
const P = [pos(11, "XYZ", "P", 95, -2, 310), pos(12, "XYZ", "P", 90, 2, 120), pos(13, "XYZ", "C", 120, 1, 50),
  pos(14, "ABC", "C", 50, 3, 200), { symbol: "XYZ", sec_type: "STK", con_id: 1, position: 100 }];
const fills = [
  { account_key: "primary", con_id: 11, order_ref: "EXEC|a|option-spread-parent", time: "2026-10-01T14:00:00Z" },
  { account_key: "primary", con_id: 12, order_ref: "EXEC|a|option-spread-parent", time: "2026-10-01T14:00:00Z" },
  { account_key: "primary", con_id: 13, order_ref: "EXEC|b|option-spread-parent", time: "2026-10-02T14:00:00Z" },
  { account_key: "pa", con_id: 14, order_ref: "EXEC|a|option-spread-parent", time: "2026-10-01T14:00:00Z" },
];
const refs = X.optOrderRefByConId(fills, "primary");
assert.strictEqual(refs.get(11), "EXEC|a|option-spread-parent");
assert.ok(!refs.has(14), "another account's fill never groups a leg");
let groups = plain(X.optGroupPositions(P, "primary", refs, new Set()));
const bySize = groups.map((g) => [g.symbol, g.legs.map((l) => l.con_id).sort(), g.source]).sort();
assert.deepStrictEqual(bySize, [["ABC", [14], "single"], ["XYZ", [11, 12], "order_ref"], ["XYZ", [13], "single"]]);
// manual ticks override the orderRef grouping (legs of one underlying only)
groups = plain(X.optGroupPositions(P, "primary", refs, new Set([12, 13])));
assert.deepStrictEqual(groups[0].legs.map((l) => l.con_id), [12, 13]);
assert.strictEqual(groups[0].source, "manual");
assert.ok(groups.some((g) => g.legs.length === 1 && g.legs[0].con_id === 11));

// reverse legs + held units (gcd)
const cl = plain(X.optCloseLegs([pos(11, "XYZ", "P", 95, -2, 310), pos(12, "XYZ", "P", 90, 4, 120)]));
assert.strictEqual(cl.units, 2);
assert.deepStrictEqual(cl.legs.map((l) => [l.con_id, l.action, l.ratio]), [[11, "BUY", 1], [12, "SELL", 2]]);

// natural / mid of the closing legs (BUY at ask, SELL at bid), debit +
const q = { 11: { bid: 3.0, ask: 3.2 }, 12: { bid: 1.0, ask: 1.1, mid: 1.05 } };
const net = plain(X.optNetQuote(cl.legs, (l) => q[l.con_id]));
assert.ok(Math.abs(net.mid - (3.1 - 2 * 1.05)) < 1e-9 && Math.abs(net.nat - (3.2 - 2 * 1.0)) < 1e-9);

// P&L vs average cost: short put sold at 3.10 (avg 310), long put bought at 1.20 (x2)
const single = X.optCloseLegs([pos(11, "XYZ", "P", 95, -1, 310)]).legs;
assert.strictEqual(X.optClosePnl(single, 1, +0.5), 260);                // buy back at 0.50 debit
const longCall = X.optCloseLegs([pos(13, "XYZ", "C", 120, 1, 200)]).legs;
assert.strictEqual(X.optClosePnl(longCall, 1, -3.0), 100);              // sell at 3.00 credit
assert.strictEqual(X.optClosePnl([{ ...longCall[0], avg_cost: null }], 1, -3), null);

// snapping toward safe and payload sign checks
assert.strictEqual(X.optSnap(1.23, "BUY"), 1.2);
assert.strictEqual(X.optSnap(1.21, "SELL"), 1.25);
let b = X.optBuildClosePayload({ symbol: "XYZ", legs: longCall, units: 1, limit: 2.97, action: "SELL", tif: "DAY" });
assert.deepStrictEqual(plain(b.payload), { symbol: "XYZ", action: "SELL", quantity: 1, limit: 3, tif: "DAY",
  legs: [{ con_id: 13, action: "SELL", ratio: 1, right: "C", strike: 120, expiry: "20261016" }] });
assert.ok(X.optBuildClosePayload({ symbol: "XYZ", legs: longCall, units: 1, limit: 3, action: "BUY" }).error, "all-SELL close is a credit");
assert.ok(/exceeds held/.test(X.optBuildClosePayload({ symbol: "XYZ", legs: longCall, units: 2, limit: 3, action: "SELL" }).error));
const text = X.optCloseConfirmText({ account: "primary", symbol: "XYZ", legs: longCall, units: 1, payload: b.payload,
  freshness: { stale: true, text: "FROZEN quotes, 30s old" }, pnl: 100 });
assert.ok(/SELL 1 XYZ 20261016 120C/.test(text) && /CREDIT/.test(text) && /\[WARN\] Quotes are FROZEN/.test(text));

// quote freshness: frozen / delayed / stale during RTH / live
const now = 1_800_000_000_000;
assert.strictEqual(X.optQuoteFreshness({ market_data_type: 2, asof: now / 1000 - 5 }, now, true).label, "FROZEN");
assert.strictEqual(X.optQuoteFreshness({ market_data_type: 3, asof: now / 1000 }, now, true).label, "DELAYED");
assert.strictEqual(X.optQuoteFreshness({ market_data_type: 4, asof: now / 1000 }, now, false).stale, true);
assert.strictEqual(X.optQuoteFreshness({ market_data_type: 1, asof: now / 1000 - 300 }, now, true).label, "STALE");
assert.strictEqual(X.optQuoteFreshness({ market_data_type: 1, asof: now / 1000 - 300 }, now, false).label, "LIVE");
assert.strictEqual(X.optQuoteFreshness({ market_data_type: 1, asof: now / 1000 - 10 }, now, true).stale, false);

// roll deep link carries the reverse legs, held units and account
const href = X.optRollHref({ symbol: "XYZ", legs: [pos(11, "XYZ", "P", 95, -2, 310)] }, "primary");
const hp = new URLSearchParams(href.split("?")[1]);
assert.strictEqual(hp.get("section"), "custom"); assert.strictEqual(hp.get("qty"), "2");
assert.strictEqual(hp.get("roll"), "11:BUY:1:P:95:20261016:-2"); assert.strictEqual(hp.get("acct"), "primary");

// row actions are hidden for an agent without capabilities; shown when advertised
const XR = ctxWith(["option-positions.js"]);
vm.runInContext(`var state = { account: "primary", book: { accounts: [{ key: "primary", positions: [] }] } };
  function acctBook() { return state.book.accounts[0]; }
  function fetchJSONOrNull() { return Promise.resolve({ fills: [] }); }
  function set() {} function renderPositions() { return ""; }`, XR);
let html = XR.optPositionActions(pos(13, "XYZ", "C", 120, 1, 50));
assert.ok(!/optCloseTicket/.test(html) && /newer agent/.test(html));
vm.runInContext(`state.book.capabilities = ["option_close", "option_roll"];`, XR);
html = XR.optPositionActions(pos(13, "XYZ", "C", 120, 1, 50));
assert.ok(/optCloseTicket\(13\)/.test(html) && /optRollTicket\(13\)/.test(html));

// the attach-exits dead end now points at Close/Roll
const exSrc = fs.readFileSync(path.join(ASSETS, "execution.js"), "utf8");
assert.ok(!exSrc.includes("option positions not supported"));
assert.ok(exSrc.includes("window.optPositionActions(p)"));
const exHtml = fs.readFileSync(path.join(__dirname, "..", "..", "site", "execution.html"), "utf8");
assert.ok(exHtml.indexOf("assets/execution.js") < exHtml.indexOf("assets/option-positions.js"));

/* ---------------- Options builder ---------------- */
const C = ctxWith(["bsm.js", "options.js", "option-positions.js", "options-custom.js", "options-roll.js"]);

// ladder: default columns, ATM highlight, bid = SELL / ask = BUY, explicit nulls
const row = (right, strike, extra) => ({ right, strike, bid: 1.0, ask: 1.2, mid: 1.1, con_id: strike * 10 + (right === "C" ? 1 : 2),
  iv: 0.25, delta: right === "C" ? 0.5 : -0.5, gamma: 0.0123, theta: -0.045, vega: 0.11, oi: 1500, volume: null, last: null, ...extra });
const chain = { expiry: "20261120", dte: 42, strikes: [row("C", 95), row("P", 95), row("C", 100, { oi: null }), row("P", 100), row("C", 105), row("P", 105)] };
let lad = C.ocLadderHtml({ chain, spot: 100.4, legs: [], cols: null, ticker: "XYZ", fresh: { label: "FROZEN", stale: true, text: "FROZEN quotes, 5s old" } });
assert.ok(/<th class="r">bid<\/th><th class="r">ask<\/th><th class="r">IV<\/th><th class="r">delta<\/th><th class="r">OI<\/th>/.test(lad), "default columns");
assert.ok(/class="oc-atm"[^>]*>.*<b>100<\/b>/.test(lad), "ATM row highlighted");
assert.ok(lad.includes('data-oc-add="C|95|SELL"') && lad.includes('data-oc-add="P|105|BUY"'));
assert.ok(/>1500</.test(lad) && /FROZEN/.test(lad));
assert.ok(!/>gamma</.test(lad.split("<thead>")[1] || ""));
lad = C.ocLadderHtml({ chain, spot: 100, legs: [], cols: ["gamma", "volume", "last"], ticker: "XYZ" });
assert.ok(/>bid<\/th><th class="r">ask<\/th><th class="r">last<\/th><th class="r">gamma<\/th><th class="r">vol</.test(lad), "bid/ask are always kept");
assert.ok(/>0\.0123</.test(lad) && /<td class="r">-<\/td>/.test(lad), "null volume/last render as -");
assert.deepStrictEqual(plain(C.ocNormCols(["oi", "nonsense"])), ["bid", "ask", "oi"]);

// multi-expiry prefill: covered calendar accepted on both expiries, never trimmed
const pl = (side, right, strike, expiry, ratio = 1) => ({ side, right, strike, expiry, ratio });
assert.deepStrictEqual(plain(C.ocPrefillPlan([pl("SELL", "C", 100, "20261016"), pl("BUY", "C", 100, "20261120")])), { expiries: ["20261016", "20261120"] });
assert.ok(/not an accepted calendar/.test(C.ocPrefillPlan([pl("BUY", "C", 100, "20261016"), pl("SELL", "C", 100, "20261120")]).error),
  "a reverse calendar (naked short back month) is refused whole");
assert.ok(/3 expiries/.test(C.ocPrefillPlan([pl("BUY", "C", 1, "20261016"), pl("BUY", "C", 2, "20261120"), pl("BUY", "C", 3, "20261218")]).error));
assert.deepStrictEqual(plain(C.ocPrefillPlan([pl("BUY", "P", 95, "20261016"), pl("SELL", "P", 90, "20261016")])), { expiries: ["20261016"] });
const cSrc = fs.readFileSync(path.join(ASSETS, "options-custom.js"), "utf8");
assert.ok(!/filter\(\(l\) => l\.expiry === want\)/.test(cSrc), "no silent first-expiry filter");

// confirm text: no automatic exits, points at Close/Roll
assert.ok(/No automatic stop, target or time exit is attached/.test(vm.runInContext("OC_NO_EXITS_TEXT", C)) && /Close\.\.\. \/ Roll\.\.\./.test(vm.runInContext("OC_NO_EXITS_TEXT", C)));
assert.ok(/\[WARN\] Quotes are/.test(C.ocStaleText({ stale: true, text: "DELAYED quotes, 1m old" })));
assert.strictEqual(C.ocStaleText({ stale: false, text: "LIVE" }), "");

// roll legs from the link
const rl = plain(C.ocParseRollLegs("11:BUY:1:P:95:20261016:-1"));
assert.deepStrictEqual(rl, [{ con_id: 11, action: "BUY", ratio: 1, right: "P", strike: 95, expiry: "20261016", held: -1 }]);
assert.deepStrictEqual(plain(C.ocParseRollLegs("11:SELL:1:P:95:20261016:-1")), [], "a closing leg must oppose the held sign");

// roll risk port matches option_position_orders.open_structure_risk
// short Oct 95P -> short Nov 90P for 0.30 credit: 100*(-0.30) + 9000 = 8970
let ev = C.ocRollRisk("SELL", 0.30, 1, [{ side: "SELL", right: "P", strike: 90, expiry: "20261120", ratio: 1 }], 1);
assert.ok(Math.abs(ev.info.unitLoss - 8970) < 1e-9 && ev.info.debitRisk === 89.7);
assert.ok(Math.abs(ev.info.riskUsd - (8970 + 0.65 * 3)) < 1e-9);
// short 105C -> short 110C for 0.20 credit: unbounded, needs ack, stress 130 -> 1980
ev = C.ocRollRisk("SELL", 0.20, 1, [{ side: "SELL", right: "C", strike: 110, expiry: "20261120", ratio: 1 }], 1, { spot: 100 });
assert.ok(ev.needsAck && /UNBOUNDED_ACK_REQUIRED/.test(ev.error));
ev = C.ocRollRisk("SELL", 0.20, 1, [{ side: "SELL", right: "C", strike: 110, expiry: "20261120", ratio: 1 }], 1, { spot: 100, unboundedAck: true });
assert.ok(ev.info.unbounded && Math.abs(ev.info.unitLoss - 1980) < 1e-6 && ev.info.debitRisk === 19.8);
// credit roll into a long-only leg has zero new cash risk (allowed)
ev = C.ocRollRisk("SELL", 0.5, 1, [{ side: "BUY", right: "P", strike: 90, expiry: "20261120", ratio: 1 }], 1);
assert.strictEqual(ev.info.unitLoss, 0);
// uncovered diagonal refused
ev = C.ocRollRisk("BUY", 0.5, 1, [{ side: "SELL", right: "P", strike: 90, expiry: "20261120", ratio: 1 },
  { side: "BUY", right: "P", strike: 95, expiry: "20261016", ratio: 1 }], 1);
assert.ok(/calendar/.test(ev.error));

// roll payload
const roll = { symbol: "XYZ", account: "primary", legs: rl, rows: new Map() };
const open = [{ side: "SELL", right: "P", strike: 90, expiry: "20261120", ratio: 1, row: { con_id: 902, bid: 2.4, ask: 2.6 } }];
let rp = C.ocBuildRollPayload({ symbol: "XYZ", roll, open, qty: 1, limit: 0.3, action: "SELL", tif: "DAY", spot: 100 });
assert.deepStrictEqual(plain(rp.payload), { symbol: "XYZ", action: "SELL", quantity: 1, limit: 0.3, tif: "DAY",
  close_legs: [{ con_id: 11, action: "BUY", ratio: 1, right: "P", strike: 95, expiry: "20261016" }],
  open_legs: [{ side: "SELL", right: "P", expiry: "20261120", strike: 90, ratio: 1, con_id: 902 }], debit_risk: 89.7 });
assert.ok(/exceeds held/.test(C.ocBuildRollPayload({ symbol: "XYZ", roll, open, qty: 2, limit: 0.3, action: "SELL" }).error));
assert.ok(C.ocBuildRollPayload({ symbol: "XYZ", roll, open: [{ ...open[0], row: { con_id: 11 } }], qty: 1, limit: 0.3, action: "SELL" }).error);
rp = C.ocBuildRollPayload({ symbol: "XYZ", roll: { ...roll, legs: [{ ...rl[0], right: "C", strike: 105 }] },
  open: [{ side: "SELL", right: "C", strike: 110, expiry: "20261120", ratio: 1, row: { con_id: 1101 } }], qty: 1, limit: 0.2, action: "SELL", spot: 100 });
assert.ok(rp.needsAck, "unbounded roll needs the explicit second confirmation");

// options page loads the shared helpers before the builder, roll mode after it
const oHtml = fs.readFileSync(path.join(__dirname, "..", "..", "site", "options.html"), "utf8");
const at = (f) => oHtml.indexOf(`assets/${f}`);
assert.ok(at("options.js") < at("option-positions.js") && at("option-positions.js") < at("options-custom.js") && at("options-custom.js") < at("options-roll.js"));
const pp = plain(C.parseParams.call(null));
assert.ok("roll" in pp && pp.acct === "primary");

/* ---------------- exec-command shape check ---------------- */
(async () => {
  const src = fs.readFileSync(path.join(__dirname, "..", "..", "functions", "exec-command.js"), "utf8");
  const uri = (t) => "data:text/javascript;base64," + Buffer.from(t).toString("base64");
  const mod = await import(uri(src.replace("./_access.js", uri("export async function requireAccess(){return null;}"))));
  const prev = globalThis.fetch, sent = [];
  globalThis.fetch = async (_url, opt) => { sent.push(JSON.parse(JSON.parse(opt.body).signed)); return new Response(JSON.stringify({ ok: true, state: "pushed" }), { status: 202 }); };
  const env = { EXEC_BROKER_URL: "https://mock-broker.invalid", STATUS_TOKEN: "NONSECRET_TEST_TOKEN" };
  const post = (type, payload) => mod.onRequestPost({ env, request: new Request("https://mock.invalid/exec-command",
    { method: "POST", body: JSON.stringify({ account: "primary", dry_run: true, type, payload }) }) });
  try {
    const good = { symbol: "XYZ", action: "SELL", quantity: 1, limit: 3, tif: "DAY", legs: [{ con_id: 13, action: "SELL", ratio: 1 }] };
    assert.strictEqual((await post("option_close", good)).status, 202);
    assert.strictEqual(sent.at(-1).type, "option_close");
    assert.ok(Number.isFinite(sent.at(-1).expires_at) && sent.at(-1).dry_run === true);
    for (const bad of [{ ...good, legs: [] }, { ...good, legs: [{ con_id: 0, action: "SELL" }] }, { ...good, action: "HOLD" },
      { ...good, limit: 0 }, { ...good, tif: "IOC" }, { ...good, legs: [{ con_id: 13, action: "SELL", ratio: 1.5 }] },
      { ...good, legs: [{ con_id: 13, action: "SELL" }, { con_id: 13, action: "BUY" }] }]) {
      const r = await post("option_close", bad);
      assert.strictEqual(r.status, 400, JSON.stringify(bad));
    }
    const roll = { symbol: "XYZ", action: "SELL", quantity: 1, limit: 0, tif: "DAY", debit_risk: 89.7,
      close_legs: [{ con_id: 11, action: "BUY", ratio: 1 }], open_legs: [{ side: "SELL", right: "P", expiry: "20261120", strike: 90, ratio: 1 }] };
    assert.strictEqual((await post("option_roll", roll)).status, 202, "an even roll (limit 0) is accepted");
    assert.strictEqual((await post("option_roll", { ...roll, open_legs: [] })).status, 400);
    assert.strictEqual((await post("option_roll", { ...roll, debit_risk: -1 })).status, 400);
    assert.strictEqual((await post("option_roll", { ...roll, unbounded_ack: "yes" })).status, 400);
    assert.strictEqual((await post("option_spread", { anything: 1 })).status, 202, "other types pass through unchanged");
  } finally { globalThis.fetch = prev; }
  console.log("test_option_positions ok");
})().catch((e) => { console.error(e); process.exit(1); });
