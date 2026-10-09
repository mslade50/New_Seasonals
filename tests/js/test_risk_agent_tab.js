"use strict";

/* Risk Agent tab: renders a fixture payload, escapes every text field, shows
   the 404 "no run yet" state, and offers no staging or execution control. */

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const ASSETS = path.join(__dirname, "..", "..", "site", "assets");
const source = fs.readFileSync(path.join(ASSETS, "risk-agent.js"), "utf8");
const context = { console, Date, Intl, document: { addEventListener() {} }, window: {}, module: { exports: {} } };
vm.createContext(context);
vm.runInContext(source, context, { filename: "risk-agent.js" });
const RA = context.module.exports;

const XSS = '<img src=x onerror=alert(1)>';
const FIX = {
  asof: "2026-10-09", decision_id: "RAD-2026-10-09", mode: "decision", model: "opus", effort: "xhigh",
  published_at: "2026-10-09T22:20:00+00:00",
  posture: { summary: "Defensive: " + XSS, net_beta: 0.3, cash_pct: 55 },
  forecasts: [
    { horizon_td: 21, p_up: 0.55, q10_pct: -5, q90_pct: 6, basis: "base rate" },
    { horizon_td: 5, p_up: 0.52, q10_pct: -2.5, q90_pct: 2.8, basis: "tape" },
  ],
  new_orders: [{
    id: "RA-2026-10-09-1", kind: "etf", instrument: "XLE", side: "long", qty: 100, ref_price: 91.2,
    risk_bps: 40, notional: 9120, entry: { type: "MOO" }, exit: { time_td: 21, stop: 88.5 },
    thesis: "t " + XSS, evidence: "e", survived: "s", what_kills_it: "k",
  }, {
    id: "RA-2026-10-09-2", kind: "option", instrument: "SPY x2: +1 P 500 2026-11-20", side: "long", qty: 2,
    risk_bps: 45, notional: 1800, entry: { type: "CHAIN" }, exit: { time_td: 30 },
    option: { status: "PASS_TERMINAL", max_loss: 905.2, legs: [{ right: "P", strike: 500, expiry: "2026-11-20", qty: 1 }] },
  }],
  verdicts: [{ id: "RA-2026-10-02-1", action: "hold", symbol: "GLD", reason: "still valid " + XSS }],
  book: { nav: 201234.5, cash: 150000, realized_pnl: -120.4, pending: 1,
    positions: [{ id: "RA-2026-10-02-1", symbol: "GLD", side: "long", qty: 50, entry: 240.1, mark: 243.0, pnl: 145, risk_bps: 30, stale_mark: true }] },
  book_after: { positions_after: 3, risk_bps_after: 115, gross_notional_after: 90000, gross_x_nav: 0.45 },
  scoreboard: { headline: { nav: 201234.5, total_return_pct: 0.62, max_drawdown_pct: -1.1, vs_spy_pct: 0.4, brier_5: 0.25 },
    nav_curve: [["2026-10-05", 200000], ["2026-10-06", 200500], ["2026-10-07", 200100], ["2026-10-08", 201234.5]] },
  considered_and_rejected: [{ idea: "Long TLT", reason: "no edge " + XSS }],
  watchlist: [{ idea: "IWM", trigger: "close above 230", expires: "2026-10-16" }],
  warnings: ["stale CBOE " + XSS],
};

const html = RA.renderRiskAgent(FIX);
for (const needle of ["RAD-2026-10-09", "XLE", "PASS_TERMINAL", "Defensive", "GLD", "145", "Brier 5d", "0.250",
  "Long TLT", "close above 230", "stale CBOE", "(stale)", "<svg", "polyline", "SPY forecasts"])
  assert.ok(html.includes(needle), "missing " + needle);
// forecasts are sorted by horizon: 5 td before 21 td
assert.ok(html.indexOf("5 td") < html.indexOf("21 td"));
// escaping: no raw markup from any text field survives
assert.ok(!html.includes("<img"), "raw img tag leaked");
assert.ok(html.includes("&lt;img src=x onerror=alert(1)&gt;"));
// display only: no buttons, no execution links; option cards (only) link to the Options builder
assert.ok(!/<button/i.test(html) && !/execution\.html|stage=/i.test(html), "must be display only");
assert.strictEqual((html.match(/Stage in Options/g) || []).length, 1, "option card only, not the ETF card");
assert.ok(html.includes('href="options.html?ticker=SPY&amp;legs=P:500:2026-11-20:%2B1&amp;qty=2"'), "stage href");
const vert = RA.raStageHref({ instrument: "SPY x3", qty: 3, structure_qty: 3, option: { legs: [
  { right: "P", strike: 748, expiry: "2026-11-30", qty: 1 }, { right: "P", strike: 720, expiry: "2026-11-30", qty: -1 }] } });
assert.strictEqual(vert, "options.html?ticker=SPY&legs=P:748:2026-11-30:%2B1,P:720:2026-11-30:-1&qty=3");
assert.strictEqual(RA.raStageHref({ kind: "etf", instrument: "XLE" }), null);
assert.strictEqual(RA.raStageHref({ instrument: "X", option: { legs: [{ right: "P", strike: 1, expiry: "bad", qty: 1 }] } }), null);

// stand-down and sparse payloads do not throw
const hold = RA.renderRiskAgent({ asof: "2026-10-09", mode: "stand_down", reason: "feed late", posture: {}, scoreboard: {} });
assert.ok(hold.includes("DATA HOLD") && hold.includes("feed late") && hold.includes("No scoreboard yet"));

assert.strictEqual(RA.raCurve({ nav_curve: [{ nav: 1 }, { value: 2 }, "x", 3] }).length, 3);
assert.strictEqual(RA.raSparkline([1]), "");

(async () => {
  const el = { innerHTML: "" };
  assert.strictEqual(await RA.loadRiskAgent(el, async () => ({ status: 404, ok: false })), null);
  assert.ok(el.innerHTML.includes("No Risk Agent run has been published yet"));
  await RA.loadRiskAgent(el, async () => ({ status: 503, ok: false }));
  assert.ok(el.innerHTML.includes("Could not load") && el.innerHTML.includes("HTTP 503"));
  await RA.loadRiskAgent(el, async () => { throw new Error("boom <b>"); });
  assert.ok(el.innerHTML.includes("boom &lt;b&gt;"));
  const got = await RA.loadRiskAgent(el, async (url) => {
    assert.strictEqual(url, "/risk-agent-today");
    return { status: 200, ok: true, json: async () => FIX };
  });
  assert.strictEqual(got.asof, "2026-10-09");
  assert.ok(el.innerHTML.includes("XLE"));

  // wiring: nav entry right after Pitch, function reads the right key, page loads the script
  const common = fs.readFileSync(path.join(ASSETS, "common.js"), "utf8");
  assert.ok(common.includes('{ href: "risk-agent.html", label: "Risk Agent" }'));
  assert.ok(common.indexOf("pitch.html") < common.indexOf("risk-agent.html"));
  const fn = fs.readFileSync(path.join(__dirname, "..", "..", "functions", "risk-agent-today.js"), "utf8");
  assert.ok(fn.includes('"risk_agent/today.json"') && !/\.put\(|\.delete\(/.test(fn));
  const page = fs.readFileSync(path.join(__dirname, "..", "..", "site", "risk-agent.html"), "utf8");
  assert.ok(page.includes("assets/risk-agent.js") && page.includes('data-page="risk-agent"'));
  console.log("test_risk_agent_tab ok");
})().catch((e) => { console.error(e); process.exit(1); });
