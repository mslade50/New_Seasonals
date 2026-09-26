"use strict";
// Strategies tab: summary table, family/status filters, cards with anchors,
// status badges, components, live line, escaping, missing-payload empty state.
// Also the Portfolio page's live-only strategies card built from the same payload.
const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const read = rel => fs.readFileSync(path.join(__dirname, "../..", rel), "utf8");

function makeDom() {
  const nodes = new Map();
  const tables = [];
  const node = (id, tag = "div") => ({
    id, tag, innerHTML: "", value: "all", style: {}, listeners: {}, className: "",
    appendChild() {}, querySelectorAll() { return []; }, querySelector() { return null; },
    addEventListener(event, fn) { this.listeners[event] = fn; },
  });
  for (const id of ["topbar", "navAsof", "stratSummary", "stratFilters", "stratCards",
    "stratFamily", "stratStatus", "liveOnlyStrats", "stratCoverage"]) nodes.set(id, node(id));
  const document = {
    addEventListener() {},
    getElementById: id => nodes.get(id) || null,
    createElement(tag) { const n = node("", tag); if (tag === "table") tables.push(n); return n; },
  };
  return { nodes, tables, document };
}

function loadStrategies() {
  const dom = makeDom();
  const context = { console, document: dom.document, window: {}, location: { hash: "" } };
  vm.createContext(context);
  vm.runInContext(read("site/assets/common.js"), context, { filename: "common.js" });
  vm.runInContext(read("site/assets/strategies.js"), context, { filename: "strategies.js" });
  return { context, dom };
}

const PAYLOAD = {
  generated_at: "2026-09-26T12:00:00Z",
  strategies: [
    {
      id: "olv", name: "Oversold Low Volume", family: "Systematic equity book", status: "live",
      live_since: "2025-01-02", direction: "long", instruments: ["US equities"], universe_size: 500,
      captures: "Mean reversion <script>alert(1)</script> after quiet selloffs",
      entry: "Limit below close", exit: "Target or time stop", sizing: "ATR risk bps",
      risk_controls: ["Vol-confirm stop", "Earnings override"], order_ref_tags: ["Oversold Low Volume"],
      doc: "docs/claude_ref/olv.md", stats_source: "ledger_replay",
      stats: { span: "2000-2026", n_trades: 5200, trades_per_year: 200, trades_per_month: 16.7,
        win_rate: 0.58, avg_r: 0.21, median_hold_days: 4, profit_factor: 1.6, sharpe: 1.1 },
      frozen_stats: null, live: null,
    },
    {
      id: "event_sleeve", name: "Event sleeve", family: "Sleeves", status: "live",
      live_since: "2026-08-21", direction: "both", instruments: "SPY, SVXY",
      captures: "Calendar anchored index drift", stats_source: "frozen_evidence",
      stats: null, frozen_stats: { span: "2006-2026", n_trades: 240, trades_per_year: 12, win_rate: 0.6 },
      components: [
        { name: "V4 SVXY", n: 40, avg_bps: 55, t: 2.4, hit: 0.65, span: "2012-2026" },
        { name: "T2 FOMC", n: 60, avg_bps: -12, t: 1.1, hit: 0.52, span: "2006-2026" },
      ],
      live: null,
    },
    {
      id: "open_breakout", name: "NQ/ES Opening Breakout", family: "Futures", status: "pilot",
      live_since: "2026-09-25", direction: "both", instruments: ["MES", "MNQ"],
      captures: "First-range breakout in index futures", order_ref_tags: ["OpenBreakout"],
      stats_source: "research_backtest",
      stats: { span: "2019-2026", trades_per_year: 180, trades_per_month: 15, win_rate: 48, avg_r: 0.1, median_hold: "intraday" },
      live: { fills: 6, first_fill: "2026-09-25T13:30:00Z", last_fill: "2026-09-25T19:55:00Z",
        symbols: ["MES"], realized_pnl: -22.47, accounts: ["primary"], notes: "1-lot pilot" },
    },
  ],
};

// ---- full render
{
  const { context, dom } = loadStrategies();
  assert.strictEqual(context.renderStrategiesPage(JSON.parse(JSON.stringify(PAYLOAD))), true);
  const cards = dom.nodes.get("stratCards").innerHTML;
  // anchors
  for (const id of ["strat-olv", "strat-event_sleeve", "strat-open_breakout"])
    assert.ok(cards.includes(`id="${id}"`), `missing anchor ${id}`);
  // families in the fixed order
  const fam = ["Systematic equity book", "Futures", "Sleeves"].map(f => cards.indexOf(`>${f} <`));
  assert.ok(fam.every(i => i >= 0) && fam[0] < fam[1] && fam[1] < fam[2], "family order");
  // status badges
  assert.ok(/strat-badge strat-live">LIVE</.test(cards));
  assert.ok(/strat-badge strat-pilot">PILOT</.test(cards));
  // escaping of captures
  assert.ok(!cards.includes("<script>"), "raw script tag leaked");
  assert.ok(cards.includes("&lt;script&gt;alert(1)&lt;/script&gt;"));
  // stats source captions
  assert.ok(cards.includes("source: ledger replay of today&#39;s config over full history"));
  assert.ok(cards.includes("source: frozen evidence"));
  assert.ok(cards.includes("source: research backtest"));
  // components mini table
  assert.ok(cards.includes("V4 SVXY") && cards.includes("T2 FOMC") && cards.includes("+55"));
  // live line
  assert.ok(cards.includes("Live fills: 6 since 2026-09-25 (MES)"), "live line");
  assert.ok(cards.includes("docs/claude_ref/olv.md"));
  // win rate given as a percent (48) is not multiplied again
  assert.ok(cards.includes("48.0%") && cards.includes("58.0%"));
  // summary table
  const table = dom.tables.at(-1).innerHTML;
  for (const label of ["Strategy", "Family", "Status", "Direction", "Instruments", "Trades/yr",
    "Trades/mo", "Win rate", "Avg R", "Median hold", "Live since", "Stats source"])
    assert.ok(table.includes(`>${label}<`), `summary column ${label}`);
  assert.ok(table.includes("Oversold Low Volume") && table.includes("NQ/ES Opening Breakout"));
  assert.ok(table.includes("intraday"));
  // filters render both selects
  const filters = dom.nodes.get("stratFilters").innerHTML;
  assert.ok(filters.includes('id="stratFamily"') && filters.includes('id="stratStatus"'));
  assert.ok(filters.includes('value="Futures"') && filters.includes('value="pilot"'));

  // family filter
  const shown = context.stratSetFilter("family", "Futures");
  assert.deepStrictEqual(shown.map(s => s.id), ["open_breakout"]);
  const filtered = dom.nodes.get("stratCards").innerHTML;
  assert.ok(filtered.includes('id="strat-open_breakout"') && !filtered.includes('id="strat-olv"'));
  const filteredTable = dom.tables.at(-1).innerHTML;
  assert.ok(filteredTable.includes("NQ/ES Opening Breakout") && !filteredTable.includes("Oversold Low Volume"));
  // status filter composes with family
  assert.deepStrictEqual(context.stratSetFilter("status", "live").map(s => s.id), []);
  context.stratSetFilter("family", "all");
  assert.deepStrictEqual(context.stratSetFilter("status", "live").map(s => s.id), ["olv", "event_sleeve"]);
  // pure helpers
  assert.strictEqual(context.stratAnchor("a b/c"), "strat-a-b-c");
  assert.strictEqual(context.stratStats(PAYLOAD.strategies[1]).label, "frozen evidence");
}

// ---- producer shapes: ledger_stats, live.n_fills + by_account, cases, manual status,
//      ledger strategy without a replay this build falls back to labelled frozen stats
{
  const { context, dom } = loadStrategies();
  const payload = { strategies: [
    { id: "wcds", name: "WCDS", family: "Systematic equity book", status: "live", stats_source: "ledger_replay",
      ledger_stats: { trades_per_year: 90, win_rate: 0.55, avg_r: 0.3, median_hold_days: 6 },
      live: { n_fills: 12, first_fill: "2026-09-01", symbols: ["AAPL"], by_account: { primary: {}, pa: {} },
        realized_pnl: 120.5, notes: "store note" }, order_ref_tags: ["Weak Close Decent Sznls"] },
    { id: "mwc", name: "MWC", family: "Systematic equity book", status: "pilot", stats_source: "ledger_replay",
      ledger_stats: null, frozen_stats: { trades_per_year: 3 }, live: { n_fills: 0 }, order_ref_tags: ["Monthly Weak Close"] },
    { id: "board", name: "Seasonal board", family: "Agent products", status: "manual", stats_source: "none",
      frozen_stats: null, live: { n_fills: 0 }, order_ref_tags: [] },
    { id: "legend_ema", name: "Legend EMA", family: "Intraday ETF", status: "live", stats_source: "research_backtest",
      frozen_stats: { trades_per_year: 7.4, avg_bps: 15.2, median_hold: "59 minutes",
        cases: [{ case: "SPY long (ES signal)", n_trades: 28, win_rate: 0.571, avg_bps: 19.6, profit_factor: 3.24, t_stat: 1.2 }] } },
  ] };
  context.renderStrategiesPage(payload);
  const cards = dom.nodes.get("stratCards").innerHTML;
  assert.ok(cards.includes("Live fills: 12 since 2026-09-01 (AAPL)"));
  assert.ok(cards.includes("accounts: pa, primary"));
  assert.ok(cards.includes("source: ledger replay of today&#39;s config over full history"));
  assert.ok(cards.includes("frozen evidence (ledger replay unavailable in this build)"));
  assert.ok(cards.includes("Live fills: none in the fills store."));
  assert.ok(cards.includes("Live fills: none attributable (no orderRef tag)."));
  assert.ok(/strat-badge strat-manual">MANUAL</.test(cards));
  assert.ok(cards.includes("SPY long (ES signal)") && cards.includes("+19.6 bps") && cards.includes("t 1.20"));
  assert.ok(cards.includes("+15.2") && cards.includes("59 minutes"));
  const table = dom.tables.at(-1).innerHTML;
  assert.ok(table.includes("90.0") && table.includes("55.0%"));
}

// ---- account ids never render; live: null says so; fills coverage line
{
  const { context, dom } = loadStrategies();
  context.renderStrategiesPage({
    sources: { fills: { available: true, rows: 447, untagged_rows: 167,
      first_session: "2026-07-24", last_session: "2026-09-25" } },
    strategies: [
      { id: "a", name: "A", family: "Futures", status: "pilot", order_ref_tags: ["A"],
        live: { n_fills: 2, by_account: { U16584234: {}, primary: {} } } },
      { id: "b", name: "B", family: "Futures", status: "pilot", order_ref_tags: ["B"], live: null },
    ] });
  const cards = dom.nodes.get("stratCards").innerHTML;
  assert.ok(!/U16584234/.test(cards), "raw account id rendered");
  assert.ok(cards.includes("accounts: [account], primary"));
  assert.ok(cards.includes("Live fills: not available in this build."));
  const coverage = dom.nodes.get("stratCoverage").textContent;
  assert.ok(coverage.includes("447 executions from 2026-07-24 to 2026-09-25"));
  assert.ok(coverage.includes("167 (37%) carry no orderRef strategy"));
  context.renderStrategiesPage({ sources: { fills: { available: false } }, strategies: [] });
  assert.ok(dom.nodes.get("stratCoverage").textContent.includes("not available to this build"));
}

// ---- missing payload: empty state, no table, no cards
{
  const { context, dom } = loadStrategies();
  assert.strictEqual(context.renderStrategiesPage(null), false);
  assert.ok(dom.nodes.get("stratSummary").innerHTML.includes("Strategies payload not built"));
  assert.strictEqual(dom.nodes.get("stratCards").innerHTML, "");
  assert.strictEqual(dom.tables.length, 0);
  assert.strictEqual(context.renderStrategiesPage({ generated_at: "x" }), false);
}

// ---- page wiring
{
  const html = read("site/strategies.html");
  assert.ok(html.includes("assets/common.js") && html.includes("assets/strategies.js"));
  assert.ok(html.includes("Every signal the book trades, what it is meant to capture, and how often it fires."));
  assert.ok(read("site/assets/strategies.js").includes('renderNav("strategies.html")'));
  assert.ok(!/—/.test(read("site/assets/strategies.js")) && !/—/.test(html), "no em dashes in copy");
}

// ---- Portfolio live-only strategies card
{
  const dom = makeDom();
  const context = { console, document: dom.document, window: { location: { search: "" } } };
  vm.createContext(context);
  vm.runInContext(read("site/assets/common.js"), context, { filename: "common.js" });
  vm.runInContext(read("site/assets/portfolio.js"), context, { filename: "portfolio.js" });
  const catalog = JSON.parse(JSON.stringify(PAYLOAD));
  catalog.strategies.push({ id: "shadow_x", name: "Shadow <b>X</b>", family: "Futures", status: "shadow", stats_source: "none" });
  catalog.strategies.push({ id: "legend_ema", name: "Legend <i>EMA</i>", family: "Intraday ETF", status: "live", stats_source: "live_journal" });
  const names = context.liveOnlyStrategies(catalog).map(s => s.id);
  assert.deepStrictEqual(names, ["event_sleeve", "open_breakout", "legend_ema"]);
  context.renderLiveOnlyStrategies(catalog);
  const html = dom.nodes.get("liveOnlyStrats").innerHTML;
  assert.ok(html.includes("Live-only strategies (no ledger replay)"));
  assert.ok(html.includes('href="strategies.html#strat-open_breakout"'));
  assert.ok(html.includes("6 fills since 2026-09-25 (MES)"));
  assert.ok(html.includes("Legend &lt;i&gt;EMA&lt;/i&gt;") && !html.includes("<i>EMA"));
  assert.ok(!html.includes("Oversold Low Volume"), "ledger strategies stay out");
  context.renderLiveOnlyStrategies(null);
  assert.strictEqual(dom.nodes.get("liveOnlyStrats").innerHTML, "");
}

console.log("PASS strategies tab: summary, filters, badges, anchors, components, live line, escaping, empty state, portfolio live-only card");
