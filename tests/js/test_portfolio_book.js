"use strict";
// Portfolio page Book scope control (swing / intraday / combined) over the
// intraday_daily.json payload: every daily figure recomputes per book, trade
// stats show n/a for intraday, correlation gains intraday rows, and a build
// without the payload renders exactly as before.
const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const read = rel => fs.readFileSync(path.join(__dirname, "../..", rel), "utf8");

class Node {
  constructor(id, tag) {
    this.id = id; this.tag = tag; this._html = ""; this.children = [];
    this.textContent = ""; this.value = ""; this.style = {}; this.dataset = {};
    this.listeners = {};
    const cls = new Set();
    this.classList = {
      toggle(c, on) { if (on === undefined ? !cls.has(c) : on) cls.add(c); else cls.delete(c); },
      add(c) { cls.add(c); }, remove(c) { cls.delete(c); }, contains(c) { return cls.has(c); },
    };
  }
  get innerHTML() { return this._html; }
  set innerHTML(v) { this._html = v; this.children = []; }
  appendChild(c) { this.children.push(c); return c; }
  append(...cs) { cs.forEach(c => this.children.push(c)); }
  querySelector() { return new Node("", "q"); }
  querySelectorAll(sel) { return sel === "button" ? this.children.filter(c => c.tag === "button") : []; }
  addEventListener(ev, fn) { this.listeners[ev] = fn; }
}
const html = n => n.innerHTML + n.children.map(html).join("") + (n.textContent || "");

function load(search = "") {
  const nodes = new Map();
  const document = {
    addEventListener() {},
    getElementById(id) { if (!nodes.has(id)) nodes.set(id, new Node(id, "div")); return nodes.get(id); },
    createElement: tag => new Node("", tag),
  };
  // index.html ships these two hidden
  document.getElementById("bookScope").style.display = "none";
  document.getElementById("bookScopeNote").style.display = "none";
  const Plotly = {
    react(el, traces, layout) { el._traces = traces; el._layout = layout; el._fullLayout = {}; },
    purge(el) { el._fullLayout = undefined; el._traces = null; },
  };
  const context = { console, document, Plotly, window: { location: { search } } };
  vm.createContext(context);
  vm.runInContext(read("site/assets/common.js"), context, { filename: "common.js" });
  vm.runInContext(read("site/assets/portfolio.js"), context, { filename: "portfolio.js" });
  const run = code => vm.runInContext(code, context);
  return { context, run, el: id => document.getElementById(id) };
}

// ---- fixture: two swing strategies, two intraday strategies (+ one without a series)
const SWING_DATES = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07",
  "2020-01-08", "2020-01-09", "2020-01-10", "2020-01-13"];
const ALPHA = [100, -50, 200, 0, -100, 300, 50, -20];
const BETA = [0, 80, -40, 60, 0, -30, 90, 10];
const INTRA_DATES = ["2020-01-08", "2020-01-09", "2020-01-10", "2020-01-13",
  "2020-01-14", "2020-01-15", "2020-01-16", "2020-01-17"];
const OB = [500, -200, 0, 300, -100, 400, 0, -50];
const LE = [0, 0, 40, -10, 0, 20, 0, 30];

const SD = {
  dates: SWING_DATES,
  series: { "Alpha||Liquid": ALPHA, "Beta||Liquid": BETA },
  total_flat: ALPHA.map((v, i) => v + BETA[i]),
  equity_compounded: SWING_DATES.map(() => 750000),
  start_equity: 750000,
};
const TRADES = [
  { trade_id: 1, Strategy: "Alpha", Tier: "Liquid", Ticker: "AAA", Direction: "Long",
    Signal_Date: "2020-01-02", Entry_Date: "2020-01-02", Exit_Date: "2020-01-06", R: 1.5, PnL_flat: 250, Hold_Days: 2 },
  { trade_id: 2, Strategy: "Alpha", Tier: "Liquid", Ticker: "AAB", Direction: "Long",
    Signal_Date: "2020-01-07", Entry_Date: "2020-01-07", Exit_Date: "2020-01-13", R: 0.8, PnL_flat: 230, Hold_Days: 4 },
  { trade_id: 3, Strategy: "Beta", Tier: "Liquid", Ticker: "BBB", Direction: "Short",
    Signal_Date: "2020-01-03", Entry_Date: "2020-01-03", Exit_Date: "2020-01-07", R: 0.6, PnL_flat: 100, Hold_Days: 2 },
  { trade_id: 4, Strategy: "Beta", Tier: "Liquid", Ticker: "BBC", Direction: "Long",
    Signal_Date: "2020-01-08", Entry_Date: "2020-01-08", Exit_Date: "2020-01-13", R: 0.4, PnL_flat: 70, Hold_Days: 3 },
];
const CORR = {
  strategies: ["Alpha", "Beta"],
  matrix: [[1, 0.2], [0.2, 1]],
  diversification: [{ strategy: "Alpha", avg_corr: 0.2, max_corr: 0.2, max_with: "Beta" },
    { strategy: "Beta", avg_corr: 0.2, max_corr: 0.2, max_with: "Alpha" }],
};
const INTRA = {
  book: "intraday", label: "research replay at live sizing", start_equity: 750000,
  dates: INTRA_DATES,
  series: { "Open Breakout||Intraday": OB, "Legend EMA||Intraday": LE },
  total_flat: OB.map((v, i) => v + LE[i]),
  strategies: [
    { id: "open_breakout", name: "Open Breakout", Strategy: "Open Breakout", Tier: "Intraday", book: "intraday",
      key: "Open Breakout||Intraday", has_daily: true, span: ["2020-01-08", "2020-01-17"] },
    { id: "legend_ema", name: "Legend EMA", Strategy: "Legend EMA", Tier: "Intraday", book: "intraday",
      key: "Legend EMA||Intraday", has_daily: true, span: ["2020-01-10", "2020-01-17"] },
    { id: "later_one", name: "Later One", Strategy: "Later One", Tier: "Intraday", book: "intraday",
      key: "Later One||Intraday", has_daily: false, span: null, notes: "no replay yet" },
  ],
};

function setup(page, intra) {
  page.context.__fx = { SD, TRADES, CORR, INTRA: intra ? JSON.parse(JSON.stringify(intra)) : null };
  page.run(`
    S.meta = { strategies: [{ Strategy: "Alpha", Tier: "Liquid", n: 2, book: "swing" },
                            { Strategy: "Beta", Tier: "Liquid", n: 2, book: "swing" }],
               date_min: "2020-01-02", date_max: "2020-01-13" };
    S.trades = JSON.parse(JSON.stringify(__fx.TRADES));
    S.sd = JSON.parse(JSON.stringify(__fx.SD));
    S.corr = JSON.parse(JSON.stringify(__fx.CORR));
    S.dateIdx = S.sd.dates;
    S.sd.dates.forEach((d, i) => S.dateToI.set(d, i));
    S.midMask = new Uint8Array(S.sd.dates.length);
    S.sizing = "flat";
    setupIntraday(__fx.INTRA);
    S.f.strategies = new Set(allStrategyNames());
    buildBookScope();
    apply();
    renderCorrelation();
  `);
}

const eq = page => {
  const t = page.el("eqChart")._traces[0];
  return { x: Array.from(t.x), y: Array.from(t.y) };
};
const flatEq = pnl => { let e = 10000; return pnl.map(v => (e += v * 10000 / 750000)); };
const close = (a, b, msg) => assert.ok(Math.abs(a - b) < 1e-6, `${msg}: ${a} vs ${b}`);
function pearson(a, b) {
  const n = a.length, ma = a.reduce((x, y) => x + y) / n, mb = b.reduce((x, y) => x + y) / n;
  let c = 0, va = 0, vb = 0;
  for (let i = 0; i < n; i++) { c += (a[i] - ma) * (b[i] - mb); va += (a[i] - ma) ** 2; vb += (b[i] - mb) ** 2; }
  return c / Math.sqrt(va * vb);
}
const swingPnl = ALPHA.map((v, i) => v + BETA[i]);
const intraPnl = OB.map((v, i) => v + LE[i]);

// ---- missing intraday payload: today's page
const base = load();
setup(base, null);
{
  assert.strictEqual(base.run("S.intraday"), null);
  assert.strictEqual(base.run("S.book"), "swing");
  assert.strictEqual(base.el("bookScope").style.display, "none", "control stays hidden");
  assert.strictEqual(base.el("bookScope").innerHTML, "");
  assert.strictEqual(base.el("bookCorrNote").textContent, "");
  const e = eq(base);
  assert.deepStrictEqual(e.x, SWING_DATES);
  flatEq(swingPnl).forEach((v, i) => close(e.y[i], v, "swing equity"));
  const kpis = base.el("kpis").innerHTML;
  assert.ok(!kpis.includes("Avg Annual $") && !kpis.includes("Up Days"), "no new KPI cards");
  assert.strictEqual((kpis.match(/class="kpi"/g) || []).length, 18);
  const strat = html(base.el("stratTable"));
  assert.ok(!strat.includes(">Book<") && !strat.includes("Intraday"), "no Book column");
  assert.deepStrictEqual(Array.from(base.el("corrChart")._traces[0].x), ["Alpha", "Beta"]);
  assert.ok(base.el("cumRChart")._traces, "trade charts render");
  assert.strictEqual(base.run("allStrategyNames().join()"), "Alpha,Beta");
  assert.strictEqual(base.run("setBook('intraday')"), false, "no book switching without the payload");
  // live-only card keeps its original copy
  const card = base.run(`liveOnlyStrategiesHtml({ strategies: [{ id: "open_breakout", name: "OB",
    status: "pilot", stats_source: "research_backtest", family: "Futures", live: null }] })`);
  assert.ok(card.includes("so none of the views on this page") && !card.includes("Intraday book"));
}

// ---- combined (default when the payload exists)
const page = load();
setup(page, INTRA);
{
  assert.strictEqual(page.run("S.book"), "combined");
  assert.strictEqual(page.el("bookScope").style.display, "");
  assert.ok(html(page.el("bookScope")).includes("Book scope"));
  const note = page.el("bookScopeNote").innerHTML;
  assert.strictEqual((note.match(/research replay at live sizing/g) || []).length, 1, "label once");
  assert.ok(note.includes("first swing date (2020-01-02)"));
  assert.ok(note.includes("Open Breakout from 2020-01-08") && note.includes("Legend EMA from 2020-01-10"));
  assert.ok(note.includes("Later One has no replay series yet (no replay yet)"));
  assert.ok(!note.includes(String.fromCharCode(0x2014)), "no em dashes");

  const e = eq(page);
  const union = [...new Set([...SWING_DATES, ...INTRA_DATES])].sort();
  assert.deepStrictEqual(e.x, union, "combined runs over the union of dates");
  const comb = union.map(d => {
    const i = SWING_DATES.indexOf(d), j = INTRA_DATES.indexOf(d);
    return (i >= 0 ? swingPnl[i] : 0) + (j >= 0 ? intraPnl[j] : 0);
  });
  flatEq(comb).forEach((v, i) => close(e.y[i], v, "combined equity"));
  assert.strictEqual(e.x[0], "2020-01-02", "combined starts at the first swing date");

  const kpis = page.el("kpis").innerHTML;
  const annUsd = comb.reduce((x, y) => x + y) / comb.length * 252;
  assert.ok(kpis.includes(page.run(`fmt.money(${annUsd})`)), "Avg Annual $ on combined");
  assert.ok(kpis.includes("Up Days") && kpis.includes(">4<"), "trade count stays the swing count");

  const strat = html(page.el("stratTable"));
  assert.ok(strat.includes(">Book<"), "Book column");
  assert.ok(strat.includes(">Swing<") && strat.includes(">Intraday<"));
  assert.ok(strat.includes("Open Breakout") && strat.includes("Legend EMA") && strat.includes("Later One"));
  assert.ok(strat.includes('<span class="cap">n/a</span>'), "trade stats n/a for intraday rows");
  assert.ok(strat.includes(page.run(`fmt.money(${OB.reduce((x, y) => x + y)})`)), "OB dollars");

  // trade charts still render (swing trades)
  assert.ok(page.el("cumRChart")._traces && !page.el("cumRChart")._na);

  // correlation: intraday rows/columns appended to the ledger matrix
  const t = page.el("corrChart")._traces[0];
  assert.deepStrictEqual(Array.from(t.x), ["Alpha", "Beta", "Open Breakout", "Legend EMA"]);
  assert.strictEqual(t.z[0][1], 0.2, "swing pair from the ledger matrix");
  const lo = INTRA_DATES.indexOf("2020-01-08"), hi = SWING_DATES.length;
  const want = pearson(ALPHA.slice(4, hi), OB.slice(lo, lo + 4));
  close(t.z[0][2], +want.toFixed(3), "Alpha vs Open Breakout over the overlap");
  close(t.z[2][3], +pearson(OB.slice(2), LE.slice(2)).toFixed(3), "OB vs Legend over Legend's span");
  const book = pearson(swingPnl.slice(4), intraPnl.slice(0, 4));
  const sentence = page.el("bookCorrNote").textContent;
  assert.ok(sentence.startsWith("Swing vs intraday, book level: daily P&L correlation " + book.toFixed(2)), sentence);
  assert.ok(sentence.includes("over 2020-01-08 to 2020-01-13 (4 sessions"));
  assert.ok(html(page.el("divTable")).includes("Open Breakout"));

  // live-only card links to the intraday view
  const card = page.run(`liveOnlyStrategiesHtml({ strategies: [{ id: "open_breakout", name: "OB",
    status: "pilot", stats_source: "research_backtest", family: "Futures", live: null }] })`);
  assert.ok(card.includes('href="index.html?scope=intraday"') && card.includes("in the Intraday book"));
}

// ---- swing view with the payload: same daily figures as the missing-payload page
{
  assert.strictEqual(page.run("setBook('swing')"), true);
  assert.deepStrictEqual(eq(page), eq(base), "swing view equals today's curve");
  assert.deepStrictEqual(JSON.parse(JSON.stringify(page.el("monthlyChart")._traces[0].z)),
    JSON.parse(JSON.stringify(base.el("monthlyChart")._traces[0].z)));
  assert.deepStrictEqual(Array.from(page.el("corrChart")._traces[0].x), ["Alpha", "Beta"]);
  const strat = html(page.el("stratTable"));
  assert.ok(strat.includes(">Book<") && !strat.includes("Open Breakout"), "swing rows only, Book column kept");
}

// ---- intraday view
{
  assert.strictEqual(page.run("setBook('intraday')"), true);
  const e = eq(page);
  assert.deepStrictEqual(e.x, INTRA_DATES);
  flatEq(intraPnl).forEach((v, i) => close(e.y[i], v, "intraday equity"));
  const kpis = page.el("kpis").innerHTML;
  assert.ok(/Trades<\/div>\s*<div class="v ">n\/a/.test(kpis), "Trades n/a");
  assert.ok(/Win Rate<\/div>\s*<div class="v ">n\/a/.test(kpis), "win rate n/a");
  assert.ok(kpis.includes("no per-trade data (intraday replay)"));
  const m = intraPnl.reduce((x, y) => x + y) / intraPnl.length / 750000;
  assert.ok(kpis.includes(page.run(`fmt.pct(${m * 252}, 1)`)), "Ann Return recomputed on intraday");
  for (const id of ["cumRChart", "histChart", "monthSeasChart", "weekdaySeasChart", "holdChart"])
    assert.ok(page.el(id).innerHTML.includes("Not available for the intraday book"), id);
  const strat = html(page.el("stratTable"));
  assert.ok(!strat.includes("Alpha") && strat.includes("Open Breakout"));
  const years = html(page.el("yearTable"));
  assert.ok(years.includes("2020") && years.includes('<span class="cap">n/a</span>'));
  const t = page.el("corrChart")._traces[0];
  assert.deepStrictEqual(Array.from(t.x), ["Open Breakout", "Legend EMA"]);
  // monthly table only holds intraday months
  assert.deepStrictEqual(Array.from(page.el("monthlyChart")._traces[0].y), ["2020"]);
}

// ---- KPIs differ across books; back to combined restores the trade charts
{
  const sharpe = () => /Sharpe<\/div>\s*<div class="v [a-z]*">([^<]+)</.exec(page.el("kpis").innerHTML)[1];
  const s = {};
  for (const b of ["swing", "intraday", "combined"]) { page.run(`setBook('${b}')`); s[b] = sharpe(); }
  assert.ok(new Set(Object.values(s)).size === 3, JSON.stringify(s));
  assert.ok(page.el("cumRChart")._traces && !page.el("cumRChart").innerHTML.includes("Not available"));
  assert.strictEqual(page.run("setBook('bogus')"), false);
}

// ---- leverage scales the intraday book; tier/direction filters exclude it
{
  page.run("setBook('intraday'); S.lev = 2; apply();");
  flatEq(intraPnl.map(v => v * 2)).forEach((v, i) => close(eq(page).y[i], v, "2x leverage"));
  page.run("S.lev = 1; S.f.tier = 'Liquid'; setBook('combined');");
  assert.deepStrictEqual(eq(page), eq(base), "tier filter drops intraday from combined");
  assert.ok(page.el("bookScopeNote").innerHTML.includes("excluding the intraday book"));
  page.run("S.f.tier = 'All'; S.f.strategies.delete('Open Breakout'); apply();");
  const e = eq(page);
  const union = [...new Set([...SWING_DATES, ...INTRA_DATES])].sort();
  const comb = union.map(d => {
    const i = SWING_DATES.indexOf(d), j = INTRA_DATES.indexOf(d);
    return (i >= 0 ? swingPnl[i] : 0) + (j >= 0 ? LE[j] : 0);
  });
  flatEq(comb).forEach((v, i) => close(e.y[i], v, "strategy filter drops Open Breakout"));
}

// ---- ?scope= picks the initial book; a malformed payload is ignored
{
  const p = load("?scope=intraday");
  setup(p, INTRA);
  assert.strictEqual(p.run("S.book"), "intraday");
  const bad = load();
  const broken = JSON.parse(JSON.stringify(INTRA));
  broken.series["Open Breakout||Intraday"].pop();
  setup(bad, broken);
  assert.strictEqual(bad.run("S.intraday"), null, "length mismatch rejected");
  assert.deepStrictEqual(eq(bad), eq(base));
}

// Trade-complete replays: the same tables and filters as the swing book.
const detailed = JSON.parse(JSON.stringify(INTRA));
detailed.has_trades = true;
detailed.trades = [];
for (const [name, ticker, values] of [["Open Breakout", "MNQ", OB], ["Legend EMA", "SPY", LE]]) {
  values.forEach((v, i) => {
    if (!v) return;
    detailed.trades.push({ trade_id: `intra:${ticker}:${i}`, book: "intraday", Tier: "Intraday",
      Strategy: name, Ticker: ticker, Direction: v < 0 ? "Short" : "Long",
      Entry_Date: INTRA_DATES[i], Exit_Date: INTRA_DATES[i], PnL_flat: v,
      R: name === "Legend EMA" ? null : v > 0 ? 1 : -.5, Hold_Days: 0, Hold_Minutes: 59 });
  });
}
const rich = load();
setup(rich, detailed);
// Real startup builds the book control before the strategy filters.
rich.run("S.f.strategies = null; buildBookScope(); S.f.strategies = new Set(allStrategyNames())");
const dailyOnly = load();
setup(dailyOnly, INTRA);
assert.strictEqual(rich.run("bookTrades().length"), 14, "combined includes both trade books");
assert.deepStrictEqual(eq(rich), eq(dailyOnly), "trade rows must not double count daily P&L");
rich.run("setBook('intraday')");
assert.strictEqual(rich.run("bookTrades().length"), 10);
assert.ok(!rich.el("cumRChart")._na, "intraday R chart is enabled");
assert.ok(html(rich.el("tradeLog")).includes("MNQ"), "intraday trades rendered");
assert.ok(html(rich.el("stratTable")).includes("59.0m"), "hold minutes in strategy table");
assert.ok(!rich.el("kpis").innerHTML.includes("no per-trade data"));
rich.run("S.f.strategies = new Set(['Legend EMA']); apply()");
assert.strictEqual(rich.run("tradeMetrics(bookTrades()).winRate"), .75);
assert.strictEqual(rich.run("tradeMetrics(bookTrades()).pf"), 9);
assert.strictEqual(rich.run("tradeMetrics(bookTrades()).totR"), null, "no invented R for un-stopped Legend EMA");
rich.run("S.f.strategies = new Set(allStrategyNames()); S.f.tickerQ = 'MNQ'; S.f.dir = 'Short'; S.midScalar = .1; S.lev = 2; apply()");
assert.strictEqual(rich.run("bookTrades().length"), 3);
assert.strictEqual(rich.run("bookSeries(bookTrades()).pnl.reduce((a,b) => a+b, 0)"), -700,
  "ticker/direction filters apply; intraday is exempt from swing midterm overlays");
assert.strictEqual(rich.run("tradeMetrics(bookTrades()).totPnl"), -700);
rich.run("S.f.tickerQ = ''; S.f.dir = 'All'; S.f.tier = 'Intraday'; apply()");
assert.strictEqual(rich.run("bookTrades().length"), 10, "Intraday tier selector works");
rich.run("S.f.from = '2020-01-14'; S.f.to = '2020-01-15'; apply()");
assert.strictEqual(rich.run("bookTrades().length"), 3, "date filter applies to intraday trades");
assert.strictEqual(rich.run("bookSeries(bookTrades()).pnl.reduce((a,b) => a+b, 0)"), 640);
rich.run("S.f.from = null; S.f.to = null; S.sd.dates.push('2020-02-03'); renderBookNote()");
assert.ok(rich.el("bookScopeNote").innerHTML.includes("Historical coverage is incomplete"));
console.log("PASS portfolio books: trade rows, stats, filters, exact P&L, coverage, legacy fallback");

// OVS scale-outs are one position result; daily cash flows keep both exits.
{
  const p = load();
  setup(p, INTRA);
  p.run(`
    var ovsNear = { trade_id: 100, Strategy: "Overbot Vol Spike", Tier: "Liquid",
      Ticker: "OVS", Direction: "Short", Signal_Date: "2020-01-02",
      Entry_Date: "2020-01-03", Exit_Date: "2020-01-06", Entry_Price: 100,
      Exit_Price: 98, Return_Pct: 2, R: 1, PnL_flat: 80, Risk_flat: 80,
      Shares_flat: 40, Hold_Days: 1, Exit_Type: "Target", Tranche: "near", Open: false };
    var ovsFar = { ...ovsNear, trade_id: 101, Exit_Date: "2020-01-07",
      Exit_Price: 101, Return_Pct: -1, R: -0.5, PnL_flat: -60, Risk_flat: 120,
      Shares_flat: 60, Hold_Days: 2, Exit_Type: "Time", Tranche: "far" };
    var ovsLegs = [ovsNear, ovsFar];
    var ovsResult = blendOvsTrades(ovsLegs);
  `);
  assert.strictEqual(p.run('ovsResult.length'), 1);
  close(p.run('ovsResult[0].Exit_Price'), 99.8, 'share-weighted exit');
  close(p.run('ovsResult[0].Return_Pct'), 0.2, 'share-weighted return');
  close(p.run('ovsResult[0].R'), 0.1, 'share-weighted R');
  assert.strictEqual(p.run('ovsResult[0].PnL_flat'), 20);
  assert.strictEqual(p.run('ovsResult[0].Risk_flat'), 200);
  assert.strictEqual(p.run('ovsResult[0].Hold_Days'), 2);
  assert.strictEqual(p.run('ovsResult[0].Exit_Date'), '2020-01-07');
  assert.strictEqual(p.run('ovsResult[0].Exit_Type'), 'Blended (Target + Time)');
  assert.strictEqual(p.run('tradeMetrics(ovsResult).n'), 1);
  assert.strictEqual(p.run('tradeMetrics(ovsResult).winRate'), 1, 'net winning position');
  assert.strictEqual(p.run('ovsNear.Exit_Price'), 98, 'source legs unchanged');
  assert.strictEqual(p.run('blendOvsTrades([ovsNear, {...ovsFar, Tier:"Overflow"}]).length'), 2);
  assert.strictEqual(p.run('blendOvsTrades([ovsNear, {...ovsFar, Signal_Date:"2020-01-01"}]).length'), 2);
  assert.strictEqual(p.run('blendOvsTrades([ovsNear, {...ovsFar, Entry_Date:"2020-01-04"}]).length'), 2);
  assert.strictEqual(p.run('blendOvsTrades([ovsNear, {...ovsFar, GateBlocked:true}]).length'), 2);
  assert.strictEqual(p.run('blendOvsTrades(ovsLegs.map(t => ({...t, Strategy:"Other"}))).length'), 2);
  assert.strictEqual(p.run('blendOvsTrades(ovsLegs.map(t => ({...t, book:"intraday"}))).length'), 2);
  assert.strictEqual(p.run('blendOvsTrades([ovsNear])[0] === ovsNear'), true);
  assert.strictEqual(p.run('blendOvsTrades([ovsNear, {...ovsFar, Open:true}])[0].Open'), true);
  p.run('renderTradeLog(blendOvsTrades([ovsNear, {...ovsFar, Open:true}]))');
  assert.ok(!html(p.el('tradeLog')).includes('Blended'), 'partially open positions excluded');
  p.run('renderTradeLog(ovsResult)');
  assert.ok(html(p.el('tradeLog')).includes('Blended (Target + Time)'));
  assert.ok(html(p.el('tradeLog')).includes('1 rows'), 'log/export table holds one result');
  p.run(`
    S.trades = ovsLegs;
    S.f.strategies = new Set(["Overbot Vol Spike"]);
    S.f.dir = "Short";
    S.f.tickerQ = "OVS";
    apply();
  `);
  assert.ok(p.el('kpis').innerHTML.includes('>1</div>'), 'apply counts one position');
  const daily = p.run('bookSeries(bookTrades())');
  close(daily.pnl[daily.dates.indexOf('2020-01-06')], 80, 'near cash flow date');
  close(daily.pnl[daily.dates.indexOf('2020-01-07')], -60, 'far cash flow date');
  assert.strictEqual(p.run('bookTrades().length'), 2, 'daily series still sees both legs');
  assert.ok(html(p.el('stratTable')).includes('Overbot Vol Spike'));
  assert.ok(html(p.el('tradeLog')).includes('1 rows'));
  p.run('S.f.to = \"2020-01-02\"; apply()');
  assert.strictEqual(p.run('bookTrades().length'), 0, 'date filters keep the position together');
  p.run('S.f.to = null; apply()');
  p.run('S.lev = 2; apply()');
  close(p.run('tradeMetrics(blendOvsTrades(bookTrades())).totPnl'), 40, 'sizing applies once');
  p.run('ovsFar.OvsExt = true; ovsFar.Exit_Type = "Time5";');
  assert.strictEqual(p.run('blendOvsTrades(ovsLegs)[0].OvsExt'), true, 'extension badge retained');
  close(p.run('blendOvsTrades(ovsLegs.map(t => ({...t, Shares_flat:null})))[0].R'), 0.1,
    'older payloads fall back to risk weighting');
}
console.log("OVS blended position results: all checks passed");
