"""Run the Exec-tab display-only hedge browser contract in the Python suite."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXECUTION_JS = ROOT / "site" / "assets" / "execution.js"


def _exit_orders(symbol, strategy, signal_date, qty, exit_date, group, base_id):
    ref = f"{symbol}|BUY|{strategy}|{signal_date}"
    common = {
        "symbol": symbol,
        "sec_type": "STK",
        "action": "SELL",
        "qty": qty,
        "status": "Submitted",
        "order_ref": ref,
        "oca_group": group,
        "parent_id": 0,
    }
    return [
        {**common, "order_id": base_id, "order_type": "LMT", "lmt": 50},
        {
            **common,
            "order_id": base_id + 1,
            "order_type": "MKT",
            "good_after": f"{exit_date} 15:59:00 US/Eastern",
        },
    ]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_execution_hedge_javascript_contract():
    orders = []
    orders += _exit_orders("LUV", "Oversold Low Volume", "2026-08-14", 1069, "20260901", "luv-1", 101)
    orders += _exit_orders("LUV", "Oversold Low Volume", "2026-08-17", 1577, "20260902", "luv-2", 111)
    orders += _exit_orders("LUV", "Oversold Low Volume", "2026-08-18", 2268, "20260903", "luv-3", 121)
    orders += _exit_orders("UNH", "Oversold Low Volume", "2026-08-19", 115, "20260902", "unh-1", 201)
    orders += _exit_orders("UNH", "Oversold Low Volume", "2026-08-20", 185, "20260903", "unh-2", 211)

    # Parent deliberately has no OCA group and its children deliberately have
    # parent_id=0, matching persisted TWS brackets after a restart. Ref fallback
    # must classify the whole group as an unfilled entry, not filled exposure.
    pending_ref = "UNH|BUY|Oversold Low Volume|2026-08-21"
    orders.append(
        {
            "symbol": "UNH",
            "sec_type": "STK",
            "action": "BUY",
            "qty": 292,
            "status": "Submitted",
            "order_type": "LMT",
            "lmt": 382.51,
            "order_ref": pending_ref,
            "order_id": 901,
            "parent_id": 0,
        }
    )
    orders += [
        {
            "symbol": "UNH",
            "sec_type": "STK",
            "action": "SELL",
            "qty": 292,
            "status": "Submitted",
            "order_type": "LMT",
            "lmt": 430,
            "order_ref": pending_ref,
            "oca_group": "unh-pending",
            "order_id": 902,
            "parent_id": 0,
        },
        {
            "symbol": "UNH",
            "sec_type": "STK",
            "action": "SELL",
            "qty": 292,
            "status": "Submitted",
            "order_type": "MKT",
            "good_after": "20260904 15:59:00 US/Eastern",
            "order_ref": pending_ref,
            "oca_group": "unh-pending",
            "order_id": 903,
            "parent_id": 0,
        },
    ]

    account = {
        "key": "primary",
        "label": "Primary (TWS)",
        "nlv": 812345,
        "positions": [
            {
                "symbol": "LUV",
                "sec_type": "STK",
                "position": 4914,
                "avg_cost": 39.5,
                "market_price": 40.81,
            },
            {
                "symbol": "UNH",
                "sec_type": "STK",
                "position": 396,
                "avg_cost": 390,
                "market_price": 396.46,
            },
            {
                "symbol": "MESU6",
                "sec_type": "FUT",
                "position": -6,
                "avg_cost": 7640,
                "market_price": 7635,
            },
            {
                "symbol": "DXU6",
                "sec_type": "FUT",
                "position": 2,
                "avg_cost": 96.8,
                "market_price": 97.2,
                "multiplier": 1000,
            },
        ],
        "orders": orders,
    }
    pa = {**account, "key": "pa", "label": "PA (Gateway)", "nlv": 66557}
    betas = {
        "asof": "2026-08-24",
        "spy_last": 763.47,
        "account_value": 750000,
        "tickers": {
            "LUV": {"beta63": 1.5, "beta252": 1.5},
            "UNH": {"beta63": 0.4, "beta252": 0.4},
            "SPY": {"beta63": 1.0, "beta252": 1.0},
        },
    }
    specs = {"MES": {"multiplier": 5}, "ES": {"multiplier": 50}}

    script = r'''
const assert = require("assert");
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(__EXECUTION_JS__, "utf8");
const storage = new Map();
const sandbox = {
  console,
  document: { addEventListener() {}, getElementById() { return null; }, querySelectorAll() { return []; } },
  window: {}, location: { search: "" }, URLSearchParams,
  setTimeout, clearTimeout, setInterval, clearInterval,
  localStorage: {
    getItem(key) { return storage.has(key) ? storage.get(key) : null; },
    setItem(key, value) { storage.set(key, String(value)); },
  },
  fmt: {
    num(value, digits = 0) { return Number(value).toFixed(digits); },
    money(value) { return "$" + Math.round(Number(value)).toLocaleString("en-US"); },
    pct(value) { return String(value); }, signed(value) { return String(value); },
  },
  clsSign(value) { return value > 0 ? "pos" : value < 0 ? "neg" : "neu"; },
};
vm.createContext(sandbox);
vm.runInContext(source, sandbox, { filename: "execution.js" });
sandbox.ACCOUNT = __ACCOUNT__;
sandbox.PA = __PA__;
sandbox.BETAS = __BETAS__;
sandbox.SPECS = __SPECS__;

function value(expr) {
  return JSON.parse(vm.runInContext(`JSON.stringify(${expr})`, sandbox));
}
function near(actual, expected, tolerance = 1e-6) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
}

const model = value('attributeBook(ACCOUNT, BETAS, SPECS, {betaKey:"beta252", today:"2026-08-25"})');
assert.strictEqual(model.navBasis.kind, "sizing");
assert.strictEqual(model.navBasis.value, 750000); // never the primary live NLV
const olv = model.byStrategy.find(row => row.strategy === "Oversold Low Volume");
const unattributed = model.byStrategy.find(row => row.strategy === "Unattributed");
assert.ok(olv && unattributed);
assert.strictEqual(olv.legs, 5);
const olvNotional = 4914 * 40.81 + 300 * 396.46;
near(olv.notionalLong, olvNotional);
near(unattributed.notionalLong, 96 * 396.46);
assert.strictEqual(unattributed.legs, 1);
assert.strictEqual(model.workingEntries.length, 1);
assert.strictEqual(model.workingEntries[0].symbol, "UNH");
assert.strictEqual(model.workingEntries[0].strategy, "Oversold Low Volume");
assert.strictEqual(model.workingEntries[0].qty, 292);
assert.strictEqual(model.workingEntries[0].lmt, 382.51);

const mes = model.futures.find(row => row.root === "MES");
assert.ok(mes, "MESU6 root did not resolve to MES");
assert.strictEqual(mes.counted, true);
near(mes.spyEquiv, -6 * 5 * 7635);
const dx = model.futures.find(row => row.symbol === "DXU6");
assert.ok(dx && !dx.counted, "DX must be excluded from equity-index hedge total");
near(model.futuresSpyEquiv, -229050);

// Index futures remain counted if the snapshot and optional specs payload both
// omit the multiplier; the trusted root fallback must be used consistently.
const noSpecModel = value('attributeBook(ACCOUNT, BETAS, {}, {betaKey:"beta252", today:"2026-08-25"})');
const fallbackMes = noSpecModel.futures.find(row => row.root === "MES");
assert.strictEqual(fallbackMes.multiplier, 5);
near(fallbackMes.spyEquiv, -6 * 5 * 7635);
assert.strictEqual(noSpecModel.futuresComplete, true);

const expectedOlvSpy = 4914 * 40.81 * 1.5 + 300 * 396.46 * 0.4;
near(olv.spyEquiv, expectedOlvSpy);
sandbox.MODEL = model;
const target = value('hedgeTarget(MODEL, 0.5, {multiplier:5,indexLevel:7635,scopeStrategies:["Oversold Low Volume"]})');
near(target.targetDollars, 375000);
near(target.excess, expectedOlvSpy - 229050 - 375000);
assert.strictEqual(target.contracts, Math.round((expectedOlvSpy - 229050 - 375000) / 38175));

const scenarios = value('hedgeScenarios(MODEL, 0.5, ["Oversold Low Volume"])');
near(scenarios[0].pnlNow, -0.02 * (expectedOlvSpy - 229050));
near(scenarios[1].pnlAtTarget, -0.05 * 375000);
const afterAllTaggedExits = model.rolloff.find(row => row.date === "20260903");
assert.ok(afterAllTaggedExits, "roll-off window did not include 2026-09-03");
near(afterAllTaggedExits.remainingSpyEquiv, unattributed.spyEquiv);

// Short books invert entry/exit sides: a working SELL parent is pending while
// BUY time/target children without one are attributed as negative exposure.
const shortAccount = {
  key: "primary", nlv: 999999,
  positions: [{symbol:"AAPL", sec_type:"STK", position:-50, avg_cost:210, market_price:200}],
  orders: [
    {symbol:"AAPL", sec_type:"STK", action:"BUY", qty:30, status:"Submitted", order_type:"LMT", lmt:180,
      good_after:null, order_ref:"AAPL|SELL|Short Strategy|2026-08-20", oca_group:"short-filled", parent_id:0},
    {symbol:"AAPL", sec_type:"STK", action:"BUY", qty:30, status:"Submitted", order_type:"MKT",
      good_after:"20260902 15:59:00 US/Eastern", order_ref:"AAPL|SELL|Short Strategy|2026-08-20", oca_group:"short-filled", parent_id:0},
    {symbol:"AAPL", sec_type:"STK", action:"SELL", qty:10, status:"Submitted", order_type:"LMT", lmt:205,
      order_ref:"AAPL|SELL|Short Strategy|2026-08-21", order_id:77, parent_id:0},
    {symbol:"AAPL", sec_type:"STK", action:"BUY", qty:10, status:"Submitted", order_type:"MKT",
      good_after:"20260903 15:59:00 US/Eastern", order_ref:"AAPL|SELL|Short Strategy|2026-08-21", oca_group:"short-pending", parent_id:0},
  ],
};
sandbox.SHORT_ACCOUNT = shortAccount;
const shortModel = value('attributeBook(SHORT_ACCOUNT, {account_value:750000,tickers:{AAPL:{beta252:1.2}}}, SPECS, {today:"2026-08-25"})');
const shortRow = shortModel.byStrategy.find(row => row.strategy === "Short Strategy");
near(shortRow.notionalShort, -30 * 200);
near(shortRow.spyEquiv, -30 * 200 * 1.2);
near(shortModel.byStrategy.find(row => row.strategy === "Unattributed").notionalShort, -20 * 200);
assert.strictEqual(shortModel.workingEntries.length, 1);
assert.strictEqual(shortModel.workingEntries[0].notional, -10 * 205);

// Missing beta/mark data degrades deterministically and PA uses live NLV.
const degradedAccount = {
  key:"primary", positions:[{symbol:"XYZ",sec_type:"STK",position:10,avg_cost:12,market_price:null}], orders:[], nlv:1,
};
sandbox.DEGRADED = degradedAccount;
const degraded = value('attributeBook(DEGRADED, null, SPECS, {today:"2026-08-25"})');
assert.strictEqual(degraded.navBasis.value, 750000);
assert.ok(degraded.flags.some(flag => flag.includes("XYZ beta assumed")));
assert.ok(degraded.flags.some(flag => flag.includes("XYZ marked at avg_cost")));
const paModel = value('attributeBook(PA, BETAS, SPECS, {today:"2026-08-25"})');
assert.deepStrictEqual(paModel.navBasis, {kind:"live", value:66557});

// Missing live NLV or an unmarked held index future must suppress target,
// scenario, and verdict arithmetic instead of silently treating exposure as 0.
sandbox.PA_NO_NLV = {...sandbox.PA, nlv:null};
const paNoNlv = value('attributeBook(PA_NO_NLV, BETAS, SPECS, {today:"2026-08-25"})');
assert.deepStrictEqual(paNoNlv.navBasis, {kind:"live", value:null});
sandbox.PA_NO_NLV_MODEL = paNoNlv;
const noNlvTarget = value('hedgeTarget(PA_NO_NLV_MODEL, 0.5, {multiplier:5,indexLevel:7635,scopeStrategies:["Oversold Low Volume"]})');
assert.strictEqual(noNlvTarget.available, false);
assert.strictEqual(noNlvTarget.unavailableReason, "live NLV unavailable");
assert.deepStrictEqual(value('hedgeScenarios(PA_NO_NLV_MODEL, 0.5, ["Oversold Low Volume"])'), []);
assert.ok(vm.runInContext('hedgeVerdict(' + JSON.stringify(noNlvTarget) + ', "MES")', sandbox).includes("unavailable"));

sandbox.INCOMPLETE_FUT = {
  key:"primary", nlv:1, positions:[{symbol:"MESU6",sec_type:"FUT",position:-2,avg_cost:null,market_price:null}], orders:[],
};
const incompleteFut = value('attributeBook(INCOMPLETE_FUT, BETAS, {}, {today:"2026-08-25"})');
assert.strictEqual(incompleteFut.futuresComplete, false);
sandbox.INCOMPLETE_FUT_MODEL = incompleteFut;
const incompleteTarget = value('hedgeTarget(INCOMPLETE_FUT_MODEL, 0.5, {multiplier:5,indexLevel:7635,scopeStrategies:["Oversold Low Volume"]})');
assert.strictEqual(incompleteTarget.available, false);
assert.strictEqual(incompleteTarget.unavailableReason, "index-futures exposure incomplete");

// The mount is between positions and orders, and rendered HTML is strictly
// display-only in fresh, stale, offline, and missing-beta states.
const shellHtml = vm.runInContext('shell()', sandbox);
assert.ok(shellHtml.indexOf('id="positions"') < shellHtml.indexOf('id="hedge"'));
assert.ok(shellHtml.indexOf('id="hedge"') < shellHtml.indexOf('id="orders"'));
vm.runInContext(`
  state.account = "primary";
  state.status = {online:true, configured:true};
  state.book = {at:Date.now(), accounts:[ACCOUNT, PA]};
  FUT_SPECS = SPECS; HEDGE_BETAS = BETAS; HEDGE_BETA_STATUS = "loaded";
`, sandbox);
let html = vm.runInContext('renderHedge()', sandbox);
assert.ok(html.includes("Hedge (display only)"));
assert.ok(html.includes("$750,000 sizing"));
assert.ok(html.includes("Display only — nothing here sends orders."));
assert.ok(html.includes("Market component only — idiosyncratic moves"));
assert.ok(!html.includes("data-mutation"));

// Scope survives the same account's 4-second-style rerender, but is isolated
// by selected account. Persistence uses the three specified localStorage keys.
vm.runInContext('hedgeScopeForAccount().add("Unattributed")', sandbox);
html = vm.runInContext('renderHedge()', sandbox);
assert.match(html, /data-hedge-strategy="Unattributed" checked/);
vm.runInContext('state.account="pa"', sandbox);
const paHtml = vm.runInContext('renderHedge()', sandbox);
assert.ok(paHtml.includes("$66,557 live NLV"));
assert.doesNotMatch(paHtml, /data-hedge-strategy="Unattributed" checked/);
vm.runInContext(`
  hedgePrefs.betaKey="beta63"; hedgePersist("hedge.betaKey", hedgePrefs.betaKey);
  hedgePrefs.contract="ES"; hedgePersist("hedge.contract", hedgePrefs.contract);
  hedgePrefs.targetPct=75; hedgePersist("hedge.targetPct", hedgePrefs.targetPct);
`, sandbox);
assert.strictEqual(storage.get("hedge.betaKey"), "beta63");
assert.strictEqual(storage.get("hedge.contract"), "ES");
assert.strictEqual(storage.get("hedge.targetPct"), "75");

vm.runInContext('state.account="primary"; state.book.at=Date.now()-100000; state.status.online=true', sandbox);
const stale = vm.runInContext('renderHedge()', sandbox);
assert.ok(stale.includes("is-stale") && stale.includes("book stale (100s)"));
assert.ok(!stale.includes("data-mutation"));
vm.runInContext('state.book.at=Date.now(); state.status.online=false', sandbox);
const offline = vm.runInContext('renderHedge()', sandbox);
assert.ok(offline.includes("is-stale") && offline.includes("agent offline"));
assert.ok(!offline.includes("data-mutation"));
vm.runInContext('state.status.online=true; HEDGE_BETAS=null; HEDGE_BETA_STATUS="absent-in-build"', sandbox);
const noBetas = vm.runInContext('renderHedge()', sandbox);
assert.ok(noBetas.includes("no beta table in this build (build_betas.py skipped)"));
assert.ok(!noBetas.includes("data-mutation"));

console.log("PASS execution hedge attribution, targeting, scenarios, roll-off, persistence, degradation, and display-only contract");
'''
    script = (
        script.replace("__EXECUTION_JS__", json.dumps(str(EXECUTION_JS)))
        .replace("__ACCOUNT__", json.dumps(account))
        .replace("__PA__", json.dumps(pa))
        .replace("__BETAS__", json.dumps(betas))
        .replace("__SPECS__", json.dumps(specs))
    )

    result = subprocess.run(
        [shutil.which("node"), "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
