"""Trade Log tab guards: nav + page + proxy wiring, the broker DO's fills
ring, and the client-side order aggregation (partial-fill roll-up, VWAP,
orderRef strategy parse)."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMMON_JS = ROOT / "site" / "assets" / "common.js"
TRADELOG_JS = ROOT / "site" / "assets" / "tradelog.js"


def test_nav_has_tradelog_entry():
    src = COMMON_JS.read_text(encoding="utf-8")
    assert "tradelog.html" in src
    assert "Trade Log" in src


def test_page_and_proxy_wired():
    html = (ROOT / "site" / "tradelog.html").read_text(encoding="utf-8")
    assert "assets/tradelog.js" in html
    assert "assets/common.js" in html
    fn = (ROOT / "functions" / "exec-fills.js").read_text(encoding="utf-8")
    assert "requireAccess" in fn
    assert "/fills" in fn
    assert "STATUS_TOKEN" in fn


def test_broker_do_fills_ring():
    src = (ROOT / "execution-broker" / "src" / "index.js").read_text(encoding="utf-8")
    assert '"/fills"' in src            # route present + registered in DO_PATHS
    assert src.count("/fills") >= 2
    assert "_mergeFills" in src
    assert "_reconcileCommandFills" in src
    assert "mergeExecutionFill" in src
    assert "reconcileCommandFills" in src
    assert "FILLS_RETENTION_DAYS" in src
    assert "FILLS_DAY_CAP" in src
    # the stored book must be stripped of fills (DO per-value size limit)
    assert "({ fills, ...rest })" in src


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_aggregation_rolls_partials_and_parses_strategy():
    script = r"""
const fs = require("fs");
const vm = require("vm");
const sandbox = { document: { addEventListener() {} }, console };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(__COMMON_JS__, "utf8"), sandbox);
vm.runInContext(fs.readFileSync(__TRADELOG_JS__, "utf8"), sandbox);
const fills = [
  {exec_id: "a1", time: "2026-07-23T14:31:00+00:00", account_key: "primary",
   account_label: "Primary (TWS)", symbol: "OXY", sec_type: "STK", side: "BOT",
   qty: 60, price: 50.0, perm_id: 111, order_ref: "OXY|BUY|OLV|2026-07-22",
   commission: 1.0},
  {exec_id: "a2", time: "2026-07-23T14:32:00+00:00", account_key: "primary",
   account_label: "Primary (TWS)", symbol: "OXY", sec_type: "STK", side: "BOT",
   qty: 40, price: 50.5, perm_id: 111, order_ref: "OXY|BUY|OLV|2026-07-22",
   commission: 0.5},
  {exec_id: "b1", time: "2026-07-23T15:00:00+00:00", account_key: "pa",
   account_label: "PA (Gateway)", symbol: "SPY", sec_type: "STK", side: "SLD",
   qty: 10, price: 700, perm_id: 222, order_ref: null, realized_pnl: 123.4},
];
const rows = sandbox.aggregateOrders(fills);
if (rows.length !== 2) throw new Error("expected 2 order rows, got " + rows.length);
const oxy = rows.find(r => r.symbol === "OXY");
if (oxy.qty !== 100) throw new Error("qty roll-up wrong: " + oxy.qty);
const vwap = (60 * 50.0 + 40 * 50.5) / 100;
if (Math.abs(oxy.avg_price - vwap) > 1e-9) throw new Error("vwap wrong: " + oxy.avg_price);
if (oxy.strategy !== "OLV") throw new Error("strategy parse wrong: " + oxy.strategy);
if (oxy.side !== "BUY" || oxy.n_fills !== 2) throw new Error("side/fill-count wrong");
if (Math.abs(oxy.commission - 1.5) > 1e-9) throw new Error("commission sum wrong");
const spy = rows.find(r => r.symbol === "SPY");
if (spy.side !== "SELL") throw new Error("SLD -> SELL mapping wrong");
if (spy.realized_pnl !== 123.4) throw new Error("realized pnl wrong");
if (spy.account !== "PA (Gateway)") throw new Error("account label wrong");
// a short orderRef (no 4 pipe fields) falls through verbatim
if (sandbox.stratFromRef("manual") !== "manual") throw new Error("short ref handling wrong");
console.log("OK");
""".replace("__COMMON_JS__", json.dumps(str(COMMON_JS))).replace(
        "__TRADELOG_JS__", json.dumps(str(TRADELOG_JS)))
    out = subprocess.run([shutil.which("node"), "-e", script],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout


def test_tradelog_history_payload_from_canonical_fills(tmp_path):
    """build_tradelog_history ships every stored execution (not just the DO's
    14-day window) with the page's fields, and never the raw broker account id."""
    import sys
    for path in (ROOT, ROOT / "scripts"):   # build_site's own import convention
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from scripts import build_site
    from scripts.harvest_fills import normalize

    rows = [
        {"exec_id": "0001.01", "time": "2025-01-06T14:31:00+00:00", "account": "U1234567",
         "account_key": "primary", "account_label": "Primary (TWS)", "symbol": "OXY",
         "sec_type": "STK", "side": "BOT", "qty": 100, "price": 50.25, "perm_id": 11,
         "order_ref": "OXY|BUY|Oversold Low Volume|2025-01-03", "commission": 1.0},
        {"exec_id": "0002.01", "time": "2026-09-28T19:59:00+00:00", "account": "DU7654321",
         "account_key": "pa", "account_label": "DU7654321", "symbol": "MES", "sec_type": "FUT",
         "side": "SLD", "qty": 2, "price": 6698.25, "perm_id": 22, "con_id": 9001,
         "expiry": "202612", "realized_pnl": 117.5},
        {"exec_id": "0003.01", "time": "2026-09-28T20:00:00+00:00", "account": "U1234567",
         "symbol": "XLE", "sec_type": "STK", "side": "BOT", "qty": 1, "price": 86.7},
    ]
    path = tmp_path / "live_fills.parquet"
    normalize(rows).to_parquet(path, index=False)

    out = build_site.build_tradelog_history(fills_path=str(path))
    assert out["schema"] == 1
    assert out["first_session"] == "2025-01-06" and out["last_session"] == "2026-09-28"
    # the row without an account_key is dropped rather than falling back to the raw id
    assert out["rows"] == 2 and [r["exec_id"] for r in out["fills"]] == ["0001.01", "0002.01"]
    first, second = out["fills"]
    assert first["time"] == "2025-01-06T14:31:00Z"
    assert first["account_label"] == "Primary (TWS)" and first["perm_id"] == 11
    assert first["order_ref"].split("|")[2] == "Oversold Low Volume"
    assert second["con_id"] == 9001 and second["realized_pnl"] == 117.5
    assert "account_label" not in second          # id-shaped label withheld
    blob = json.dumps(out)
    assert "U1234567" not in blob and "DU7654321" not in blob
    assert '"account"' not in blob

    assert build_site.build_tradelog_history(fills_path=str(tmp_path / "absent.parquet")) is None
    src = (ROOT / "scripts" / "build_site.py").read_text(encoding="utf-8")
    assert 'best_effort("tradelog_history", build_tradelog_history)' in src
    assert '"tradelog_history": False' in src


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_history_and_live_window_merge_per_execution():
    script = r"""
const fs = require("fs");
const vm = require("vm");
const sandbox = { document: { addEventListener() {} }, console };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(__COMMON_JS__, "utf8"), sandbox);
vm.runInContext(fs.readFileSync(__TRADELOG_JS__, "utf8"), sandbox);
const history = [
  {exec_id: "old.01", time: "2025-01-06T14:31:00Z", account_key: "primary", symbol: "OXY", qty: 100, price: 50},
  {exec_id: "dup.01", time: "2026-09-28T14:31:00Z", account_key: "primary", symbol: "SPY", qty: 10, price: 600,
   commission: 1.25},
  {exec_id: "dup.01", time: "2026-09-28T14:31:00Z", account_key: "pa", symbol: "SPY", qty: 3, price: 600},
];
const live = [
  // IBKR correction of the stored execution: replaces it, keeps stored commission
  {exec_id: "dup.02", time: "2026-09-28T14:31:00Z", account_key: "primary", symbol: "SPY", qty: 12, price: 601,
   commission: null},
  {exec_id: "new.01", time: "2026-09-29T14:31:00Z", account_key: "primary", symbol: "XLE", qty: 5, price: 86},
];
const merged = sandbox.mergeFillSources(history, live);
const byId = Object.fromEntries(merged.map((f) => [f.account_key + ":" + f.exec_id, f]));
if (merged.length !== 4) throw new Error("expected 4 executions, got " + merged.length);
const spy = byId["primary:dup.02"];
if (!spy || spy.qty !== 12 || spy.price !== 601 || spy.commission !== 1.25) throw new Error("live override wrong");
if (!byId["pa:dup.01"]) throw new Error("same exec id on another account must stay separate");
if (!byId["primary:old.01"] || !byId["primary:new.01"]) throw new Error("history/live rows lost");
if (sandbox.tlCutoffDate(0) !== "") throw new Error("All must have no cutoff");
if (sandbox.mergeFillSources([], live).length !== 2) throw new Error("live-only merge wrong");
console.log("OK");
"""
    script = (script.replace("__COMMON_JS__", json.dumps(str(COMMON_JS)))
              .replace("__TRADELOG_JS__", json.dumps(str(TRADELOG_JS))))
    out = subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout
