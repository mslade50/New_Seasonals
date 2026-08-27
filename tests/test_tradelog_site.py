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
