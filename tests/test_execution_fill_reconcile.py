"""Execution-broker delayed fill-price reconciliation contracts."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "execution-broker" / "src" / "fill-reconcile.mjs"
BROKER = ROOT / "execution-broker" / "src" / "index.js"
LOCAL_BOOK_SNAPSHOT = Path(r"C:\Users\McKinley Slade\OneDrive\trading_ibkr\book_snapshot.py")
LOCAL_EXECUTOR = Path(r"C:\Users\McKinley Slade\OneDrive\trading_ibkr\execute_order.py")


def _run_node(script: str) -> str:
    out = subprocess.run(
        [shutil.which("node"), "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_delayed_price_backfills_command_and_sparse_push_cannot_erase_it():
    script = f"""
import {{ commandFillMatch, executionFamilyId, mergeCommandResult, mergeExecutionFill,
          reconcileCommandFills }} from {json.dumps(HELPERS.as_uri())};

const command = {{
  id: "c1", type: "entry_bracket", account: "primary",
  payload: {{symbol: "OXY", sec_type: "STK", action: "BUY", quantity: 100}},
  result: {{fill: {{order_ids: [101, 102, 103], parent_status: "Submitted"}}}},
}};
command.fill_match = commandFillMatch(command);

const first = mergeExecutionFill(null, {{
  exec_id: "e1", order_id: 101, account_key: "primary", symbol: "OXY",
  sec_type: "STK", client_id:123, side: "BOT", qty: 100, price: null, avg_price: null,
}}, 10);
if (first.price !== null || first.ingested_at !== 10) throw new Error("initial sparse fill changed");

const settled = mergeExecutionFill(first, {{
  exec_id: "e1", order_id: 101, account_key: "primary", symbol: "OXY",
  sec_type: "STK", client_id:123, side: "BOT", qty: 100, price: 50.25, avg_price: 50.25,
}}, 20);
if (settled.price !== 50.25 || settled.ingested_at !== 10) throw new Error("delayed price not merged");
const preserved = mergeExecutionFill(settled, {{exec_id: "e1", price: null, avg_price: 0}}, 30);
if (preserved.price !== 50.25 || preserved.avg_price !== 50.25) throw new Error("sparse push erased price");

const original = mergeExecutionFill(null, {{exec_id:"0001.01", qty:100, price:50}}, 31);
const correction = mergeExecutionFill(original, {{exec_id:"0001.02", qty:80, price:51}}, 32);
if (executionFamilyId(correction.exec_id) !== "0001" || correction.exec_id !== "0001.02"
    || correction.qty !== 80 || correction.price !== 51)
  throw new Error("execution correction did not supersede its original");
const outOfOrderOriginal = mergeExecutionFill(correction,
  {{exec_id:"0001.01", qty:100, price:50}}, 33);
if (outOfOrderOriginal.exec_id !== "0001.02" || outOfOrderOriginal.qty !== 80)
  throw new Error("older execution revision overwrote its correction");

const reconciled = reconcileCommandFills([command], [preserved], 40);
if (!reconciled.changed) throw new Error("command was not reconciled");
const fill = reconciled.commands[0].result.fill;
if (fill.filled !== 100 || fill.avg_fill !== 50.25 || fill.status !== "Filled")
  throw new Error("wrong command backfill: " + JSON.stringify(fill));
if (fill.order_id != null || fill.matched_order_ids[0] !== "101" || fill.reconciled_at !== 40)
  throw new Error("missing or unsafe reconciliation metadata");

const staleResult = mergeCommandResult(reconciled.commands[0].result, {{
  ok: true, detail:"submitted", fill: {{order_id: 101, filled: 10, avg_fill: 49, status: "Submitted"}},
}});
if (staleResult.fill.avg_fill !== 50.25 || staleResult.fill.filled !== 100 || staleResult.fill.status !== "Filled")
  throw new Error("late sparse result erased authoritative backfill");
if (!staleResult.detail.includes("later execution reconciled from IBKR"))
  throw new Error("reconciliation context was lost");
console.log("OK");
"""
    assert "OK" in _run_node(script)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_partial_vwap_ignores_children_other_accounts_and_unrelated_orders():
    script = f"""
import {{ commandFillMatch, reconcileCommandFills }} from {json.dumps(HELPERS.as_uri())};
const command = {{
  id: "c2", type: "entry_bracket", account: "pa",
  created_at: Date.parse("2026-08-27T13:30:00Z"),
  payload: {{symbol: "LUV", sec_type: "STK", action: "BUY", quantity: 100}},
  result: {{fill: {{order_ids: [201, 202, 203], parent_status: "Submitted"}}}},
}};
command.fill_match = commandFillMatch(command);
const fills = [
  {{exec_id:"a", order_id:201, account_key:"pa", time:"2026-08-27T13:31:00Z", symbol:"LUV", sec_type:"STK", client_id:147, side:"BOT", qty:60, price:40.0}},
  {{exec_id:"b", order_id:201, account_key:"pa", time:"2026-08-27T13:32:00Z", symbol:"LUV", sec_type:"STK", client_id:147, side:"BOT", qty:40, price:40.5}},
  // Same bracket child, but opposite side: must never contaminate the entry VWAP.
  {{exec_id:"child", order_id:202, account_key:"pa", time:"2026-08-27T13:33:00Z", symbol:"LUV", sec_type:"STK", client_id:147, side:"SLD", qty:100, price:39.0}},
  {{exec_id:"acct", order_id:201, account_key:"primary", time:"2026-08-27T13:31:00Z", symbol:"LUV", sec_type:"STK", client_id:147, side:"BOT", qty:100, price:99}},
  {{exec_id:"symbol", order_id:201, account_key:"pa", time:"2026-08-27T13:31:00Z", symbol:"OXY", sec_type:"STK", client_id:147, side:"BOT", qty:100, price:88}},
  {{exec_id:"type", order_id:201, account_key:"pa", time:"2026-08-27T13:31:00Z", symbol:"LUV", sec_type:"FUT", client_id:147, side:"BOT", qty:100, price:66}},
  {{exec_id:"client", order_id:201, account_key:"pa", time:"2026-08-27T13:31:00Z", symbol:"LUV", sec_type:"STK", client_id:999, side:"BOT", qty:100, price:44}},
  {{exec_id:"old", order_id:201, account_key:"pa", time:"2026-08-27T12:00:00Z", symbol:"LUV", sec_type:"STK", client_id:147, side:"BOT", qty:100, price:55}},
  {{exec_id:"order", order_id:999, account_key:"pa", time:"2026-08-27T13:31:00Z", symbol:"LUV", sec_type:"STK", client_id:147, side:"BOT", qty:100, price:77}},
];
const out = reconcileCommandFills([command], fills, 50);
const f = out.commands[0].result.fill;
const expected = (60 * 40.0 + 40 * 40.5) / 100;
if (f.filled !== 100 || Math.abs(f.avg_fill - expected) > 1e-10)
  throw new Error("unsafe or wrong VWAP: " + JSON.stringify(f));
console.log("OK");
"""
    assert "OK" in _run_node(script)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_cumulative_average_is_used_only_as_order_level_fallback():
    script = f"""
import {{ reconcileCommandFills }} from {json.dumps(HELPERS.as_uri())};
const command = {{
  type:"add_to_position", account:"primary",
  result:{{fill:{{add:{{order_id:301, status:"Submitted", filled:0, avg_fill:0}}}}}},
}};
const fills = [
  {{exec_id:"p1", order_id:301, account_key:"primary", client_id:123, qty:40, price:10.0, avg_price:10.0, cum_qty:40}},
  // ex.price is delayed on the second partial; ex.avgPrice is cumulative for all 100 shares.
  {{exec_id:"p2", order_id:301, account_key:"primary", client_id:123, qty:60, price:null, avg_price:10.6, cum_qty:100}},
];
const out = reconcileCommandFills([command], fills, 60);
const f = out.commands[0].result.fill;
if (f.filled !== 100 || f.avg_fill !== 10.6)
  throw new Error("cumulative average mishandled: " + JSON.stringify(f));

const incompleteAverage = [
  {{exec_id:"q1", order_id:301, account_key:"primary", client_id:123, qty:40,
    price:10.0, avg_price:10.0, cum_qty:40}},
  // The later partial exists, but IBKR has not supplied either its print or a
  // cumulative average covering all 100 shares. The broker must wait.
  {{exec_id:"q2", order_id:301, account_key:"primary", client_id:123, qty:60,
    price:null, avg_price:null, cum_qty:100}},
];
const deferred = reconcileCommandFills([command], incompleteAverage, 61);
if (deferred.changed || deferred.commands[0].result.fill.avg_fill != null)
  throw new Error("incomplete cumulative average was published");

// If an older partial aged out or was capped, cum_qty can exceed the retained
// rows. Its all-order average cannot price only the retained tail.
const retainedTail = [
  {{exec_id:"q2", order_id:301, account_key:"primary", client_id:123, qty:60,
    price:null, avg_price:10.6, cum_qty:100}},
];
const missingHead = reconcileCommandFills([command], retainedTail, 62);
if (missingHead.changed || missingHead.commands[0].result.fill.avg_fill != null)
  throw new Error("cumulative average with missing partials was published");

const completeDirect = [
  {{exec_id:"r1", order_id:301, account_key:"primary", client_id:123, qty:40,
    price:10.0, avg_price:10.0, cum_qty:40}},
  {{exec_id:"r2", order_id:301, account_key:"primary", client_id:123, qty:60,
    price:11.0, avg_price:10.6, cum_qty:100}},
];
const completed = reconcileCommandFills([command], completeDirect, 63);
const truncated = reconcileCommandFills(completed.commands, [completeDirect[1]], 64);
const kept = truncated.commands[0].result.fill;
if (truncated.changed || kept.filled !== 100 || kept.avg_fill !== 10.6)
  throw new Error("retention truncation regressed a completed command");
console.log("OK");
"""
    assert "OK" in _run_node(script)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_futures_alias_and_option_spread_backfill():
    script = f"""
import {{ commandFillMatch, reconcileCommandFills }} from {json.dumps(HELPERS.as_uri())};
const future = {{type:"entry_bracket", account:"primary", created_at:1,
  payload:{{symbol:"6E", fut_ib_symbol:"EUR", sec_type:"FUT", action:"BUY", quantity:1}},
  result:{{fill:{{order_ids:[401,402]}}}}}};
future.fill_match = commandFillMatch(future);
const option = {{type:"option_spread", account:"primary", created_at:1,
  payload:{{symbol:"SPY", action:"BUY", quantity:2, legs:[
    {{side:"BUY", right:"C", strike:500}}, {{side:"SELL", right:"C", strike:505}}
  ]}},
  result:{{fill:{{order_id:501, perm_id:9001, status:"Submitted"}}}}}};
option.fill_match = commandFillMatch(option);
const fills = [
  {{exec_id:"f",order_id:401,account_key:"primary",client_id:123,symbol:"EUR",sec_type:"FUT",side:"BOT",qty:1,price:1.19}},
  // IBKR can emit these component OPT rows alongside the authoritative BAG
  // execution. Neither leg premium may be mistaken for the net spread fill.
  {{exec_id:"longleg",order_id:501,perm_id:9001,account_key:"primary",client_id:123,symbol:"SPY",sec_type:"OPT",side:"BOT",qty:2,price:8.0}},
  {{exec_id:"shortleg",order_id:501,perm_id:9001,account_key:"primary",client_id:123,symbol:"SPY",sec_type:"OPT",side:"SLD",qty:2,price:5.5}},
  {{exec_id:"o",order_id:501,perm_id:9001,account_key:"primary",client_id:123,symbol:"SPY",sec_type:"BAG",side:"BOT",qty:2,price:2.5}},
  {{exec_id:"wrongperm",order_id:501,perm_id:9999,account_key:"primary",client_id:123,symbol:"SPY",sec_type:"BAG",side:"BOT",qty:2,price:99}},
];
const out = reconcileCommandFills([future, option], fills, 70).commands;
if (out[0].result.fill.avg_fill !== 1.19) throw new Error("futures alias did not reconcile");
if (out[1].result.fill.avg_fill !== 2.5 || out[1].result.fill.filled !== 2)
  throw new Error("option spread did not reconcile safely");

const noBag = reconcileCommandFills([option], fills.filter((row) => row.sec_type === "OPT"), 71);
if (noBag.changed || noBag.commands[0].result.fill.avg_fill != null)
  throw new Error("option legs were accepted without a BAG aggregate");
console.log("OK");
"""
    assert "OK" in _run_node(script)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_broker_reconciles_durable_union_across_staggered_snapshots():
    script = f"""
import fs from "node:fs";
import {{ commandFillMatch }} from {json.dumps(HELPERS.as_uri())};
let source = fs.readFileSync({json.dumps(str(BROKER))}, "utf8");
source = source.replace('import {{ DurableObject }} from "cloudflare:workers";',
  'class DurableObject {{ constructor(ctx, env) {{ this.ctx = ctx; this.env = env; }} }}');
source = source.replace('from "./fill-reconcile.mjs";',
  'from {json.dumps(HELPERS.as_uri())};');
const mod = await import("data:text/javascript;base64," + Buffer.from(source).toString("base64"));
class Storage {{
  constructor() {{ this.values = new Map(); }}
  async get(key) {{ const v=this.values.get(key); return v == null ? v : structuredClone(v); }}
  async put(key, value) {{ this.values.set(key, structuredClone(value)); }}
  async delete(key) {{ this.values.delete(key); }}
  async list({{prefix=""}}={{}}) {{
    return new Map([...this.values].filter(([key]) => key.startsWith(prefix))
      .map(([key,value]) => [key, structuredClone(value)]));
  }}
}}
const storage = new Storage();
const broker = new mod.ExecBroker({{storage, getWebSockets(){{return [];}}}}, {{}});
const day1 = Date.now() - 86_400_000, day2 = Date.now();
const command = {{type:"entry_bracket", account:"primary", created_at:day1-60_000,
  payload:{{symbol:"OXY",sec_type:"STK",action:"BUY",quantity:100}},
  result:{{detail:"submitted",fill:{{order_ids:[601,602],parent_status:"Submitted"}}}}}};
command.fill_match = commandFillMatch(command);
await storage.put("recent_commands", [command]);
await storage.put("scheduled_commands", []);
await broker._mergeFills({{accounts:[{{key:"primary",label:"Primary",fills:[
  {{exec_id:"first",time:new Date(day1).toISOString(),order_id:601,account_key:"primary",client_id:123,symbol:"OXY",sec_type:"STK",side:"BOT",qty:40,price:50}},
]}}]}});
await broker._mergeFills({{accounts:[{{key:"primary",label:"Primary",fills:[
  {{exec_id:"second",time:new Date(day2).toISOString(),order_id:602,account_key:"primary",client_id:123,symbol:"OXY",sec_type:"STK",side:"BOT",qty:60,price:51}},
]}}]}});
const final = (await storage.get("recent_commands"))[0].result.fill;
if (final.filled !== 100 || Math.abs(final.avg_fill - 50.6) > 1e-10)
  throw new Error("durable union was not reconciled: " + JSON.stringify(final));

const correctedCommand = {{type:"close_only", account:"primary", created_at:day2-60_000,
  payload:{{symbol:"OXY",sec_type:"STK",action:"BUY",qty:80}},
  result:{{detail:"submitted",fill:{{order_id:701,status:"Submitted"}}}}}};
correctedCommand.fill_match = commandFillMatch(correctedCommand);
await storage.put("recent_commands", [correctedCommand]);
await broker._mergeFills({{accounts:[{{key:"primary",label:"Primary",fills:[
  {{exec_id:"ibkr-fill.01",time:new Date(day2).toISOString(),order_id:701,account_key:"primary",client_id:123,symbol:"OXY",sec_type:"STK",side:"BOT",qty:100,price:50}},
]}}]}});
await broker._mergeFills({{accounts:[{{key:"primary",label:"Primary",fills:[
  {{exec_id:"ibkr-fill.02",time:new Date(day2).toISOString(),order_id:701,account_key:"primary",client_id:123,symbol:"OXY",sec_type:"STK",side:"BOT",qty:80,price:51}},
]}}]}});
const corrected = (await storage.get("recent_commands"))[0].result.fill;
if (corrected.filled !== 80 || corrected.avg_fill !== 51)
  throw new Error("corrected execution was double-counted: " + JSON.stringify(corrected));
const correctionDay = new Date(day2).toISOString().slice(0, 10);
const storedRevisions = (await storage.get(`fills:${{correctionDay}}`))
  .filter((row) => row.order_id === 701);
if (storedRevisions.length !== 1 || storedRevisions[0].exec_id !== "ibkr-fill.02")
  throw new Error("execution revision family was not replaced");

// Migrate correction duplicates written by the pre-reconciliation broker,
// including the harder newest-then-oldest retained order.
const legacyAt = Date.now() - 3 * 86_400_000;
const legacyDay = new Date(legacyAt).toISOString().slice(0, 10);
await storage.put(`fills:${{legacyDay}}`, [
  {{exec_id:"legacy.02",time:new Date(legacyAt).toISOString(),order_id:702,account_key:"primary",qty:80,price:51}},
  {{exec_id:"legacy.01",time:new Date(legacyAt).toISOString(),order_id:702,account_key:"primary",qty:100,price:50}},
]);
await broker._mergeFills({{accounts:[]}});
const migrated = await storage.get(`fills:${{legacyDay}}`);
if (migrated.length !== 1 || migrated[0].exec_id !== "legacy.02" || migrated[0].qty !== 80)
  throw new Error("retained correction chain was not migrated");

// At the per-day cap, a newly arriving execution must remain eligible for
// delayed-price reconciliation instead of being discarded as row 501.
const burstAt = Date.now() - 2 * 86_400_000;
const burstDay = new Date(burstAt).toISOString().slice(0, 10);
const burst = Array.from({{length:501}}, (_, i) => ({{
  exec_id:`burst-${{i}}`, time:new Date(burstAt + i * 1000).toISOString(),
  order_id:800+i, client_id:123, symbol:"OXY", sec_type:"STK", side:"BOT", qty:1, price:50,
}}));
await broker._mergeFills({{accounts:[{{key:"primary",label:"Primary",fills:burst}}]}});
const capped = await storage.get(`fills:${{burstDay}}`);
if (capped.length !== 500 || capped.some((row) => row.exec_id === "burst-0")
    || !capped.some((row) => row.exec_id === "burst-500"))
  throw new Error("fill cap did not retain the newest execution");

const tieAt = Date.now() - 4 * 86_400_000;
const tieDay = new Date(tieAt).toISOString().slice(0, 10);
const tieTime = new Date(tieAt).toISOString();
await storage.put(`fills:${{tieDay}}`, Array.from({{length:500}}, (_, i) => ({{
  exec_id:`old-tie-${{i}}`, time:tieTime, ingested_at:1, order_id:1400+i,
  account_key:"primary", client_id:123, symbol:"OXY", sec_type:"STK", side:"BOT", qty:1, price:50,
}})));
await broker._mergeFills({{accounts:[{{key:"primary",label:"Primary",fills:[{{
  exec_id:"new-tie", time:tieTime, order_id:2000, client_id:123,
  symbol:"OXY", sec_type:"STK", side:"BOT", qty:1, price:51,
}}]}}]}});
const tied = await storage.get(`fills:${{tieDay}}`);
if (tied.length !== 500 || !tied.some((row) => row.exec_id === "new-tie"))
  throw new Error("same-second new execution was dropped at the cap");
console.log("OK");
"""
    assert "OK" in _run_node(script)


@pytest.mark.skipif(not LOCAL_BOOK_SNAPSHOT.exists(), reason="live execution checkout is unavailable")
def test_local_book_snapshot_exports_execution_client_id():
    src = LOCAL_BOOK_SNAPSHOT.read_text(encoding="utf-8")
    assert '"client_id": int(getattr(ex, "clientId", 0) or 0)' in src


@pytest.mark.skipif(not LOCAL_EXECUTOR.exists(), reason="live execution checkout is unavailable")
def test_broker_client_ids_match_live_executor():
    helper = HELPERS.read_text(encoding="utf-8")
    executor = LOCAL_EXECUTOR.read_text(encoding="utf-8")
    assert "const EXECUTION_CLIENT_IDS = { primary: 123, pa: 147 };" in helper
    assert 'PORTS = {"primary": ("127.0.0.1", 7496, 123), "pa": ("127.0.0.1", 4001, 147)}' in executor
