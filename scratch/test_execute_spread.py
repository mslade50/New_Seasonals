"""Unit test for execute_order.py's option_spread (BAG combo) path, with a mocked
broker — no connection, no real order. Verifies the pure helpers (snap, risk,
combo construction), the live gates, conId cross-check aborts, the vertical
debit>=width sanity abort, and the OPT-flatten guard."""
import contextlib
import io
import json
import sys

sys.path.insert(0, r"C:\Users\McKinley Slade\OneDrive\trading_ibkr")
import execute_order as ex

ok = True
def chk(n, c):
    global ok; ok = ok and bool(c); print(("OK   " if c else "FAIL ") + n)

# ---- snap_combo_limit (pure) ----
chk("BUY 1.53 snaps down to 1.50", ex.snap_combo_limit(1.53, "BUY", 0.05) == 1.50)
chk("BUY 1.55 stays 1.55", ex.snap_combo_limit(1.55, "BUY", 0.05) == 1.55)
chk("SELL 1.53 snaps up to 1.55", ex.snap_combo_limit(1.53, "SELL", 0.05) == 1.55)
chk("BUY 0.04 snaps to 0.00", ex.snap_combo_limit(0.04, "BUY", 0.05) == 0.0)
chk("tick 0 -> passthrough", ex.snap_combo_limit(1.234, "BUY", 0) == 1.234)

# ---- opt_spread_risk (pure) ----
LEGS2 = [{"ratio": 1}, {"ratio": 1}]
chk("risk 13x 1.55 vertical = 2015 + 33.80 comm",
    abs(ex.opt_spread_risk(1.55, 13, LEGS2) - (1.55 * 100 * 13 + 0.65 * 26 * 2)) < 1e-9)

# ---- build_combo (pure) ----
combo = ex.build_combo("XOM", [(111, 1, "BUY"), (222, 1, "SELL")])
chk("BAG SMART USD", combo.secType == "BAG" and combo.exchange == "SMART" and combo.currency == "USD" and combo.symbol == "XOM")
chk("2 combo legs, conIds + actions", len(combo.comboLegs) == 2
    and combo.comboLegs[0].conId == 111 and combo.comboLegs[0].action == "BUY"
    and combo.comboLegs[1].conId == 222 and combo.comboLegs[1].action == "SELL"
    and all(l.exchange == "SMART" for l in combo.comboLegs))

# ---- main() gates (mocked IB) ----
CONIDS = {105.0: 71001, 115.0: 71002}   # strike -> conId the fake qualifier assigns

class FakeClient:
    def __init__(s): s._i = 1000
    def getReqId(s): s._i += 1; return s._i
class FT:
    def __init__(s, o): s.order = o; s.orderStatus = type("S", (), {"status": "PreSubmitted", "filled": 0, "avgFillPrice": 0})()
class _Ev:
    def __iadd__(s, f): return s
class FakePos:
    def __init__(s, sec):
        s.position = 100
        s.contract = type("C", (), {"symbol": "XOM", "secType": sec,
                                    "lastTradeDateOrContractMonth": "20260814",
                                    "conId": 9, "exchange": ""})()
        s.account = "DU999"
class FakeIB:
    PLACED = []
    POSITIONS = []
    def __init__(s): s.client = FakeClient(); s.errorEvent = _Ev()
    def connect(s, *a, **k): pass
    def managedAccounts(s): return ["DU999"]
    def positions(s): return FakeIB.POSITIONS
    def qualifyContracts(s, *a):
        out = []
        for c in a:
            cid = CONIDS.get(getattr(c, "strike", None))
            if cid:
                c.conId = cid
                out.append(c)
        return out
    def placeOrder(s, c, o): FakeIB.PLACED.append((c, o)); return FT(o)
    def reqAllOpenOrders(s): pass
    def openTrades(s): return []
    def sleep(s, *a): pass
    def disconnect(s): pass
ex.IB = FakeIB

def run(cmd, env):
    for k, v in env.items():
        setattr(ex, k, v)
    FakeIB.PLACED = []
    sys.argv = ["execute_order.py", json.dumps(cmd)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ex.main()
    return json.loads(buf.getvalue())

def SPREAD(**kw):
    p = {"symbol": "XOM", "action": "BUY", "quantity": 5, "limit": 1.55, "tif": "DAY",
         "structure": "call_debit_vertical", "debit_risk": 1.55,
         "legs": [{"side": "BUY", "right": "C", "expiry": "20260814", "strike": 105.0, "ratio": 1, "con_id": 71001},
                  {"side": "SELL", "right": "C", "expiry": "20260814", "strike": 115.0, "ratio": 1, "con_id": 71002}]}
    p.update(kw)
    return {"type": "option_spread", "account": "pa", "payload": p}

ARM = dict(LIVE_ENABLED=True, LIVE_ACCOUNTS={"pa", "primary"}, LIVE_TYPES={"option_spread", "flatten"},
           LIVE_MAX_OPT_CONTRACTS=10, LIVE_MAX_OPT_RISK=2500.0, LIVE_MAX_OPT_RISK_BY_ACCT={},
           OPT_COMBO_TICK=0.05)

chk("disarmed -> rejected, nothing placed",
    (lambda r: r["state"] == "rejected" and not FakeIB.PLACED)(run(SPREAD(), {**ARM, "LIVE_ENABLED": False})))
chk("type not armed -> rejected",
    (lambda r: r["state"] == "rejected" and "not armed" in r["detail"])(run(SPREAD(), {**ARM, "LIVE_TYPES": {"flatten"}})))
chk("SELL (credit) -> rejected phase 1",
    (lambda r: r["state"] == "rejected" and "BUY/debit only" in r["detail"])(run(SPREAD(action="SELL"), ARM)))
chk("over contract cap -> rejected",
    (lambda r: r["state"] == "rejected" and "LIVE_MAX_OPT_CONTRACTS" in r["detail"])(run(SPREAD(quantity=11), ARM)))
chk("over risk cap -> rejected",
    (lambda r: r["state"] == "rejected" and "exceeds" in r["detail"])(run(SPREAD(quantity=10, debit_risk=3.0, limit=3.0), ARM)))
chk("per-acct risk cap override honored",
    (lambda r: r["state"] == "rejected" and "pa cap" in r["detail"])(
        run(SPREAD(), {**ARM, "LIVE_MAX_OPT_RISK_BY_ACCT": {"pa": 100.0}})))
chk("debit >= width -> aborted",   # qty 2 keeps risk under the cap so the width check is reached
    (lambda r: r["state"] == "rejected" and "width" in r["detail"])(run(SPREAD(quantity=2, limit=10.0, debit_risk=10.0), ARM)))
chk("1-leg payload -> rejected",
    (lambda r: r["state"] == "rejected" and "2-4 legs" in r["detail"])(
        run(SPREAD(legs=[{"side": "BUY", "right": "C", "expiry": "20260814", "strike": 105.0}]), ARM)))
chk("bad right -> rejected",
    (lambda r: r["state"] == "rejected" and "right" in r["detail"])(
        run(SPREAD(legs=[{"side": "BUY", "right": "X", "expiry": "20260814", "strike": 105.0},
                         {"side": "SELL", "right": "C", "expiry": "20260814", "strike": 115.0}]), ARM)))
chk("conId mismatch -> aborted",
    (lambda r: r["state"] == "rejected" and "conId mismatch" in r["detail"] and not FakeIB.PLACED)(
        run(SPREAD(legs=[{"side": "BUY", "right": "C", "expiry": "20260814", "strike": 105.0, "con_id": 99999},
                         {"side": "SELL", "right": "C", "expiry": "20260814", "strike": 115.0, "con_id": 71002}]), ARM)))
chk("unqualifiable strike -> aborted",
    (lambda r: r["state"] == "rejected" and "did not qualify" in r["detail"])(
        run(SPREAD(legs=[{"side": "BUY", "right": "C", "expiry": "20260814", "strike": 999.0},
                         {"side": "SELL", "right": "C", "expiry": "20260814", "strike": 115.0}]), ARM)))

r = run(SPREAD(limit=1.53), ARM)     # 1.53 must snap down to 1.50
chk("valid armed -> executed, one BAG order", r["state"] == "executed" and len(FakeIB.PLACED) == 1)
if FakeIB.PLACED:
    c, o = FakeIB.PLACED[0]
    chk("placed BAG with 2 legs", c.secType == "BAG" and len(c.comboLegs) == 2
        and c.comboLegs[0].conId == 71001 and c.comboLegs[1].conId == 71002)
    chk("LMT BUY 5 @1.50 DAY transmit", o.orderType == "LMT" and o.action == "BUY"
        and o.totalQuantity == 5 and o.lmtPrice == 1.50 and o.tif == "DAY")
    chk("no account pin", not getattr(o, "account", ""))

# missing con_id in payload is allowed (qualification is authoritative)
r = run(SPREAD(legs=[{"side": "BUY", "right": "C", "expiry": "20260814", "strike": 105.0},
                     {"side": "SELL", "right": "C", "expiry": "20260814", "strike": 115.0}]), ARM)
chk("no payload conId -> still executes", r["state"] == "executed")

# ---- OPT flatten guard ----
FakeIB.POSITIONS = [FakePos("OPT")]
r = run({"type": "flatten", "account": "pa", "payload": {"symbol": "XOM", "sec_type": "OPT", "expiry": "202608"}}, ARM)
chk("flatten OPT -> rejected with combo-ticket message",
    r["state"] == "rejected" and "combo ticket" in r["detail"] and not FakeIB.PLACED)

print("\n" + ("ALL PASS" if ok else "FAILURES"))
sys.exit(0 if ok else 1)
