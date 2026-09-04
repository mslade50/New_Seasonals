"""Unit test for execute_order.py's entry-bracket construction + gates, with a
mocked broker — no connection, no real order. Verifies the legs, prices, OCA,
parentId, transmit chain, and the live gates."""
import contextlib
import io
import json
import sys

sys.path.insert(0, r"C:\Users\McKinley Slade\OneDrive\trading_ibkr")
import execute_order as ex

ok = True
def chk(n, c):
    global ok; ok = ok and bool(c); print(("OK   " if c else "FAIL ") + n)

# ---- build_bracket (pure construction) ----
ids = iter(range(500, 600))
parent, kids = ex.build_bracket("BUY", 10, 100.0, 95.0, 120.0, "DU1", lambda: next(ids))
chk("parent LMT BUY 10 @100", parent.orderType == "LMT" and parent.action == "BUY" and parent.totalQuantity == 10 and parent.lmtPrice == 100.0)
chk("parent transmit=False, DAY, acct", parent.transmit is False and parent.tif == "DAY" and parent.account == "DU1")
tgt, stp = kids
chk("target LMT SELL @120, child, transmit F", tgt.orderType == "LMT" and tgt.action == "SELL" and tgt.lmtPrice == 120.0 and tgt.parentId == parent.orderId and tgt.ocaType == 1 and tgt.transmit is False)
chk("stop STP SELL @95, child, transmit T", stp.orderType == "STP" and stp.action == "SELL" and stp.auxPrice == 95.0 and stp.parentId == parent.orderId and stp.ocaType == 1 and stp.transmit is True)
chk("same OCA group", tgt.ocaGroup == stp.ocaGroup and stp.ocaGroup.startswith("OCA_"))

ps, ks = ex.build_bracket("SELL", 5, 50.0, 55.0, 40.0, "DU1", lambda: next(ids))
chk("SELL parent LMT SELL @50", ps.action == "SELL" and ps.lmtPrice == 50.0)
chk("SELL exit BUY target @40 / stop @55", ks[0].action == "BUY" and ks[0].lmtPrice == 40.0 and ks[1].action == "BUY" and ks[1].auxPrice == 55.0)

pn, kn = ex.build_bracket("BUY", 1, 10.0, 9.0, None, "DU1", lambda: next(ids))
chk("no target -> only stop, transmit T", len(kn) == 1 and kn[0].orderType == "STP" and kn[0].transmit is True)

# with a time stop: target + stop + TIME(MKT goodAfterTime), time leg transmits last
pt, kt = ex.build_bracket("BUY", 10, 100.0, 95.0, 120.0, "DU1", lambda: next(ids), "20260715 15:59:00")
chk("time stop -> 3 children", len(kt) == 3)
ttg, sttp, tmk = kt
chk("TIME leg MKT SELL @goodAfterTime, OCA, transmit T", tmk.orderType == "MKT" and tmk.action == "SELL" and tmk.totalQuantity == 10 and tmk.goodAfterTime == "20260715 15:59:00" and tmk.outsideRth is True and tmk.parentId == pt.orderId and tmk.ocaType == 1 and tmk.transmit is True)
chk("with time: stop transmit F (no longer last)", sttp.transmit is False)
chk("with time: one OCA across target/stop/time", ttg.ocaGroup == sttp.ocaGroup == tmk.ocaGroup)

# entry expiry: parent becomes GTD to the given datetime (DAY default tested above)
pg, kg = ex.build_bracket("BUY", 5, 100.0, 95.0, None, "DU1", lambda: next(ids), None, "20260715 16:00:00")
chk("entry expiry -> parent GTD + goodTillDate", pg.tif == "GTD" and pg.goodTillDate == "20260715 16:00:00")

# ---- main() gates (mocked IB) ----
class FakeClient:
    def __init__(s): s._i = 1000
    def getReqId(s): s._i += 1; return s._i
class FT:
    def __init__(s, o): s.order = o; s.orderStatus = type("S", (), {"status": "PreSubmitted", "filled": 0, "avgFillPrice": 0})()
class _Ev:
    def __iadd__(s, f): return s
class FakeIB:
    PLACED = []
    def __init__(s): s.client = FakeClient(); s.errorEvent = _Ev()
    def connect(s, *a, **k): pass
    def managedAccounts(s): return ["DU999"]
    def positions(s): return []
    def qualifyContracts(s, *a):
        for c in a: c.conId = 265598          # real IB qualifies in place
        return list(a)
    def placeOrder(s, c, o): FakeIB.PLACED.append(o); return FT(o)
    def sleep(s, *a): pass
    def disconnect(s): pass
ex.IB = FakeIB
ex.Stock = lambda *a, **k: type("C", (), {"symbol": a[0] if a else "X"})()

def run(cmd, env):
    for k, v in env.items():
        setattr(ex, k, v)
    FakeIB.PLACED = []
    sys.argv = ["execute_order.py", json.dumps(cmd)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ex.main()
    return json.loads(buf.getvalue())

EB = lambda **kw: {"type": "entry_bracket", "account": "pa",
                   "payload": {"symbol": "AAPL", "action": "BUY", "quantity": 10, "entry": 100, "stop": 95, "target": 120, **kw}}
ARM = dict(LIVE_ENABLED=True, LIVE_ACCOUNTS={"pa", "primary"}, LIVE_TYPES={"entry_bracket"}, LIVE_MAX_QTY=1000.0, LIVE_MAX_NOTIONAL=1e7)

chk("disarmed -> rejected, nothing placed", (lambda r: r["state"] == "rejected" and not FakeIB.PLACED)(run(EB(), {**ARM, "LIVE_ENABLED": False})))
chk("type not armed -> rejected", (lambda r: r["state"] == "rejected" and not FakeIB.PLACED)(run({"type": "flatten", "account": "pa", "payload": {"symbol": "X"}}, ARM)))
chk("over notional -> rejected", (lambda r: r["state"] == "rejected" and "notional" in r["detail"] and not FakeIB.PLACED)(run(EB(), {**ARM, "LIVE_MAX_NOTIONAL": 500.0})))
chk("BUY stop>entry -> rejected", (lambda r: r["state"] == "rejected" and not FakeIB.PLACED)(run(EB(stop=105), ARM)))
r = run(EB(), ARM)
chk("valid armed -> executed, 3 legs", r["state"] == "executed" and len(FakeIB.PLACED) == 3)
if len(FakeIB.PLACED) == 3:
    pp, _, ss = FakeIB.PLACED
    chk("placed parent BUY @100 transmit F", pp.action == "BUY" and pp.lmtPrice == 100 and pp.transmit is False)
    chk("placed last leg (stop) transmit T", ss.transmit is True)

print("\n" + ("ALL PASS" if ok else "FAILURES"))
sys.exit(0 if ok else 1)
