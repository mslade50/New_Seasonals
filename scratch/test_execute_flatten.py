"""Unit test for safe flatten + cancel with a mocked broker. Verifies cancel-then-
close, the FAIL-SAFE abort (no close if cancels don't confirm), and cancel by
perm_id / symbol. No connection, no real order."""
import contextlib
import io
import json
import sys

sys.path.insert(0, r"C:\Users\McKinley Slade\OneDrive\trading_ibkr")
import execute_order as ex

ok = True
def chk(n, c):
    global ok; ok = ok and bool(c); print(("OK   " if c else "FAIL ") + n)

class FClient:
    def __init__(s): s._i = 1000
    def getReqId(s): s._i += 1; return s._i
class FOrder:
    def __init__(s, oid, perm, action="SELL", cid=98): s.orderId = oid; s.permId = perm; s.action = action; s.clientId = cid
class FStatus:
    def __init__(s, st="Submitted"): s.status = st; s.filled = 0; s.avgFillPrice = 0.0
class FC:
    def __init__(s, sym): s.symbol = sym; s.secType = "STK"; s.conId = 0; s.exchange = ""
class FTrade:
    def __init__(s, sym, oid, perm, action="SELL"): s.order = FOrder(oid, perm, action); s.contract = FC(sym); s.orderStatus = FStatus()
class FPos:
    def __init__(s, sym, qty): s.contract = FC(sym); s.position = qty
class _Ev:
    def __iadd__(s, f): return s
class FIB:
    OPEN = []; POS = []; PLACED = []; CANCEL_WORKS = True
    def __init__(s): s.client = FClient(); s.errorEvent = _Ev()
    def connect(s, *a, **k): pass
    def managedAccounts(s): return ["DU1"]
    def positions(s): return FIB.POS
    def reqAllOpenOrders(s): pass
    def openTrades(s): return FIB.OPEN
    def qualifyContracts(s, *a): pass
    def cancelOrder(s, o):
        if FIB.CANCEL_WORKS:
            for t in FIB.OPEN:
                if t.order is o:
                    t.orderStatus.status = "Cancelled"
    def placeOrder(s, c, o):
        FIB.PLACED.append(o)
        # a mock market close FILLS: _do_flatten now polls for Filled (2026-07-01)
        # instead of reporting a blind "executed" after 2s
        tr = FTrade(getattr(c, "symbol", "X"), 0, 0, getattr(o, "action", "")); tr.order = o; tr.orderStatus = FStatus("Filled"); return tr
    def sleep(s, *a): pass
    def disconnect(s): pass
ex.IB = FIB

ARM = dict(LIVE_ENABLED=True, LIVE_ACCOUNTS={"pa"}, LIVE_TYPES={"flatten", "cancel"}, LIVE_MAX_QTY=1e6, LIVE_MAX_NOTIONAL=1e9)

def run(cmd, pos=(), open_orders=(), cancel_works=True):
    for k, v in ARM.items():
        setattr(ex, k, v)
    FIB.POS = list(pos); FIB.OPEN = list(open_orders); FIB.PLACED = []; FIB.CANCEL_WORKS = cancel_works
    sys.argv = ["execute_order.py", json.dumps(cmd)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ex.main()
    return json.loads(buf.getvalue())

FL = lambda **kw: {"type": "flatten", "account": "pa", "payload": {"symbol": "AAPL", "fraction": 1.0, **kw}}

r = run(FL(), pos=[FPos("AAPL", 100)], open_orders=[])
chk("flatten no working orders -> executed, 1 placed, cancelled 0", r["state"] == "executed" and len(FIB.PLACED) == 1 and r["fill"]["cancelled"] == 0)

r = run(FL(), pos=[FPos("AAPL", 100)], open_orders=[FTrade("AAPL", 11, 111), FTrade("AAPL", 12, 112)], cancel_works=True)
chk("flatten cancels 2 then closes", r["state"] == "executed" and r["fill"]["cancelled"] == 2 and len(FIB.PLACED) == 1)

r = run(FL(), pos=[FPos("AAPL", 100)], open_orders=[FTrade("AAPL", 11, 111), FTrade("AAPL", 12, 112)], cancel_works=False)
chk("flatten cancel-FAIL -> ABORTED, nothing placed", r["state"] == "rejected" and "ABORT" in r["detail"].upper() and len(FIB.PLACED) == 0)

r = run(FL(), pos=[], open_orders=[])
chk("flatten no position -> rejected", r["state"] == "rejected" and len(FIB.PLACED) == 0)

r = run({"type": "cancel", "account": "pa", "payload": {"perm_id": 111}}, open_orders=[FTrade("AAPL", 11, 111), FTrade("MSFT", 12, 222)])
chk("cancel by perm_id -> 1", r["state"] == "executed" and "1 order" in r["detail"])

r = run({"type": "cancel", "account": "pa", "payload": {"symbol": "AAPL"}}, open_orders=[FTrade("AAPL", 11, 111), FTrade("AAPL", 13, 113), FTrade("MSFT", 12, 222)])
chk("cancel by symbol -> 2", r["state"] == "executed" and "2 order" in r["detail"])

r = run({"type": "cancel", "account": "pa", "payload": {"perm_id": 999}}, open_orders=[FTrade("AAPL", 11, 111)])
chk("cancel not found -> rejected", r["state"] == "rejected")

print("\n" + ("ALL PASS" if ok else "FAILURES"))
sys.exit(0 if ok else 1)
