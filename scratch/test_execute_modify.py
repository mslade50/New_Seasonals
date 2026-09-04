"""Unit test for the modify command with a mocked broker (mirrors
test_execute_flatten.py). Verifies field routing (qty/lmt/stop), type
compatibility rejects, qty-increase caps, owner-client re-place, and the
agent-side validate/preview. No connection, no real order."""
import contextlib
import io
import json
import sys

sys.path.insert(0, r"C:\Users\McKinley Slade\OneDrive\trading_ibkr")
import execute_order as ex

ok = True
def chk(n, c):
    global ok; ok = ok and bool(c); print(("OK   " if c else "FAIL ") + n)

UNSET = 1.7976931348623157e+308

class FClient:
    def __init__(s): s._i = 1000
    def getReqId(s): s._i += 1; return s._i
class FOrder:
    def __init__(s, oid, perm, action="SELL", cid=98, typ="LMT", qty=100, lmt=UNSET, aux=UNSET):
        s.orderId = oid; s.permId = perm; s.action = action; s.clientId = cid
        s.orderType = typ; s.totalQuantity = qty; s.lmtPrice = lmt; s.auxPrice = aux
        s.transmit = False
class FStatus:
    def __init__(s, st="Submitted"): s.status = st; s.filled = 0; s.avgFillPrice = 0.0
class FC:
    def __init__(s, sym, sec="STK"): s.symbol = sym; s.secType = sec
class FTrade:
    def __init__(s, sym, order): s.order = order; s.contract = FC(sym); s.orderStatus = FStatus()
class _Ev:
    def __iadd__(s, f): return s
class FIB:
    OPEN = []; PLACED = []
    def __init__(s): s.client = FClient(); s.errorEvent = _Ev()
    def connect(s, *a, **k): pass
    def managedAccounts(s): return ["DU1"]
    def positions(s): return []
    def reqAllOpenOrders(s): pass
    def openTrades(s): return FIB.OPEN
    def qualifyContracts(s, *a): pass
    def placeOrder(s, c, o): FIB.PLACED.append(o); return FTrade(c.symbol, o)
    def sleep(s, *a): pass
    def disconnect(s): pass
ex.IB = FIB

ARM = dict(LIVE_ENABLED=True, LIVE_ACCOUNTS={"pa"}, LIVE_TYPES={"modify"},
           LIVE_MAX_QTY=500, LIVE_MAX_NOTIONAL=100_000, LIVE_MAX_FUT_CONTRACTS=3)

def run(payload, open_orders=(), types=None):
    for k, v in ARM.items():
        setattr(ex, k, v)
    if types is not None:
        ex.LIVE_TYPES = types
    FIB.OPEN = list(open_orders); FIB.PLACED = []
    sys.argv = ["execute_order.py",
                json.dumps({"type": "modify", "account": "pa", "payload": payload})]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ex.main()
    return json.loads(buf.getvalue())

def lmt_order(**kw):
    return FTrade("OLV1", FOrder(11, 111, typ="LMT", qty=100, lmt=33.50, **kw))
def stp_order(**kw):
    return FTrade("OLV1", FOrder(12, 112, typ="STP", qty=100, aux=31.00, **kw))

# 1. qty change on a LMT via perm_id (owner != main cid -> owner-client path)
r = run({"perm_id": 111, "new_qty": 80}, [lmt_order()])
o = FIB.PLACED[0] if FIB.PLACED else None
chk("qty 100->80 executed via owner client", r["state"] == "executed" and o
    and o.totalQuantity == 80 and o.lmtPrice == 33.50 and o.transmit is True)

# 2. limit reprice only — qty untouched
r = run({"perm_id": 111, "new_limit": 33.10}, [lmt_order()])
o = FIB.PLACED[0] if FIB.PLACED else None
chk("lmt 33.50->33.10, qty preserved", r["state"] == "executed" and o
    and o.lmtPrice == 33.10 and o.totalQuantity == 100)

# 3. stop-trigger change on a STP
r = run({"perm_id": 112, "new_stop": 30.50}, [stp_order()])
o = FIB.PLACED[0] if FIB.PLACED else None
chk("stop 31.00->30.50 on STP", r["state"] == "executed" and o and o.auxPrice == 30.50)

# 4. wrong-field rejects
r = run({"perm_id": 111, "new_stop": 30.0}, [lmt_order()])
chk("new_stop on LMT rejected", r["state"] == "rejected" and not FIB.PLACED)
r = run({"perm_id": 112, "new_limit": 30.0}, [stp_order()])
chk("new_limit on STP rejected", r["state"] == "rejected" and not FIB.PLACED)

# 5. nothing to change / bad values / missing id / not found
r = run({"perm_id": 111}, [lmt_order()])
chk("no fields -> rejected", r["state"] == "rejected")
r = run({"perm_id": 111, "new_qty": -5}, [lmt_order()])
chk("negative qty -> rejected", r["state"] == "rejected")
r = run({"perm_id": 111, "new_qty": 10.5}, [lmt_order()])
chk("fractional qty -> rejected", r["state"] == "rejected")
r = run({"new_qty": 80}, [lmt_order()])
chk("no id -> rejected", r["state"] == "rejected")
r = run({"perm_id": 999, "new_qty": 80}, [lmt_order()])
chk("unknown perm_id -> rejected", r["state"] == "rejected")

# 6. qty-increase caps (decrease always passes; increase re-checked)
r = run({"perm_id": 111, "new_qty": 600}, [lmt_order()])
chk("qty 100->600 > LIVE_MAX_QTY rejected", r["state"] == "rejected" and not FIB.PLACED)
r = run({"perm_id": 111, "new_qty": 400}, [lmt_order()])  # 400*33.50 = 13,400 < cap
chk("qty 100->400 within caps executed", r["state"] == "executed")
r = run({"perm_id": 111, "new_qty": 400, "new_limit": 300.0}, [lmt_order()])  # 120k > 100k
chk("qty up + notional > cap rejected", r["state"] == "rejected")

# 7. type not armed
r = run({"perm_id": 111, "new_qty": 80}, [lmt_order()], types={"flatten"})
chk("modify not in LIVE_TYPES -> rejected", r["state"] == "rejected")

# 8. main-cid owner path (pa main cid = 147)
r = run({"perm_id": 111, "new_qty": 90}, [lmt_order(cid=147)])
o = FIB.PLACED[0] if FIB.PLACED else None
chk("owner == main cid path", r["state"] == "executed" and o and o.totalQuantity == 90)

# --- agent-side validate/preview ---
import exec_agent as ag
ag._BOOK["book"] = {"accounts": [{"key": "pa", "nlv": 50_000, "positions": [],
    "orders": [{"symbol": "OLV1", "sec_type": "STK", "action": "SELL", "qty": 100,
                "order_type": "LMT", "lmt": 33.5, "aux": None, "tif": "GTC",
                "status": "Submitted", "parent_id": 0, "order_id": 11, "perm_id": 111}]}]}
CMD = lambda **kw: {"type": "modify", "account": "pa", "payload": {"perm_id": 111, **kw}}
okv, why = ag._validate(CMD(new_qty=80))
chk("agent validate ok", okv and not why)
okv, why = ag._validate(CMD(new_stop=30.0))
chk("agent validate rejects stop-on-LMT", not okv and any("stop" in r for r in why))
okv, why = ag._validate({"type": "modify", "account": "pa", "payload": {"perm_id": 999, "new_qty": 5}})
chk("agent validate rejects unknown order", not okv)
okv, why = ag._validate(CMD())
chk("agent validate rejects empty change", not okv)
pv = ag._preview(CMD(new_qty=80, new_limit=33.1))
chk("agent preview shows old->new", "MODIFY" in pv["legs"][0] and any("33.5" in l for l in pv["legs"]))

print("\nALL PASS" if ok else "\nFAILURES ABOVE")
sys.exit(0 if ok else 1)
