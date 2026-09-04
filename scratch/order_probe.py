"""Read-only: dump open orders so we can design safe cross-client cancel. Shows
each order's symbol/type/side/status + clientId / orderId / permId / parentId."""
import sys

from ib_insync import IB

port = int(sys.argv[1]) if len(sys.argv) > 1 else 7496   # 7496 primary, 4001 PA
ib = IB()
ib.connect("127.0.0.1", port, clientId=136, timeout=8, readonly=True)
try:
    ib.reqAllOpenOrders()
    ib.sleep(2.0)
    trades = ib.openTrades()
    print(f"port {port}: {len(trades)} open orders")
    seen = {}
    for t in trades:
        o, c = t.order, t.contract
        seen[o.clientId] = seen.get(o.clientId, 0) + 1
        print(f"  {c.symbol:6} {o.action:4} {o.orderType:9} lmt={o.lmtPrice} aux={o.auxPrice} "
              f"st={t.orderStatus.status:11} cid={o.clientId} oid={o.orderId} perm={o.permId} "
              f"parent={o.parentId} oca={o.ocaGroup}")
    print("orders by clientId:", seen)
finally:
    ib.disconnect()
