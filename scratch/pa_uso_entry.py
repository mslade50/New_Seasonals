"""Place the 75-share USO version on the PA (small) account gateway, reusing
sznl_entry's exact bracket construction (same entry 104.80 / stop 103.29 /
target 123.21 / 3-day window / 7-9 time exit), just QTY=75 on gateway :4001."""
import sys
sys.path.insert(0, r"C:\Users\McKinley Slade\OneDrive\trading_ibkr")
import sznl_entry as se
from ib_insync import IB

PA_HOST, PA_PORT, PA_CLIENT = "127.0.0.1", 4001, 101
se.QUANTITY = 75  # override (sznl_entry config is set to the primary's 692)

se.validate_config()
parent_gtd = se.fmt_gtd(se.PARENT_GTD, se.TIMEZONE)
bracket_gtd = se.fmt_gtd(se.BRACKET_GTD, se.TIMEZONE) if se.BRACKET_GTD else None
time_stop = se.fmt_gtd(se.TIME_STOP, se.TIMEZONE) if se.TIME_STOP else None

print(f"PA bracket: USO {se.ACTION} {se.QUANTITY} @ {se.ENTRY_PRICE}  "
      f"stop {se.STOP_PRICE}  target {se.TARGET_PRICE}")
print(f"  parent GTD {parent_gtd} | bracket GTD {bracket_gtd} | time stop {time_stop}")

ib = IB()
ib.connect(PA_HOST, PA_PORT, clientId=PA_CLIENT, timeout=15)
try:
    print("PA managedAccounts:", ib.managedAccounts())
    contract = se.build_contract()
    ib.qualifyContracts(contract)
    print("qualified:", contract.localSymbol, contract.primaryExchange)

    parent, target, stop, time_close = se.build_orders(
        ib, parent_gtd, bracket_gtd, time_stop)
    chain = [("PARENT", parent)]
    if target is not None:
        chain.append(("TARGET", target))
    chain.append(("STOP", stop))
    if time_close is not None:
        chain.append(("TIME_CLOSE", time_close))

    trades = []
    for label, o in chain:
        trades.append(ib.placeOrder(contract, o))
    ib.sleep(2)

    print("\nSubmitted PA orders:")
    for t in trades:
        o = t.order
        px = o.lmtPrice if getattr(o, "lmtPrice", 0) else getattr(o, "auxPrice", 0)
        print(f"  id={o.orderId:>5} {o.action:>4} {int(o.totalQuantity):>4} "
              f"{type(o).__name__:<11} px={px} tif={o.tif} status={t.orderStatus.status}")
finally:
    ib.disconnect()
