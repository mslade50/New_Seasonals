"""LIVE single-order probe: validate that eq_order_entry's GTD parent change is
accepted by IBKR and the goodTillDate sticks.

SAFETY:
  * ONE order, 1 share, BUY LIMIT at $1.00 on a high-priced name (AAPL) -> can
    never fill (rests far below market).
  * separate clientId (88) so it can't collide with prod's eq_order_entry (99).
  * placed, read back from the broker, then CANCELLED in a finally block so it
    is always removed even if a check fails.
  * does NOT read or submit the prod staged_orders.csv; does NOT touch any other
    open order in the account.

It replicates eq_order_entry's parent-construction block VERBATIM (including the
2026-06-24 Entry_Expire_Time edit) and uses the real format_ib_time + host/port.
"""
import os
import sys

TRADING_IBKR = r"C:\Users\McKinley Slade\OneDrive\trading_ibkr"
sys.path.insert(0, TRADING_IBKR)

from ib_insync import IB, Stock, Order
from eq_order_entry import format_ib_time, IB_IP, IB_PORT  # real code

PROBE_CLIENT_ID = 88
SYMBOL = "AAPL"
LIMIT_PRICE = 1.00          # far below market -> guaranteed no fill
QTY = 1

# A staged row exactly as order_staging now emits it for an OLV signal:
# Entry_Expire_Time = signal + 3 BDays (the shortened fill window). This is the
# value eq_order_entry reads from row['Entry_Expire_Time'].
row = {"Entry_Expire_Time": "2026-07-02 15:59:00",
       "Exit_Condition_Time": "2026-07-15 15:59:00"}
gat_time = format_ib_time(row["Exit_Condition_Time"])   # the time-exit (fallback)

ib = IB()
trade = None
try:
    print(f"Connecting to TWS {IB_IP}:{IB_PORT} clientId={PROBE_CLIENT_ID} ...")
    ib.connect(IB_IP, IB_PORT, clientId=PROBE_CLIENT_ID, timeout=15)
    print(f"[OK] connected. accounts={ib.managedAccounts()}")

    contract = Stock(SYMBOL, "SMART", "USD")
    ib.qualifyContracts(contract)

    # ---- VERBATIM parent block from eq_order_entry.py (incl. the edit) ----
    tif = "GTC"
    parent = Order()
    parent.action = "BUY"
    parent.totalQuantity = QTY
    parent.orderType = "LMT"
    parent.lmtPrice = LIMIT_PRICE
    parent.transmit = True   # lone order (no children) -> transmit now

    entry_expire_raw = row.get("Entry_Expire_Time", "")
    if entry_expire_raw is None:
        entry_expire_raw = ""
    entry_expire_raw = str(entry_expire_raw).strip()
    gtd_time = format_ib_time(entry_expire_raw) if entry_expire_raw else gat_time

    if tif == "GTC":
        parent.tif = "GTD"
        parent.goodTillDate = gtd_time
        print(f"      GTD Parent: expires {gtd_time}")
    else:
        parent.tif = tif
    # ----------------------------------------------------------------------

    print(f"\nPlacing: BUY {QTY} {SYMBOL} LMT ${LIMIT_PRICE} TIF=GTD GTD={gtd_time}")
    trade = ib.placeOrder(contract, parent)
    for _ in range(40):                      # up to ~4s for broker ack
        ib.sleep(0.1)
        if trade.orderStatus.status not in ("", "PendingSubmit"):
            break

    st = trade.orderStatus.status
    print(f"\n--- BROKER READBACK ---")
    print(f"  status        : {st}")
    print(f"  orderId       : {trade.order.orderId}")
    print(f"  tif           : {trade.order.tif}")
    print(f"  goodTillDate  : {trade.order.goodTillDate!r}")
    if trade.log:
        print(f"  last log      : {trade.log[-1].message}")

    expected_gtd = format_ib_time("2026-07-02 15:59:00")   # -> 20260702 15:59:00
    accepted = st in ("PreSubmitted", "Submitted")
    # broker may echo a timezone suffix; match on the YYYYMMDD date portion
    gtd_ok = trade.order.tif == "GTD" and "20260702" in str(trade.order.goodTillDate)
    no_fill = (trade.orderStatus.filled or 0) == 0
    print(f"\n  accepted (not rejected) : {accepted}")
    print(f"  GTD value correct       : {gtd_ok}  (expected date 20260702, got {trade.order.goodTillDate!r})")
    print(f"  unfilled                : {no_fill}")
    PASS = accepted and gtd_ok and no_fill
finally:
    # ALWAYS cancel + disconnect
    try:
        if trade is not None and trade.orderStatus.status not in (
                "Cancelled", "ApiCancelled", "Filled", "Inactive", ""):
            print(f"\nCancelling order {trade.order.orderId} ...")
            ib.cancelOrder(trade.order)
            for _ in range(40):
                ib.sleep(0.1)
                if trade.orderStatus.status in ("Cancelled", "ApiCancelled"):
                    break
            print(f"  cancel status : {trade.orderStatus.status}")
    finally:
        ib.disconnect()
        print("[OK] disconnected.")

print("\n" + ("LIVE GTD PROBE PASSED" if 'PASS' in dir() and PASS else "*** PROBE FAILED / INCOMPLETE ***"))
