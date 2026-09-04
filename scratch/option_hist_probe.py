"""What historical option data can we pull from IBKR? Probes reqHistoricalData on
a real near-ATM AAPL call: MIDPOINT bars + implied-vol history. Read-only."""
import datetime

from ib_insync import IB, Option, Stock

ib = IB()
ib.connect("127.0.0.1", 7496, clientId=134, timeout=8, readonly=True)
try:
    ib.reqMarketDataType(3)
    stk = Stock("AAPL", "SMART", "USD"); ib.qualifyContracts(stk)
    [u] = ib.reqTickers(stk); spot = u.marketPrice() or u.close
    params = ib.reqSecDefOptParams(stk.symbol, "", stk.secType, stk.conId)
    std = [p for p in params if p.tradingClass == "AAPL"] or params
    exps = sorted(set().union(*[set(p.expirations) for p in std]))
    strikes = sorted(set().union(*[set(p.strikes) for p in std]))
    today = datetime.date.today()
    # a ~monthly expiry (more history than a brand-new weekly) ~30-45 DTE
    exp = next((e for e in exps if 25 <= (datetime.date(int(e[:4]), int(e[4:6]), int(e[6:8])) - today).days <= 60), exps[-1])
    strike = min(strikes, key=lambda s: abs(s - spot))
    opt = Option("AAPL", exp, strike, "C", "SMART", tradingClass="AAPL")
    ib.qualifyContracts(opt)
    print(f"AAPL {strike}C {exp}  (spot {spot})")

    for what in ("MIDPOINT", "OPTION_IMPLIED_VOLATILITY", "TRADES"):
        try:
            bars = ib.reqHistoricalData(opt, endDateTime="", durationStr="60 D",
                                        barSizeSetting="1 day", whatToShow=what,
                                        useRTH=True, formatDate=1)
            if bars:
                print(f"  {what}: {len(bars)} daily bars  {bars[0].date}..{bars[-1].date}  "
                      f"first.close={bars[0].close}  last.close={bars[-1].close}")
            else:
                print(f"  {what}: no bars")
        except Exception as e:
            print(f"  {what}: error {type(e).__name__}: {e}")
finally:
    ib.disconnect()
