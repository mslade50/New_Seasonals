"""Read-only probe: can we pull option chains + quotes + greeks from IBKR?
Checks SPY: spot, nearest expiries, and bid/ask/delta/IV for a few near-ATM calls.
Tells us whether live option market data (mid + greeks) is available on this account."""
from ib_insync import IB, Stock, Option

ib = IB()
try:
    ib.connect("127.0.0.1", 7496, clientId=131, timeout=8, readonly=True)
except Exception as e:
    print("TWS not reachable:", e); raise SystemExit

try:
    spy = Stock("SPY", "SMART", "USD")
    ib.qualifyContracts(spy)
    ib.reqMarketDataType(3)  # 3 = delayed if no live sub, so we still see something
    [u] = ib.reqTickers(spy)
    spot = u.marketPrice() or u.close
    print(f"SPY spot ~ {spot}")

    params = ib.reqSecDefOptParams(spy.symbol, "", spy.secType, spy.conId)
    smart = next((p for p in params if p.exchange == "SMART"), params[0] if params else None)
    if not smart:
        print("no option params"); raise SystemExit
    expiries = sorted(smart.expirations)[:4]
    strikes = sorted(smart.strikes)
    print(f"expiries (nearest 4): {expiries}")
    print(f"strike count: {len(strikes)}  range {strikes[0]}..{strikes[-1]}")

    exp = expiries[0]
    near = [s for s in strikes if spot and abs(s - spot) <= spot * 0.03][:6] or strikes[:6]
    opts = [Option("SPY", exp, s, "C", "SMART", tradingClass="SPY") for s in near]
    ib.qualifyContracts(*opts)
    tks = ib.reqTickers(*opts)
    print(f"\n{exp} calls:")
    print("  strike   bid    ask    mid    delta   IV")
    for tk in tks:
        g = tk.modelGreeks
        mid = (tk.bid + tk.ask) / 2 if (tk.bid and tk.ask and tk.bid > 0 and tk.ask > 0) else None
        print(f"  {tk.contract.strike:>6}  {tk.bid!s:>5}  {tk.ask!s:>5}  "
              f"{('%.2f' % mid) if mid else '   -':>5}  "
              f"{('%.3f' % g.delta) if g and g.delta is not None else '   -':>6}  "
              f"{('%.3f' % g.impliedVol) if g and g.impliedVol is not None else '  -'}")
finally:
    ib.disconnect()
