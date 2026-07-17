"""SPY hold-when-calm strategy: long SPY when (within 5% of 52w high) AND
(63d dial 10d-MA < threshold), else T-bills. Signals lagged 1 day.

Baselines: buy-and-hold, and the trend-only rule (near-high, no dial) so the
dial's MARGINAL contribution is visible. Thresholds 15/27/40 bracket the
jurisdiction-table break (~27). Caveats: pre-2026-07-02 dial is recompute
vintage; signal code is today's (lookahead); one decade; threshold chosen
after inspecting the same data (in-sample).
"""
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

frag = pd.read_parquet(os.path.join(_ROOT, "data", "rd2_fragility.parquet"))
s63 = frag["63d"].dropna().sort_index()
ma10 = s63.rolling(10, min_periods=1).mean()

mp = pd.read_parquet(os.path.join(_ROOT, "data", "master_prices.parquet"),
                     filters=[("ticker", "in", ["SPY", "^IRX"])])
mp["date"] = pd.to_datetime(mp["date"])


def ser(tkr):
    return (mp[mp["ticker"] == tkr].set_index("date")["Close"]
            .sort_index().reindex(s63.index).ffill())


spy = ser("SPY")
tbill_daily = (ser("^IRX").fillna(4.0) / 100.0) / 252.0

ret = spy.pct_change().fillna(0.0)
near5 = (spy / spy.rolling(252, min_periods=60).max() - 1) >= -0.05

variants = {
    "buy_and_hold": pd.Series(True, index=spy.index),
    "near5_only": near5,
    "near5_dial<15": near5 & (ma10 < 15),
    "near5_dial<27": near5 & (ma10 < 27),
    "near5_dial<40": near5 & (ma10 < 40),
}


def stats(sig: pd.Series, label: str) -> dict:
    pos = sig.shift(1).fillna(False)          # trade next day on today's state
    r = np.where(pos, ret, tbill_daily)
    r = pd.Series(r, index=ret.index)
    eq = (1 + r).cumprod()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    dd = (eq / eq.cummax() - 1).min()
    sharpe = (r.mean() - tbill_daily.mean()) / r.std() * np.sqrt(252) if r.std() > 0 else 0
    switches = int((pos != pos.shift(1)).sum())
    return {
        "strategy": label,
        "CAGR%": round(cagr * 100, 1),
        "vol%": round(vol * 100, 1),
        "Sharpe": round(sharpe, 2),
        "maxDD%": round(dd * 100, 1),
        "in_mkt%": round(pos.mean() * 100),
        "switches": switches,
        "_eq": eq, "_r": r,
    }


pd.set_option("display.width", 150)
rows = [stats(sig, name) for name, sig in variants.items()]
print(pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')}
                    for r in rows]).set_index("strategy").to_string())

print("\nper-year returns (%):")
tbl = {}
for r in rows:
    tbl[r["strategy"]] = (r["_r"].groupby(r["_r"].index.year)
                          .apply(lambda x: ((1 + x).prod() - 1) * 100).round(1))
print(pd.DataFrame(tbl).to_string())

print("\nworst 5 drawdown troughs, near5_dial<27:")
r27 = next(r for r in rows if r["strategy"] == "near5_dial<27")
eq = r27["_eq"]
ddser = eq / eq.cummax() - 1
print(ddser.nsmallest(5).round(3).to_string())
