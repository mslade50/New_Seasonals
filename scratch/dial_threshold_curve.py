"""Threshold sensitivity: near-high + dial<T for T in a fine grid.
Realistic accounting: next-open execution, 2 bps/side. If Sharpe is smooth
in T, the threshold isn't load-bearing; a spike would mean overfit."""
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
spy_df = (mp[mp["ticker"] == "SPY"].set_index("date")[["Open", "Close"]]
          .sort_index().reindex(s63.index).ffill())
irx = (mp[mp["ticker"] == "^IRX"].set_index("date")["Close"]
       .sort_index().reindex(s63.index).ffill().fillna(4.0))
tbill_daily = (irx / 100.0) / 252.0

close = spy_df["Close"]
o2o = (spy_df["Open"].shift(-1) / spy_df["Open"] - 1).fillna(0.0)
near5 = (close / close.rolling(252, min_periods=60).max() - 1) >= -0.05
YEARS = (s63.index[-1] - s63.index[0]).days / 365.25
SLIP = 2 / 1e4

rows = []
for t in [15, 18, 20, 22, 25, 27, 30, 32, 35, 38, 40, 42, 45, 50, 60]:
    sig = near5 & (ma10 < t)
    pos = sig.shift(1)
    pos = pos.astype(object).where(pd.notna(pos), False).astype(bool)
    r = pd.Series(np.where(pos, o2o, tbill_daily), index=sig.index)
    switch = (pos != pos.shift(1))
    switch.iloc[0] = False
    r = r - switch.astype(float) * SLIP
    eq = (1 + r).cumprod()
    ex_in = (r - tbill_daily)[pos]
    inmkt_sharpe = (ex_in.mean() / ex_in.std() * np.sqrt(252)
                    if len(ex_in) > 20 and ex_in.std() > 0 else np.nan)
    rows.append({
        "T": t,
        "CAGR%": round((eq.iloc[-1] ** (1 / YEARS) - 1) * 100, 2),
        "Sharpe": round((r.mean() - tbill_daily.mean()) / r.std() * np.sqrt(252), 2),
        "inmkt_Sharpe": round(inmkt_sharpe, 2),
        "maxDD%": round((eq / eq.cummax() - 1).min() * 100, 1),
        "in_mkt%": round(pos.mean() * 100),
        "switches/yr": round(switch.sum() / YEARS, 1),
    })
print(pd.DataFrame(rows).set_index("T").to_string())
print("\nreference: SPY all days o2o in-market Sharpe = "
      f"{((o2o - tbill_daily).mean() / (o2o - tbill_daily).std() * np.sqrt(252)):.2f}")
