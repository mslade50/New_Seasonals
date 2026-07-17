"""Overall vs in-market-only Sharpe for the near-high + dial<T strategy."""
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
tbill = (irx / 100.0) / 252.0

close = spy_df["Close"]
o2o = (spy_df["Open"].shift(-1) / spy_df["Open"] - 1).fillna(0.0)
near5 = (close / close.rolling(252, min_periods=60).max() - 1) >= -0.05
SLIP = 2 / 1e4

print(f"{'T':>4} {'in_mkt%':>8} {'overall_Sharpe':>15} {'inmkt_Sharpe':>13} "
      f"{'sqrt(f)*inmkt':>14}")
for t in (27, 40):
    sig = near5 & (ma10 < t)
    pos = sig.shift(1)
    pos = pos.astype(object).where(pd.notna(pos), False).astype(bool)
    r = pd.Series(np.where(pos, o2o, tbill), index=sig.index)
    switch = (pos != pos.shift(1))
    switch.iloc[0] = False
    r = r - switch.astype(float) * SLIP
    overall = (r.mean() - tbill.mean()) / r.std() * np.sqrt(252)
    ex_in = (r - tbill)[pos]
    inmkt = ex_in.mean() / ex_in.std() * np.sqrt(252)
    f = pos.mean()
    print(f"{t:>4} {f * 100:>7.0f}% {overall:>15.2f} {inmkt:>13.2f} "
          f"{np.sqrt(f) * inmkt:>14.2f}")

spy_ex = (o2o - tbill)
print(f"\nSPY all days, o2o basis: Sharpe "
      f"{spy_ex.mean() / spy_ex.std() * np.sqrt(252):.2f}")
