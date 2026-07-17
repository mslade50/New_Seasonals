"""Execution realism for the near-high + dial<40 SPY strategy.

Timing: the dial exists after the 21:15 UTC risk report, so the earliest
real execution is the NEXT OPEN (MOO/OPG — the trend-sleeve pattern).
Accounting here: open-to-open returns, position(T) = signal(close T-1),
executed at open(T). Compare vs the optimistic same-close assumption and a
worst-case full-day delay (next close). Slippage charged per SIDE on every
switch day. Also: switch counts per year per variant.
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
spy_df = (mp[mp["ticker"] == "SPY"].set_index("date")[["Open", "Close"]]
          .sort_index().reindex(s63.index).ffill())
irx = (mp[mp["ticker"] == "^IRX"].set_index("date")["Close"]
       .sort_index().reindex(s63.index).ffill().fillna(4.0))
tbill_daily = (irx / 100.0) / 252.0

close = spy_df["Close"]
open_ = spy_df["Open"]
c2c = close.pct_change().fillna(0.0)
o2o = (open_.shift(-1) / open_ - 1).fillna(0.0)   # open(T) -> open(T+1)

near5 = (close / close.rolling(252, min_periods=60).max() - 1) >= -0.05
signals = {
    "dial<27": near5 & (ma10 < 27),
    "dial<40": near5 & (ma10 < 40),
}

YEARS = (s63.index[-1] - s63.index[0]).days / 365.25


def run(sig, mode, slip_bps):
    """mode: close_exec (optimistic), open_exec (realistic), nextclose (worst)."""
    if mode == "close_exec":
        pos = sig.shift(1).fillna(False).astype(bool)
        mkt = c2c
    elif mode == "open_exec":
        pos = sig.shift(1).fillna(False).astype(bool)
        mkt = o2o
    else:  # nextclose
        pos = sig.shift(2).fillna(False).astype(bool)
        mkt = c2c
    r = pd.Series(np.where(pos, mkt, tbill_daily), index=sig.index)
    switch = (pos != pos.shift(1)).fillna(False)
    r = r - switch.astype(float) * (slip_bps / 1e4)
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (1 / YEARS) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() - tbill_daily.mean()) / r.std() * np.sqrt(252)
    dd = (eq / eq.cummax() - 1).min()
    return {"CAGR%": round(cagr * 100, 2), "Sharpe": round(sharpe, 2),
            "maxDD%": round(dd * 100, 1),
            "switches/yr": round(switch.sum() / YEARS, 1)}


pd.set_option("display.width", 150)
for name, sig in signals.items():
    print(f"\n== {name} ==")
    rows = []
    for mode in ("close_exec", "open_exec", "nextclose"):
        for slip in (0, 2, 5, 10):
            rows.append({"mode": mode, "slip_bps/side": slip,
                         **run(sig, mode, slip)})
    print(pd.DataFrame(rows).set_index(["mode", "slip_bps/side"]).to_string())

# how clustered are switches? distribution of holding/flat spell lengths
sig = signals["dial<40"]
pos = sig.shift(1).fillna(False).astype(bool)
grp = (pos != pos.shift(1)).cumsum()
spells = pos.groupby(grp).agg(["first", "size"])
print("\ndial<40 spell lengths (td): in-market median "
      f"{spells[spells['first']]['size'].median():.0f} "
      f"(min {spells[spells['first']]['size'].min()}), "
      f"flat median {spells[~spells['first']]['size'].median():.0f} "
      f"(min {spells[~spells['first']]['size'].min()})")
short = (spells['size'] <= 3).sum()
print(f"spells lasting <=3 td: {short} of {len(spells)} "
      f"(whipsaw fraction {short / len(spells):.0%})")
