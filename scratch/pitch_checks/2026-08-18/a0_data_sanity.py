import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

px = load_prices(["^MOVE", "TLT", "SPY", "^VIX", "^TNX", "IEF"])
m = px["^MOVE"]["Close"]
r = m.pct_change()
print("^MOVE n=", len(m), m.index[0].date(), m.index[-1].date())
print("zero-change days (stale repeats):", int((r == 0).sum()), "=", round(100*(r==0).mean(),2), "%")
print("gaps > 5 calendar days:", int((m.index.to_series().diff().dt.days > 5).sum()))
print("last 6:", [(str(d.date()), round(v,2)) for d, v in m.tail(6).items()])
print("1d pct today:", round(100*r.iloc[-1], 3))
print("pctile of today's 1d among all 1d:", round(100*(r < r.iloc[-1]).mean(), 1))
print("level pctile full:", round(100*(m < m.iloc[-1]).mean(),1),
      " 252d:", round(100*(m.tail(252) < m.iloc[-1]).mean(),1))
# by-era count of >=8% up days
sp = r >= 0.08
print("\n>=8% up days by year:")
print(sp.groupby(sp.index.year).sum().to_string())
# TLT 52w low distance
t = px["TLT"]["Close"]
d52 = t / t.rolling(252).min() - 1.0
print("\nTLT dist above 52w low today: %.4f%%" % (100*d52.iloc[-1]))
