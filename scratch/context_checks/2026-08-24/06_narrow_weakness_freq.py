"""How often does the drill-05 state print, by year? Is 2026 unusual?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pandas as pd
from pitch_lab import load_prices, declusters  # noqa

ASOF = pd.Timestamp("2026-08-24")
px = load_prices(["QQQ", "^NYA"])
pan = pd.concat({k: px[k]["Close"] for k in px}, axis=1).dropna().loc[:ASOF]
q, n = pan["QQQ"], pan["^NYA"]
rank_q5 = q.pct_change(5).rolling(252).apply(lambda w: (w[:-1] < w[-1]).mean() * 100, raw=True)
near_hi = n >= n.rolling(252).max() * 0.995
dts = pan.index[((rank_q5 <= 10) & near_hi).fillna(False)]
dc = declusters(dts, 10, pan.index)
by_y = pd.Series(dc.year).value_counts().sort_index()
print("declustered episodes by year:")
print(by_y.to_string())
print(f"\ntotal {len(dc)} over {pan.index[0].year}-2026")
print(f"2026 so far: {by_y.get(2026, 0)}  dates {[str(d.date()) for d in dc if d.year == 2026]}")
yrs_with = (by_y > 0).sum()
print(f"years with at least one: {yrs_with} of {pan.index[-1].year - pan.index[0].year + 1}")
print(f"max in any single year: {by_y.max()} ({list(by_y[by_y == by_y.max()].index)})")
