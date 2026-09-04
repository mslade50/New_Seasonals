"""Follow-up: is the late-August gap in ZC=F / KC=F a contract roll signature?

Test 1: for each ticker, the distribution of overnight gaps by (month, day-bucket).
        A roll shows up as one calendar slot carrying gaps many times the median.
Test 2: the same calendar slot in every prior year - if Aug 20-31 always carries a
        one-day jump of the same sign, the continuous series is rolling, not moving.
Test 3: sanity on today's raw bars (CT=F printed Open=0).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices  # noqa

ASOF = pd.Timestamp("2026-08-24")
px = load_prices(["ZC=F", "KC=F", "CT=F", "ZW=F", "ZS=F", "SB=F"])

print("=== raw bars 2026-08-17 .. 2026-08-24 ===")
for t in ["ZC=F", "KC=F", "CT=F", "ZW=F"]:
    df = px[t].loc["2026-08-17":"2026-08-24", ["Open", "High", "Low", "Close"]]
    print(f"-- {t}")
    print(df.round(2).to_string())

print()
print("=== gap (Open/prevClose-1) by month, mean and count of |gap|>3% ===")
for t in ["ZC=F", "KC=F", "CT=F", "ZW=F", "SB=F"]:
    df = px[t]
    gap = (df["Open"] / df["Close"].shift(1) - 1).replace([np.inf, -np.inf], np.nan)
    gap = gap[(gap.abs() < 0.5) & gap.notna()]
    big = gap[gap.abs() > 0.03]
    by_m = big.groupby(big.index.month).size()
    print(f"{t:<7} n_gaps={len(gap)}  |gap|>3% n={len(big)}  by month: {dict(by_m)}")

print()
print("=== late-August window (Aug 18 - Sep 2), largest single gap per year ===")
for t in ["ZC=F", "KC=F"]:
    df = px[t]
    gap = (df["Open"] / df["Close"].shift(1) - 1).replace([np.inf, -np.inf], np.nan)
    sel = gap[(gap.index.month == 8) & (gap.index.day >= 18) |
              ((gap.index.month == 9) & (gap.index.day <= 2))]
    sel = sel.dropna()
    print(f"-- {t}")
    rows = []
    for y, g in sel.groupby(sel.index.year):
        i = g.abs().idxmax()
        rows.append((y, i.date(), round(g.loc[i] * 100, 2)))
    for r in rows:
        print(f"   {r[0]}  {r[1]}  biggest gap {r[2]:>7}%")
    vals = np.array([r[2] for r in rows])
    print(f"   mean {vals.mean():.2f}%  median {np.median(vals):.2f}%  "
          f"same-sign-as-2026 {int((np.sign(vals) == np.sign(vals[-1])).sum())}/{len(vals)}")
