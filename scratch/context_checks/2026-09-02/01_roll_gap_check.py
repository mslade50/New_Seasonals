"""Are tonight's commodity bars trade or continuous-contract roll?

Third night running this. The 2026-08-31 and 2026-09-01 grain/softs bars were
94-98% overnight gap and were excluded from both briefs. Tonight KC=F printed
-13.12%, ZC=F +4.03%, and ZC/ZS/ZW all closed AT 52-week highs again. Gate
every commodity verdict in tonight's cell map on this.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TKRS = ["KC=F", "ZC=F", "ZS=F", "ZW=F", "SB=F", "CT=F", "CL=F", "LE=F"]
px = load_prices(TKRS)

print(f"{'tkr':<7}{'date':<12}{'ret_1d%':>9}{'gap%':>9}{'intraday%':>11}"
      f"{'gap share':>11}{'volume':>12}{'med vol 20':>12}")
for t in TKRS:
    df = px.get(t)
    if df is None or len(df) < 25:
        print(f"{t:<7} no data")
        continue
    for k in (-2, -1):
        row = df.iloc[k]
        prev = df.iloc[k - 1]
        ret = row["Close"] / prev["Close"] - 1.0
        gap = row["Open"] / prev["Close"] - 1.0
        intr = row["Close"] / row["Open"] - 1.0
        share = abs(gap) / abs(ret) * 100 if ret else float("nan")
        vol = row.get("Volume", float("nan"))
        medv = df["Volume"].iloc[k - 20:k].median() if "Volume" in df else float("nan")
        print(f"{t:<7}{str(df.index[k].date()):<12}{100*ret:>9.2f}{100*gap:>9.2f}"
              f"{100*intr:>11.2f}{share:>10.0f}%{vol:>12,.0f}{medv:>12,.0f}")
    print()

# The 52-week-high claim on the grains: is the HIGH itself a gap artifact?
print("=== grains: distance to trailing-252 close max, and how it was reached ===")
for t in ["ZC=F", "ZS=F", "ZW=F"]:
    c = px[t]["Close"]
    hi = c.rolling(252, min_periods=200).max()
    at_hi = c.iloc[-1] >= hi.iloc[-1] - 1e-9
    # how many of the last 10 sessions were majority-gap moves
    o = px[t]["Open"]
    ret = c.pct_change()
    gap = o / c.shift(1) - 1.0
    tail = (gap.abs() / ret.abs()).iloc[-10:]
    majority_gap = int((tail > 0.7).sum())
    print(f"{t}: close {c.iloc[-1]:.2f}, 252d max {hi.iloc[-1]:.2f}, at high={at_hi}, "
          f"{majority_gap}/10 recent sessions were >70% gap")
