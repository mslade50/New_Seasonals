"""Kill check: are the lean hog and live cattle collapses real, or roll gaps?

HE=F prints 5d -15.80% and 21d -20.29%, the 0.4th percentile of its own year,
and fired three separate price triggers. LE=F fired one at 5d -5.84%. Both are
CONTINUOUS futures contracts stitched across expiries, and this repo has a
standing note that HE=F roll gaps fire price-state triggers as real moves.

This script exists to kill those cells, not to publish them. A move that shows
up as an overnight gap with a quiet intraday range is a contract change, not a
market event, and it must not reach the brief either way without being checked.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices, summarize, show  # noqa: E402

px = load_prices(["HE=F", "LE=F", "ZC=F"])

for t in ["HE=F", "LE=F"]:
    df = px[t].dropna(subset=["Close"]).tail(30).copy()
    df["ret"] = df["Close"].pct_change() * 100
    df["gap"] = (df["Open"] / df["Close"].shift(1) - 1.0) * 100
    df["intraday"] = (df["Close"] / df["Open"] - 1.0) * 100
    df["range"] = (df["High"] / df["Low"] - 1.0) * 100
    print("=" * 78)
    print(f"{t}: last 15 sessions, decomposed into overnight gap + intraday")
    print("=" * 78)
    out = df.tail(15)[["Open", "High", "Low", "Close", "Volume",
                       "ret", "gap", "intraday", "range"]].round(3)
    print(out.to_string())

    last10 = df.tail(10)
    print(f"\n  sum of last 10 daily returns : {last10['ret'].sum():+.2f}%")
    print(f"  sum of overnight gaps        : {last10['gap'].sum():+.2f}%")
    print(f"  sum of intraday moves        : {last10['intraday'].sum():+.2f}%")
    big_gap = last10[np.abs(last10["gap"]) > 2.0]
    print(f"  sessions with a >2% overnight gap: {len(big_gap)}")
    if len(big_gap):
        print("   ", [(str(d.date()), round(g, 2))
                      for d, g in zip(big_gap.index, big_gap["gap"])])
    # a roll shows as a gap with a NORMAL intraday range, not a wide one
    print(f"  median daily range, last 10  : {last10['range'].median():.2f}%")
    print(f"  median daily range, prior 250: "
          f"{((px[t]['High']/px[t]['Low']-1)*100).tail(260).head(250).median():.2f}%")
    print()

print("=" * 78)
print("CONTROL: ZC=F, the same decomposition, where the move is believed real")
print("=" * 78)
df = px["ZC=F"].dropna(subset=["Close"]).tail(10).copy()
df["ret"] = df["Close"].pct_change() * 100
df["gap"] = (df["Open"] / df["Close"].shift(1) - 1.0) * 100
df["intraday"] = (df["Close"] / df["Open"] - 1.0) * 100
print(df[["Open", "High", "Low", "Close", "ret", "gap", "intraday"]].round(3).to_string())
print(f"\n  corn last 5: gaps {df['gap'].tail(5).sum():+.2f}%, "
      f"intraday {df['intraday'].tail(5).sum():+.2f}%")

print("\n" + "=" * 78)
print("VERDICT INPUTS: share of the 21d move that is overnight gap")
print("=" * 78)
for t in ["HE=F", "LE=F", "ZC=F"]:
    df = px[t].dropna(subset=["Close"]).tail(22).copy()
    gap = (df["Open"] / df["Close"].shift(1) - 1.0).sum() * 100
    intr = (df["Close"] / df["Open"] - 1.0).sum() * 100
    tot = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1.0) * 100
    print(f"  {t}: 21d total {tot:+.2f}%  |  gaps {gap:+.2f}%  "
          f"intraday {intr:+.2f}%  |  gap share {100*gap/(gap+intr):.0f}%"
          if abs(gap + intr) > 1e-9 else f"  {t}: degenerate")
