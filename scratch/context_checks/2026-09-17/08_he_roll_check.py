"""Is the lean-hog collapse a contract roll?

HE=F closed -11.53% today, on a 52-week low, and it fired four price triggers
(P2, P2b, P5, P6, P7b). Continuous lean-hog futures roll between contract
months and the roll shows up in the adjusted series as a single enormous
session that never traded. Check the bar itself: a real -11.5% session has an
intraday range that contains the move; a roll gap does not.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices  # noqa: E402

px = load_prices(["HE=F", "LE=F"])
for t in ["HE=F", "LE=F"]:
    df = px[t].tail(8)
    df = df.assign(
        gap_pct=100 * (df["Open"] / df["Close"].shift(1) - 1.0),
        intraday_pct=100 * (df["Close"] / df["Open"] - 1.0),
        c2c_pct=100 * (df["Close"] / df["Close"].shift(1) - 1.0),
        range_pct=100 * (df["High"] - df["Low"]) / df["Close"],
    )
    print(f"\n=== {t} ===")
    print(df[["Open", "High", "Low", "Close", "Volume", "gap_pct",
              "intraday_pct", "c2c_pct", "range_pct"]].round(3).to_string())

d = px["HE=F"]
last = d.index[-1]
o, h, lo, c = d.loc[last, ["Open", "High", "Low", "Close"]]
prev_c = d["Close"].iloc[-2]
print(f"\nHE=F {last.date()}: prev close {prev_c:.3f}, open {o:.3f} "
      f"(gap {100 * (o / prev_c - 1):+.2f}%), close {c:.3f} "
      f"(intraday {100 * (c / o - 1):+.2f}%), "
      f"session low {lo:.3f} vs prev close {prev_c:.3f}")
print(f"  did the bar's RANGE contain the close-to-close move? "
      f"low {lo:.3f} {'<=' if lo <= prev_c else '>'} prev close {prev_c:.3f}")
if lo > prev_c * 0.999:
    print("  VERDICT: the whole move is a GAP the tape never traded through. "
          "Roll artifact. Every HE=F price trigger today is fake.")
elif o / prev_c - 1 < -0.05:
    print("  VERDICT: opened more than 5% below the prior close with the low "
          "beneath it. Consistent with a roll gap, not a traded decline.")
else:
    print("  VERDICT: the decline was traded intraday. Not a roll artifact.")

vol = d["Volume"].tail(30)
print(f"\n  volume today {int(vol.iloc[-1]):,} vs 20d median "
      f"{int(vol.iloc[-21:-1].median()):,}")
