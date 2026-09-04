"""Guard: are today's big futures moves real, or continuous-contract roll gaps?

Memory note futures-continuous-roll-gaps: HE=F/LE=F (and possibly other thin
contracts) gap on the front-month roll and every price trigger reads it as a
market move. Decompose today's session into gap (Open/prevClose) vs intraday
(Close/Open) for every futures ticker that fired a trigger, then check whether
that ticker's history of comparable moves is gap-dominated and calendar-clustered.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices  # noqa

TICKERS = ["ZC=F", "LE=F", "KC=F", "GC=F", "SI=F", "CT=F", "SB=F", "ZW=F", "HE=F"]
ASOF = pd.Timestamp("2026-08-24")

px = load_prices(TICKERS)

print("=== today's session decomposition ===")
print(f"{'tkr':<8}{'close':>10}{'sess%':>8}{'gap%':>8}{'intra%':>8}{'range%':>8}")
for t in TICKERS:
    df = px.get(t)
    if df is None or ASOF not in df.index:
        print(f"{t:<8} no bar")
        continue
    i = df.index.get_loc(ASOF)
    row, prev = df.iloc[i], df.iloc[i - 1]
    gap = row["Open"] / prev["Close"] - 1
    intra = row["Close"] / row["Open"] - 1
    sess = row["Close"] / prev["Close"] - 1
    rng = (row["High"] - row["Low"]) / row["Open"]
    print(f"{t:<8}{row['Close']:>10.2f}{sess*100:>8.2f}{gap*100:>8.2f}"
          f"{intra*100:>8.2f}{rng*100:>8.2f}")

print()
print("=== history of comparable-magnitude sessions (same sign, |ret| >= |today|*0.8) ===")
for t in TICKERS:
    df = px.get(t)
    if df is None or ASOF not in df.index:
        continue
    c = df["Close"]
    ret = c.pct_change()
    today = ret.loc[ASOF]
    if abs(today) < 0.01:
        continue
    thr = abs(today) * 0.8
    m = (ret.abs() >= thr) & (np.sign(ret) == np.sign(today)) & (df.index <= ASOF)
    d = df.loc[m]
    if len(d) < 3:
        print(f"{t:<8} n={len(d)} too few comparable sessions")
        continue
    gaps = (d["Open"] / c.shift(1).loc[d.index] - 1) * 100
    intras = (d["Close"] / d["Open"] - 1) * 100
    months = pd.Series(d.index.month).value_counts().sort_index()
    print(f"{t:<8} n={len(d):<4} mean gap {gaps.mean():>6.2f}%  mean intraday "
          f"{intras.mean():>6.2f}%  gap-dominated {int((gaps.abs() > intras.abs()).sum())}/{len(d)}")
    print(f"{'':<8} month spread: {dict(months)}")
