"""Next week IS August opex week (opex Fri 2026-08-21, VIX expiry Wed
8/19). The meme is that August opex week is weak. Check it: SPY return over
the 5 trading days ending at the August third Friday (tonight's close is the
anchor, so the cell is exactly next week's shape). Controls: the same
5-td-into-opex window in every other month, and all rolling 5-td returns.
Era split. One read per year, non-overlapping by construction."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import era_split, show, sign_test, summarize
from pitch_lab import load_prices

px = load_prices(["SPY"])["SPY"]
close = px["Close"].dropna()
idx = close.index


def third_friday(yr: int, mo: int) -> pd.Timestamp:
    d = pd.Timestamp(yr, mo, 1)
    fridays = pd.date_range(d, d + pd.offsets.MonthEnd(0), freq="W-FRI")
    return fridays[2]


def opex_week_ret(yr: int, mo: int):
    tf = third_friday(yr, mo)
    # last session at or before the third Friday (holiday-safe)
    pos = idx.searchsorted(tf, side="right") - 1
    if pos < 5 or pos >= len(idx):
        return None
    if abs((idx[pos] - tf).days) > 3:
        return None
    return close.iloc[pos] / close.iloc[pos - 5] - 1.0


aug_vals, aug_dates = [], []
other_vals = []
for yr in range(int(idx[0].year), int(idx[-1].year) + 1):
    for mo in range(1, 13):
        r = opex_week_ret(yr, mo)
        if r is None:
            continue
        if mo == 8:
            if yr == 2026:
                continue  # next week, not history
            aug_vals.append(r)
            aug_dates.append(third_friday(yr, 8))
        else:
            other_vals.append(r)

aug = np.array(aug_vals)
oth = np.array(other_vals)
all5 = (close / close.shift(5) - 1.0).dropna().values

rows = [
    summarize(aug, "SPY 5td into AUGUST opex Fri"),
    summarize(oth, "5td into opex Fri, other months"),
    summarize(all5, "all rolling 5td (baseline)"),
]
show(rows, "August opex week, SPY")
wins = int((aug > 0).sum())
print(f"green {wins}/{len(aug)}, sign_p(down) {sign_test(len(aug)-wins, len(aug)):.4f}")
for e in era_split(pd.DatetimeIndex(aug_dates), aug):
    print(e)
print("last 10:", [f"{d.year}:{v*100:+.1f}%" for d, v in zip(aug_dates[-10:], aug[-10:])])
