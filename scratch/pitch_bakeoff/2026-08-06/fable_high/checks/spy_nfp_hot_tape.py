"""Check: short SPY MOC into the August NFP day, hot-tape interaction.

Candidate: enter short SPY at today's close (2026-08-06), exit MOC on the
NFP release day (2026-08-07). Base cell from scratch/august_nfp_cross_asset.py:
Aug NFP day0 -35.8 bps, hit 0.35, N=26. This check interrogates:
  1. does the hot-tape interaction (5d run into the print) make it better
     or is the Aug cell just noise?
  2. era stability (pre/post 2018)
  3. worst windows
  4. day0..+1 in case of a 2-day hold
No bar after 2026-08-05 is used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from macro_calendar import event_dates  # noqa: E402

CUTOFF = pd.Timestamp("2026-08-05")


def load(tkr: str) -> pd.Series:
    mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                         columns=["ticker", "date", "Close"])
    df = mp[mp["ticker"] == tkr].set_index("date").sort_index()[["Close"]]
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")]
    return df.loc[:CUTOFF, "Close"]


def win(c: pd.Series, anchor: pd.Timestamp, a: int, b: int) -> float:
    idx = c.index
    p = idx.searchsorted(anchor)
    lo, hi = p + a, p + b
    if lo - 1 < 0 or hi >= len(idx) or p >= len(idx):
        return np.nan
    return float(c.iloc[hi] / c.iloc[lo - 1] - 1)


def stats(x: pd.Series, label: str) -> None:
    x = x.dropna()
    if len(x) < 3:
        print(f"{label:52s} N={len(x)}")
        return
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    worst = x.min()
    print(f"{label:52s} {x.mean()*1e4:+7.1f} bps  t {t:+5.2f}  N {len(x):3d}"
          f"  hit {(x > 0).mean():.2f}  worst {worst*1e4:+.0f}")


spy = load("SPY")
ret5 = spy.pct_change(5)
nfp = [d for d in event_dates("nfp") if d <= CUTOFF]

# 5d return into the print = as of the session BEFORE the anchor
into = {}
for d in nfp:
    p = spy.index.searchsorted(d)
    if p - 1 < 0 or p >= len(spy.index):
        continue
    into[d] = ret5.iloc[p - 1]
into = pd.Series(into).dropna()

for wname, a, b in [("day0", 0, 0), ("day0..+1", 0, 1), ("+1..+5", 1, 5)]:
    w_all = pd.Series({d: win(spy, d, a, b) for d in into.index}).dropna()
    aug = w_all[[d.month == 8 for d in w_all.index]]
    hot = w_all[into.loc[w_all.index] >= 0.03]
    hot_aug = aug[into.loc[aug.index] >= 0.03]
    cold = w_all[into.loc[w_all.index] <= 0.0]
    print(f"--- window {wname} ---")
    stats(w_all, "all NFP (control)")
    stats(aug, "Aug NFP")
    stats(hot, "all NFP, 5d into >= +3% (hot)")
    stats(cold, "all NFP, 5d into <= 0 (cold, contrast)")
    stats(hot_aug, "Aug NFP, hot")
    if wname == "day0":
        mid = aug[[d.year % 4 == 2 for d in aug.index]]
        print("  Aug NFP midterm day0:",
              " ".join(f"{d.year}:{v*1e4:+.0f}" for d, v in mid.items()))
        for era, cut in [("pre-2013", None), ("2013-2019", None), ("2020+", None)]:
            pass
        e1 = aug[aug.index < "2013-01-01"]
        e2 = aug[(aug.index >= "2013-01-01") & (aug.index < "2020-01-01")]
        e3 = aug[aug.index >= "2020-01-01"]
        stats(e1, "Aug NFP day0 pre-2013")
        stats(e2, "Aug NFP day0 2013-2019")
        stats(e3, "Aug NFP day0 2020+")
        print("  Aug NFP day0 listing:",
              " ".join(f"{d.year}:{v*1e4:+.0f}" for d, v in aug.items()))
        stats(hot[hot.index >= "2018-01-01"], "hot day0 2018+")
        stats(hot[hot.index < "2018-01-01"], "hot day0 pre-2018")
