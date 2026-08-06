"""August NFP cross-asset study: dollar, gold, silver, bonds.

Question (McKinley, 2026-08-06): has the NFP released in August (July
reference month) historically been bullish USD, bearish gold/silver,
bullish bonds — especially in midterm years? Tomorrow (2026-08-07) is an
August NFP in a midterm year.

Assets: UUP (dollar, 2007+), GLD (2004+), SLV (2006+), TLT (2002+),
SPY for context. Windows: release day (close-to-close), +1..+5 after,
and release day through +5. Controls: the same stat on ALL NFP days
(is August special vs a generic NFP?) and on all August days (seasonal
drift control). Midterm subset listed per event (N is tiny — read it as
a listing, not a t-stat).

Run: python scratch/august_nfp_cross_asset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from macro_calendar import event_dates  # noqa: E402

TICKERS = ["UUP", "GLD", "SLV", "TLT", "SPY"]


def load(tkr: str) -> pd.Series:
    mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                         columns=["ticker", "date", "Close"])
    df = mp[mp["ticker"] == tkr].set_index("date").sort_index()[["Close"]]
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")]
    return df["Close"]


def stats(x: pd.Series, label: str) -> str:
    x = x.dropna()
    if len(x) < 3:
        return f"{label:40s} N={len(x)}"
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    return (f"{label:40s} {x.mean()*1e4:+7.1f} bps  t {t:+5.2f}  N {len(x):3d}"
            f"  hit {(x>0).mean():.2f}")


def win(c: pd.Series, anchor: pd.Timestamp, a: int, b: int) -> float:
    idx = c.index
    p = idx.searchsorted(anchor)
    lo, hi = p + a, p + b
    if lo - 1 < 0 or hi >= len(idx) or p >= len(idx):
        return np.nan
    return float(c.iloc[hi] / c.iloc[lo - 1] - 1)


px = {t: load(t) for t in TICKERS}
nfp = event_dates("nfp")
aug = [d for d in nfp if d.month == 8]
non_aug = [d for d in nfp if d.month != 8]

WINDOWS = [("day0", 0, 0), ("+1..+5", 1, 5), ("day0..+5", 0, 5)]

for tkr in TICKERS:
    c = px[tkr]
    lo, hi = c.index.min(), c.index.max()
    print("=" * 92)
    print(f"{tkr}  ({lo:%Y-%m} .. {hi:%Y-%m})")
    print("=" * 92)
    for wname, a, b in WINDOWS:
        w_aug = pd.Series({d: win(c, d, a, b) for d in aug
                           if lo <= d <= hi}).dropna()
        w_all = pd.Series({d: win(c, d, a, b) for d in nfp
                           if lo <= d <= hi}).dropna()
        print(stats(w_aug, f"AUG NFP {wname}"))
        print(stats(w_all, f"  all-months NFP {wname} (control)"))
        if wname == "day0":
            # seasonal control: all August days
            ret = c.pct_change()
            print(stats(ret[ret.index.month == 8],
                        "  all August days (seasonal control)"))
        mid = w_aug[[d.year % 4 == 2 for d in w_aug.index]]
        if len(mid):
            line = " ".join(f"{d.year}:{v*1e4:+.0f}" for d, v in mid.items())
            print(f"  midterm only {wname}: mean {mid.mean()*1e4:+.0f} bps, "
                  f"{int((mid>0).sum())}/{len(mid)} up  [{line}]")
    print()
