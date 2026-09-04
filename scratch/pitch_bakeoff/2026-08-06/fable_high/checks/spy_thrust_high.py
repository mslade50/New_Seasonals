"""Check: what follows a +5% 5d SPY thrust ending within 1% of a 52w high?

Context cell for the current tape (5d +5.53%, 0.2% off the high). Determines
whether the NFP-day short candidate is fighting a strong continuation drift,
and whether a standalone thrust-continuation long has legs.
Cells: thrust = 5d ret >= +5% AND close within 1% of 252d max -> fwd 1/5/10d.
Splits: midterm years, 2018+, VIX-era proxy via realized vol. Declustered 10td.
No bar after 2026-08-05 is used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
CUTOFF = pd.Timestamp("2026-08-05")

mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                     columns=["ticker", "date", "Close"])
df = mp[mp["ticker"] == "SPY"].set_index("date").sort_index()[["Close"]]
df.index = pd.to_datetime(df.index).normalize()
df = df[~df.index.duplicated(keep="last")]
spy = df.loc[:CUTOFF, "Close"]

r5 = spy.pct_change(5)
near = spy / spy.rolling(252).max() - 1 >= -0.01
mask = (r5 >= 0.05) & near


def decluster(mask: pd.Series, gap: int = 10) -> pd.DatetimeIndex:
    out, last = [], None
    idx = mask.index
    for i, v in enumerate(mask.values):
        if v and (last is None or i - last > gap):
            out.append(idx[i])
            last = i
    return pd.DatetimeIndex(out)


def stats(x: pd.Series, label: str) -> None:
    x = x.dropna()
    if len(x) < 3:
        print(f"{label:48s} N={len(x)}")
        return
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    print(f"{label:48s} {x.mean()*1e4:+7.1f} bps  t {t:+5.2f}  N {len(x):3d}"
          f"  hit {(x > 0).mean():.2f}  worst {x.min()*1e4:+.0f}")


days = decluster(mask.fillna(False))
print(f"thrust-at-high episodes (declustered): {len(days)}")
print(" ", " ".join(d.strftime("%Y-%m-%d") for d in days))
for h in (1, 5, 10):
    fwd = spy.shift(-h) / spy - 1
    stats(fwd.reindex(days), f"fwd {h}d, all")
    stats(fwd.reindex(days[days >= "2018-01-01"]), f"fwd {h}d, 2018+")
    mid = days[[d.year % 4 == 2 for d in days]]
    stats(fwd.reindex(mid), f"fwd {h}d, midterm years")
    stats(fwd.dropna(), f"fwd {h}d unconditional")
    print()
