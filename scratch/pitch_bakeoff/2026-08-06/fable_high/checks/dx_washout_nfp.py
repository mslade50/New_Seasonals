"""Check: long the dollar (DX futures) after a washout, into/after NFP.

Candidate: DXY z10 -1.59, 5d -1.1%, 5d rank 6 as of 2026-08-05, NFP on
2026-08-07. Does a washed-out dollar bounce over the NFP week?
Cells: NFP events where the dollar's 5d return into the print <= -1%;
controls: all NFP, and ALL washout days (event-independent).
Series: DX-Y.NYB if present in master_prices (falls back to UUP).
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

mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                     columns=["ticker", "date", "Close"])


def load(tkr: str) -> pd.Series:
    df = mp[mp["ticker"] == tkr].set_index("date").sort_index()[["Close"]]
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")]
    return df.loc[:CUTOFF, "Close"]


tkr = "DX-Y.NYB" if (mp["ticker"] == "DX-Y.NYB").any() else "UUP"
c = load(tkr)
print(f"series: {tkr}  {c.index.min():%Y-%m}..{c.index.max():%Y-%m}  n={len(c)}")

ret5 = c.pct_change(5)


def win(anchor: pd.Timestamp, a: int, b: int) -> float:
    idx = c.index
    p = idx.searchsorted(anchor)
    lo, hi = p + a, p + b
    if lo - 1 < 0 or hi >= len(idx) or p >= len(idx):
        return np.nan
    return float(c.iloc[hi] / c.iloc[lo - 1] - 1)


def stats(x: pd.Series, label: str) -> None:
    x = x.dropna()
    if len(x) < 3:
        print(f"{label:48s} N={len(x)}")
        return
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    print(f"{label:48s} {x.mean()*1e4:+7.1f} bps  t {t:+5.2f}  N {len(x):3d}"
          f"  hit {(x > 0).mean():.2f}  worst {x.min()*1e4:+.0f}")


nfp = [d for d in event_dates("nfp") if d <= CUTOFF]
into = {}
for d in nfp:
    p = c.index.searchsorted(d)
    if p - 1 < 0 or p >= len(c.index):
        continue
    into[d] = ret5.iloc[p - 1]
into = pd.Series(into).dropna()

for wname, a, b in [("day0", 0, 0), ("day0..+5", 0, 5)]:
    w_all = pd.Series({d: win(d, a, b) for d in into.index}).dropna()
    washed = w_all[into.loc[w_all.index] <= -0.01]
    print(f"--- window {wname} ---")
    stats(w_all, "all NFP (control)")
    stats(washed, "NFP, 5d into <= -1% (washed out)")
    stats(washed[washed.index >= "2018-01-01"], "  washed 2018+")
    stats(washed[washed.index < "2018-01-01"], "  washed pre-2018")

# event-independent control: any washout day -> fwd 5d
fwd5 = c.shift(-5) / c - 1
mask = ret5 <= -0.01
# decluster: keep first day of each run
first = mask & ~mask.shift(1, fill_value=False)
uncond = fwd5[first].dropna()
stats(uncond, "ANY washout day (declustered), fwd 5d")
stats(fwd5.dropna(), "unconditional fwd 5d drift")
