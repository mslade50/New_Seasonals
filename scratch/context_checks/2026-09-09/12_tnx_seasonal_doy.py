"""The Sep-10 seasonal cell on the 10-year yield, midterm years.

The sweep put ^TNX midterm h1 at n=6, mean +1.355%, 5 up 1 down, sign p 0.1094.
An index, so unlike the copper version of this cell there is no contract-roll
exposure in either the entry state or the history. Kept as the clean fallback
for the seasonal slot.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, cluster_note, era_split, fwd_ret,  # noqa
                       sign_test, summarize)

px = close_panel(["^TNX", "^GSPC", "IEF"])
TARGET_MD = (9, 10)


def doy_anchors(s: pd.Series, window: int = 2) -> pd.DatetimeIndex:
    """One pick per prior year: the session nearest Sep 10, within +/-2 td."""
    s = s.dropna()
    idx = pd.DatetimeIndex(s.index)
    picks = []
    for yr in sorted({d.year for d in idx}):
        target = pd.Timestamp(year=yr, month=TARGET_MD[0], day=TARGET_MD[1])
        yr_idx = idx[(idx >= target - pd.Timedelta(days=10))
                     & (idx <= target + pd.Timedelta(days=10))]
        if len(yr_idx) == 0:
            continue
        # nearest by TRADING-day distance, capped at +/-window sessions
        pos = int(idx.searchsorted(target))
        if pos >= len(idx):
            continue
        cand = idx[max(0, pos - window):min(len(idx), pos + window + 1)]
        cand = cand[(cand.year == yr)]
        if len(cand) == 0:
            continue
        best = min(cand, key=lambda d: abs((d - target).days))
        picks.append(best)
    return pd.DatetimeIndex(sorted(set(picks)))


for name in ["^TNX", "^GSPC", "IEF"]:
    s = px[name].dropna()
    anchors = doy_anchors(s)
    anchors = anchors[anchors < pd.Timestamp("2026-09-09")]
    print(f"\n{'=' * 70}\n{name}: {len(anchors)} yearly anchors, "
          f"{anchors[0].date()} to {anchors[-1].date()}")
    for h in (1, 5):
        f = fwd_ret(s, h).reindex(anchors).dropna()
        mid = f[[d.year % 4 == 2 for d in f.index]]
        for label, v in (("all years", f), ("midterm", mid)):
            if len(v) == 0:
                continue
            st = summarize(v.values, label)
            up = int((v.values > 0).sum())
            n = len(v)
            print(f"  h={h} {label:<10} n={n:<3} mean {st['mean_pct']:+.3f}%  "
                  f"med {st['median_pct']:+.3f}%  hit {st['hit']:.1f}%  "
                  f"t {st['t']:+.2f}  {up}-{n - up} up  "
                  f"sign p {sign_test(up, n):.4f}")
            if label == "midterm":
                for d, val in v.items():
                    print(f"        {d.date()}  {100 * val:+.3f}%")
            else:
                eras = era_split(pd.DatetimeIndex(v.index), v.values)
                print("        era: " + " | ".join(
                    f"{e['label']} n={e.get('n', 0)} "
                    f"{e.get('mean_pct', float('nan')):+.3f}%" for e in eras))
                print(f"        {cluster_note(pd.DatetimeIndex(v.index), v.values)}")

print("\nentry state 2026-09-09 (for the record, not part of the cell):")
tnx = px["^TNX"].dropna()
print(f"  ^TNX close {tnx.iloc[-1]:.3f}, "
      f"{100 * (tnx.iloc[-1] / tnx.rolling(252).max().iloc[-1] - 1):+.2f}% "
      f"vs its 252d max")
ief = px["IEF"].dropna()
print(f"  IEF  close {ief.iloc[-1]:.3f}, "
      f"{100 * (ief.iloc[-1] / ief.rolling(252).min().iloc[-1] - 1):+.2f}% "
      f"above its 252d min")
