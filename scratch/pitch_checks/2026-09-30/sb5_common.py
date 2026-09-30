"""sb5: sector longs XLP / XLF from Oct 1-2 MOC, 21td, SPY as control.
Anchor T = last session on/before Sep 30 (sb2_common.anchors). lag=1 -> entry
Oct 1 close (today's MOC-tomorrow), lag=2 -> Oct 2 close (the checkers' window).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from sb2_common import (MIDTERMS, anchors, yearly, full_report, cohort_line,
                        drop_best, concentration, era)

import numpy as np
import pandas as pd

H = 21
PX = close_panel(["XLP", "XLF", "SPY"]).dropna()
IDX = PX.index


def near_high_flags(anc: dict[int, int]) -> pd.Series:
    spy = PX["SPY"]
    hi = spy.rolling(252).max()
    off = 1 - spy / hi
    return pd.Series({y: bool(off.iloc[p] <= 0.02) for y, p in anc.items()}), off


def excess(tk: str, anc, lag, h) -> pd.Series:
    return (yearly(PX[tk], anc, lag, h, 1.0) - yearly(PX["SPY"], anc, lag, h, 1.0)).dropna()


def round1_2(tk: str) -> dict:
    out = {}
    print(f"### {tk}  data {IDX[0].date()}..{IDX[-1].date()}")
    for lag in (1, 2):
        anc = anchors(IDX, 9, 30)
        s = full_report(PX[tk], anc, lag, H, 1.0, f"{tk} lag{lag}")
        sp = yearly(PX["SPY"], anc, lag, H, 1.0)
        ex = excess(tk, anc, lag, H)
        print(f"  SPY same window: {cohort_line(sp, 'SPY')}")
        print(f"  SPY midterm: {cohort_line(sp[sp.index.isin(MIDTERMS)], 'SPYmid')}")
        print(f"  EXCESS {tk}-SPY: {cohort_line(ex, 'all')}")
        print(f"  EXCESS midterm: {cohort_line(ex[ex.index.isin(MIDTERMS)], 'mid')}")
        print("  EXCESS " + era(ex, 2010) + " || " + era(ex, 2018))
        exd = (fwd_lag(PX[tk], H, lag) - fwd_lag(PX["SPY"], H, lag)).dropna()
        print(f"  EXCESS all-day drift {100*exd.mean():+.2f}%  hit {100*(exd>0).mean():.0f}%")
        print("  last10: " + ", ".join(f"{y}:{100*s[y]:+.2f}/ex{100*ex[y]:+.2f}"
                                       for y in s.index[-10:]))
        # (d) near-high branch
        nh, off = near_high_flags(anc)
        nh = nh.reindex(s.index)
        print(f"  SPY off-high at 2026-09-29: {100*off.iloc[-1]:.2f}%")
        print("  NEAR-HIGH (<=2%): " + cohort_line(s[nh], tk) + " || excess " + cohort_line(ex[nh.reindex(ex.index)], 'ex'))
        print("  FAR (>2%):        " + cohort_line(s[~nh], tk) + " || excess " + cohort_line(ex[~nh.reindex(ex.index)], 'ex'))
        r = fwd_lag(PX[tk], H, lag)
        rs = fwd_lag(PX["SPY"], H, lag)
        m = (off <= 0.02) & r.notna()
        print(f"  own drift ALL days SPY near-high: {tk} {100*r[m].mean():+.2f}% hit {100*(r[m]>0).mean():.0f}% | "
              f"excess {100*(r-rs)[m].mean():+.2f}% | Oct-Dec near-high days {tk} "
              f"{100*r[m & (r.index.month>=10)].mean():+.2f}%")
        out[lag] = (s, sp, ex)
    # (b) neighbours
    print(f"\n  NEIGHBOURS {tk} lag1 (shift x horizon): mean / hit / excess-mean / ex-hit")
    for sh in (-3, -2, -1, 0, 1, 2, 3):
        anc = anchors(IDX, 9, 30, shift=sh)
        cells = []
        for h in (10, 15, 21, 26):
            s = yearly(PX[tk], anc, 1, h, 1.0)
            ex = excess(tk, anc, 1, h)
            cells.append(f"h{h}: {100*s.mean():+.2f} {int((s>0).sum())}/{len(s)} ex{100*ex.mean():+.2f} {int((ex>0).sum())}/{len(ex)}")
        print(f"   sh{sh:+d} | " + " | ".join(cells))
    return out
