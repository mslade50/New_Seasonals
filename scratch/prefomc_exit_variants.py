"""Pre-FOMC drift: where should the exit land?

Variants (ex-midterm years, SPY, 2000+):
  A. close td-4 -> close td-1  (out before decision day entirely)
  B. close td-4 -> close td0   (hold through statement + presser; original)
  C. close td-4 -> open td0    (out at the decision-day open)
  D. decision day alone: overnight (close-1 -> open0) and intraday
     (open0 -> close0), to see what day 0 actually contributes.

Run: python scratch/prefomc_exit_variants.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from macro_calendar import event_dates  # noqa: E402


def load(tkr: str) -> pd.DataFrame:
    mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                         columns=["ticker", "date", "Open", "Close"])
    df = mp[mp["ticker"] == tkr].set_index("date").sort_index()[["Open", "Close"]]
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")]
    return df[df.index >= "2000-01-01"]


def stats(x: pd.Series, label: str) -> str:
    x = x.dropna()
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    ann = x.mean() * 1e4
    return (f"{label:44s} mean {ann:+7.1f} bps  t {t:+5.2f}  N {len(x):3d}  "
            f"hit {(x>0).mean():.2f}  worst {x.min()*1e4:+7.0f}")


for tkr in ("SPY", "QQQ"):
    df = load(tkr)
    idx = df.index
    c, o = df["Close"], df["Open"]
    fomc = event_dates("fomc_decision")
    fomc = fomc[(fomc >= idx.min()) & (fomc <= idx.max())]
    recs = []
    for d in fomc:
        if d.year % 4 == 2:
            continue
        p = idx.searchsorted(d)
        if p < 4 or p >= len(idx):
            continue
        recs.append({
            "date": idx[p],
            "A_to_dm1": float(c.iloc[p - 1] / c.iloc[p - 4] - 1),
            "B_to_d0close": float(c.iloc[p] / c.iloc[p - 4] - 1),
            "C_to_d0open": float(o.iloc[p] / c.iloc[p - 4] - 1),
            "d0_overnight": float(o.iloc[p] / c.iloc[p - 1] - 1),
            "d0_intraday": float(c.iloc[p] / o.iloc[p] - 1),
            "d0_full": float(c.iloc[p] / c.iloc[p - 1] - 1),
        })
    w = pd.DataFrame(recs).set_index("date")
    print(f"===== {tkr} ex-midterm, N={len(w)} =====")
    print(stats(w.A_to_dm1, "A: exit close DAY BEFORE (td-4c..td-1c)"))
    print(stats(w.B_to_d0close, "B: exit DECISION-DAY CLOSE (td-4c..td0c)"))
    print(stats(w.C_to_d0open, "C: exit decision-day OPEN"))
    print(stats(w.d0_overnight, "   day0 overnight leg alone"))
    print(stats(w.d0_intraday, "   day0 intraday leg alone (2pm inside)"))
    print(stats(w.d0_full, "   day0 full close-to-close alone"))
    # era stability of the day-0 intraday leg
    for era, lo, hi in (("2000-2012", "2000", "2012"), ("2013+", "2013", "2027")):
        sub = w.loc[lo:hi]
        if len(sub) > 5:
            print(stats(sub.d0_intraday, f"   day0 intraday, {era}"))
    print()
