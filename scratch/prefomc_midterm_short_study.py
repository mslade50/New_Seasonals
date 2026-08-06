"""Pre-FOMC MIDTERM-YEAR SHORT: exit variants, shorter windows, and
overbought (5d/21d percentile rank) filters. Also re-checks the ex-midterm
LONG under the same rank filters, and a MOO-friendly entry (open td-3).

Windows (all returns reported from the SHORT side for midterm, LONG side
for ex-midterm):
  W1 full:        close td-4 -> close td0
  W2 skip-day0:   close td-4 -> close td-1
  W3 to-open:     close td-4 -> open td0
  W4 daybefore:   close td-2 -> close td-1
  W5 day0 alone:  close td-1 -> close td0
  W6 overnight0:  close td-1 -> open td0
  W7 moo-entry:   open td-3 -> open td0

Rank convention: trailing 5d / 21d return, percentile-ranked against the
prior 252 sessions, measured at the ENTRY close (td-4 for W1-W3, td-2 for
W4, td-1 for W5/W6; W7 uses td-4 close ranks, the last known at its open
entry... conservative: use td-4 close for W7 too).

Run: python scratch/prefomc_midterm_short_study.py
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
    df = df[df.index >= "1999-01-01"]
    for n in (5, 21):
        r = df["Close"].pct_change(n)
        df[f"rank{n}"] = r.rolling(252).rank(pct=True) * 100
    return df


def stats(x: pd.Series, label: str) -> str:
    x = x.dropna()
    if len(x) < 4:
        return f"{label:52s} N={len(x)} (too small)"
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    return (f"{label:52s} {x.mean()*1e4:+7.1f} bps  t {t:+5.2f}  N {len(x):3d}"
            f"  hit {(x>0).mean():.2f}  worst {x.min()*1e4:+6.0f}")


def build(tkr: str) -> pd.DataFrame:
    df = load(tkr)
    idx, c, o = df.index, df["Close"], df["Open"]
    rows = []
    for d in event_dates("fomc_decision"):
        p = idx.searchsorted(d)
        if p < 300 or p >= len(idx):
            continue
        rows.append({
            "date": idx[p], "midterm": d.year % 4 == 2,
            "W1_full": float(c.iloc[p] / c.iloc[p - 4] - 1),
            "W2_skipd0": float(c.iloc[p - 1] / c.iloc[p - 4] - 1),
            "W3_toopen": float(o.iloc[p] / c.iloc[p - 4] - 1),
            "W4_daybefore": float(c.iloc[p - 1] / c.iloc[p - 2] - 1),
            "W5_day0": float(c.iloc[p] / c.iloc[p - 1] - 1),
            "W6_ovn0": float(o.iloc[p] / c.iloc[p - 1] - 1),
            "W7_moo": float(o.iloc[p] / o.iloc[p - 3] - 1),
            "rank5": float(df["rank5"].iloc[p - 4]),
            "rank21": float(df["rank21"].iloc[p - 4]),
            "rank5_d2": float(df["rank5"].iloc[p - 2]),
            "rank21_d2": float(df["rank21"].iloc[p - 2]),
        })
    return pd.DataFrame(rows).set_index("date")


WINDOWS = ["W1_full", "W2_skipd0", "W3_toopen", "W4_daybefore",
           "W5_day0", "W6_ovn0", "W7_moo"]

for tkr in ("SPY", "QQQ"):
    w = build(tkr)
    mid, ex = w[w.midterm], w[~w.midterm]
    print("=" * 100)
    print(f"{tkr}: MIDTERM SHORT (returns shown from the short side)")
    print("=" * 100)
    for win in WINDOWS:
        print(stats(-mid[win], f"{win}  unconditioned"))
    print()
    rank_cuts = [("rank21>70", mid.rank21 > 70), ("rank21>80", mid.rank21 > 80),
                 ("rank5>70", mid.rank5 > 70), ("rank5>80", mid.rank5 > 80),
                 ("rank5>70 & rank21>70", (mid.rank5 > 70) & (mid.rank21 > 70)),
                 ("rank21<50 (not overbot)", mid.rank21 < 50)]
    for win in ("W1_full", "W3_toopen", "W4_daybefore", "W5_day0"):
        for name, m in rank_cuts:
            # W4 conditions on ranks at its own entry (td-2)
            mm = m
            if win == "W4_daybefore":
                mm = {"rank21>70": mid.rank21_d2 > 70,
                      "rank21>80": mid.rank21_d2 > 80,
                      "rank5>70": mid.rank5_d2 > 70,
                      "rank5>80": mid.rank5_d2 > 80,
                      "rank5>70 & rank21>70": (mid.rank5_d2 > 70) & (mid.rank21_d2 > 70),
                      "rank21<50 (not overbot)": mid.rank21_d2 < 50}[name]
            print(stats(-mid.loc[mm, win], f"{win}  {name}"))
        print()

    print(f"{tkr}: EX-MIDTERM LONG under the same rank filters (W3 to-open)")
    for name, m in [("all", ex.rank5.notna()), ("rank5<20 EXCLUDED (>=20)", ex.rank5 >= 20),
                    ("rank5<20 only", ex.rank5 < 20), ("rank21>80 only", ex.rank21 > 80),
                    ("rank21<80", ex.rank21 < 80)]:
        print(stats(ex.loc[m, "W3_toopen"], f"long W3  {name}"))
    print()
