"""PIT-correct (lag-1) variants of the event-sleeve specs.

Live staging happens pre-market (4:47 AM ET), so any filter can only use
the PRIOR session's close. Checks that the tested edges survive the lag:

1. Midterm short W3 (close td-4 -> open td0) with rank21 measured at td-5
   close instead of td-4.
2. Sep post-quad short with the washout exception z10 measured at the
   session BEFORE opex instead of the opex close.
3. Ex-midterm long W3 (no filter, just restated for the prereg table).

Run: python scratch/event_sleeve_pit_variants.py
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
    df["ret"] = df["Close"].pct_change()
    df["vol21"] = df["ret"].rolling(21).std()
    r21 = df["Close"].pct_change(21)
    df["rank21"] = r21.rolling(252).rank(pct=True) * 100
    return df


def stats(x: pd.Series, label: str) -> str:
    x = x.dropna()
    if len(x) < 4:
        return f"{label:56s} N={len(x)}"
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    return (f"{label:56s} {x.mean()*1e4:+7.1f} bps  t {t:+5.2f}  N {len(x):3d}"
            f"  hit {(x>0).mean():.2f}")


print("--- 1. midterm short W3, rank21 lag comparison ---")
for tkr in ("SPY", "QQQ"):
    df = load(tkr)
    idx, c, o = df.index, df["Close"], df["Open"]
    rows = []
    for d in event_dates("fomc_decision"):
        p = idx.searchsorted(d)
        if p < 300 or p >= len(idx) or d.year % 4 != 2:
            continue
        rows.append({"w3": float(o.iloc[p] / c.iloc[p - 4] - 1),
                     "rank_td4": float(df["rank21"].iloc[p - 4]),
                     "rank_td5": float(df["rank21"].iloc[p - 5])})
    w = pd.DataFrame(rows)
    print(stats(-w.loc[w.rank_td4 < 50, "w3"], f"{tkr} short W3 rank21(td-4)<50 (as tested)"))
    print(stats(-w.loc[w.rank_td5 < 50, "w3"], f"{tkr} short W3 rank21(td-5)<50 (PIT lag-1)"))

print("\n--- 2. Sep post-quad short, washout-exception lag comparison ---")
for tkr in ("IWM", "SPY"):
    df = load(tkr)
    idx, c = df.index, df["Close"]
    rows = []
    for d in event_dates("opex"):
        if d.month != 9:
            continue
        p = idx.searchsorted(d)
        if p < 300 or p >= len(idx) - 11 or idx[p] > pd.Timestamp("2026-01-01"):
            continue
        me = idx.searchsorted(pd.Timestamp(d.year, 9, 28) + pd.Timedelta(days=4),
                              side="left") - 1
        vol = df["vol21"].iloc[p - 10]
        rows.append({
            "yr": d.year,
            "to_me": float(c.iloc[me] / c.iloc[p] - 1),
            "z10_at_opex": float((c.iloc[p] / c.iloc[p - 10] - 1)
                                 / (vol * np.sqrt(10))) if vol > 0 else np.nan,
            "z10_lag1": float((c.iloc[p - 1] / c.iloc[p - 11] - 1)
                              / (vol * np.sqrt(10))) if vol > 0 else np.nan,
        })
    w = pd.DataFrame(rows)
    print(stats(-w["to_me"], f"{tkr} Sep short unconditional"))
    print(stats(-w.loc[w.z10_at_opex >= -1, "to_me"],
                f"{tkr} Sep short, skip washout z10(opex)<-1 (as tested)"))
    print(stats(-w.loc[w.z10_lag1 >= -1, "to_me"],
                f"{tkr} Sep short, skip washout z10(td-1)<-1 (PIT lag-1)"))
    skipped = w.loc[w.z10_lag1 < -1, "yr"].tolist()
    print(f"    lag-1 exception would have skipped: {skipped}")

print("\n--- 3. ex-midterm long W3 restated ---")
for tkr in ("SPY", "QQQ"):
    df = load(tkr)
    idx, c, o = df.index, df["Close"], df["Open"]
    w = []
    for d in event_dates("fomc_decision"):
        p = idx.searchsorted(d)
        if p < 300 or p >= len(idx) or d.year % 4 == 2:
            continue
        w.append(float(o.iloc[p] / c.iloc[p - 4] - 1))
    print(stats(pd.Series(w), f"{tkr} long W3 (close td-4 -> open td0)"))
