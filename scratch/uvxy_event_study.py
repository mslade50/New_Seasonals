"""UVXY / VIX around macro events + vol seasonality.

Data: UVXY 2011-10+ (adjusted, so reverse splits are handled; the series
carries structural decay of roughly -30 to -50 bps/day from futures roll,
which is THE hurdle for any long-vol window and THE harvest for short-vol),
^VIX 2000+ (level, untradeable, clean seasonal shape).

Sections:
  A. all-days baseline (the decay hurdle)
  B. event-relative day grids (UVXY ret + VIX chg): fomc, cpi, nfp, opex,
     quad, vix_expiry, jackson_hole
  C. event windows: pre-FOMC short vol (w/ midterm split), Sep post-quad
     long vol (T3 companion), Dec post-opex short vol, CPI day by era
  D. seasonality: UVXY mean daily ret + VIX avg change by calendar month;
     long-vol window Aug->mid-Oct per year; short-vol Nov-Dec per year;
     midterm-year splits

Run: python scratch/uvxy_event_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from macro_calendar import event_dates, td_offset  # noqa: E402


def load(tkr: str) -> pd.DataFrame:
    mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                         columns=["ticker", "date", "Close"])
    df = mp[mp["ticker"] == tkr].set_index("date").sort_index()[["Close"]]
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")]
    df["ret"] = df["Close"].pct_change()
    return df


def stats(x: pd.Series, label: str) -> str:
    x = x.dropna()
    if len(x) < 3:
        return f"{label:44s} N={len(x)}"
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    return (f"{label:44s} {x.mean()*1e4:+8.1f} bps  t {t:+5.2f}  N {len(x):4d}"
            f"  hit {(x>0).mean():.2f}")


def win(df: pd.DataFrame, anchor: pd.Timestamp, a: int, b: int) -> float:
    idx = df.index
    p = idx.searchsorted(anchor)
    lo, hi = p + a, p + b
    if lo - 1 < 0 or hi >= len(idx) or p >= len(idx):
        return np.nan
    return float(df["Close"].iloc[hi] / df["Close"].iloc[lo - 1] - 1)


uvxy = load("UVXY")
vix = load("^VIX")

print("=" * 96)
print("A. BASELINE (the decay hurdle)")
print("=" * 96)
print(stats(uvxy["ret"], "UVXY all days (2011-10+)"))
print(stats(vix["ret"], "^VIX all days (2000+)"))
print(stats(vix.loc[uvxy.index.min():, "ret"], "^VIX same era as UVXY"))

print()
print("=" * 96)
print("B. EVENT DAY GRIDS (UVXY mean daily ret / ^VIX mean daily chg by td)")
print("=" * 96)
events = ["fomc_decision", "cpi", "nfp", "opex", "quad_witching",
          "vix_expiry", "jackson_hole"]
for name, df in (("UVXY", uvxy), ("VIX ", vix)):
    off = {ev: td_offset(df.index, ev) for ev in events}
    print(f"\n{name}  td:" + "".join(f"{k:>8d}" for k in range(-3, 4)))
    for ev in events:
        cells = []
        for k in range(-3, 4):
            r = df.loc[off[ev] == k, "ret"].dropna()
            cells.append(f"{r.mean()*1e4:+8.0f}")
        print(f"  {ev:14s}" + "".join(cells))
    print(f"  (t-stats row for |t|>=2 flags)")
    for ev in events:
        cells = []
        for k in range(-3, 4):
            r = df.loc[off[ev] == k, "ret"].dropna()
            t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 3 else np.nan
            cells.append(f"{t:+8.1f}")
        print(f"  {ev:14s}" + "".join(cells))

print()
print("=" * 96)
print("C. EVENT WINDOWS (UVXY, windows in event_sleeve terms)")
print("=" * 96)
fomc = event_dates("fomc_decision")
fomc = fomc[(fomc >= uvxy.index.min()) & (fomc <= uvxy.index.max())]
w_pre = pd.Series([win(uvxy, d, -3, 0) for d in fomc], index=fomc)
print(stats(w_pre, "UVXY pre-FOMC (-3..0) ALL"))
print(stats(w_pre[[d.year % 4 != 2 for d in w_pre.index]],
            "UVXY pre-FOMC ex-midterm (T1 window)"))
print(stats(w_pre[[d.year % 4 == 2 for d in w_pre.index]],
            "UVXY pre-FOMC midterm (T2 window)"))
w_post = pd.Series([win(uvxy, d, 1, 3) for d in fomc], index=fomc)
print(stats(w_post, "UVXY post-FOMC (+1..+3) ALL"))

opex = event_dates("opex")
opex = opex[(opex >= uvxy.index.min()) & (opex <= uvxy.index.max())]
sep = [d for d in opex if d.month == 9]
dec = [d for d in opex if d.month == 12]


def to_eom(df, d):
    idx = df.index
    p = idx.searchsorted(d)
    me = idx.searchsorted(pd.Timestamp(d.year, d.month, 28)
                          + pd.Timedelta(days=4), side="left") - 1
    if p >= len(idx) or me <= p:
        return np.nan
    return float(df["Close"].iloc[me] / df["Close"].iloc[p] - 1)


w = pd.Series([to_eom(uvxy, d) for d in sep], index=sep)
print(stats(w, "UVXY Sep opex -> month-end (T3 window)"))
for d, v in w.dropna().items():
    print(f"      {d.year}: {v*1e4:+8.0f} bps")
w = pd.Series([to_eom(uvxy, d) for d in dec], index=dec)
print(stats(w, "UVXY Dec opex -> year-end (T4 window)"))

cpi = event_dates("cpi")
off_cpi = td_offset(uvxy.index, "cpi")
d0 = uvxy.loc[off_cpi == 0, "ret"]
print(stats(d0, "UVXY CPI day ALL"))
print(stats(d0.loc["2021-06-01":], "UVXY CPI day 2021-06+ (inflation era)"))

vexp = event_dates("vix_expiry")
off_ve = td_offset(uvxy.index, "vix_expiry")
for k in (-2, -1, 0, 1, 2):
    print(stats(uvxy.loc[off_ve == k, "ret"], f"UVXY vix-expiry td{k:+d}"))

print()
print("=" * 96)
print("D. SEASONALITY")
print("=" * 96)
print("\nUVXY mean daily ret by month (2011+), ^VIX mean daily chg by month (2000+):")
for mo in range(1, 13):
    u = uvxy.loc[uvxy.index.month == mo, "ret"]
    v = vix.loc[vix.index.month == mo, "ret"]
    tu = u.mean() / (u.std(ddof=1) / np.sqrt(len(u)))
    print(f"  {mo:2d}: UVXY {u.mean()*1e4:+7.1f} bps (t {tu:+4.1f})   "
          f"VIX {v.mean()*1e4:+7.1f} bps")

print("\nLong-vol seasonal window: UVXY Aug 1 -> Oct 15, per year:")
rows = []
for y in range(2012, 2026):
    a = uvxy.index.searchsorted(pd.Timestamp(y, 8, 1))
    b = uvxy.index.searchsorted(pd.Timestamp(y, 10, 15), side="right") - 1
    if a >= len(uvxy) or b <= a:
        continue
    r = float(uvxy["Close"].iloc[b] / uvxy["Close"].iloc[a - 1] - 1)
    rows.append((y, r))
    mid = " MIDTERM" if y % 4 == 2 else ""
    print(f"  {y}: {r*1e4:+9.0f} bps{mid}")
s = pd.Series([r for _, r in rows])
print(stats(s, "  all years"))
print(stats(pd.Series([r for y, r in rows if y % 4 == 2]), "  midterm only"))

print("\nShort-vol seasonal window: UVXY Nov 1 -> Dec 31 (return shown from"
      " the SHORT side), per year:")
rows = []
for y in range(2011, 2026):
    a = uvxy.index.searchsorted(pd.Timestamp(y, 11, 1))
    b = uvxy.index.searchsorted(pd.Timestamp(y, 12, 31), side="right") - 1
    if a >= len(uvxy) or b <= a:
        continue
    r = float(uvxy["Close"].iloc[b] / uvxy["Close"].iloc[a - 1] - 1)
    rows.append((y, -r))
print("  " + " ".join(f"{y}:{v*1e2:+.0f}%" for y, v in rows))
print(stats(pd.Series([v for _, v in rows]), "  short Nov-Dec all years"))
