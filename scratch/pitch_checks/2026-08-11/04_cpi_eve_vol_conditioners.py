"""Does today's STATE sit in the good bucket of the CPI-eve vol cell?

Three conditioners nobody else was briefed on, each one a way today could be
unrepresentative of the sample:

  1. CYCLE YEAR. 2026 is a midterm year and the seasonal board's standing
     prior is de-risk (book win 56.4% vs 64.9% all-years). The 2026-08-07 run
     died on exactly this cut, so it gets run before anything ships.
  2. VIX LEVEL going in. VIX closed 15.46 at a 63d rank of 28.6. A crush
     measured from elevated vol is a different trade from one entered at the
     bottom of the range: there is less premium to release.
  3. MONTH. It is August, and the sample spans every month.

Measured on ^VIX (N=317 back to 2000, no vehicle confound, no leverage break)
and on SVXY (the tradeable, 2011+). Entry is the CPI eve: mask two sessions
before the print, lag=1 puts the MOC entry on the eve, h counts from there.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, load_events, fwd_lag, declusters, summarize, sign_test  # noqa: E402

warnings.filterwarnings("ignore")

px = close_panel(["^VIX", "SVXY", "SPY"])
idx = px.index
ev = load_events(["cpi"])
cpi = pd.DatetimeIndex(sorted(ev.loc[ev["event"] == "cpi", "date"].unique()))

# eve entry -> mask sits 2 sessions before the print
mask = []
for d in cpi:
    loc = idx.searchsorted(d)
    if 0 <= loc - 2 and loc < len(idx):
        mask.append(idx[loc - 2])
mask = declusters(pd.DatetimeIndex(sorted(set(mask))), 5, idx)
print(f"CPI-eve anchors: {len(mask)}  ({mask[0].date()} -> {mask[-1].date()})\n")

vix = px["^VIX"].dropna()
vix_pct = vix.rolling(252).apply(lambda w: (w[-1] > w[:-1]).mean() * 100, raw=True)
vix_rank63 = vix.rolling(63).apply(lambda w: (w[-1] > w[:-1]).mean() * 100, raw=True)


def cell(s, dates, h, label):
    v = fwd_lag(s, h, lag=1).reindex(dates).dropna()
    if len(v) < 4:
        return f"{label:<34} N={len(v):<4} -- too few"
    st = summarize(v.values)
    p_up = sign_test(int((v.values > 0).sum()), len(v))
    return (f"{label:<34} N={st['n']:<4} mean={st['mean_pct']:+7.3f} "
            f"med={st['median_pct']:+7.3f} up={st['hit']:5.1f}% "
            f"p(up)={p_up:.4f} worst={st['worst_pct']:+8.2f}")


print("=" * 96)
print("1. CYCLE YEAR  (2026 is midterm, year %% 4 == 2)")
print("=" * 96)
for name, s in (("^VIX", vix), ("SVXY", px["SVXY"].dropna())):
    m = mask[mask.isin(s.index)]
    mid = m[(m.year % 4) == 2]
    non = m[(m.year % 4) != 2]
    for h in (1, 3, 5):
        print(cell(s, m,   h, f"{name} h={h} ALL"))
        print(cell(s, mid, h, f"{name} h={h}   midterm (TODAY)"))
        print(cell(s, non, h, f"{name} h={h}   non-midterm"))
    print()

print("=" * 96)
print("2. VIX LEVEL GOING IN  (today: VIX 15.46, 63d rank 28.6, 252d pctile below)")
print("=" * 96)
today_pct = vix_pct.iloc[-1]
today_r63 = vix_rank63.iloc[-1]
print(f"today's ^VIX 252d percentile = {today_pct:.1f}, 63d rank = {today_r63:.1f}\n")
for name, s in (("^VIX", vix), ("SVXY", px["SVXY"].dropna())):
    m = mask[mask.isin(s.index)]
    p = vix_pct.reindex(m)
    lo = m[(p <= 33).values]
    mi = m[((p > 33) & (p <= 66)).values]
    hi = m[(p > 66).values]
    for h in (3,):
        print(cell(s, lo, h, f"{name} h={h} VIX pctile <=33 (TODAY-ish)"))
        print(cell(s, mi, h, f"{name} h={h} VIX pctile 33-66"))
        print(cell(s, hi, h, f"{name} h={h} VIX pctile >66"))
    print()

print("=" * 96)
print("3. MONTH (today is August)")
print("=" * 96)
for name, s in (("^VIX", vix), ("SVXY", px["SVXY"].dropna())):
    m = mask[mask.isin(s.index)]
    rows = []
    for mo in range(1, 13):
        mm = m[m.month == mo]
        v = fwd_lag(s, 3, lag=1).reindex(mm).dropna()
        if len(v) < 4:
            continue
        st = summarize(v.values)
        rows.append((mo, st["n"], st["mean_pct"], st["hit"]))
    print(f"-- {name} h=3 by month --")
    for mo, n, mean, hit in rows:
        star = "  <-- AUGUST" if mo == 8 else ""
        print(f"   month {mo:>2}  N={n:<3} mean={mean:+7.3f}  up={hit:5.1f}%{star}")
    print()

print("=" * 96)
print("4. SPY CONTROL on the same anchor and buckets (is any of this direction?)")
print("=" * 96)
spy = px["SPY"].dropna()
m = mask[mask.isin(spy.index)]
for h in (1, 3, 5):
    print(cell(spy, m, h, f"SPY h={h} CPI eve"))
    print(cell(spy, spy.index, h, f"SPY h={h} ALL DAYS (drift)"))
mid = m[(m.year % 4) == 2]
print(cell(spy, mid, 3, "SPY h=3 CPI eve, midterm only"))
