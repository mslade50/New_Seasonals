"""TLT off the session nearest September 4 -- the descriptive year-by-year table.

The context product reported a midterm-only cell: n=6, mean -1.258%, hit 0%
(0-6) for TLT over the five sessions after the session nearest Sept 4. This
script lays out the raw arithmetic behind that shape: every year's return, for
TLT, IEF and SPY, at h=1, h=5 and h=10.

CONVENTION HERE IS LAG 0 AND DESCRIPTIVE, by instruction. The anchor is the
session nearest calendar September 4 in that instrument's own bars, and the
return is close(anchor) -> close(anchor + h). There is no tradeable claim in
this file: the 2026 anchor is Friday 2026-09-04, which has already closed, so
a live version of the cell would have to enter Tuesday 2026-09-08 and would
be measuring a different five sessions.

Requested window is 2003..2025. TLT and IEF both start 2002-07-30, so a
2002 anchor also exists; the 2002-inclusive midterm set is printed separately
because 2003..2025 contains only FIVE midterm years (2006, 2010, 2014, 2018,
2022) while the context product's cell was n=6.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices, sign_test, summarize  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

ASOF = pd.Timestamp("2026-09-04")
TK = ["TLT", "IEF", "SPY"]
raw = load_prices(TK)
close = {t: raw[t]["Close"].dropna() for t in TK}
for t in TK:
    print(f"{t}: bars {close[t].index[0].date()} .. {close[t].index[-1].date()}  "
          f"n={len(close[t])}")


def anchor_for(s, year):
    """Session NEAREST calendar Sept 4 of `year` in this instrument's bars."""
    target = pd.Timestamp(year, 9, 4)
    idx = s.index
    if len(idx) == 0 or target < idx[0] - pd.Timedelta(days=10) or \
            target > idx[-1] + pd.Timedelta(days=10):
        return None
    diffs = np.abs((idx - target).days)
    loc = int(np.argmin(diffs))
    if diffs[loc] > 10:
        return None
    return idx[loc]


def fwd(s, anchor, h):
    idx = s.index
    p = int(idx.get_loc(anchor))
    if p + h >= len(idx):
        return np.nan, None
    return s.iloc[p + h] / s.iloc[p] - 1.0, idx[p + h]


# ------------------------------------------------------------ anchor table
print("\n=== anchor sessions (nearest calendar Sept 4), TLT bars ===")
rows = []
for y in range(2002, 2027):
    a = anchor_for(close["TLT"], y)
    if a is None:
        continue
    exit5 = fwd(close["TLT"], a, 5)[1]
    rows.append({"year": y, "anchor": str(a.date()),
                 "weekday": a.day_name()[:3],
                 "days_from_sep4": int((a - pd.Timestamp(y, 9, 4)).days),
                 "midterm": y % 4 == 2,
                 "exit_close_h5": str(exit5.date()) if exit5 is not None else "n/a"})
print(pd.DataFrame(rows).to_string(index=False))
a26 = anchor_for(close["TLT"], 2026)
print(f"\n2026 anchor session: {a26.date()}  (asof bar {ASOF.date()}; "
      f"equal: {a26 == ASOF})")
print("  the FIRST of the five forward sessions from that anchor is Tuesday "
      "2026-09-08 (Monday 09-07 is Labor Day).")


# --------------------------------------------------------- per-instrument
def table(tkr, h, y0=2003, y1=2025):
    s = close[tkr]
    recs = []
    for y in range(y0, y1 + 1):
        a = anchor_for(s, y)
        if a is None:
            continue
        r, ex = fwd(s, a, h)
        if not np.isfinite(r):
            continue
        recs.append({"year": y, "anchor": str(a.date()),
                     "midterm": y % 4 == 2, "ret_pct": round(100 * r, 3)})
    return pd.DataFrame(recs)


def score(df, label):
    if df.empty:
        print(f"  {label:<30} n=0")
        return
    v = df["ret_pct"].values / 100.0
    st = summarize(v)
    nup = int((v > 0).sum())
    print(f"  {label:<30} n={st['n']:<3} mean={st['mean_pct']:+.3f}%  "
          f"med={st['median_pct']:+.3f}%  {nup}-{len(v) - nup} "
          f"({st['hit']:.1f}%)  t={st['t']:+.2f}  "
          f"sp={sign_test(nup, len(v)):.4f}  worst {st['worst_pct']:+.2f}% "
          f"({df.loc[df.ret_pct.idxmin(), 'year']})  best {st['best_pct']:+.2f}% "
          f"({df.loc[df.ret_pct.idxmax(), 'year']})")


for tkr in TK:
    for h in (1, 5, 10):
        df = table(tkr, h)
        print(f"\n================ {tkr}  h={h}  (lag 0, close(anchor) -> "
              f"close(anchor+{h})), years 2003..2025 ================")
        print(df.to_string(index=False))
        score(df, "ALL years")
        score(df[df.midterm], "MIDTERM (year %% 4 == 2)")
        score(df[~df.midterm], "NON-midterm")
        mid = df[df.midterm]
        non = df[~df.midterm]
        print(f"    midterm years: "
              f"{[(int(r.year), r.ret_pct) for r in mid.itertuples()]}")
        print(f"    non-midterm  : "
              f"{[(int(r.year), r.ret_pct) for r in non.itertuples()]}")

# ------------------------------------------------- the 2002-inclusive midterm
print("\n\n================ RECONCILIATION with the context product's n=6 "
      "midterm cell ================")
print("2003..2025 holds only 5 midterm years. 2002 has TLT/IEF bars, so the "
      "2002..2025 midterm set is n=6.")
for tkr in TK:
    for h in (1, 5, 10):
        df = table(tkr, h, 2002, 2025)
        mid = df[df.midterm]
        if mid.empty:
            continue
        v = mid["ret_pct"].values / 100.0
        st = summarize(v)
        nup = int((v > 0).sum())
        print(f"  {tkr:<4} h={h:<3} midterm 2002..2025  n={st['n']} mean="
              f"{st['mean_pct']:+.3f}%  {nup}-{len(v) - nup} "
              f"({st['hit']:.1f}%)  sp={sign_test(nup, len(v)):.4f}   "
              f"years {[(int(r.year), r.ret_pct) for r in mid.itertuples()]}")
print("\n  context product reported: TLT midterm n=6, mean -1.258%, hit 0% (0-6)")

# The 0-6 record does NOT appear under the "nearest Sept 4" anchor: 2006's
# nearest session is Tue 2006-09-05 (Sept 4 2006 was Labor Day) and it printed
# +0.55%. Grid every plausible anchor/lag/start-year combination so the
# mismatch is a stated fact rather than a guess.
print("\n--- anchor-definition grid, TLT midterm h=5 ---")
tlt = close["TLT"]
tidx = tlt.index


def anchor_mode(year, mode):
    t = pd.Timestamp(year, 9, 4)
    if mode == "on_or_before":
        return tidx[tidx <= t][-1]
    if mode == "on_or_after":
        return tidx[tidx >= t][0]
    return tidx[int(np.argmin(np.abs((tidx - t).days)))]


for mode in ("on_or_before", "on_or_after", "nearest"):
    for lag in (0, 1):
        for y0 in (2002, 2003):
            vals, yrs = [], []
            for y in range(y0, 2026):
                if y % 4 != 2:
                    continue
                p = int(tidx.get_loc(anchor_mode(y, mode)))
                if p + lag + 5 >= len(tidx):
                    continue
                vals.append(100 * (tlt.iloc[p + lag + 5] / tlt.iloc[p + lag] - 1))
                yrs.append(y)
            v = np.asarray(vals)
            print(f"  {mode:<13} lag{lag} start{y0}  n={len(v)}  "
                  f"mean={v.mean():+.3f}%  hit={100 * (v > 0).mean():.1f}%  "
                  f"{[(y, round(x, 3)) for y, x in zip(yrs, vals)]}")
print("  -> the 0-6 record reproduces ONLY under 'last session on or before "
      "Sept 4'; no variant reproduces -1.258% exactly on these bars.")

# --------------------------------------------------------------- all-days base
print("\n\n=== all-days base rates on the same lag-0 form (whole history) ===")
for tkr in TK:
    s = close[tkr]
    for h in (1, 5, 10):
        f = (s.shift(-h) / s - 1.0).dropna()
        print(f"  {tkr:<4} h={h:<3} n={len(f):<5} mean={100 * f.mean():+.3f}%  "
              f"hit {100 * (f > 0).mean():.1f}%")

# ----------------------------------------------------- September-wide context
print("\n\n=== every September session as a control (lag-0 h=5) ===")
for tkr in TK:
    s = close[tkr]
    f = (s.shift(-5) / s - 1.0).dropna()
    sep = f[f.index.month == 9]
    sep_mid = sep[[d.year % 4 == 2 for d in sep.index]]
    print(f"  {tkr:<4} all Sept sessions n={len(sep):<5} "
          f"mean={100 * sep.mean():+.3f}% hit {100 * (sep > 0).mean():.1f}%   |   "
          f"midterm Septembers n={len(sep_mid):<4} "
          f"mean={100 * sep_mid.mean():+.3f}% hit {100 * (sep_mid > 0).mean():.1f}%")

print("\nDONE.")
