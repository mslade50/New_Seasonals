"""C9 -- Jackson Hole x US large equity (SPY).

Prior is strongly negative: 2026-08-13 found the JH offset ladder is a PLATEAU
on rates/gold/FX, i.e. the anchor is August month position wearing an event
label. Order of operations per the brief:
  1. build the anchor set with the searchsorted end-guard
  2. OFFSET LADDER FIRST (entry sessions -10..+5 around the symposium)
  3. August month-position control (same trading days, all years, no event)
  4. midterm split
  5. only then the conditional mean / N / sign test
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["SPY"])
spy = px["SPY"].dropna()
dates = spy.index
n = len(dates)
print(f"SPY {dates[0].date()} .. {dates[-1].date()}  n={n}")

ev = load_events(["jackson_hole"])
print("\njackson_hole rows in the events file:")
print(ev[["date", "detail"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 1. anchor set, with the end-guard the brief demands
# ---------------------------------------------------------------------------
anchors = []  # (year, jh_date, loc of last session <= jh)
skipped = []
for _, r in ev.iterrows():
    d = pd.Timestamp(r["date"])
    loc = int(np.searchsorted(dates.values, np.datetime64(d), side="right")) - 1
    if loc < 0 or loc >= n:
        skipped.append((d.date(), "loc out of range"))
        continue
    # a JH date past the end of the price index must NOT mint a fake anchor
    if d > dates[-1]:
        skipped.append((d.date(), "event past end of price index"))
        continue
    anchors.append((d.year, d, loc))
print(f"\nusable anchors: {len(anchors)}  skipped: {skipped}")
print("anchor sessions:", ", ".join(f"{y}:{dates[l].date()}" for y, _, l in anchors))

years = np.array([y for y, _, _ in anchors])
locs = np.array([l for _, _, l in anchors])
midterm = (years % 4) == 2

# ---------------------------------------------------------------------------
# 2. OFFSET LADDER.  entry at close[loc+k], exit at close[loc+k+h].
#    the LIVE trade is k=-7 (entry 2026-08-18 = 7 sessions before the 08-28
#    symposium session; signal close 08-17, MOC entry 08-18).
# ---------------------------------------------------------------------------
c = spy.values


def offset_ret(k: int, h: int, mask=None):
    out, ys = [], []
    sel = np.ones(len(locs), bool) if mask is None else mask
    for y, l, m in zip(years, locs, sel):
        if not m:
            continue
        e, x = l + k, l + k + h
        if e < 0 or x >= n:
            continue
        out.append(c[x] / c[e] - 1.0)
        ys.append(y)
    return np.array(out), np.array(ys)


for h in (5, 7, 10):
    rows = []
    for k in range(-10, 6):
        v, _ = offset_ret(k, h)
        r = summarize(v, f"k={k:+d}{'  <-- TRUE ANCHOR' if k == -7 else ''}")
        rows.append(r)
    show(rows, f"2. OFFSET LADDER, h={h} td (entry = symposium session + k)")
    means = np.array([r.get("mean_pct", np.nan) for r in rows])
    true_i = list(range(-10, 6)).index(-7)
    rank = int((means > means[true_i]).sum()) + 1
    print(f"  true anchor k=-7 ranks {rank} of {len(means)} offsets "
          f"(mean {means[true_i]:+.3f}%, ladder mean {np.nanmean(means):+.3f}%, "
          f"ladder sd {np.nanstd(means):.3f}%, "
          f"positive offsets {int((means > 0).sum())}/{len(means)})")

# ---------------------------------------------------------------------------
# 3. AUGUST MONTH-POSITION CONTROL: same trading days of August, ALL years,
#    no event condition.
# ---------------------------------------------------------------------------
tdom = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).cumcount() + 1
tdom = pd.Series(tdom.values, index=dates)
live_entry_tdom = None
# entry session in 2026 is 2026-08-18
try:
    live_entry_tdom = int(tdom.loc[pd.Timestamp("2026-08-18")])
except KeyError:
    live_entry_tdom = int(tdom.loc[dates[dates <= pd.Timestamp("2026-08-18")][-1]]) + 1
print(f"\nlive entry session 2026-08-18 has August tdom = {live_entry_tdom}")
print("anchor-year entry tdoms (k=-7):",
      {int(y): int(tdom.iloc[l - 7]) for y, l in zip(years, locs) if l - 7 >= 0})

for h in (5, 7, 10):
    aug = (dates.month == 8)
    rows = []
    for lo, hi in [(live_entry_tdom, live_entry_tdom), (10, 16), (6, 16), (1, 23)]:
        m = aug & (tdom.values >= lo) & (tdom.values <= hi)
        idx = np.where(m)[0]
        idx = idx[idx + h < n]
        v = c[idx + h] / c[idx] - 1.0
        rows.append(summarize(v, f"Aug tdom {lo}-{hi} ALL YEARS (N days)"))
    v_jh, _ = offset_ret(-7, h)
    rows.append(summarize(v_jh, f"JH anchor k=-7 (N={len(v_jh)})"))
    show(rows, f"3. August month-position control, h={h}")

# ---------------------------------------------------------------------------
# 4. midterm split
# ---------------------------------------------------------------------------
for h in (5, 7, 10):
    v_all, ys = offset_ret(-7, h)
    mm = (ys % 4) == 2
    show([summarize(v_all[mm], f"midterm (N={int(mm.sum())})"),
          summarize(v_all[~mm], f"non-midterm (N={int((~mm).sum())})")],
         f"4. midterm split, JH anchor k=-7, h={h}")
    print("  midterm years:", sorted(ys[mm].tolist()))

# ---------------------------------------------------------------------------
# 5. the conditional cell itself
# ---------------------------------------------------------------------------
for h in (5, 7, 10):
    v, ys = offset_ret(-7, h)
    w = int((v > 0).sum())
    # unconditional SPY hit rate over the same holding length
    base = c[h:] / c[:-h] - 1.0
    p0 = float((base > 0).mean())
    print(f"\n5. h={h}: N={len(v)} mean={100*v.mean():+.3f}% median="
          f"{100*np.median(v):+.3f}% hit={100*w/len(v):.1f}% "
          f"record {w}-{len(v)-w}  sign p (vs SPY's own {100*p0:.1f}% base) = "
          f"{sign_test(w, len(v), p0):.4f}   worst {100*v.min():+.2f}% "
          f"({ys[int(np.argmin(v))]})")
    print("   per-year:", ", ".join(f"{int(y)}:{100*x:+.2f}%" for y, x in zip(ys, v)))
