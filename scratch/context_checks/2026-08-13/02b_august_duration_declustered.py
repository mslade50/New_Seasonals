"""02 found the wrong thing twice, then something.

  - August Fridays add ~nothing to the plain Friday effect (two-prop z 0.95/0.79)
  - the Aug-14 h5 anchor is 75% up pre-2018 and 50% since, with 73% of the
    return in 2010 and 2011. Dead.
  - but ANY August session ran TLT +0.340% over five at 61.7% up, t=4.27,
    against +0.084% and 53.0% on all sessions.

That t is fake: 535 August sessions with five-day forward windows are ~21
overlapping reads per year. This re-tests it non-overlapping, and asks whether
it survives the state we are actually in, duration pinned at a 52-week low.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, declusters, era_split,  # noqa: E402
                       fwd_ret, sign_test, summarize, cluster_note)

px = close_panel(["TLT", "IEF", "^TNX", "SPY"])


def blk(name, v, dates=None):
    d = summarize(np.asarray(v, float), name)
    up = int((np.asarray(v) > 0).sum())
    n = len(v)
    print(f"{name:<44} n={n:>4}  mean={d['mean_pct']:+7.3f}%  "
          f"med={d['median_pct']:+7.3f}%  up={up}-{n - up} ({100 * up / n:4.1f}%)  "
          f"t={d['t']:+5.2f}  signp={sign_test(up, n):.4f}")
    if dates is not None and n >= 8:
        for e in era_split(dates, np.asarray(v, float)):
            if e["n"]:
                print(f"      era {e['label']}: n={e['n']} mean={e['mean_pct']:+.3f}% "
                      f"hit={e['hit']:.1f}%")
    return d


print("=== A. calendar-month August return, one non-overlapping read per year ===")
for tk in ("TLT", "IEF", "SPY"):
    s = px[tk].dropna()
    s = s[s.index >= "1999-01-01"]
    rows, yrs = [], []
    for y in sorted(set(s.index.year)):
        aug = s[(s.index.year == y) & (s.index.month == 8)]
        prev = s[s.index < aug.index[0]] if len(aug) else None
        if len(aug) < 15 or prev is None or not len(prev):
            continue
        rows.append(aug.iloc[-1] / prev.iloc[-1] - 1.0)
        yrs.append(pd.Timestamp(f"{y}-08-15"))
    blk(f"{tk} full-August return by year", rows, pd.DatetimeIndex(yrs))
    print("      per-year:", {d.year: round(100 * v, 2) for d, v in zip(yrs, rows)})
    print()

print("=== B. all other months, same construction, TLT (is August special?) ===")
s = px["TLT"].dropna()
s = s[s.index >= "1999-01-01"]
by_month = {}
for m in range(1, 13):
    rows = []
    for y in sorted(set(s.index.year)):
        blkm = s[(s.index.year == y) & (s.index.month == m)]
        prev = s[s.index < blkm.index[0]] if len(blkm) else None
        if len(blkm) < 15 or prev is None or not len(prev):
            continue
        rows.append(blkm.iloc[-1] / prev.iloc[-1] - 1.0)
    if rows:
        up = sum(1 for r in rows if r > 0)
        by_month[m] = (len(rows), 100 * float(np.mean(rows)), up)
for m, (n, mu, up) in sorted(by_month.items(), key=lambda kv: -kv[1][1]):
    star = "  <== August" if m == 8 else ""
    print(f"   month {m:>2}  n={n}  mean={mu:+6.2f}%  up={up}-{n - up}{star}")

print("\n=== C. h5 from an August anchor, declustered to 5td minimum gap ===")
for tk in ("TLT", "IEF"):
    s = px[tk].dropna()
    s = s[s.index >= "1999-01-01"]
    r5 = fwd_ret(s, 5)
    aug = s.index[(s.index.month == 8)]
    dc = declusters(aug, 5, s.index)
    v = r5.reindex(dc).dropna()
    blk(f"{tk} August anchors, h5, 5td gap", v.to_numpy(), v.index)
    allr = r5.dropna()
    blk(f"{tk} all sessions, h5 (overlapping baseline)", allr.to_numpy())
    print("   ", cluster_note(v.index, v.to_numpy()))
    print()

print("=== D. the state we are in: duration entering near a 52-week low ===")
print("    tonight TLT +0.82% off its 52w low, IEF +1.23% off its own\n")
for tk in ("TLT", "IEF"):
    s = px[tk].dropna()
    s = s[s.index >= "1999-01-01"]
    r5 = fwd_ret(s, 5)
    low52 = s.rolling(252).min()
    near = (s / low52 - 1.0) <= 0.03
    aug = (s.index.month == 8)
    for nm, msk in [(f"{tk} August + near 52w low", near.to_numpy() & aug),
                    (f"{tk} August, not near the low", ~near.to_numpy() & aug),
                    (f"{tk} near 52w low, any month", near.to_numpy()),
                    (f"{tk} all sessions", np.ones(len(s), dtype=bool))]:
        dates = s.index[msk]
        dc = declusters(dates, 5, s.index)
        v = r5.reindex(dc).dropna()
        if len(v) >= 5:
            blk(nm, v.to_numpy(), v.index)
    print()
