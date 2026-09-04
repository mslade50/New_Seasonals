"""Two calendar cuts pointed the same way for the dollar:

  a) August Fridays, EURUSD 35-58 down, sign p 0.0198
  b) the Aug-14 anchor, EURUSD 16-6 down all years (p 0.0262), 5-0 down in
     midterm years (p 0.0312)

A 5-0 midterm record on n=5 is corroboration or it is nothing. The 08-10 brief
killed a seasonal of exactly this shape with an anchor walk: if only the anchor
the calendar happens to land on clears p 0.10 and its eight neighbours do not,
the cell is the search, not the season. Same test here, plus the DXY leg, plus
the Friday control from drill 01.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, era_split, fwd_ret,  # noqa: E402
                       sign_test, summarize)

px = close_panel(["EURUSD=X", "DX-Y.NYB", "JPY=X"])
px = px[px.index >= "1999-01-01"]


def rep(name, s, dates, h=1, show=True):
    r = fwd_ret(s, h).reindex(pd.DatetimeIndex(dates)).dropna()
    if len(r) < 3:
        return None
    d = summarize(r.to_numpy(), name)
    down = int((r < 0).sum())
    p = sign_test(down, len(r))
    if show:
        print(f"   {name:<44} n={len(r):>4} mean={d['mean_pct']:+6.3f}% "
              f"med={d['median_pct']:+6.3f}% down={down}-{len(r) - down} "
              f"({100 * down / len(r):4.1f}%) t={d['t']:+5.2f} signp={p:.4f}")
    return {"r": r, "down": down, "n": len(r), "d": d, "p": p}


def anchors_for(s, day, month=8, per_year=True):
    idx = s.index
    m = (idx.month == month) & (np.abs(idx.day - day) <= 2)
    if not per_year:
        return idx[m]
    seen, keep = set(), []
    for dt in idx[m]:
        if dt.year not in seen:
            seen.add(dt.year)
            keep.append(dt)
    return pd.DatetimeIndex(keep)


for tk in ("EURUSD=X", "DX-Y.NYB"):
    s = px[tk].dropna()
    print(f"\n================ {tk} ================")
    print(f"   history {s.index[0].date()} .. {s.index[-1].date()}, {len(s)} sessions")

    print("\n   --- the anchor walk: Aug 8 through Aug 20, h1 ---")
    hits = []
    for day in range(8, 21):
        a = anchors_for(s, day)
        r = rep(f"Aug-{day} anchor", s, a)
        if r:
            hits.append((day, r["p"], 100 * r["down"] / r["n"], r["d"]["mean_pct"]))
    clear = [h for h in hits if h[1] < 0.10]
    print(f"   anchors clearing sign p 0.10: {[h[0] for h in clear]} of "
          f"{len(hits)} tested")

    print("\n   --- the midterm slice at the Aug-14 anchor ---")
    a14 = anchors_for(s, 14)
    mid = pd.DatetimeIndex([d for d in a14 if d.year % 4 == 2])
    non = pd.DatetimeIndex([d for d in a14 if d.year % 4 != 2])
    rep("Aug-14, midterm years only", s, mid)
    rep("Aug-14, non-midterm years", s, non)
    rep("Aug-14, all years", s, a14)

    print("\n   --- midterm Augusts generally, is it the year not the day ---")
    idx = s.index
    aug_mid = idx[(idx.month == 8) & (idx.year % 4 == 2)]
    aug_non = idx[(idx.month == 8) & (idx.year % 4 != 2)]
    rep("any August session, midterm years", s, aug_mid)
    rep("any August session, other years", s, aug_non)

    print("\n   --- the Friday leg and its control (drill 01's question) ---")
    dow, month = idx.dayofweek, idx.month
    af = idx[(month == 8) & (dow == 3)]
    of = idx[(month != 8) & (dow == 3)]
    aa = rep("August Fridays", s, af)
    oo = rep("Fridays outside August", s, of)
    rep("every session", s, idx)
    if aa and oo:
        p1, n1 = aa["down"] / aa["n"], aa["n"]
        p2, n2 = oo["down"] / oo["n"], oo["n"]
        pp = (aa["down"] + oo["down"]) / (n1 + n2)
        z = (p1 - p2) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
        print(f"   two-proportion z, August Fridays vs other Fridays: {z:+.2f}")
        for e in era_split(aa["r"].index, aa["r"].to_numpy()):
            if e["n"]:
                print(f"      era {e['label']}: n={e['n']} mean={e['mean_pct']:+.3f}% "
                      f"hit={e['hit']:.1f}%")

    print("\n   --- tomorrow is BOTH: an August Friday at the Aug-14 anchor ---")
    both = pd.DatetimeIndex(sorted(set(af) & set(anchors_for(s, 14, per_year=False))))
    rep("August Friday within 2 days of Aug 14", s, both)
