"""Month-end bond bid, re-cut from the engine's cell to the one that matches
Monday, and crossed with the state the bond market is actually in.

The engine cell is "next session is one of the month's last 3" (IEF t 5.55,
TLT t 4.38, ^TNX t -4.37, all BH pass, all era stable). Monday is the LAST
session, not one of the last three, so the publishable cell is narrower than
what fired.

Three questions:
  1. Does the effect survive the narrow cut, and is the final session where
     it lives or is it diluted by the two sessions before it?
  2. August specifically. Month-end index extension is a mechanical monthly
     event, so the August sub-cell should NOT be special. If it is, that is
     evidence the whole thing is noise.
  3. The cross. IEF sits 0.73% off its 52-week low, TLT 1.88%, and the 5-year
     yield printed a 52-week high on Friday. Does the bid still show up when
     bonds enter month end beaten down, or is it a momentum artifact that
     only appears when they were already rallying?

Convention: anchor is the second-to-last session of the month, so h=1 is the
final session's own close-to-close move. That is the engine's event-lane
anchor rule and it makes h=1 exactly Monday.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, summarize, show, sign_test, era_split, cluster_note,
)

SUBJECTS = ["TLT", "IEF", "HYG", "SPY", "^TNX", "^FVX"]
px = close_panel(SUBJECTS + ["^GSPC"])

# US equity-session calendar. Every subject here trades on it.
cal = px["^GSPC"].dropna().index
ym = pd.Series(cal.year * 100 + cal.month, index=cal)

# position of each month's LAST session, and of the session before it
# August 2026 is NOT over: Monday is still to come. Treating Friday as its
# final session would inject a fake month-end observation into every cell
# below, and into exactly the low-proximity bucket the brief wants to quote.
# A month counts only once a later month has printed.
COMPLETE = sorted(set(ym.values))[:-1]

last_pos = {}
for key, grp in ym.groupby(ym.values):
    if key not in COMPLETE:
        continue
    last_pos[key] = len(cal) - 1 - list(cal[::-1]).index(grp.index[-1])

pos_of = pd.Series(range(len(cal)), index=cal)
finals = sorted(last_pos.values())
# anchor = the session before the final one; drop the very first month
anchors = [(cal[p - 1], cal[p]) for p in finals if p >= 1]

# also build the two earlier sessions of the "last 3" window for contrast
third_last = [(cal[p - 3], cal[p - 2]) for p in finals if p >= 3]
second_last = [(cal[p - 2], cal[p - 1]) for p in finals if p >= 2]


def leg(sub: str, pairs) -> tuple[pd.DatetimeIndex, np.ndarray]:
    s = px[sub].dropna()
    ds, vs = [], []
    for a, b in pairs:
        if a in s.index and b in s.index:
            ds.append(a)
            vs.append(s.loc[b] / s.loc[a] - 1.0)
    return pd.DatetimeIndex(ds), np.asarray(vs, dtype=float)


print("=" * 78)
print("Q1. the narrow cut: the FINAL session of the month, h=1 from the")
print("    second-to-last close. Contrast against the other two sessions of")
print("    the engine's last-3 window, and against all other sessions.")
print("=" * 78)
for sub in ["IEF", "TLT", "HYG", "^TNX", "SPY"]:
    rows = []
    s = px[sub].dropna()
    allret = (s / s.shift(1) - 1.0).dropna()
    for label, pairs in [("final session", anchors),
                         ("2nd-to-last", second_last),
                         ("3rd-to-last", third_last)]:
        d, v = leg(sub, pairs)
        if len(v) == 0:
            continue
        r = summarize(v, label)
        up = int((v > 0).sum())
        r["record"] = f"{up}-{len(v) - up}"
        r["sign_p"] = round(sign_test(up, len(v)), 4)
        rows.append(r)
    base = summarize(allret.values, "all sessions")
    base["record"] = ""
    base["sign_p"] = np.nan
    rows.append(base)
    show(rows, f"{sub}: where in the month-end window the move sits")

print()
print("=" * 78)
print("Q2. is August special? It should not be. Index extension is monthly.")
print("=" * 78)
for sub in ["IEF", "TLT"]:
    d, v = leg(sub, anchors)
    rows = []
    for m in range(1, 13):
        mm = d.month == m
        if mm.sum() < 5:
            continue
        r = summarize(v[mm], pd.Timestamp(2000, m, 1).strftime("%b"))
        up = int((v[mm] > 0).sum())
        r["record"] = f"{up}-{int(mm.sum()) - up}"
        rows.append(r)
    show(rows, f"{sub}: final session of each month")

print()
print("=" * 78)
print("Q3. the cross: does the bid survive bonds entering month end beaten")
print("    down? Split the anchors on the subject's distance from its own")
print("    trailing-252 low at the anchor close. Today IEF is 0.73% off it,")
print("    TLT 1.88%.")
print("=" * 78)
for sub in ["IEF", "TLT"]:
    s = px[sub].dropna()
    low252 = s.rolling(252, min_periods=200).min()
    d, v = leg(sub, anchors)
    dist = np.array([100 * (s.loc[a] / low252.loc[a] - 1.0)
                     if a in low252.index and np.isfinite(low252.loc[a])
                     else np.nan for a in d])
    ok = np.isfinite(dist)
    rows = []
    for label, m in [("within 3% of 52w low", ok & (dist <= 3.0)),
                     ("3-10% above", ok & (dist > 3.0) & (dist <= 10.0)),
                     (">10% above", ok & (dist > 10.0))]:
        if m.sum() < 5:
            continue
        r = summarize(v[m], label)
        up = int((v[m] > 0).sum())
        r["record"] = f"{up}-{int(m.sum()) - up}"
        r["sign_p"] = round(sign_test(up, int(m.sum())), 4)
        rows.append(r)
    show(rows, f"{sub}: final session by entry distance from the 52w low")
    m = ok & (dist <= 3.0)
    if m.sum() >= 5:
        print("  era:", [(r["label"], r["n"], round(r["mean_pct"], 3),
                          round(r["hit"], 1)) for r in era_split(d[m], v[m])])
        print("  conc:", cluster_note(d[m], v[m]))
        print("  years:", sorted({x.year for x in d[m]}))

print()
print("=" * 78)
print("Q3b. the other half of the cross: the 5-year yield printed a 52-week")
print("     high on Friday. Split month-end finals on whether ^FVX was at or")
print("     near a trailing-252 high at the anchor.")
print("=" * 78)
fvx = px["^FVX"].dropna()
hi252 = fvx.rolling(252, min_periods=200).max()
for sub in ["IEF", "TLT"]:
    d, v = leg(sub, anchors)
    near = np.array([100 * (fvx.loc[a] / hi252.loc[a] - 1.0)
                     if a in hi252.index and np.isfinite(hi252.loc[a])
                     else np.nan for a in d])
    ok = np.isfinite(near)
    rows = []
    for label, m in [("5y yield within 1% of 52w high", ok & (near >= -1.0)),
                     ("5y yield 1-10% below", ok & (near < -1.0) & (near >= -10.0)),
                     ("5y yield >10% below", ok & (near < -10.0))]:
        if m.sum() < 5:
            continue
        r = summarize(v[m], label)
        up = int((v[m] > 0).sum())
        r["record"] = f"{up}-{int(m.sum()) - up}"
        r["sign_p"] = round(sign_test(up, int(m.sum())), 4)
        rows.append(r)
    show(rows, f"{sub}: final session by 5y-yield position")

print()
print("=" * 78)
print("Q4. today's readings, for the brief's setup sentence")
print("=" * 78)
for sub in ["IEF", "TLT", "HYG"]:
    s = px[sub].dropna()
    lo = s.rolling(252, min_periods=200).min().iloc[-1]
    hi = s.rolling(252, min_periods=200).max().iloc[-1]
    print(f"  {sub:5s} last {s.iloc[-1]:8.2f}  {100*(s.iloc[-1]/lo-1):+6.2f}% off 52w low"
          f"  {100*(s.iloc[-1]/hi-1):+6.2f}% off 52w high")
print(f"  ^FVX  last {fvx.iloc[-1]:8.3f}  52w high {hi252.iloc[-1]:.3f}"
      f"  ({100*(fvx.iloc[-1]/hi252.iloc[-1]-1):+.2f}% from it)")
print(f"  anchor session (today's close) = {cal[-1].date()}, "
      f"final session of August 2026 = next US session")

print()
print("=" * 78)
print("Q5. era stability and concentration for the headline cell itself")
print("    (final session, all months), plus the equity contrast.")
print("=" * 78)
for sub in ["IEF", "TLT", "^TNX", "SPY"]:
    d, v = leg(sub, anchors)
    print(f"  {sub}:")
    for r in era_split(d, v):
        up = None
        print(f"    {r['label']:10s} n={r['n']:4d} mean={r['mean_pct']:+.3f}% "
              f"hit={r['hit']:.1f}% t={r['t']:+.2f}")
    print("    conc:", cluster_note(d, v))
    # decade-by-decade, to show it is not one regime
    dec = pd.Series(v).groupby((pd.DatetimeIndex(d).year // 5) * 5).agg(
        ["count", "mean", lambda x: (x > 0).mean()])
    dec.columns = ["n", "mean", "hit"]
    dec["mean"] = (100 * dec["mean"]).round(3)
    dec["hit"] = (100 * dec["hit"]).round(1)
    print("    by 5y block:", dec.to_dict("index"))
