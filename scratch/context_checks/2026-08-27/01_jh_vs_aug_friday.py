"""Jackson Hole day-of vs the August-Friday cell it is nested inside.

26 of 27 symposium dates in macro_events.csv are Fridays and all 27 are in
August, so the engine's E:jackson_hole k1 table and the bare E:weekday_month
"Fridays in August" table are computed on overlapping days. The symposium
cell cannot publish until it is shown to be something the calendar slot does
not already give you.

Anchor convention: the session BEFORE the symposium date, so h1 is the
symposium session itself. lag=0 close-to-close, matching the context engine.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, fwd_ret, summarize, show, sign_test,
    era_split, cluster_note,
)

SUBJECTS = ["^VIX", "IWM", "CL=F", "SPY", "HG=F"]
px = close_panel(SUBJECTS)
dates = px.index

ev = load_events(["jackson_hole"])
jh = pd.DatetimeIndex(ev["date"])
print(f"symposium dates: {len(jh)}, "
      f"weekday counts {pd.Series(jh.dayofweek).value_counts().to_dict()} "
      "(4=Friday)")
print(f"months {sorted(set(jh.month))}")

pos = pd.Series(range(len(dates)), index=dates)

# anchor = last session strictly before the symposium date
anchors, anchor_of = [], {}
for d in jh:
    prior = dates[dates < d]
    if len(prior) == 0:
        continue
    a = prior[-1]
    anchors.append(a)
    anchor_of[a] = d
anchors = pd.DatetimeIndex(sorted(anchors))
print(f"anchors resolved: {len(anchors)} "
      f"({anchors.min().date()} .. {anchors.max().date()})")

# the nesting control: every session that is the last one before an August
# Friday, symposium or not. That is the slot, stripped of the Fed.
aug_fri_anchor = []
for i, d in enumerate(dates[:-1]):
    nxt = dates[i + 1]
    if nxt.month == 8 and nxt.dayofweek == 4:
        aug_fri_anchor.append(d)
aug_fri_anchor = pd.DatetimeIndex(aug_fri_anchor)
non_jh_aug_fri = aug_fri_anchor.difference(anchors)
print(f"August-Friday anchors: {len(aug_fri_anchor)}, "
      f"of which non-symposium: {len(non_jh_aug_fri)}")

# no-chair-speech years, from the event note
NO_SPEECH = {2013, 2015}


def cell(sub, idx, h=1):
    f = fwd_ret(px[sub], h)
    v = f.reindex(idx).dropna()
    return v.index, v.values


for sub in SUBJECTS:
    print(f"\n{'='*72}\n{sub}\n{'='*72}")
    rows = []

    d_jh, v_jh = cell(sub, anchors)
    up = int((v_jh > 0).sum())
    rows.append(summarize(v_jh, "symposium session (h1)"))

    d_af, v_af = cell(sub, non_jh_aug_fri)
    rows.append(summarize(v_af, "Aug Friday, no symposium"))

    # all Fridays outside August, and all days: the two wider controls
    fri_anchor = pd.DatetimeIndex(
        [d for i, d in enumerate(dates[:-1])
         if dates[i + 1].dayofweek == 4 and dates[i + 1].month != 8])
    d_f, v_f = cell(sub, fri_anchor)
    rows.append(summarize(v_f, "Friday, not August"))

    d_all, v_all = cell(sub, dates)
    rows.append(summarize(v_all, "all days"))
    show(rows, f"{sub} h1 by cell")

    # the decomposition that decides it
    if sub == "^VIX":
        w = int((v_jh < 0).sum())
        print(f"  symposium record {w}-{len(v_jh)-w} DOWN, "
              f"sign p {sign_test(w, len(v_jh)):.4f}")
        wa = int((v_af < 0).sum())
        print(f"  non-symposium Aug Friday {wa}-{len(v_af)-wa} DOWN, "
              f"sign p {sign_test(wa, len(v_af)):.4f}")
    else:
        print(f"  symposium record {up}-{len(v_jh)-up} UP, "
              f"sign p {sign_test(up, len(v_jh)):.4f}")
        ua = int((v_af > 0).sum())
        print(f"  non-symposium Aug Friday {ua}-{len(v_af)-ua} UP, "
              f"sign p {sign_test(ua, len(v_af)):.4f}")

    diff = np.mean(v_jh) - np.mean(v_af)
    # Welch t on the difference of the two cells
    s1, s2 = np.var(v_jh, ddof=1), np.var(v_af, ddof=1)
    n1, n2 = len(v_jh), len(v_af)
    se = np.sqrt(s1 / n1 + s2 / n2)
    print(f"  symposium MINUS Aug-Friday-control: {100*diff:+.3f}pp, "
          f"Welch t {diff/se:+.2f}")

    print("  era:", [(r["label"], r["n"], round(r["mean_pct"], 3),
                      round(r["hit"], 1)) for r in era_split(d_jh, v_jh)])
    print("  concentration:", cluster_note(d_jh, v_jh))

    # midterm years and the no-speech years
    yrs = pd.DatetimeIndex(d_jh).year
    mid = np.array([y % 4 == 2 for y in yrs])
    if mid.sum():
        r = summarize(v_jh[mid], "midterm symposium")
        k = int((v_jh[mid] > 0).sum())
        print(f"  midterm: n={r['n']} mean {r['mean_pct']:+.3f}% "
              f"record {k}-{r['n']-k} up sign p {sign_test(k, r['n']):.4f}")
    nos = np.array([y in NO_SPEECH for y in yrs])
    if nos.sum():
        print(f"  no-chair-speech years {sorted(set(yrs[nos]))}: "
              f"{[round(100*x, 2) for x in v_jh[nos]]}")

    # follow-on: what the session after the symposium did
    d2, v2 = cell(sub, anchors, h=2)
    nxt = v2[:len(v_jh)] - v_jh[:len(v2)]
    k = int((nxt > 0).sum())
    print(f"  session AFTER the symposium: n={len(nxt)} "
          f"mean {100*np.mean(nxt):+.3f}% record {k}-{len(nxt)-k} up")

# per-episode detail for the headline subject
print(f"\n{'='*72}\nsymposium sessions, ^VIX and IWM, per year\n{'='*72}")
fv = fwd_ret(px["^VIX"], 1).reindex(anchors)
fi = fwd_ret(px["IWM"], 1).reindex(anchors)
fc = fwd_ret(px["CL=F"], 1).reindex(anchors)
for a in anchors:
    print(f"  {anchor_of[a].date()} (anchor {a.date()}): "
          f"VIX {100*fv.get(a, np.nan):+7.2f}%  "
          f"IWM {100*fi.get(a, np.nan):+6.2f}%  "
          f"CL {100*fc.get(a, np.nan):+6.2f}%")
