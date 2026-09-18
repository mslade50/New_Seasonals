"""Post-Labor-Day specifically, against the generic post-holiday cell.

The sweep's strongest cell is E:holiday_post pooled across every market
closure. Tuesday is one specific holiday. Two questions:
  1. does the pooled result survive when restricted to Labor Day?
  2. how much of the Sep-08 seasonal_doy cell IS the post-Labor-Day session?
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, fwd_ret, summarize, sign_test, era_split, cluster_note  # noqa

TK = ["^GSPC", "SPY", "^VIX", "QQQ", "IWM", "^TNX", "TLT", "DX-Y.NYB"]
px = close_panel(TK)
idx = px.index

# --- Labor Day = first Monday of September. Post-LD session = first bar after it.
ld_anchor, ld_post = [], []
for yr in sorted(set(idx.year)):
    sept = pd.Timestamp(yr, 9, 1)
    lab = sept + pd.Timedelta(days=(7 - sept.weekday()) % 7)   # first Monday
    after = idx[idx > lab]
    before = idx[idx < lab]
    if len(after) == 0 or len(before) == 0:
        continue
    post = after[0]
    if (post - lab).days > 6:      # no session near it, bad data year
        continue
    ld_post.append(post)
    ld_anchor.append(before[-1])
ld_anchor = pd.DatetimeIndex(ld_anchor)
ld_post = pd.DatetimeIndex(ld_post)
print(f"Labor Day post-sessions found: {len(ld_post)}  "
      f"{ld_post[0].date()} .. {ld_post[-1].date()}")
print("  weekday of post session:",
      pd.Series([d.day_name() for d in ld_post]).value_counts().to_dict())

# --- every post-holiday session (weekday gap in the NYSE index)
gap = pd.Series(idx, index=idx).diff().dt.days
wk = pd.Series([d.weekday() for d in idx], index=idx)
# a session preceded by a gap that skipped at least one weekday
holpost = []
for i in range(1, len(idx)):
    prev, cur = idx[i - 1], idx[i]
    span = pd.bdate_range(prev + pd.Timedelta(days=1), cur - pd.Timedelta(days=1))
    if len(span) > 0:
        holpost.append(cur)
holpost = pd.DatetimeIndex(holpost)
hol_anchor = pd.DatetimeIndex([idx[idx.get_loc(d) - 1] for d in holpost])
print(f"All post-holiday sessions: {len(holpost)}")

# --- Sep-08 doy overlap: how many of the 26 doy anchors are post-Labor-Day?
doy_post = []
for yr in sorted(set(idx.year)):
    c = idx[(idx.year == yr) & (idx.month == 9)]
    c = c[(c >= pd.Timestamp(yr, 9, 6)) & (c <= pd.Timestamp(yr, 9, 10))]
    if len(c):
        doy_post.append(c[abs((c - pd.Timestamp(yr, 9, 8)).days).argmin()])
doy_post = pd.DatetimeIndex(doy_post)
ov = len(set(doy_post) & set(ld_post))
print(f"\nSep-08 doy sessions: {len(doy_post)}, of which post-Labor-Day: {ov} "
      f"({100*ov/len(doy_post):.0f}%)")

MID = {y for y in range(1996, 2030) if y % 4 == 2}


def block(name, anchors, posts, tk):
    s = px[tk].dropna()
    a = pd.DatetimeIndex([d for d in anchors if d in s.index])
    r = fwd_ret(s, 1).reindex(a).dropna()
    if len(r) < 3:
        print(f"  {name:34} {tk:9} n={len(r)} too thin")
        return
    v = r.values
    up = int((v > 0).sum())
    d = summarize(v * 0 + v, "")
    print(f"  {name:34} {tk:9} n={len(v):3d} mean={100*v.mean():+7.3f}% "
          f"med={100*np.median(v):+7.3f}% up={up}-{len(v)-up} "
          f"hit={100*up/len(v):5.1f}% t={d.get('t', float('nan')):+5.2f} "
          f"signp={sign_test(up, len(v)):.4f}")
    return r


print("\n=== h1 from the anchor close (h1 IS the post-holiday session) ===")
for tk in TK:
    print(f"\n-- {tk}")
    block("all post-holiday", hol_anchor, holpost, tk)
    block("post-LABOR-DAY only", ld_anchor, ld_post, tk)
    mid_a = pd.DatetimeIndex([d for d in ld_anchor if d.year in MID])
    block("post-LD, midterm years", mid_a, None, tk)
    # control: every other session
    s = px[tk].dropna()
    r_all = fwd_ret(s, 1).dropna()
    ctrl = r_all.drop(index=[d for d in hol_anchor if d in r_all.index], errors="ignore")
    up = int((ctrl.values > 0).sum())
    print(f"  {'CONTROL non-holiday-eve':34} {tk:9} n={len(ctrl):5d} "
          f"mean={100*ctrl.values.mean():+7.3f}% hit={100*up/len(ctrl):5.1f}%")

print("\n=== ^VIX post-holiday, detail ===")
s = px["^VIX"].dropna()
r = fwd_ret(s, 1).reindex(pd.DatetimeIndex([d for d in hol_anchor if d in s.index])).dropna()
print("  era:", [f"{e['label']} n={e['n']} mean={e['mean_pct']:+.3f}% hit={e['hit']:.1f}%" for e in era_split(r.index, r.values)])
print("  concentration:", cluster_note(r.index, r.values, 2))
rl = fwd_ret(s, 1).reindex(pd.DatetimeIndex([d for d in ld_anchor if d in s.index])).dropna()
print(f"  Labor Day only: {[f'{d.year}:{100*v:+.1f}%' for d, v in zip(rl.index, rl.values)]}")
