"""C3 round 2b -- WHERE inside the hold does the pair cell earn?

C3b left one thing standing: SVXY entered at first-print-2 and held 5
sessions pays +3.240% on 65 anchors (50-15), +2.17pp month-x-tdom matched,
+1.958% alpha at a beta fitted off the trigger set.  But the h=3 placebo
ladder ranked the true anchor SIXTH of twelve, with k=0 and k=-1 -- entries
placed ON or AFTER the first print -- paying 3.5x more.

That pattern has exactly one honest reading and this script tests it: the
return does not accrue in the run-in, it accrues AFTER the calendar clears,
which is watchlist 33's mechanism.  If so the pitched entry (today, two
sessions BEFORE the first print) is buying dead or negative carry to reach
the part that pays, and the correct anchor is the second print's close --
2026-09-11, not 2026-09-08.

Probes:
  a. day-by-day path of every episode, and the segment decomposition
     (entry->print1, print1->print2, print2->exit)
  b. placebo anchor ladder AT h=5, the pitched horizon
  c. whole-variant comparison: pitched entry (print1-2) vs the post-print
     entry (MOC on print 2), matched exit dates, both eras
  d. the live -0.5x leverage era on its own
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 240)

px = close_panel(["SVXY", "^VIX", "SPY"])
cal = px["SPY"].dropna().index
pos = pd.Series(range(len(cal)), index=cal)
KINDS = ("nfp", "cpi", "ppi", "fomc_decision")
ALL_PRINTS = pd.DatetimeIndex(sorted(pd.concat(
    [load_events([k])["date"] for k in KINDS]).unique()))
ppi = load_events(["ppi"])["date"]
p_all, _ = anchor_positions(cal, ppi, 0)
print_pos = {int(pos.get(d, -1)) for d in ALL_PRINTS}
pair_pos = [p for p in p_all if (p + 1) in print_pos]
A_PAIR = cal[[p - 3 for p in pair_pos if p - 3 >= 0]]

sv = px["SVXY"].dropna()
svp = pd.DataFrame({"SVXY": sv})

print("=" * 100)
print("a. DAY-BY-DAY PATH, pair anchors, h=8 (entry = print1 - 2).")
print("   day 1 = the session after entry; print1 lands on day 2, print2 on")
print("   day 3; days 4+ are the cleared calendar.")
print("=" * 100)
paths = episode_paths(svp, pd.DatetimeIndex(A_PAIR), [("SVXY", 1.0)], h=8, lag=1)
print(f"  N={len(paths)}")
cum = 100 * paths.mean()
inc = cum.diff()
inc.iloc[0] = cum.iloc[0]
hit = 100 * (paths > 0).mean()
out = pd.DataFrame({"cum_mean_pct": cum.round(3), "incr_pct": inc.round(3),
                    "hit_cum": hit.round(1)})
out.index.name = "day"
print(out.to_string())
d_all = pd.DatetimeIndex(A_PAIR)
seg = pd.DataFrame({
    "entry->print1 (d1-2)": paths[2],
    "print1->print2 (d3)": paths[3] - paths[2],
    "print2->d5 (cleared)": paths[5] - paths[3],
    "d5->d8": paths[8] - paths[5],
}, index=paths.index)
print("\n  SEGMENTS (mean %, hit %, sign p):")
for c in seg.columns:
    v = seg[c].dropna().values
    print(f"   {c:24s} {100*v.mean():+7.3f}%  hit {100*(v>0).mean():5.1f}%  "
          f"n={len(v)}  sign p {sign_test(int((v>0).sum()), len(v)):.4f}")

# the same segmentation for the LIVE leverage era
m18 = seg.index >= pd.Timestamp("2018-02-28")
print("\n  -0.5x LIVE ERA only (2018-02-28+):")
for c in seg.columns:
    v = seg.loc[m18, c].dropna().values
    if not len(v):
        continue
    print(f"   {c:24s} {100*v.mean():+7.3f}%  hit {100*(v>0).mean():5.1f}%  "
          f"n={len(v)}  sign p {sign_test(int((v>0).sum()), len(v)):.4f}")

print("\n" + "=" * 100)
print("b. PLACEBO ANCHOR LADDER AT h=5, the pitched horizon.")
print("=" * 100)
r5 = fwd_lag(sv, 5, lag=1)
rows = []
for k in range(-8, 6):
    a = cal[[p + k for p in pair_pos if 0 <= p + k < len(cal)]]
    v = r5.reindex(a).dropna().values
    if len(v) < 5:
        continue
    rows.append({"k": k, "entry_is": f"print1{k+1:+d}", "n": len(v),
                 "mean_pct": round(100 * v.mean(), 3),
                 "hit": round(100 * (v > 0).mean(), 1),
                 "t": round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2)})
d = pd.DataFrame(rows).sort_values("mean_pct", ascending=False).reset_index(drop=True)
rk = d.index[d["k"] == -3]
print(f"  TRUE ANCHOR k=-3 (entry = print1-2) RANKS {int(rk[0])+1} of {len(d)}")
print(d.to_string(index=False))

print("\n" + "=" * 100)
print("c. WHOLE-VARIANT COMPARISON. Same episodes, different entry.")
print("   PITCHED : MOC print1-2, exit +5  (today = 2026-09-08 entry)")
print("   POST    : MOC print2   , exit +3 (2026-09-11 entry, same exit day)")
print("   POST5   : MOC print2   , exit +5")
print("=" * 100)
rows = []
for lbl, k, h in (("PITCHED entry print1-2, h=5", -3, 5),
                  ("POST entry ON print2, h=3", 0, 3),
                  ("POST entry ON print2, h=5", 0, 5),
                  ("entry ON print1, h=5", -1, 5)):
    r = fwd_lag(sv, h, lag=1)
    a = cal[[p + k for p in pair_pos if 0 <= p + k < len(cal)]]
    v = r.reindex(a).dropna().values
    s = summarize(v, lbl)
    base = r.dropna()
    s["drift_pct"] = round(100 * base.mean(), 3)
    s["edge_pp"] = round(s["mean_pct"] - 100 * base.mean(), 3)
    s["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    rows.append(s)
show(rows, "SVXY, whole variants (never a marginal-fill decomposition)")

print("\n" + "=" * 100)
print("d. THE LIVE -0.5x ERA ALONE, both entries.")
print("=" * 100)
rows = []
for lbl, k, h in (("PITCHED print1-2, h=5", -3, 5),
                  ("POST on print2, h=3", 0, 3),
                  ("POST on print2, h=5", 0, 5)):
    r = fwd_lag(sv, h, lag=1)
    a = pd.DatetimeIndex(cal[[p + k for p in pair_pos if 0 <= p + k < len(cal)]])
    a = a[a >= pd.Timestamp("2018-02-28")]
    v = r.reindex(a).dropna().values
    if not len(v):
        continue
    s = summarize(v, lbl + " [2018-02-28+]")
    s["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    rows.append(s)
show(rows)

print("\n" + "=" * 100)
print("e. AND THE SAME SEGMENTATION ON SHORT ^VIX (unlevered mechanism).")
print("=" * 100)
vx = px["^VIX"].dropna()
vxp = pd.DataFrame({"VIX": vx})
pv = episode_paths(vxp, pd.DatetimeIndex(cal[[p - 3 for p in pair_pos if p - 3 >= 0]]),
                   [("VIX", -1.0)], h=8, lag=1)
cum = 100 * pv.mean()
inc = cum.diff()
inc.iloc[0] = cum.iloc[0]
print(pd.DataFrame({"cum_short_vix_pct": cum.round(3),
                    "incr_pct": inc.round(3)}).to_string())
