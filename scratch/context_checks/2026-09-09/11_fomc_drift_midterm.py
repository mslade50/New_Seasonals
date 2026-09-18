"""Pre-FOMC drift into the 2026-09-16 decision, with the cycle split.

The decision is 2026-09-16. Four trading sessions before it is 2026-09-10
(the sessions after it are 09-11, 09-14, 09-15, 09-16 -- no US holiday in the
span), so TOMORROW's close is the standard anchor and the decision session's
own close is h=4 from it.

Pre-specified hypothesis (Lucca-Moench), so no multiplicity correction is owed
to tonight's sweep. What is NOT pre-specified is the cycle split, which is
reported as the exploratory cut it is.

Scheduled-only by construction: data/macro_events.csv carries the Fed's own
fomchistorical/fomccalendars pages, 8 decisions a year and nothing else. The
intermeeting cuts (2001-01-03, 2001-04-18, 2001-09-17, 2008-01-22,
2008-10-08, 2020-03-03, 2020-03-15) are ABSENT from that file; the count check
below prints the per-year totals so the claim is auditable rather than
asserted. Forward returns are lag=0 fwd_ret, the context convention.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import (load_prices, load_events, fwd_ret, anchor_positions,
                       summarize, era_split, sign_test, cluster_note, show,
                       local_control)

ASOF = pd.Timestamp("2026-09-09")
DECISION = pd.Timestamp("2026-09-16")
OFFSET = -4  # trading days before the decision


def stats_line(dates, vals, label, indent="   "):
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    s = summarize(v, label)
    if not s["n"]:
        print(f"{indent}{label}: n=0")
        return None
    up = int((v > 0).sum())
    dn = s["n"] - up
    flag = ""
    if (s["mean_pct"] < 0 and up > dn) or (s["mean_pct"] > 0 and dn > up):
        flag += "   <== MEAN AND RECORD DISAGREE IN SIGN"
    if s["n"] < 15:
        flag += "   <== n<15"
    print(f"{indent}{label}: n={s['n']:3d} mean={s['mean_pct']:+.3f}% "
          f"med={s['median_pct']:+.3f}% hit={s['hit']:5.1f}% t={s['t']:+.2f} "
          f"rec {up}-{dn}  signp(up)={sign_test(up, s['n']):.4f} "
          f"signp(down)={sign_test(dn, s['n']):.4f} "
          f"worst={s['worst_pct']:+.2f}% best={s['best_pct']:+.2f}%{flag}")
    cn = cluster_note(pd.DatetimeIndex(dates), v, k=2)
    print(f"{indent}     cluster: {cn}")
    return s


px = load_prices(["^GSPC", "^VIX", "SPY"])
gspc = px["^GSPC"]["Close"].astype(float).dropna()
vix = px["^VIX"]["Close"].astype(float).dropna()
gidx = gspc.index

# ------------------------------------------------------------ f. the calendar
print("=" * 78)
print("f. EVENT SET: scheduled FOMC decisions only")
print("=" * 78)
ev = load_events(["fomc_decision"])
dates_all = pd.DatetimeIndex(ev["date"])
print(f"  macro_events.csv fomc_decision rows: {len(dates_all)}  "
      f"({dates_all.min().date()} .. {dates_all.max().date()})")
per_year = pd.Series(1, index=dates_all).groupby(dates_all.year).sum()
print("  per-year counts (8 = the scheduled cadence; a 9th would be an "
      "intermeeting move leaking in):")
print("   ", {int(y): int(c) for y, c in per_year.items()})
odd = {int(y): int(c) for y, c in per_year.items() if c != 8}
print(f"  years that are NOT 8: {odd}")
print("  known intermeeting decisions checked for presence "
      "(all should be ABSENT):")
for d in ["2001-01-03", "2001-04-18", "2001-09-17", "2008-01-22",
          "2008-10-08", "2020-03-03", "2020-03-15"]:
    print(f"    {d}: {'PRESENT (would need excluding)' if pd.Timestamp(d) in set(dates_all) else 'absent'}")
print("  sanity, a SCHEDULED two-day meeting that is easy to mistake for an "
      "intermeeting move (Nov 2-3 2010, QE2):")
print(f"    2010-11-03: "
      f"{'present, correctly KEPT' if pd.Timestamp('2010-11-03') in set(dates_all) else 'absent'}")
print("  -> no exclusion filter is applied because the source file is "
      "scheduled-only; 2020 carries 7 because the March 17-18 meeting was "
      "superseded by the unscheduled 03-15 cut, which the file omits.")
print(f"\n  ^GSPC cache: {gidx.min().date()} .. {gidx.max().date()}  "
      f"({len(gspc)} sessions) -> the usable event window starts in 2000, "
      f"not 1999.")
print(f"  ^VIX cache: {vix.index.min().date()} .. {vix.index.max().date()} "
      f"({len(vix)} sessions)")

# live anchor arithmetic, stated
print(f"\n  LIVE EVENT: decision {DECISION.date()}. Sessions between the "
      f"anchor and the decision, inclusive of the decision: 09-11, 09-14, "
      f"09-15, 09-16 = 4 -> anchor = 2026-09-10 (tomorrow's close).")
print(f"  Tonight's close {ASOF.date()} is the 5-td-before session; "
      f"section g measures that anchor separately.")

# -------------------------------------------------------- a. the base drift
print()
print("=" * 78)
print("a. PRE-FOMC DRIFT: ^GSPC from the 4-td-before anchor, lag=0")
print("=" * 78)
pos, kept = anchor_positions(gidx, dates_all, offset=OFFSET)
anchors = pd.DatetimeIndex([gidx[p] for p in pos])
print(f"  decisions in the calendar: {len(dates_all)}")
print(f"  decisions surviving anchor_positions guards (inside the ^GSPC "
      f"span, offset resolvable): {len(kept)}")
dropped = sorted(set(dates_all) - set(kept))
print(f"  dropped: {len(dropped)}  "
      f"first/last -> {[str(pd.Timestamp(d).date()) for d in dropped[:3]]} ... "
      f"{[str(pd.Timestamp(d).date()) for d in dropped[-3:]]}")
print("    (the tail drops are the 2026-09-16 .. 2027 decisions that have not "
     "happened; guard 1 in anchor_positions)")

f = {h: fwd_ret(gspc, h) for h in (1, 2, 3, 4, 5)}
print("\n  ALL SCHEDULED DECISIONS (anchor -> h):")
for h in (1, 4, 5):
    v = f[h].reindex(anchors).dropna()
    stats_line(v.index, v.values, f"h={h} ({'decision close' if h == 4 else 'session after the anchor' if h == 1 else 'decision +1'})")
print("\n  full horizon ladder from the anchor (h=1..5):")
for h in (1, 2, 3, 4, 5):
    v = f[h].reindex(anchors).dropna()
    s = summarize(v.values, f"h={h}")
    up = int((v.values > 0).sum())
    print(f"    h={h}: n={s['n']:3d} mean={s['mean_pct']:+.3f}% "
          f"med={s['median_pct']:+.3f}% hit={s['hit']:5.1f}% t={s['t']:+.2f} "
          f"rec {up}-{s['n']-up}")

for h in (1, 4, 5):
    v = f[h].reindex(anchors).dropna()
    show(era_split(v.index, v.values), f"  era split, h={h}")

# --------------------------------------------------------- b. the cycle split
print()
print("=" * 78)
print("b. CYCLE SPLIT: midterm (year %% 4 == 2) vs every other year")
print("=" * 78)
kept_year = pd.Series(pd.DatetimeIndex(kept).year, index=anchors)
mid = anchors[(kept_year.values % 4) == 2]
non = anchors[(kept_year.values % 4) != 2]
print(f"  midterm anchors: {len(mid)}   other-year anchors: {len(non)}")
print(f"  midterm years present: "
      f"{sorted(set(pd.DatetimeIndex(mid).year.tolist()))}")

for lbl, grp in (("MIDTERM", mid), ("NON-MIDTERM", non)):
    print(f"\n  --- {lbl} (anchors={len(grp)}) ---")
    for h in (1, 4, 5):
        v = f[h].reindex(grp).dropna()
        stats_line(v.index, v.values, f"h={h}")
        show(era_split(v.index, v.values), f"     era split h={h}")

print("\n  --- every MIDTERM episode, h=4 (decision-session close) ---")
print(f"    {'decision':12s} {'anchor':12s} {'h1':>9s} {'h4':>9s} {'h5':>9s}")
kept_list = list(kept)
for a, d in zip(anchors, kept_list):
    if pd.Timestamp(d).year % 4 != 2:
        continue
    row = [f[h].get(a, np.nan) for h in (1, 4, 5)]
    print(f"    {str(pd.Timestamp(d).date()):12s} {str(a.date()):12s} "
          + " ".join(f"{100*r:+8.2f}%" if pd.notna(r) else f"{'  n/a':>9s}"
                     for r in row))

# ------------------------------------------------------------- d. the controls
print()
print("=" * 78)
print("d. CONTROLS")
print("=" * 78)
for h in (1, 4, 5):
    base = f[h].dropna()
    v = f[h].reindex(anchors).dropna()
    loc = local_control(base.index, anchors, win=126)
    lv = f[h].reindex(loc).dropna()
    print(f"  h={h}: CTRL-b all days n={len(base)} mean={100*base.mean():+.3f}%"
          f"   CTRL-c local +/-126td ex-anchor n={len(lv)} "
          f"mean={100*lv.mean():+.3f}%")
    print(f"        pre-FOMC mean {100*v.mean():+.3f}%  -> edge vs all-days "
          f"{100*(v.mean()-base.mean()):+.3f}pp, vs local "
          f"{100*(v.mean()-lv.mean()):+.3f}pp")
    for lbl, grp in (("midterm", mid), ("non-midterm", non)):
        g = f[h].reindex(grp).dropna()
        print(f"        {lbl:12s} mean {100*g.mean():+.3f}% -> edge vs "
              f"all-days {100*(g.mean()-base.mean()):+.3f}pp")

# ------------------------------------------------------------------ e. the VIX
print()
print("=" * 78)
print("e. ^VIX OVER THE SAME WINDOW (the vol-compression telling)")
print("=" * 78)
vpos, vkept = anchor_positions(vix.index, dates_all, offset=OFFSET)
vanch = pd.DatetimeIndex([vix.index[p] for p in vpos])
vf = {h: fwd_ret(vix, h) for h in (1, 4, 5)}
vyear = pd.Series(pd.DatetimeIndex(vkept).year, index=vanch)
vmid = vanch[(vyear.values % 4) == 2]
vnon = vanch[(vyear.values % 4) != 2]
print(f"  ^VIX anchors: {len(vanch)}  (midterm {len(vmid)} / other {len(vnon)})")
print("  NOTE: these are ^VIX INDEX LEVEL changes, not a tradeable vehicle.")
for lbl, grp in (("ALL", vanch), ("MIDTERM", vmid), ("NON-MIDTERM", vnon)):
    print(f"\n  --- ^VIX {lbl} ---")
    for h in (1, 4, 5):
        v = vf[h].reindex(grp).dropna()
        stats_line(v.index, v.values, f"h={h}")
    if lbl == "ALL":
        for h in (4,):
            base = vf[h].dropna()
            print(f"     CTRL-b ^VIX all days h={h}: n={len(base)} "
                  f"mean={100*base.mean():+.3f}%")

# ------------------------------- g. tonight's own anchor (5 td before), extra
print()
print("=" * 78)
print("g. EXTRA, NOT REQUESTED BUT TONIGHT'S ACTUAL ANALOGUE: the 5-td-before "
      "anchor")
print("=" * 78)
print("  Tonight's close is 5 sessions before the decision, so the cell that "
      "maps to TODAY is the 5-td-before anchor with h=5 to the decision close.")
pos5, kept5 = anchor_positions(gidx, dates_all, offset=-5)
a5 = pd.DatetimeIndex([gidx[p] for p in pos5])
y5 = pd.Series(pd.DatetimeIndex(kept5).year, index=a5)
m5 = a5[(y5.values % 4) == 2]
n5 = a5[(y5.values % 4) != 2]
for lbl, grp in (("ALL", a5), ("MIDTERM", m5), ("NON-MIDTERM", n5)):
    print(f"\n  --- 5-td anchor, {lbl} (anchors={len(grp)}) ---")
    for h in (1, 5):
        v = f[h].reindex(grp).dropna()
        stats_line(v.index, v.values,
                   f"h={h} ({'tomorrow' if h == 1 else 'decision close'})")
