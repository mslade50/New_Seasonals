"""C6 — the pre-opex WEEK itself: entry MOC on the Friday BEFORE opex week,
exit MOC at opex (h=5), both vol events inside the hold.

Anchor arithmetic: opex is the 3rd Friday. The Friday before opex week is
opex - 5 td. With the lag=1 MOC-tomorrow convention the SIGNAL day is
opex - 6 td, entry at opex - 5 td (that Friday's close), exit at opex.

Owes, per the brief: the Friday/weekday placebo, a tdom month-position
control, the placebo ANCHOR LADDER, an era split, and the August/midterm
crossing. Prior expectation is that this is month position.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import (close_panel, fwd_lag, load_events, show, sign_test,  # noqa: E402
                       summarize, cluster_note, bootstrap_p_le0)

ASOF = pd.Timestamp("2026-08-13")
H = 5

px = close_panel(["SPY", "IWM"])
px = px[px.index <= ASOF]
spy = px["SPY"].dropna()
idx = spy.index
pos = pd.Series(range(len(idx)), index=idx)
ret5 = fwd_lag(spy, H, 1)

opex = load_events(["opex"])["date"]
opex = opex[(opex >= idx[0]) & (opex <= idx[-1])]

# map each opex to its trading-day position; anchor = opex - 6 td
anchors, opex_used = [], []
for d in opex:
    p = pos.get(d)
    if p is None:                       # opex on a holiday -> nearest prior bar
        prior = idx[idx <= d]
        if len(prior) == 0:
            continue
        p = pos[prior[-1]]
    if p - 6 < 0:
        continue
    anchors.append(idx[p - 6])
    opex_used.append(idx[p])
anchors = pd.DatetimeIndex(anchors)
print(f"opex events mapped: {len(anchors)}  ({anchors[0].date()} .. {anchors[-1].date()})")
print("entry weekday distribution (should be all Friday=4):",
      pd.Series(anchors).apply(lambda d: idx[pos[d] + 1].weekday()).value_counts().to_dict())

valid = ret5.notna()
a = pd.DatetimeIndex([d for d in anchors if valid.get(d, False)])
cond = ret5.loc[a].values
wins = int((cond > 0).sum())

print("\n" + "=" * 74)
print("1. THE CELL vs ITS CONTROLS  (long SPY, entry Fri-before-opex MOC, h=5)")
print("=" * 74)
fri = idx[(idx.weekday == 4) & valid.values]
allday = idx[valid.values]
in_span = idx[(idx >= a[0]) & (idx <= a[-1]) & valid.values]
rows = [summarize(cond, f"COND pre-opex week (N={len(a)})"),
        summarize(ret5.loc[in_span].values, "CTRL-a SPY all days, same span"),
        summarize(ret5.loc[allday].values, "CTRL-b SPY all days, full history"),
        summarize(ret5.loc[fri].values, f"CTRL-Friday all Fridays (N={len(fri)})")]
show(rows, "conditional vs controls")
print(f"  record {wins}-{len(a)-wins}, sign p = {sign_test(wins, len(a)):.4f}   "
      f"bootstrap P(mean<=0) = {bootstrap_p_le0(cond):.3f}")
print(f"  edge vs all-Fridays = {100*(cond.mean()-ret5.loc[fri].mean()):+.3f}pp")
print(f"  concentration: {cluster_note(a, cond)}")

print("\n" + "=" * 74)
print("2. MONTH-POSITION CONTROL — 'pre-opex week' IS 'the third week'")
print("=" * 74)
tdom = pd.Series(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).cumcount().values + 1,
                 index=idx)
print("  entry-day tdom distribution:",
      pd.Series([tdom[idx[pos[d] + 1]] for d in a]).value_counts().sort_index().to_dict())
rows = []
for lo, hi in [(1, 5), (6, 9), (10, 12), (13, 15), (16, 21)]:
    m = (tdom >= lo) & (tdom <= hi)
    rows.append(summarize(ret5.loc[idx[m.values & valid.values]].values, f"tdom {lo}-{hi}"))
show(rows, "SPY h=5 unconditional by entry tdom")
# Fridays MATCHED on tdom to the cell's own entry tdoms
ent_tdom = pd.Series([tdom[idx[pos[d] + 1]] for d in a])
lo_t, hi_t = int(ent_tdom.min()), int(ent_tdom.max())
mfri = idx[(idx.weekday == 4) & (tdom >= lo_t).values & (tdom <= hi_t).values & valid.values]
matched = ret5.loc[mfri.difference(pd.DatetimeIndex([idx[pos[d] + 1] for d in a]))].values
show([summarize(cond, "COND pre-opex Fridays"),
      summarize(matched, f"CTRL Fridays at tdom {lo_t}-{hi_t}, ex-cell")],
     "tdom-and-weekday MATCHED control")

print("\n" + "=" * 74)
print("3. PLACEBO ANCHOR LADDER — offsets around the true anchor (h=5 fixed)")
print("=" * 74)
lad = []
for off in range(-10, 11):
    dd = []
    for d in anchors:
        p = pos[d] + off
        if 0 <= p < len(idx) and valid.iloc[p]:
            dd.append(idx[p])
    if not dd:
        continue
    v = ret5.loc[pd.DatetimeIndex(dd)].values
    r = summarize(v, f"offset {off:+d}" + ("  <-- TRUE ANCHOR" if off == 0 else ""))
    lad.append(r)
show(lad, "anchor ladder")
real = [r for r in lad if r["label"].startswith("offset +0")][0]
rank = 1 + sum(1 for r in lad if r["mean_pct"] > real["mean_pct"])
print(f"  TRUE anchor mean {real['mean_pct']:+.3f}% ranks {rank} of {len(lad)} offsets. "
      f"A PLATEAU here = month position, not an event.")

print("\n" + "=" * 74)
print("4. ERA, MONTH AND CYCLE-YEAR SPLITS")
print("=" * 74)
d = pd.DatetimeIndex(a)
show([summarize(cond[d < pd.Timestamp("2010-01-01")], "pre-2010"),
      summarize(cond[(d >= pd.Timestamp("2010-01-01")) & (d < pd.Timestamp("2018-01-01"))], "2010-2017"),
      summarize(cond[d >= pd.Timestamp("2018-01-01")], "2018+")], "era")
aug = d.month == 8
mid = (d.year % 4) == 2
show([summarize(cond[aug], f"August only (N={int(aug.sum())})"),
      summarize(cond[~aug], "all other months"),
      summarize(cond[mid], f"midterm years (N={int(mid.sum())})"),
      summarize(cond[aug & mid], f"August AND midterm (N={int((aug & mid).sum())})")],
     "month / cycle crossing (2026 is August + midterm)")
augw = int((cond[aug] > 0).sum())
print(f"  August record {augw}-{int(aug.sum())-augw}, sign p = "
      f"{sign_test(augw, int(aug.sum())):.4f}")

print("\n" + "=" * 74)
print("5. COST + IWM LEG")
print("=" * 74)
print(f"  SPY 1 leg ~2 bps round trip; episode mean {100*cond.mean():.3f}% = "
      f"{10000*cond.mean():.1f} bps -> {10000*cond.mean()/2.0:.1f}x cost (need >=5x)")
iwm = px["IWM"].dropna()
ir = fwd_lag(iwm, H, 1)
ai = pd.DatetimeIndex([x for x in a if ir.notna().get(x, False)])
show([summarize(ir.loc[ai].values, f"IWM pre-opex week (N={len(ai)})"),
      summarize(ir.loc[iwm.index[ir.notna().values]].values, "IWM all days")], "IWM leg")
