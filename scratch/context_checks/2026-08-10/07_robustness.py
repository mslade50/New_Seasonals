"""Controls and robustness for the three cells that survived drills 1 and 2.

A. VIX rises on a CPI eve when SPY sits at a 52w high (n=76, +2.58%, 51-25).
   The control that matters: does VIX rise on ANY session after SPY closes at a
   52w high? If it does, the CPI part is decoration.
B. TLT rallies on the CPI session when crude has run 10%+ over 21 sessions
   (n=49, +0.39%, 71% hit, 35-14). Control: the same crude state on non-CPI
   sessions.
C. ^NYA prints a 52w high while SPY, QQQ and IWM all close lower (n=29 episodes,
   SPY h1 -0.24%, IWM -0.38%). Controls: each half on its own, and the cell with
   its two largest episodes removed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, fwd_ret, declusters, local_control, summarize,
    era_split, sign_test, cluster_note,
)

TIX = ["SPY", "QQQ", "IWM", "^NYA", "^VIX", "TLT", "^TNX", "CL=F", "^GSPC"]
px = close_panel(TIX)
px = px[px.index >= "1999-01-01"]
dates = px.index

ev = load_events(["cpi"])
cpi_anch, cpi_day = [], []
for d in ev["date"]:
    pos = dates.searchsorted(pd.Timestamp(d))
    if pos >= len(dates) or dates[pos] != pd.Timestamp(d) or pos - 2 < 0:
        continue
    cpi_anch.append(dates[pos - 2])
    cpi_day.append(dates[pos])
cpi_anch = pd.DatetimeIndex(cpi_anch)

spy_dh = px["SPY"] / px["SPY"].rolling(252).max() - 1.0
cl21 = px["CL=F"] / px["CL=F"].shift(21) - 1.0
nya_dh = px["^NYA"] / px["^NYA"].rolling(252).max() - 1.0


def stat(sub, idx, h, label, indent="   "):
    f = fwd_ret(px[sub].dropna(), h)
    v = f.reindex(pd.DatetimeIndex(sorted(set(idx)))).dropna()
    if len(v) < 3:
        print(f"{indent}{label:58s} n={len(v)} too few")
        return None
    r = summarize(v.values, "")
    up = int((v.values > 0).sum())
    print(f"{indent}{label:58s} n={r['n']:4d} mean {r['mean_pct']:+.3f}% "
          f"med {r['median_pct']:+.3f}% hit {r['hit']:.1f}% t={r['t']:+.2f} | "
          f"{up}-{len(v)-up} up-p {sign_test(up, len(v)):.4f} "
          f"dn-p {sign_test(len(v)-up, len(v)):.4f}")
    return v


def detail(v, indent="        "):
    if v is None:
        return
    print(f"{indent}{cluster_note(v.index, v.values)}")
    for e in era_split(v.index, v.values):
        if e.get("n", 0):
            print(f"{indent}era n={e['n']:4d} mean {e['mean_pct']:+.3f}% "
                  f"hit {e['hit']:.1f}% t={e['t']:+.2f}")
    keep = v.iloc[np.argsort(-np.abs(v.values))[2:]]
    r = summarize(keep.values, "")
    up = int((keep.values > 0).sum())
    print(f"{indent}ex the 2 largest moves: n={r['n']} mean {r['mean_pct']:+.3f}% "
          f"hit {r['hit']:.1f}% {up}-{len(keep)-up}")


print("=" * 78)
print("A. VIX on the CPI eve, conditioned on SPY sitting at a 52w high")
print("=" * 78)
at_high = spy_dh > -0.005
a_idx = cpi_anch[at_high.reindex(cpi_anch).fillna(False).values]
v = stat("^VIX", a_idx, 1, "CPI eve, SPY within 0.5% of 52w high  [the cell]")
detail(v)
stat("SPY", a_idx, 1, "   ...and SPY itself over the same session")
print("\n   controls:")
stat("^VIX", dates, 1, "ALL sessions")
stat("^VIX", dates[at_high.reindex(dates).fillna(False).values], 1,
     "SPY at a 52w high, ANY session (no CPI)  [the key control]")
non_cpi_high = dates[at_high.reindex(dates).fillna(False).values].difference(cpi_anch)
stat("^VIX", non_cpi_high, 1, "SPY at a 52w high, NOT a CPI eve")
stat("^VIX", cpi_anch, 1, "ALL CPI eves (high or not)")
stat("^VIX", cpi_anch[~at_high.reindex(cpi_anch).fillna(False).values], 1,
     "CPI eves with SPY NOT at a 52w high")
print("\n   declustered (10 td), since consecutive at-high CPI eves are rare anyway:")
dc = declusters(a_idx, 10, dates)
stat("^VIX", dc, 1, f"the cell, declustered ({len(dc)} episodes)")

print()
print("=" * 78)
print("B. TLT on the CPI session when crude ran 10%+ over the prior 21 sessions")
print("=" * 78)
hot = cl21 >= 0.10
b_idx = cpi_anch[hot.reindex(cpi_anch).fillna(False).values]
v = stat("TLT", b_idx, 2, "CPI day, crude 21d >= +10%  [the cell]")
detail(v)
stat("^TNX", b_idx, 2, "   ...^TNX over the same session")
stat("TLT", b_idx, 1, "   ...TLT on the eve instead")
print("\n   controls:")
stat("TLT", dates, 2, "ALL sessions, 2-day forward")
hot_dates = dates[hot.reindex(dates).fillna(False).values]
stat("TLT", hot_dates, 2, "crude 21d >= +10%, ANY session  [the key control]")
stat("TLT", hot_dates.difference(cpi_anch), 2, "crude 21d >= +10%, NOT a CPI eve")
stat("TLT", cpi_anch, 2, "ALL CPI days (crude hot or not)")
stat("TLT", cpi_anch[~hot.reindex(cpi_anch).fillna(False).values], 2,
     "CPI days with crude NOT hot")
print("\n   tighter, at today's actual crude reading (+15.25%):")
b15 = cpi_anch[(cl21 >= 0.15).reindex(cpi_anch).fillna(False).values]
v15 = stat("TLT", b15, 2, "CPI day, crude 21d >= +15%")
detail(v15)
print("\n   and with TLT already near a 52w low, which is tonight's state:")
tlt_dl = px["TLT"] / px["TLT"].rolling(252).min() - 1.0
low = tlt_dl < 0.03
stat("TLT", cpi_anch[(hot & low).reindex(cpi_anch).fillna(False).values], 2,
     "CPI day, crude hot AND TLT within 3% of its 52w low")

print()
print("=" * 78)
print("C. ^NYA at a 52w high while SPY, QQQ and IWM all close lower")
print("=" * 78)
down3 = (px["SPY"].pct_change() < 0) & (px["QQQ"].pct_change() < 0) & (px["IWM"].pct_change() < 0)
c_mask = (nya_dh > -0.0005) & down3
c_raw = dates[c_mask.fillna(False).values]
c_idx = declusters(c_raw[c_raw <= dates[-2]], 5, dates)
for sub in ("SPY", "IWM", "QQQ"):
    v = stat(sub, c_idx, 1, f"{sub} h1  [the cell, {len(c_idx)} episodes]")
    detail(v)
print("\n   controls:")
stat("SPY", dates, 1, "ALL sessions")
stat("SPY", dates[(nya_dh > -0.0005).fillna(False).values], 1,
     "^NYA at a 52w high, any session")
stat("SPY", dates[down3.fillna(False).values], 1, "SPY, QQQ and IWM all lower, any session")
stat("IWM", dates[down3.fillna(False).values], 1, "IWM, same")
stat("IWM", dates[(nya_dh > -0.0005).fillna(False).values], 1,
     "IWM, ^NYA at a 52w high, any session")
print("\n   episode list:")
f1 = fwd_ret(px["IWM"].dropna(), 1).reindex(c_idx)
for d, r in f1.items():
    print(f"      {d.date()}  IWM next session {100*r:+.2f}%" if not np.isnan(r) else f"      {d.date()}  n/a")
