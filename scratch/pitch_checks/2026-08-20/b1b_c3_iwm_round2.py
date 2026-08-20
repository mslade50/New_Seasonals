"""C3 round 2. Round 1 left the h=10 cell alive (ladder rank 1 of 16, +1.603%
over 26 against an Aug tdom-band control of +0.852%). This round attacks the
LIVE conditioners rather than the pooled cell.

Fixes a contamination in round 1's ladder: sliding k from ALL opex dates and
then filtering on `anchor.month == 8` let September opex minus 10 sessions leak
in (n=31 at k=-10). Here the AUGUST OPEX DATES are selected first and k slides
off those, so N is 26 at every rung.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
warnings.filterwarnings("ignore")
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

px = close_panel(["IWM", "SPY"])
d = px.index
ev = load_events(["opex"])
opex = pd.DatetimeIndex(sorted(set(ev["date"]) & set(d)))
aug_opex = pd.DatetimeIndex([x for x in opex if x.month == 8])


def anchor_at(off, src=None):
    src = aug_opex if src is None else src
    pos = d.get_indexer(src) + off
    pos = pos[(pos >= 0) & (pos < len(d))]
    return d[pos]


entry = anchor_at(-1)
print(f"August opex dates {len(aug_opex)}; entry anchors {len(entry)}")

print("\n" + "=" * 78)
print("1. LADDER, CONTAMINATION FIXED (August opex selected first, N=26 "
      "at every rung)")
print("=" * 78)
for h in (3, 5, 10):
    rows = []
    for k in range(-10, 6):
        a = anchor_at(k)
        s = fwd_lag(px["IWM"], h, lag=0).reindex(a).dropna()
        rows.append({"k": k, "n": len(s), "mean_pct": 100 * s.mean(),
                     "hit": 100 * (s > 0).mean(),
                     "true": "<== TRUE" if k == -1 else ""})
    df = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    print(f"\n--- h={h} ---")
    print(df.round(3).to_string(index=False))
    print(f"  >>> TRUE ANCHOR RANKS "
          f"{int(df[df.k == -1]['rank'].iloc[0])} of {len(df)}; "
          f"ladder spread {df.mean_pct.max()-df.mean_pct.min():.3f}pp, "
          f"true minus ladder-median "
          f"{df[df.k==-1].mean_pct.iloc[0]-df.mean_pct.median():+.3f}pp")

# --------------------------------------------------------- tdom + midterm
aug = d[d.month == 8]
tmap = {}
for y in sorted(set(aug.year)):
    m = d[(d.year == y) & (d.month == 8)]
    for i, x in enumerate(m):
        tmap[x] = i + 1
tdom_anchor = sorted(set(tmap[x] for x in entry))
band = pd.DatetimeIndex([x for x in aug if min(tdom_anchor) <= tmap[x]
                         <= max(tdom_anchor)])

print("\n\n" + "=" * 78)
print("2. THE LIVE CONDITIONER: midterm. Against a MIDTERM-restricted "
      "unconditional August control, not an all-years one.")
print("=" * 78)
for h in (3, 5, 10):
    s = fwd_lag(px["IWM"], h, lag=0)
    ev_all = s.reindex(entry).dropna()
    bd = s.reindex(band).dropna()
    for lbl, sel in (("ALL YEARS", lambda i: np.ones(len(i), bool)),
                     ("MIDTERM ONLY", lambda i: (i.year % 4 == 2))):
        e = ev_all[sel(ev_all.index)]
        b = bd[sel(bd.index)]
        print(f"  h={h:2d} {lbl:13s}  opex anchor {100*e.mean():+.3f}% "
              f"(N={len(e)}, hit {100*(e>0).mean():.0f}%)  |  "
              f"unconditional Aug tdom {min(tdom_anchor)}-{max(tdom_anchor)} "
              f"{100*b.mean():+.3f}% (N={len(b)})  ->  ANCHOR EXCESS "
              f"{100*(e.mean()-b.mean()):+.3f}pp")
    print()

print("\n" + "=" * 78)
print("3. GATE ATTRIBUTION: run the trade WITHOUT its August gate, and "
      "without its opex gate")
print("=" * 78)
for h in (3, 5, 10):
    s = fwd_lag(px["IWM"], h, lag=0)
    a_aug = s.reindex(entry).dropna()
    a_all = s.reindex(anchor_at(-1, src=opex)).dropna()
    a_ex = a_all[a_all.index.month != 8]
    aug_only = s.reindex(band).dropna()
    base = s.dropna()
    print(f"  h={h:2d}  BOTH gates (Aug + opex) {100*a_aug.mean():+.3f}% "
          f"N={len(a_aug)}  |  opex only, no Aug "
          f"{100*a_ex.mean():+.3f}% N={len(a_ex)}  |  Aug only, no opex "
          f"{100*aug_only.mean():+.3f}% N={len(aug_only)}  |  neither "
          f"{100*base.mean():+.3f}% N={len(base)}")

print("\n" + "=" * 78)
print("4. LIVE PRICE STATE: IWM 1.10% off its 52w high. Cross the calendar "
      "cell with the state that is actually live.")
print("=" * 78)
hi = rolling_on_valid(px["IWM"], lambda x: x.rolling(252).max())
off = (px["IWM"] / hi - 1.0) * 100
for h in (5, 10):
    s = fwd_lag(px["IWM"], h, lag=0)
    # pooled opex anchors, split on the live state
    a_all = s.reindex(anchor_at(-1, src=opex)).dropna()
    o = off.reindex(a_all.index)
    for thr in (-1.5, -2.0, -3.0):
        near = (o >= thr).values
        wins = int((a_all[near] > 0).sum())
        print(f"  h={h:2d} POOLED opex-1, IWM within {abs(thr):.1f}% of its "
              f"52w high: {100*a_all[near].mean():+.3f}% N={int(near.sum())} "
              f"(hit {100*(a_all[near]>0).mean():.0f}%, sign p "
              f"{sign_test(wins, int(near.sum())):.3f})  |  farther out "
              f"{100*a_all[~near].mean():+.3f}% N={int((~near).sum())}")
    # and the unconditional near-high control (is it the state, not the opex?)
    ab = s.dropna()
    ob = off.reindex(ab.index)
    n2 = (ob >= -2.0).values
    print(f"  h={h:2d}   control: ALL DAYS with IWM within 2% of its 52w "
          f"high {100*ab[n2].mean():+.3f}% N={int(n2.sum())}  ->  the opex "
          f"gate's effect INSIDE the live state = "
          f"{100*(a_all[(o>=-2.0).values].mean()-ab[n2].mean()):+.3f}pp")
    print()

print("\n" + "=" * 78)
print("5. LEAVE-ONE-YEAR-OUT floor on the August cell")
print("=" * 78)
for h in (5, 10):
    s = fwd_lag(px["IWM"], h, lag=0).reindex(entry).dropna()
    loyo = {int(y): 100 * s[s.index.year != y].mean()
            for y in sorted(set(s.index.year))}
    lo = min(loyo.values())
    lo_y = [y for y, v in loyo.items() if v == lo][0]
    print(f"  h={h:2d}  full {100*s.mean():+.3f}%   LOYO floor "
          f"{lo:+.3f}% (dropping {lo_y})   LOYO ceiling "
          f"{max(loyo.values()):+.3f}%")
    # against the matched Aug window control
    b = fwd_lag(px["IWM"], h, lag=0).reindex(band).dropna()
    print(f"        matched Aug window control {100*b.mean():+.3f}%  ->  "
          f"LOYO floor EXCESS {lo-100*b.mean():+.3f}pp")

print("\n" + "=" * 78)
print("6. MULTIPLICITY: how many cells were looked at before this one won?")
print("=" * 78)
print("  03_recon_events.py priced 15 vehicles x 5 horizons x 2 anchors x")
print("  {pooled, August} = 300 cells. This candidate is the maximum of that")
print("  grid. Under a global null the max of 300 correlated cells is")
print("  routinely +1 to +2 sd. Charge the cell for the grid.")
grid = []
for tk in ["IWM", "SPY"]:
    for h in (1, 2, 3, 5, 10):
        s = fwd_lag(px[tk], h, lag=0)
        for mo in range(1, 13):
            a = anchor_at(-1, src=pd.DatetimeIndex(
                [x for x in opex if x.month == mo]))
            v = s.reindex(a).dropna()
            base = s.dropna()
            grid.append({"tk": tk, "h": h, "mo": mo,
                         "excess": 100 * (v.mean() - base.mean())})
g = pd.DataFrame(grid)
print(f"\n  month x horizon x {{IWM,SPY}} grid at the opex-1 anchor: "
      f"{len(g)} cells")
print(f"  observed IWM/August/h=10 excess = "
      f"{g[(g.tk=='IWM')&(g.h==10)&(g.mo==8)].excess.iloc[0]:+.3f}pp")
print(f"  that cell's rank among all {len(g)}: "
      f"{int((g.excess > g[(g.tk=='IWM')&(g.h==10)&(g.mo==8)].excess.iloc[0]).sum())+1}")
print(f"  cells with |excess| >= 1.0pp: {int((g.excess.abs()>=1.0).sum())} "
      f"of {len(g)}   sd of the excess distribution "
      f"{g.excess.std():.3f}pp")
