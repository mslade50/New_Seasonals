"""^FCHI broke its 200d mean for the first time in 63+ sessions while the S&P
sits 1.99% from its own 52w high. Idiosyncrasy or travelling signal?

Base cell reproduces build_context_state P8:sma200_cross exactly: a close/200d
crossing in EITHER direction, declustered by _first_in_sessions(63), then
filtered to the DOWN side (close below the mean), on ^FCHI's OWN calendar.
Forward returns are lag=0 fwd_ret, the context convention.

The crossing question is measured by aligning each ^FCHI cross to the most
recent ^GSPC session AT OR BEFORE it (asof alignment, never the next US
session), then reading ^GSPC's own forward path from that bar.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import (load_prices, fwd_ret, summarize, era_split, sign_test,
                       cluster_note, show, pct_rank)

ASOF = pd.Timestamp("2026-09-09")


def first_in_sessions(mask: pd.Series, sessions: int) -> pd.Series:
    """build_context_state._first_in_sessions: keep a hit, ignore the next
    `sessions` trading days. `last` advances only on KEPT hits."""
    out = pd.Series(False, index=mask.index)
    pos = {d: i for i, d in enumerate(mask.index)}
    last = -10 ** 9
    for d in mask.index[mask.fillna(False).values]:
        p = pos[d]
        if p - last >= sessions:
            out.loc[d] = True
            last = p
    return out


def stats_line(dates, vals, label):
    s = summarize(np.asarray(vals), label)
    if not s["n"]:
        print(f"   {label}: n=0")
        return None
    up = int((np.asarray(vals) > 0).sum())
    dn = s["n"] - up
    p_up = sign_test(up, s["n"])
    p_dn = sign_test(dn, s["n"])
    disagree = ""
    if (s["mean_pct"] < 0 and up > dn) or (s["mean_pct"] > 0 and dn > up):
        disagree = "   <== MEAN AND RECORD DISAGREE IN SIGN"
    small = "   <== n<15" if s["n"] < 15 else ""
    print(f"   {label}: n={s['n']:3d} mean={s['mean_pct']:+.3f}% "
          f"med={s['median_pct']:+.3f}% hit={s['hit']:5.1f}% t={s['t']:+.2f} "
          f"rec {up}-{dn}  signp(up)={p_up:.4f} signp(down)={p_dn:.4f} "
          f"worst={s['worst_pct']:+.2f}% best={s['best_pct']:+.2f}%"
          f"{disagree}{small}")
    print(f"        cluster: {cluster_note(pd.DatetimeIndex(dates), np.asarray(vals), k=2)}")
    return s


px = load_prices(["^FCHI", "^GSPC", "SPY"])
fchi = px["^FCHI"]["Close"].astype(float).dropna()
gspc = px["^GSPC"]["Close"].astype(float).dropna()
spy = px["SPY"]["Close"].astype(float).dropna()

print("=" * 78)
print("HISTORY")
print("=" * 78)
for nm, s in (("^FCHI", fchi), ("^GSPC", gspc), ("SPY", spy)):
    print(f"  {nm}: {s.index.min().date()} .. {s.index.max().date()}  "
          f"{len(s)} sessions")

# ---------------------------------------------------------------- 1. base cell
ma = fchi.rolling(200).mean()
above = fchi > ma
cross = ((above != above.shift(1)) & ma.notna() & ma.shift(1).notna()).fillna(False)
kept = first_in_sessions(cross, 63)
side = np.sign(fchi - ma)
down_cross = kept & (side < 0)
anchors = fchi.index[down_cross.values]

dist200 = (fchi / ma - 1.0) * 100.0
rank21 = pct_rank(fchi, 21)

print()
print("=" * 78)
print("1. BASE CELL: ^FCHI first 200d cross in 63+ sessions, crossing DOWN")
print("=" * 78)
print(f"  raw crossings (either direction), 200d defined : {int(cross.sum())}")
print(f"  after first_in_sessions(63) declustering       : {int(kept.sum())}")
print(f"  of those, DOWN side (close < 200d)             : {len(anchors)}")
print(f"  sweep reported n=20 for h=1")
print(f"  TODAY {ASOF.date()} in the anchor set: {ASOF in anchors}")
if ASOF in anchors:
    print(f"    today's close {fchi.loc[ASOF]:.2f} vs 200d {ma.loc[ASOF]:.2f} "
          f"= {dist200.loc[ASOF]:+.2f}%   21d rank = {rank21.loc[ASOF]:.1f}")
print("\n  every anchor, with the state at the cross:")
print(f"    {'date':12s} {'dist to 200d':>13s} {'21d rank':>9s} {'h1':>8s} "
      f"{'h5':>8s} {'h21':>8s}")
f = {h: fwd_ret(fchi, h) for h in (1, 5, 21)}
for d in anchors:
    r = {h: f[h].get(d, np.nan) for h in (1, 5, 21)}
    print(f"    {str(d.date()):12s} {dist200.loc[d]:+12.2f}% {rank21.loc[d]:9.1f} "
          + " ".join(f"{100*r[h]:+7.2f}%" if pd.notna(r[h]) else f"{'  n/a':>8s}"
                     for h in (1, 5, 21)))

# --------------------------------------------------- 2. forward on the cell
print()
print("=" * 78)
print("2. ^FCHI FORWARD (lag=0 close-to-close) ON THE FULL CELL")
print("=" * 78)
cell = {}
for h in (1, 5, 21):
    v = f[h].reindex(anchors).dropna()
    cell[h] = v
    stats_line(v.index, v.values, f"h={h:2d}")
    show(era_split(v.index, v.values), f"   era split h={h}")
    base = f[h].dropna()
    print(f"   CTRL-b ^FCHI all days h={h}: n={len(base)} "
          f"mean={100*base.mean():+.3f}%  -> edge "
          f"{100*v.mean()-100*base.mean():+.3f}pp")

# --------------------------------------------------- 3. the crossing: US high
print()
print("=" * 78)
print("3. THE CROSSING: was the US within 3% of its own 252d high at the cross?")
print("=" * 78)
gs_hi = gspc.rolling(252).max()
gs_dist = (gspc / gs_hi - 1.0) * 100.0
spy_hi = spy.rolling(252).max()
spy_dist = (spy / spy_hi - 1.0) * 100.0
print(f"  today: ^GSPC {gs_dist.loc[ASOF]:+.2f}% from its 252d high, "
      f"SPY {spy_dist.loc[ASOF]:+.2f}%   (state note said SPY -1.99%)")
print("  alignment: each ^FCHI cross maps to the most recent ^GSPC session "
      "AT OR BEFORE it")

gidx = gspc.index
near, far, unmapped = [], [], []
gpos = {}
for d in anchors:
    loc = int(gidx.searchsorted(d, side="right")) - 1
    if loc < 0:
        unmapped.append(d)
        continue
    gd = gidx[loc]
    gpos[d] = gd
    dd = gs_dist.loc[gd]
    if pd.isna(dd):
        unmapped.append(d)
    elif dd >= -3.0:
        near.append(d)
    else:
        far.append(d)
print(f"  NEAR-HIGH side (^GSPC within 3% of its 252d high): {len(near)} anchors")
print("    " + ", ".join(str(d.date()) for d in near))
print(f"  FAR side (^GSPC more than 3% below): {len(far)} anchors")
print("    " + ", ".join(str(d.date()) for d in far))
if unmapped:
    print(f"  unmapped/undefined: {[str(d.date()) for d in unmapped]}")

for lbl, grp in (("NEAR-HIGH US", near), ("FAR-FROM-HIGH US", far)):
    print(f"\n  --- ^FCHI forward, {lbl} side (n_anchors={len(grp)}) ---")
    if len(grp) < 15:
        print(f"      *** n<15 on this side ({len(grp)} anchors): "
              f"anecdote, sign test only ***")
    for h in (1, 5, 21):
        v = f[h].reindex(pd.DatetimeIndex(grp)).dropna()
        stats_line(v.index, v.values, f"h={h:2d}")

print("\n  --- DID IT TRAVEL: ^GSPC's own forward from the SAME anchors, "
      "NEAR-HIGH side ---")
gf = {h: fwd_ret(gspc, h) for h in (1, 5, 21)}
gnear = pd.DatetimeIndex([gpos[d] for d in near])
print(f"      mapped ^GSPC sessions: {len(gnear)}")
if len(gnear) < 15:
    print(f"      *** n<15 ({len(gnear)}): anecdote ***")
for h in (1, 5, 21):
    v = gf[h].reindex(gnear).dropna()
    stats_line(v.index, v.values, f"^GSPC h={h:2d}")
    base = gf[h].dropna()
    if len(v):
        print(f"        CTRL-b ^GSPC all days: n={len(base)} "
              f"mean={100*base.mean():+.3f}% -> edge "
              f"{100*v.mean()-100*base.mean():+.3f}pp")
print("\n      per-episode ^GSPC h1 / h5 / h21 on the near-high side:")
for d in near:
    gd = gpos[d]
    print(f"        FCHI {str(d.date()):12s} -> GSPC {str(gd.date()):12s} "
          + " ".join(f"{100*gf[h].get(gd, np.nan):+7.2f}%"
                     if pd.notna(gf[h].get(gd, np.nan)) else f"{'  n/a':>8s}"
                     for h in (1, 5, 21)))

print("\n  --- ^GSPC forward, FAR side, for contrast ---")
gfar = pd.DatetimeIndex([gpos[d] for d in far])
for h in (1, 5, 21):
    v = gf[h].reindex(gfar).dropna()
    stats_line(v.index, v.values, f"^GSPC h={h:2d}")

# ---------------------------------------- 3b. is the 3% cut load-bearing?
print()
print("=" * 78)
print("3b. THRESHOLD SENSITIVITY on the near-high cut, and SPY vs ^GSPC")
print("=" * 78)
print("  The 3% figure is the cell map's, not a scanned optimum. If the split "
      "only\n  exists at 3% it is a fitted line; these rows say whether it "
      "moves.")
for thr in (1.0, 2.0, 3.0, 5.0, 10.0):
    grp = [d for d in anchors
           if d in gpos and pd.notna(gs_dist.loc[gpos[d]])
           and gs_dist.loc[gpos[d]] >= -thr]
    oth = [d for d in anchors
           if d in gpos and pd.notna(gs_dist.loc[gpos[d]])
           and gs_dist.loc[gpos[d]] < -thr]
    v = f[1].reindex(pd.DatetimeIndex(grp)).dropna()
    w = f[1].reindex(pd.DatetimeIndex(oth)).dropna()
    up = int((v.values > 0).sum())
    upw = int((w.values > 0).sum())
    print(f"  within {thr:4.1f}% of the 252d high: n={len(v):2d} "
          f"mean={100*v.mean():+.3f}% rec {up}-{len(v)-up} "
          f"signp(up)={sign_test(up, len(v)):.4f}   |   "
          f"beyond: n={len(w):2d} mean={100*w.mean():+.3f}% "
          f"rec {upw}-{len(w)-upw} signp(down)={sign_test(len(w)-upw, len(w)):.4f}")
print("\n  same split computed on SPY instead of ^GSPC (agreement check):")
sidx = spy.index
for thr in (3.0,):
    sn, sf = [], []
    for d in anchors:
        loc = int(sidx.searchsorted(d, side="right")) - 1
        if loc < 0 or pd.isna(spy_dist.iloc[loc]):
            continue
        (sn if spy_dist.iloc[loc] >= -thr else sf).append(d)
    v = f[1].reindex(pd.DatetimeIndex(sn)).dropna()
    w = f[1].reindex(pd.DatetimeIndex(sf)).dropna()
    up = int((v.values > 0).sum())
    upw = int((w.values > 0).sum())
    print(f"  SPY within {thr:.0f}%: n={len(v)} mean={100*v.mean():+.3f}% "
          f"rec {up}-{len(v)-up}   |   beyond: n={len(w)} "
          f"mean={100*w.mean():+.3f}% rec {upw}-{len(w)-upw}")
    print(f"  anchors classified the same way by SPY and ^GSPC: "
          f"{len(set(sn) & set(near))} of {len(near)} near-high, "
          f"{len(set(sf) & set(far))} of {len(far)} far")

print("\n  h=5 and h=21 sensitivity on the ^GSPC 3% cut, for completeness:")
for h in (5, 21):
    v = f[h].reindex(pd.DatetimeIndex(near)).dropna()
    w = f[h].reindex(pd.DatetimeIndex(far)).dropna()
    up = int((v.values > 0).sum())
    upw = int((w.values > 0).sum())
    print(f"    h={h:2d}: near n={len(v)} mean={100*v.mean():+.3f}% "
          f"rec {up}-{len(v)-up}   |   far n={len(w)} "
          f"mean={100*w.mean():+.3f}% rec {upw}-{len(w)-upw}")

# ------------------------------------------- 4. depth + already-washed-out
print()
print("=" * 78)
print("4. DEPTH BELOW THE 200d AND THE ALREADY-WASHED-OUT SUB-CELL")
print("=" * 78)
depths = dist200.reindex(anchors)
ranks = rank21.reindex(anchors)
print(f"  distance below the 200d at the cross (today {dist200.loc[ASOF]:+.2f}%):")
print(f"    mean {depths.mean():+.2f}%  median {depths.median():+.2f}%  "
      f"min {depths.min():+.2f}%  max {depths.max():+.2f}%")
shallower = int((depths > dist200.loc[ASOF]).sum())
print(f"    crosses SHALLOWER than today's {dist200.loc[ASOF]:+.2f}%: "
      f"{shallower} of {len(depths)}")
print(f"\n  21d return rank at the cross (today {rank21.loc[ASOF]:.1f}):")
print(f"    mean {ranks.mean():.1f}  median {ranks.median():.1f}  "
      f"min {ranks.min():.1f}  max {ranks.max():.1f}")
washed = pd.DatetimeIndex([d for d in anchors if pd.notna(rank21.loc[d])
                           and rank21.loc[d] <= 10.0])
print(f"\n  SUB-CELL 'already washed out' (21d rank <= 10 at the cross): "
      f"n_anchors={len(washed)}")
print("    " + (", ".join(str(d.date()) for d in washed) if len(washed) else "(none)"))
if len(washed):
    if len(washed) < 15:
        print(f"    *** n<15 ({len(washed)}): anecdote, sign test only ***")
    for h in (1, 5, 21):
        v = f[h].reindex(washed).dropna()
        stats_line(v.index, v.values, f"h={h:2d}")
    notw = anchors.difference(washed)
    print("    contrast, NOT washed out (21d rank > 10):")
    for h in (1, 5, 21):
        v = f[h].reindex(notw).dropna()
        stats_line(v.index, v.values, f"h={h:2d}")
