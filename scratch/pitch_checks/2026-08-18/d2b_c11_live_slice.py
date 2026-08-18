"""C11 round 2 -- the LIVE slice only.

Today is NOT the average trigger day: SPY is above its 200d, only 4 names fire,
and all 4 (CAT, CSCO, GOOG, INTC) are ABOVE their own 200d. The pooled cell in
d2 is name-day weighted toward SPY-below-200d tape (48.3% vs a 25.7% base). So
re-measure the cell conditioned on today's state, and re-run the alphabetical
placebo inside it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import strategy_config as sc

RANK_THR, H = 5.0, 5
uni = [t for t in sc.LIQUID_PLUS_COMMODITIES if t != "SOXS"]
px_all = close_panel(sorted(set(uni + ["SPY"])))
have = [t for t in uni if t in px_all.columns]
P, idx, spy = px_all[have], px_all.index, px_all["SPY"]

rank63 = P.apply(lambda s: pct_rank(s.dropna(), 63)).reindex(idx)
MASK = (rank63 <= RANK_THR) & rank63.notna()
above200 = P > P.rolling(200).mean()
spy_up = (spy > spy.rolling(200).mean()).reindex(idx)

F = P.shift(-(1 + H)) / P.shift(-1) - 1.0
U = F.mean(axis=1)
EX = F.sub(U, axis=0)

n_fire = MASK.sum(axis=1)
print(f"laggards firing today: {int(n_fire.iloc[-1])}  "
      f"(median across all dates {n_fire[n_fire > 0].median():.0f}, "
      f"90th pct {n_fire[n_fire > 0].quantile(0.9):.0f})")
print(f"dates with <=6 laggards firing: {int(((n_fire > 0) & (n_fire <= 6)).sum())}")

# ---------------------------------------------------------------------------
# 1. name-episode cell, sliced by today's two conditions
# ---------------------------------------------------------------------------
slices = {
    "ALL trigger name-days": MASK,
    "name ABOVE own 200d": MASK & above200,
    "name BELOW own 200d": MASK & ~above200,
    "SPY above 200d": MASK & pd.DataFrame(
        np.repeat(spy_up.values[:, None], len(have), axis=1),
        index=idx, columns=have),
    "LIVE: name>200d & SPY>200d": MASK & above200 & pd.DataFrame(
        np.repeat(spy_up.values[:, None], len(have), axis=1),
        index=idx, columns=have),
}
rows = []
for lbl, M in slices.items():
    vr, vx = [], []
    for t in have:
        d = idx[M[t].fillna(False).values & F[t].notna().values]
        if len(d) == 0:
            continue
        epi = declusters(d, H, idx)
        vr.append(F.loc[epi, t].values)
        vx.append(EX.loc[epi, t].values)
    vr = np.concatenate(vr) if vr else np.array([])
    vx = np.concatenate(vx) if vx else np.array([])
    rows.append(summarize(vr, f"{lbl} RAW"))
    rows.append(summarize(vx, f"{lbl} EXCESS"))
show(rows, f"1. name-episode slices, h={H}")

# ---------------------------------------------------------------------------
# 2. LIVE-shaped basket: narrow laggard tape (<=6 fire), SPY>200d, names>200d
# ---------------------------------------------------------------------------
live_M = MASK & above200
sel_ex, sel_raw, alp_ex, alp_raw, keep = [], [], [], [], []
alpha_order = sorted(have)
narrow = (n_fire > 0) & (n_fire <= 6) & spy_up
for d in idx[narrow.values]:
    fired = [t for t in have if bool(live_M.loc[d, t]) and np.isfinite(F.loc[d, t])]
    if not fired:
        continue
    k = min(4, len(fired))
    pick = list(rank63.loc[d, fired].sort_values().index[:k])
    avail = [t for t in alpha_order if np.isfinite(F.loc[d, t])][:k]
    sel_raw.append(F.loc[d, pick].mean())
    sel_ex.append(EX.loc[d, pick].mean())
    alp_raw.append(F.loc[d, avail].mean())
    alp_ex.append(EX.loc[d, avail].mean())
    keep.append(d)

keep = pd.DatetimeIndex(keep)
print(f"\nLIVE-shaped dates (SPY>200d, 1-6 laggards, >=1 above its 200d): {len(keep)}")
if len(keep) > 2:
    epi = declusters(keep, H, idx)
    s_r = pd.Series(sel_raw, index=keep).loc[epi]
    s_x = pd.Series(sel_ex, index=keep).loc[epi]
    a_r = pd.Series(alp_raw, index=keep).loc[epi]
    a_x = pd.Series(alp_ex, index=keep).loc[epi]
    show([summarize(s_r.values, "LIVE basket signal-selected RAW"),
          summarize(a_r.values, "LIVE basket alphabetical RAW"),
          summarize(s_x.values, "LIVE basket signal-selected EXCESS"),
          summarize(a_x.values, "LIVE basket alphabetical EXCESS")],
         f"2. live-shaped date-episodes N={len(epi)}")
    w = int((s_x.values > 0).sum())
    print(f"  selection premium (raw): {100*(s_r.mean()-a_r.mean()):+.3f}pp")
    print(f"  excess record {w}-{len(s_x)-w}, sign p={sign_test(w, len(s_x)):.4f}, "
          f"bootstrap P(mean<=0)={bootstrap_p_le0(s_x.values):.3f}")
    print("  concentration:", cluster_note(epi, s_x.values))
    show(era_split(epi, s_x.values), "  era split (live-shaped excess)")

# ---------------------------------------------------------------------------
# 3. horizon scan on the live-shaped cell
# ---------------------------------------------------------------------------
print("\n=== 3. horizon scan, live-shaped basket EXCESS ===")
for h in (1, 2, 3, 5, 10):
    Fh = P.shift(-(1 + h)) / P.shift(-1) - 1.0
    Uh = Fh.mean(axis=1)
    EXh = Fh.sub(Uh, axis=0)
    vals, dts = [], []
    for d in idx[narrow.values]:
        fired = [t for t in have if bool(live_M.loc[d, t]) and np.isfinite(Fh.loc[d, t])]
        if not fired:
            continue
        pick = list(rank63.loc[d, fired].sort_values().index[:min(4, len(fired))])
        vals.append(EXh.loc[d, pick].mean())
        dts.append(d)
    if not vals:
        continue
    dts = pd.DatetimeIndex(dts)
    epi = declusters(dts, h, idx)
    s = pd.Series(vals, index=dts).loc[epi]
    r = summarize(s.values, f"h={h} live-shaped excess")
    print(f"  {r['label']}: N={r['n']} mean {r['mean_pct']:+.3f}% hit {r['hit']:.1f}% "
          f"t={r['t']:+.2f}")
