"""C5b -- round-2 attack on the ONE survivor-shaped cell in the whole family.

C5's pooled numbers blend a -1.0x instrument (pre 2018-02-28) with the -0.5x
product you can actually buy. On the LIVE instrument only, the live band is:

    SVXY, ^SKEW r21 >= 98, h=10, post-2018-02-28:
    +1.415% raw, +0.718pp excess, n=27, 19-8, sign p 0.0261

That is the last thing standing after C4 and C6 died. This script tries to kill
it four ways that do not depend on N:
  1. THRESHOLD LADDER inside the -0.5x era. If the cell exists at >=98 and
     nowhere else, the threshold was scanned, not specified.
  2. CONCENTRATION. Which years, and what is left after the best one goes.
  3. THE BETA RESIDUAL inside the -0.5x era. The whole premise of the
     instrument_translation axis is that SVXY expresses something SPY does not.
     If the residual is a coin flip, the cell is a 2.3x levered copy of C4,
     which is already dead.
  4. TAIL. Worst episode, and the paths of the losers, with PPI at +2 td, CPI
     at +3 td and FOMC at +6 td inside a 10-session hold.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _c456_common import (  # noqa: E402
    align, cluster_note, declusters, fwd_lag, load_prices, np, pct_rank, pd,
    row, show, sign_test, summarize,
)
from pitch_lab import episode_paths, event_in_window  # noqa: E402

REBAL = pd.Timestamp("2018-02-28")
PX = load_prices(["SPY", "SVXY", "^SKEW"])
IDX = PX["SPY"].index
SKEW, SPY = PX["^SKEW"]["Close"], PX["SPY"]["Close"]
r21 = pct_rank(SKEW, 21)

print("=" * 78)
print("C5b  SVXY on skew r21, LIVE (-0.5x) INSTRUMENT ONLY, post 2018-02-28")
print("=" * 78)


def svxy(h):
    return align(fwd_lag(PX["SVXY"]["Close"], h, 1), IDX)


LIVE_ERA = pd.Series(IDX >= REBAL, index=IDX)

# ------------------------------------------------- 1. threshold ladder, live era
print("\n1. THRESHOLD LADDER inside the -0.5x era (was 98 specified or scanned?)")
rows = []
for h in (3, 5, 7, 10):
    ret = svxy(h)
    valid = ret.dropna().index
    valid_live = valid[valid >= REBAL]
    base = float(ret.loc[valid_live].mean())
    for th in (80, 85, 90, 95, 98):
        m = align(r21 >= th, IDX).fillna(0).astype(bool) & LIVE_ERA
        t = pd.DatetimeIndex(IDX[m.values]).intersection(valid_live)
        epi = declusters(t, h, valid_live)
        if len(epi) == 0:
            continue
        v = ret.loc[epi].values
        w = int((v > 0).sum())
        rows.append({"h": h, "thresh": f">= {th}", "n": len(epi),
                     "mean_pct": round(100 * v.mean(), 3),
                     "era_base_pct": round(100 * base, 3),
                     "excess_pct": round(100 * (v.mean() - base), 3),
                     "med_pct": round(100 * float(np.median(v)), 3),
                     "hit": round(100 * float((v > 0).mean()), 1),
                     "worst_pct": round(100 * v.min(), 2),
                     "rec": f"{w}-{len(epi) - w}",
                     "sign_p": round(sign_test(w, len(epi)), 4)})
    for lo, hi in [(85, 90), (90, 95), (95, 98), (98, 101)]:
        m = align((r21 >= lo) & (r21 < hi), IDX).fillna(0).astype(bool) & LIVE_ERA
        t = pd.DatetimeIndex(IDX[m.values]).intersection(valid_live)
        epi = declusters(t, h, valid_live)
        if len(epi) == 0:
            continue
        v = ret.loc[epi].values
        w = int((v > 0).sum())
        rows.append({"h": h, "thresh": f"band [{lo},{hi})", "n": len(epi),
                     "mean_pct": round(100 * v.mean(), 3),
                     "era_base_pct": round(100 * base, 3),
                     "excess_pct": round(100 * (v.mean() - base), 3),
                     "med_pct": round(100 * float(np.median(v)), 3),
                     "hit": round(100 * float((v > 0).mean()), 1),
                     "worst_pct": round(100 * v.min(), 2),
                     "rec": f"{w}-{len(epi) - w}",
                     "sign_p": round(sign_test(w, len(epi)), 4)})
print(pd.DataFrame(rows).to_string(index=False))

# --------------------------------------------------------- 2. the cell itself
H = 10
ret = svxy(H)
valid = ret.dropna().index
valid_live = valid[valid >= REBAL]
base = float(ret.loc[valid_live].mean())
m98 = align(r21 >= 98, IDX).fillna(0).astype(bool) & LIVE_ERA
t98 = pd.DatetimeIndex(IDX[m98.values]).intersection(valid_live)
epi = declusters(t98, H, valid_live)
ep = ret.loc[epi]

print(f"\n2. CONCENTRATION, the cell (r21>=98, -0.5x era, h={H}), n={len(epi)}")
print("  " + cluster_note(epi, ep.values, k=2))
byyr = ep.groupby(epi.year).agg(["size", "mean", "sum"])
byyr.columns = ["n", "mean", "sum"]
byyr["mean_pct"] = (100 * byyr["mean"]).round(3)
byyr["sum_pct"] = (100 * byyr["sum"]).round(2)
print(byyr[["n", "mean_pct", "sum_pct"]].to_string())
for drop in (1, 2):
    top = byyr["sum"].sort_values(ascending=False).head(drop).index
    keep = ~np.isin(epi.year, top)
    v = ep.values[keep]
    w = int((v > 0).sum())
    print(f"  drop best {drop} year(s) {list(top)} -> n={len(v)} mean "
          f"{100 * v.mean():+.3f}%  excess {100 * (v.mean() - base):+.3f}pp  "
          f"{w}-{len(v) - w}  sign p {sign_test(w, len(v)):.4f}")
print("  episodes: " + ", ".join(f"{d.date()}:{100 * r:+.1f}%"
                                 for d, r in zip(epi, ep.values)))
for g in (10, 21, 42):
    e = declusters(t98, g, valid_live)
    v = ret.loc[e].values
    w = int((v > 0).sum())
    print(f"  min_gap {g:>2d} -> n={len(e):3d} mean {100 * v.mean():+.3f}% "
          f"excess {100 * (v.mean() - base):+.3f}pp {w}-{len(v) - w}")

# ------------------------------------------------- 3. the beta residual, live era
print("\n3. BETA RESIDUAL inside the -0.5x era. Is there a vol-specific view?")
for h in (5, 10):
    rs = align(fwd_lag(SPY, h, 1), IDX)
    d = pd.DataFrame({"spy": rs, "svxy": svxy(h)}).dropna()
    d = d[d.index >= REBAL]
    b, a = np.polyfit(d["spy"], d["svxy"], 1)
    resid = d["svxy"] - (a + b * d["spy"])
    r2 = float(np.corrcoef(d["spy"], d["svxy"])[0, 1] ** 2)
    print(f"\n  h={h}: SVXY = {100 * a:+.3f}% + {b:.2f} x SPY  R2 {r2:.3f} (n={len(d)})")
    for th in (90, 95, 98):
        mm = align(r21 >= th, IDX).fillna(0).astype(bool) & LIVE_ERA
        t = pd.DatetimeIndex(IDX[mm.values]).intersection(d.index)
        e = declusters(t, h, d.index)
        v = resid.loc[e].values
        w = int((v > 0).sum())
        print(f"    r21>={th}: RESIDUAL mean {100 * v.mean():+.3f}%  median "
              f"{100 * float(np.median(v)):+.3f}%  n={len(v)}  {w}-{len(v) - w}"
              f"  sign p {sign_test(w, len(v)):.4f}")
    # and the raw SPY leg on the same episodes, for the like-for-like read
    for th in (98,):
        mm = align(r21 >= th, IDX).fillna(0).astype(bool) & LIVE_ERA
        t = pd.DatetimeIndex(IDX[mm.values]).intersection(d.index)
        e = declusters(t, h, d.index)
        sv, sp = d.loc[e, "svxy"].values, d.loc[e, "spy"].values
        base_sp = float(d["spy"].mean())
        print(f"    like-for-like on the SAME {len(e)} episodes: SVXY "
              f"{100 * sv.mean():+.3f}% | SPY {100 * sp.mean():+.3f}% "
              f"(SPY era base {100 * base_sp:+.3f}%, SPY excess "
              f"{100 * (sp.mean() - base_sp):+.3f}pp)")

# -------------------------------------------------------------- 4. tail + events
print("\n4. TAIL and EVENT EXPOSURE on the cell")
print(f"  worst episode {100 * ep.min():+.2f}% on {epi[int(np.argmin(ep.values))].date()}"
      f"   sd {100 * ep.std(ddof=1):.2f}%   "
      f"P(<-5%) {100 * float((ep.values < -0.05).mean()):.1f}%")
fl = event_in_window(epi, IDX, H, 1, ("cpi", "ppi", "fomc_decision"))
show([summarize(ep.values[fl], f"print/FOMC IN window (N={int(fl.sum())})"),
      summarize(ep.values[~fl], f"OUT (N={int((~fl).sum())})")],
     "cpi+ppi+fomc in the 10-session hold (today has all three)")
losers = epi[ep.values < 0]
paths = episode_paths(pd.DataFrame({"SVXY": PX["SVXY"]["Close"]}).reindex(IDX).ffill(),
                      losers, [("SVXY", 1.0)], H, 1)
if len(paths):
    print(f"\n  LOSER PATHS (n={len(paths)}), mean cumulative % by session:")
    print((100 * paths.mean()).round(2).to_string())
    print(f"  worst single-session drawdown inside a losing hold: "
          f"{100 * paths.min().min():.2f}%")
    print(f"  of {len(paths)} losers, {int((paths.iloc[:, 2] > 0).sum())} were "
          f"GREEN at session 3 and still finished red")

print("\n5. MIDTERM inside the live era (2018, 2022, 2026 are the midterms here)")
for h in (5, 10):
    r = svxy(h)
    v_ = r.dropna().index
    v_live = v_[v_ >= REBAL]
    b_ = float(r.loc[v_live].mean())
    mm = align(r21 >= 98, IDX).fillna(0).astype(bool) & LIVE_ERA
    t = pd.DatetimeIndex(IDX[mm.values]).intersection(v_live)
    e = declusters(t, h, v_live)
    x = r.loc[e]
    for lbl, sel in (("MIDTERM", e.year % 4 == 2), ("non-midterm", e.year % 4 != 2)):
        v = x.values[sel]
        if not len(v):
            continue
        w = int((v > 0).sum())
        print(f"  h={h:>2d} {lbl:<12s} n={len(v):2d} mean {100 * v.mean():+.3f}% "
              f"excess {100 * (v.mean() - b_):+.3f}pp med "
              f"{100 * float(np.median(v)):+.3f}% {w}-{len(v) - w} "
              f"sign p {sign_test(w, len(v)):.4f}")

print("\nDONE C5b")
