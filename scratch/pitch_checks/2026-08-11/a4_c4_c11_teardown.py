"""C4 / C11 teardown: the trigger definition, the threshold sweep, the
drawdown carve, cluster depth, LOYO and the tail.

Opens with the finding that reorders everything else. `02_price_state_recon.py`
builds its metals trigger as

    rank(r(px["GDX"], 5), 5)      where rank(s, n) = pct_rank(s, n)
                                  and  pct_rank(s, n) = s.pct_change(n).rolling(252).rank(pct=True)

so the series being ranked is ``GDX.pct_change(5).pct_change(5)`` -- the
5-day percent CHANGE OF the 5-day return. That is a second difference taken
on a series that crosses zero constantly, so a small move through zero
produces an arbitrarily large "rank". It is not "GDX 5d return rank", which
is what the surface map, the candidate text and today's tape reading all say.
Every number quoted for C4, C11 (and C8, same construction on XLV) inherits
this.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, fwd_lag, declusters, summarize, sign_test,  # noqa: E402
                       pct_rank, bootstrap_p_le0, show)

warnings.filterwarnings("ignore")

TK = ["GDX", "GLD", "SLV", "SPY"]
px = close_panel(TK)
idx = px.index
H = 5

gdx = px["GDX"]
correct = pct_rank(gdx, 5)                       # 5d return, ranked
buggy = pct_rank(gdx.pct_change(5), 5)           # what the recon actually built

slv = fwd_lag(px["SLV"], H, lag=1)
gld = fwd_lag(px["GLD"], H, lag=1)
valid = slv.notna() & gld.notna()

print("=" * 96)
print("TEST 4a  THE TRIGGER IN THE RECON IS NOT THE TRIGGER IN THE CANDIDATE")
print("=" * 96)
mc = (correct >= 95).fillna(False) & valid
mb = (buggy >= 95).fillna(False) & valid
dc, db = idx[mc.values], idx[mb.values]
ec = declusters(dc, 5, idx)
eb = declusters(db, 5, idx)
print(f"  stated trigger  GDX 5d return rank >= 95 : {len(dc)} days, {len(ec)} episodes")
print(f"  recon's trigger pct_change(5) of the 5d return, ranked >= 95 : "
      f"{len(db)} days, {len(eb)} episodes   <-- the recon's N=81")
inter = dc.intersection(db)
print(f"  overlap: {len(inter)} days = {100*len(inter)/max(len(db),1):.1f}% of the recon's "
      f"population, {100*len(inter)/max(len(dc),1):.1f}% of the real one")
last = gdx.dropna().index[-1]
print(f"\n  TODAY ({last.date()}): real 5d rank = {correct.loc[last]:.1f}  "
      f"(5d ret {100*gdx.pct_change(5).loc[last]:+.2f}%)")
print(f"                  recon's statistic = {buggy.loc[last]:.1f}  "
      f"-> today is {'IN' if buggy.loc[last] >= 95 else 'NOT IN'} the population "
      f"the recon measured")
rows = []
for lbl, e in (("recon trigger (buggy)", eb), ("stated trigger (correct)", ec)):
    for nm, leg in (("SLV", slv), ("GLD", gld)):
        s = summarize(leg.loc[e].values, f"{nm} | {lbl}")
        base = leg[valid]
        s["own_drift_pct"] = round(100 * base.mean(), 3)
        s["excess_pct"] = round(s["mean_pct"] - 100 * base.mean(), 3)
        rows.append(s)
show(rows, "h=5 episodes under each definition")

# ---------------------------------------------------------------------------
print()
print("=" * 96)
print("TEST 4b  DEFINITION FRAGILITY on the CORRECT trigger: rank + raw 5d sweeps")
print("=" * 96)
r5 = gdx.pct_change(5)
rows = []
for thr in (90, 93, 95, 97, 99):
    m = (correct >= thr).fillna(False) & valid
    e = declusters(idx[m.values], 5, idx)
    for nm, leg in (("SLV", slv), ("GLD", gld)):
        s = summarize(leg.loc[e].values, f"{nm} rank>={thr}")
        s["excess_pct"] = round(s["mean_pct"] - 100 * leg[valid].mean(), 3)
        s["signp"] = round(sign_test(int((leg.loc[e].values > 0).sum()), len(e)), 4)
        rows.append(s)
show(rows, "rank threshold sweep (episodes, h=5)")

rows = []
for thr in (0.05, 0.08, 0.10, 0.12, 0.15, 0.18):
    m = (r5 >= thr).fillna(False) & valid
    e = declusters(idx[m.values], 5, idx)
    for nm, leg in (("SLV", slv), ("GLD", gld)):
        s = summarize(leg.loc[e].values, f"{nm} 5d>={100*thr:.0f}%")
        s["excess_pct"] = round(s["mean_pct"] - 100 * leg[valid].mean(), 3)
        rows.append(s)
show(rows, "raw 5d-return threshold sweep (episodes, h=5).  TODAY GDX 5d = +18.99%")

print("\n  --- where does TODAY sit, and does the top of the distribution behave "
      "like the rest? ---")
m95 = (correct >= 95).fillna(False) & valid
d95 = idx[m95.values]
q = r5.loc[d95]
print(f"  among rank>=95 days the 5d return has median {100*q.median():+.2f}%, "
      f"90th pctile {100*q.quantile(0.90):+.2f}%, max {100*q.max():+.2f}%")
print(f"  today's +18.99% is the {100*(q < 0.1899).mean():.1f}th percentile OF the "
      f"trigger population itself")
for lo, hi in ((0.0, 0.06), (0.06, 0.10), (0.10, 0.14), (0.14, 1.0)):
    m = (r5 >= lo) & (r5 < hi) & (correct >= 95) & valid
    e = declusters(idx[m.fillna(False).values], 5, idx)
    if len(e) < 3:
        print(f"  5d in [{100*lo:.0f}%,{100*hi:.0f}%): N={len(e)} too few")
        continue
    a, b = slv.loc[e].values, gld.loc[e].values
    print(f"  5d in [{100*lo:>3.0f}%,{100*hi:>4.0f}%): N={len(e):<3} "
          f"SLV {100*a.mean():+7.3f}% hit {100*(a>0).mean():5.1f}%  |  "
          f"GLD {100*b.mean():+7.3f}% hit {100*(b>0).mean():5.1f}%")

# ---------------------------------------------------------------------------
print()
print("=" * 96)
print("TEST 5a  REGISTRY COLLISION: the drawdown carve, re-derived on THIS trigger")
print("=" * 96)
print("  registry (2026-08-10, c7_slv_drawdown_thrust.py): 'deep-dd pays +1.378% at")
print("  h=10 against +1.780% for the same thrust near a 52w high ... a U-shaped noise")
print("  carve.'  SLV today is 43.7% below its 52w high.")
slv_dd = px["SLV"] / px["SLV"].rolling(252).max() - 1.0
last_dd = slv_dd.loc[last]
print(f"  SLV drawdown today = {100*last_dd:.1f}%")
for h in (5, 10):
    leg = fwd_lag(px["SLV"], h, lag=1)
    v2 = leg.notna()
    rows = []
    for lbl, sel in (("deep dd (<= -30%)", slv_dd <= -0.30),
                     ("mid dd (-30%..-10%)", (slv_dd > -0.30) & (slv_dd <= -0.10)),
                     ("near high (> -10%)", slv_dd > -0.10)):
        m = (correct >= 95) & sel & v2
        e = declusters(idx[m.fillna(False).values], 5, idx)
        if len(e) < 3:
            rows.append({"label": f"h={h} {lbl}", "n": len(e)})
            continue
        s = summarize(leg.loc[e].values, f"h={h} {lbl}")
        s["excess_pct"] = round(s["mean_pct"] - 100 * leg[v2].mean(), 3)
        rows.append(s)
    show(rows, f"SLV drawdown carve on the GDX thrust, h={h}")

# ---------------------------------------------------------------------------
print()
print("=" * 96)
print("TEST 5b  CLUSTER DEPTH -- today is several sessions into the run")
print("=" * 96)
tf = (correct >= 95).fillna(False)
depth = tf.groupby((~tf).cumsum()).cumsum()   # consecutive trigger days ending here
print(f"  today's depth (consecutive rank>=95 sessions through {last.date()}) = "
      f"{int(depth.loc[last])}")
print(f"  trigger population median depth = {depth[tf].median():.0f}")
for nm, leg in (("SLV", slv), ("GLD", gld)):
    rows = []
    for lbl, sel in (("depth 1 (fresh)", depth == 1), ("depth 2-3", (depth >= 2) & (depth <= 3)),
                     ("depth >= 4", depth >= 4), ("depth >= 5", depth >= 5)):
        m = tf & sel & valid
        d = idx[m.values]
        if len(d) < 4:
            rows.append({"label": f"{nm} {lbl}", "n": len(d)})
            continue
        s = summarize(leg.loc[d].values, f"{nm} {lbl}")
        s["excess_pct"] = round(s["mean_pct"] - 100 * leg[valid].mean(), 3)
        s["signp"] = round(sign_test(int((leg.loc[d].values > 0).sum()), len(d)), 4)
        rows.append(s)
    show(rows, f"{nm} h=5 by cluster depth (day level -- depth cells cannot be declustered)")

# ---------------------------------------------------------------------------
print()
print("=" * 96)
print("TEST 5c  LEAVE-ONE-YEAR-OUT and the year histogram (episodes, h=5)")
print("=" * 96)
e = declusters(idx[((correct >= 95).fillna(False) & valid).values], 5, idx)
for nm, leg in (("SLV", slv), ("GLD", gld)):
    v = leg.loc[e]
    yrs = v.index.year
    by = v.groupby(yrs).agg(["count", "mean"])
    print(f"\n  {nm} by year (mean %):")
    print("   " + "  ".join(f"{y}:{100*r['mean']:+.2f}({int(r['count'])})"
                            for y, r in by.iterrows()))
    loyo = []
    for y in sorted(set(yrs)):
        keep = v[yrs != y]
        loyo.append((y, 100 * keep.mean(), len(keep)))
    worst = min(loyo, key=lambda x: x[1])
    best = max(loyo, key=lambda x: x[1])
    print(f"  LOYO mean range: {worst[1]:+.3f}% (drop {worst[0]}) .. "
          f"{best[1]:+.3f}% (drop {best[0]});  full-sample {100*v.mean():+.3f}%")
    own = 100 * leg[valid].mean()
    n_below = sum(1 for _, m_, _ in loyo if m_ < own)
    print(f"  own-drift control = {own:+.3f}%;  LOYO folds BELOW the control: "
          f"{n_below}/{len(loyo)}")

# ---------------------------------------------------------------------------
print()
print("=" * 96)
print("TEST 6  TAIL at 30 bps risk sizing")
print("=" * 96)
atr_pct = 3.31  # SLV ATR as % of price, from today's tape
for nm, leg, ap in (("SLV", slv, 3.31), ("GLD", gld, 1.10)):
    v = leg.loc[e].values
    worst = v.min()
    d_worst = e[int(np.argmin(v))]
    # 30 bps risk with a 1 ATR stop-equivalent risk unit -> notional = 0.30% / ATR%
    notional_pct_nav = 0.30 / ap
    print(f"  {nm}: worst episode {100*worst:.2f}% on {d_worst.date()}; ATR {ap:.2f}% of "
          f"price -> 30 bps risk = {notional_pct_nav*100:.1f}% NAV notional -> "
          f"worst-episode NAV hit {100*worst*notional_pct_nav:.2f}% "
          f"({100*worst*notional_pct_nav/0.30:.2f}R)")
    tail = v[v < np.quantile(v, 0.05)]
    print(f"     5% tail mean {100*tail.mean():.2f}% = "
          f"{100*tail.mean()*notional_pct_nav/0.30:.2f}R")
