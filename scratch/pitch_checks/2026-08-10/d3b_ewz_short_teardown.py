"""D3 round 2 -- the ONLY surviving sign is the short, so tear the short down.

Round 1 (d3_ewz_decoupler.py) killed the CATCH-UP reading outright: long EWZ
on the decoupling trigger is -0.710% over h=5 episodes against a SPY-gate
control of -0.132% and an all-days drift of +0.301%. -14.2x cost.

That leaves the INFORMATION reading, which was pre-registered in the same
breath: the market that will not rally in the best possible tape keeps
lagging -> SHORT EWZ / LONG SPY. Round 1's eq-dollar spread inverts to
+0.862% per episode (22-14) with the SPY-gate control at +0.408%. That is
not nothing, so it gets the hostile treatment rather than a pass.

The three questions that decide it:
  A. MONOTONICITY. The trigger is "EWZ 5d < 0". TODAY IS -3.57%. Does the
     edge grow as the decoupling deepens, or does it live in the shallow
     bucket and vanish (or invert) at today's reading? Round 1's sensitivity
     table already smells: the "EWZ 5d < -3.5%" variant came back +1.042%
     for the LONG, i.e. NEGATIVE for the short.
  B. CONCENTRATION. Drop-worst / drop-best on the short's episodes.
  C. Does the short survive a borrow-inclusive cost, and is the drift it
     harvests just EWZ's structural underperformance of SPY?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TK = ["EWZ", "EEM", "SPY"]
px = close_panel(TK).dropna()
px = px.loc[px.index >= "2003-05-01"]
idx = px.index

r5 = {t: px[t].pct_change(5) for t in TK}
d52 = {t: px[t] / px[t].rolling(252).max() - 1.0 for t in TK}
spy_gate = ((d52["SPY"] >= -0.010) & (r5["SPY"] >= 0.020)).fillna(False)

H = 5
LEGS_SHORT = [("EWZ", -1.0), ("SPY", 1.0)]      # THE information reading
ret = vehicle_ret(px, LEGS_SHORT, H, 1)
val = ret.notna()

print("TODAY: EWZ 5d = %+.2f%%   rank5d %.1f" %
      (100 * r5["EWZ"].iloc[-1], pct_rank(px["EWZ"], 5).iloc[-1]))

# ------------------------------------------------ A. MONOTONICITY IN DEPTH
print("\n" + "=" * 78)
print("A. MONOTONICITY -- does the short's edge grow with the depth of the")
print("   decoupling?  Today's reading is EWZ 5d = -3.57%.")
print("=" * 78)
buckets = [(-99.0, -0.030), (-0.030, -0.020), (-0.020, -0.010),
           (-0.010, 0.0)]
tot_lo, tot_hi = [], []
for lo, hi in buckets:
    m = (spy_gate & (r5["EWZ"] >= lo) & (r5["EWZ"] < hi)).fillna(False)
    s = idx[m.values & val.values]
    if len(s) == 0:
        print(f"  EWZ 5d in [{100*lo:6.1f}%, {100*hi:5.1f}%): no triggers")
        continue
    e = declusters(s, H, idx)
    v = ret.loc[e].values
    w = int((v > 0).sum())
    print(f"  EWZ 5d in [{100*lo:6.1f}%, {100*hi:5.1f}%): SHORT-spread "
          f"{100*v.mean():+.3f}%  (N_epi={len(e)}, {w}-{len(v)-w}, "
          f"sign p {sign_test(w, len(v)):.3f})  median {100*np.median(v):+.3f}%")

print("\n  CUMULATIVE from the deep tail inward (the honest read of 'today'):")
for thr in (-0.035, -0.030, -0.025, -0.020, -0.015, -0.010, -0.005, 0.0):
    m = (spy_gate & (r5["EWZ"] < thr)).fillna(False)
    s = idx[m.values & val.values]
    if len(s) < 2:
        print(f"  EWZ 5d < {100*thr:5.1f}%: N={len(s)} days, too few to decluster")
        continue
    e = declusters(s, H, idx)
    v = ret.loc[e].values
    w = int((v > 0).sum())
    print(f"  EWZ 5d < {100*thr:5.1f}%: SHORT-spread {100*v.mean():+.3f}%  "
          f"(N_epi={len(e)}, {w}-{len(v)-w}, sign p {sign_test(w, len(v)):.3f}, "
          f"boot P<=0 {bootstrap_p_le0(v):.3f})")

print("\n  SAME on rank5d (today = 10.7):")
rk = pct_rank(px["EWZ"], 5)
for thr in (10, 15, 20, 30, 50):
    m = (spy_gate & (rk < thr)).fillna(False)
    s = idx[m.values & val.values]
    if len(s) < 2:
        print(f"  EWZ rank5d < {thr:3d}: N={len(s)} days")
        continue
    e = declusters(s, H, idx)
    v = ret.loc[e].values
    w = int((v > 0).sum())
    print(f"  EWZ rank5d < {thr:3d}: SHORT-spread {100*v.mean():+.3f}%  "
          f"(N_epi={len(e)}, {w}-{len(v)-w}, sign p {sign_test(w, len(v)):.3f})"
          f"  dates {[str(d.date()) for d in e]}")

# --------------------------------------------------- B. CONCENTRATION
print("\n" + "=" * 78)
print("B. CONCENTRATION on the headline short cell (EWZ 5d<0, h=5)")
print("=" * 78)
m = (spy_gate & (r5["EWZ"] < 0)).fillna(False)
e = declusters(idx[m.values & val.values], H, idx)
v = ret.loc[e].values
w = int((v > 0).sum())
print(f"  headline SHORT {100*v.mean():+.3f}%  N={len(v)}  {w}-{len(v)-w}  "
      f"sign p {sign_test(w, len(v)):.4f}  boot P<=0 {bootstrap_p_le0(v):.3f}")
for k in (1, 2, 3):
    ordr = np.argsort(-v)
    drop = v[ordr[k:]]
    print(f"  drop-best-{k}: {100*drop.mean():+.3f}%  "
          f"(dropped {[str(pd.Timestamp(e[i]).date()) for i in ordr[:k]]} "
          f"= {[round(100*v[i], 2) for i in ordr[:k]]})")
yrs = pd.DatetimeIndex(e).year
byyr = pd.Series(100 * v, index=yrs).groupby(level=0).sum()
print(f"\n  by year (pp, short side): {byyr.round(2).to_dict()}")
print(f"  top-2 years {byyr.nlargest(2).round(2).to_dict()} = "
      f"{100*byyr.nlargest(2).sum()/byyr.sum():.0f}% of the +{byyr.sum():.2f}pp total")
for cut in ("2013-01-01", "2018-01-01", "2021-01-01"):
    show(era_split(e, v, cut), f"era {cut[:4]} (short)")
mid = yrs % 4 == 2
print(f"\n  MIDTERM (2026 is one): {100*v[mid].mean():+.3f}% (N={int(mid.sum())})"
      f"   non-midterm {100*v[~mid].mean():+.3f}% (N={int((~mid).sum())})")

# ------------------------------------------- day-vs-episode sign stability
print("\n  DAY-vs-EPISODE stability, pre-2018 only (round 1 flagged a flip):")
sd = idx[m.values & val.values]
vd = ret.loc[sd].values
pre = pd.DatetimeIndex(sd) < pd.Timestamp("2018-01-01")
pre_e = pd.DatetimeIndex(e) < pd.Timestamp("2018-01-01")
print(f"    day-level pre-2018 {100*vd[pre].mean():+.3f}% (N={int(pre.sum())})"
      f"   episode pre-2018 {100*v[pre_e].mean():+.3f}% (N={int(pre_e.sum())})"
      f"   <- signs {'AGREE' if np.sign(vd[pre].mean())==np.sign(v[pre_e].mean()) else 'DISAGREE'}")

# ------------------------------------- C. is it just structural drift?
print("\n" + "=" * 78)
print("C. IS THE SHORT JUST HARVESTING EWZ's STRUCTURAL UNDERPERFORMANCE?")
print("=" * 78)
base = ret[val]
lo = idx[m.values & val.values][0]
same = ret[val & (idx >= lo)]
gate_only = declusters(idx[spy_gate.values & val.values & (idx >= lo)], H, idx)
print(f"  unconditional SHORT-spread, full history {100*base.mean():+.3f}%")
print(f"  unconditional SHORT-spread, trigger span {100*same.mean():+.3f}%")
print(f"  SPY-gate-only episodes            {100*ret.loc[gate_only].mean():+.3f}% "
      f"(N={len(gate_only)})")
print(f"  headline trigger episodes         {100*v.mean():+.3f}% (N={len(v)})")
print(f"  -> EWZ leg adds {100*(v.mean()-ret.loc[gate_only].mean()):+.3f}pp "
      f"over the SPY gate alone; the SPY gate itself adds "
      f"{100*(ret.loc[gate_only].mean()-same.mean()):+.3f}pp over doing nothing.")
print(f"  CAGR-equivalent of the unconditional drift: EWZ/SPY total ratio "
      f"{px['EWZ'].iloc[-1]/px['EWZ'].iloc[0]:.2f}x vs "
      f"{px['SPY'].iloc[-1]/px['SPY'].iloc[0]:.2f}x over "
      f"{(idx[-1]-idx[0]).days/365.25:.1f}y")

print("\n  COST including borrow (short EWZ):")
edge_bps = 100 * v.mean() * 100
for borrow_ann in (0.5, 1.5, 3.0):
    borrow_bps = borrow_ann * 100 * (H + 1) / 252
    tot = 5.0 + 2.0 + borrow_bps
    print(f"    borrow {borrow_ann:.1f}%/yr -> {borrow_bps:.1f} bps over "
          f"{H+1} sessions; total {tot:.1f} bps vs edge {edge_bps:.1f} bps "
          f"= {edge_bps/tot:.1f}x")

# -------------------------------------------- what today's cell says
print("\n" + "=" * 78)
print("D. THE CELL THAT ACTUALLY DESCRIBES TODAY")
print("=" * 78)
m_today = (spy_gate & (r5["EWZ"] < -0.035)).fillna(False)
s = idx[m_today.values & val.values]
e_t = declusters(s, H, idx)
v_t = ret.loc[e_t].values
w = int((v_t > 0).sum())
print(f"  EWZ 5d < -3.5% (today -3.57%) under the SPY gate:")
print(f"    SHORT-spread {100*v_t.mean():+.3f}%  N={len(v_t)}  {w}-{len(v_t)-w}"
      f"  sign p {sign_test(w, len(v_t)):.3f}")
print(f"    LONG-spread  {-100*v_t.mean():+.3f}%  <- the sign the headline "
      f"cell says is wrong")
print(f"    episodes: {[(str(pd.Timestamp(d).date()), round(100*x,2)) for d,x in zip(e_t, v_t)]}")
lo_only = (spy_gate & (r5["EWZ"] < 0) & (r5["EWZ"] >= -0.02)).fillna(False)
e_l = declusters(idx[lo_only.values & val.values], H, idx)
v_l = ret.loc[e_l].values
w = int((v_l > 0).sum())
print(f"\n  SHALLOW bucket, EWZ 5d in [-2%, 0):  SHORT-spread "
      f"{100*v_l.mean():+.3f}%  N={len(v_l)}  {w}-{len(v_l)-w}  "
      f"sign p {sign_test(w, len(v_l)):.4f}")
print(f"  DEEP bucket,    EWZ 5d < -2%:        SHORT-spread ", end="")
dp = (spy_gate & (r5["EWZ"] < -0.02)).fillna(False)
e_d = declusters(idx[dp.values & val.values], H, idx)
v_d = ret.loc[e_d].values
w = int((v_d > 0).sum())
print(f"{100*v_d.mean():+.3f}%  N={len(v_d)}  {w}-{len(v_d)-w}  "
      f"sign p {sign_test(w, len(v_d)):.4f}")
print(f"\n  SPREAD BETWEEN THE TWO BUCKETS: "
      f"{100*(v_l.mean()-v_d.mean()):+.3f}pp in favour of the SHALLOW cell.")
print("  A trigger whose edge lives entirely in its weakest readings and")
print("  inverts at today's reading is not a trigger, it is a definition.")
