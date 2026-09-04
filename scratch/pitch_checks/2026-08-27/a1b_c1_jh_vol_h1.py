"""C1 round 2 -- the ONE-SESSION vehicle form, which is the only horizon the
mechanism in a1 actually claims.

a1 established: spot VIX from the JH-1 close to the JH+0 close is -2.60%,
21 of 26 down, sign p 0.0010, 68% of it intraday (10:00 ET speech). a1 then
tested the vehicle only at h=4 and h=2. This runs h=1.

Lag convention, stated explicitly:
    signal date = JH-2  (the session BEFORE the entry close)
    entry  MOC  = JH-1
    exit   MOC  = JH+0  (the speech session close)
For 2026: JH = 2026-08-28 (Fri). signal 2026-08-26, ENTRY 2026-08-27 (today),
EXIT 2026-08-28.

Plus the translation question: does the FRONT VIX FUTURE (and therefore SVXY)
move with the spot drop on JH+0, or is this the SVXY translation trap?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

JH = load_events(["jackson_hole"])["date"]
JH_FUT = JH[JH.dt.year == 2026]
JH = JH[JH.dt.year <= 2025]

pxd = load_prices(["^VIX", "^VIX3M", "SPY", "SVXY", "UVXY"])
px = pd.DataFrame({t: pxd[t]["Close"] for t in ("SVXY", "UVXY", "SPY")}).dropna()
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)

sig_pos = [p - 2 for d in JH if (p := pos.get(d)) is not None and p - 2 >= 0]
sig = idx[sig_pos]
print(f"SVXY-era JH anchors: N={len(sig)}  {sig.year.min()}..{sig.year.max()}")
print("  live instance: JH =", JH_FUT.iloc[0].date(),
      "-> signal 2026-08-26, ENTRY (MOC) 2026-08-27, EXIT (MOC) 2026-08-28")
print("  historical map signal/entry/exit:")
for d in sig:
    p = pos.get(d)
    print(f"    signal {d.date()}  entry {idx[p+1].date()}  exit {idx[p+2].date()}")

H = 1
ret1 = vehicle_ret(px, [("SVXY", 1.0)], H, 1)
allv = ret1.dropna()
span = (ret1.notna()) & (idx >= sig[0]) & (idx <= sig[-1])
v = ret1.loc[sig].values
w = int((v > 0).sum())

print("\n" + "=" * 78)
print("1. LONG SVXY, entry MOC JH-1 close, exit MOC JH+0 close (h=1)")
print("=" * 78)
rows = [summarize(v, f"COND h=1 (N={len(v)})"),
        summarize(ret1[span].values, "CTRL-a own drift, same span"),
        summarize(allv.values, "CTRL-b all days, full history")]
show(rows)
ctrl_a = ret1[span].mean()
print(f"  excess over CTRL-a  = {100*(v.mean()-ctrl_a):+.4f}pp")
print(f"  excess over CTRL-b  = {100*(v.mean()-allv.mean()):+.4f}pp")
print(f"  record {w}-{len(v)-w}   exact sign_test p = {sign_test(w, len(v)):.4f}")
se = np.sqrt(v.var(ddof=1)/len(v) + ret1[span].var(ddof=1)/span.sum())
print(f"  Welch t vs CTRL-a = {(v.mean()-ctrl_a)/se:+.2f}   "
      f"bootstrap P(mean<=0) = {bootstrap_p_le0(v):.3f}")
print(f"  concentration: {cluster_note(sig, v)}")
print("  per-anchor h=1 SVXY:")
for d, x in zip(sig, v):
    print(f"    {d.year}  entry {idx[pos.get(d)+1].date()}  {100*x:+6.2f}%"
          f"{'   [MIDTERM]' if d.year % 4 == 2 else ''}")

print("\n" + "=" * 78)
print("2. SPY-BETA-HEDGED RESIDUAL at h=1 (trailing-252d OLS beta, lag-1)")
print("=" * 78)
rs = px["SVXY"].pct_change(fill_method=None)
rp = px["SPY"].pct_change(fill_method=None)
beta = (rs.rolling(252).cov(rp) / rp.rolling(252).var()).shift(1)
resid = ret1 - beta * vehicle_ret(px, [("SPY", 1.0)], H, 1)
rc = resid.loc[sig].dropna()
ra = resid.dropna()
wr = int((rc > 0).sum())
print(f"  mean trailing beta at the anchors = {beta.loc[sig].mean():.2f}")
print(f"  residual COND {100*rc.mean():+.4f}% (N={len(rc)}, t={rc.mean()/(rc.std(ddof=1)/np.sqrt(len(rc))):+.2f})"
      f"   ALL {100*ra.mean():+.4f}%   excess {100*(rc.mean()-ra.mean()):+.4f}pp")
print(f"  residual record {wr}-{len(rc)-wr}, sign p = {sign_test(wr, len(rc)):.4f}")
print(f"  SPY's own h=1 return on the same anchors: "
      f"{100*vehicle_ret(px, [('SPY',1.0)],1,1).loc[sig].mean():+.4f}% "
      f"(all days {100*vehicle_ret(px, [('SPY',1.0)],1,1).dropna().mean():+.4f}%)")

print("\n" + "=" * 78)
print("3. OFFSET PLACEBO LADDER at h=1, k = -12..+12 (signal = JH-2+k)")
print("=" * 78)
rows = []
for k in range(-12, 13):
    sp = [p - 2 + k for d in JH
          if (p := pos.get(d)) is not None and 0 <= p - 2 + k < len(idx)]
    vv = ret1.loc[idx[sp]].dropna().values
    if not len(vv):
        continue
    rows.append({"k": k, "n": len(vv), "mean_pct": round(100*vv.mean(), 3),
                 "hit": round(100*(vv > 0).mean(), 1),
                 "excess_pp": round(100*(vv.mean()-allv.mean()), 3)})
d = pd.DataFrame(rows).sort_values("mean_pct", ascending=False).reset_index(drop=True)
rank = int(d.index[d["k"] == 0][0]) + 1
print(f"  TRUE ANCHOR k=0 RANKS {rank} of {len(d)}")
print(d.to_string(index=False))

print("\n" + "=" * 78)
print("4+5. LEVERAGE REGIME (-1x <=2018-02-26 / -0.5x after) and MIDTERM, h=1")
print("=" * 78)
cut = pd.Timestamp("2018-02-27")
rows = []
for lbl, m in (("-1x era (<=2017)", sig < cut), ("-0.5x era (2018+)", sig >= cut),
               ("MIDTERM", (sig.year % 4) == 2), ("non-midterm", (sig.year % 4) != 2)):
    sub = v[m]
    if not len(sub):
        continue
    r = summarize(sub, f"{lbl} N={len(sub)}")
    r["excess_pp"] = round(r["mean_pct"] - 100*ctrl_a, 3)
    r["sign_p"] = round(sign_test(int((sub > 0).sum()), len(sub)), 4)
    rows.append(r)
show(rows)

print("\n" + "=" * 78)
print("6. COST")
print("=" * 78)
edge = 100*100*v.mean()
for lbl, bps in (("MOC-to-MOC auction, optimistic", 5.0),
                 ("realistic SVXY round trip (spread + auction impact)", 15.0)):
    print(f"  {lbl}: {bps} bps -> edge {edge:.1f} bps = {edge/bps:.1f}x (need >=5x)")

print("\n" + "=" * 78)
print("7. THE TRANSLATION QUESTION: does the FRONT VIX FUTURE move with the")
print("   spot drop on JH+0?  SVXY/UVXY track front VIX futures, not spot.")
print("=" * 78)
vix = pxd["^VIX"]["Close"]
v3m = pxd["^VIX3M"]["Close"]
LEV_SVXY = pd.Series(np.where(idx < cut, -1.0, -0.5), index=idx)
LEV_UVXY = pd.Series(np.where(idx < cut, 2.0, 1.5), index=idx)
rows = []
for d in sig:
    p = pos.get(d)
    e, x = idx[p+1], idx[p+2]        # entry close (JH-1), exit close (JH+0)
    if e not in vix.index or x not in vix.index:
        continue
    sp_ = vix.loc[x]/vix.loc[e] - 1.0
    s3 = (v3m.loc[x]/v3m.loc[e] - 1.0) if (e in v3m.index and x in v3m.index) else np.nan
    rsvxy = px["SVXY"].loc[x]/px["SVXY"].loc[e] - 1.0
    ruvxy = px["UVXY"].loc[x]/px["UVXY"].loc[e] - 1.0
    rows.append({"yr": d.year, "spot_VIX_%": 100*sp_, "VIX3M_%": 100*s3,
                 "SVXY_%": 100*rsvxy, "UVXY_%": 100*ruvxy,
                 "impl_fut_from_SVXY_%": 100*rsvxy/LEV_SVXY.loc[x],
                 "impl_fut_from_UVXY_%": 100*ruvxy/LEV_UVXY.loc[x]})
t = pd.DataFrame(rows).round(2)
print(t.to_string(index=False))
print(f"\n  MEANS on the JH+0 session (N={len(t)}):")
print(f"    spot VIX            {t['spot_VIX_%'].mean():+.2f}%   "
      f"down {int((t['spot_VIX_%']<0).sum())}/{len(t)}  "
      f"sign p {sign_test(int((t['spot_VIX_%']<0).sum()), len(t)):.4f}")
print(f"    VIX3M               {t['VIX3M_%'].mean():+.2f}%   "
      f"down {int((t['VIX3M_%']<0).sum())}/{len(t)}")
print(f"    implied front fut   {t['impl_fut_from_SVXY_%'].mean():+.2f}% (from SVXY)   "
      f"{t['impl_fut_from_UVXY_%'].mean():+.2f}% (from UVXY)   "
      f"down {int((t['impl_fut_from_UVXY_%']<0).sum())}/{len(t)}")
print(f"    SVXY                {t['SVXY_%'].mean():+.2f}%   "
      f"up {int((t['SVXY_%']>0).sum())}/{len(t)}")
print(f"\n  TRANSLATION RATIO: implied front future / spot VIX = "
      f"{t['impl_fut_from_UVXY_%'].mean()/t['spot_VIX_%'].mean():.2f}x  "
      f"(1.00 would be full pass-through)")
# same ratio unconditionally, for the reference class
rv = vix.pct_change(fill_method=None).reindex(idx)
ru = px["UVXY"].pct_change(fill_method=None)/LEV_UVXY
m = rv.notna() & ru.notna() & (rv < 0)
print(f"  reference class (ALL down-VIX sessions, N={int(m.sum())}): spot "
      f"{100*rv[m].mean():+.2f}%, implied front fut {100*ru[m].mean():+.2f}% "
      f"-> ratio {ru[m].mean()/rv[m].mean():.2f}x")
