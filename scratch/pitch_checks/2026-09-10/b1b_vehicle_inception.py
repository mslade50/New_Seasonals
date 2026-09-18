"""B1 round 2 -- is the 6-0 UUP record an artefact of UUP's 2007 inception?

The trigger fires 31 times on the SPY calendar. UUP was not alive for 18 of
them (all 2006). This script asks whether the two vehicles are substitutable
where both exist, and what the evidence UUP cannot see actually says.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 210)

TICKS = ["SPY", "UUP", "DX-Y.NYB", "^TNX"]
raw = load_prices(TICKS)
cal = raw["SPY"].index
px = pd.DataFrame({t: raw[t]["Close"].reindex(cal) for t in TICKS})
dx, tnx, uup = px["DX-Y.NYB"], px["^TNX"], px["UUP"]


def at_high(s, n):
    hi = rolling_on_valid(s, lambda x: x.rolling(n).max())
    return (s >= hi - 1e-9) & s.notna() & hi.notna()


TRIG = ((pct_rank(dx, 63) <= 20) & at_high(tnx, 252)).fillna(False)
H = 5
r_uup = vehicle_ret(px, [("UUP", 1.0)], H, 1)
r_dx = vehicle_ret(px, [("DX-Y.NYB", 1.0)], H, 1)

print("=" * 78)
print("A. ARE THE TWO VEHICLES SUBSTITUTABLE? (overlap 2007-03-01 onward)")
print("=" * 78)
both = px[["UUP", "DX-Y.NYB"]].dropna()
du = both["UUP"].pct_change()
dd = both["DX-Y.NYB"].pct_change()
ok = du.notna() & dd.notna()
print(f"  daily-return corr (UUP vs DX-Y.NYB, {int(ok.sum())} sessions) = "
      f"{du[ok].corr(dd[ok]):.4f}")
print(f"  beta of UUP on DX = {np.polyfit(dd[ok], du[ok], 1)[0]:.4f}")
print(f"  ann. tracking drift (UUP - DX) = "
       f"{252*100*(du[ok].mean()-dd[ok].mean()):+.2f}%/yr (expense + T-bill carry)")

ov = px.index[TRIG.values & r_uup.notna().values & r_dx.notna().values]
print(f"\n  state days where BOTH vehicles are measurable: N={len(ov)}")
show([summarize(r_uup.loc[ov].values, "UUP, overlap days"),
      summarize(r_dx.loc[ov].values, "DX,  overlap days")],
     "same days, both vehicles")
print(f"  per-day |UUP - DX| mean = {100*np.abs(r_uup.loc[ov]-r_dx.loc[ov]).mean():.3f}%"
      f"  -> the vehicles say the SAME thing where both exist")

print("\n" + "=" * 78)
print("B. THE EVIDENCE UUP CANNOT SEE (pre-2007-03-01 state days)")
print("=" * 78)
pre = px.index[TRIG.values & r_dx.notna().values & (px.index < pd.Timestamp("2007-03-01"))]
post = px.index[TRIG.values & r_dx.notna().values & (px.index >= pd.Timestamp("2007-03-01"))]
for lbl, d in (("PRE-UUP (2006)", pre), ("UUP-ERA (2007+)", post)):
    e = declusters(d, H, px.index)
    v = r_dx.loc[e].values
    w = int((v > 0).sum())
    print(f"\n  {lbl}: day-level N={len(d)}  episodes N={len(e)}")
    show([summarize(r_dx.loc[d].values, f"{lbl} day-level [DX]"),
          summarize(v, f"{lbl} episodes [DX]")], "")
    print(f"    record {w}-{len(v)-w}   sign p(>= wins) = {sign_test(w, len(v)):.4f}")
    print(f"    dates: {', '.join(str(x.date()) for x in e)}")

print("\n  >>> ALL AVAILABLE EVIDENCE on the underlying, one cell:")
alld = px.index[TRIG.values & r_dx.notna().values]
alle = declusters(alld, H, px.index)
av = r_dx.loc[alle].values
w = int((av > 0).sum())
show([summarize(r_dx.loc[alld].values, f"DX day-level, ALL history (N={len(alld)})"),
      summarize(av, f"DX episodes, ALL history (N={len(alle)})")], "")
print(f"    record {w}-{len(av)-w}  sign p = {sign_test(w, len(av)):.4f}  "
      f"bootstrap P(mean<=0) = {bootstrap_p_le0(av):.3f}")

print("\n" + "=" * 78)
print("C. HOW MANY INDEPENDENT EPISODES IS THE UUP CELL REALLY?")
print("=" * 78)
uupd = px.index[TRIG.values & r_uup.notna().values]
for gap in (5, 10, 21, 42, 63, 126):
    e = declusters(uupd, gap, px.index)
    v = r_uup.loc[e].values
    w = int((v > 0).sum())
    print(f"  min_gap={gap:4d} td -> N={len(e)}  mean {100*v.mean():+.3f}%  "
          f"record {w}-{len(v)-w}  sign p {sign_test(w, len(v)):.4f}  "
          f"dates {[str(x.date()) for x in e]}")
print("\n  calendar clusters present:", sorted({f"{d.year}-{d.month:02d}" for d in uupd}))

print("\n" + "=" * 78)
print("D. SIGN INSTABILITY: same state, same direction, by cluster (DX vehicle)")
print("=" * 78)
alle_l = list(alle)
grp = {}
for d, v in zip(alle_l, av):
    grp.setdefault(f"{d.year}", []).append(v)
for k in sorted(grp):
    v = np.array(grp[k])
    print(f"  {k}: N={len(v):2d}  mean {100*v.mean():+7.3f}%  "
          f"record {(v>0).sum()}-{(v<=0).sum()}")

print("\n" + "=" * 78)
print("E. HORIZON ROBUSTNESS of the sign flip (DX, all history vs UUP era)")
print("=" * 78)
rows = []
for h in (1, 2, 3, 5, 7, 10, 21):
    rr = vehicle_ret(px, [("DX-Y.NYB", 1.0)], h, 1)
    d = px.index[TRIG.values & rr.notna().values]
    e = declusters(d, h, px.index)
    s = summarize(rr.loc[e].values, f"DX all-hist h={h}")
    ru = vehicle_ret(px, [("UUP", 1.0)], h, 1)
    du_ = px.index[TRIG.values & ru.notna().values]
    eu = declusters(du_, h, px.index)
    su = summarize(ru.loc[eu].values, f"UUP 2007+ h={h}")
    rows += [s, su]
show(rows, "horizon x vehicle")
