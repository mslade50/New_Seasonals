"""C6 round 2: is the midterm drag real, and does any horizon survive it?

Round 1: dip+skew all years N=41 +1.214% (excess +1.020, welch +2.14, boot
0.002, 27-14). Live cell dip+skew+MIDTERM N=8 -0.473% at a 37.5% hit. The
midterm drag reproduces on three independent parents (skew alone -0.618pp
welch -1.78; dip alone -0.290pp; all days -0.210pp welch -1.39), so it is a
conditioner rather than a slice.

Two honest counter-attacks before finalising a kill:
  (a) is the midterm drag just 2002 and 2022 bear tape?
  (b) does a SHORTER horizon survive inside midterms?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change  # noqa

import warnings
warnings.filterwarnings("ignore")

ASOF = pd.Timestamp("2026-08-20")
px = close_panel(["SPY", "^SKEW"]).loc[:ASOF].dropna()
idx = px.index
sk5 = pct_rank(px["^SKEW"], 5)
spy5r = _valid_pct_change(px["SPY"], 5)
spy_dd = px["SPY"] / rolling_on_valid(px["SPY"], lambda x: x.rolling(252).max()) - 1.0
mid = pd.Series([d.year % 4 == 2 for d in idx], index=idx)
skew = (sk5 >= 95).fillna(False)
dip = (spy5r <= -0.01).fillna(False)
below = (spy_dd < -0.01).fillna(False)

print("=" * 100)
print("C6b-1  the 8 midterm dip+skew episodes, one by one (h=5)")
print("=" * 100)
leg = fwd_lag(px["SPY"], 5, 1)
e = declusters(idx[((dip & skew & mid) & leg.notna()).values], 5, idx)
for d in e:
    print(f"   {d.date()}   SPY5d {100*spy5r.loc[d]:+6.2f}%  dd52wh {100*spy_dd.loc[d]:+6.2f}%  "
          f"skew rk5 {sk5.loc[d]:5.1f}  ->  fwd h=5 {100*leg.loc[d]:+6.2f}%")
v = leg.loc[e].values
print(f"   mean {100*v.mean():+.3f}%  hit {100*(v>0).mean():.1f}%  "
      f"sign p {sign_test(int((v>0).sum()), len(v)):.4f}")

print("\n" + "=" * 100)
print("C6b-2  midterm drag by year -- is it 2002/2022 bear tape only?")
print("=" * 100)
for lbl, m in (("all days", pd.Series(True, index=idx)), ("dip alone", dip), ("skew alone", skew)):
    print(f"  --- {lbl}")
    for y in sorted({d.year for d in idx if d.year % 4 == 2}):
        ee = declusters(idx[(m & (pd.Series([d.year == y for d in idx], index=idx))
                             & leg.notna()).values], 5, idx)
        if len(ee) < 3:
            continue
        vv = leg.loc[ee].values
        print(f"       {y}: N={len(ee):<4} {100*vv.mean():+7.3f}%  hit {100*(vv>0).mean():5.1f}%")

print("\n" + "=" * 100)
print("C6b-3  does a SHORTER horizon survive inside midterms?")
print("=" * 100)
for form, m in (("dip+skew", dip & skew), ("skew + >1% below 52wh (the watchlist leg)", skew & below)):
    print(f"  --- {form}")
    for h in (1, 2, 3, 5, 10):
        lg = fwd_lag(px["SPY"], h, 1)
        b = lg.dropna()
        for tag, mm in (("MIDTERM", m & mid), ("non-mid", m & ~mid)):
            ee = declusters(idx[(mm & lg.notna()).values], max(h, 5), idx)
            if len(ee) < 3:
                print(f"      h={h:<3} {tag:<8} N={len(ee)}")
                continue
            vv = lg.loc[ee].values
            print(f"      h={h:<3} {tag:<8} N={len(ee):<4} {100*vv.mean():+7.3f}%  "
                  f"excess {100*vv.mean()-100*b.mean():+7.3f}%  hit {100*(vv>0).mean():5.1f}%  "
                  f"signp {sign_test(int((vv>0).sum()), len(vv)):.4f}")

print("\n" + "=" * 100)
print("C6b-4  what would turn it on?  the non-midterm cell, for the watchlist")
print("=" * 100)
for h in (3, 5):
    lg = fwd_lag(px["SPY"], h, 1)
    b = lg.dropna()
    ee = declusters(idx[((dip & skew & ~mid) & lg.notna()).values], 5, idx)
    vv = lg.loc[ee].values
    print(f"  h={h} dip+skew NON-midterm: N={len(ee)}  {100*vv.mean():+.3f}%  "
          f"excess {100*vv.mean()-100*b.mean():+.3f}%  hit {100*(vv>0).mean():.1f}%  "
          f"signp {sign_test(int((vv>0).sum()), len(vv)):.4f}  boot {bootstrap_p_le0(vv):.3f}")
    print(f"     {cluster_note(ee, vv)}")
