"""C11 round-2c: the intersection that is actually live TODAY.

c11b showed the SKEW>=95 cell dies separately on TWO of today's conditions
(midterm excess -0.134% at a 50.0% hit; SPY within 1% of its 52w high excess
-0.029%). This measures the INTERSECTION rather than the marginals, for both
the joint C11 trigger and the SKEW-alone reframing, so the kill quotes the
state we would actually be entering into rather than two separate slices.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["SPY", "^SKEW", "^VIX"]).dropna(subset=["SPY"])
idx = px.index
sk5, vx5 = pct_rank(px["^SKEW"], 5), pct_rank(px["^VIX"], 5)
dh = px["SPY"] / px["SPY"].rolling(252).max() - 1.0

H = 5
ret = vehicle_ret(px, [("SPY", 1.0)], H, 1)
valid = ret.notna()
base = ret[valid].dropna()
bh = float((base > 0).mean())
print(f"all-days control h={H}: mean {100*base.mean():+.3f}%, hit {100*bh:.1f}%\n")


def row(lbl, mask):
    s = idx[mask.reindex(idx, fill_value=False).fillna(False).values & valid.values]
    e = declusters(s, 5, idx)
    if len(e) == 0:
        print(f"  {lbl:52s} n=0  DEAD ON COUNT")
        return
    v = ret.loc[e].values
    w = int((v > 0).sum())
    print(f"  {lbl:52s} n={len(v):4d}  mean {100*v.mean():+6.3f}%  "
          f"excess {100*(v.mean()-base.mean()):+6.3f}%  hit {100*w/len(v):5.1f}%  "
          f"signp {sign_test(w, len(v), bh):.4f}")


JOINT = (sk5 >= 95) & (vx5 <= 35)
SKONLY = sk5 >= 95
mid = pd.Series(idx.year % 4 == 2, index=idx)
nh = dh >= -0.01
flat5 = px["SPY"].pct_change(5).abs() <= 0.02   # today's SPY 5d is -0.10%

print("=== C11 AS FRAMED (joint skew>=95 & vix<=35) ===")
row("joint, all", JOINT)
row("joint & midterm", JOINT & mid)
row("joint & SPY within 1% of 52w high", JOINT & nh)
row("joint & midterm & near-high = TODAY", JOINT & mid & nh)
row("joint & midterm & near-high & |SPY 5d|<=2% = TODAY", JOINT & mid & nh & flat5)

print("\n=== THE REFRAMING (skew>=95 alone, no vol condition) ===")
row("skew-alone, all", SKONLY)
row("skew-alone & midterm", SKONLY & mid)
row("skew-alone & near-high", SKONLY & nh)
row("skew-alone & midterm & near-high = TODAY", SKONLY & mid & nh)
row("skew-alone & midterm & near-high & |SPY 5d|<=2% = TODAY", SKONLY & mid & nh & flat5)

print("\n=== the complement, for contrast (what carries the pooled number) ===")
row("skew-alone & NOT midterm & NOT near-high", SKONLY & ~mid & ~nh)
