"""C1 round 2b: BOOK OVERLAP, measured not reasoned.

The systematic book is staged SHORT four gold-complex names for 2026-08-21
(NEM 690 liquid, AGI 2284 / AU 666 / CGAU 3537 overflow), all Overbot Vol
Spike, 2-day hold, time exit 2026-08-25. A 5-day C1 long-GLD hold straddles
that window almost exactly.

Question: on the cell's own historical trigger episodes, does long GLD pay
when the miner short pays, or does it lose at the same time? If GLD falls
whenever GDX mean-reverts, the pitch is a partial HEDGE of the book's winning
position rather than a new bet.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import warnings
warnings.filterwarnings("ignore")

ASOF = pd.Timestamp("2026-08-20")
px = close_panel(["GLD", "GDX", "NEM", "AU", "AGI"]).loc[:ASOF]
idx = px.index
gdx5 = pct_rank(px["GDX"], 5)
gld5 = pct_rank(px["GLD"], 5)
mask = ((gdx5 >= 95) & (gld5 < 95)).fillna(False)

for H, tag in ((2, "the book's OVS hold"), (5, "the C1 hold")):
    gld = fwd_lag(px["GLD"], H, 1)
    gdx = fwd_lag(px["GDX"], H, 1)
    nem = fwd_lag(px["NEM"], H, 1)
    e = declusters(idx[(mask & gld.notna() & gdx.notna()).values], 5, idx)
    a, b, c = gld.loc[e].values, gdx.loc[e].values, nem.loc[e].values
    print("=" * 100)
    print(f"C1c  h={H} ({tag}) -- {len(e)} trigger episodes")
    print("=" * 100)
    print(f"  corr(GLD fwd, GDX fwd) over the hold = {np.corrcoef(a, b)[0,1]:+.3f}   "
          f"corr(GLD, NEM) = {np.corrcoef(a[~np.isnan(c)], c[~np.isnan(c)])[0,1]:+.3f}")
    rev = b < 0
    print(f"  GDX MEAN-REVERTS (fwd<0, the book's short wins): N={int(rev.sum())} of {len(e)} "
          f"({100*rev.mean():.0f}%)")
    print(f"     -> GLD over the same window: mean {100*a[rev].mean():+.3f}%  "
          f"hit {100*(a[rev]>0).mean():.1f}%  worst {100*a[rev].min():+.2f}%")
    print(f"  GDX keeps running (fwd>=0, the book's short loses): N={int((~rev).sum())}")
    print(f"     -> GLD: mean {100*a[~rev].mean():+.3f}%  hit {100*(a[~rev]>0).mean():.1f}%")
    joint = (a > 0) & rev
    print(f"  BOTH win (GLD up AND GDX down): {int(joint.sum())} of {len(e)} = {100*joint.mean():.0f}%")
    print(f"  BOTH lose (GLD down AND GDX up): {int(((a<=0)&(~rev)).sum())} of {len(e)} "
          f"= {100*((a<=0)&(~rev)).mean():.0f}%")
    # dollar-neutral read: the pitch's marginal contribution to a short-miner book
    for w in (0.0, 0.5, 1.0):
        comb = -1.0 * c + w * a   # 1 unit short NEM (the liquid leg) + w long GLD
        v = comb[~np.isnan(comb)]
        print(f"  book = 1x SHORT NEM + {w:.1f}x long GLD: mean {100*v.mean():+7.3f}%  "
              f"sd {100*v.std(ddof=1):6.3f}%  mean/sd {v.mean()/v.std(ddof=1):+.4f}  "
              f"hit {100*(v>0).mean():5.1f}%")
    print()

print("=" * 100)
print("C1c-2  unconditional daily corr GLD vs GDX (context for the numbers above)")
print("=" * 100)
r = px[["GLD", "GDX", "NEM"]].pct_change().dropna()
print(f"  full sample daily corr GLD/GDX = {r['GLD'].corr(r['GDX']):.3f}   "
      f"GLD/NEM = {r['GLD'].corr(r['NEM']):.3f}")
print(f"  trailing 252d           GLD/GDX = {r['GLD'].iloc[-252:].corr(r['GDX'].iloc[-252:]):.3f}   "
      f"GLD/NEM = {r['GLD'].iloc[-252:].corr(r['NEM'].iloc[-252:]):.3f}")
