import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from sb5_common import PX, IDX, H, anchors, excess, near_high_flags

# XLF 2010+ residual: is the +0.9% excess cell-specific or the post-2010 XLF-SPY drift?
ex_all = (fwd_lag(PX["XLF"], H, 1) - fwd_lag(PX["SPY"], H, 1)).dropna()
e10 = ex_all[ex_all.index.year >= 2010]
print(f"XLF-SPY 21d all-day excess 2010+: {100*e10.mean():+.2f}% hit {100*(e10>0).mean():.0f}%")
oq = e10[(e10.index.month >= 9) & (e10.index.month <= 11)]
print(f"  Sep-Nov anchors 2010+: {100*oq.mean():+.2f}% hit {100*(oq>0).mean():.0f}%")
for sh in (-3, 0, 3):
    for lag in (1, 2):
        ex = excess("XLF", anchors(IDX, 9, 30, shift=sh), lag, H)
        a = ex[ex.index >= 2010]
        w = int((a > 0).sum())
        print(f"  sh{sh:+d} lag{lag} 2010+ excess {100*a.mean():+.2f}% {w}/{len(a)} p={sign_test(w, len(a)):.3f} "
              f"drop2 {100*a.sort_values().iloc[:-2].mean():+.2f}%")
nh, off = near_high_flags(anchors(IDX, 9, 30))
ex = excess("XLF", anchors(IDX, 9, 30), 1, H)
a = ex[(ex.index >= 2010) & nh.reindex(ex.index).values]
print(f"  2010+ & near-high excess {100*a.mean():+.2f}% {int((a>0).sum())}/{len(a)}  yrs {[(y, round(100*v, 2)) for y, v in a.items()]}")
