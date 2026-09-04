"""Tonight's exact vol shape: the tail bid thrusts while at-the-money vol falls.

^SKEW 5d +7.26% (98.4th pctile of its year) while ^VIX 5d -7.39%. Drill 05
showed the SKEW thrust alone has no S&P forward content once the local
neighbourhood is controlled for. This isolates the DIVERGENCE version and
prices it against the same control, so the item can say what the state is
worth rather than how unusual it looks.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = load_prices(["^SKEW", "SPY", "^VIX"])
skew = px["^SKEW"]["Close"].dropna()
spy = px["SPY"]["Close"].dropna()
vix = px["^VIX"]["Close"].dropna()
common = skew.index.intersection(spy.index).intersection(vix.index)
skew, spy, vix = skew.reindex(common), spy.reindex(common), vix.reindex(common)

sk5 = skew.pct_change(5)
rank5 = sk5.rolling(252, min_periods=252).rank(pct=True) * 100
vx5 = vix.pct_change(5)

print(f"today: SKEW 5d {sk5.iloc[-1]*100:+.2f}% (pctile {rank5.iloc[-1]:.1f})  "
      f"VIX 5d {vx5.iloc[-1]*100:+.2f}%  SKEW level {skew.iloc[-1]:.2f}  VIX {vix.iloc[-1]:.2f}")

trig = pd.DatetimeIndex([d for d in common
                         if pd.notna(rank5.get(d, np.nan)) and rank5[d] >= 95
                         and pd.notna(vx5.get(d, np.nan)) and vx5[d] < 0])
dec = declusters(trig, 5, common)
ctrl = local_control(common, dec, 126)
print(f"\nSKEW 5d in the top 5% of its year WHILE VIX fell over the same week")
print(f"  raw {len(trig)} days -> {len(dec)} episodes, {dec[0].date()} .. {dec[-1].date()}")
print(f"  years: {sorted(set(dec.year))}")

for h in (1, 2, 3, 5, 10, 21):
    fw = (spy.shift(-h) / spy - 1.0)
    v = fw.reindex(dec).dropna()
    vc = fw.reindex(ctrl).dropna()
    s = summarize(v.values, f"h{h}")
    sc = summarize(vc.values, "ctrl")
    print(f"  SPY h{h:<2} n={s['n']:<4} mean {s['mean_pct']:+7.3f}%  hit {s['hit']:5.1f}%  t={s['t']:+5.2f}  "
          f"{int((v>0).sum())}-{int((v<0).sum())}  sign p {sign_test(int((v>0).sum()), int(len(v))):.4f}  "
          f"| local ctrl n={sc['n']} {sc['mean_pct']:+.3f}% hit {sc['hit']:.1f}%  "
          f"EDGE {s['mean_pct']-sc['mean_pct']:+.3f}%")

# is the S&P's own realized vol different afterwards? that is what a tail bid claims
fwvol = spy.pct_change().rolling(10).std().shift(-10) * np.sqrt(252) * 100
v = fwvol.reindex(dec).dropna()
vc = fwvol.reindex(ctrl).dropna()
print(f"\n  SPY realized vol over the NEXT 10 sessions: {v.mean():.2f}% annualized "
      f"vs local control {vc.mean():.2f}%  (n={len(v)} vs {len(vc)})")
print(f"  median {v.median():.2f}% vs {vc.median():.2f}%")
hi = (v > vc.median()).mean() * 100
print(f"  share of episodes above the control median: {hi:.1f}%")

# and the same for a SKEW thrust WITHOUT the VIX divergence, as the contrast
trig2 = pd.DatetimeIndex([d for d in common
                          if pd.notna(rank5.get(d, np.nan)) and rank5[d] >= 95
                          and pd.notna(vx5.get(d, np.nan)) and vx5[d] >= 0])
dec2 = declusters(trig2, 5, common)
v2 = fwvol.reindex(dec2).dropna()
print(f"  contrast, SKEW thrust with VIX also UP: n={len(v2)} next-10d vol {v2.mean():.2f}%")

# how often has this state landed on a CPI eve before?
ev = load_events(["cpi"])
cpi = pd.DatetimeIndex(sorted(pd.to_datetime(ev["date"]).unique()))
anc = []
for x in cpi:
    pos = common.searchsorted(x)
    if pos <= 0 or pos >= len(common) or common[pos] != x:
        continue
    anc.append(common[pos - 1])
anc = pd.DatetimeIndex(anc)
overlap = dec.intersection(anc)
print(f"\n  episodes that were ALSO a CPI eve: {len(overlap)} -> {[str(x.date()) for x in overlap]}")
if len(overlap) >= 3:
    fw1 = (spy.shift(-1) / spy - 1.0)
    v = fw1.reindex(overlap).dropna()
    print(f"    those print sessions: {[(str(d.date()), round(x*100,2)) for d, x in v.items()]}")
