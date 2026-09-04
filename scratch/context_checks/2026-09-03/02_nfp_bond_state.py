"""Payrolls with the long end already pinned at a 52-week low.

Tonight IEF sits 0.47% off its 52-week low, LQD 0.27%, TLT 1.27%, and the 10y
yield printed a 52-week high on 09-02 before giving back 3bp today. So the
question is not "what does payrolls do" but "what does payrolls do when the
bond market has already sold off into it".
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

SUBJ = ["IEF", "TLT", "^TNX", "^GSPC", "SPY", "LQD", "HYG", "GC=F"]
px = close_panel(SUBJ)
ref = px["^GSPC"].dropna().index
nfp = pd.DatetimeIndex(sorted(set(load_events(["nfp"])["date"]) & set(ref)))
pos = {d: i for i, d in enumerate(ref)}
anchors_all = pd.DatetimeIndex([ref[pos[d] - 1] for d in nfp if pos.get(d, 0) > 0])

ief = px["IEF"].dropna()
dist_low = 100 * (ief / ief.rolling(252).min() - 1.0)
print(f"IEF distance above its 252d low, today: {dist_low.iloc[-1]:.2f}%")
print(f"that percentile in history:            "
      f"{100*(dist_low.dropna() <= dist_low.iloc[-1]).mean():.1f}%")
print()

tnx = px["^TNX"].dropna()
tnx_pct = 100 * tnx.rolling(252).apply(lambda w: (w <= w[-1]).mean(), raw=True)
print(f"^TNX trailing-252 percentile today: {tnx_pct.iloc[-1]:.1f}")
print()

def row(label, a, tick, h=1, note=False):
    a = pd.DatetimeIndex([d for d in a if d in dist_low.index])
    f = fwd_ret(px[tick].dropna(), h).reindex(a).dropna()
    if len(f) < 4:
        print(f"    {label:34s} {tick:6s} n={len(f)} (too few)")
        return None
    r = summarize(f.to_numpy())
    up = int((f > 0).sum())
    p = sign_test(max(up, len(f) - up), len(f))
    print(f"    {label:34s} {tick:6s} n={r['n']:3d} mean={r['mean_pct']:+7.3f}% "
          f"med={r['median_pct']:+7.3f}% hit={r['hit']:5.1f}% t={r['t']:+6.2f} "
          f"{up}-{len(f)-up} up p={p:.4f}")
    if note:
        print("      ", cluster_note(f.index, f.to_numpy()))
        for e in era_split(f.index, f.to_numpy()):
            print(f"       {e['label']:9s} n={e['n']:3d} mean={e['mean_pct']:+6.3f}% "
                  f"hit={e['hit']:5.1f}%")
    return f

for thr in (1.0, 2.0):
    tight = pd.DatetimeIndex([d for d in anchors_all
                              if d in dist_low.index
                              and np.isfinite(dist_low[d]) and dist_low[d] <= thr])
    loose = pd.DatetimeIndex([d for d in anchors_all
                              if d in dist_low.index
                              and np.isfinite(dist_low[d]) and dist_low[d] > thr])
    print(f"=== payroll eves with IEF within {thr:.0f}% of its 52w low: "
          f"{len(tight)} of {len(tight)+len(loose)} ===")
    for tick in SUBJ:
        row(f"IEF <= {thr:.0f}% off low", tight, tick,
            note=(tick in ("IEF", "^TNX", "^GSPC")))
        row(f"IEF >  {thr:.0f}% off low", loose, tick)
    print()

print("=== the pinned-bond payroll episodes (IEF within 1% of its 52w low) ===")
tight = pd.DatetimeIndex([d for d in anchors_all
                          if d in dist_low.index
                          and np.isfinite(dist_low[d]) and dist_low[d] <= 1.0])
f_ief = fwd_ret(px["IEF"].dropna(), 1)
f_spx = fwd_ret(px["^GSPC"].dropna(), 1)
for d in tight:
    print(f"  eve {d.date()} -> payrolls {ref[pos[d]+1].date()}  "
          f"IEF {100*f_ief.get(d, float('nan')):+6.2f}%  "
          f"^GSPC {100*f_spx.get(d, float('nan')):+6.2f}%")

print()
print("=== h5 (through the following week) ===")
for tick in ("IEF", "TLT", "^TNX", "^GSPC"):
    row("IEF <= 1% off low", tight, tick, h=5)
