"""C12 round 2 - the local control (CTRL-c) and the confirming-leg trap.

Registry (2026-08-14): "Long SPY with VIX's LEVEL in its bottom decile while SPY
is within 0.5% of its 52w high ... dies on CTRL-c: the local neighbourhood pays
+0.634%, an edge of +0.003pp", and "Adding confirming legs to a momentum state
does not create a state". C12 has exactly that shape (a momentum state at a high
plus a rates leg that confirms the same regime), so run CTRL-c early.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["SPY", "TLT", "^TNX"]).dropna()
IDX = px.index
off_hi = (px["SPY"] / px["SPY"].rolling(252).max() - 1.0) * 100
off_lo = (px["TLT"] / px["TLT"].rolling(252).min() - 1.0) * 100
r21 = pct_rank(px["^TNX"], 21)
LIVE = (off_hi >= -0.5) & (r21 >= 70) & (off_lo <= 1.0)

for leg in ("SPY", "TLT"):
    for h in (5, 10):
        f = fwd_lag(px[leg], h, 1)
        valid = f.notna()
        trig = IDX[(LIVE & valid).values]
        epi = declusters(trig, 10, IDX)
        loc = local_control(IDX[valid.values], trig, 126)
        rows = [summarize(f.reindex(epi).dropna().values, f"LIVE cell episodes (N={len(epi)})"),
                summarize(f.reindex(trig).dropna().values, f"LIVE cell day-level (N={len(trig)})"),
                summarize(f.reindex(loc).dropna().values, "CTRL-c local +/-126td ex-trigger"),
                summarize(f[valid].values, "CTRL-b all days")]
        show(rows, f"{leg} h={h}")
        c = f.reindex(loc).dropna()
        v = f.reindex(epi).dropna()
        print(f"  edge vs LOCAL neighbourhood: {100*(v.mean()-c.mean()):+.3f}pp")

print("\n" + "=" * 92)
print("confirming-leg trap: are the two 'legs' the same regime?")
print("=" * 92)
print(f"  corr(TNX 21d rank, -TLT %off 52w low) = "
      f"{r21.corr(-off_lo):.3f}")
print(f"  P(TLT within 1% of low | TNX r21>=70) = "
      f"{100*((off_lo<=1.0) & (r21>=70)).sum()/max(1,(r21>=70).sum()):.1f}%   "
      f"P(TLT within 1% of low | all days) = {100*(off_lo<=1.0).mean():.1f}%")
