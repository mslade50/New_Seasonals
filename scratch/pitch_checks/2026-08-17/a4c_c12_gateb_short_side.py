"""C12 round 2b - the only piece of C12 with a pulse, examined on the short side.

Round 1 settled the pitched conditioner: TNX 21d rank >= 70 moves the near-52w-
high parent by -0.059pp (h=5) / -0.082pp (h=10). It does not filter.

What is left is gate B - SPY within 0.5% of a 52w high WHILE TLT is pinned
within 1% of its 52w low - where forward SPY was -0.775% at h=5 over 10
episodes, i.e. a short. Adjudicate it properly before letting it near a
watchlist line: concentration, declustering sensitivity, era, threshold
neighbours, and whether the SPY-high leg is doing any work at all.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["SPY", "TLT"]).dropna()
IDX = px.index
off_hi = (px["SPY"] / px["SPY"].rolling(252).max() - 1.0) * 100
off_lo = (px["TLT"] / px["TLT"].rolling(252).min() - 1.0) * 100
CELL = (off_hi >= -0.5) & (off_lo <= 1.0)

for h in (3, 5, 10):
    f = -fwd_lag(px["SPY"], h, 1)          # SHORT basis
    valid = f.notna()
    trig = IDX[(CELL & valid).values]
    for gap in (5, 10, 21, 42):
        epi = declusters(trig, gap, IDX)
        v = f.reindex(epi).dropna()
        w = int((v > 0).sum())
        loc = local_control(IDX[valid.values], trig, 126)
        print(f"h={h} min_gap={gap:>2}: N={len(v):>2} mean {100*v.mean():+.3f}% "
              f"med {100*np.median(v.values):+.3f}% {w}-{len(v)-w} "
              f"signp {sign_test(w, len(v)):.4f} | vs local "
              f"{100*(v.mean()-f.reindex(loc).dropna().mean()):+.3f}pp | "
              f"boot P(<=0) {bootstrap_p_le0(v.values):.3f}")
    epi = declusters(trig, 10, IDX)
    v = f.reindex(epi).dropna()
    print(f"   concentration: {cluster_note(v.index, v.values)}")
    print(f"   years: {sorted(set(v.index.year))}")
    show(era_split(v.index, v.values), f"   h={h} era split")

print("\n" + "=" * 92)
print("does the SPY-52w-high leg do any work? (gate attribution, short SPY)")
print("=" * 92)
for h in (5, 10):
    f = -fwd_lag(px["SPY"], h, 1)
    valid = f.notna()
    rows = []
    for lbl, m in (("TLT<=1% of low ALONE", off_lo <= 1.0),
                   ("TLT<=1% x SPY near high", CELL),
                   ("TLT<=1% x SPY NOT near high", (off_lo <= 1.0) & (off_hi < -0.5)),
                   ("SPY near high ALONE", off_hi >= -0.5)):
        e = declusters(IDX[(m & valid).values], 10, IDX)
        r = summarize(f.reindex(e).dropna().values, lbl)
        r["n_days"] = int((m & valid).sum())
        rows.append(r)
    show(rows, f"short SPY h={h}")

print("\n" + "=" * 92)
print("threshold neighbours (short SPY h=5)")
print("=" * 92)
f = -fwd_lag(px["SPY"], 5, 1)
valid = f.notna()
rows = []
for hi_thr in (-0.25, -0.5, -1.0, -2.0):
    for lo_thr in (0.5, 1.0, 2.0, 3.0):
        m = (off_hi >= hi_thr) & (off_lo <= lo_thr)
        e = declusters(IDX[(m & valid).values], 10, IDX)
        v = f.reindex(e).dropna()
        rows.append({"spy_hi": hi_thr, "tlt_lo": lo_thr, "n_epi": len(v),
                     "mean_pct": round(100 * v.mean(), 3) if len(v) else np.nan,
                     "hit": round(100 * (v > 0).mean(), 1) if len(v) else np.nan})
show(rows, "grid of neighbours")
