"""Detail for 08's one live cell: the 10y up 15bp+ over two sessions, closing at a 52w high
(20 episodes with a forward week). A week later IEF was 14-6 higher. Check era, concentration,
the horizon profile, and the exact analogue: today is the SECOND trigger day of the cluster."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = close_panel(["^TNX", "TLT", "IEF", "SPY"])
tnx = px["^TNX"].dropna()
idx = tnx.index
px = px.reindex(idx)
bp2 = tnx.diff(2) * 100
hi = tnx >= tnx.rolling(252, min_periods=200).max() - 1e-9
trig_mask = (hi & (bp2 >= 15)).fillna(False)


def fwd_bp(h):
    return (tnx.shift(-h) - tnx) * 100


def profile(epi, label):
    ctl = local_control(idx, epi, 126)
    print(f"\n##### {label}: n {len(epi)}")
    print("   dates:", [str(d.date()) for d in epi])
    for h in (1, 2, 3, 5, 10, 21):
        v = fwd_bp(h).reindex(epi).dropna()
        dn = int((v < 0).sum())
        i = fwd_ret(px["IEF"], h).reindex(epi).dropna()
        print(f"   h{h:2d}: 10y {v.mean():+5.1f}bp down {dn}/{len(v)} (local {fwd_bp(h).reindex(ctl).mean():+.1f}) | "
              f"IEF {100 * i.mean():+.2f}% up {int((i > 0).sum())}/{len(i)} (local {100 * fwd_ret(px['IEF'], h).reindex(ctl).mean():+.2f})")
    i5 = fwd_ret(px["IEF"], 5).reindex(epi).dropna()
    s = summarize(i5.values, "IEF h5")
    print("   IEF h5:", {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()}, "sign_p", round(sign_test(int((i5 > 0).sum()), len(i5)), 4))
    for part in era_split(i5.index, i5.values):
        print("   era IEF h5:", part["label"], part["n"], round(part.get("mean_pct", np.nan), 2), round(part.get("hit", np.nan), 1))
    print("   cluster IEF h5:", cluster_note(i5.index, i5.values))
    b5 = fwd_bp(5).reindex(epi).dropna()
    for part in era_split(b5.index, b5.values / 100):
        print("   era 10y h5 (x100 = bp):", part["label"], part["n"], round(part.get("mean_pct", np.nan), 2), "hit(up)", round(part.get("hit", np.nan), 1))
    print("   per-episode 10y h5 bp:", {str(d.date()): round(x, 1) for d, x in b5.items()})
    x = i5[~((i5.index.year == 2022) | (i5.index.year == 2023))]
    print(f"   ex 2022-23: n {len(x)} IEF h5 {100 * x.mean():+.2f}% up {int((x > 0).sum())}/{len(x)}")


trig = idx[trig_mask.values]
trig = trig[trig < idx[-1]]
epi = declusters(trig, 5, idx)
profile(epi, "first day of each cluster (08's cell)")

# exact analogue: a trigger day whose previous session was also a trigger day
day2 = idx[(trig_mask & trig_mask.shift(1).fillna(False).astype(bool)).values]
day2 = day2[day2 < idx[-1]]
profile(declusters(day2, 5, idx), "second consecutive trigger day (today's position)")

# any trigger day, 5-day decluster, but anchored on the LAST day of each run
runs = []
for d in trig:
    p = idx.get_loc(d)
    if p + 1 < len(idx) and trig_mask.iloc[p + 1]:
        continue
    runs.append(d)
profile(declusters(pd.DatetimeIndex(runs), 5, idx), "last day of each trigger run")
