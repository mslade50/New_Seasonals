"""Today's shape from the yield side: the 10y rose 19.4bp over two sessions and closed at a
52-week high on both (5.114 then 5.162). Last night told the one-day jump and its next-session
follow-through. This is the session AFTER the second high: does the move extend or give back?
Cross-check against 03's TLT two-day -1.2% cell (6 of 7 up next session)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = close_panel(["^TNX", "TLT", "IEF", "SPY", "IWM", "^GSPC"])
tnx = px["^TNX"].dropna()
idx = tnx.index
px = px.reindex(idx)
bp1 = tnx.diff() * 100
bp2 = tnx.diff(2) * 100
hi = tnx >= tnx.rolling(252, min_periods=200).max() - 1e-9
two_hi = hi & hi.shift(1).fillna(False).astype(bool)
print("today: bp2", round(bp2.iloc[-1], 1), "two_hi", bool(two_hi.iloc[-1]), "bp1", round(bp1.iloc[-1], 1))


def fwd_bp(h):
    return (tnx.shift(-h) - tnx) * 100


def report(mask, label, gap=5):
    trig = idx[mask.fillna(False).values]
    trig = trig[trig < idx[-1]]
    epi = declusters(trig, gap, idx)
    ctl = local_control(idx, epi, 126)
    print(f"\n##### {label}: raw {len(trig)}, declustered({gap}) {len(epi)}")
    print("   dates:", [str(d.date()) for d in epi])
    for h in (1, 2, 5, 21):
        f = fwd_bp(h)
        v = f.reindex(epi).dropna()
        dn = int((v < 0).sum())
        print(f"   10y h{h}: n {len(v)} mean {v.mean():+.1f}bp median {v.median():+.1f} down {dn}/{len(v)} "
              f"sign_p(down) {sign_test(dn, len(v)):.3f} sign_p(up) {sign_test(len(v) - dn, len(v)):.3f} | local {f.reindex(ctl).mean():+.2f}bp")
    rows = []
    for t in ("TLT", "IEF", "SPY", "IWM"):
        for h in (1, 5):
            f = fwd_ret(px[t], h)
            v = f.reindex(epi).dropna()
            if len(v) == 0:
                continue
            row = summarize(v.values, f"{t} h{h}")
            row["rec"] = f"{int((v > 0).sum())}-{int((v < 0).sum())}"
            k = int((v > 0).sum()) if row["hit"] >= 50 else int((v < 0).sum())
            row["sign_p"] = round(sign_test(k, len(v)), 4)
            row["local"] = 100 * f.reindex(ctl).mean()
            rows.append(row)
    show(rows, label)
    f1 = fwd_bp(1).reindex(epi).dropna()
    for part in era_split(f1.index, f1.values / 100):
        print("   era 10y h1 (x100 = bp):", part["label"], part["n"], round(part.get("mean_pct", np.nan), 2), "hit(up)", round(part.get("hit", np.nan), 1))
    t1 = fwd_ret(px["TLT"], 1).reindex(epi).dropna()
    for part in era_split(t1.index, t1.values):
        print("   era TLT h1:", part["label"], part["n"], round(part.get("mean_pct", np.nan), 2), round(part.get("hit", np.nan), 1))
    print("   cluster 10y h1:", cluster_note(f1.index, f1.values / 100))
    if len(t1):
        print("   cluster TLT h1:", cluster_note(t1.index, t1.values))
    return epi


e1 = report(two_hi & (bp2 >= 15), "10y up 15bp+ over 2 sessions, 52w-high close on both")
e2 = report(two_hi & (bp2 >= 12), "10y up 12bp+ over 2 sessions, 52w-high close on both")
e3 = report(hi & (bp2 >= 15), "10y up 15bp+ over 2 sessions, 52w-high close today (any yesterday)")
e4 = report(two_hi & (bp2 >= 15) & (bp1 > 0) & (bp1.shift(1) >= 10), "10y +10bp or more, then up again, both 52w-high closes (today's order)")
e5 = report((bp2 >= 15), "10y up 15bp+ over 2 sessions, any level")
