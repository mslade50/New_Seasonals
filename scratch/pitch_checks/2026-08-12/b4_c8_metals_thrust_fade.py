"""C8 round 1+2: FADE the metals joint-thrust state (SHORT GDX).

State: GDX 5d rank >= 99 AND GLD 5d rank >= 95. Recon: forward excess
-0.452 / -0.600 / -0.479 at h=1/3/5 over 29-30 declustered episodes, hit
40.0 / 41.4 / 51.7%.

THE FIRST QUESTION IS NOT THE NUMBER. A live pitch leg is LONG GDX, entered
MOC 2026-08-10, exiting MOC 2026-08-17 -- three sessions remain after tonight
(08-13, 08-14, 08-17). A short GDX entered tonight at h=3 exits on exactly the
same close. So this script prices the NET book before it prices the cell.

Kill angles:
  A. NET EXPOSURE against the live leg. What does the combined position
     actually own, and what does it cost to own it?
  B. parent cells: GDX r5>=99 alone, GLD r5>=95 alone. Does the joint state
     add anything, or is it a nested subset that reverses a parent's sign?
  C. concentration: are the negatives a couple of episodes?
  D. definition neighbours on both thresholds.
  E. era / midterm / year histogram.
  F. cost + the tail of a naked short on a miner thrust (GDX 1d upside).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["GDX", "GLD", "SLV", "NEM", "SPY"])
idx = px.index

g5 = pct_rank(px["GDX"], 5)
d5 = pct_rank(px["GLD"], 5)
print(f"today's bar 2026-08-11: GDX rank5={g5.iloc[-1]:.1f}  "
      f"GLD rank5={d5.iloc[-1]:.1f}  "
      f"GDX 5d={100*(px['GDX'].iloc[-1]/px['GDX'].iloc[-6]-1):+.2f}%  "
      f"GLD 5d={100*(px['GLD'].iloc[-1]/px['GLD'].iloc[-6]-1):+.2f}%")

M = ((g5 >= 99) & (d5 >= 95)).fillna(False)
print(f"trigger days: {int(M.sum())}  today fires: {bool(M.iloc[-1])}")

# =========================================================== A. NET EXPOSURE
print("\n" + "="*72)
print("A. NET EXPOSURE vs the live long-GDX pitch leg (2026-08-10 -> 08-17)")
print("="*72)
print("  live leg : LONG GDX, MOC 2026-08-10 entry, MOC 2026-08-17 exit")
print("  candidate: SHORT GDX, MOC 2026-08-12 entry, h=3 -> MOC 2026-08-17 exit")
print("  -> the two legs share the SAME instrument and the SAME exit close.")
print("     Sessions 08-13, 08-14, 08-17 are the entire remaining life of the")
print("     live leg, and the short covers all three of them.")
print("  net GDX exposure over those 3 sessions = LONG 1.0 + SHORT 1.0 = 0.00")
print("  what the pair actually is: an EARLY EXIT of the live leg, executed")
print("  with two extra round trips instead of one cancel.")
# price the two ways of expressing the same view
ret3 = vehicle_ret(px, [("GDX", 1.0)], 3, 1)
gdx_sd = px["GDX"].pct_change().std()
print(f"\n  GDX daily sd = {100*gdx_sd:.2f}%; a flat book earns 0.00% +/- 0.00%")
print(f"  cost of expressing it as a short instead of a close-out:")
print(f"    close the live leg   : 1 round trip  ~= 6 bps")
print(f"    open an offsetting   : 2 round trips ~= 12 bps + GDX borrow")
print(f"    -> strictly dominated by {12+2-6:.0f}+ bps, with identical exposure")
# and if the horizon is NOT 3, what is left?
for h in (1, 2, 3, 5):
    net_days = max(0, 3 - h) if h <= 3 else h - 3
    side = "still-flat" if h <= 3 else "net SHORT after 08-17"
    print(f"    h={h}: overlap {min(h,3)} of 3 live sessions; residual "
          f"{net_days} session(s) {side}")

# =========================================================== B. parent cells
print("\n=== B. parent cells vs the joint state (episodes, min_gap=5) ===")
cells = {
    "JOINT: GDX r5>=99 & GLD r5>=95": M,
    "PARENT: GDX r5>=99 alone": (g5 >= 99).fillna(False),
    "PARENT: GLD r5>=95 alone": (d5 >= 95).fillna(False),
    "COMPLEMENT: GDX r5>=99 & GLD r5<95": ((g5 >= 99) & (d5 < 95)).fillna(False),
}
for h in (1, 3, 5):
    rows = []
    ret = vehicle_ret(px, [("GDX", -1.0)], h, 1)
    base = ret.dropna()
    for lbl, c in cells.items():
        t = idx[c.reindex(idx, fill_value=False).values].intersection(base.index)
        epi = declusters(t, 5, base.index)
        v = ret.loc[epi].values
        if len(v) < 3:
            rows.append({"label": lbl, "n": len(v)})
            continue
        w = int((v > 0).sum())
        rows.append({"label": lbl, "n_days": len(t), "n_epi": len(epi),
                     "short_mean_pct": round(100*v.mean(), 3),
                     "excess_pct": round(100*(v.mean()-base.mean()), 3),
                     "hit": round(100*(v > 0).mean(), 1),
                     "sign_p": round(sign_test(w, len(v),
                                               float((base > 0).mean())), 4),
                     "worst_pct": round(100*v.min(), 2)})
    show(rows, f"SHORT GDX, h={h}")

# ------------------------------------------------------------- full battery
variants = {
    "GDX r5>=95 & GLD r5>=95": ((g5 >= 95) & (d5 >= 95)).fillna(False),
    "GDX r5>=97 & GLD r5>=95": ((g5 >= 97) & (d5 >= 95)).fillna(False),
    "GDX r5>=99 & GLD r5>=90": ((g5 >= 99) & (d5 >= 90)).fillna(False),
    "GDX r5>=99 & GLD r5>=97": ((g5 >= 99) & (d5 >= 97)).fillna(False),
    "GDX r5>=99 & GLD r5>=99": ((g5 >= 99) & (d5 >= 99)).fillna(False),
    "GDX r5>=99 alone": (g5 >= 99).fillna(False),
}
for h in (1, 3, 5):
    battery(px, M, [("GDX", -1.0)], h,
            f"C8 SHORT GDX on the metals joint thrust, h={h}",
            cost_bps=8.0, variants=variants, min_gap=5,
            event_kinds=("cpi", "ppi"))

# ------------------------------------------------- C/E concentration + eras
for h in (3, 5):
    print(f"\n=== C/E. episode detail, SHORT GDX h={h} ===")
    ret = vehicle_ret(px, [("GDX", -1.0)], h, 1)
    t = idx[M.values].intersection(ret.dropna().index)
    epi = declusters(t, 5, ret.dropna().index)
    v = ret.loc[epi].values
    yr = pd.Series(100*v, index=epi).groupby(epi.year).agg(["sum", "count"])
    print(yr.round(2).to_string())
    print(f"  positive years {int((yr['sum']>0).sum())}/{len(yr)}")
    mid = np.array([y % 4 == 2 for y in epi.year])
    show([summarize(v[mid], f"midterm (N={int(mid.sum())})"),
          summarize(v[~mid], f"non-midterm (N={int((~mid).sum())})")], "midterm")
    print(f"  {cluster_note(epi, v, k=3)}")
    print("  episodes:", ", ".join(f"{d.date()}:{100*x:+.1f}"
                                   for d, x in zip(epi, v)))
    # drop the best 2 episodes
    order = np.argsort(-v)
    keep = np.ones(len(v), bool)
    keep[order[:2]] = False
    print(f"  drop-2-best: {100*v[keep].mean():+.3f}% over {keep.sum()} episodes")

# ------------------------------------------------------------------ F. tail
print("\n=== F. tail of a naked short GDX on a thrust ===")
g1 = px["GDX"].pct_change().dropna()
t = idx[M.values]
nxt = []
for d in t:
    p = idx.searchsorted(d)
    if p + 4 < len(idx):
        nxt.append(px["GDX"].iloc[p+4] / px["GDX"].iloc[p+1] - 1)
nxt = np.array(nxt)
print(f"  worst 3-session move AGAINST a short entered on trigger: "
      f"{100*nxt.max():+.2f}%")
print(f"  GDX unconditional daily sd {100*g1.std():.2f}%, "
      f"max 1d {100*g1.max():+.2f}% on {g1.idxmax().date()}")
