"""C9 round 1+2: XLU 21d-rank washout (<=5), outright and paired, plus the
rate-sensitive family companions XLRE and VNQ.

Registry traps this is written against:
- pct_rank takes a PRICE series. TODAY's value is printed first and must match
  data/pitch_tape.json (XLU rank_21d = 4.8).
- Utilities are dead in FOUR previously tested expressions, all of which used a
  z10 washout. This trigger is a 21d RANK. If the two masks overlap heavily the
  cell is the same corpse, so the overlap is measured explicitly.
- The 2026-08-07 kill found the SPY-near-high gate HURTS this family
  (+0.605% ungated vs -0.123% gated). SPY is 0.35% off its high TODAY, so the
  gate is ON and the gated cell is the one that matters.
- Episode-level only; day-level t-stats on overlapping triggers are illegal.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TKRS = ["XLU", "XLRE", "VNQ", "XLP", "SPY", "TLT"]
px = close_panel(TKRS)
idx = px.index

rank21_xlu = pct_rank(px["XLU"], 21)
z10_xlu = zscore(px["XLU"], 10)
spy_dist_high = px["SPY"] / px["SPY"].rolling(252).max() - 1.0

print("=== 0. TRIGGER SANITY (must match pitch_tape.json) ===")
print(f"  last bar            : {idx[-1].date()}")
print(f"  XLU pct_rank(21)    : {rank21_xlu.iloc[-1]:.1f}   (tape rank_21d 4.8)")
print(f"  XLU z10             : {z10_xlu.iloc[-1]:.2f}   (tape z10 -1.49)")
print(f"  XLRE pct_rank(5)    : {pct_rank(px['XLRE'], 5).iloc[-1]:.1f}  (tape 7.1)")
print(f"  VNQ z10             : {zscore(px['VNQ'], 10).iloc[-1]:.2f}  (tape -1.61)")
print(f"  SPY dist 52w high   : {100*spy_dist_high.iloc[-1]:.2f}%  (tape -0.35%)")

MASK = (rank21_xlu <= 5).reindex(idx, fill_value=False).fillna(False)

# --- cluster depth today ----------------------------------------------------
depth = 0
for i in range(len(idx) - 1, -1, -1):
    if bool(MASK.iloc[i]):
        depth += 1
    else:
        break
print(f"\n=== 0b. TODAY'S CLUSTER DEPTH === consecutive trigger days incl today: {depth}")
run = MASK.values[-21:]
print(f"  trigger days in last 21 sessions: {int(run.sum())}")

# --- registry overlap: is this the z10 corpse? ------------------------------
z_mask = (z10_xlu <= -1.5).reindex(idx, fill_value=False).fillna(False)
both = (MASK & z_mask).sum()
print("\n=== 0c. REGISTRY OVERLAP vs the killed z10<=-1.5 utilities washout ===")
print(f"  rank21<=5 days      : {int(MASK.sum())}")
print(f"  z10<=-1.5 days      : {int(z_mask.sum())}")
print(f"  BOTH                : {int(both)}  "
      f"({100*both/max(1,MASK.sum()):.1f}% of the rank cell is inside the corpse)")

# --- round 1 battery, three horizons ---------------------------------------
variants = {
    "rank21<=3": pct_rank(px["XLU"], 21) <= 3,
    "rank21<=5": MASK,
    "rank21<=10": pct_rank(px["XLU"], 21) <= 10,
    "rank21<=15": pct_rank(px["XLU"], 21) <= 15,
    "rank5<=5": pct_rank(px["XLU"], 5) <= 5,
    "z10<=-1.5 (the corpse)": z_mask,
}
for h in (1, 3, 5):
    battery(px, MASK, [("XLU", 1.0)], h, f"C9 XLU outright, rank21<=5", 6.0,
            variants=variants if h == 3 else None,
            event_kinds=("cpi", "ppi"))

print("\n\n########## C9 PAIRED FORMS (registry: price the legs first) ##########")
for legs, lbl in ([("XLU", 1.0), ("SPY", -1.0)], "XLU vs SPY"), \
                 ([("XLU", 1.0), ("XLP", -1.0)], "XLU vs XLP"):
    battery(px, MASK, legs, 3, f"C9 pair {lbl}", 6.0, event_kinds=("cpi", "ppi"))

print("\n\n########## C9 FAMILY COMPANIONS ##########")
battery(px, (pct_rank(px["XLRE"], 5) <= 8).reindex(idx, fill_value=False).fillna(False),
        [("XLRE", 1.0)], 3, "C9b XLRE 5d rank<=8", 6.0, event_kinds=("cpi", "ppi"))
battery(px, (zscore(px["VNQ"], 10) <= -1.5).reindex(idx, fill_value=False).fillna(False),
        [("VNQ", 1.0)], 3, "C9c VNQ z10<=-1.5", 6.0, event_kinds=("cpi", "ppi"))

# --- round 2: gate attribution (SPY near high) + midterm split -------------
print("\n\n########## C9 ROUND 2: GATE ATTRIBUTION + REGIME SPLITS ##########")
for h in (1, 3, 5):
    ret = vehicle_ret(px, [("XLU", 1.0)], h, 1)
    valid = ret.notna()
    sig = idx[MASK.values & valid.values]
    epi = declusters(sig, max(h, 5), idx)
    v = ret.loc[epi].values
    span = (idx >= sig[0]) & (idx <= sig[-1])
    ctrl = ret[span].dropna()
    base = float((ctrl > 0).mean())

    near_high = spy_dist_high.reindex(epi).values >= -0.01
    mid = (pd.DatetimeIndex(epi).year % 4 == 2)
    rows = [
        summarize(v, f"h={h} ALL episodes (N={len(v)})"),
        summarize(v[near_high], f"h={h} SPY within 1% of 52w high (TODAY, N={int(near_high.sum())})"),
        summarize(v[~near_high], f"h={h} SPY NOT near high (N={int((~near_high).sum())})"),
        summarize(v[mid], f"h={h} midterm years (TODAY, N={int(mid.sum())})"),
        summarize(v[~mid], f"h={h} non-midterm (N={int((~mid).sum())})"),
    ]
    show(rows, f"gate attribution h={h}   [drift {100*ctrl.mean():+.3f}%, base hit {100*base:.1f}%]")
    w = int((v > 0).sum())
    print(f"  ALL episodes sign p vs own base rate = {sign_test(w, len(v), base):.4f}")
    if near_high.sum() >= 3:
        wg = int((v[near_high] > 0).sum())
        print(f"  GATED (today's state) sign p = {sign_test(wg, int(near_high.sum()), base):.4f}, "
              f"excess vs drift = {100*(v[near_high].mean()-ctrl.mean()):+.3f}%")
    print(f"  concentration ALL : {cluster_note(epi, v)}")
    yrs = pd.Series(v).groupby(pd.DatetimeIndex(epi).year.values).count()
    print(f"  episode year histogram: {dict(yrs)}")
