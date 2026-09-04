"""C10 round 1+2: SMH bottom-percentile 63d laggard (pct_rank(close,63) <= 2),
OUTRIGHT only (the SMH/QQQ pair is registry-dead and this morning's recon
agrees), plus the XLK companion.

Registry traps this is written against:
- the pair form over-selects bear tape by +29pp vs base rate, so it is a regime
  bet. THE SAME TEST IS RUN HERE ON THE OUTRIGHT: if rank63<=2 fires mostly
  below the 200d, "the edge" is just buying dips in downtrends.
- TODAY IS THE OPPOSITE OF THAT: SMH sits 25.0% ABOVE its 200d while rank63 is
  0.8. So the decisive number is the conditional cell restricted to days that
  look like today, not the pooled cell.
- an outright long-semis bet must beat QQQ and SPY as a vehicle, so both
  relative forms are priced.
- CPI tonight and PPI tomorrow land inside any hold; AMAT prints 08-13 and NVDA
  08-26, which is single-name event risk inside SMH.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TKRS = ["SMH", "QQQ", "SPY", "XLK", "SOXX"]
px = close_panel([t for t in TKRS if t != "SOXX"] + ["SOXX"])
px = px.dropna(subset=["SMH"], how="any") if "SMH" in px else px
idx = px.index

r63 = pct_rank(px["SMH"], 63)
sma200 = px["SMH"].rolling(200).mean()
above200 = px["SMH"] / sma200 - 1.0
spy_above200 = px["SPY"] / px["SPY"].rolling(200).mean() - 1.0

print("=== 0. TRIGGER SANITY (must match pitch_tape.json) ===")
print(f"  last bar          : {idx[-1].date()}")
print(f"  SMH pct_rank(63)  : {r63.iloc[-1]:.1f}    (tape rank_63d 0.8)")
print(f"  SMH dist sma200   : {100*above200.iloc[-1]:+.2f}%  (tape +25.02%)")
print(f"  XLK pct_rank(63)  : {pct_rank(px['XLK'], 63).iloc[-1]:.1f}   (tape 32.1)")
print(f"  SPY dist sma200   : {100*spy_above200.iloc[-1]:+.2f}%  (tape +9.78%)")

MASK = (r63 <= 2).reindex(idx, fill_value=False).fillna(False)

depth = 0
for i in range(len(idx) - 1, -1, -1):
    if bool(MASK.iloc[i]):
        depth += 1
    else:
        break
print(f"\n=== 0b. TODAY'S CLUSTER DEPTH === consecutive trigger days incl today: {depth}")
print(f"  trigger days in last 21 sessions: {int(MASK.values[-21:].sum())}")

# --- the decisive regime question ------------------------------------------
print("\n=== 0c. DOES THE TRIGGER OVER-SELECT BEAR TAPE? (registry's pair kill) ===")
base_bear = float((spy_above200 < 0).reindex(idx).fillna(False).mean())
trig_bear = float((spy_above200 < 0).reindex(idx).fillna(False)[MASK.values].mean())
print(f"  SPY below 200d, all days      : {100*base_bear:.1f}%")
print(f"  SPY below 200d, trigger days  : {100*trig_bear:.1f}%   "
      f"(over-selection {100*(trig_bear-base_bear):+.1f}pp)")
smh_bear = float((above200 < 0).reindex(idx).fillna(False)[MASK.values].mean())
print(f"  SMH below its OWN 200d on trigger days: {100*smh_bear:.1f}%   "
      f"(TODAY SMH is +25.0% ABOVE)")
hi = (above200 >= 0.15).reindex(idx, fill_value=False).fillna(False)
print(f"  trigger days ALSO >=15% above own 200d (today's state): "
      f"{int((MASK & hi).sum())} of {int(MASK.sum())}")

variants = {
    "rank63<=1": pct_rank(px["SMH"], 63) <= 1,
    "rank63<=2": MASK,
    "rank63<=5": pct_rank(px["SMH"], 63) <= 5,
    "rank63<=10": pct_rank(px["SMH"], 63) <= 10,
    "rank63<=2 & >200d": MASK & (above200 > 0),
    "rank63<=2 & <200d": MASK & (above200 <= 0),
}
for h in (1, 3, 5):
    battery(px, MASK, [("SMH", 1.0)], h, "C10 SMH outright, rank63<=2", 6.0,
            variants=variants if h == 3 else None, event_kinds=("cpi", "ppi"))

print("\n\n########## C10 VEHICLE COMPETITION (must beat QQQ / SPY) ##########")
for legs, lbl in ([("SMH", 1.0), ("QQQ", -1.0)], "SMH-QQQ (registry-dead, confirm)"), \
                 ([("SMH", 1.0), ("SPY", -1.0)], "SMH-SPY"):
    battery(px, MASK, legs, 3, f"C10 pair {lbl}", 6.0, event_kinds=("cpi", "ppi"))

print("\n\n########## C10 COMPANION: XLK ##########")
xlk_mask = (pct_rank(px["XLK"], 63) <= 2).reindex(idx, fill_value=False).fillna(False)
print(f"  XLK rank63 TODAY = {pct_rank(px['XLK'], 63).iloc[-1]:.1f} -> "
      f"trigger live? {bool(xlk_mask.iloc[-1])}")
battery(px, xlk_mask, [("XLK", 1.0)], 3, "C10b XLK rank63<=2", 6.0,
        event_kinds=("cpi", "ppi"))

# --- round 2 ---------------------------------------------------------------
print("\n\n########## C10 ROUND 2: TODAY-LIKE SUBSET, ERA, MIDTERM ##########")
for h in (1, 3, 5):
    ret = vehicle_ret(px, [("SMH", 1.0)], h, 1)
    valid = ret.notna()
    sig = idx[MASK.values & valid.values]
    epi = declusters(sig, max(h, 5), idx)
    v = ret.loc[epi].values
    span = (idx >= sig[0]) & (idx <= sig[-1])
    ctrl = ret[span].dropna()
    base = float((ctrl > 0).mean())
    a200 = above200.reindex(epi).values
    mid = (pd.DatetimeIndex(epi).year % 4 == 2)
    rows = [
        summarize(v, f"h={h} ALL episodes (N={len(v)})"),
        summarize(v[a200 > 0], f"h={h} SMH above own 200d (N={int((a200>0).sum())})"),
        summarize(v[a200 >= 0.15], f"h={h} SMH >=15% above 200d = TODAY (N={int((a200>=0.15).sum())})"),
        summarize(v[a200 <= 0], f"h={h} SMH below 200d (N={int((a200<=0).sum())})"),
        summarize(v[mid], f"h={h} midterm (TODAY, N={int(mid.sum())})"),
        summarize(v[~mid], f"h={h} non-midterm (N={int((~mid).sum())})"),
    ]
    show(rows, f"today-like subset h={h}   [drift {100*ctrl.mean():+.3f}%, base hit {100*base:.1f}%]")
    w = int((v > 0).sum())
    print(f"  ALL sign p vs own base = {sign_test(w, len(v), base):.4f}")
    sub = v[a200 > 0]
    if len(sub) >= 3:
        print(f"  ABOVE-200d sign p = {sign_test(int((sub>0).sum()), len(sub), base):.4f}, "
              f"excess = {100*(sub.mean()-ctrl.mean()):+.3f}%")
    print(f"  concentration: {cluster_note(epi, v)}")
    print(f"  episode year histogram: "
          f"{dict(pd.Series(v).groupby(pd.DatetimeIndex(epi).year.values).count())}")
