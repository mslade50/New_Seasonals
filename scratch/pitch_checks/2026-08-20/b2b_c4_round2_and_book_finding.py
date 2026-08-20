"""C4 round 2, and the numbers behind the BOOK FINDING owed to McKinley.

Round 1 established that the "August inverts the post-opex vol crush" claim is
built on the PRE-BREAK -1x SVXY. This finishes the job: prices the short-vol
expression honestly, tests whether the spot-^VIX August cell is anything once
its top episode is removed, and gives V4's August-vs-September comparison the
sign tests it needs to be quotable in a book decision.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
warnings.filterwarnings("ignore")
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

BREAK = pd.Timestamp("2018-02-28")
px = close_panel(["SVXY", "^VIX", "SPY"])
d = px.index
opex = pd.DatetimeIndex(sorted(set(load_events(["opex"])["date"]) & set(d)))

# =========================================================== BOOK FINDING
print("=" * 78)
print("BOOK FINDING: should August be excluded from V4 the way September is?")
print("(long SVXY, MOC on the opex close, exit MOC +3; POST-BREAK ONLY)")
print("=" * 78)
sv3 = fwd_lag(px["SVXY"], 3, lag=0)
v4_ok = pd.DatetimeIndex([x for x in opex if x.month != 9
                          and not (x.month in (11, 12) and x.year % 4 != 2)])
groups = {
    "V4 AUGUST": [x for x in v4_ok if x.month == 8 and x >= BREAK],
    "V4 ex-August": [x for x in v4_ok if x.month != 8 and x >= BREAK],
    "SEPTEMBER (excluded by spec)": [x for x in opex
                                     if x.month == 9 and x >= BREAK],
}
for lbl, dates in groups.items():
    s = sv3.reindex(pd.DatetimeIndex(dates)).dropna()
    wins = int((s > 0).sum())
    print(f"\n  {lbl}: N={len(s)}  mean {100*s.mean():+.3f}%  median "
          f"{100*np.median(s):+.3f}%  hit {100*(s>0).mean():.0f}%  "
          f"worst {100*s.min():+.2f}%  best {100*s.max():+.2f}%")
    print(f"    record {wins}-{len(s)-wins}, sign p (>=wins) "
          f"{sign_test(wins, len(s)):.4f}, sign p (<=wins, i.e. a LOSING "
          f"cell) {sign_test(len(s)-wins, len(s)):.4f}, "
          f"bootstrap P(mean<=0) {bootstrap_p_le0(s.values):.3f}")
print("\n  -> September post-break is 0-for-8. August post-break is 5-for-8 "
      "and its mean BEATS the rest of V4. The spec's September carve-out is "
      "confirmed; an August carve-out would be backwards.")

print("\n  Robustness: does the August-vs-rest ordering hold at other exits?")
for h in (1, 2, 3, 4, 5):
    s = fwd_lag(px["SVXY"], h, lag=0)
    a = s.reindex(pd.DatetimeIndex(groups["V4 AUGUST"])).dropna()
    r = s.reindex(pd.DatetimeIndex(groups["V4 ex-August"])).dropna()
    sep = s.reindex(pd.DatetimeIndex(
        groups["SEPTEMBER (excluded by spec)"])).dropna()
    print(f"    exit +{h}: August {100*a.mean():+6.3f}% (N={len(a)})  "
          f"rest-of-V4 {100*r.mean():+6.3f}% (N={len(r)})  "
          f"September {100*sep.mean():+6.3f}% (N={len(sep)})")

# ================================================ spot VIX, concentration
print("\n\n" + "=" * 78)
print("SPOT ^VIX AUGUST CELL (opex-1 entry): is it anything without 2015?")
print("=" * 78)
entry = d[np.clip(d.get_indexer(opex) - 1, 0, len(d) - 1)]
aug_entry = entry[entry.month == 8]
for h in (2, 3, 5):
    s = fwd_lag(px["^VIX"], h, lag=0).reindex(aug_entry).dropna()
    base = fwd_lag(px["^VIX"], h, lag=0).dropna()
    ex15 = s[s.index.year != 2015]
    ex2 = s[~s.index.year.isin([2015, 2022])]
    print(f"  h={h}: full {100*s.mean():+.3f}% (N={len(s)}, median "
          f"{100*np.median(s):+.3f}%, up-rate {100*(s>0).mean():.0f}%)  |  "
          f"ex-2015 {100*ex15.mean():+.3f}%  |  ex-2015/2022 "
          f"{100*ex2.mean():+.3f}%  |  all-days base {100*base.mean():+.3f}%")
print("\n  A cell whose MEAN is positive, MEDIAN negative and up-rate 42% is "
      "a left-tail description, not a directional edge. The 'fat left tail "
      "for SVXY' reading of the same fact is correct and untradeable: you "
      "cannot enter a tail with a time stop and 50% hit.")

# =========================================== the short expression, priced
print("\n\n" + "=" * 78)
print("THE TRADEABLE FORM: short SVXY (a long-vol expression), post-break")
print("=" * 78)
for h in (2, 3, 5, 10):
    s = fwd_lag(px["SVXY"], h, lag=0).reindex(aug_entry).dropna()
    s = s[s.index >= BREAK]
    short = -s
    wins = int((short > 0).sum())
    print(f"  h={h:2d}  SHORT SVXY: {100*short.mean():+.3f}% "
          f"(N={len(short)}, hit {100*(short>0).mean():.0f}%, record "
          f"{wins}-{len(short)-wins}, sign p {sign_test(wins, len(short)):.3f})"
          f"  worst {100*short.min():+.2f}%  best {100*short.max():+.2f}%")
print("\n  Cost, stated honestly and not measurable from this repo's data:")
print("   - SVXY spread is 1-3 c on a ~$50 print, ~4-6 bps round trip.")
print("   - Short borrow on a levered vol ETP is the real cost and is NOT in")
print("     master_prices; it is routinely tens of bps to several percent")
print("     annualised and spikes exactly when vol spikes, i.e. inside the")
print("     window the trade needs. Nothing here can price it, so the")
print("     mechanism cannot be verified on the cost side either.")
print("   - The sign is wrong before any of that matters, so cost is moot.")

# ------------------------------------------------------ local control
print("\n\n" + "=" * 78)
print("LOCAL CONTROL on the August SVXY cell (post-break), +/-126td "
      "ex-trigger")
print("=" * 78)
for h in (3, 5):
    s = fwd_lag(px["SVXY"], h, lag=0)
    valid = s.dropna().index
    valid = valid[valid >= BREAK]
    trig = pd.DatetimeIndex([x for x in aug_entry if x >= BREAK])
    loc = local_control(valid, trig)
    show([summarize(s.reindex(trig).dropna().values, f"h={h} AUG post-break"),
          summarize(s.reindex(loc).dropna().values, f"h={h} local ctrl"),
          summarize(s.reindex(valid).values, f"h={h} all days post-break")])
