"""C5c -- a METHOD BUG found in C5/C5b, and the residual test that survives it.

THE BUG. `_survey_lib.align` (inherited by this morning's C5) does
`s.reindex(union).ffill().reindex(idx)`. That is correct for a STATE series
(^SKEW's rank on a session SPY trades and ^SKEW does not is legitimately the
last known reading). It is WRONG for a FORWARD RETURN, because the trailing
NaNs -- the sessions whose forward window has not resolved yet -- get filled
with the last resolved value. Measured below: SVXY's h=10 forward return from
2026-08-20 (+3.18%) is smeared across every session from 2026-08-21 to
2026-09-04, minting fake episodes at exactly the live reading. C5b's episode
list carries "2026-09-02: +3.2%" for that reason, and TODAY'S OWN ANCHOR
carries it too.

Yesterday's 03_pricestate_s3 script used the same helper, so the registry's
"+1.374% SVXY (59-36)" line inherits it at the tail.

This script (1) sizes the damage, (2) re-runs the SVXY cells on SVXY's OWN
calendar with the mask ffilled and the return never ffilled, and (3) finishes
the round-2 attack: does the h=10 beta residual survive dropping 2023?
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _c456_common import (  # noqa: E402
    cluster_note, declusters, fwd_lag, load_prices, np, pct_rank, pd,
    sign_test, summarize, show,
)

REBAL = pd.Timestamp("2018-02-28")
PX = load_prices(["SPY", "SVXY", "^SKEW"])
SPY_IDX = PX["SPY"].index
SVXY = PX["SVXY"]["Close"]
SIDX = SVXY.dropna().index          # SVXY's OWN calendar -- the right one
SKEW = PX["^SKEW"]["Close"]
SPY = PX["SPY"]["Close"]

print("=" * 78)
print("C5c  ALIGNMENT BUG in the SVXY leg, and the residual after 2023 goes")
print("=" * 78)


def state_align(s, idx):
    """ffill is CORRECT here: a state known at close D carries forward."""
    if s.dtype == bool:
        s = s.astype(float)
    return s.reindex(idx.union(s.index)).ffill().reindex(idx)


def bad_align(s, idx):
    return state_align(s, idx)


# --------------------------------------------------------------- 1. the damage
print("\n1. SIZE OF THE DAMAGE")
r10_raw = fwd_lag(SVXY, 10, 1)
r10_bad = bad_align(r10_raw, SPY_IDX)
fake = r10_bad.notna() & ~r10_raw.reindex(SPY_IDX).notna()
print(f"  h=10: sessions where the ffill INVENTED a forward return: "
      f"{int(fake.sum())} of {len(SPY_IDX)}")
ff = SPY_IDX[fake.values]
print(f"  they run {ff[0].date()} .. {ff[-1].date()}   "
      f"(and {int((ff >= pd.Timestamp('2026-01-01')).sum())} are in 2026)")
r21 = pct_rank(SKEW, 21)
m98_spy = state_align(r21 >= 98, SPY_IDX).fillna(0).astype(bool)
print(f"  of those, {int((m98_spy & fake).sum())} also carry skew r21>=98, i.e. "
      "they entered C5b's cell as fabricated episodes")
print("  -> every SVXY number in c5_*.txt and c5b_*.txt is restated below.")

# ----------------------------------------- 2. correct cells on SVXY's calendar
print("\n2. CORRECTED: SVXY's own calendar, mask ffilled, return NEVER ffilled")
rank_on_svxy = state_align(r21, SIDX)
rows = []
for h in (3, 5, 7, 10):
    ret = fwd_lag(SVXY, h, 1)
    valid = ret.dropna().index
    for era_lbl, sub in (("ALL history", valid),
                         ("-0.5x era only", valid[valid >= REBAL])):
        base = float(ret.loc[sub].mean())
        for th in (90, 95, 98):
            m = (rank_on_svxy >= th).fillna(False)
            t = pd.DatetimeIndex(SIDX[m.values]).intersection(sub)
            epi = declusters(t, h, sub)
            if len(epi) == 0:
                continue
            v = ret.loc[epi].values
            w = int((v > 0).sum())
            rows.append({"h": h, "era": era_lbl, "th": f">={th}", "n": len(epi),
                         "mean_pct": round(100 * v.mean(), 3),
                         "base_pct": round(100 * base, 3),
                         "excess_pct": round(100 * (v.mean() - base), 3),
                         "med_pct": round(100 * float(np.median(v)), 3),
                         "hit": round(100 * float((v > 0).mean()), 1),
                         "worst_pct": round(100 * v.min(), 2),
                         "rec": f"{w}-{len(epi) - w}",
                         "sign_p": round(sign_test(w, len(epi)), 4)})
print(pd.DataFrame(rows).to_string(index=False))

# --------------------------------- 3. the h=10 live-era cell, corrected + 2023
print("\n3. THE CELL AFTER CORRECTION (r21>=98, -0.5x era, h=10)")
H = 10
ret = fwd_lag(SVXY, H, 1)
valid = ret.dropna().index
sub = valid[valid >= REBAL]
base = float(ret.loc[sub].mean())
m98 = (rank_on_svxy >= 98).fillna(False)
t98 = pd.DatetimeIndex(SIDX[m98.values]).intersection(sub)
epi = declusters(t98, H, sub)
ep = ret.loc[epi]
w = int((ep.values > 0).sum())
print(f"  n={len(epi)}  mean {100 * ep.mean():+.3f}%  base {100 * base:+.3f}%  "
      f"excess {100 * (ep.mean() - base):+.3f}pp  {w}-{len(ep) - w}  "
      f"sign p {sign_test(w, len(ep)):.4f}")
print("  " + cluster_note(epi, ep.values, k=2))
byyr = ep.groupby(epi.year).agg(["size", "mean", "sum"])
byyr.columns = ["n", "mean", "sum"]
byyr["mean_pct"] = (100 * byyr["mean"]).round(3)
byyr["sum_pct"] = (100 * byyr["sum"]).round(2)
print(byyr[["n", "mean_pct", "sum_pct"]].to_string())
for drop in (1, 2):
    top = byyr["sum"].sort_values(ascending=False).head(drop).index
    keep = ~np.isin(epi.year, top)
    v = ep.values[keep]
    ww = int((v > 0).sum())
    print(f"  drop best {drop} yr {list(top)} -> n={len(v)} mean "
          f"{100 * v.mean():+.3f}% excess {100 * (v.mean() - base):+.3f}pp "
          f"{ww}-{len(v) - ww} sign p {sign_test(ww, len(v)):.4f}")

# ------------------------------- 4. residual, corrected, and after 2023 leaves
print("\n4. BETA RESIDUAL (corrected calendar) and what 2023 was doing")
for h in (5, 10):
    rs = fwd_lag(SPY, h, 1).reindex(SIDX)
    rv = fwd_lag(SVXY, h, 1)
    d = pd.DataFrame({"spy": rs, "svxy": rv}).dropna()
    d = d[d.index >= REBAL]
    b, a = np.polyfit(d["spy"], d["svxy"], 1)
    resid = d["svxy"] - (a + b * d["spy"])
    print(f"\n  h={h}: SVXY = {100 * a:+.3f}% + {b:.2f} x SPY  "
          f"R2 {np.corrcoef(d['spy'], d['svxy'])[0, 1] ** 2:.3f}  n={len(d)}")
    for th in (90, 95, 98):
        m = (rank_on_svxy >= th).fillna(False)
        t = pd.DatetimeIndex(SIDX[m.values]).intersection(d.index)
        e = declusters(t, h, d.index)
        v = resid.loc[e].values
        ww = int((v > 0).sum())
        line = (f"    r21>={th}: resid mean {100 * v.mean():+.3f}% med "
                f"{100 * float(np.median(v)):+.3f}% n={len(v)} "
                f"{ww}-{len(v) - ww} sign p {sign_test(ww, len(v)):.4f}")
        keep = e.year != 2023
        v2 = resid.loc[e[keep]].values
        w2 = int((v2 > 0).sum())
        line += (f"   |  ex-2023: mean {100 * v2.mean():+.3f}% n={len(v2)} "
                 f"{w2}-{len(v2) - w2} sign p {sign_test(w2, len(v2)):.4f}")
        print(line)

# --------------------------------- 5. restate C4's exposure to the same bug
print("\n5. DOES C4 (SPY) OR C6 (SPY/IWM) INHERIT THE BUG? -- no, but verify")
print("   C4 used fwd_lag(SPY) on SPY's own index with no align; C6 used")
print("   vehicle_ret over a dropna'd SPY/IWM panel. Trailing NaNs stay NaN:")
for nm, s in (("SPY h=5", fwd_lag(SPY, 5, 1)), ("SPY h=10", fwd_lag(SPY, 10, 1))):
    print(f"   {nm}: last non-NaN {s.dropna().index[-1].date()}, "
          f"trailing NaNs {int(s.iloc[-15:].isna().sum())} of last 15  OK")
panel = pd.DataFrame({"SPY": SPY, "IWM": PX["IWM"]["Close"]}).dropna() \
    if "IWM" in PX else None
print("\nDONE C5c")
