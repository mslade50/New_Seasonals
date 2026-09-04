"""C2 round 1 -- the month turn on the metals / commodity complex.

Today is ME-2 (August's last trading day is 2026-08-31). Signal on the ME-3
close, entry MOC at the ME-2 close, hold h sessions (h=2 reaches the ME-0
close; h=3..7 reach into the new month).

Kills this has to clear:
  1. beat the basket's OWN unconditional drift over the same span, not zero
  2. the ME anchor is closed on equities in BOTH the month-position and
     month-of-year senses, suspended on rates, closed on FX -- if the
     commodity number looks like the equity one it is the same corpse
  3. famous calendar cells (turn-of-month included) were arbitraged away
     post-2013 -> era split is mandatory
  4. the midterm split (6 independent wrong-signed results in this repo)
  5. the offset placebo ladder over ME-m, m = 1..21
  6. August-only, since today is an August month turn
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

CMDTY = ["GLD", "SLV", "GDX", "DBC", "USO", "XLE", "XME", "FCX"]
px = pd.DataFrame({t: load_prices(CMDTY + ["SPY"])[t]["Close"]
                   for t in CMDTY + ["SPY"]}).dropna()
idx = px.index
print(f"common calendar: {idx[0].date()} .. {idx[-1].date()}  N={len(idx)}")

# ---- month-end positions -------------------------------------------------
ym = pd.Series(idx.year * 100 + idx.month, index=idx)
me_pos = []
for _, g in pd.Series(range(len(idx)), index=idx).groupby(ym.values):
    me_pos.append(int(g.iloc[-1]))
me_pos = np.array(sorted(me_pos))
print(f"month-end sessions: {len(me_pos)}  last = {idx[me_pos[-1]].date()}")


def sig_dates_for(m: int) -> pd.DatetimeIndex:
    """Signal date = ME-m  (=> entry MOC at ME-(m-1))."""
    p = me_pos - m
    p = p[(p >= 0) & (p < len(idx))]
    return idx[p]


BASKET = [(t, 1.0 / len(CMDTY)) for t in CMDTY]
sig3 = sig_dates_for(3)          # today's analogue: entry at ME-2
print(f"ME-3 signal dates: N={len(sig3)}  {sig3[0].date()} .. {sig3[-1].date()}")

mask = pd.Series(False, index=idx)
mask.loc[sig3] = True

battery(px, mask, BASKET, h=2,
        title="C2: EW commodity basket, ME-2 entry, hold to the ME-0 close",
        cost_bps=5.0, min_gap=15, event_kinds=("nfp",))

battery(px, mask, BASKET, h=5,
        title="C2: EW commodity basket, ME-2 entry, hold 5td into the new month",
        cost_bps=5.0, min_gap=15, event_kinds=("nfp",))

# ---- horizon table -------------------------------------------------------
print("\n" + "=" * 78)
print("horizon table, EW basket, ME-2 entry (excess = vs the basket's own")
print("unconditional drift over the same span)")
print("=" * 78)
rows = []
for h in (1, 2, 3, 4, 5, 6, 7, 8, 10):
    r = vehicle_ret(px, BASKET, h, 1)
    v = r.loc[sig3].dropna().values
    allv = r.dropna()
    w = int((v > 0).sum())
    rows.append({"h": h, "n": len(v), "mean_pct": round(100 * v.mean(), 4),
                 "drift_pct": round(100 * allv.mean(), 4),
                 "excess_pp": round(100 * (v.mean() - allv.mean()), 4),
                 "hit": round(100 * w / len(v), 1), "t": round(
                     v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2),
                 "sign_p": round(sign_test(w, len(v)), 4)})
show(rows, "C2 horizons")

# ---- per-ticker (a GRID -- charged as one) -------------------------------
print("\n" + "=" * 78)
print("per-ticker at h=2 and h=5. This is an 8x2 GRID; its best occupant is")
print("charged for the search (Sidak over 16 cells).")
print("=" * 78)
rows = []
for t in CMDTY:
    for h in (2, 5):
        r = vehicle_ret(px, [(t, 1.0)], h, 1)
        v = r.loc[sig3].dropna().values
        allv = r.dropna()
        w = int((v > 0).sum())
        tt = v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
        rows.append({"tkr": t, "h": h, "n": len(v),
                     "mean_pct": round(100 * v.mean(), 3),
                     "drift_pct": round(100 * allv.mean(), 3),
                     "excess_pp": round(100 * (v.mean() - allv.mean()), 3),
                     "hit": round(100 * w / len(v), 1), "t": round(tt, 2),
                     "sign_p": round(sign_test(w, len(v)), 4)})
df = pd.DataFrame(rows).sort_values("excess_pp", ascending=False)
print(df.to_string(index=False))
best = df.iloc[0]
raw_p = float(best["sign_p"])
print(f"\n  best cell {best['tkr']} h={best['h']}: excess {best['excess_pp']:+.3f}pp, "
      f"raw sign p {raw_p:.4f} -> Sidak over 16 cells = "
      f"{1-(1-raw_p)**16:.4f}")

# ---- offset placebo ladder ----------------------------------------------
print("\n" + "=" * 78)
print("OFFSET PLACEBO LADDER: signal at ME-m, m = 1..21, EW basket.")
print("A plateau kills. The live cell is m=3 (entry at the ME-2 close).")
print("=" * 78)
for h in (2, 5):
    r = vehicle_ret(px, BASKET, h, 1)
    rows = []
    for m in range(1, 22):
        v = r.loc[sig_dates_for(m)].dropna().values
        rows.append({"m(ME-m)": m, "n": len(v),
                     "mean_pct": round(100 * v.mean(), 4),
                     "hit": round(100 * (v > 0).mean(), 1)})
    d = pd.DataFrame(rows).sort_values("mean_pct", ascending=False).reset_index(drop=True)
    rank = int(d.index[d["m(ME-m)"] == 3][0]) + 1
    print(f"\n  h={h}: LIVE CELL m=3 RANKS {rank} of {len(d)}")
    print(d.to_string(index=False))

# ---- eras, cycle, month-of-year -----------------------------------------
print("\n" + "=" * 78)
print("eras / midterm / August-only, EW basket")
print("=" * 78)
for h in (2, 5):
    r = vehicle_ret(px, BASKET, h, 1)
    s = sig3[r.loc[sig3].notna().values]
    v = r.loc[s].values
    allv = r.dropna().mean()
    rows = []
    for lbl, m in (("pre-2013", s.year < 2013), ("2013+", s.year >= 2013),
                   ("pre-2018", s.year < 2018), ("2018+", s.year >= 2018),
                   ("MIDTERM (y%4==2)", (s.year % 4) == 2),
                   ("non-midterm", (s.year % 4) != 2),
                   ("AUGUST only", s.month == 8),
                   ("ex-August", s.month != 8)):
        sub = v[m]
        if len(sub) == 0:
            continue
        rr = summarize(sub, lbl)
        rr["excess_pp"] = round(rr["mean_pct"] - 100 * allv, 3)
        w = int((sub > 0).sum())
        rr["sign_p"] = round(sign_test(w, len(sub)), 4)
        rows.append(rr)
    show(rows, f"h={h}")

# ---- is it the same corpse? SPY over the identical cell ------------------
print("\n" + "=" * 78)
print("SAME-CORPSE CHECK: SPY over the identical ME-2 entry cell (the equity")
print("month turn was closed 2026-08-26). If the commodity number tracks it,")
print("the basket is carrying equity beta, not a commodity flow.")
print("=" * 78)
rows = []
for h in (2, 5):
    rs = vehicle_ret(px, [("SPY", 1.0)], h, 1)
    rb = vehicle_ret(px, BASKET, h, 1)
    resid = rb - rs  # unit-beta crude hedge; also report the OLS version
    rb_ = rb.pct_change  # noqa
    for lbl, ser in (("SPY", rs), ("basket", rb), ("basket - SPY", resid)):
        v = ser.loc[sig3].dropna().values
        a = ser.dropna()
        rows.append({"h": h, "leg": lbl, "n": len(v),
                     "mean_pct": round(100 * v.mean(), 3),
                     "drift_pct": round(100 * a.mean(), 3),
                     "excess_pp": round(100 * (v.mean() - a.mean()), 3)})
print(pd.DataFrame(rows).to_string(index=False))

# beta-hedged residual, trailing-252 OLS beta of the basket on SPY
bret = sum(w * px[t].pct_change(fill_method=None) for t, w in BASKET)
sret = px["SPY"].pct_change(fill_method=None)
beta = (bret.rolling(252).cov(sret) / sret.rolling(252).var()).shift(1)
print(f"\n  mean trailing beta of the basket on SPY at the signal dates: "
      f"{beta.loc[sig3].mean():.2f}")
for h in (2, 5):
    resid = vehicle_ret(px, BASKET, h, 1) - beta * vehicle_ret(px, [("SPY", 1.0)], h, 1)
    v = resid.loc[sig3].dropna().values
    a = resid.dropna()
    w = int((v > 0).sum())
    print(f"  h={h} beta-hedged residual: cond {100*v.mean():+.3f}% (N={len(v)}) "
          f"vs all {100*a.mean():+.3f}% -> excess {100*(v.mean()-a.mean()):+.3f}pp, "
          f"record {w}-{len(v)-w}, sign p {sign_test(w, len(v)):.4f}")
