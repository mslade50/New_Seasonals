"""C10 round 1: tail hedges bid while at-the-money vol is dead.

^SKEW 21-day RETURN rank 99.6 (level 144.12) while the VIX 21-day range sits at
the 1st-4th percentile. Untested form: skew's 21-day RANK (not level, not r5)
crossed with range compression, in BOTH directions.

Registry priors:
  2026-08-12  "a skew spike with a low-vol filter attached -- the filter
              SUBTRACTS: it throws away 81 episodes and HALVES the excess from
              +0.372% to +0.175%."  -> gate attribution is mandatory here.
  watchlist 6 midterm block: the midterm intersection was N=20 at -0.174%.
  2026-08-14  ^SKEW's median has drifted; trailing-252 and full-history
              percentiles disagree violently -> print BOTH.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import (battery, close_panel, declusters, era_split, horizon_scan,
                       load_prices, pct_rank, show, sign_test, summarize,
                       vehicle_ret, bootstrap_p_le0)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)

print("=" * 78)
print("C10  ^SKEW 21d-return rank >= 95  x  VIX 21d range percentile <= 15")
print("=" * 78)

raw = load_prices(["^SKEW", "^VIX", "SPY", "SVXY", "UVXY"])
skew = raw["^SKEW"]["Close"].dropna()
vix = raw["^VIX"]["Close"].dropna()
spy = raw["SPY"]["Close"].dropna()
print("history: ^SKEW", skew.index[0].date(), "->", skew.index[-1].date(),
      "| ^VIX", vix.index[0].date(), "| SPY", spy.index[0].date(),
      "| SVXY", raw["SVXY"]["Close"].dropna().index[0].date())

# ---- the two conventions the 2026-08-14 finding demands -------------------
lvl_252 = skew.rolling(252).rank(pct=True) * 100
lvl_full = skew.expanding(252).rank(pct=True) * 100
skew_r21 = pct_rank(skew, 21, 252)
skew_r5 = pct_rank(skew, 5, 252)
print(f"\nLIVE ^SKEW {skew.iloc[-1]:.2f}")
print(f"  LEVEL pctile trailing-252   : {lvl_252.iloc[-1]:.1f}")
print(f"  LEVEL pctile FULL HISTORY   : {lvl_full.iloc[-1]:.1f}   <-- conventions disagree")
print(f"  21d-RETURN rank (252)       : {skew_r21.iloc[-1]:.1f}   (this cell's object)")
print(f"  5d-RETURN rank (252)        : {skew_r5.iloc[-1]:.1f}   (watchlist 6's object)")
print(f"  ^SKEW median: full history {skew.median():.2f}  last 252d {skew.tail(252).median():.2f}"
      f"  first 252d {skew.head(252).median():.2f}")

# ---- VIX range compression ------------------------------------------------
rng21 = vix.rolling(21).max() - vix.rolling(21).min()
rel = rng21 / vix.rolling(21).mean()
RNG = rel.rolling(252).rank(pct=True) * 100
ABS = rng21.rolling(252).rank(pct=True) * 100
print(f"\nLIVE VIX {vix.iloc[-1]:.2f}  rel-range pctile {RNG.iloc[-1]:.2f}"
      f"  abs-range pctile {ABS.iloc[-1]:.2f}")

px = close_panel(["SPY", "SVXY"])
cal = spy.index
SK = skew_r21.reindex(cal)
RG = RNG.reindex(cal)
LV252 = lvl_252.reindex(cal)
LVF = lvl_full.reindex(cal)

SKEW_HI = SK >= 95
RANGE_DEAD = RG <= 15
JOINT = SKEW_HI & RANGE_DEAD
print(f"\ntrigger day counts on SPY calendar (n={len(cal)}):")
print(f"  SKEW r21 >= 95              : {int(SKEW_HI.sum())}")
print(f"  VIX rel-range pctile <= 15  : {int(RANGE_DEAD.sum())}")
print(f"  JOINT (live cell)           : {int(JOINT.sum())}")
print(f"  SKEW_HI & range NOT dead    : {int((SKEW_HI & ~RANGE_DEAD).sum())}")

LONG = [("SPY", 1.0)]
SHORT = [("SPY", -1.0)]
VOL = [("SVXY", 1.0)]           # long SVXY = short vol, the book's short-vol vehicle

pj = JOINT.reindex(px.index, fill_value=False)
variants = {
    "skew r21>=90 & range<=15": ((SK >= 90) & RANGE_DEAD).reindex(px.index, fill_value=False),
    "skew r21>=95 & range<=15 (LIVE)": pj,
    "skew r21>=98 & range<=15": ((SK >= 98) & RANGE_DEAD).reindex(px.index, fill_value=False),
    "skew r21>=95 & range<=10": ((SK >= 95) & (RG <= 10)).reindex(px.index, fill_value=False),
    "skew r21>=95 & range<=25": ((SK >= 95) & (RG <= 25)).reindex(px.index, fill_value=False),
    "skew r21>=95 ALONE (no vol filter)": SKEW_HI.reindex(px.index, fill_value=False),
    "range<=15 ALONE": RANGE_DEAD.reindex(px.index, fill_value=False),
}
for h in (5, 10):
    battery(px, pj, LONG, h=h, title=f"C10 JOINT -> LONG SPY", cost_bps=1.5,
            variants=variants, min_gap=h, event_kinds=("nfp", "cpi", "fomc_decision"))

# ---- BOTH DIRECTIONS + the vol expression ---------------------------------
print("\n" + "=" * 78)
print("BOTH DIRECTIONS and the vol expression")
print("=" * 78)
for h in (1, 3, 5, 10):
    rows = []
    for lbl, legs in (("LONG SPY", LONG), ("SHORT SPY", SHORT), ("LONG SVXY (short vol)", VOL)):
        ret = vehicle_ret(px, legs, h, 1)
        valid = ret.notna()
        d = px.index[pj.values & valid.values]
        epi = declusters(d, h, px.index)
        r = summarize(ret.loc[epi].values, f"{lbl} JOINT episodes")
        r["ctl_all_pct"] = round(100 * ret[valid].mean(), 3)
        r["edge_pct"] = round(r.get("mean_pct", np.nan) - 100 * ret[valid].mean(), 3) if r["n"] else np.nan
        rows.append(r)
    show(rows, f"direction sweep, h={h}")

# ---- GATE ATTRIBUTION: does the range filter subtract, as in 2026-08-12? --
print("\n" + "=" * 78)
print("GATE ATTRIBUTION -- run it WITHOUT the range-compression filter")
print("=" * 78)
for h in (5, 10):
    ret = vehicle_ret(px, LONG, h, 1)
    valid = ret.notna()
    rows = []
    cells = {
        "A skew r21>=95 ALONE": SKEW_HI,
        "B JOINT = LIVE (skew hi & range dead)": JOINT,
        "C skew hi & range NOT dead (filter OFF)": SKEW_HI & ~RANGE_DEAD,
        "D range dead ALONE": RANGE_DEAD,
        "E range dead & skew NOT hi": RANGE_DEAD & ~SKEW_HI,
    }
    got = {}
    for lbl, m in cells.items():
        d = px.index[m.reindex(px.index, fill_value=False).values & valid.values]
        epi = declusters(d, h, px.index)
        r = summarize(ret.loc[epi].values, lbl)
        r["n_days"] = len(d)
        base = 100 * ret[valid].mean()
        r["edge_pct"] = round(r["mean_pct"] - base, 3) if r["n"] else np.nan
        rows.append(r)
        got[lbl] = ret.loc[epi].values
    rows.append(summarize(ret[valid].values, "CTRL-b all days"))
    show(rows, f"gate attribution LONG SPY, h={h}")
    b, c = got["B JOINT = LIVE (skew hi & range dead)"], got["C skew hi & range NOT dead (filter OFF)"]
    if len(b) > 1 and len(c) > 1:
        se = np.sqrt(b.var(ddof=1) / len(b) + c.var(ddof=1) / len(c))
        print(f"  h={h}: filter ON {100*b.mean():+.3f}% (n={len(b)}) vs filter OFF "
              f"{100*c.mean():+.3f}% (n={len(c)})  diff {100*(b.mean()-c.mean()):+.3f}pp "
              f"welch t {(b.mean()-c.mean())/se:+.2f}   "
              f"(negative diff reproduces the 2026-08-12 'filter subtracts' finding)")

# ---- MIDTERM BLOCK: does watchlist 6's block reproduce here? --------------
print("\n" + "=" * 78)
print("MIDTERM BLOCK (watchlist 6 recorded N=20 at -0.174% for its own cell)")
print("=" * 78)
yr = px.index.year
mid = pd.Series((yr % 4 == 2), index=px.index)
for h in (5, 10):
    for lbl, legs in (("LONG SPY", LONG), ("SHORT SPY", SHORT)):
        ret = vehicle_ret(px, legs, h, 1)
        valid = ret.notna()
        d = px.index[pj.values & valid.values]
        epi = declusters(d, h, px.index)
        m = mid.reindex(epi).values
        rows = [summarize(ret.loc[epi].values[m], f"{lbl} MIDTERM yrs (N={int(m.sum())})"),
                summarize(ret.loc[epi].values[~m], f"{lbl} non-midterm (N={int((~m).sum())})")]
        show(rows, f"{lbl} midterm split, h={h}")
        if m.sum():
            print("   midterm episode dates:",
                  ", ".join(str(x.date()) for x in pd.DatetimeIndex(epi)[m]))

# ---- level-convention sensitivity ----------------------------------------
print("\n" + "=" * 78)
print("LEVEL-CONVENTION SENSITIVITY (2026-08-14: the two percentiles disagree)")
print("=" * 78)
for h in (5,):
    ret = vehicle_ret(px, LONG, h, 1)
    valid = ret.notna()
    rows = []
    for lbl, m in {
        "LEVEL pctile(252) >= 95 & range dead": (LV252 >= 95) & RANGE_DEAD,
        "LEVEL pctile(FULL) >= 95 & range dead": (LVF >= 95) & RANGE_DEAD,
        "r21 rank >= 95 & range dead (LIVE)": JOINT,
        "r5 rank >= 95 & range dead (watchlist 6 form)": (skew_r5.reindex(cal) >= 95) & RANGE_DEAD,
    }.items():
        d = px.index[m.reindex(px.index, fill_value=False).values & valid.values]
        epi = declusters(d, h, px.index)
        r = summarize(ret.loc[epi].values, lbl)
        r["n_days"] = len(d)
        rows.append(r)
    rows.append(summarize(ret[valid].values, "CTRL-b all days"))
    show(rows, f"definition neighbours, h={h}")

# ---- horizon scan ---------------------------------------------------------
epi5 = declusters(px.index[pj.values & vehicle_ret(px, LONG, 5, 1).notna().values], 5, px.index)
show(horizon_scan(px, epi5, LONG, hs=(1, 2, 3, 5, 7, 10)), "horizon scan LONG SPY")
show(horizon_scan(px, epi5, VOL, hs=(1, 2, 3, 5, 7, 10)), "horizon scan LONG SVXY")
print("\nlast 12 JOINT episode dates (h=5 declustering):",
      ", ".join(str(d.date()) for d in epi5[-12:]))
print("\nDONE C10")
