"""b1 / C3: the BARE dollar 21d PIT washout (rank21 <= 2, 252d lookback),
no rate leg. Vehicles DX-Y.NYB (futures-tradeable) and UUP (the ETF).

Direction is taken from the data. Round 1: battery on both vehicles, both
signs implied, plus the definition ladder, the midterm split, and the cost
line quoted EARLY because the dollar's daily sd is tiny.

Registry adjacents honoured:
 - "UUP as a dollar vehicle" (~6 bps of edge cannot pay the drag+spread);
   DX passed cost. So DX is the vehicle that matters; UUP is reported to
   show the effect is not a vehicle artefact (2026-08-24: matched episodes
   differ by 1.3 bps, 95.5% sign agreement).
 - "Long DX after ANY weak NFP close, h=5: +0.1826%, 12.2x cost" -- that
   entry says a WEAK-DOLLAR-close cell already pays. This is a rank-extreme
   parent of the same family; if the parent is flat while a 1-day close cell
   pays, that is a definitional finding, not a new trade.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change
import pandas as pd, numpy as np

pd.set_option("display.width", 220)

TK = ["DX-Y.NYB", "UUP"]
px = close_panel(TK)
px = px.dropna(subset=["DX-Y.NYB"])

# --- trigger, PIT ---
rk_dx = pct_rank(px["DX-Y.NYB"], 21, 252)
rk_uup = pct_rank(px["UUP"], 21, 252)
print(f"TODAY: DX rank21={rk_dx.iloc[-1]:.2f}  UUP rank21={rk_uup.iloc[-1]:.2f}")

mask = rk_dx <= 2.0
print(f"trigger days (DX rank21<=2): {int(mask.sum())} of {int(rk_dx.notna().sum())} "
      f"({100*mask.sum()/rk_dx.notna().sum():.2f}%)")
print("years:", dict(pd.Series(px.index[mask.values]).dt.year.value_counts().sort_index()))

# --- COST LINE FIRST (brief's instruction) ---
# daily sd of the dollar
sd_d = _valid_pct_change(px["DX-Y.NYB"], 1).std() * 100
print(f"\nDX daily sd = {sd_d:.3f}%.  DX futures round trip ~1.5 bps; UUP ~6 bps.")
print("Bar is 5x cost: DX needs >= 7.5 bps = 0.075%; UUP needs >= 30 bps = 0.300%.")

# --- direction: horizon scan on the LONG side for both vehicles ---
trig = px.index[mask.values]
for veh in ["DX-Y.NYB", "UUP"]:
    sub = px.dropna(subset=[veh])
    t2 = pd.DatetimeIndex(trig).intersection(sub.index)
    print(f"\n=== horizon scan LONG {veh} (N days {len(t2)}) ===")
    show(horizon_scan(sub, t2, [(veh, 1.0)], hs=(1, 2, 3, 5, 10)))

# --- battery at h=5 on DX (the cost-viable vehicle) ---
variants = {
    "rank21<=1": rk_dx <= 1.0,
    "rank21<=2 (pitched)": rk_dx <= 2.0,
    "rank21<=5": rk_dx <= 5.0,
    "rank21<=10": rk_dx <= 10.0,
    "rank21<=20": rk_dx <= 20.0,
    "rank21<=50 (PARENT/half)": rk_dx <= 50.0,
}
battery(px, mask, [("DX-Y.NYB", 1.0)], 5, "C3 LONG DX on rank21<=2", 1.5,
        variants=variants, event_kinds=("nfp", "cpi", "fomc_decision"))

battery(px.dropna(subset=["UUP"]), mask, [("UUP", 1.0)], 5,
        "C3 LONG UUP on rank21<=2", 6.0, variants=variants,
        event_kinds=("nfp", "cpi", "fomc_decision"))

# --- midterm split (today is midterm) ---
print("\n=== midterm split, DX h=5, episodes ===")
for h in (3, 5, 10):
    ret = vehicle_ret(px, [("DX-Y.NYB", 1.0)], h, 1)
    valid = ret.dropna().index
    t2 = pd.DatetimeIndex(trig).intersection(valid)
    epi = declusters(t2, h, valid)
    yrs = pd.DatetimeIndex(epi).year
    mid = (yrs % 4 == 2)
    rows = [summarize(ret.loc[epi[mid]].values, f"h={h} MIDTERM"),
            summarize(ret.loc[epi[~mid]].values, f"h={h} non-midterm")]
    show(rows)

# --- continuation vs reversion: what did the dollar do NEXT vs what it just did ---
print("\n=== dose response: forward h=5 by rank21 bucket (all days, episodes not needed) ===")
ret5 = vehicle_ret(px, [("DX-Y.NYB", 1.0)], 5, 1)
buckets = [(0, 2), (2, 5), (5, 10), (10, 25), (25, 50), (50, 75), (75, 90), (90, 98), (98, 100.01)]
rows = []
for lo, hi in buckets:
    m = (rk_dx >= lo) & (rk_dx < hi) & ret5.notna()
    rows.append(summarize(ret5[m.values].values, f"rank21 [{lo},{hi})"))
show(rows)

# --- trailing-return / mean-reversion sanity: is the state itself the move? ---
print("\n=== trailing 21d return on trigger days (the vol-carry lesson) ===")
r21 = _valid_pct_change(px["DX-Y.NYB"], 21)
print(f"  median trailing 21d on trigger days: {100*r21[mask.values].median():+.2f}%  "
      f"(all days {100*r21.median():+.2f}%)")
