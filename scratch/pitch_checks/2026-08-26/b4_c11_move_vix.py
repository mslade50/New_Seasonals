"""b4 / C11: MOVE/VIX at a trailing-252d PIT percentile >= 80 -- bond vol
rich relative to equity vol. Vehicles TLT / IEF (duration) and SPY (cross).

Data span established in b0: ^MOVE runs 2002-11-12 .. 2026-08-25, 5,881
observations, 94.8% business-day coverage, one gap > 7 days. Usable.
NOT a data kill; the candidate gets measured.

Conditioning window: the trailing-252d PIT percentile is the only statistic
knowable in advance. The full-history percentile is lookahead and the
registry has already killed a cell on exactly that (2026-08-18: "the map's
97.9th percentile was a FULL-HISTORY percentile"). Both are printed; only
the PIT one is used as the trigger.

Registry adjacents honoured:
 - the two ^MOVE FLOOR entries and the 2026-08-18 MOVE SPIKE kill -- this is
   a RATIO, a third object, so it is measured rather than assumed dead.
 - "a vol-carry state can be a LAGGING marker of the move that created it":
   the trailing return of the vehicle on trigger days is checked BEFORE any
   premium story is believed.
 - "^MOVE's LEVEL is not its return rank": both are quoted.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change
import pandas as pd, numpy as np

pd.set_option("display.width", 240)

TK = ["^MOVE", "^VIX", "TLT", "IEF", "SPY", "LQD", "^VIX3M"]
raw = load_prices(TK)
px = pd.DataFrame({t: raw[t]["Close"] for t in TK}).dropna(subset=["^MOVE", "^VIX", "TLT", "IEF", "SPY"])
print(f"usable joint span {px.index[0].date()} .. {px.index[-1].date()}, {len(px)} sessions")

ratio = px["^MOVE"] / px["^VIX"]
pit = ratio.rolling(252).rank(pct=True) * 100.0          # PIT, knowable
full = ratio.rank(pct=True) * 100.0                      # LOOKAHEAD, contrast
mv_pit = px["^MOVE"].rolling(252).rank(pct=True) * 100.0
vx_pit = px["^VIX"].rolling(252).rank(pct=True) * 100.0

print(f"\nTODAY  ratio={ratio.iloc[-1]:.4f}  PIT252 pctile={pit.iloc[-1]:.1f}  "
      f"full-history pctile={full.iloc[-1]:.1f}")
print(f"       ^MOVE level={px['^MOVE'].iloc[-1]:.2f} PIT252 pctile={mv_pit.iloc[-1]:.1f}")
print(f"       ^VIX  level={px['^VIX'].iloc[-1]:.2f} PIT252 pctile={vx_pit.iloc[-1]:.1f}")
print("  -> the ratio is high because the DENOMINATOR is low, not because bond vol is high.")

mask = pit >= 80.0
print(f"\ntrigger days (PIT ratio pctile>=80): {int(mask.sum())} of {int(pit.notna().sum())}")

# ---------- WHICH LEG DRIVES THE TRIGGER? ----------
print("\n=== decomposition on trigger days: which leg makes the ratio rich? ===")
rows = []
for lbl, m in [("ALL DAYS", pit.notna()), ("ratio PIT>=80", mask),
               ("ratio PIT>=80 & MOVE PIT>=80 (bond vol really high)", mask & (mv_pit >= 80)),
               ("ratio PIT>=80 & VIX PIT<=25 (equity vol cheap) [TODAY]", mask & (vx_pit <= 25)),
               ("ratio PIT>=80 & MOVE PIT<=50 [TODAY: MOVE pctile %.0f]" % mv_pit.iloc[-1],
                mask & (mv_pit <= 50))]:
    sub = m & pit.notna()
    rows.append({"cell": lbl, "n_days": int(sub.sum()),
                 "median_MOVE_pit": round(float(mv_pit[sub.values].median()), 1),
                 "median_VIX_pit": round(float(vx_pit[sub.values].median()), 1)})
print(pd.DataFrame(rows).to_string(index=False))

# ---------- direction from the data ----------
for veh in ["TLT", "IEF", "SPY"]:
    trig = px.index[mask.values]
    print(f"\n=== horizon scan LONG {veh} on ratio PIT>=80 ===")
    show(horizon_scan(px, trig, [(veh, 1.0)], hs=(1, 2, 3, 5, 10)))

# ---------- battery ----------
variants = {f"PIT pctile>={k}": (pit >= k) for k in (60, 70, 80, 90, 95)}
variants["FULL-history pctile>=80 (lookahead)"] = full >= 80
variants["MOVE PIT level>=80 alone"] = mv_pit >= 80
variants["VIX PIT level<=25 alone"] = vx_pit <= 25
for veh, cost in [("TLT", 6.0), ("IEF", 6.0), ("SPY", 3.0)]:
    battery(px, mask, [(veh, 1.0)], 5, f"C11 LONG {veh} on MOVE/VIX PIT>=80", cost,
            variants=variants, event_kinds=("cpi", "fomc_decision", "nfp"))

# ---------- GATE ATTRIBUTION: does the RATIO add over its two legs? ----------
print("\n" + "=" * 78)
print("GATE ATTRIBUTION -- does the ratio add anything over either leg alone?")
print("=" * 78)
for veh in ["TLT", "IEF"]:
    ret = vehicle_ret(px, [(veh, 1.0)], 5, 1)
    valid = ret.dropna().index
    gates = {
        "ALL DAYS": pd.Series(True, index=px.index),
        "ratio PIT>=80 (pitched)": mask,
        "VIX PIT<=25 alone": vx_pit <= 25,
        "MOVE PIT>=80 alone": mv_pit >= 80,
        "ratio>=80 & VIX PIT<=25 (today's shape)": mask & (vx_pit <= 25),
        "ratio>=80 & VIX PIT>25": mask & (vx_pit > 25),
        "ratio<80 & VIX PIT<=25 (cheap equity vol, ratio NOT rich)": (~mask) & (vx_pit <= 25),
    }
    rows = []
    for lbl, g in gates.items():
        tt = px.index[g.reindex(px.index, fill_value=False).values].intersection(valid)
        epi = declusters(tt, 5, valid)
        d = summarize(ret.loc[epi].values, lbl)
        d["n_days"] = len(tt)
        rows.append(d)
    show(rows, f"LONG {veh} h=5, episodes")

# ---------- lagging-marker check ----------
print("\n=== trailing returns on trigger days (the vol-carry lesson) ===")
trig = px.index[mask.values]
for t in ["TLT", "IEF", "SPY"]:
    tr21 = _valid_pct_change(px[t], 21)
    tr5 = _valid_pct_change(px[t], 5)
    print(f"  {t}: median trailing 21d on triggers {100*float(tr21.loc[trig].median()):+.2f}% "
          f"(all {100*float(tr21.median()):+.2f}%) | trailing 5d {100*float(tr5.loc[trig].median()):+.2f}% "
          f"(all {100*float(tr5.median()):+.2f}%) | today 21d {100*float(tr21.iloc[-1]):+.2f}% "
          f"5d {100*float(tr5.iloc[-1]):+.2f}%")

# ---------- era + midterm + bond-regime split ----------
print("\n=== TLT h=5 episode splits ===")
ret = vehicle_ret(px, [("TLT", 1.0)], 5, 1)
valid = ret.dropna().index
tt = px.index[mask.values].intersection(valid)
epi = declusters(tt, 5, valid)
yrs = pd.DatetimeIndex(epi).year
show([summarize(ret.loc[epi[yrs < 2018]].values, "pre-2018"),
      summarize(ret.loc[epi[yrs >= 2018]].values, "2018+"),
      summarize(ret.loc[epi[yrs < 2021]].values, "pre-2021 (bond bull)"),
      summarize(ret.loc[epi[yrs >= 2021]].values, "2021+ (post bond bull)"),
      summarize(ret.loc[epi[(yrs % 4) == 2]].values, "MIDTERM"),
      summarize(ret.loc[epi[(yrs % 4) != 2]].values, "non-midterm")])
print("  episode dates:", ", ".join(str(d.date()) for d in epi[-25:]))
print(f"  episodes by year: {dict(pd.Series(yrs).value_counts().sort_index())}")
