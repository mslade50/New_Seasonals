"""C3 round 1 -- a 10 td hold entered today that contains BOTH the Sep 10 PPI
and the Sep 11 CPI.

The object is the JOINT containment, not either print. Two forms are tested
because the candidate is ambiguous between them:

  (A) LOOSE gate: both a PPI and a CPI land inside the (entry, exit] window.
  (B) EXACT configuration, which is what today actually is: PPI at signal+10
      and CPI at signal+11 (i.e. entry+9 / entry+10), back-to-back, with the
      CPI on the exit session itself.

Kills it has to clear:
  1. beat SPY's own unconditional 10 td drift, not zero
  2. GATE ATTRIBUTION -- the registry killed the macro-vacuum gate for
     agreeing with a trivial alternative on 278 of 318 anchors. What is this
     gate's BASE RATE, and does BOTH differ from CPI-alone?
  3. the gap-share test: an 08:30 release is fully inside the prior-close-to-
     open gap. If the hold's return does not accrue there, no release
     mechanism is operating.
  4. era + midterm splits
  5. offset placebo ladder on form (B)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TK = ["SPY", "TLT", "IEF", "GLD"]
pxd = load_prices(TK)
px = pd.DataFrame({t: pxd[t]["Close"] for t in TK}).dropna()
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)
print(f"common calendar {idx[0].date()} .. {idx[-1].date()}  N={len(idx)}")

H, LAG = 10, 1

# ---------------------------------------------------------------- the gates
in_ppi = event_in_window(idx, idx, H, LAG, ("ppi",))
in_cpi = event_in_window(idx, idx, H, LAG, ("cpi",))
both = in_ppi & in_cpi
neither = (~in_ppi) & (~in_cpi)
one_only = in_ppi ^ in_cpi

ret10 = vehicle_ret(px, [("SPY", 1.0)], H, LAG)
valid = ret10.notna().values

print("\n" + "=" * 78)
print("2. GATE ATTRIBUTION FIRST -- does this gate filter anything?")
print("=" * 78)
v = valid
print(f"  usable days N={v.sum()}")
print(f"  BOTH ppi+cpi in the 10td hold : {100*both[v].mean():.1f}% of all days "
      f"(N={int(both[v].sum())})")
print(f"  CPI in the hold               : {100*in_cpi[v].mean():.1f}% "
      f"(N={int(in_cpi[v].sum())})")
print(f"  PPI in the hold               : {100*in_ppi[v].mean():.1f}% "
      f"(N={int(in_ppi[v].sum())})")
print(f"  NEITHER                       : {100*neither[v].mean():.1f}% "
      f"(N={int(neither[v].sum())})")
agree = (both[v] == in_cpi[v]).mean()
print(f"  BOTH agrees with 'CPI in the hold' on {100*agree:.1f}% of days "
      f"({int((both[v]==in_cpi[v]).sum())} of {int(v.sum())}) "
      f"-> the PPI leg adds {int((in_cpi[v]&~both[v]).sum())} disagreeing days")

print("\n" + "=" * 78)
print("1. FORM (A) loose containment, SPY, h=10, entry lag=1")
print("=" * 78)
rows = []
for lbl, m in (("BOTH ppi+cpi", both), ("CPI only", in_cpi & ~in_ppi),
               ("PPI only", in_ppi & ~in_cpi), ("exactly one", one_only),
               ("NEITHER", neither), ("ALL DAYS (drift)", np.ones(len(idx), bool))):
    vals = ret10.values[m & v]
    r = summarize(vals, lbl)
    rows.append(r)
show(rows, "A. day-level (overlapping, for shape only)")
base = ret10.values[v].mean()
print(f"  BOTH minus all-days drift = "
      f"{100*(ret10.values[both & v].mean() - base):+.4f}pp")

# declustered episodes (min gap = H so windows do not overlap)
print("\n  DECLUSTERED (min gap 10 td):")
rows = []
for lbl, m in (("BOTH", both), ("NEITHER", neither), ("exactly one", one_only)):
    d = declusters(idx[m & v], H, idx[v])
    vals = ret10.loc[d].values
    r = summarize(vals, f"{lbl} episodes")
    w = int((vals > 0).sum())
    r["sign_p"] = round(sign_test(w, len(vals)), 4)
    rows.append(r)
d_all = declusters(idx[v], H, idx[v])
rows.append(summarize(ret10.loc[d_all].values, "ALL episodes (drift)"))
show(rows)

# other vehicles
print("\n  other vehicles, BOTH vs NEITHER, day-level:")
for t in ("TLT", "IEF", "GLD"):
    r = vehicle_ret(px, [(t, 1.0)], H, LAG)
    vv = r.notna().values
    b = r.values[both & vv]
    n_ = r.values[neither & vv]
    a = r.values[vv]
    print(f"   {t}: BOTH {100*b.mean():+.3f}% (N={len(b)})  NEITHER "
          f"{100*n_.mean():+.3f}% (N={len(n_)})  ALL {100*a.mean():+.3f}%  "
          f"BOTH-ALL {100*(b.mean()-a.mean()):+.3f}pp")

# ---------------------------------------------- FORM B: exact configuration
print("\n" + "=" * 78)
print("FORM (B) EXACT configuration -- PPI at signal+10, CPI at signal+11")
print("(= entry+9 / entry+10, back to back, CPI on the exit session).")
print("This is literally today.")
print("=" * 78)
ppi = load_events(["ppi"])["date"]
cpi = set(load_events(["cpi"])["date"])
exact = []
for d in ppi:
    p = pos.get(d)
    if p is None or p - 10 < 0 or p + 1 >= len(idx):
        continue
    if idx[p + 1] in cpi:                      # CPI the very next session
        exact.append(idx[p - 10])              # the signal date
exact = pd.DatetimeIndex(sorted(set(exact)))
exact = exact[ret10.reindex(exact).notna().values]
print(f"  N exact-configuration signal days = {len(exact)}  "
      f"{exact[0].date()} .. {exact[-1].date()}")

mask_b = pd.Series(False, index=idx)
mask_b.loc[exact] = True
battery(px, mask_b, [("SPY", 1.0)], h=H,
        title="C3-B: SPY 10td hold, PPI at +9 / CPI at +10 from entry",
        cost_bps=3.0, min_gap=10, event_kinds=("fomc_decision",))

# ---------------------------------------------------- 3. the gap-share test
print("\n" + "=" * 78)
print("3. GAP-SHARE TEST. An 08:30 release is fully contained in the")
print("   prior-close-to-open gap of the print session. If a containment")
print("   cell earns nothing there, no release mechanism is operating.")
print("=" * 78)
sp = pxd["SPY"].dropna(subset=["Open", "Close"])
sidx = sp.index
spos = pd.Series(range(len(sidx)), index=sidx)
tot_l, gap_l, rest_l = [], [], []
for d in exact:
    p = spos.get(d)
    if p is None or p + 1 + H >= len(sidx):
        continue
    e = p + 1                                   # entry close
    c = sp["Close"].values
    o = sp["Open"].values
    tot = c[e + H] / c[e] - 1.0
    # the two release sessions inside the hold: entry+9 (PPI), entry+10 (CPI)
    g = 0.0
    for k in (9, 10):
        g += o[e + k] / c[e + k - 1] - 1.0
    tot_l.append(tot)
    gap_l.append(g)
    rest_l.append(tot - g)
tot_a, gap_a, rest_a = map(np.array, (tot_l, gap_l, rest_l))
print(f"  N={len(tot_a)}  hold total {100*tot_a.mean():+.3f}%  "
      f"= 2 release gaps {100*gap_a.mean():+.3f}%  + the other 8 sessions "
      f"{100*rest_a.mean():+.3f}%")
if tot_a.mean() != 0:
    print(f"  release-gap share of the hold's return: "
          f"{100*gap_a.mean()/tot_a.mean():.0f}%")
# baseline: the same two gaps on all days
allgap = (sp["Open"] / sp["Close"].shift(1) - 1.0).dropna()
print(f"  baseline: mean SPY overnight gap on ALL days "
      f"{100*allgap.mean():+.4f}% -> 2 gaps = {200*allgap.mean():+.4f}%")

# --------------------------------------------------- 4. eras / midterm
print("\n" + "=" * 78)
print("4. era + midterm splits, FORM (B)")
print("=" * 78)
vals = ret10.loc[exact].values
yrs = exact.year
rows = []
for lbl, m in (("pre-2013", yrs < 2013), ("2013+", yrs >= 2013),
               ("pre-2018", yrs < 2018), ("2018+", yrs >= 2018),
               ("MIDTERM", (yrs % 4) == 2), ("non-midterm", (yrs % 4) != 2),
               ("SEPTEMBER only", exact.month == 9)):
    sub = vals[m]
    if not len(sub):
        continue
    r = summarize(sub, lbl)
    r["excess_pp"] = round(r["mean_pct"] - 100 * base, 3)
    r["sign_p"] = round(sign_test(int((sub > 0).sum()), len(sub)), 4)
    rows.append(r)
show(rows, "FORM B splits (excess vs SPY's all-day 10td drift)")

# ------------------------------------------------ 5. offset placebo ladder
print("\n" + "=" * 78)
print("5. OFFSET PLACEBO LADDER, FORM (B). Signal = PPI-10+k, k in -12..+12.")
print("   A plateau kills; only a spike at k=0 survives.")
print("=" * 78)
pairs = []
for d in ppi:
    p = pos.get(d)
    if p is None or p + 1 >= len(idx):
        continue
    if idx[p + 1] in cpi:
        pairs.append(int(p))
rows = []
for k in range(-12, 13):
    sp_ = [p - 10 + k for p in pairs if 0 <= p - 10 + k < len(idx)]
    dts = idx[sp_]
    vv = ret10.loc[dts].dropna().values
    if not len(vv):
        continue
    rows.append({"k": k, "n": len(vv), "mean_pct": round(100 * vv.mean(), 3),
                 "excess_pp": round(100 * (vv.mean() - base), 3),
                 "hit": round(100 * (vv > 0).mean(), 1)})
d = pd.DataFrame(rows).sort_values("mean_pct", ascending=False).reset_index(drop=True)
rank = int(d.index[d["k"] == 0][0]) + 1
print(f"  TRUE ANCHOR k=0 RANKS {rank} of {len(d)}")
print(d.to_string(index=False))
