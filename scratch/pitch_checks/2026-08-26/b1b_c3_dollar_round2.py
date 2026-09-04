"""b1b / C3 round 2. The only thing in C3 that looked alive was UUP at
+0.275% / 4.6x cost while DX paid +0.029% / 2.0x. Two possible readings:
(a) a real vehicle difference, or (b) a sample-span artefact, since DX runs
from 2002 and UUP only from 2007.

Also: concentration, drop-best, the midterm sign test scored against the
instrument's OWN up-rate (not a coin), and whether the cell is just the
already-registered "long DX after a weak NFP close" entry wearing a rank.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change
import pandas as pd, numpy as np

pd.set_option("display.width", 220)

px = close_panel(["DX-Y.NYB", "UUP"]).dropna(subset=["DX-Y.NYB"])
rk = pct_rank(px["DX-Y.NYB"], 21, 252)
mask = rk <= 2.0

# ---- (b) MATCHED SAMPLE: same episodes, both vehicles ----
both = px.dropna(subset=["UUP"])
r_dx = vehicle_ret(both, [("DX-Y.NYB", 1.0)], 5, 1)
r_uup = vehicle_ret(both, [("UUP", 1.0)], 5, 1)
valid = pd.DataFrame({"d": r_dx, "u": r_uup}).dropna().index
tt = both.index[mask.reindex(both.index, fill_value=False).values].intersection(valid)
epi = declusters(tt, 5, valid)
print("=== MATCHED-SAMPLE vehicle comparison (identical episodes, 2007+) ===")
show([summarize(r_dx.loc[epi].values, f"DX  (N={len(epi)})"),
      summarize(r_uup.loc[epi].values, f"UUP (N={len(epi)})"),
      summarize((r_uup - r_dx).loc[epi].values, "UUP minus DX, per episode"),
      summarize((r_uup - r_dx).loc[valid].values, "UUP minus DX, ALL DAYS")])
d = (r_uup - r_dx).loc[epi]
print(f"  sign agreement DX/UUP on episodes: "
      f"{100*float((np.sign(r_dx.loc[epi]) == np.sign(r_uup.loc[epi])).mean()):.1f}%")
print(f"  structural per-5td gap (all days): {100*float((r_uup-r_dx).loc[valid].mean()):+.3f}%")
print("  -> the 2002-2006 DX-only era is what makes the two headline numbers differ,")
print("     not the vehicle. Registry's standing 'UUP is dead' entry stands on drag alone.")

# ---- DX on the SAME 2007+ window, for the like-for-like cost line ----
print("\n=== DX on the matched 2007+ window vs its own full sample ===")
r_dx_full = vehicle_ret(px, [("DX-Y.NYB", 1.0)], 5, 1)
vfull = r_dx_full.dropna().index
tfull = px.index[mask.values].intersection(vfull)
efull = declusters(tfull, 5, vfull)
show([summarize(r_dx_full.loc[efull].values, "DX 2002+ (N=%d)" % len(efull)),
      summarize(r_dx.loc[epi].values, "DX 2007+ matched (N=%d)" % len(epi))])

# ---- concentration / drop-best on the UUP cell ----
print("\n=== UUP cell concentration ===")
u = r_uup.loc[epi].values
print(" ", cluster_note(epi, u, k=2))
print(" ", cluster_note(epi, u, k=3))
order = np.argsort(-u)
for k in (1, 2, 3):
    keep = np.ones(len(u), bool); keep[order[:k]] = False
    print(f"  drop-best-{k}: {100*u[keep].mean():+.3f}%  (N={keep.sum()})  "
          f"vs headline {100*u.mean():+.3f}%")
by_yr = pd.Series(u).groupby(pd.DatetimeIndex(epi).year.values).sum().sort_values(ascending=False)
print(f"  best year {by_yr.index[0]} = {100*by_yr.iloc[0]:+.2f}pp of {100*u.sum():+.2f}pp total "
      f"({100*by_yr.iloc[0]/u.sum():.0f}%)")

# ---- sign test against the instrument's OWN up-rate ----
print("\n=== sign tests against each instrument's OWN unconditional up-rate ===")
for lbl, rr, ee, vv in [("DX 2002+", r_dx_full, efull, vfull), ("UUP", r_uup, epi, valid)]:
    base = float((rr.loc[vv] > 0).mean())
    w = int((rr.loc[ee] > 0).sum()); n = len(ee)
    print(f"  {lbl}: {w}-{n-w} ({100*w/n:.1f}%) against base rate {100*base:.1f}% "
          f"-> sign p = {sign_test(w, n, p=base):.4f}")

# ---- MIDTERM: scored properly, and today is midterm ----
print("\n=== MIDTERM cell, both vehicles (TODAY IS MIDTERM) ===")
for lbl, rr, ee, vv in [("DX 2002+", r_dx_full, efull, vfull), ("UUP", r_uup, epi, valid)]:
    yrs = pd.DatetimeIndex(ee).year
    mid = (yrs % 4 == 2)
    base = float((rr.loc[vv] > 0).mean())
    v = rr.loc[ee[mid]].values
    w = int((v > 0).sum()); n = len(v)
    print(f"  {lbl} MIDTERM: N={n} mean {100*np.mean(v):+.3f}% hit {100*w/n:.1f}% "
          f"(base {100*base:.1f}%)  P(<= this many wins) = "
          f"{1-sign_test(w+1, n, p=base):.4f}   bootstrap P(mean<=0) = {bootstrap_p_le0(v):.3f}")
    print(f"    midterm years present: {sorted(set(pd.DatetimeIndex(ee[mid]).year))}")
    nm = rr.loc[ee[~mid]].values
    print(f"  {lbl} non-midterm: N={len(nm)} mean {100*np.mean(nm):+.3f}%  "
          f"-> midterm-minus-nonmidterm {100*(np.mean(v)-np.mean(nm)):+.3f}pp")

# ---- is the cell just the registered NFP-close entry? ----
print("\n=== overlap with the registered 'weak DX close into NFP' family ===")
nfp = load_events(["nfp"])["date"]
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)
nfp_pos = set(int(pos[d]) for d in nfp if d in pos.index)
trig_pos = [int(pos[d]) for d in efull]
within = [p for p in trig_pos if any(abs(p - q) <= 3 for q in nfp_pos)]
print(f"  DX episodes within 3 td of an NFP print: {len(within)} of {len(efull)} "
      f"({100*len(within)/len(efull):.0f}%)")
r = r_dx_full.loc[efull].values
near = np.array([any(abs(p - q) <= 3 for q in nfp_pos) for p in trig_pos])
show([summarize(r[near], "episodes NEAR an NFP (<=3td)"),
      summarize(r[~near], "episodes AWAY from NFP")])
