"""C8 - FXI 5d break inside its own intact 21d thrust, vs EEM.

Cell: FXI rank5 <= 20 & FXI rank21 >= 80 & EEM 5d return > 0.
Registry collision to clear (EWZ decoupler, killed 08-10 and again 08-12):
 (a) which 5d-drop DEPTH bucket does today's -2.36% sit in, and does the sign
     reverse at the deep readings that make the idea interesting;
 (b) bear-tape selection rate of the gate vs the base rate (over-selection);
 (c) year concentration / top-2 episodes;
 (d) beta-neutral residual, not the equal-dollar spread.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TK = ["FXI", "EEM", "SPY"]
px = close_panel(TK).dropna()
raw = load_prices(TK)
fxi, eem, spy = raw["FXI"]["Close"], raw["EEM"]["Close"], raw["SPY"]["Close"]

f_r5, f_r21 = pct_rank(fxi, 5), pct_rank(fxi, 21)


def mask(r5=20, r21=80, eem_pos=True):
    m = (f_r5 <= r5) & (f_r21 >= r21)
    if eem_pos:
        m &= eem.pct_change(5) > 0
    return m.reindex(px.index).fillna(False)


print("=== occurrence counts, definition neighbourhood ===")
rows = []
for r5 in (10, 20, 30, 40):
    for r21 in (70, 80, 90):
        for ep in (True, False):
            m = mask(r5, r21, ep)
            d = px.index[m.values]
            rows.append({"fxi_r5<=": r5, "fxi_r21>=": r21, "eem5>0": ep,
                         "n_days": len(d), "n_epi": len(declusters(d, 5, px.index)),
                         "first": str(d[0].date()) if len(d) else "-",
                         "last": str(d[-1].date()) if len(d) else "-"})
print(pd.DataFrame(rows).to_string(index=False))

# beta: FXI on EEM, daily returns, full history
al = pd.concat([fxi.pct_change(), eem.pct_change(), spy.pct_change()],
               axis=1).dropna()
al.columns = ["fxi", "eem", "spy"]
b_eem = np.polyfit(al["eem"], al["fxi"], 1)[0]
b_spy = np.polyfit(al["spy"], al["fxi"], 1)[0]
print(f"\nbeta FXI~EEM = {b_eem:.3f} (corr {al['fxi'].corr(al['eem']):.3f})   "
      f"beta FXI~SPY = {b_spy:.3f} (corr {al['fxi'].corr(al['spy']):.3f})")

# --- the as-specified cell ---
m_spec = mask()
trig = px.index[m_spec.values]
print(f"\n### AS SPECIFIED: n_days={len(trig)} dates={[str(d.date()) for d in trig]}")

# --- registry test (b): bear-tape selection rate ---
print("\n=== (b) over-selection: SPY below its own 200d ===")
below200 = (spy < spy.rolling(200).mean()).reindex(px.index).fillna(False)
base = 100 * below200[px.index >= fxi.index[0]].mean()
for lbl, m in [("as-specified", m_spec), ("loose r5<=30,r21>=70", mask(30, 70, True)),
               ("no-EEM-gate r5<=20,r21>=80", mask(20, 80, False))]:
    d = px.index[m.values]
    if len(d):
        print(f"  {lbl}: SPY<200d on {100*below200.loc[d].mean():.1f}% of "
              f"{len(d)} trigger days vs base rate {base:.1f}%")
    # also FXI's own 200d
    f200 = (fxi < fxi.rolling(200).mean()).reindex(px.index).fillna(False)
    if len(d):
        print(f"      FXI<its own 200d on {100*f200.loc[d].mean():.1f}% of trigger days "
              f"vs base {100*f200[px.index >= fxi.index[0]].mean():.1f}%  "
              f"(today: {bool(f200.iloc[-1])})")

# --- registry test (a): depth buckets on the 5d drop, using the widest cell ---
print("\n=== (a) 5d-drop DEPTH buckets (widest measurable cell, r5<=40 & r21>=70 & eem>0) ===")
mw = mask(40, 70, True)
tw = px.index[mw.values]
d5 = fxi.pct_change(5).reindex(px.index)
print(f"  today's FXI 5d = {100*d5.iloc[-1]:+.2f}%")
for h in (3, 5):
    ret_bn = vehicle_ret(px, [("FXI", 1.0), ("EEM", -b_eem)], h)
    ret_out = vehicle_ret(px, [("FXI", 1.0)], h)
    epi = declusters(tw, 5, px.index)
    dv = d5.loc[epi].values
    rows = []
    for lo, hi in [(-99, -3.5), (-3.5, -2.0), (-2.0, -1.0), (-1.0, 99)]:
        sel = (100 * dv > lo) & (100 * dv <= hi)
        if sel.sum():
            rows.append(summarize(ret_bn.loc[epi[sel]].values,
                                  f"h={h} bn 5d in ({lo},{hi}] N={int(sel.sum())}"))
            rows.append(summarize(ret_out.loc[epi[sel]].values,
                                  f"h={h} outright 5d in ({lo},{hi}]"))
    show(rows)

# --- full battery on the widest measurable cell + the no-EEM-gate parent ---
variants = {"r5<=20": mask(20, 70, True), "r5<=30": mask(30, 70, True),
            "r21>=80": mask(40, 80, True), "r21>=90": mask(40, 90, True),
            "no eem gate": mask(40, 70, False)}
for h in (3, 5):
    battery(px, mw, [("FXI", 1.0), ("EEM", -b_eem)], h,
            f"C8 WIDE beta-neutral long FXI / short EEM (beta {b_eem:.2f})", 12.0,
            variants)
    battery(px, mw, [("FXI", 1.0), ("EEM", -1.0)], h,
            "C8 WIDE equal-dollar long FXI / short EEM", 12.0)
    battery(px, mw, [("FXI", 1.0)], h, "C8 WIDE long FXI outright", 8.0, variants)
    battery(px, mw, [("FXI", -1.0), ("EEM", b_eem)], h,
            "C8 WIDE SHORT side (beta-neutral)", 12.0)

print("\n=== parent: FXI r5<=20 & r21>=80, no EEM gate (the measurable population) ===")
mp = mask(20, 80, False)
for h in (3, 5):
    battery(px, mp, [("FXI", 1.0), ("EEM", -b_eem)], h,
            f"C8 PARENT beta-neutral (beta {b_eem:.2f})", 12.0)

print("\n=== midterm split + era, wide cell, h=3 beta-neutral ===")
r = vehicle_ret(px, [("FXI", 1.0), ("EEM", -b_eem)], 3)
epi = declusters(tw, 5, px.index)
mt = np.array([d.year % 4 == 2 for d in epi])
show([summarize(r.loc[epi[mt]].values, f"midterm (N={int(mt.sum())})"),
      summarize(r.loc[epi[~mt]].values, f"non-midterm (N={int((~mt).sum())})")])
print("  by year:", (pd.Series(r.loc[epi].values, index=epi.year)
                     .groupby(level=0).sum().mul(100).round(2).to_dict()))
