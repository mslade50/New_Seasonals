"""C6 round 2 — gate attribution, breadth monotonicity, definition neighbours.

Round 1 (c1) showed the index-distance half of the trigger carries everything.
This script prices that properly (EPISODE vs EPISODE, not episode vs day-level),
asks whether MORE breadth is BETTER (the stated mechanism), and walks the
definition in three independent directions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BAR = pd.Timestamp("2026-08-21")

tape_json = json.load(open(ROOT / "data" / "pitch_tape.json"))["tickers"]
TAPE = sorted(tape_json)
SECT9 = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
SECT11 = SECT9 + ["XLRE", "XLC"]
VEH = ["SPY", "QQQ", "IWM", "DIA"]

px_all = load_prices(sorted(set(TAPE + SECT11 + VEH)))
spy = px_all["SPY"]["Close"].dropna()
CAL = spy.index[spy.index <= BAR]
px = pd.DataFrame({t: px_all[t]["Close"] for t in VEH}).reindex(CAL)


def dist_52wh(c, look=252):
    return c / c.rolling(look).max() - 1.0


def breadth(univ, tol):
    num = pd.Series(0.0, index=CAL)
    den = pd.Series(0.0, index=CAL)
    for t in univ:
        d = px_all.get(t)
        if d is None:
            continue
        c = d["Close"].dropna()
        c = c[c.index <= BAR]
        if len(c) < 300:
            continue
        dd = dist_52wh(c)
        flag = (dd >= -tol).astype(float)
        flag[dd.isna()] = np.nan
        f = flag.reindex(CAL)
        ok = f.notna()
        num[ok] += f[ok].values
        den[ok] += 1.0
    return (num / den).where(den >= 5)


def pit_pct(s, look=252):
    return rolling_on_valid(s, lambda x: x.rolling(look).rank(pct=True) * 100.0)


spy_d = dist_52wh(spy).reindex(CAL)
idx_gate = (spy_d > -0.05) & (spy_d <= -0.005)

b_tape = breadth(TAPE, 0.0025)
b_s9 = breadth(SECT9, 0.0025)
pit_tape, pit_s9 = pit_pct(b_tape), pit_pct(b_s9)


def epi_summary(mask, h, label, min_gap=10, veh="SPY"):
    ret = fwd_lag(px[veh], h, 1)
    valid = ret.dropna().index
    t = CAL[mask.reindex(CAL, fill_value=False).values].intersection(valid)
    if len(t) == 0:
        return {"label": label, "n": 0}
    epi = declusters(t, min_gap, valid)
    r = summarize(ret.loc[epi].values, label)
    r["n_days"] = len(t)
    return r


print("=" * 100)
print("A. GATE ATTRIBUTION, EPISODE vs EPISODE (min_gap = h, so like-for-like)")
print("=" * 100)
for h in (1, 3, 5, 10):
    base = epi_summary(idx_gate, h, "index off-high ALONE", min_gap=h)
    tp = epi_summary(idx_gate & (pit_tape >= 80), h, "+ tape218 breadth>=80", min_gap=h)
    s9 = epi_summary(idx_gate & (pit_s9 >= 80), h, "+ sect9 breadth>=80", min_gap=h)
    rows = [base, tp, s9]
    for r in (tp, s9):
        r["gate_worth_pp"] = round(r["mean_pct"] - base["mean_pct"], 3)
    show(rows, f"h={h} td, long SPY")

print("\n" + "=" * 100)
print("B. MONOTONICITY: is MORE breadth BETTER? (the stated mechanism)")
print("=" * 100)
h = 5
ret5 = fwd_lag(px["SPY"], h, 1)
for lbl, pit in (("tape218", pit_tape), ("sect9", pit_s9)):
    rows = []
    for lo, hi in ((0, 20), (20, 40), (40, 60), (60, 80), (80, 95), (95, 101)):
        m = idx_gate & (pit >= lo) & (pit < hi)
        rows.append(epi_summary(m, h, f"{lbl} PIT breadth [{lo},{hi})", min_gap=h))
    show(rows, f"{lbl}: SPY h=5 by PIT-breadth bucket, WITHIN the index gate")

print("\n  raw-count buckets on the survivorship-free sect9 (n of 9 at a high):")
rows = []
for k in range(0, 6):
    m = idx_gate & (b_s9 >= (k / 9) - 1e-9) & (b_s9 < ((k + 1) / 9) - 1e-9)
    rows.append(epi_summary(m, h, f"exactly {k} of 9 at a 52w high", min_gap=h))
rows.append(epi_summary(idx_gate & (b_s9 >= 5 / 9 - 1e-9), h, ">=5 of 9", min_gap=h))
show(rows, "sect9 raw count buckets, SPY h=5")

print("\n" + "=" * 100)
print("C. DEFINITION NEIGHBOURS in three directions")
print("=" * 100)
print("\n  C1. the 'at a high' tolerance")
rows = []
for tol in (0.0025, 0.005, 0.01, 0.02, 0.05):
    bt = breadth(TAPE, tol)
    bs = breadth(SECT9, tol)
    rows.append(epi_summary(idx_gate & (pit_pct(bt) >= 80), h,
                            f"tape tol={100*tol:.2f}%", min_gap=h))
    rows.append(epi_summary(idx_gate & (pit_pct(bs) >= 80), h,
                            f"sect9 tol={100*tol:.2f}%", min_gap=h))
show(rows, "tolerance walk, SPY h=5")

print("\n  C2. the index-distance rung")
rows = []
for lo, hi in ((-0.02, -0.005), (-0.03, -0.005), (-0.05, -0.005),
               (-0.10, -0.005), (-0.05, -0.01), (-0.05, -0.02), (-1.0, -0.005)):
    g = (spy_d > lo) & (spy_d <= hi)
    rows.append(epi_summary(g & (pit_s9 >= 80), h,
                            f"sect9 & idx ({100*lo:.0f}%,{100*hi:.1f}%]", min_gap=h))
show(rows, "index rung walk, SPY h=5")

print("\n  C3. the universe")
rows = []
for lbl, u in (("tape218", TAPE), ("sect9", SECT9), ("sect11", SECT11)):
    bb = breadth(u, 0.0025)
    rows.append(epi_summary(idx_gate & (pit_pct(bb) >= 80), h, lbl, min_gap=h))
# a survivorship-free non-sector alternative: the 4 broad index ETFs
rows.append(epi_summary(idx_gate & (pit_pct(breadth(["SPY", "QQQ", "IWM", "DIA"], 0.0025)) >= 80),
                        h, "4 index ETFs", min_gap=h))
show(rows, "universe swap, SPY h=5")

print("\n" + "=" * 100)
print("D. ERA / REGIME SPLITS on the survivorship-free cell (the better of the two)")
print("=" * 100)
mask_s9 = idx_gate & (pit_s9 >= 80)
t = CAL[mask_s9.reindex(CAL, fill_value=False).values].intersection(ret5.dropna().index)
epi = declusters(t, 5, ret5.dropna().index)
v = ret5.loc[epi].values
show(era_split(epi, v), "era")
mid = np.array([d.year % 4 == 2 for d in epi])
show([summarize(v[mid], f"MIDTERM years (N={int(mid.sum())})"),
      summarize(v[~mid], f"non-midterm (N={int((~mid).sum())})")], "cycle split")
sma200 = rolling_on_valid(spy, lambda x: x.rolling(200).mean()).reindex(CAL)
ab = (spy > sma200).reindex(epi).values
show([summarize(v[ab], f"above 200d (N={int(ab.sum())})"),
      summarize(v[~ab], f"below 200d (N={int((~ab).sum())})")], "trend split")
print(f"  concentration: {cluster_note(epi, v)}")
order = np.argsort(-np.abs(v))
keep = np.ones(len(v), bool)
keep[order[:2]] = False
show([summarize(v, "all episodes"), summarize(v[keep], "drop-top-2 by |R|")],
     "drop-two")

print("\n" + "=" * 100)
print("E. AUG-SEASONAL / TDOM CONTROL: trigger days vs same-calendar-position days")
print("=" * 100)
tdom = pd.Series(index=CAL, dtype=float)
for (y, m), grp in pd.Series(CAL, index=CAL).groupby([CAL.year, CAL.month]):
    tdom.loc[grp.index] = np.arange(1, len(grp) + 1)
n_in_month = pd.Series(index=CAL, dtype=float)
for (y, m), grp in pd.Series(CAL, index=CAL).groupby([CAL.year, CAL.month]):
    n_in_month.loc[grp.index] = len(grp)
me_off = (n_in_month - tdom).astype(int)      # sessions until the month-end close
trig_me = me_off.reindex(epi)
print(f"  trigger episodes' median ME offset = {trig_me.median():.0f}; today is ME-5")
m5 = (me_off >= 3) & (me_off <= 7)
show([summarize(ret5[m5.values & ret5.notna().values].values,
                f"ALL days at ME-3..ME-7 (N={int((m5 & ret5.notna()).sum())})"),
      summarize(v, f"trigger episodes (N={len(v)})")],
     "calendar-position control")
