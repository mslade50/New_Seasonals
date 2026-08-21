"""C3/C4/C5 round 2 — the three controls the round-1 grid does NOT survive on
its own: trading-day-of-month matching, month-of-year ranking, and the offset
placebo ladder.

Same entry convention as b1: anchor = the session BEFORE opex, lag=1, so entry
is MOC on the OPEX CLOSE (the live form for 2026-08-21).

Why each control:
  tdom  — the third Friday sits at a near-fixed trading day of the month, and
          this repo has repeatedly found mid-month position swallowing an
          "event" whole (TLT h=3 swings -0.202% at tdom 2 to +0.215% at tdom 14).
  month — an August cell owes a month-of-year rank, not an all-days control.
  ladder— slide the anchor -8..+8 sessions. Mid-pack = calendar, not event.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

VEH = ["GLD", "SLV", "TLT", "IEF", "HYG", "LQD", "USO", "XLE", "UUP", "FXI"]
px = close_panel(VEH)
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)

opex = pd.DatetimeIndex([d for d in load_events(["opex"])["date"] if d in pos.index])

# trading day of month for every session in the calendar
tdom = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).cumcount() + 1
tdom = pd.Series(tdom.values, index=idx)
opex_tdom = tdom.loc[opex]
print("opex trading-day-of-month distribution:")
print(opex_tdom.value_counts().sort_index().to_string())
TDOM_SET = sorted(opex_tdom.value_counts()[lambda s: s >= 10].index)
print(f"tdom band used for matching: {TDOM_SET}")

aA = pd.DatetimeIndex([idx[pos[d] - 1] for d in opex if pos[d] >= 1])
aA_tdom = tdom.loc[aA]
ATD = sorted(aA_tdom.value_counts()[lambda s: s >= 10].index)
print(f"ANCHOR (opex-1) tdom band: {ATD}")


def ret_of(v, h, lag=1):
    return fwd_lag(px[v].dropna(), h, lag)


def stats(vals):
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return dict(n=0, mean=np.nan, med=np.nan, hit=np.nan)
    return dict(n=len(v), mean=100 * v.mean(), med=100 * float(np.median(v)),
                hit=100 * (v > 0).mean())


print("\n\n===== 1. tdom-MATCHED control (all months) =====")
print("COND = anchor(opex-1); CTRL = every OTHER session at the same tdom band")
rows = []
for v in VEH:
    r = ret_of(v, 10).dropna()
    a = pd.DatetimeIndex(aA).intersection(r.index)
    span = (a[0], a[-1])
    same = r.index[(tdom.reindex(r.index).isin(ATD)) &
                   (r.index >= span[0]) & (r.index <= span[1])]
    ctrl = same.difference(a)
    for h in (3, 5, 10):
        rr = ret_of(v, h)
        c = stats(rr.loc[a].values)
        k = stats(rr.loc[ctrl].values)
        rows.append({"v": v, "h": h, "cond_n": c["n"], "cond": c["mean"],
                     "tdom_ctrl_n": k["n"], "tdom_ctrl": k["mean"],
                     "excess_vs_tdom": c["mean"] - k["mean"]})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n\n===== 2. AUGUST cell vs AUGUST tdom-matched control =====")
rows = []
for v in VEH:
    r = ret_of(v, 10).dropna()
    a_all = pd.DatetimeIndex(aA).intersection(r.index)
    aug = pd.DatetimeIndex([d for d in a_all if idx[pos[d] + 1].month == 8])
    if len(aug) == 0:
        continue
    span = (aug[0], aug[-1])
    augsess = r.index[(r.index.month == 8) & (tdom.reindex(r.index).isin(ATD)) &
                      (r.index >= span[0]) & (r.index <= span[1])]
    ctrl = augsess.difference(aug)
    for h in (3, 5, 10):
        rr = ret_of(v, h)
        c = stats(rr.loc[aug].values)
        k = stats(rr.loc[ctrl].values)
        rows.append({"v": v, "h": h, "aug_n": c["n"], "aug_cond": c["mean"],
                     "augtdom_n": k["n"], "aug_tdom_ctrl": k["mean"],
                     "excess": c["mean"] - k["mean"]})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n\n===== 3. month-of-year rank of the opex anchor cell (h=10) =====")
for v in VEH:
    rr = ret_of(v, 10)
    a = pd.DatetimeIndex(aA).intersection(rr.dropna().index)
    by = {}
    for d in a:
        m = idx[pos[d] + 1].month
        by.setdefault(m, []).append(rr.loc[d])
    s = pd.Series({m: 100 * np.mean(x) for m, x in by.items()}).sort_values(ascending=False)
    rank = list(s.index).index(8) + 1
    print(f"{v:4s} August {s.get(8):+.3f}% ranks {rank} of {len(s)} months  "
          f"| best {s.index[0]} {s.iloc[0]:+.3f}%  worst {s.index[-1]} {s.iloc[-1]:+.3f}%")

print("\n\n===== 4. OFFSET PLACEBO LADDER, off=-8..+8 around the opex close =====")
print("off=0 is the TRUE anchor (entry MOC on the opex close).")
for v, hs in [("SLV", (5, 10)), ("FXI", (5, 10)), ("XLE", (5, 10)),
              ("USO", (5, 10)), ("HYG", (5, 10)), ("GLD", (10,)),
              ("TLT", (10,)), ("LQD", (10,))]:
    for h in hs:
        rr = ret_of(v, h)
        lad = {}
        for off in range(-8, 9):
            anc = pd.DatetimeIndex(
                [idx[pos[d] - 1 + off] for d in opex
                 if 0 <= pos[d] - 1 + off < len(idx)])
            anc = anc.intersection(rr.dropna().index)
            lad[off] = 100 * rr.loc[anc].mean()
        s = pd.Series(lad)
        rank = int((s > s[0]).sum()) + 1
        print(f"{v} h={h}: TRUE off=0 {s[0]:+.3f}% ranks {rank} of {len(s)}   "
              f"best off={s.idxmax()} {s.max():+.3f}%  "
              f"| ladder {' '.join(f'{o:+d}:{val:+.2f}' for o, val in s.items())}")

print("\n\n===== 4b. AUGUST-only ladder for the August cells =====")
for v, h in [("SLV", 10), ("FXI", 10), ("XLE", 10), ("GLD", 10), ("HYG", 10),
             ("USO", 10), ("SLV", 5), ("XLE", 5)]:
    rr = ret_of(v, h)
    aug_opex = [d for d in opex if d.month == 8]
    lad = {}
    for off in range(-8, 9):
        anc = pd.DatetimeIndex([idx[pos[d] - 1 + off] for d in aug_opex
                                if 0 <= pos[d] - 1 + off < len(idx)])
        anc = anc.intersection(rr.dropna().index)
        lad[off] = 100 * rr.loc[anc].mean()
    s = pd.Series(lad)
    rank = int((s > s[0]).sum()) + 1
    print(f"{v} Aug h={h}: TRUE {s[0]:+.3f}% ranks {rank} of {len(s)}  "
          f"best off={s.idxmax()} {s.max():+.3f}%  "
          f"| {' '.join(f'{o:+d}:{val:+.2f}' for o, val in s.items())}")

print("\n\n===== 5. concentration of the August cells (drop best years) =====")
for v, h in [("SLV", 10), ("FXI", 10), ("XLE", 10), ("GLD", 10), ("HYG", 10),
             ("USO", 10), ("TLT", 10), ("LQD", 10)]:
    rr = ret_of(v, h)
    a = pd.DatetimeIndex(aA).intersection(rr.dropna().index)
    aug = pd.DatetimeIndex([d for d in a if idx[pos[d] + 1].month == 8])
    vals = rr.loc[aug].values
    if len(vals) < 5:
        continue
    o = np.argsort(-vals)
    d1 = np.delete(vals, o[:1]).mean() * 100
    d2 = np.delete(vals, o[:2]).mean() * 100
    d3 = np.delete(vals, o[:3]).mean() * 100
    mid = np.array([d.year % 4 == 2 for d in aug])
    print(f"{v} Aug h={h}: N={len(vals)} mean {100*vals.mean():+.3f}% "
          f"| drop1 {d1:+.3f} drop2 {d2:+.3f} drop3 {d3:+.3f} "
          f"| midterm(N={int(mid.sum())}) {100*vals[mid].mean():+.3f}% "
          f"vs non {100*vals[~mid].mean():+.3f}% "
          f"| best yrs {[ (aug[i].year, round(100*vals[i],1)) for i in o[:3]]}")
