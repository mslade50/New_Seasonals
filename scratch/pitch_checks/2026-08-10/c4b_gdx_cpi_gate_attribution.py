"""C4 round 2 -- gate attribution + multiplicity for the pre-CPI GDX cell.

Round 1's best cell was CPI-2 anchor x GDX rank5>=80, h=2: +0.962%, N=49,
67.3% hit, sign p=0.011.  Three ways that can be an artefact:

  A. the CPI anchor does nothing and it is just "GDX after a 5d thrust"
     (run the thrust with NO CPI anchor -- the decisive gate-attribution test)
  B. it is the best of a 32-cell k x threshold x horizon grid I built
     (price it against the grid, which IS a search I performed)
  C. it is one or two episodes / one era / one cycle-year bucket
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

LAG = 1
P = close_panel(["GDX", "GLD"]).dropna(subset=["GDX"])
g = P["GDX"]
idx = g.index
rk5 = pct_rank(g, 5)
cpi = load_events(["cpi"])["date"]


def anchors(k: int) -> pd.DatetimeIndex:
    out = []
    for d in cpi:
        p = int(np.searchsorted(idx.values, np.datetime64(d)))
        if p - k < 0 or p >= len(idx):
            continue
        out.append(idx[p - k])
    return pd.DatetimeIndex(sorted(set(out)))


# ---------------------------------------------------------------- A. attribution
print("### A. GATE ATTRIBUTION: is it the CPI anchor, or just the thrust? ###")
for thr in (80.0, 90.0):
    rows = []
    for h in (1, 2, 3, 5):
        fw = fwd_lag(g, h, LAG)
        ok = fw.notna()
        m = (rk5 >= thr).fillna(False)
        # (i) thrust, NO CPI anchor
        t_all = idx[m.values & ok.values]
        e_all = declusters(t_all, h, idx[ok.values])
        r = summarize(fw.loc[e_all].values, f"thrust rk>={thr:.0f} ANY day, h={h}")
        r["n_days"] = len(t_all)
        rows.append(r)
        # (ii) thrust AND on a CPI-2 anchor
        a = anchors(2)
        t_cpi = pd.DatetimeIndex(a).intersection(idx[m.values & ok.values])
        e_cpi = declusters(t_cpi, h, idx[ok.values])
        r = summarize(fw.loc[e_cpi].values, f"thrust rk>={thr:.0f} AND CPI-2, h={h}")
        r["n_days"] = len(t_cpi)
        rows.append(r)
        # (iii) thrust NOT on a CPI-2 anchor  <- the placebo
        t_no = idx[m.values & ok.values].difference(a)
        e_no = declusters(t_no, h, idx[ok.values])
        r = summarize(fw.loc[e_no].values, f"thrust rk>={thr:.0f} NOT CPI-2, h={h}")
        r["n_days"] = len(t_no)
        rows.append(r)
    show(rows, f"attribution at rank5 >= {thr:.0f}")

# placebo anchors: the same k-td-before construction on OTHER events
print("\n### A2. anchor placebo: same construction on PPI, opex, FOMC ###")
h, thr = 2, 80.0
fw = fwd_lag(g, h, LAG)
ok = fw.notna()
m = (rk5 >= thr).fillna(False)
rows = []
for kind in ("cpi", "ppi", "opex", "fomc_decision", "nfp"):
    ev = load_events([kind])["date"]
    an = []
    for d in ev:
        p = int(np.searchsorted(idx.values, np.datetime64(d)))
        if p - 2 < 0 or p >= len(idx):
            continue
        an.append(idx[p - 2])
    an = pd.DatetimeIndex(sorted(set(an)))
    t = pd.DatetimeIndex(an).intersection(idx[m.values & ok.values])
    if len(t) == 0:
        rows.append({"label": kind, "n": 0})
        continue
    e = declusters(t, h, idx[ok.values])
    v = fw.loc[e].values
    r = summarize(v, f"{kind}-2 x thrust, h={h}")
    r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    rows.append(r)
show(rows, "placebo: does ANY event -2 anchor 'work' with the thrust?")

# ---------------------------------------------------------------- B. the grid
print("\n### B. THE GRID I BUILT (k x threshold x horizon). Best-of-N pricing. ###")
grid = []
for k in (1, 2, 3, 4):
    a = anchors(k)
    for thr in (0.0, 80.0, 90.0):
        m = (rk5 >= thr).fillna(False)
        for h in (1, 2, 3, 5):
            fw = fwd_lag(g, h, LAG)
            ok = fw.notna()
            t = pd.DatetimeIndex(a).intersection(idx[m.values & ok.values])
            if len(t) < 5:
                continue
            e = declusters(t, h, idx[ok.values])
            v = fw.loc[e].values
            grid.append({"k": k, "thr": thr, "h": h, "n": len(v),
                         "mean_pct": round(100 * v.mean(), 3),
                         "t": round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2),
                         "hit": round(100 * (v > 0).mean(), 1),
                         "edge_pct": round(100 * (v.mean() - fw[ok].mean()), 3)})
gdf = pd.DataFrame(grid).sort_values("t", ascending=False)
print(gdf.to_string(index=False))
nc = len(gdf)
best_t = gdf.iloc[0]["t"]
print(f"\ncells searched = {nc}; best |t| = {best_t:.2f}")
print(f"positive-mean cells = {(gdf.mean_pct > 0).sum()}/{nc}; "
      f"cells with t>=2 = {(gdf.t >= 2).sum()}/{nc} (expect ~{0.025*nc:.1f} by chance)")

# ---------------------------------------------------------------- C. stability
print("\n### C. stability of the headline cell (k=2, rank5>=80, h=2) ###")
k, thr, h = 2, 80.0, 2
a = anchors(k)
fw = fwd_lag(g, h, LAG)
ok = fw.notna()
m = (rk5 >= thr).fillna(False)
t = pd.DatetimeIndex(a).intersection(idx[m.values & ok.values])
e = declusters(t, h, idx[ok.values])
v = fw.loc[e].values
print(f"N={len(v)} mean={100*v.mean():+.3f}% hit={100*(v>0).mean():.1f}% "
      f"t={v.mean()/(v.std(ddof=1)/np.sqrt(len(v))):.2f} "
      f"sign_p={sign_test(int((v>0).sum()), len(v)):.4f} "
      f"boot_p={bootstrap_p_le0(v):.3f}")
print("  concentration:", cluster_note(e, v, k=2))
o = np.argsort(v)
print("  drop-2-best:", round(100 * np.delete(v, o[-2:]).mean(), 3),
      "% | drop-2-worst:", round(100 * np.delete(v, o[:2]).mean(), 3), "%")
show(era_split(e, v), "era split")
mt = np.array([d.year % 4 == 2 for d in e])
show([summarize(v[mt], f"MIDTERM (N={int(mt.sum())})"), summarize(v[~mt], "non-midterm")],
     "cycle split -- 2026 IS midterm")
yrs = pd.DatetimeIndex(e).year
by = pd.Series(v).groupby(yrs.values).agg(["count", "mean"])
by["mean"] = (100 * by["mean"]).round(2)
print("\nyear histogram:\n", by.to_string())
loyo = [(int(y), round(100 * v[yrs.values != y].mean(), 3)) for y in sorted(set(yrs))]
print("LOYO means:", loyo)
print("LOYO min:", min(x[1] for x in loyo))

# does GLD do the same thing on the same days?  (transfer, conditioned form)
gl = P["GLD"]
fwl = fwd_lag(gl, h, LAG)
show([summarize(v, "GDX cell"), summarize(fwl.loc[e].values, "GLD, same days"),
      summarize(fw[ok].values, "GDX all days"), summarize(fwl[fwl.notna()].values, "GLD all days")],
     "does the registry's GLD kill transfer? (conditioned form)")
