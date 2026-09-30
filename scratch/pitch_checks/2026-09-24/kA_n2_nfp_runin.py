"""kA N2 round 1: long TLT from the k=-6 close into the NFP close with ^TNX at
(or within 1% of) its 252 high. Pre-specified sign LONG. Job: kill it.

Order of reading, as instructed: (1) entry-offset ladder k=-10..-1 exiting at
the NFP close, (2) placebo ladder of non-NFP anchors and a tdom-matched
month-turn control, THEN (3) the headline k=-6 gated cell.
Gate is measured on the SIGNAL close (entry - 1), which is what is knowable
before a MOC entry: today that is 2026-09-23 (^TNX 5.114, at its 252 high).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

raw = load_prices(["TLT", "^TNX"])
tlt = raw["TLT"]["Close"]
IDX = tlt.index
C = tlt.to_numpy()
tnx_v = raw["^TNX"]["Close"].dropna()
tnx_hi = tnx_v.rolling(252).max()
tnx_ratio = (tnx_v / tnx_hi).reindex(IDX).ffill()
TR = tnx_ratio.to_numpy()

nfp = load_events(["nfp"])["date"]
nfp = nfp[nfp <= IDX[-1]]
pos0, kept = anchor_positions(IDX, nfp, 0)
pos0 = np.array(pos0)
print(f"NFP anchors measured: {len(pos0)} ({kept[0].date()}..{kept[-1].date()})")
print(f"LIVE: ^TNX ratio to 252 high on 2026-09-23 = {TR[-1]:.4f}; next NFP 2026-10-02")
nfp_set = set(pos0.tolist())

YEARS = np.array([IDX[p].year for p in pos0])
MONTH = np.array([IDX[p].month for p in pos0])


def window(p_end: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """return from close p_end+k (entry) to close p_end; gate at entry-1."""
    ok = (p_end + k - 1 >= 0) & (p_end < len(C))
    pe = p_end[ok]
    r = C[pe] / C[pe + k] - 1.0
    g = TR[pe + k - 1]
    return r, g, ok


def drift(n: int) -> float:
    s = pd.Series(C)
    return float((s.shift(-n) / s - 1).mean())


GATE = 0.99
rows = []
for k in range(-10, 0):
    r, g, ok = window(pos0, k)
    gm = g >= GATE
    n = -k
    d = drift(n)
    wg = int((r[gm] > 0).sum())
    rows.append({"k": k, "n_all": len(r), "all_pct": 100 * r.mean(), "all_hit": 100 * (r > 0).mean(),
                 "drift_pct": 100 * d, "n_gate": int(gm.sum()), "gate_pct": 100 * r[gm].mean(),
                 "gate_rec": f"{wg}-{int(gm.sum())-wg}", "gate_signp": sign_test(wg, int(gm.sum())),
                 "compl_pct": 100 * r[~gm & ~np.isnan(g)].mean(),
                 "gate_per_day_bp": 1e4 * r[gm].mean() / n})
show(rows, "1. ENTRY LADDER k=-10..-1 -> NFP close (gate: TNX >= 99% of 252 high at entry-1)")

# 2a. placebo ladder: same k=-6 construction, anchors shifted off the NFP day
prow = []
for s in (-20, -15, -10, -5, 0, 5, 10, 15):
    pe = pos0 + s
    pe = pe[(pe < len(C)) & (pe - 7 >= 0)]
    r, g, ok = window(pe, -6)
    gm = g >= GATE
    wg = int((r[gm] > 0).sum())
    prow.append({"anchor_shift": s, "n": len(r), "all_pct": 100 * r.mean(), "n_gate": int(gm.sum()),
                 "gate_pct": 100 * r[gm].mean(), "gate_rec": f"{wg}-{int(gm.sum())-wg}"})
show(prow, "2a. PLACEBO anchors (NFP pos + shift), window k=-6 -> anchor; shift 0 = the cell")

# 2b. tdom-matched month-turn control: every 6-td window ending on tdom 1..7
tdom = pd.Series(1, index=IDX).groupby([IDX.year, IDX.month]).cumsum().to_numpy()
ends = np.array([p for p in range(7, len(C)) if tdom[p] <= 7])
r_all, g_all, _ = window(ends, -6)
is_nfp = np.array([p in nfp_set for p in ends])
gm_all = g_all >= GATE
show([summarize(r_all[is_nfp], "tdom1-7 end, IS NFP day"),
      summarize(r_all[~is_nfp], "tdom1-7 end, NOT NFP day"),
      summarize(r_all[is_nfp & gm_all], "IS NFP & TNX gate"),
      summarize(r_all[~is_nfp & gm_all], "NOT NFP & TNX gate (day-level, overlapping)")],
     "2b. tdom-matched month-turn control (6-td windows)")

# 3. headline cell k=-6
r, g, ok = window(pos0, -6)
ends6 = pos0[ok]
dates6 = IDX[ends6]
gm = g >= GATE
at_hi = g >= 0.99999
ep = r[gm]
w = int((ep > 0).sum())
print(f"\n3. HEADLINE k=-6 gated: N={len(ep)} mean {100*ep.mean():+.3f}% hit "
      f"{100*(ep>0).mean():.0f}%  record {w}-{len(ep)-w} sign p {sign_test(w, len(ep)):.4f}"
      f"  | ungated {100*r.mean():+.3f}% (N={len(r)})  complement {100*r[~gm].mean():+.3f}%"
      f"  | 6d own drift {100*drift(6):+.3f}%")
show([summarize(ep, "gated k=-6 (within 1%)"), summarize(r[at_hi], "gated AT 252 high"),
      summarize(r[~gm], "complement (TNX < 99% of high)"), summarize(r, "all NFP k=-6")],
     "3. cell vs parent")
gd = dates6[gm]
show(era_split(gd, ep), "3b. gated era split")
mid = np.array([d.year % 4 == 2 for d in gd])
show([summarize(ep[mid], "gated midterm"), summarize(ep[~mid], "gated non-midterm"),
      summarize(r[np.array([d.year % 4 == 2 for d in dates6])], "ungated midterm"),
      summarize(r[np.array([d.year % 4 != 2 for d in dates6])], "ungated non-midterm")],
     "3c. midterm split")
octp = np.array([d.month == 10 for d in dates6])
show([summarize(r[octp], "ungated October prints (Sept run-in)"),
      summarize(r[octp & gm], "gated October prints")], "3d. the live month")
# run-in vs print session decomposition at gated anchors
run = C[ends6 - 1] / C[ends6 - 6] - 1.0
prt = C[ends6] / C[ends6 - 1] - 1.0
show([summarize(run[gm], "gated run-in k=-6 -> k=-1"), summarize(prt[gm], "gated print session"),
      summarize(run, "all run-in"), summarize(prt, "all print session")], "3e. decomposition")
print(f"  concentration: {cluster_note(gd, ep)}")
print("  gated episodes:")
for d, v, gg in zip(gd, ep, g[gm]):
    print(f"    NFP {d.date()}  {100*v:+.3f}%  TNX/hi {gg:.4f}")
print("\n4. book overlap: see kA_r1 notes (no book strategy trades TLT; TMF only in 3x fades)")
