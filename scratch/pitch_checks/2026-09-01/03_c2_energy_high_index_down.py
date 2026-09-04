"""C2 KILL CHECK — a sector ETF closing AT a fresh trailing-252 high on a
session the INDEX fell (a divergence at a high, not a plain new high).

Live: XLE closed 0.00% from its 52-week high, +2.04% on a session SPY fell
-0.30%.

The whole candidate stands or falls on ONE question, so it is asked first and
loudly: does the index's direction on the trigger day MATTER? If XLE's forward
return after a fresh high is the same whether SPY rose or fell, "divergence"
is a label on a plain new-high buy and the candidate is dead. That is the gate
attribution test and it is the likely kill.

Then the NINE-SECTOR reference class, which has killed this shape repeatedly
here: if every sector shows the same excess, there is nothing about energy.

BOOK OVERLAP DISCLOSED: the scanner has a live staged OVS SHORT in SLB today,
same complex, opposite side.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd
import numpy as np

ASOF = pd.Timestamp("2026-08-31")
SEC9 = ["XLE", "XLK", "XLF", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]
EXTRA = ["XOP", "OIH", "USO", "SPY"]
PX = load_prices(SEC9 + EXTRA)
PX = {t: d[d.index <= ASOF] for t, d in PX.items()}
SPY = PX["SPY"]["Close"].dropna()
SPY_1D = SPY.pct_change()

TOL = 0.0005  # "at the high" tolerance: within 5 bps of the trailing-252 max


def at_high(s, tol=TOL):
    hi = s.rolling(252, min_periods=252).max()
    return (s >= hi * (1 - tol))


def masks_for(t):
    """(at a fresh high AND index down, at a fresh high AND index up,
        at a fresh high unconditional) on the ticker's own index."""
    s = PX[t]["Close"].dropna()
    ah = at_high(s)
    sd = SPY_1D.reindex(s.index)
    return (ah & (sd < 0)).fillna(False), (ah & (sd >= 0)).fillna(False), ah.fillna(False), s


# ---------------------------------------------------------------------------
# 1. COUNT FIRST
# ---------------------------------------------------------------------------
print("== LIVE XLE: close %.4f, trailing-252 max %.4f -> %.3f%% off high; "
      "SPY 1d %+.2f%%" % (
          PX["XLE"]["Close"].iloc[-1],
          PX["XLE"]["Close"].dropna().rolling(252).max().iloc[-1],
          100 * (PX["XLE"]["Close"].dropna().iloc[-1]
                 / PX["XLE"]["Close"].dropna().rolling(252).max().iloc[-1] - 1),
          100 * SPY_1D.iloc[-1]))

dn, up, ah, xle = masks_for("XLE")
xidx = xle.index
print("\n===== 1. COUNT FIRST (XLE, 2001+) =====")
for lbl, m in (("at 252-high, ANY session", ah),
               ("at 252-high & SPY DOWN (the cell)", dn),
               ("at 252-high & SPY UP", up)):
    t = xidx[m.reindex(xidx, fill_value=False).values]
    print("  %-38s %4d days | %3d episodes(gap10) | %3d episodes(gap21)"
          % (lbl, len(t), len(declusters(t, 10, xidx)), len(declusters(t, 21, xidx))))
t_dn = xidx[dn.reindex(xidx, fill_value=False).values]
print("  cell year histogram:", pd.Series(t_dn.year).value_counts().sort_index().to_dict())

# ---------------------------------------------------------------------------
# 2. horizon scan + battery
# ---------------------------------------------------------------------------
px_xle = pd.DataFrame({"XLE": xle})
print("\n===== 2. HORIZON SCAN (XLE long, cell, lag=1) =====")
show(horizon_scan(px_xle, t_dn, [("XLE", 1.0)], hs=(1, 2, 3, 5, 10, 21), min_gap=10),
     "horizon scan (episodes, gap 10)")

variants = {
    "at high & SPY down (CELL)": dn,
    "at high & SPY up": up,
    "at high, ANY session (no gate)": ah,
    "at high & SPY down >0.5%": (ah & (SPY_1D.reindex(xidx) < -0.005)).fillna(False),
    "at high & XLE up >1% & SPY down": (
        ah & (xle.pct_change() > 0.01) & (SPY_1D.reindex(xidx) < 0)).fillna(False),
    "within 1% of high & SPY down": (
        at_high(xle, 0.01) & (SPY_1D.reindex(xidx) < 0)).fillna(False),
    "within 2% of high & SPY down": (
        at_high(xle, 0.02) & (SPY_1D.reindex(xidx) < 0)).fillna(False),
}
for H in (5, 10):
    battery(px_xle, dn, [("XLE", 1.0)], H,
            "C2 XLE long, at 252-high on a DOWN-SPY session",
            cost_bps=4.0, variants=variants, min_gap=10)

# ---------------------------------------------------------------------------
# 3. GATE ATTRIBUTION — the decisive test
# ---------------------------------------------------------------------------
print("\n===== 3. GATE ATTRIBUTION: does SPY's direction on the trigger day "
      "matter at all? (XLE, lag=1) =====")
for H in (3, 5, 10, 21):
    r = fwd_lag(xle, H, 1)
    rows = []
    for lbl, m in (("at high & SPY DOWN (cell)", dn),
                   ("at high & SPY UP", up),
                   ("at high, ANY (no gate)", ah)):
        t = xidx[m.reindex(xidx, fill_value=False).values].intersection(r.dropna().index)
        e = declusters(t, 10, xidx)
        s = summarize(r.reindex(e).values, lbl)
        s["n_days"] = len(t)
        rows.append(s)
    rows.append(summarize(r.dropna().values, "XLE all days (own drift)"))
    show(rows, f"h={H}")
    # direct DOWN-vs-UP contrast at DAY level (no declustering loss)
    td = xidx[dn.reindex(xidx, fill_value=False).values].intersection(r.dropna().index)
    tu = xidx[up.reindex(xidx, fill_value=False).values].intersection(r.dropna().index)
    a, b = r.reindex(td).values, r.reindex(tu).values
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) > 1 and len(b) > 1:
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        print(f"  DOWN minus UP (day level) = {100*(a.mean()-b.mean()):+.3f}pp "
              f"welch t = {(a.mean()-b.mean())/se:+.2f}   "
              f"(if this is ~0 the 'divergence' is a LABEL)")

# ---------------------------------------------------------------------------
# 4. NINE-SECTOR REFERENCE CLASS
# ---------------------------------------------------------------------------
print("\n===== 4. NINE-SECTOR REFERENCE CLASS (identical cell on each sector) =====")
for H in (5, 10):
    rows, pooled = [], []
    for t in SEC9:
        m_dn, m_up, m_ah, s = masks_for(t)
        r = fwd_lag(s, H, 1)
        idx = s.index[m_dn.reindex(s.index, fill_value=False).values]
        idx = idx.intersection(r.dropna().index)
        e = declusters(idx, 10, s.index)
        drift = 100 * r.dropna().mean()
        rec = summarize(r.reindex(e).values, t)
        rec["n_days"] = len(idx)
        rec["own_drift_pct"] = round(drift, 3)
        if rec["n"]:
            rec["excess_pct"] = round(rec["mean_pct"] - drift, 3)
            wins = int((r.reindex(e).values > 0).sum())
            rec["sign_p"] = round(sign_test(wins, rec["n"]), 4)
            # the up-gate contrast for the SAME sector
            iu = s.index[m_up.reindex(s.index, fill_value=False).values]
            iu = iu.intersection(r.dropna().index)
            eu = declusters(iu, 10, s.index)
            rec["UPgate_mean_pct"] = round(100 * r.reindex(eu).mean(), 3)
            pooled.extend(list(r.reindex(e).values - r.dropna().mean()))
        rows.append(rec)
    show(rows, f"per-sector, h={H}")
    pe = np.asarray(pooled, float)
    pe = pe[~np.isnan(pe)]
    if len(pe) > 2:
        wins = int((pe > 0).sum())
        print(f"  POOLED 9-sector EXCESS: N={len(pe)} mean {100*pe.mean():+.3f}pp "
              f"t={pe.mean()/(pe.std(ddof=1)/np.sqrt(len(pe))):+.2f} "
              f"record {wins}-{len(pe)-wins} sign p={sign_test(wins, len(pe)):.4f}")
        # XLE distinguishable?
        m_dn, _, _, s = masks_for("XLE")
        r = fwd_lag(s, H, 1)
        idx = s.index[m_dn.reindex(s.index, fill_value=False).values].intersection(r.dropna().index)
        e = declusters(idx, 10, s.index)
        xe = r.reindex(e).values - r.dropna().mean()
        xe = xe[~np.isnan(xe)]
        se = np.sqrt(xe.var(ddof=1) / len(xe) + pe.var(ddof=1) / len(pe))
        print(f"  XLE excess {100*xe.mean():+.3f}pp (N={len(xe)}) vs family "
              f"{100*pe.mean():+.3f}pp -> welch t = {(xe.mean()-pe.mean())/se:+.2f}")

# ---------------------------------------------------------------------------
# 5. LAG PROFILE + declustering sensitivity
# ---------------------------------------------------------------------------
print("\n===== 5. LAG PROFILE (XLE cell) =====")
for H in (5, 10):
    for lag in (0, 1, 2, 3):
        r = fwd_lag(xle, H, lag)
        e = declusters(t_dn.intersection(r.dropna().index), 10, xidx)
        s = summarize(r.reindex(e).values, "")
        print("  h=%2d lag=%d  N=%2d mean %+.3f%% hit %.1f%% t %s"
              % (H, lag, s["n"], s["mean_pct"], s["hit"],
                 f"{s['t']:+.2f}" if s["n"] > 1 else "na"))

print("\n===== 5b. DECLUSTERING SENSITIVITY (h=10, lag=1) =====")
r = fwd_lag(xle, 10, 1)
for gap in (5, 10, 21, 42):
    e = declusters(t_dn.intersection(r.dropna().index), gap, xidx)
    s = summarize(r.reindex(e).values, f"gap={gap}")
    print("  gap=%2d N=%2d mean %+.3f%% hit %.1f%% t %s  drop-best-2 %+.3f%% "
          "drop-best-3 %+.3f%%"
          % (gap, s["n"], s["mean_pct"], s["hit"],
             f"{s['t']:+.2f}" if s["n"] > 1 else "na",
             100 * np.sort(r.reindex(e).values)[:-2].mean() if s["n"] > 3 else np.nan,
             100 * np.sort(r.reindex(e).values)[:-3].mean() if s["n"] > 4 else np.nan))
    print("      concentration:", cluster_note(e, r.reindex(e).values))

# ---------------------------------------------------------------------------
# 6. EFFECTIVE-N on the energy complex (standing caveat) + book overlap
# ---------------------------------------------------------------------------
print("\n===== 6. STANDING ENERGY CAVEAT =====")
ec = ["XLE", "XOP", "OIH", "USO"]
rr = pd.DataFrame({t: PX[t]["Close"].pct_change() for t in ec}).dropna()
ev = np.linalg.eigvalsh(np.corrcoef(rr.values.T))[::-1]
print("  4-vehicle energy correlation eigenvalues:", np.round(ev, 3),
      "-> PC1 explains %.1f%%, effective names %.2f of 4"
      % (100 * ev[0] / ev.sum(), (ev.sum() ** 2) / (ev ** 2).sum()))
print("  BOOK OVERLAP: live staged OVS SHORT in SLB (same complex, opposite side).")
