"""ROUND 2 item 1 + independent verification.

The candidate recovered from C1's corpse: LONG TLT (and the TLT-SPY spread),
entered MOC on 2026-08-18, exited at the 2026-08-31 month-end close (h=9
sessions after the entry close), gated on TLT's trailing 21d return <= -2.5%.

(a) verify every headline number from scratch, my own construction
(b) build the FULL grid the checker implicitly searched and rank the chosen
    cell in it
(c) rotation-permutation null: how often does SOME cell in that grid look
    this good when the association is broken but the calendar, the gate and
    the return autocorrelation are all preserved
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

LAG = 1
TK = ["SPY", "TLT", "IEF", "LQD"]
raw = load_prices(TK)
idx = raw["SPY"]["Close"].index
for t in TK[1:]:
    idx = idx.intersection(raw[t]["Close"].index)
px = pd.DataFrame({t: raw[t]["Close"].reindex(idx) for t in TK}).dropna()
idx = px.index
print("panel %s .. %s  N=%d rows" % (idx[0].date(), idx[-1].date(), len(idx)))

ymv = pd.Series(idx.year * 100 + idx.month, index=idx)
is_last = ymv.ne(ymv.shift(-1)).values
is_last[-1] = False           # last row is not a known month end
pos = pd.Series(range(len(idx)), index=idx)
T21 = px["TLT"].pct_change(21)


def anchor(h: int, k: int = 0) -> np.ndarray:
    """True on signal day D when D+LAG+h+k is the month's last session."""
    t = pos.values + LAG + h + k
    m = np.zeros(len(idx), dtype=bool)
    ok = t < len(idx)
    m[ok] = is_last[t[ok]]
    return m


def rets(h: int) -> dict:
    rT = fwd_lag(px["TLT"], h, LAG)
    rS = fwd_lag(px["SPY"], h, LAG)
    rI = fwd_lag(px["IEF"], h, LAG)
    rL = fwd_lag(px["LQD"], h, LAG)
    return {"TLT": rT, "TLT-SPY": rT - rS, "IEF": rI, "IEF-SPY": rI - rS,
            "LQD-SPY": rL - rS, "-SPY": -rS}


# ---------------------------------------------------------------- (a) verify
H = 9
R = rets(H)
A9 = anchor(H)
print("\nanchor fires on the final row (2026-08-17)? %s   live TLT21d = %+.3f%%"
      % (bool(A9[-1]), 100 * T21.iloc[-1]))
G = (T21 <= -0.025).fillna(False).values

rows = []
for lab in ("TLT", "TLT-SPY", "IEF", "IEF-SPY", "LQD-SPY", "-SPY"):
    r = R[lab]
    v = r.notna().values
    d = idx[A9 & G & v]
    rows.append(summarize(r.loc[d].values, f"{lab} | anchor+gate (N={len(d)})"))
show(rows, "(a1) VERIFY headline vehicles, h=9 lag=1, TLT21d<=-2.5%")

r = R["TLT-SPY"]; v = r.notna().values
dS = idx[A9 & G & v]
epi = declusters(dS, 21, idx)
ve = r.loc[epi].values
w = int((ve > 0).sum())
print("(a2) SPREAD episodes N=%d mean %+.3f%% hit %.1f%% sign p=%.4f boot P(<=0)=%.3f"
      % (len(epi), 100 * ve.mean(), 100 * (ve > 0).mean(),
         sign_test(w, len(epi)), bootstrap_p_le0(ve)))
rt = R["TLT"]
epiT = declusters(idx[A9 & G & rt.notna().values], 21, idx)
vt = rt.loc[epiT].values
wt = int((vt > 0).sum())
print("(a2) TLT-ONLY episodes N=%d mean %+.3f%% hit %.1f%% sign p=%.4f boot P(<=0)=%.3f"
      % (len(epiT), 100 * vt.mean(), 100 * (vt > 0).mean(),
         sign_test(wt, len(epiT)), bootstrap_p_le0(vt)))

# local +/-126td control, both vehicles
for lab in ("TLT", "TLT-SPY"):
    rr = R[lab]; vv = rr.notna().values
    dd = idx[A9 & G & vv]
    loc = local_control(idx[vv], dd, 126)
    print("(a3) %-8s local +/-126td ex-trigger ctrl: N=%d mean %+.3f%% | "
          "all-days %+.3f%% | anchor-only(no gate) %+.3f%% | gate-only(no anchor) %+.3f%%"
          % (lab, len(loc), 100 * rr.loc[loc].mean(), 100 * rr[vv].mean(),
             100 * rr.loc[idx[A9 & vv]].mean(),
             100 * rr.loc[idx[G & vv & ~A9]].mean()))

# anchor specificity: same gate away from month end
for lab in ("TLT", "TLT-SPY"):
    rr = R[lab]; vv = rr.notna().values
    off = idx[G & vv & ~A9]
    print("(a4) %-8s gate ON, anchor OFF: N=%d mean %+.3f%% t=%.2f"
          % (lab, len(off), 100 * rr.loc[off].mean(),
             summarize(rr.loc[off].values)["t"]))

# ------------------------------------------------------------- (b) the grid
RUNGS = [None, 0.0, -0.005, -0.01, -0.015, -0.02, -0.025, -0.03, -0.0336,
         -0.04, -0.045, -0.05, -0.06]
VEH = ["TLT", "TLT-SPY", "IEF", "IEF-SPY", "LQD-SPY"]
KOFF = list(range(0, 13))          # exit offset from month-end (anchor family)
HS = [8, 9, 10]                    # the horizons the checker moved between

cache = {h: rets(h) for h in HS}


def grid_stats(shift: int = 0) -> pd.DataFrame:
    out = []
    for h in HS:
        Rh = cache[h]
        for k in KOFF:
            A = anchor(h, k)
            for thr in RUNGS:
                g = np.ones(len(idx), bool) if thr is None else (T21 <= thr).fillna(False).values
                for vlab in VEH:
                    rr = Rh[vlab]
                    arr = rr.values
                    if shift:
                        arr = np.roll(arr, shift)
                    vv = ~np.isnan(arr)
                    sel = A & g & vv
                    n = int(sel.sum())
                    if n < 15:
                        continue
                    x = arr[sel]
                    t = x.mean() / (x.std(ddof=1) / np.sqrt(n))
                    out.append({"h": h, "k": k, "thr": thr, "veh": vlab,
                                "n": n, "mean_pct": 100 * x.mean(), "t": t})
    return pd.DataFrame(out)


gs = grid_stats()
print("\n(b1) grid size = %d cells (h %s x exit-offset %s x %d rungs x %d vehicles, n>=15)"
      % (len(gs), HS, KOFF, len(RUNGS), len(VEH)))
chosen = gs[(gs.h == 9) & (gs.k == 0) & (gs.thr == -0.025) & (gs.veh == "TLT-SPY")]
chosenT = gs[(gs.h == 9) & (gs.k == 0) & (gs.thr == -0.025) & (gs.veh == "TLT")]
for lab, c in (("TLT-SPY", chosen), ("TLT", chosenT)):
    row = c.iloc[0]
    rk_t = int((gs.t > row.t).sum()) + 1
    rk_m = int((gs.mean_pct > row.mean_pct).sum()) + 1
    print("(b2) chosen cell %-8s mean %+.3f%% t=%.2f  ->  RANK %d/%d by t, %d/%d by mean"
          % (lab, row.mean_pct, row.t, rk_t, len(gs), rk_m, len(gs)))
print("\n(b3) top 12 cells in the whole grid by t:")
print(gs.sort_values("t", ascending=False).head(12).round(3).to_string(index=False))
print("\n(b4) how many grid cells beat t=2.0 / t=2.5 / t=2.82: %d / %d / %d  (%.1f%% / %.1f%% / %.1f%%)"
      % ((gs.t > 2).sum(), (gs.t > 2.5).sum(), (gs.t > 2.82).sum(),
         100 * (gs.t > 2).mean(), 100 * (gs.t > 2.5).mean(), 100 * (gs.t > 2.82).mean()))

# ------------------------------------------------- (c) rotation permutation
rng = np.random.default_rng(7)
obs_max_t = gs.t.max()
obs_t = float(chosen.iloc[0].t)
obs_m = float(chosen.iloc[0].mean_pct)
NB = 300
maxts, maxms = [], []
for _ in range(NB):
    sh = int(rng.integers(63, len(idx) - 63))
    g2 = grid_stats(shift=sh)
    maxts.append(g2.t.max())
    maxms.append(g2.mean_pct.max())
maxts = np.array(maxts); maxms = np.array(maxms)
print("\n(c1) rotation null, %d draws (returns rotated, calendar+gate+autocorr preserved)" % NB)
print("     observed GRID-MAX t = %.2f ; null max-t mean %.2f, 90th %.2f, 95th %.2f -> familywise p = %.3f"
      % (obs_max_t, maxts.mean(), np.percentile(maxts, 90), np.percentile(maxts, 95),
         (maxts >= obs_max_t).mean()))
print("     observed CHOSEN t = %.2f -> P(null grid-max >= chosen t) = %.3f"
      % (obs_t, (maxts >= obs_t).mean()))
print("     observed CHOSEN mean = %+.3f%% -> P(null grid-max mean >= it) = %.3f"
      % (obs_m, (maxms >= obs_m).mean()))
