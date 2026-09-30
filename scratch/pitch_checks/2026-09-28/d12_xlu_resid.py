"""d12 round 1: long XLU against ex-ante beta_TLT*TLT + beta_SPY*SPY after the XLU
residual on a rolling TLT+SPY regression hits a trailing-252 extreme low.

Residual: rolling 252d OLS of XLU daily returns on TLT and SPY (min 200 obs); the residual
on day t uses the coefficients estimated through t-1 (strictly ex-ante). cum21 = 21-session
sum of residuals; gate = trailing-252 percentile of cum21 <= 5 (pre-stated; 2 / 10 and
lookbacks 10 / 15 / 42 are neighbours). Traded object: long XLU, short bT*TLT, short bS*SPY
at the signal-date betas, entry lag 1, h = 5 / 10 pre-specified.
Contrast with watchlist 61 (XLU r21 <= 5 AND TLT r21 < 25, long XLU outright).
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

px = close_panel(["XLU", "TLT", "SPY"])
px = px[px["TLT"].notna() & px["XLU"].notna() & px["SPY"].notna()]
idx = px.index
r = px.pct_change(fill_method=None)
W, MINP = 252, 200


def rolling_ols(y: pd.Series, x1: pd.Series, x2: pd.Series, w: int = W, mp: int = MINP):
    m = lambda s: s.rolling(w, min_periods=mp).mean()
    c11 = m(x1 * x1) - m(x1) ** 2
    c22 = m(x2 * x2) - m(x2) ** 2
    c12 = m(x1 * x2) - m(x1) * m(x2)
    c1y = m(x1 * y) - m(x1) * m(y)
    c2y = m(x2 * y) - m(x2) * m(y)
    det = c11 * c22 - c12 ** 2
    b1 = (c22 * c1y - c12 * c2y) / det
    b2 = (c11 * c2y - c12 * c1y) / det
    a = m(y) - b1 * m(x1) - b2 * m(x2)
    return a, b1, b2


a, bT, bS = rolling_ols(r["XLU"], r["TLT"], r["SPY"])
res = r["XLU"] - (a.shift(1) + bT.shift(1) * r["TLT"] + bS.shift(1) * r["SPY"])
cum = {n: res.rolling(n).sum() for n in (10, 15, 21, 42)}
pctl = {n: cum[n].rolling(252, min_periods=200).rank(pct=True) * 100 for n in cum}

print("LIVE GATE (as of close 2026-09-25)")
for d in idx[-4:]:
    print(f"  {d.date()} bT {bT[d]:+.3f} bS {bS[d]:+.3f}  cum21 {100*cum[21][d]:+.2f}%  "
          f"pctl21 {pctl[21][d]:.1f}  pctl10 {pctl[10][d]:.1f}  pctl42 {pctl[42][d]:.1f}")
xr21 = pct_rank(px["XLU"], 21)
tr21 = pct_rank(px["TLT"], 21)
print(f"  XLU r21 {xr21.iloc[-1]:.2f}  TLT r21 {tr21.iloc[-1]:.2f}  "
      f"(W61 arm: XLU r21<=5 AND TLT r21<25 -> {bool(xr21.iloc[-1] <= 5 and tr21.iloc[-1] < 25)})")
print(f"  beta history: bT median {bT.median():+.3f}, bS median {bS.median():+.3f}; "
      f"today's bT pctile {100*(bT.dropna() <= bT.iloc[-1]).mean():.0f}")


def pair_ret(h: int, betas_at=None) -> pd.Series:
    fx, ft, fs = (fwd_lag(px[t], h) for t in ("XLU", "TLT", "SPY"))
    return fx - bT * ft - bS * fs


def ep(mask, ret, lbl, gap):
    valid = ret.notna()
    d = idx[mask.reindex(idx, fill_value=False).values & valid.values]
    e = declusters(d, gap, idx)
    v = ret.loc[e].values
    rr = summarize(v, lbl)
    if rr["n"]:
        w = int((v > 0).sum())
        rr["rec"] = f"{w}-{len(v)-w}"
        rr["sign_p"] = round(sign_test(w, len(v)), 4)
        rr["edge_pp"] = rr["mean_pct"] - 100 * ret[valid].mean()
        rr["n_days"] = len(d)
    return rr, e, v


gate = pctl[21] <= 5
w61 = (xr21 <= 5) & (tr21 < 25)
wash = xr21 <= 5
print(f"\nmask overlap: gate days {int(gate.sum())}, W61 days {int(w61.sum())}, both {int((gate & w61).sum())}, "
      f"gate & XLU r21<=5 {int((gate & wash).sum())}")

for h in (3, 5, 10):
    gap = max(h, 10)
    P = pair_ret(h)
    X = fwd_lag(px["XLU"], h)
    rows = []
    for lbl, m in [("RESID pctl21<=5", gate), ("RESID pctl21<=2", pctl[21] <= 2),
                   ("RESID pctl21<=10", pctl[21] <= 10), ("RESID pctl10<=5", pctl[10] <= 5),
                   ("RESID pctl15<=5", pctl[15] <= 5), ("RESID pctl42<=5", pctl[42] <= 5),
                   ("W61 joint", w61), ("XLU r21<=5 (plain washout)", wash),
                   ("gate & NOT washout", gate & ~wash), ("washout & NOT gate", wash & ~gate)]:
        rr, e, v = ep(m, P, f"PAIR {lbl}", gap)
        rows.append(rr)
    rows.append(summarize(P.dropna().values, "PAIR all days"))
    for lbl, m in [("RESID pctl21<=5", gate), ("W61 joint", w61)]:
        rr, _, _ = ep(m, X, f"XLU outright {lbl}", gap)
        rows.append(rr)
    rows.append(summarize(X.dropna().values, "XLU outright all days"))
    show(rows, f"h={h}: pair vs controls (episodes, gap {gap})")
    rr, e, v = ep(gate, P, "g", gap)
    show(era_split(e, v), f"h={h} gate pair era split")
    print(f"  {cluster_note(e, v)}")
    if h in (5, 10):
        print("  episodes:", ", ".join(f"{x.date()}:{100*y:+.2f}" for x, y in zip(e, v)))
        # leg attribution on the gate episodes
        legs = {t: fwd_lag(px[t], h).loc[e].values for t in ("XLU", "TLT", "SPY")}
        print(f"  legs on gate episodes: XLU {100*np.nanmean(legs['XLU']):+.3f}%  "
              f"-bT*TLT {100*np.nanmean(-bT.loc[e].values*legs['TLT']):+.3f}%  "
              f"-bS*SPY {100*np.nanmean(-bS.loc[e].values*legs['SPY']):+.3f}%")
        # filter vs reanchor: parent = plain XLU washout, child = residual gate
        pe = declusters(idx[wash.reindex(idx, fill_value=False).values & P.notna().values], gap, idx)
        ce = declusters(idx[gate.reindex(idx, fill_value=False).values & P.notna().values], gap, idx)
        filter_vs_reanchor(P, pd.Series(idx.isin(pe), index=idx), pd.Series(idx.isin(ce), index=idx),
                           idx, window_td=21, label=f"PAIR h={h} parent XLU r21<=5 -> child resid gate")

battery(px.assign(), gate, [("XLU", 1.0)], 5, "XLU OUTRIGHT on resid gate (contrast)", cost_bps=3, min_gap=10,
        event_kinds=("nfp",))
