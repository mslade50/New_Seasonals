"""C4 / C11 round 1 -- the FIRST test is basket correlation with the live book.

The Daily Pitch published LONG GDX on 2026-08-10 (MOC at that close, 5 td,
exit MOC 2026-08-17). That position is live RIGHT NOW and covers 4 of the 5
sessions C4/C11 would trade (signal close 2026-08-10 -> MOC entry 2026-08-11
-> exit 2026-08-18).

So the question is never "does long SLV pay on a GDX thrust". It is "does
long SLV pay ANYTHING McKinley does not already own". This script answers
that with a correlation, a beta and a residual alpha, and only then runs the
standard battery.

Windows, stated once so the overlap is honest:
  LIVE GDX leg : entry close D+0 (the signal close), exit close D+5   [lag=0]
  C4  SLV leg  : entry close D+1,                    exit close D+6   [lag=1]
  C11 GLD leg  : entry close D+1,                    exit close D+6   [lag=1]
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, fwd_lag, declusters, summarize, sign_test,  # noqa: E402
                       pct_rank, local_control, bootstrap_p_le0, cluster_note,
                       era_split, show, battery)

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

TK = ["GDX", "GLD", "SLV", "SPY", "NEM", "XME", "GDXJ"]
px = close_panel(TK)
idx = px.index

H = 5
gdx_rank5 = pct_rank(px["GDX"], 5)
mask = (gdx_rank5 >= 95).fillna(False)

# the live GDX leg's window is lag=0 (it entered at the signal close)
live_gdx = fwd_lag(px["GDX"], H, lag=0)
slv = fwd_lag(px["SLV"], H, lag=1)
gld = fwd_lag(px["GLD"], H, lag=1)

valid = live_gdx.notna() & slv.notna() & gld.notna()
sig_all = idx[mask.reindex(idx, fill_value=False).values & valid.values]
epi = declusters(sig_all, 5, idx)

print("=" * 96)
print("TODAY'S READING")
print("=" * 96)
last = px["GDX"].dropna().index[-1]
print(f"freshest bar {last.date()}   GDX close {px['GDX'].loc[last]:.2f}  "
      f"5d ret {100*(px['GDX'].pct_change(5).loc[last]):+.2f}%  "
      f"5d rank {gdx_rank5.loc[last]:.1f}")
print(f"SLV {px['SLV'].loc[last]:.2f}  5d {100*px['SLV'].pct_change(5).loc[last]:+.2f}%   "
      f"GLD {px['GLD'].loc[last]:.2f}  5d {100*px['GLD'].pct_change(5).loc[last]:+.2f}%")
print(f"triggers: {len(sig_all)} days, {len(epi)} declustered episodes, "
      f"span {epi[0].date()} .. {epi[-1].date()}")

# ---------------------------------------------------------------------------
print()
print("=" * 96)
print("TEST 1  BASKET CORRELATION vs THE LIVE GDX POSITION  (the decisive one)")
print("=" * 96)
for name, leg in (("C4  SLV", slv), ("C11 GLD", gld)):
    a = leg.loc[epi].values
    b = live_gdx.loc[epi].values
    c_all = leg.loc[idx[valid]].values
    b_all = live_gdx.loc[idx[valid]].values
    r_trig = np.corrcoef(a, b)[0, 1]
    r_all = np.corrcoef(c_all, b_all)[0, 1]
    beta = np.polyfit(b, a, 1)
    resid = a - (beta[0] * b + beta[1])
    # alpha = intercept, i.e. what the leg pays when the live GDX leg pays 0
    n = len(a)
    se_a = np.std(resid, ddof=2) / np.sqrt(n) * np.sqrt(1 + b.mean() ** 2 / b.var())
    same_sign = 100 * np.mean(np.sign(a) == np.sign(b))
    print(f"\n{name} vs LIVE GDX leg (both over the overlapping hold):")
    print(f"   corr on trigger episodes (N={n})   = {r_trig:+.3f}")
    print(f"   corr on ALL days, full history      = {r_all:+.3f}")
    print(f"   beta of leg on the live GDX leg     = {beta[0]:+.3f}")
    print(f"   intercept (alpha when GDX pays 0)   = {100*beta[1]:+.3f}%   "
          f"t = {beta[1]/se_a:+.2f}")
    print(f"   same-sign agreement with the live leg = {same_sign:.1f}% of episodes")
    print(f"   leg mean {100*a.mean():+.3f}%   live-GDX-leg mean over the same "
          f"episodes {100*b.mean():+.3f}%")
    # what does the leg add on the episodes where the live leg LOST?
    lost = b < 0
    print(f"   when the live GDX leg LOST (N={int(lost.sum())}): leg pays "
          f"{100*a[lost].mean():+.3f}%, hit {100*(a[lost] > 0).mean():.1f}%")
    won = ~lost
    print(f"   when the live GDX leg WON  (N={int(won.sum())}): leg pays "
          f"{100*a[won].mean():+.3f}%, hit {100*(a[won] > 0).mean():.1f}%")

# ---------------------------------------------------------------------------
print()
print("=" * 96)
print("TEST 2  IS SLV JUST LEVERED GDX/GLD?  multiple regression on the same window")
print("=" * 96)
# same-window (lag=1) GDX and GLD, so this is a pure instrument question
gdx1 = fwd_lag(px["GDX"], H, lag=1)
for name, leg in (("SLV", slv), ("GLD", gld)):
    y = leg.loc[epi].values
    if name == "SLV":
        X = np.column_stack([np.ones(len(epi)), gdx1.loc[epi].values, gld.loc[epi].values])
        cols = ["alpha", "b_GDX", "b_GLD"]
    else:
        X = np.column_stack([np.ones(len(epi)), gdx1.loc[epi].values])
        cols = ["alpha", "b_GDX"]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ coef
    res = y - fit
    dof = len(y) - X.shape[1]
    s2 = res @ res / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    r2 = 1 - res.var() / y.var()
    print(f"\n{name} (lag=1, h={H}) regressed on same-window metals legs, N={len(y)}:")
    for c, b_, s_ in zip(cols, coef, se):
        unit = "%" if c == "alpha" else ""
        val = 100 * b_ if c == "alpha" else b_
        print(f"   {c:<7} {val:+8.3f}{unit:<2}  se {(100*s_ if c=='alpha' else s_):7.3f}  "
              f"t {b_/s_:+6.2f}")
    print(f"   R^2 = {r2:.3f}   residual mean {100*res.mean():+.4f}%  "
          f"residual sd {100*res.std(ddof=1):.3f}%")
    print(f"   -> the cell's edge net of the metals complex is the ALPHA row above.")

# ---------------------------------------------------------------------------
print()
print("=" * 96)
print("TEST 3  ROUND-1 BATTERY, both legs")
print("=" * 96)
battery(px, mask, [("SLV", 1.0)], H, "C4 long SLV on GDX 5d rank>=95",
        cost_bps=8.0, min_gap=5)
battery(px, mask, [("GLD", 1.0)], H, "C11 long GLD on GDX 5d rank>=95",
        cost_bps=6.0, min_gap=5)
