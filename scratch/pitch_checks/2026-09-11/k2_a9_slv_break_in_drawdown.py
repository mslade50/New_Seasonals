"""A9 round 1 - SLV reversal after a >= 4% single-session break inside a deep
drawdown but a strong year.

Pre-specified cell (from the handover, unchanged):
  SLV 1-session return <= -4.0%
  AND distance below the trailing-252 high >= 25%
  AND trailing-252 return > 0
Long SLV, lag=1 MOC entry.

Live 2026-09-10: SLV -5.30%, 45.55% below the 252-day high, +54.65% / 252d.

Round-1 obligations:
 1/2/5/6 via battery
 gate attribution on EACH of the two conditioners separately (the registry's
   2026-08-10 finding is that SLV's distance-from-high is a U-shaped noise carve)
 the registry adjacency: is this the 2026-09-10 "long SLV" continuation cell
   under a different trigger, and does watchlist 29's SHORT have content at h=1
   on today's exact state?
 GLD beta: regress the SLV forward on the GLD forward over the same windows
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-09-10")
BREAK = -0.040
DD_MIN = 0.25
H = 5
COST_BPS = 4.0

TK = ["SLV", "GLD", "GDX", "USO", "DBC", "XME", "SPY", "PPLT", "CPER"]
px = load_prices(TK)
have = [t for t in TK if t in px]
print("cached:", have)

close = {t: px[t]["Close"] for t in have}


def state(t):
    s = close[t]
    r1 = s.pct_change()
    hi = rolling_on_valid(s, lambda x: x.rolling(252).max())
    dd = 1.0 - s / hi                      # fraction BELOW the 252d high
    r252 = s / s.shift(252) - 1.0
    return s, r1, dd, r252


s, r1, dd, r252 = state("SLV")
print("\n" + "=" * 78)
print("0. LIVE STATE, 2026-09-10 close")
print(f"  SLV 1d = {100*float(r1.loc[ASOF]):+.3f}%   (gate <= -4.0%)")
print(f"  SLV below 252d high = {100*float(dd.loc[ASOF]):.2f}%  (gate >= 25%)")
print(f"  SLV 252d return = {100*float(r252.loc[ASOF]):+.2f}%  (gate > 0)")
for t in ["GLD", "GDX"]:
    print(f"  {t} 1d = {100*float(close[t].pct_change().loc[ASOF]):+.2f}%")

frame = pd.DataFrame({"SLV": s.dropna()})
mask = ((r1 <= BREAK) & (dd >= DD_MIN) & (r252 > 0)).reindex(
    frame.index, fill_value=False).fillna(False)
print(f"\n  cell fires today: {bool(mask.loc[ASOF])}")

variants = {
    "1d<=-3%": ((r1 <= -0.03) & (dd >= DD_MIN) & (r252 > 0)),
    "1d<=-5%": ((r1 <= -0.05) & (dd >= DD_MIN) & (r252 > 0)),
    "dd>=20%": ((r1 <= BREAK) & (dd >= 0.20) & (r252 > 0)),
    "dd>=35%": ((r1 <= BREAK) & (dd >= 0.35) & (r252 > 0)),
    "NO dd gate": ((r1 <= BREAK) & (r252 > 0)),
    "NO 252ret gate": ((r1 <= BREAK) & (dd >= DD_MIN)),
    "BREAK ALONE": (r1 <= BREAK),
    "COMPLEMENT dd<25 (break+strong yr)": ((r1 <= BREAK) & (dd < DD_MIN) & (r252 > 0)),
    "COMPLEMENT 252ret<=0 (break+deep dd)": ((r1 <= BREAK) & (dd >= DD_MIN) & (r252 <= 0)),
}
variants = {k: v.reindex(frame.index, fill_value=False).fillna(False)
            for k, v in variants.items()}

battery(frame, mask, [("SLV", 1.0)], H, "A9 DEFENDED: SLV break in deep dd, "
        "strong year", COST_BPS, variants=variants,
        event_kinds=("cpi", "fomc_decision"))

sig = frame.index[mask.values]
print("\n  horizon scan:")
show(horizon_scan(frame, sig, [("SLV", 1.0)], hs=(1, 2, 3, 5, 7, 10)),
     "SLV horizon scan")

# ---------------------------------------------------------------------------
# gate attribution, stated explicitly at several horizons
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("GATE ATTRIBUTION (day level AND episode level, several horizons)")
for h in (1, 3, 5, 10):
    f = fwd_lag(s, h, 1)
    val = f.dropna().index
    print(f"\n  h={h}: all days {100*f.loc[val].mean():+.3f}% "
          f"(N={len(val)}, hit {100*(f.loc[val]>0).mean():.1f}%)")
    for lbl, m in [("DEFENDED join", mask),
                   ("break alone", (r1 <= BREAK)),
                   ("break & dd>=25 (no yr gate)", (r1 <= BREAK) & (dd >= DD_MIN)),
                   ("break & yr>0 (no dd gate)", (r1 <= BREAK) & (r252 > 0)),
                   ("DISCARDED: break & dd<25 & yr>0",
                    (r1 <= BREAK) & (dd < DD_MIN) & (r252 > 0)),
                   ("DISCARDED: break & dd>=25 & yr<=0",
                    (r1 <= BREAK) & (dd >= DD_MIN) & (r252 <= 0))]:
        mm = m.reindex(val, fill_value=False).fillna(False)
        v = f.loc[val][mm.values]
        if len(v) == 0:
            print(f"    {lbl:<34} N=0")
            continue
        ep = declusters(v.index, h, val)
        print(f"    {lbl:<34} N={len(v):>4} mean={100*v.mean():+.3f}% "
              f"hit={100*(v>0).mean():>5.1f}%  epi N={len(ep):>3} "
              f"mean={100*f.loc[ep].mean():+.3f}%")

# ---------------------------------------------------------------------------
# dose response on each conditioner (is the carve monotone or U-shaped?)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("DOSE RESPONSE inside the break cell (h=5, day level)")
f5 = fwd_lag(s, 5, 1)
val5 = f5.dropna().index
brk = (r1 <= BREAK).reindex(val5, fill_value=False).fillna(False)
ddv = dd.reindex(val5)
yrv = r252.reindex(val5)
rows = []
for lo, hi in [(0, .10), (.10, .20), (.20, .30), (.30, .40), (.40, 1.0)]:
    m = brk & (ddv >= lo) & (ddv < hi)
    v = f5.loc[val5][m.fillna(False).values]
    rows.append(summarize(v.values, f"break & dd in [{100*lo:.0f},{100*hi:.0f})%"))
show(rows, "drawdown-depth dose response")
rows = []
for lo, hi in [(-10, -.20), (-.20, 0), (0, .20), (.20, .50), (.50, 10)]:
    m = brk & (yrv >= lo) & (yrv < hi)
    v = f5.loc[val5][m.fillna(False).values]
    rows.append(summarize(v.values, f"break & 252ret in [{100*lo:.0f},{100*hi:.0f})%"))
show(rows, "trailing-year dose response")

# ---------------------------------------------------------------------------
# GLD beta: is the whole thing a gold trade?
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("GLD BETA on the defended windows (h=5, lag=1)")
fg = fwd_lag(close["GLD"], 5, 1)
common = f5.dropna().index.intersection(fg.dropna().index)
ms = mask.reindex(common, fill_value=False).fillna(False)
y = f5.loc[common][ms.values].values
x = fg.loc[common][ms.values].values
ya, xa = f5.loc[common].values, fg.loc[common].values
b_all = np.polyfit(xa, ya, 1)
print(f"  all-days OLS SLV_fwd = {b_all[1]*100:+.3f}% + {b_all[0]:.3f} * GLD_fwd")
if len(y) > 2:
    resid = y - (b_all[0] * x + b_all[1])
    print(f"  cell N={len(y)}  SLV fwd {100*y.mean():+.3f}%  "
          f"GLD fwd {100*x.mean():+.3f}%  "
          f"beta-charged alpha {100*resid.mean():+.3f}% "
          f"t={resid.mean()/(resid.std(ddof=1)/np.sqrt(len(resid))):+.2f} "
          f"record {(resid>0).sum()}-{(resid<=0).sum()}")

# ---------------------------------------------------------------------------
# registry adjacency 1: the 2026-09-10 "long SLV" continuation cell
# registry adjacency 2: watchlist 29's SHORT at h=1 on today's exact state
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("REGISTRY ADJACENCY")
gld, gdx = close["GLD"], close["GDX"]
g1, x1 = gld.pct_change(), gdx.pct_change()
zg = zscore(gld, 10)
# W29: whole metals complex breaks together on the same session
comp_break = ((r1 < 0) & (g1 < 0) & (x1 < 0) &
              (r1 <= -0.02))
print(f"  W29 complex-break state live today: "
      f"{bool(comp_break.reindex([ASOF]).fillna(False).iloc[0])}")
for h in (1, 3, 5):
    f = fwd_lag(s, h, 1)
    val = f.dropna().index
    for lbl, m in [("complex break (SLV<=-2%, GLD<0, GDX<0)", comp_break),
                   ("SLV break alone <=-2%", (r1 <= -0.02)),
                   ("TODAY'S JOIN: complex break AND dd>=25 AND yr>0",
                    comp_break & (dd >= DD_MIN) & (r252 > 0))]:
        mm = m.reindex(val, fill_value=False).fillna(False)
        v = f.loc[val][mm.values]
        if len(v) < 2:
            print(f"    h={h} {lbl:<50} N={len(v)}")
            continue
        ep = declusters(v.index, max(h, 5), val)
        w = int((v > 0).sum())
        print(f"    h={h} {lbl:<50} N={len(v):>4} LONG mean="
              f"{100*v.mean():+.3f}% hit={100*(v>0).mean():.1f}% "
              f"(SHORT = {-100*v.mean():+.3f}%) epi={len(ep)} "
              f"epi_mean={100*f.loc[ep].mean():+.3f}% sign_p={sign_test(w, len(v)):.4f}")

# ---------------------------------------------------------------------------
# reference class: identical rule across the commodity/metal family
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("REFERENCE CLASS - identical rule on every cached commodity vehicle")
res = []
for t in ["SLV", "GLD", "GDX", "USO", "DBC", "XME", "PPLT", "CPER"]:
    if t not in close:
        continue
    st, rr1, ddt, rr252 = state(t)
    f = fwd_lag(st, H, 1)
    val = f.dropna().index
    m = ((rr1 <= BREAK) & (ddt >= DD_MIN) & (rr252 > 0)).reindex(
        val, fill_value=False).fillna(False)
    v = f.loc[val][m.values]
    if len(v) < 2:
        res.append((t, len(v), 0, np.nan, np.nan, np.nan))
        continue
    ep = declusters(v.index, H, val)
    e = f.loc[ep].values
    drift = f.loc[val].mean()
    res.append((t, len(v), len(ep), 100 * e.mean(), 100 * (e.mean() - drift),
                e.mean() / (e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 1 else np.nan))
rd = pd.DataFrame(res, columns=["ticker", "n_days", "n_epi", "epi_mean_pct",
                                "excess_pct", "t"])
print(rd.round(3).to_string(index=False))
ok = rd.dropna(subset=["excess_pct"])
if len(ok) > 1 and "SLV" in set(ok["ticker"]):
    d = float(ok.set_index("ticker").loc["SLV", "excess_pct"])
    print(f"  SLV excess {d:+.3f}% ranks "
          f"{int((ok['excess_pct'] > d).sum()) + 1} of {len(ok)}")
