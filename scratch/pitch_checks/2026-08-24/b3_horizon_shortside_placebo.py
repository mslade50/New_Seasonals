"""b3 — close both candidates over their FULL stated horizon range (1-10 td),
check the short side (so nobody pitches the inverse off these kills), and run
the alphabetical-selection placebo on C7's "or the complex" basket form.

C3 as stated: long the copper complex on a 5d thrust to a fresh 52w high, 1-10 td.
C7 as stated: k>=5 of the energy complex at z10>=2, long XLE or the complex, 1-10 td.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, declusters, horizon_scan, load_prices, rolling_on_valid,
    show, sign_test, summarize, vehicle_ret,
)
from pitch_lab import _valid_pct_change as vpc  # noqa: E402

pd.set_option("display.width", 240)
HS = (1, 2, 3, 4, 5, 7, 10)


def tape_z10(close: pd.Series, n: int = 10) -> pd.Series:
    r = close.pct_change(n)
    v = close.pct_change().rolling(21).std()
    return r / (v * np.sqrt(n))


# ---------------------------------------------------------------------------
print("=== C3: horizon scan 1-10, FCX leg, r5>=15% into a fresh 52w high ===")
EQ = ["FCX", "COPX", "XME", "XLB", "SCCO", "TECK", "SPY"]
px = close_panel(EQ)
r5f = vpc(px["FCX"], 5)
hif = rolling_on_valid(px["FCX"], lambda x: x.rolling(252).max())
m15 = ((r5f >= 0.15) & (px["FCX"] >= hif * (1 - 1e-9))).fillna(False)
m10 = ((r5f >= 0.10) & (px["FCX"] >= hif * (1 - 1e-9))).fillna(False)
d15 = px.index[m15.values]
d10 = px.index[m10.values]
show(horizon_scan(px, d15, [("FCX", 1.0)], HS, min_gap=10),
     "C3 r5>=15% & fresh high -> long FCX (episodes, 10td gap)")
show(horizon_scan(px, d10, [("FCX", 1.0)], HS, min_gap=10),
     "C3 r5>=10% & fresh high -> long FCX (the loosest non-negative variant)")
show(horizon_scan(px, d15, [("XME", 1.0)], HS, min_gap=10),
     "C3 same trigger -> long XME (the complex ETF)")

print("\n  short side of the C3 cell (is the negative tradeable?):")
for h in (3, 5, 10):
    ret = vehicle_ret(px, [("FCX", -1.0)], h)
    epi = declusters(px.index[m15.values & ret.notna().values], 10, px.index)
    v = ret.loc[epi].values
    w = int((v > 0).sum())
    print(f"   SHORT FCX h={h:2d}: N={len(v)} mean {100*v.mean():+.3f}% "
          f"record {w}-{len(v)-w} sign p {sign_test(w, len(v)):.4f} "
          f"worst {100*v.min():+.2f}%")

# ---------------------------------------------------------------------------
print("\n=== C7: horizon scan 1-10, count(z10>=2)>=5 ===")
COMPLEX = ["XLE", "XOP", "USO", "COP", "CVX", "VLO", "OXY", "SLB", "EOG",
           "HAL", "WMB"]
raw = load_prices(sorted(set(COMPLEX + ["SPY"])))
pan = close_panel(sorted(set(COMPLEX + ["SPY"]))).dropna(subset=["XLE", "SPY"])
IDX = pan.index
z = pd.DataFrame({t: tape_z10(raw[t]["Close"]) for t in COMPLEX}).reindex(IDX)
allv = z.notna().all(axis=1)
cnt = (z >= 2.0).sum(axis=1).where(allv)
TRIG = (cnt >= 5).fillna(False)
dT = IDX[TRIG.values]
show(horizon_scan(pan, dT, [("XLE", 1.0)], HS, min_gap=10),
     "C7 count>=5 -> long XLE (episodes, 10td gap)")
show(horizon_scan(pan, dT, [("XOP", 1.0)], HS, min_gap=10),
     "C7 count>=5 -> long XOP")
eqw = [(t, 1.0 / 9) for t in COMPLEX if t not in ("XLE", "XOP")]
show(horizon_scan(pan, dT, eqw, HS, min_gap=10),
     "C7 count>=5 -> equal-weight the 9 non-ETF members ('or the complex')")
d23 = IDX[((cnt >= 2) & (cnt <= 3)).fillna(False).values]
show(horizon_scan(pan, d23, [("XLE", 1.0)], HS, min_gap=10),
     "contrast: count in [2,3], the band that IS positive (does NOT fire today)")

print("\n  short side of the C7 cell:")
for h in (3, 5, 10):
    ret = vehicle_ret(pan, [("XLE", -1.0)], h)
    epi = declusters(IDX[TRIG.values & ret.notna().values], 10, IDX)
    v = ret.loc[epi].values
    w = int((v > 0).sum())
    print(f"   SHORT XLE h={h:2d}: N={len(v)} mean {100*v.mean():+.3f}% "
          f"record {w}-{len(v)-w} sign p {sign_test(w, len(v)):.4f} "
          f"worst {100*v.min():+.2f}%")

# ---------------------------------------------------------------------------
print("\n=== C7 basket form: the alphabetical-selection placebo ===")
print("  on each trigger day, long the k FIRING names equally vs the k")
print("  ALPHABETICALLY-FIRST complex members, h=5, market-relative to SPY.")
H = 5
fwd = {t: (pan[t].shift(-(1 + H)) / pan[t].shift(-1) - 1.0) for t in COMPLEX}
fspy = pan["SPY"].shift(-(1 + H)) / pan["SPY"].shift(-1) - 1.0
alpha_order = sorted(COMPLEX)
epi = declusters(dT, 10, IDX)
sel, plac, k_used = [], [], []
for d in epi:
    fire = [t for t in COMPLEX if z.loc[d, t] >= 2.0]
    if not fire:
        continue
    k = len(fire)
    alt = alpha_order[:k]
    a = np.nanmean([fwd[t].loc[d] for t in fire])
    b = np.nanmean([fwd[t].loc[d] for t in alt])
    s = fspy.loc[d]
    if np.isnan(a) or np.isnan(b) or np.isnan(s):
        continue
    sel.append(a - s)
    plac.append(b - s)
    k_used.append(k)
sel, plac = np.array(sel), np.array(plac)
show([summarize(sel, f"the k FIRING names (N={len(sel)}, mean k={np.mean(k_used):.1f})"),
      summarize(plac, "the k ALPHABETICALLY-FIRST members"),
      summarize(sel - plac, "difference (firing minus alphabetical)")],
     "market-relative, h=5, episodes")
w = int((sel - plac > 0).sum())
print(f"  firing beats alphabetical on {w} of {len(sel)} episodes, "
      f"sign p = {sign_test(w, len(sel)):.4f}")
print(f"  alphabetically-first {alpha_order[:6]}")
