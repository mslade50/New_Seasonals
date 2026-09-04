"""C2 -- the same crude thrust taken in CRUDE (USO) rather than the producers.

Whole-variant comparison only (registry rule: no marginal-fill / marginal-name
decompositions). USO, XOP, OIH, an equal-weight producer basket and XLE on the
IDENTICAL trigger days, with USO's roll decay charged explicitly.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403,E402
from pitch_lab import (close_panel, load_prices, fwd_lag, declusters, summarize,
                       sign_test, wilder_atr, show, horizon_scan)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)

PROD = ["COP", "CVX", "VLO", "OXY", "SLB", "EOG", "HAL", "WMB"]
TK = ["USO", "XLE", "XOP", "OIH", "CL=F"] + PROD
px = close_panel(TK)
uso = load_prices(["USO"])["USO"]
uso_1d_own = uso["Close"] / uso["Close"].shift(1) - 1.0
atrpct_own = pd.Series(wilder_atr(uso["High"], uso["Low"], uso["Close"]),
                       index=uso.index) / uso["Close"].shift(1)
uso_1d = uso_1d_own.reindex(px.index)
thrust_atr = (uso_1d_own / atrpct_own).reindex(px.index)

band = (uso_1d >= 0.05) & (uso_1d < 0.06)
armed = band & (thrust_atr >= 1.50)

# ---------------------------------------------------------------------------
# 0. USO ROLL DECAY, charged explicitly
# ---------------------------------------------------------------------------
print("=" * 110)
print("0. USO ROLL DECAY -- USO against front crude (CL=F) over the common span")
print("=" * 110)
both = pd.concat([px["USO"], px["CL=F"]], axis=1).dropna()
both.columns = ["USO", "CL"]
n_yr = (both.index[-1] - both.index[0]).days / 365.25
uso_cagr = (both["USO"].iloc[-1] / both["USO"].iloc[0]) ** (1 / n_yr) - 1
cl_cagr = (both["CL"].iloc[-1] / both["CL"].iloc[0]) ** (1 / n_yr) - 1
print(f"  span {both.index[0].date()} .. {both.index[-1].date()} ({n_yr:.1f}y)")
print(f"  USO CAGR {100*uso_cagr:+.2f}%/yr   CL=F CAGR {100*cl_cagr:+.2f}%/yr   "
      f"drag {100*(uso_cagr-cl_cagr):+.2f}pp/yr = {100*(uso_cagr-cl_cagr)/252*3:+.4f}%/3 sessions")
for t in ("USO", "XLE", "XOP", "OIH"):
    f3 = fwd_lag(px[t], 3, lag=1).dropna()
    print(f"  {t:>5} unconditional h=3 drift {100*f3.mean():+.4f}%  (N={len(f3)})")

# ---------------------------------------------------------------------------
# 1. WHOLE-VARIANT vehicle table on the identical trigger days
# ---------------------------------------------------------------------------
prod_px = px[PROD].dropna(how="all")
basket = pd.Series(np.nan, index=px.index)
_r = px[PROD].pct_change().mean(axis=1, skipna=True)
basket = (1 + _r.fillna(0)).cumprod()   # EW daily-rebalanced producer basket
px2 = px.copy()
px2["PRODEW"] = basket

print("\n" + "=" * 110)
print("1. VEHICLE TABLE -- whole variants, identical trigger days, h=3, lag=1")
print("=" * 110)
for lbl, mask in (("ARMED band+atr>=1.50", armed), ("BAND [5,6)% alone", band)):
    rows = []
    for t in ("USO", "CL=F", "XLE", "XOP", "OIH", "PRODEW"):
        ss = px2[t].dropna()
        f = fwd_lag(ss, 3, lag=1)
        mm = mask.reindex(ss.index).fillna(False)
        e = declusters(ss.index[mm.values], 5, ss.index)
        v = f.reindex(e).dropna()
        if len(v) < 3:
            rows.append({"vehicle": t, "n": len(v)})
            continue
        st = summarize(v.values, t)
        drift = 100 * f.dropna().mean()
        rows.append({"vehicle": t, "n": st["n"],
                     "mean_pct": round(st["mean_pct"], 3),
                     "own_drift_pct": round(drift, 3),
                     "excess_pp": round(st["mean_pct"] - drift, 3),
                     "hit": round(st["hit"], 1),
                     "sd_pct": round(st["sd_pct"], 2),
                     "risk_adj": round(st["mean_pct"] / st["sd_pct"], 3),
                     "worst_pct": round(st["worst_pct"], 2),
                     "signp": round(sign_test(int((v.values > 0).sum()), len(v)), 4)})
    print(f"\n--- {lbl} ---")
    print(pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------------------------------
# 2. vol-matched comparison: scale each vehicle to XLE's sd
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("2. VOL-MATCHED to XLE (scale each vehicle so its unconditional h=3 sd == XLE's)")
print("=" * 110)
xle_sd = fwd_lag(px["XLE"], 3, lag=1).dropna().std(ddof=1)
rows = []
for t in ("USO", "XLE", "XOP", "OIH", "PRODEW"):
    ss = px2[t].dropna()
    f = fwd_lag(ss, 3, lag=1)
    k = xle_sd / f.dropna().std(ddof=1)
    mm = armed.reindex(ss.index).fillna(False)
    e = declusters(ss.index[mm.values], 5, ss.index)
    v = f.reindex(e).dropna() * k
    drift = k * f.dropna().mean()
    st = summarize(v.values, t)
    rows.append({"vehicle": t, "scale": round(k, 3), "n": st["n"],
                 "vm_mean_pct": round(st["mean_pct"], 3),
                 "vm_excess_pp": round(st["mean_pct"] - 100 * drift, 3),
                 "hit": round(st["hit"], 1),
                 "vm_worst_pct": round(st["worst_pct"], 2)})
print(pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------------------------------
# 3. horizon scan per vehicle
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("3. HORIZON SCAN per vehicle, ARMED trigger days")
print("=" * 110)
mm = armed.reindex(px2.index).fillna(False)
trig = px2.index[mm.values]
for t in ("USO", "XLE", "XOP"):
    rows = horizon_scan(px2, trig, [(t, 1.0)], hs=(1, 2, 3, 5, 10), lag=1, min_gap=5)
    for r in rows:
        r["vehicle"] = t
    show(rows, f"{t}")

# ---------------------------------------------------------------------------
# 4. cost
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("4. COST -- assume 4 bps round trip on USO/XLE/XOP (liquid ETFs, ~1c spread),")
print("   8 bps on OIH, 15 bps on the 8-name EW basket (8 round trips).")
print("=" * 110)
for t, c in (("USO", 4.0), ("XLE", 4.0), ("XOP", 4.0), ("OIH", 8.0), ("PRODEW", 15.0)):
    ss = px2[t].dropna()
    f = fwd_lag(ss, 3, lag=1)
    mmm = armed.reindex(ss.index).fillna(False)
    e = declusters(ss.index[mmm.values], 5, ss.index)
    v = f.reindex(e).dropna()
    if len(v) < 3:
        continue
    edge = 100 * 100 * v.mean()
    print(f"  {t:>7}: edge {edge:7.1f} bps vs {c:5.1f} bps cost -> {edge/c:5.1f}x")
