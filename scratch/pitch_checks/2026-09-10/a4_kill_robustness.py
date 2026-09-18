"""a4 — kill robustness. Three questions the round-1 scripts leave open:

1. Are the A1 / A2 kills an artifact of the h=5 I chose? Scan h=1,2,3,5,10.
   This grid is CHARGED TO THE CHECKER: it is a search for a surviving
   horizon, and nothing here is used to defend a cell.
2. Is A2's trigger actually live today? The candidate asserts the XLE-XLY
   21d spread is at/above the trailing-252 95th percentile.
3. A3's discarded complement (the "still falling" laggard) — quantify it for
   the watchlist, and check whether it is live on SMH today.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change  # noqa

SPDR = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU",
        "XLV", "XLY"]
T = SPDR + ["^TNX", "IYR", "VNQ", "SPY", "USO", "SMH"]
raw = close_panel(T)
CAL = raw["SPY"].dropna().index
px = raw.reindex(CAL)


def epi(mask, ret, h):
    d = CAL[mask.reindex(CAL, fill_value=False).values & ret.notna().values]
    return declusters(d, h, CAL)


def cell(long_t, short_t, mask, h):
    r = vehicle_ret(px, [(long_t, 1.0), (short_t, -1.0)], h, 1)
    e = epi(mask, r, h)
    v = r.loc[e].values
    if len(v) == 0:
        return {"h": h, "n": 0}
    base = r[r.notna()].values
    w = int((v > 0).sum())
    return {"h": h, "n": len(v), "mean_pct": round(100 * v.mean(), 3),
            "all_days_pct": round(100 * base.mean(), 3),
            "edge_pct": round(100 * (v.mean() - base.mean()), 3),
            "hit": round(100 * (v > 0).mean(), 1),
            "t": round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2),
            "record": f"{w}-{len(v)-w}",
            "sign_p": round(sign_test(w, len(v)), 4)}


# ---------------------------------------------------------------- A1
tnx = px["^TNX"]
hi = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
TNX_HI = (tnx >= hi - 1e-12) & tnx.notna() & hi.notna()
print("=== A1 horizon robustness — LONG(real estate) / SHORT XLF at "
      "^TNX 252d high AND r63<=10 (CHECKER grid, charged) ===")
for proxy in ("XLRE", "IYR", "VNQ"):
    m = TNX_HI & (pct_rank(px[proxy], 63) <= 10)
    rows = [cell(proxy, "XLF", m, h) for h in (1, 2, 3, 5, 10, 21)]
    print(f"\n LONG {proxy} / SHORT XLF")
    print(pd.DataFrame(rows).to_string(index=False))
    pos = [r for r in rows if r.get("n", 0) > 2 and r.get("mean_pct", 0) > 0]
    print(f"   horizons with a positive mean: {len(pos)}/6")

# ---------------------------------------------------------------- A2
R21 = {t: _valid_pct_change(px[t], 21) for t in SPDR + ["USO"]}
sp = R21["XLE"] - R21["XLY"]
rk = rolling_on_valid(sp, lambda x: x.rolling(252).rank(pct=True) * 100.0)
print("\n\n=== A2 trigger liveness (the candidate asserts >= 95th pctile) ===")
tail = pd.DataFrame({"spread_pp": (100 * sp).round(2), "pctile": rk.round(1)}).tail(8)
print(tail.to_string())
hi252 = rolling_on_valid(sp, lambda x: x.rolling(252).max())
q95 = rolling_on_valid(sp, lambda x: x.rolling(252).quantile(0.95))
print(f"  today spread {100*sp.iloc[-1]:+.2f}pp   trailing-252 95th pct level "
      f"{100*q95.iloc[-1]:+.2f}pp   trailing-252 max {100*hi252.iloc[-1]:+.2f}pp")
print(f"  TRIGGER FIRES TODAY: {bool(rk.iloc[-1] >= 95)}  "
      f"(needs +{100*(q95.iloc[-1]-sp.iloc[-1]):.2f}pp more spread)")

MASK2 = (rk >= 95) & sp.notna()
print("\n=== A2 horizon robustness — LONG XLY / SHORT XLE (CHECKER grid, charged) ===")
print(pd.DataFrame([cell("XLY", "XLE", MASK2, h)
                    for h in (1, 2, 3, 5, 10, 21)]).to_string(index=False))
print("\n  same, momentum direction (LONG XLE / SHORT XLY):")
print(pd.DataFrame([{**cell("XLE", "XLY", MASK2, h)} for h in (1, 2, 3, 5, 10, 21)
                    ]).to_string(index=False))

# A2 at the LIVE reading's own threshold (88th pctile) — does the idea exist
# at the level that is actually live? Charged: this is a threshold the
# checker moved to fit today.
print("\n  A2 at the pctile that is actually live today (>=88), h=5, CHARGED:")
print(pd.DataFrame([cell("XLY", "XLE", (rk >= 88) & sp.notna(), h)
                    for h in (3, 5, 10)]).to_string(index=False))

# ---------------------------------------------------------------- A3 complement
UNIV = ["SPY", "QQQ", "IWM", "DIA"] + SPDR + [
    "SMH", "XBI", "IBB", "ITA", "IHI", "ITB", "XHB", "XRT", "XME", "XOP",
    "OIH", "KRE", "IYR", "GDX", "VNQ", "EFA", "EEM"]
raw2 = close_panel(UNIV)
CAL2 = raw2["SPY"].dropna().index
px2 = raw2.reindex(CAL2)
print("\n\n=== A3 discarded complement: 'the laggard that is STILL falling' ===")
for h in (5, 10, 21):
    vs, ds, vj, dj = [], [], [], []
    for t in UNIV:
        r = fwd_lag(px2[t], h)
        r63 = pct_rank(px2[t], 63) <= 10
        r5 = pct_rank(px2[t], 5) >= 75
        ok = r.notna().values
        mc = r63.reindex(CAL2, fill_value=False).values & ~r5.reindex(CAL2, fill_value=False).values & ok
        mj = r63.reindex(CAL2, fill_value=False).values & r5.reindex(CAL2, fill_value=False).values & ok
        vs.append(r.values[mc]); ds.extend(list(CAL2[mc]))
        vj.append(r.values[mj]); dj.extend(list(CAL2[mj]))
    vs, vj = np.concatenate(vs), np.concatenate(vj)
    sc = pd.Series(vs).groupby(pd.DatetimeIndex(ds).values).mean()
    sj = pd.Series(vj).groupby(pd.DatetimeIndex(dj).values).mean()
    allv = np.concatenate([fwd_lag(px2[t], h).dropna().values for t in UNIV])
    print(f"  h={h}: COMPLEMENT (r63<=10 & r5<75) dcl mean "
          f"{100*sc.mean():+.3f}% t {sc.mean()/(sc.std(ddof=1)/np.sqrt(len(sc))):+.2f} "
          f"on {len(sc)} dates  |  JOIN (candidate) dcl mean {100*sj.mean():+.3f}% "
          f"t {sj.mean()/(sj.std(ddof=1)/np.sqrt(len(sj))):+.2f} on {len(sj)} dates "
          f"|  pool all-days {100*allv.mean():+.3f}%")
print(f"\n  SMH today: r63={pct_rank(px2['SMH'],63).iloc[-1]:.1f} "
      f"r5={pct_rank(px2['SMH'],5).iloc[-1]:.1f} -> complement fires: "
      f"{bool(pct_rank(px2['SMH'],63).iloc[-1] <= 10 and pct_rank(px2['SMH'],5).iloc[-1] < 75)}")
