"""A4 ROUND 2 -- finishing the CPI x ^TNX-252d-high cell.

Round 1 found:
  - 6 anchors at the 0.25% whisker (5 exact), years 2006/2013/2016/2018/2022x2
  - the DISCARDED complement (CPI without the rates gate) pays MORE at every
    lag x horizon except lag=0 h=1
  - the sign flips between lag=0 and lag=1 ON THE SAME SIX ANCHORS
  - top-2 episodes (both 2022) carry 96-120% of every total

Round 2 closes it properly:
  1. leave-one-year-out and drop-2022
  2. the max-of-K charge the UNSPECIFIED DIRECTION earns. The composer handed
     me a cell with no sign, so the search space is
     {2 signs} x {2 lags} x {5 whiskers} x {5 horizons} = 100 cells, and the
     permutation must be run against the CELL BEING DEFENDED, not the max.
  3. reference class across index vehicles
  4. the rates parent alone, declustered properly, as its own object
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

raw = load_prices(["SPY", "^TNX", "QQQ", "IWM", "TLT"])
SP = raw["SPY"].index
PX = pd.DataFrame({t: raw[t]["Close"].reindex(SP).ffill() for t in raw})
PX = PX.rename(columns={"^TNX": "TNX"})
tnx = PX["TNX"]
tnx_dist = (tnx / rolling_on_valid(tnx, lambda x: x.rolling(252).max()) - 1.0) * 100
cpi_days = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(
    load_events(["cpi"])["date"]).normalize()) & set(SP)))
IS_CPI = pd.Series(False, index=SP)
IS_CPI.loc[cpi_days] = True
GATE = (tnx_dist >= -0.25).fillna(False)
A = SP[(IS_CPI & GATE).values]

print("=" * 78)
print("1. LEAVE-ONE-YEAR-OUT AND DROP-2022")
print("=" * 78)
for lag in (0, 1):
    for h in (1, 3, 5):
        r = fwd_lag(PX["SPY"], h, lag)
        v = r.loc[A].dropna()
        parts = [f"ALL {100*v.mean():+.3f}% ({len(v)})"]
        for y in sorted(set(v.index.year)):
            s = v[v.index.year != y]
            parts.append(f"-{y} {100*s.mean():+.3f}%")
        print(f"  lag={lag} h={h}: " + "  ".join(parts))

print("\n" + "=" * 78)
print("2. MAX-OF-K CHARGE FOR THE UNSPECIFIED DIRECTION")
print("=" * 78)
print("  grid: 2 signs x 2 lags x 5 whiskers x 5 horizons = 100 cells")
WH = [-1e-9, -0.10, -0.25, -0.50, -1.00]
HS = [1, 2, 3, 5, 10]
grid = []
for lag in (0, 1):
    for wh in WH:
        m = IS_CPI & (tnx_dist >= wh).fillna(False)
        d = SP[m.values]
        for h in HS:
            r = fwd_lag(PX["SPY"], h, lag)
            v = r.loc[d].dropna()
            c = r.loc[cpi_days].dropna()
            if len(v) < 3:
                continue
            e = v.mean() - c.mean()
            grid.append((lag, wh, h, len(v), 100 * e))
grid.sort(key=lambda x: -abs(x[4]))
print("  top 8 |edge| cells in the grid (edge vs the all-CPI control):")
for lag, wh, h, n, e in grid[:8]:
    print(f"    lag={lag} whisker {wh:+.2f} h={h:2d} n={n} edge {e:+.3f}pp")
DEF_LAG, DEF_WH, DEF_H = 0, -0.25, 5        # the strongest SHORT the grid offers
r = fwd_lag(PX["SPY"], DEF_H, DEF_LAG)
v = r.loc[A].dropna()
obs = -v.mean()                              # SHORT SPY
print(f"\n  DEFENDED CELL (short SPY, lag={DEF_LAG}, whisker -0.25, h={DEF_H}): "
      f"n={len(v)} {100*obs:+.3f}%")
rng = np.random.default_rng(11)
pool = r.loc[cpi_days].dropna()
null_max = []
for _ in range(4000):
    best = 0.0
    for lag in (0, 1):
        for h in HS:
            rr = fwd_lag(PX["SPY"], h, lag).loc[cpi_days].dropna()
            for _wh in WH:
                s = rng.choice(rr.values, size=len(v), replace=False)
                for sgn in (1, -1):
                    best = max(best, sgn * (s.mean() - rr.mean()))
    null_max.append(best)
null_max = np.array(null_max)
edge = obs - (-pool.mean())
print(f"  edge over the all-CPI control = {100*edge:+.3f}pp")
print(f"  null-max median {100*np.median(null_max):+.3f}pp, "
      f"95th {100*np.quantile(null_max, 0.95):+.3f}pp")
print(f"  P(a no-effect grid of this shape produces a best cell >= "
      f"{100*edge:+.3f}pp) = {(null_max >= edge).mean():.4f}")

print("\n" + "=" * 78)
print("3. REFERENCE CLASS ACROSS INDEX VEHICLES (same anchors, same rule)")
print("=" * 78)
for lag in (0, 1):
    for h in (1, 5):
        out = []
        for t in ("SPY", "QQQ", "IWM", "TLT"):
            rr = fwd_lag(PX[t], h, lag)
            vv = rr.loc[A].dropna()
            cc = rr.loc[cpi_days].dropna()
            out.append(f"{t} {100*vv.mean():+.3f}% (edge "
                       f"{100*(vv.mean()-cc.mean()):+.3f})")
        print(f"  lag={lag} h={h}: " + "  |  ".join(out))

print("\n" + "=" * 78)
print("4. THE RATES PARENT AS ITS OWN OBJECT (^TNX at a 252d high, any session)")
print("=" * 78)
pd_days = SP[GATE.values]
for h in (1, 3, 5, 10):
    r = fwd_lag(PX["SPY"], h, 1)
    valid = r.dropna().index
    e = declusters(pd.DatetimeIndex(pd_days).intersection(valid), 10, valid)
    v = r.loc[e].dropna()
    w = int((v > 0).sum())
    b = r.loc[valid].mean()
    print(f"  h={h:2d}: episodes n={len(v)} {100*v.mean():+.3f}% "
          f"edge {100*(v.mean()-b):+.3f}pp hit {100*(v>0).mean():.1f}% "
          f"sign p(>=) {sign_test(w, len(v)):.4f}")
print("  (watchlist 44 reported this parent at roughly -0.169% at h=3; the "
       "composer asked whether the rates leg alone had already been measured.)")

print("\n" + "=" * 78)
print("5. THE LIVE DOSE: is +87 bp / 252 sessions inside the episode range?")
print("=" * 78)
chg = (tnx - tnx.shift(252)) * 100
print(pd.DataFrame({"tnx": tnx.loc[A].round(3),
                    "chg252bp": chg.loc[A].round(1),
                    "spy_h5_lag0": (100 * fwd_lag(PX['SPY'], 5, 0).loc[A]).round(3),
                    "spy_h5_lag1": (100 * fwd_lag(PX['SPY'], 5, 1).loc[A]).round(3)}
                   ).to_string())
print(f"  LIVE 2026-09-10: ^TNX {tnx.iloc[-1]:.3f}, 252-session change "
      f"{chg.iloc[-1]:+.1f} bp")
