"""A2 round 3 - finishing the rescued rung r5<=2 & r63>=90.

Round 2 left it alive on the numbers I had run: 34 obs / 29 dates, +1.608%
day-level at a 70.6% hit, date-clustered t 1.84, sign p 0.0122, gate
attribution clean AT THAT RUNG (parent r5<=2 alone +0.247%, discarded
complement +0.251%), positive in both eras, charged P 0.0120 over the 12-cell
threshold ladder.

Four things it has NOT yet paid:
 1. the charge over the FULL walk - the horizon was also selected (the profile
    peaks exactly at h=5), so K is 12 thresholds x 6 horizons, not 12.
 2. member robustness - KRE alone is +8.920% on n=4. Leave-one-member-out.
 3. definition fragility on the RANKING WINDOW (252 is a choice; 126 and 504
    are equally defensible) and on the return window (5 and 63 likewise).
 4. what the LIVE instrument's own record is - IBB has fired this rung 4 times
    ever, 3 of them consecutive sessions in June 2003.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

FAMILY = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLE", "XLU", "XLB",
          "SMH", "IBB", "XBI", "IHI", "KRE", "ITA", "XME", "XRT", "XHB",
          "IYR", "OIH"]
px = load_prices(FAMILY)
panel = pd.DataFrame({t: px[t]["Close"] for t in FAMILY})


def ranks(n, lb):
    return pd.DataFrame({t: pct_rank(panel[t], n, lb) for t in FAMILY})


R5 = ranks(5, 252)
R63 = ranks(63, 252)


def cell_dates(R5d, R63d, a, b):
    out = {}
    for t in FAMILY:
        m = ((R5d[t] <= a) & (R63d[t] >= b)).fillna(False)
        out[t] = R5d.index[m.values]
    return out


def score(dates_by_t, h, drop=None):
    """date-averaged mean and the pool drift at that horizon."""
    fw = {t: fwd_lag(panel[t].dropna(), h, 1) for t in FAMILY}
    d, v = [], []
    for t in FAMILY:
        if drop and t in drop:
            continue
        f = fw[t]
        for x in dates_by_t[t]:
            if x in f.index and not np.isnan(f.loc[x]):
                d.append(x); v.append(float(f.loc[x]))
    if not v:
        return np.nan, np.nan, 0, np.nan
    s = pd.Series(v, index=pd.DatetimeIndex(d)).groupby(level=0).mean()
    keep = [t for t in FAMILY if not (drop and t in drop)]
    allv = np.concatenate([fw[t].dropna().values for t in keep])
    t_ = s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else np.nan
    return s.mean() - allv.mean(), s.mean(), len(s), t_


print("=" * 78)
print("1. CHARGED MAX-OF-K over the FULL walk: 12 thresholds x 6 horizons")
GRID = [(a, b) for a in (2, 5, 10, 15) for b in (75, 85, 90)]
HS = (1, 2, 3, 5, 7, 10)
FW = {h: {t: fwd_lag(panel[t].dropna(), h, 1) for t in FAMILY} for h in HS}
POOL = {h: {t: FW[h][t].dropna().values for t in FAMILY} for h in HS}
DRIFT = {h: np.concatenate([POOL[h][t] for t in FAMILY]).mean() for h in HS}
cells = []
for a, b in GRID:
    dbt = cell_dates(R5, R63, a, b)
    for h in HS:
        cn = {}
        for t in FAMILY:
            f = FW[h][t]
            cn[t] = int(sum(1 for x in dbt[t]
                            if x in f.index and not np.isnan(f.loc[x])))
        if sum(cn.values()) >= 5:
            cells.append((h, cn))
defended_exc, defended_mean, nd, td = score(cell_dates(R5, R63, 2, 90), 5)
print(f"  defended cell: dates={nd} mean={100*defended_mean:+.3f}% "
      f"excess={100*defended_exc:+.3f}% t={td:+.2f}")
rng = np.random.default_rng(7)
nulls = []
for _ in range(1500):
    best = -np.inf
    for h, cn in cells:
        tot, num = 0.0, 0
        for t in FAMILY:
            n = cn[t]
            if n:
                arr = POOL[h][t]
                tot += arr[rng.integers(0, len(arr), size=n)].sum(); num += n
        if num:
            best = max(best, tot / num - DRIFT[h])
    nulls.append(best)
nulls = np.asarray(nulls)
print(f"  K={len(cells)} cells; null-max median {100*np.median(nulls):+.3f}%   "
      f"P(null max >= defended) = {float((nulls >= defended_exc).mean()):.4f}")

print("\n" + "=" * 78)
print("2. LEAVE-ONE-MEMBER-OUT on the defended cell (h=5)")
dbt = cell_dates(R5, R63, 2, 90)
rows = []
for t in FAMILY:
    e, m, n, tt = score(dbt, 5, drop={t})
    rows.append((t, n, 100 * m if not np.isnan(m) else np.nan,
                 100 * e if not np.isnan(e) else np.nan, tt))
lo = pd.DataFrame(rows, columns=["dropped", "dates", "mean_pct", "excess_pct", "t"])
print(lo.round(3).sort_values("excess_pct").to_string(index=False))
print(f"  LOMO floor excess = {lo['excess_pct'].min():+.3f}% "
      f"(dropping {lo.loc[lo['excess_pct'].idxmin(), 'dropped']})")

print("\n" + "=" * 78)
print("3. DEFINITION FRAGILITY - ranking lookback and return windows")
for lb in (126, 252, 504):
    for n5, n63 in ((5, 63), (4, 63), (7, 63), (5, 42), (5, 84)):
        A, B = ranks(n5, lb), ranks(n63, lb)
        e, m, n, tt = score(cell_dates(A, B, 2, 90), 5)
        print(f"  lookback={lb:>3} r{n5:<2}<=2 & r{n63:<2}>=90: dates={n:>3} "
              f"mean={100*m:+.3f}% excess={100*e:+.3f}% t={tt:+.2f}"
              if n else f"  lookback={lb} r{n5}/r{n63}: EMPTY")

print("\n" + "=" * 78)
print("4. THE LIVE INSTRUMENT'S OWN RECORD")
f5 = fwd_lag(panel["IBB"].dropna(), 5, 1)
for lbl, a, b in [("r5<=2 & r63>=90", 2, 90), ("r5<=5 & r63>=85", 5, 85),
                  ("r5<=5 (parent)", 5, None)]:
    m = (R5["IBB"] <= a)
    if b is not None:
        m &= (R63["IBB"] >= b)
    d = R5.index[m.fillna(False).values]
    vals = [float(f5.loc[x]) for x in d if x in f5.index and not np.isnan(f5.loc[x])]
    print(f"  IBB {lbl:<18} fires {len(d)} days "
          f"({', '.join(str(x.date()) for x in d[-8:])})"
          f"  fwd5 mean={100*np.mean(vals):+.3f}%" if vals else
          f"  IBB {lbl}: none")

print("\n" + "=" * 78)
print("5. DATE LIST of the defended cell + which member fired")
recs = []
for t in FAMILY:
    for x in dbt[t]:
        f = FW[5][t]
        if x in f.index and not np.isnan(f.loc[x]):
            recs.append((str(x.date()), t, round(100 * float(f.loc[x]), 2)))
rdf = pd.DataFrame(recs, columns=["date", "ticker", "fwd5_pct"]).sort_values("date")
print(rdf.to_string(index=False))
