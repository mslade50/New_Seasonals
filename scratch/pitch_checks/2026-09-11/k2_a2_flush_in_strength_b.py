"""A2 round 2 - the only positive rung in the ladder, charged properly.

Round 1 result: the PRE-SPECIFIED cell (r5<=5 & r63>=85) is dead. Pooled it
pays -0.069% against the pool's all-days +0.218%, the parent r5<=5 alone pays
+0.353% and the DISCARDED COMPLEMENT (r5<=5 & r63<85) pays +0.385% at t 3.33 —
the 63-day gate is an anti-filter, the third instance of that registry shape.
On IBB alone the defended cell is 4 episodes, all pre-2018, at -1.186%.

BUT the ladder threw up one positive cell, r5<=2 & r63>=90 at +1.608% (N=34
obs / 29 dates, date-clustered t 1.84), and IBB's live reading (r5 0.4,
r63 91.3) sits INSIDE it. That is a post-hoc rescue found by walking a
4x3 grid, so this script charges it:
  - per-member breakdown, era split, concentration, decluster
  - gate attribution at the r5<=2 rung specifically
  - the generic cross-sectional-reversal control on the same dates
  - a charged max-of-K permutation against the DEFENDED cell over the
    12-cell ladder that produced it
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-09-10")
FAMILY = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLE", "XLU", "XLB",
          "SMH", "IBB", "XBI", "IHI", "KRE", "ITA", "XME", "XRT", "XHB",
          "IYR", "OIH"]
H = 5
R5_THR, R63_THR = 2, 90

px = load_prices(FAMILY)
panel = pd.DataFrame({t: px[t]["Close"] for t in FAMILY})
R5 = pd.DataFrame({t: pct_rank(panel[t], 5) for t in FAMILY})
R63 = pd.DataFrame({t: pct_rank(panel[t], 63) for t in FAMILY})
FW = {t: fwd_lag(panel[t].dropna(), H, 1) for t in FAMILY}


def pull(mask_fn):
    d, v, k = [], [], []
    for t in FAMILY:
        f = FW[t]
        idx = f.dropna().index
        m = mask_fn(t).reindex(idx, fill_value=False).fillna(False)
        dd = idx[m.values]
        d.extend(list(dd)); v.extend(list(f.loc[dd].values)); k.extend([t] * len(dd))
    return pd.DataFrame({"ticker": k, "date": pd.DatetimeIndex(d), "ret": v})


def rep(df, label):
    if df.empty:
        print(f"  {label:<44} EMPTY")
        return
    daily = df.groupby("date")["ret"].mean()
    t = daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily))) if len(daily) > 1 else np.nan
    w = int((df["ret"] > 0).sum())
    # sign_test's exact-Fraction p=0.5 path is O(n) big-integer binomials; only
    # meaningful (and only affordable) on the small conditional cells.
    sp = sign_test(w, len(df)) if len(df) <= 400 else float("nan")
    print(f"  {label:<44} N={len(df):>5} dates={len(daily):>4} "
          f"mean={100*df['ret'].mean():+.3f}% hit={100*(df['ret']>0).mean():>5.1f}% "
          f"t_date={t:+.2f} signp={sp:.4f}")


def mk(r5=None, r63lo=None, r63hi=None):
    def f(t):
        m = pd.Series(True, index=R5.index)
        if r5 is not None:
            m &= (R5[t] <= r5)
        if r63lo is not None:
            m &= (R63[t] >= r63lo)
        if r63hi is not None:
            m &= (R63[t] < r63hi)
        return m.fillna(False)
    return f


print("=" * 78)
print(f"1. THE RESCUED RUNG r5<={R5_THR} & r63>={R63_THR}, h={H}, lag=1")
allp = pull(lambda t: pd.Series(True, index=R5.index))
rep(allp, "pool all days")
cell = pull(mk(R5_THR, R63_THR))
rep(cell, f"RESCUED r5<={R5_THR} & r63>={R63_THR}")
print("\n  --- gate attribution at the r5<=2 rung ---")
rep(pull(mk(R5_THR)), f"r5<={R5_THR} alone (parent)")
rep(pull(mk(R5_THR, None, R63_THR)), f"DISCARDED r5<={R5_THR} & r63<{R63_THR}")
for lo, hi in [(0, 25), (25, 50), (50, 75), (75, 90), (90, 101)]:
    rep(pull(mk(R5_THR, lo, hi)), f"   r5<={R5_THR} & r63 in [{lo},{hi})")

print("\n2. PER-MEMBER BREAKDOWN of the rescued cell")
g = cell.groupby("ticker")["ret"]
bd = pd.DataFrame({"n": g.size(), "mean_pct": 100 * g.mean(),
                   "hit": 100 * g.apply(lambda x: (x > 0).mean())})
bd["drift_pct"] = [100 * FW[t].mean() for t in bd.index]
bd["excess_pct"] = bd["mean_pct"] - bd["drift_pct"]
print(bd.round(3).sort_values("n", ascending=False).to_string())
print(f"  distinct members firing: {len(bd)} of {len(FAMILY)}; "
      f"largest member share {100*bd['n'].max()/bd['n'].sum():.1f}%")
print(f"  members with positive excess: {int((bd['excess_pct']>0).sum())} of {len(bd)}")

print("\n3. ERA + CONCENTRATION (date-averaged episodes)")
daily = cell.groupby("date")["ret"].mean().sort_index()
pre = daily[daily.index < "2018-01-01"]
post = daily[daily.index >= "2018-01-01"]
show([summarize(daily.values, "all dates"),
      summarize(pre.values, "pre-2018"),
      summarize(post.values, "2018+")], "era split on date-averaged returns")
print("  " + cluster_note(daily.index, daily.values, k=2))
print(f"  by year: {dict((y, round(100*v,2)) for y, v in daily.groupby(daily.index.year).sum().items())}")
ordv = np.sort(daily.values)
print(f"  drop-best-2 mean = {100*ordv[:-2].mean():+.3f}% (n={len(ordv)-2}); "
      f"drop-worst-2 mean = {100*ordv[2:].mean():+.3f}%")
ep = declusters(daily.index, H, pd.DatetimeIndex(sorted(allp['date'].unique())))
print(f"  declustered (gap {H} td): N={len(ep)} "
      f"mean={100*daily.loc[ep].mean():+.3f}%")

print("\n4. GENERIC CROSS-SECTIONAL REVERSAL on the same dates")
r5_raw = pd.DataFrame({t: panel[t] / panel[t].shift(5) - 1.0 for t in FAMILY})
fwd_all = pd.DataFrame({t: FW[t] for t in FAMILY})
gen, defd = [], []
for d in daily.index:
    if d not in r5_raw.index:
        continue
    row = r5_raw.loc[d].dropna(); fr = fwd_all.loc[d].dropna()
    common = row.index.intersection(fr.index)
    if len(common) < 10:
        continue
    gen.append(fr[row[common].idxmin()]); defd.append(daily.loc[d])
gen, defd = np.asarray(gen, float), np.asarray(defd, float)
show([summarize(defd, "RESCUED cell (date-averaged)"),
      summarize(gen, "generic worst-5d-in-family")], "generic control")
if len(gen) > 2:
    dif = defd - gen
    print(f"  paired diff {100*dif.mean():+.3f}% "
          f"t={dif.mean()/(dif.std(ddof=1)/np.sqrt(len(dif))):+.2f} "
          f"record {(dif>0).sum()}-{(dif<=0).sum()}")

print("\n5. HORIZON PROFILE of the rescued cell (date-averaged)")
for h in (1, 2, 3, 5, 7, 10):
    fwh = {t: fwd_lag(panel[t].dropna(), h, 1) for t in FAMILY}
    d, v = [], []
    for t in FAMILY:
        idx = fwh[t].dropna().index
        m = mk(R5_THR, R63_THR)(t).reindex(idx, fill_value=False).fillna(False)
        dd = idx[m.values]
        d.extend(list(dd)); v.extend(list(fwh[t].loc[dd].values))
    s = pd.Series(v, index=pd.DatetimeIndex(d)).groupby(level=0).mean()
    allv = np.concatenate([fwh[t].dropna().values for t in FAMILY])
    print(f"  h={h:>2}: dates={len(s):>3} mean={100*s.mean():+.3f}% "
          f"hit={100*(s>0).mean():>5.1f}% t={s.mean()/(s.std(ddof=1)/np.sqrt(len(s))):+.2f} "
          f"pool all-days={100*allv.mean():+.3f}% edge={100*(s.mean()-allv.mean()):+.3f}pp")

print("\n6. CHARGED MAX-OF-K over the 4x3 ladder that produced this cell")
grid = [(a, b) for a in (2, 5, 10, 15) for b in (75, 85, 90)]
pool_series = {t: FW[t].dropna() for t in FAMILY}
masks = {}
for a, b in grid:
    mm = {}
    for t in FAMILY:
        idx = pool_series[t].index
        mm[t] = mk(a, b)(t).reindex(idx, fill_value=False).fillna(False).values
    masks[(a, b)] = mm
defended = float(daily.mean() - np.concatenate(
    [pool_series[t].values for t in FAMILY]).mean())
rng = np.random.default_rng(2)
POOL_ARR = {t: pool_series[t].values for t in FAMILY}
DRIFT = np.concatenate([POOL_ARR[t] for t in FAMILY]).mean()
COUNTS = {k: {t: int(mm[t].sum()) for t in FAMILY} for k, mm in masks.items()}
nulls = []
for _ in range(1500):
    best = -np.inf
    for key, cn in COUNTS.items():
        tot, num = 0.0, 0
        for t in FAMILY:
            n = cn[t]
            if n:
                tot += POOL_ARR[t][rng.integers(0, len(POOL_ARR[t]), size=n)].sum()
                num += n
        if num:
            best = max(best, tot / num - DRIFT)
    nulls.append(best)
nulls = np.asarray(nulls)
print(f"  defended excess (date-averaged vs pool drift) = {100*defended:+.3f}%")
print(f"  null-max median = {100*np.median(nulls):+.3f}%   "
      f"P(null max >= defended) = {float((nulls >= defended).mean()):.4f}")

print("\n7. LIVE INSTANCE CONTEXT: how often does IBB itself fire this rung?")
m_ibb = mk(R5_THR, R63_THR)("IBB")
d_ibb = R5.index[m_ibb.reindex(R5.index, fill_value=False).values]
print(f"  IBB fires r5<={R5_THR} & r63>={R63_THR} on "
      f"{len(d_ibb)} days: {[str(x.date()) for x in d_ibb]}")
f = FW["IBB"]
for d in d_ibb:
    if d in f.index and not np.isnan(f.loc[d]):
        print(f"    {d.date()}  fwd{H} = {100*f.loc[d]:+.2f}%")
