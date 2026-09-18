"""a3 — ADVERSARIAL round 1 on CANDIDATE A3.

"The laggard that has stopped falling": LONG the ETF, no short leg, entry
lag=1 MOC, when its 63d return rank (trailing-252 PIT) <= 10 AND its 5d
return rank (trailing-252 PIT) >= 75.

Pooled over a PRE-DECLARED index-and-industry universe, fixed for the whole
script (the prompt says "29" and lists 32 names; the 32 listed names are
used and named here so the class is auditable).

Order convention (rule 7, fixed and stated): FILTER first, then DECLUSTER
(min_gap = h), per member, then pool.

Cost: single leg. 2 bp round trip on a liquid index ETF, 4 bp on an
SMH-class industry ETF; the 4 bp figure is used for the headline multiple.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change  # noqa

H = 5
COST_BPS = 4.0

UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
    "SMH", "XBI", "IBB", "ITA", "IHI", "ITB", "XHB", "XRT", "XME", "XOP",
    "OIH", "KRE", "IYR", "GDX", "VNQ", "EFA", "EEM",
]
print(f"DECLARED UNIVERSE ({len(UNIVERSE)} names): {' '.join(UNIVERSE)}")

raw = close_panel(UNIVERSE)
CAL = raw["SPY"].dropna().index
px = raw.reindex(CAL)
POS = pd.Series(range(len(CAL)), index=CAL)
SPY_R = {h: fwd_lag(px["SPY"], h) for h in (1, 2, 3, 5, 10)}


def masks(t: str, floor63: float = 10.0, top5: float = 75.0):
    r63 = pct_rank(px[t], 63)
    r5 = pct_rank(px[t], 5)
    return (r63 <= floor63), (r5 >= top5)


def epi(mask: pd.Series, ret: pd.Series, h: int = H) -> pd.DatetimeIndex:
    days = CAL[mask.reindex(CAL, fill_value=False).values & ret.notna().values]
    return declusters(days, h, CAL)


# live reading check
print("\nlive readings (last bar", CAL[-1].date(), "):")
live = []
for t in UNIVERSE:
    m63, m5 = masks(t)
    if bool(m63.iloc[-1]) and bool(m5.iloc[-1]):
        live.append(t)
    if t in ("SMH", "XLRE", "XLI", "XLY"):
        print(f"  {t:5s} r63={pct_rank(px[t],63).iloc[-1]:5.1f} "
              f"r5={pct_rank(px[t],5).iloc[-1]:5.1f} fires={bool(m63.iloc[-1] and m5.iloc[-1])}")
print("  LIVE instances today:", live)


# ==========================================================================
# 1. per-member table + fixed-effect meta-analysis
# ==========================================================================
def member_stats(h: int = H, floor63: float = 10.0, top5: float = 75.0):
    rows, pooled_v, pooled_d, pooled_t, pooled_spy = [], [], [], [], []
    for t in UNIVERSE:
        ret = fwd_lag(px[t], h)
        m63, m5 = masks(t, floor63, top5)
        e = epi(m63 & m5, ret, h)
        if len(e) == 0:
            rows.append({"ticker": t, "n": 0})
            continue
        v = ret.loc[e].values
        span = (CAL >= e[0]) & (CAL <= e[-1])
        base = ret[span & ret.notna().values].values
        exc = v.mean() - base.mean()
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
        sp = SPY_R[h].loc[e].values
        rows.append({
            "ticker": t, "n": len(v),
            "mean_pct": 100 * v.mean(),
            "drift_pct": 100 * base.mean(),
            "excess_pct": 100 * exc,
            "vs_spy_pct": 100 * np.nanmean(v - sp),
            "hit": 100 * (v > 0).mean(),
            "t": v.mean() / se if se and se > 0 else np.nan,
            "se_pct": 100 * se,
            "first": str(e[0].date()), "last": str(e[-1].date()),
        })
        pooled_v.append(v)
        pooled_d.extend(list(e))
        pooled_t.extend([t] * len(v))
        pooled_spy.append(sp)
    return (pd.DataFrame(rows), np.concatenate(pooled_v),
            pd.DatetimeIndex(pooled_d), np.asarray(pooled_t),
            np.concatenate(pooled_spy))


df, V, D, T, SP = member_stats()
print(f"\n--- PER-MEMBER TABLE (h={H}, r63<=10 & r5>=75), FILTER-then-DECLUSTER")
print(df.round(3).to_string(index=False))

ok = df["n"].fillna(0) > 1
ex = df.loc[ok, "excess_pct"].values
se = df.loc[ok, "se_pct"].values
w = 1.0 / se**2
fe = float((w * ex).sum() / w.sum())
fe_se = float(np.sqrt(1.0 / w.sum()))
Q = float((w * (ex - fe) ** 2).sum())
dfree = len(ex) - 1
I2 = max(0.0, 100 * (Q - dfree) / Q) if Q > 0 else 0.0
from scipy import stats as _st  # noqa: E402
print(f"\nFIXED-EFFECT common excess (member return minus own drift): "
      f"{fe:+.3f}%  se {fe_se:.3f}  t {fe/fe_se:+.2f}")
print(f"  Cochran Q = {Q:.1f} on {dfree} df, p = {1 - _st.chi2.cdf(Q, dfree):.3f}, "
      f"I^2 = {I2:.1f}%   (members with n>1: {int(ok.sum())})")
print(f"  members with POSITIVE excess: {int((ex > 0).sum())}/{len(ex)}, "
      f"sign p = {sign_test(int((ex > 0).sum()), len(ex)):.4f}")

# pooled, naive vs date-clustered
print(f"\nPOOLED episodes N={len(V)}  mean {100*V.mean():+.3f}%  "
      f"hit {100*(V>0).mean():.1f}%  naive t {V.mean()/(V.std(ddof=1)/np.sqrt(len(V))):+.2f}")
by_date = pd.Series(V).groupby(D.values).mean()
print(f"  DATE-CLUSTERED (mean per calendar date, N={len(by_date)} dates): "
      f"mean {100*by_date.mean():+.3f}%  t "
      f"{by_date.mean()/(by_date.std(ddof=1)/np.sqrt(len(by_date))):+.2f}   "
      f"record {int((by_date>0).sum())}-{int((by_date<=0).sum())}  sign p "
      f"{sign_test(int((by_date>0).sum()), len(by_date)):.4f}")
print(f"  bootstrap P(mean<=0) on date-clustered = {bootstrap_p_le0(by_date.values):.4f}")
print(f"  pooled vs SPY same window: {100*np.nanmean(V - SP):+.3f}%  "
      f"(N={len(V)})")


# ==========================================================================
# 2. GATE ATTRIBUTION — the likely killer
# ==========================================================================
def pooled_cell(sel, h: int = H) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """sel(t) -> boolean mask; returns pooled DAY-LEVEL values."""
    vs, ds, sps = [], [], []
    for t in UNIVERSE:
        ret = fwd_lag(px[t], h)
        m = sel(t).reindex(CAL, fill_value=False).values & ret.notna().values
        if not m.any():
            continue
        vs.append(ret.values[m])
        ds.extend(list(CAL[m]))
        sps.append(SPY_R[h].values[m])
    return (np.concatenate(vs), pd.DatetimeIndex(ds), np.concatenate(sps))


def dcl(v, d):
    s = pd.Series(v).groupby(pd.DatetimeIndex(d).values).mean()
    tt = s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else np.nan
    return len(s), 100 * s.mean(), tt


cells = {
    "JOIN r63<=10 & r5>=75": lambda t: masks(t)[0] & masks(t)[1],
    "(a) r63<=10 ALONE": lambda t: masks(t)[0],
    "    DISCARDED by the r5 gate (r63<=10 & r5<75)":
        lambda t: masks(t)[0] & ~masks(t)[1],
    "(b) r5>=75 ALONE": lambda t: masks(t)[1],
    "    DISCARDED by the r63 gate (r5>=75 & r63>10)":
        lambda t: masks(t)[1] & ~masks(t)[0],
    "neither gate": lambda t: ~masks(t)[0] & ~masks(t)[1],
    "ALL DAYS": lambda t: pd.Series(True, index=CAL),
}
rows = []
for lbl, fn in cells.items():
    v, d, sp = pooled_cell(fn)
    nd, mn, tt = dcl(v, d)
    rows.append({"cell": lbl, "n_days": len(v), "mean_pct": round(100 * v.mean(), 3),
                 "hit": round(100 * (v > 0).mean(), 1),
                 "vs_spy_pct": round(100 * np.nanmean(v - sp), 3),
                 "n_dates": nd, "date_clustered_mean_pct": round(mn, 3),
                 "date_clustered_t": round(tt, 2)})
print(f"\n--- GATE ATTRIBUTION (pooled, day level, h={H})")
print(pd.DataFrame(rows).to_string(index=False))


# ==========================================================================
# 3. threshold neighbours — 4x4 grid, CHARGED
# ==========================================================================
print(f"\n--- THRESHOLD NEIGHBOURS (grid of 16 = 4 r63 floors x 4 r5 tops, h={H})")
grid = []
for f63 in (5, 10, 15, 20):
    for t5 in (70, 75, 80, 90):
        _, vv, dd, _, ss = member_stats(H, f63, t5)
        nd, mn, tt = dcl(vv, dd)
        grid.append({"r63<=": f63, "r5>=": t5, "n_epi": len(vv),
                     "mean_pct": round(100 * vv.mean(), 3),
                     "vs_spy_pct": round(100 * np.nanmean(vv - ss), 3),
                     "hit": round(100 * (vv > 0).mean(), 1),
                     "n_dates": nd, "dcl_mean_pct": round(mn, 3),
                     "dcl_t": round(tt, 2)})
gdf = pd.DataFrame(grid)
print(gdf.to_string(index=False))
print(f"  defended cell (10, 75) rank in the grid by dcl_mean: "
      f"{1 + int((gdf['dcl_mean_pct'] > gdf.loc[(gdf['r63<=']==10)&(gdf['r5>=']==75),'dcl_mean_pct'].iloc[0]).sum())}"
      f" of {len(gdf)}  -- this grid is CHARGED to the checker, not the candidate")
print(f"  plateau check: grid dcl_mean spread {gdf['dcl_mean_pct'].min():+.3f}% .. "
      f"{gdf['dcl_mean_pct'].max():+.3f}%, sd {gdf['dcl_mean_pct'].std():.3f}, "
      f"cells positive {int((gdf['dcl_mean_pct']>0).sum())}/{len(gdf)}")


# ==========================================================================
# 4. horizon table (h = 1..10), CHARGED as a grid
# ==========================================================================
print("\n--- HORIZON TABLE (charged grid; pitched h would come from round 3)")
hrows = []
for h in (1, 2, 3, 5, 10):
    _, vv, dd, _, ss = member_stats(h)
    nd, mn, tt = dcl(vv, dd)
    _, bv, bd, _, _ = pooled_cell(lambda t: pd.Series(True, index=CAL), h), None, None, None, None
    allv, alld, allsp = pooled_cell(lambda t: pd.Series(True, index=CAL), h)
    hrows.append({"h": h, "n_epi": len(vv), "mean_pct": round(100 * vv.mean(), 3),
                  "all_days_pct": round(100 * allv.mean(), 3),
                  "edge_pct": round(100 * (vv.mean() - allv.mean()), 3),
                  "vs_spy_pct": round(100 * np.nanmean(vv - ss), 3),
                  "n_dates": nd, "dcl_t": round(tt, 2)})
print(pd.DataFrame(hrows).to_string(index=False))


# ==========================================================================
# 5. concentration, era, midterm, SPY residual regression
# ==========================================================================
print("\n--- CONCENTRATION / STABILITY (date-clustered episode series)")
s = pd.Series(V).groupby(D.values).mean()
sd = pd.DatetimeIndex(s.index)
sv = s.values
print("  ", cluster_note(sd, sv))
order = np.argsort(-sv)
keep = np.ones(len(sv), bool)
keep[order[:2]] = False
show([summarize(sv, "all dates"), summarize(sv[keep], "drop best 2")], "drop-best-2")
yrs = sd.year
by = pd.Series(sv).groupby(yrs.values).agg(["count", "mean", "sum"])
by["mean"] = (100 * by["mean"]).round(3)
by["sum"] = (100 * by["sum"]).round(3)
print("by year:\n", by.to_string())
bestyr = by["sum"].idxmax()
show([summarize(sv[yrs != bestyr], f"drop best year {bestyr}")], "drop-best-year")
show(era_split(sd, sv), "era split")
mid = (yrs % 4) == 2
show([summarize(sv[mid], "midterm years"), summarize(sv[~mid], "non-midterm")],
     "midterm split")

# member-count drift over time (inception bias, rule 8 analogue)
cnt = pd.Series(1, index=D).groupby(D.year).sum()
print("\n  episodes per year (pool grows as ETFs incept):")
print("  ", dict(cnt))

print("\n--- SPY RESIDUAL: OLS of pooled episode return on same-window SPY return")
x = SP
y = V
m = ~np.isnan(x) & ~np.isnan(y)
b, a = np.polyfit(x[m], y[m], 1)
resid = y[m] - (a + b * x[m])
tse = resid.std(ddof=2) / np.sqrt(m.sum())
print(f"  beta {b:.3f}   alpha {100*a:+.3f}%   alpha t {a/tse:+.2f}   N={int(m.sum())}")
sdates = pd.DatetimeIndex(D)[m]
rs = pd.Series(y[m] - (a + b * x[m])).groupby(sdates.values).mean()
print(f"  date-clustered alpha t = "
      f"{rs.mean()/(rs.std(ddof=1)/np.sqrt(len(rs))):+.2f} on {len(rs)} dates")

print(f"\n--- COST: single leg, 4 bp round trip (SMH class). pooled date-clustered "
      f"mean {100*s.mean()*100:.1f} bps = {100*s.mean()*100/COST_BPS:.1f}x cost "
      f"(need >=5x). vs-SPY {100*np.nanmean(V-SP)*100:.1f} bps "
      f"= {100*np.nanmean(V-SP)*100/COST_BPS:.1f}x")
