"""2026 inversion: trade-level anatomy (Q1), below-50 weakness (Q3), base rates (Q4)."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]

trades = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")

# live basis: 63d, 10d MA, as-of signal date (ffill limit 5d)
frag_ma = frag["63d"].dropna().rolling(10, min_periods=1).mean()
frag_ma.index = pd.to_datetime(frag_ma.index).normalize()

trades["Signal Date"] = pd.to_datetime(trades["Signal Date"]).dt.normalize()
start = frag_ma.index.min() + pd.Timedelta(days=20)
t = trades[trades["Signal Date"] >= start].copy().sort_values("Signal Date")
t["frag"] = pd.merge_asof(
    t[["Signal Date"]], frag_ma.rename("frag").reset_index(),
    left_on="Signal Date", right_on="Date", tolerance=pd.Timedelta(days=5),
)["frag"].values
t = t.dropna(subset=["frag", "R_Multiple"])

is_ovs = t["Strategy"].str.contains("Overbot Vol|OVS", case=False, na=False)
nb = t[~is_ovs].copy()
nb["yr"] = nb["Signal Date"].dt.year
nb["ym"] = nb["Signal Date"].dt.to_period("M")
nb["hi"] = nb.frag >= 50

print(f"non-OVS trades joined: {len(nb)} ({nb['Signal Date'].min().date()}..{nb['Signal Date'].max().date()})")

# ============ Q1: 2026 anatomy ============
y26 = nb[nb.yr == 2026].copy()
print(f"\n2026 non-OVS: N={len(y26)}, >=50 N={y26.hi.sum()} avgR={y26[y26.hi].R_Multiple.mean():+.3f}, "
      f"<50 N={(~y26.hi).sum()} avgR={y26[~y26.hi].R_Multiple.mean():+.3f}")

cols = ["Signal Date", "Strategy", "Ticker", "Direction", "Exit Type", "R_Multiple", "frag", "Tier"]
for side, sub in [(">=50", y26[y26.hi]), ("<50", y26[~y26.hi])]:
    print(f"\n--- 2026 trades frag {side} (N={len(sub)}) ---")
    s = sub[cols].copy()
    s["Signal Date"] = s["Signal Date"].dt.date
    s["frag"] = s["frag"].round(1)
    s["R_Multiple"] = s["R_Multiple"].round(2)
    print(s.sort_values("Signal Date").to_string(index=False))

print("\n--- 2026 by strategy x side ---")
g = y26.groupby(["Strategy", "hi"])["R_Multiple"].agg(["size", "mean", "sum"]).round(3)
print(g.to_string())

print("\n--- 2026 by month x side ---")
g = y26.groupby([y26["Signal Date"].dt.to_period("M"), "hi"])["R_Multiple"].agg(["size", "mean", "sum"]).round(3)
print(g.to_string())

# frag regime in 2026: when was the 10dMA >= 50?
f26 = frag_ma[frag_ma.index >= "2026-01-01"]
above = f26[f26 >= 50]
print(f"\n2026 frag(63d,10dMA): days>=50: {len(above)} of {len(f26)}; "
      f"range of >=50 dates: {above.index.min().date() if len(above) else '-'} .. {above.index.max().date() if len(above) else '-'}")
# episodes
if len(above):
    d = above.index.to_series()
    gaps = d.diff().dt.days.fillna(1)
    ep = (gaps > 7).cumsum()
    for e, dd in d.groupby(ep):
        print(f"  episode: {dd.min().date()} .. {dd.max().date()} ({len(dd)} days, peak {f26.loc[dd].max():.1f})")

# concentration: does one strategy or a handful of trades drive the +0.49?
hi26 = y26[y26.hi].sort_values("R_Multiple")
print(f"\n>=50 2026: totR={hi26.R_Multiple.sum():+.1f}; top-3 trades sum "
      f"{hi26.R_Multiple.nlargest(3).sum():+.1f}; median {hi26.R_Multiple.median():+.2f}; "
      f"win% {(hi26.R_Multiple > 0).mean()*100:.0f}")
lo26 = y26[~y26.hi]
print(f"<50 2026: totR={lo26.R_Multiple.sum():+.1f}; bottom-3 trades sum "
      f"{lo26.R_Multiple.nsmallest(3).sum():+.1f}; median {lo26.R_Multiple.median():+.2f}; "
      f"win% {(lo26.R_Multiple > 0).mean()*100:.0f}")

# drop-worst sensitivity on the >=50 side
r = hi26.R_Multiple.sort_values(ascending=False).values
for k in [1, 2, 3]:
    print(f">=50 avgR dropping top {k} winners: {r[k:].mean():+.3f} (N={len(r)-k})")

# ============ Q3: is 2026 below-50 weakness unusual? ============
print("\n\n=== Q3: below-50 avgR by year (non-OVS) ===")
rows = []
for yr, g in nb.groupby("yr"):
    lo, hi = g[~g.hi], g[g.hi]
    rows.append({
        "year": yr,
        "lo_N": len(lo), "lo_avgR": round(lo.R_Multiple.mean(), 3) if len(lo) else np.nan,
        "lo_med": round(lo.R_Multiple.median(), 3) if len(lo) else np.nan,
        "lo_win%": round((lo.R_Multiple > 0).mean() * 100, 0) if len(lo) else np.nan,
        "hi_N": len(hi), "hi_avgR": round(hi.R_Multiple.mean(), 3) if len(hi) else np.nan,
        "diff(hi-lo)": round(hi.R_Multiple.mean() - lo.R_Multiple.mean(), 3) if len(hi) >= 5 and len(lo) >= 5 else np.nan,
    })
yr_tab = pd.DataFrame(rows)
print(yr_tab.to_string(index=False))

# which strategies drove 2026 below-50 weakness
print("\n2026 below-50 by strategy:")
g = lo26.groupby("Strategy")["R_Multiple"].agg(["size", "mean", "sum"]).round(3).sort_values("sum")
print(g.to_string())

print("\n2026 below-50 by month:")
g = lo26.groupby(lo26["Signal Date"].dt.to_period("M"))["R_Multiple"].agg(["size", "mean", "sum"]).round(3)
print(g.to_string())

# same strategies' below-50 performance 2016-2025 (baseline)
lo_hist = nb[(~nb.hi) & (nb.yr < 2026)]
print("\nBelow-50 2016-2025 by strategy (baseline for comparison):")
g = lo_hist.groupby("Strategy")["R_Multiple"].agg(["size", "mean"]).round(3)
g26 = lo26.groupby("Strategy")["R_Multiple"].agg(["size", "mean"]).round(3)
cmp = g.join(g26, lsuffix="_hist", rsuffix="_2026", how="outer")
print(cmp.to_string())

# monthly-clustered test: 2026 below-50 vs historical below-50
m26 = lo26.groupby("ym")["R_Multiple"].mean()
mh = lo_hist.groupby("ym")["R_Multiple"].mean()
tt = stats.ttest_ind(m26, mh, equal_var=False)
print(f"\nmonthly-clustered below-50: 2026 {m26.mean():+.3f} ({len(m26)} mo, {len(lo26)} tr) vs "
      f"2016-25 {mh.mean():+.3f} ({len(mh)} mo, {len(lo_hist)} tr)  t={tt.statistic:+.2f} p={tt.pvalue:.3f}")

# how many prior single years had below-50 avgR <= -0.09?
print("prior years with below-50 avgR <= 2026's -0.09:",
      yr_tab[(yr_tab.year < 2026) & (yr_tab.lo_avgR <= lo26.R_Multiple.mean())][["year", "lo_N", "lo_avgR"]].to_dict("records"))

# ============ Q4: base rate of inversion ============
print("\n\n=== Q4: how often does a single year invert? ===")
# inversion := hi avgR > lo avgR in that year (with min N=5 both sides)
yr_tab["inverted"] = yr_tab["diff(hi-lo)"] > 0
valid = yr_tab.dropna(subset=["diff(hi-lo)"])
print(valid[["year", "lo_N", "lo_avgR", "hi_N", "hi_avgR", "diff(hi-lo)", "inverted"]].to_string(index=False))
inv_years = valid[valid.inverted].year.tolist()
print(f"\ninverted years (min N=5/side): {inv_years} out of {len(valid)} measurable years")

# did an inversion in year t predict year t+1's diff?
print("\nyear t inversion -> year t+1 diff:")
v = valid.set_index("year")["diff(hi-lo)"]
for y in v.index[:-1]:
    if y + 1 in v.index:
        print(f"  {y}: diff {v[y]:+.3f} -> {y+1}: diff {v[y+1]:+.3f}")

# with a lower N floor (N>=3) to include more years
rows2 = []
for yr, g in nb.groupby("yr"):
    lo, hi = g[~g.hi], g[g.hi]
    if len(lo) >= 3 and len(hi) >= 3:
        rows2.append((yr, len(lo), len(hi), hi.R_Multiple.mean() - lo.R_Multiple.mean()))
print("\nlooser floor N>=3/side:")
for yr, ln, hn, d in rows2:
    print(f"  {yr}: loN={ln} hiN={hn} diff={d:+.3f} {'INVERTED' if d > 0 else ''}")

# bootstrap: under the pooled 2016-2025 relationship, how often would a
# random 'year' of 2026's size show hi-lo diff >= 2026's observed +0.58?
obs26 = y26[y26.hi].R_Multiple.mean() - y26[~y26.hi].R_Multiple.mean()
pool_hi = nb[(nb.hi) & (nb.yr < 2026)].R_Multiple.values
pool_lo = nb[(~nb.hi) & (nb.yr < 2026)].R_Multiple.values
rng = np.random.default_rng(7)
n_hi, n_lo = y26.hi.sum(), (~y26.hi).sum()
sims = np.array([
    rng.choice(pool_hi, n_hi).mean() - rng.choice(pool_lo, n_lo).mean()
    for _ in range(20000)
])
print(f"\n2026 observed hi-lo diff: {obs26:+.3f}")
print(f"iid bootstrap from 2016-25 pools (nhi={n_hi}, nlo={n_lo}): "
      f"P(diff >= obs) = {(sims >= obs26).mean():.4f}  (mean sim diff {sims.mean():+.3f})")
print("NOTE: iid bootstrap understates tail prob because trades cluster; treat as lower bound on rarity.")

# block bootstrap by month: resample months with replacement
mo_hi = nb[(nb.hi) & (nb.yr < 2026)].groupby("ym")["R_Multiple"].agg(list)
mo_lo = nb[(~nb.hi) & (nb.yr < 2026)].groupby("ym")["R_Multiple"].agg(list)
sims2 = []
for _ in range(10000):
    acc_h = []
    while len(acc_h) < n_hi:
        acc_h += mo_hi.iloc[rng.integers(len(mo_hi))]
    acc_l = []
    while len(acc_l) < n_lo:
        acc_l += mo_lo.iloc[rng.integers(len(mo_lo))]
    sims2.append(np.mean(acc_h[:n_hi]) - np.mean(acc_l[:n_lo]))
sims2 = np.array(sims2)
print(f"month-block bootstrap: P(diff >= obs) = {(sims2 >= obs26).mean():.4f}  (mean {sims2.mean():+.3f}, sd {sims2.std():.3f})")
