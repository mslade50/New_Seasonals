"""Adversarial verification of track 'inversion-2026' claims.

Fresh implementation — does not import or reuse the researcher's scripts.
Live basis: rd2_fragility '63d' -> .rolling(10, min_periods=1).mean(),
reindexed daily, ffill limit 5, as-of trade Signal Date. Non-OVS only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.width", 200)

# ---------------------------------------------------------------- load
trades = pd.read_parquet("data/backtest_trades_full.parquet")
frag = pd.read_parquet("data/rd2_fragility.parquet").sort_index()

ma = frag["63d"].rolling(10, min_periods=1).mean()
daily = pd.date_range(frag.index.min(), frag.index.max() + pd.Timedelta(days=5), freq="D")
ma_daily = ma.reindex(daily).ffill(limit=5)
raw_daily = frag["63d"].reindex(daily).ffill(limit=5)

t = trades[trades["Strategy"] != "Overbot Vol Spike"].copy()
t["Signal Date"] = pd.to_datetime(t["Signal Date"])
t["frag_ma"] = t["Signal Date"].map(ma_daily)
t["frag_raw"] = t["Signal Date"].map(raw_daily)
t = t.dropna(subset=["frag_ma"]).copy()
t["year"] = t["Signal Date"].dt.year
t["month"] = t["Signal Date"].dt.to_period("M")
R = "R_Multiple"
print(f"Joined non-OVS trades: N={len(t)}, span {t['Signal Date'].min().date()}..{t['Signal Date'].max().date()}")

# ---------------------------------------------------------------- claim 1: 2026 >=50
y26 = t[t["year"] == 2026]
hi26 = y26[y26["frag_ma"] >= 50]
lo26 = y26[y26["frag_ma"] < 50]
print("\n=== CLAIM 1: 2026 >=50 ===")
print(f"N={len(hi26)}, avgR={hi26[R].mean():+.3f}, totR={hi26[R].sum():+.2f}, "
      f"win%={(hi26[R] > 0).mean()*100:.0f}")
print(f"signal-date span: {hi26['Signal Date'].min().date()}..{hi26['Signal Date'].max().date()}")
uso = hi26[(hi26["Ticker"] == "USO")]
print("USO hi-frag trades:")
print(uso[["Signal Date", "Strategy", "Direction", R]].to_string(index=False))
if len(uso):
    ex_uso = hi26.drop(uso[R].idxmax())
    print(f"drop max USO trade -> avgR={ex_uso[R].mean():+.3f} (N={len(ex_uso)})")
print("\n2026 >=50 by strategy:")
print(hi26.groupby("Strategy")[R].agg(["count", "mean"]).to_string())

# episode windows in the MA series during 2026
m26 = ma[(ma.index >= "2026-01-01")]
on = m26 >= 50
runs = []
start = None
for d, v in on.items():
    if v and start is None:
        start = d
    elif not v and start is not None:
        runs.append((start, prev))
        start = None
    prev = d
if start is not None:
    runs.append((start, prev))
print("2026 MA>=50 windows:", [(a.date(), b.date()) for a, b in runs])

# ---------------------------------------------------------------- claim 2: composition
print("\n=== CLAIM 2: composition-adjusted expectation ===")
hist = t[(t["year"] >= 2016) & (t["year"] <= 2025)]
hist_hi = hist[hist["frag_ma"] >= 50]
strat_hist = hist_hi.groupby("Strategy")[R].agg(["count", "mean"])
print("hist 2016-25 >=50 per-strategy:")
print(strat_hist.to_string())
mix = hi26.groupby("Strategy").size()
cov, wsum, nsum = 0, 0.0, 0
for s, n in mix.items():
    if s in strat_hist.index:
        wsum += n * strat_hist.loc[s, "mean"]
        nsum += n
print(f"2026 mix: {dict(mix)}")
print(f"reweighted expectation = {wsum/nsum:+.3f} (covered {nsum}/{len(hi26)} trades)")
print(f"pooled hist >=50 avgR = {hist_hi[R].mean():+.3f} (N={len(hist_hi)})")

# ---------------------------------------------------------------- claim 3: SPY phase split
print("\n=== CLAIM 3: pre-break vs post-break ===")
mp = pd.read_parquet("data/master_prices.parquet")
spy = mp[mp["ticker"] == "SPY"].set_index("date").sort_index()["Close"]
spy.index = pd.to_datetime(spy.index)
roll_hi = spy.rolling(252, min_periods=60).max()
dd = spy / roll_hi - 1.0
dd_daily = dd.reindex(daily).ffill(limit=5)
t["spy_dd"] = t["Signal Date"].map(dd_daily)

def phase(x: float) -> str:
    if x > -0.02:
        return "pre"
    if x < -0.03:
        return "post"
    return "mid"

t["phase"] = t["spy_dd"].apply(phase)
hh = t[(t["frag_ma"] >= 50) & (t["year"].between(2016, 2025))]
for ph in ["pre", "mid", "post"]:
    g = hh[hh["phase"] == ph]
    print(f"2016-25 >=50 {ph}: N={len(g)}, avgR={g[R].mean():+.3f}, win%={(g[R]>0).mean()*100:.0f}")
h26 = t[(t["frag_ma"] >= 50) & (t["year"] == 2026)]
print("2026 >=50 phase counts:", h26["phase"].value_counts().to_dict())
for ph in ["pre", "mid", "post"]:
    g = h26[h26["phase"] == ph]
    if len(g):
        print(f"2026 {ph}: N={len(g)}, avgR={g[R].mean():+.3f}")
# clustered test pre vs post (2016-25)
pre_m = hh[hh["phase"] == "pre"].groupby("month")[R].mean()
post_m = hh[hh["phase"] == "post"].groupby("month")[R].mean()
tt = stats.ttest_ind(pre_m, post_m, equal_var=False)
print(f"monthly-clustered pre vs post: t={tt.statistic:+.2f}, p={tt.pvalue:.2f} "
      f"(months: pre {len(pre_m)}, post {len(post_m)})")

# ---------------------------------------------------------------- claim 4: June / <50
print("\n=== CLAIM 4: 2026 <50 / June ===")
print(f"2026 <50: N={len(lo26)}, avgR={lo26[R].mean():+.3f}, totR={lo26[R].sum():+.2f}, "
      f"win%={(lo26[R]>0).mean()*100:.0f}")
jun = lo26[lo26["Signal Date"].dt.month == 6]
print(f"June <50: N={len(jun)}, totR={jun[R].sum():+.2f}")
print("June by ticker:")
print(jun.groupby("Ticker")[R].agg(["count", "sum"]).sort_values("sum").to_string())
energy = ["OXY", "USO", "PBR", "LYB", "DBC", "CL=F", "BP", "WLK"]
je = jun[jun["Ticker"].isin(energy)]
print(f"energy-complex (their 8 tickers): N={len(je)}, totR={je[R].sum():+.2f}")
exjun = lo26[lo26["Signal Date"].dt.month != 6]
print(f"2026 <50 ex-June: N={len(exjun)}, avgR={exjun[R].mean():+.3f}")
# monthly clustered: 2026 <50 month means vs 2016-25 <50 month means
lo_hist = t[(t["frag_ma"] < 50) & (t["year"].between(2016, 2025))]
m_hist = lo_hist.groupby("month")[R].mean()
m_26 = lo26.groupby("month")[R].mean()
tt2 = stats.ttest_ind(m_26, m_hist, equal_var=False)
print(f"2026 <50 month-means: {[f'{v:+.2f}' for v in m_26]} mean={m_26.mean():+.3f}")
print(f"hist <50 month-means: N={len(m_hist)}, mean={m_hist.mean():+.3f}")
print(f"clustered t={tt2.statistic:+.2f}, p={tt2.pvalue:.2f}")

# ---------------------------------------------------------------- claim 5: base rates
print("\n=== CLAIM 5: yearly inversion base rate ===")
rows = []
for y, g in t.groupby("year"):
    lo, hi = g[g["frag_ma"] < 50], g[g["frag_ma"] >= 50]
    rows.append({"year": y, "loN": len(lo), "lo": lo[R].mean(), "hiN": len(hi),
                 "hi": hi[R].mean() if len(hi) else np.nan,
                 "diff": (hi[R].mean() - lo[R].mean()) if len(hi) else np.nan})
yr = pd.DataFrame(rows).set_index("year")
print(yr.round(3).to_string())
meas = yr[(yr["loN"] >= 5) & (yr["hiN"] >= 5)]
print(f"measurable years (>=5 both sides): {list(meas.index)}; "
      f"inverted (diff>0): {list(meas[meas['diff'] > 0].index)}")

# distinct >=50 episodes full history (contiguous runs, merge gaps <=5 bdays)
on_all = (ma >= 50)
eps = []
start = None
prev_d = None
for d, v in on_all.items():
    if v:
        if start is None:
            start = d
        prev_d = d
    else:
        if start is not None:
            eps.append((start, prev_d))
            start = None
if start is not None:
    eps.append((start, prev_d))
merged = []
for a, b in eps:
    if merged and (a - merged[-1][1]).days <= 7:
        merged[-1] = (merged[-1][0], b)
    else:
        merged.append((a, b))
print(f"MA>=50 raw runs: {len(eps)}; merged (gap<=7cal days): {len(merged)}")
print([(a.date(), b.date()) for a, b in merged])

# ---------------------------------------------------------------- claim 6: throttle replay
print("\n=== CLAIM 6: pending-rec taper replay ===")
def taper_mult(s: float) -> float:
    if s < 50:
        return 1.0
    return max(0.5, 1.0 - 0.05 * (s - 50))

t["mult"] = t["frag_ma"].apply(taper_mult)
t["deltaR"] = (t["mult"] - 1.0) * t[R]
rep = t[t["mult"] < 1.0].groupby("year").agg(n=("mult", "size"), dR=("deltaR", "sum"))
print(rep.round(2).to_string())
print(f"total delta R 2016-2026: {t['deltaR'].sum():+.2f} over {(t['mult']<1).sum()} throttled trades")
print(f"June 2026 <50 trades above 50? {(jun['frag_ma']>=50).sum()} (should be 0)")

# ---------------------------------------------------------------- claim 7 (checkable parts)
print("\n=== CLAIM 7: SPY drawdown + raw-vs-MA lag ===")
feb_hi = spy["2026-02-01":"2026-02-28"].max()
mar30 = spy.loc["2026-03-30"] if pd.Timestamp("2026-03-30") in spy.index else spy["2026-03-25":"2026-04-02"].min()
trough = spy["2026-02-15":"2026-04-15"].min()
print(f"SPY Feb-2026 high {feb_hi:.2f} ({spy['2026-02-01':'2026-02-28'].idxmax().date()}), "
      f"2026-03-30 close {mar30:.2f} -> {(mar30/feb_hi-1)*100:+.1f}%")
print(f"Feb15-Apr15 trough {trough:.2f} ({spy['2026-02-15':'2026-04-15'].idxmin().date()}) "
      f"-> {(trough/feb_hi-1)*100:+.1f}%")
for d in ["2026-03-12", "2026-03-19"]:
    print(f"{d}: raw63d={frag.loc[d,'63d']:.1f}, MA10={ma.loc[d]:.1f}")
