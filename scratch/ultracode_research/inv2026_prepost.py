"""High-frag trades: pre-break (SPY near high) vs post-break (already in dip).
The 10d-MA lag means the throttle keeps flagging >=50 during the dip; dip-buyers
signed there face a different distribution than trades signed at the highs."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]

trades = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
frag_ma = frag["63d"].dropna().rolling(10, min_periods=1).mean()
frag_ma.index = pd.to_datetime(frag_ma.index).normalize()

mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
spy = mp[mp.ticker == "SPY"].set_index("date")["Close"].sort_index()
spy.index = pd.to_datetime(spy.index).normalize()
dd = (spy / spy.rolling(252).max() - 1) * 100  # % off 52w high

trades["Signal Date"] = pd.to_datetime(trades["Signal Date"]).dt.normalize()
t = trades[trades["Signal Date"] >= frag_ma.index.min() + pd.Timedelta(days=20)].copy().sort_values("Signal Date")
t["frag"] = pd.merge_asof(t[["Signal Date"]], frag_ma.rename("frag").reset_index(),
                          left_on="Signal Date", right_on="Date",
                          tolerance=pd.Timedelta(days=5))["frag"].values
t["spy_dd"] = dd.reindex(t["Signal Date"], method="ffill").values
t = t.dropna(subset=["frag", "R_Multiple", "spy_dd"])
is_ovs = t["Strategy"].str.contains("Overbot Vol|OVS", case=False, na=False)
nb = t[~is_ovs].copy()
nb["yr"] = nb["Signal Date"].dt.year

hi = nb[nb.frag >= 50].copy()
hi["phase"] = np.where(hi.spy_dd > -2, "pre-break (dd>-2%)",
               np.where(hi.spy_dd < -3, "post-break (dd<-3%)", "transition"))

print("=== >=50 trades by phase, 2016-2025 vs 2026 ===")
for era, sub in [("2016-2025", hi[hi.yr < 2026]), ("2026", hi[hi.yr == 2026])]:
    g = sub.groupby("phase")["R_Multiple"].agg(["size", "mean", "median",
                                                lambda s: (s > 0).mean() * 100]).round(3)
    g.columns = ["N", "avgR", "medR", "win%"]
    print(f"\n{era}:")
    print(g.to_string())

# short-horizon index dip buyers vs the rest, by phase (historical)
short_dip = ["Weak Close Decent Sznls", "SPY QQQ MonFri Reversion", "Monday Dip"]
hh = hi[hi.yr < 2026].copy()
hh["grp"] = np.where(hh.Strategy.isin(short_dip), "short-dip-buyers", "other")
print("\n=== historical >=50 by phase x strategy-group ===")
g = hh.groupby(["grp", "phase"])["R_Multiple"].agg(["size", "mean"]).round(3)
print(g.to_string())

# monthly-clustered: historical >=50 pre-break vs post-break
hh["ym"] = hh["Signal Date"].dt.to_period("M")
pre = hh[hh.phase.str.startswith("pre")]
post = hh[hh.phase.str.startswith("post")]
pm = pre.groupby("ym")["R_Multiple"].mean()
qm = post.groupby("ym")["R_Multiple"].mean()
tt = stats.ttest_ind(pm, qm, equal_var=False)
print(f"\nmonthly-clustered hist >=50: pre-break {pm.mean():+.3f} ({len(pm)} mo, {len(pre)} tr) vs "
      f"post-break {qm.mean():+.3f} ({len(qm)} mo, {len(post)} tr)  t={tt.statistic:+.2f} p={tt.pvalue:.3f}")

# 2026 >=50 trades: dd at signal
y26 = hi[hi.yr == 2026]
print(f"\n2026 >=50: spy_dd at signal — min {y26.spy_dd.min():.1f}%, median {y26.spy_dd.median():.1f}%, "
      f"max {y26.spy_dd.max():.1f}%; N pre-break={len(y26[y26.spy_dd > -2])}, "
      f"post-break={len(y26[y26.spy_dd < -3])}")

# raw (unsmoothed) 63d score at those 2026 signal dates: already decaying?
raw = frag["63d"].dropna()
raw.index = pd.to_datetime(raw.index).normalize()
y26 = y26.copy()
y26["raw63"] = raw.reindex(y26["Signal Date"], method="ffill").values
print("\n2026 >=50 trades: 10dMA vs raw 63d at signal date:")
print(y26[["Signal Date", "Strategy", "Ticker", "frag", "raw63", "spy_dd", "R_Multiple"]]
      .assign(**{"Signal Date": y26["Signal Date"].dt.date})
      .round(2).to_string(index=False))
print(f"\nshare of 2026 >=50 trades where raw63 < 50: {(y26.raw63 < 50).mean()*100:.0f}%")

hh2 = hi[hi.yr < 2026].copy()
hh2["raw63"] = raw.reindex(hh2["Signal Date"], method="ffill").values
print(f"share of 2016-25 >=50(MA) trades where raw63 < 50: {(hh2.raw63 < 50).mean()*100:.0f}%")
for era, sub in [("hist", hh2), ("2026", y26)]:
    a = sub[sub.raw63 >= 50]; b = sub[sub.raw63 < 50]
    print(f"{era}: raw>=50 avgR {a.R_Multiple.mean():+.3f} (N={len(a)}) | "
          f"raw<50 (MA-lag trades) avgR {b.R_Multiple.mean():+.3f} (N={len(b)})")
