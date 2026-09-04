"""Is the 2026 'inversion' a strategy-mix effect? Reweight historical per-strategy
high-frag avgR by 2026's actual high-frag strategy mix."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

trades = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
frag_ma = frag["63d"].dropna().rolling(10, min_periods=1).mean()
frag_ma.index = pd.to_datetime(frag_ma.index).normalize()

trades["Signal Date"] = pd.to_datetime(trades["Signal Date"]).dt.normalize()
t = trades[trades["Signal Date"] >= frag_ma.index.min() + pd.Timedelta(days=20)].copy().sort_values("Signal Date")
t["frag"] = pd.merge_asof(t[["Signal Date"]], frag_ma.rename("frag").reset_index(),
                          left_on="Signal Date", right_on="Date",
                          tolerance=pd.Timedelta(days=5))["frag"].values
t = t.dropna(subset=["frag", "R_Multiple"])
is_ovs = t["Strategy"].str.contains("Overbot Vol|OVS", case=False, na=False)
nb = t[~is_ovs].copy()
nb["yr"] = nb["Signal Date"].dt.year
nb["hi"] = nb.frag >= 50

hist_hi = nb[(nb.hi) & (nb.yr < 2026)]
y26_hi = nb[(nb.hi) & (nb.yr == 2026)]

print("=== historical (2016-2025) >=50 avgR by strategy ===")
h = hist_hi.groupby("Strategy")["R_Multiple"].agg(["size", "mean"]).round(3)
print(h.to_string())

print("\n=== 2026 >=50 mix vs historical mix ===")
mix26 = y26_hi.groupby("Strategy").size()
mixh = hist_hi.groupby("Strategy").size()
cmp = pd.DataFrame({"N_2026": mix26, "share_2026": (mix26 / mix26.sum()).round(2),
                    "N_hist": mixh, "share_hist": (mixh / mixh.sum()).round(2)})
print(cmp.to_string())

# composition-adjusted expectation: 2026 mix x historical per-strategy >=50 avgR
exp = 0.0
n_used = 0
for strat, n in mix26.items():
    if strat in h.index and h.loc[strat, "size"] >= 5:
        exp += n * h.loc[strat, "mean"]
        n_used += n
    else:
        print(f"  (no/thin historical >=50 sample for {strat}, N26={n})")
print(f"\ncomposition-adjusted expected avgR for 2026 >=50 mix "
      f"(hist per-strategy rates, {n_used}/{mix26.sum()} trades covered): {exp / max(n_used,1):+.3f}")
print(f"actual 2026 >=50 avgR: {y26_hi.R_Multiple.mean():+.3f}")
print(f"pooled historical >=50 avgR (all strategies): {hist_hi.R_Multiple.mean():+.3f} (N={len(hist_hi)})")

# what share of historical >=50 damage came from strategies ABSENT in 2026's episode?
absent = set(mixh.index) - set(mix26.index)
pres = set(mix26.index)
a = hist_hi[hist_hi.Strategy.isin(absent)]
p = hist_hi[hist_hi.Strategy.isin(pres)]
print(f"\nhistorical >=50, strategies ABSENT from 2026 episode ({sorted(absent)}): "
      f"N={len(a)} avgR={a.R_Multiple.mean():+.3f}")
print(f"historical >=50, strategies PRESENT in 2026 episode ({sorted(pres)}): "
      f"N={len(p)} avgR={p.R_Multiple.mean():+.3f}")

# same but with the 55+ band from the decile study
hi55 = nb[(nb.frag >= 55) & (nb.yr < 2026)]
a55 = hi55[hi55.Strategy.isin(absent)]
p55 = hi55[hi55.Strategy.isin(pres)]
print(f"\n55+ band historical: absent-strategies avgR {a55.R_Multiple.mean():+.3f} (N={len(a55)}), "
      f"present-strategies avgR {p55.R_Multiple.mean():+.3f} (N={len(p55)})")

# June 2026 OLV cluster detail: energy concentration
lo26 = nb[(nb.yr == 2026) & (~nb.hi)]
jun = lo26[lo26["Signal Date"].dt.month == 6]
print(f"\nJune 2026 below-50: N={len(jun)}, totR={jun.R_Multiple.sum():+.1f}")
energy = {"USO", "OXY", "CL=F", "PBR", "BP", "DBC", "WLK", "LYB", "PBA", "CVE", "GLNG"}
je = jun[jun.Ticker.isin(energy)]
print(f"  energy/commodity-complex tickers: N={len(je)}, totR={je.R_Multiple.sum():+.1f}, avgR={je.R_Multiple.mean():+.3f}")
jo = jun[~jun.Ticker.isin(energy)]
print(f"  everything else: N={len(jo)}, totR={jo.R_Multiple.sum():+.1f}, avgR={jo.R_Multiple.mean():+.3f}")
print(f"2026 below-50 EX-June: N={len(lo26)-len(jun)}, avgR={lo26[lo26['Signal Date'].dt.month != 6].R_Multiple.mean():+.3f}")

# distinct-ticker view of June (OLV re-signals same ticker daily)
print("\nJune below-50 by ticker:")
print(jun.groupby("Ticker")["R_Multiple"].agg(["size", "sum"]).round(2).sort_values("sum").to_string())

# per-strategy 2026 >=50 vs their own historical >=50 (did LT Trend / OLV do better than their own hist rates?)
print("\nper-strategy: 2026 >=50 avgR vs own historical >=50 avgR:")
for strat in mix26.index:
    o = y26_hi[y26_hi.Strategy == strat].R_Multiple
    hh = hist_hi[hist_hi.Strategy == strat].R_Multiple
    print(f"  {strat}: 2026 {o.mean():+.3f} (N={len(o)}) vs hist {hh.mean():+.3f} (N={len(hh)})")
