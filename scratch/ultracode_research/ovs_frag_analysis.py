"""OVS exemption from the fragility multiplier — is the 21-44 mid-band dip real?

Bands follow the established decile-band cuts: [0,3), [3,21), [21,44), [44,55), [55,100].
Live basis: 63d fragility, 10d rolling mean, as-of signal date (ffill limit 5d).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")

trades = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")

frag_ma = frag["63d"].dropna().rolling(10, min_periods=1).mean()
frag_ma.index = pd.to_datetime(frag_ma.index).normalize()

trades["Signal Date"] = pd.to_datetime(trades["Signal Date"]).dt.normalize()
start = frag_ma.index.min() + pd.Timedelta(days=20)

ovs = trades[trades.Strategy.str.contains("Overbot Vol", na=False)].copy()
ovs = ovs[ovs["Signal Date"] >= start].sort_values("Signal Date").reset_index(drop=True)

f = frag_ma.rename("frag").reset_index()
f.columns = ["Date", "frag"]
ovs["frag"] = pd.merge_asof(
    ovs[["Signal Date"]], f, left_on="Signal Date", right_on="Date",
    tolerance=pd.Timedelta(days=5),
)["frag"].values
ovs = ovs.dropna(subset=["frag", "R_Multiple"]).copy()

ovs["gap_atr"] = (ovs["T+1 Open"] - ovs["Signal Close"]) / ovs["ATR"]
ovs["path"] = np.where(ovs.gap_atr > 0.25, "P1", "P2")
ovs["yr"] = ovs["Signal Date"].dt.year
ovs["ym"] = ovs["Signal Date"].dt.to_period("M")
ovs["midterm"] = (ovs.yr % 4 == 2)

EDGES = [0, 3, 21, 44, 55, 100.001]
LABELS = ["0-3", "3-21", "21-44", "44-55", "55+"]
ovs["band"] = pd.cut(ovs.frag, bins=EDGES, labels=LABELS, include_lowest=True)

print(f"OVS trades in frag window: {len(ovs)} ({ovs['Signal Date'].min().date()}..{ovs['Signal Date'].max().date()})")
print(f"P1={len(ovs[ovs.path=='P1'])} P2={len(ovs[ovs.path=='P2'])} midterm-yr trades={ovs.midterm.sum()}\n")


def band_table(df, by="band"):
    g = df.groupby(by, observed=False)["R_Multiple"]
    out = pd.DataFrame({
        "N": g.size(), "avgR": g.mean().round(3), "medR": g.median().round(3),
        "win%": g.apply(lambda s: (s > 0).mean() * 100).round(1),
        "totR": g.sum().round(1),
    })
    return out


print("=== OVS R by fragility band (all) ===")
print(band_table(ovs).to_string(), "\n")

print("=== by band x path ===")
for p in ["P1", "P2"]:
    print(f"-- {p} --")
    print(band_table(ovs[ovs.path == p]).to_string(), "\n")

print("=== by band x tier ===")
for tier in ["Liquid", "Overflow"]:
    print(f"-- {tier} --")
    print(band_table(ovs[ovs.Tier == tier]).to_string(), "\n")

print("=== by band x midterm ===")
for mt in [False, True]:
    print(f"-- midterm={mt} --")
    print(band_table(ovs[ovs.midterm == mt]).to_string(), "\n")

# ---------- Q1: is the 21-44 dip statistically real? ----------
mid = ovs[ovs.band == "21-44"]
rest = ovs[ovs.band != "21-44"]
low = ovs[ovs.frag < 21]
hi = ovs[ovs.frag >= 44]

def cluster_t(a, b, la, lb):
    am = a.groupby("ym")["R_Multiple"].mean()
    bm = b.groupby("ym")["R_Multiple"].mean()
    tt = stats.ttest_ind(am, bm, equal_var=False)
    print(f"{la} {am.mean():+.3f} ({len(am)} mo, {len(a)} tr) vs {lb} {bm.mean():+.3f} "
          f"({len(bm)} mo, {len(b)} tr)  t={tt.statistic:+.2f} p={tt.pvalue:.3f}")
    return tt

print("=== Q1: monthly-clustered tests ===")
cluster_t(mid, rest, "21-44", "all-other")
cluster_t(mid, low, "21-44", "frag<21")
cluster_t(mid, hi, "21-44", "frag>=44")
cluster_t(mid[mid.path == "P1"], low[low.path == "P1"], "21-44 P1", "frag<21 P1")
cluster_t(mid[mid.path == "P2"], low[low.path == "P2"], "21-44 P2", "frag<21 P2")

# yearly composition of the 21-44 band
print("\n21-44 band by year:")
g = mid.groupby("yr")["R_Multiple"]
print(pd.DataFrame({"N": g.size(), "avgR": g.mean().round(3), "totR": g.sum().round(1)}).to_string())

# LOYO on the mid-vs-low contrast
print("\nLOYO (drop one year, monthly-clustered t of 21-44 vs frag<21):")
years = sorted(ovs.yr.unique())
for drop in years:
    a = mid[mid.yr != drop]
    b = low[low.yr != drop]
    am = a.groupby("ym")["R_Multiple"].mean()
    bm = b.groupby("ym")["R_Multiple"].mean()
    tt = stats.ttest_ind(am, bm, equal_var=False)
    print(f"  drop {drop}: 21-44 {am.mean():+.3f} (N={len(a)}) vs <21 {bm.mean():+.3f} (N={len(b)}) "
          f"t={tt.statistic:+.2f} p={tt.pvalue:.3f}")

# ---------- Q4: is 55+ recovery just 2022? ----------
print("\n=== Q4: 55+ band by year ===")
g = hi55 = ovs[ovs.band == "55+"]
gg = g.groupby("yr")["R_Multiple"]
print(pd.DataFrame({"N": gg.size(), "avgR": gg.mean().round(3), "totR": gg.sum().round(1)}).to_string())
print("\n55+ excluding 2022:", end=" ")
ex = g[g.yr != 2022]
print(f"N={len(ex)} avgR={ex.R_Multiple.mean():+.3f}")
ex2 = g[~g.yr.isin([2022, 2020])]
print(f"55+ excluding 2020 AND 2022: N={len(ex2)} avgR={ex2.R_Multiple.mean():+.3f}")
print("55+ by path:")
print(band_table(g, by="path").to_string())

# 44-55 by year for completeness
print("\n44-55 band by year:")
b45 = ovs[ovs.band == "44-55"]
gg = b45.groupby("yr")["R_Multiple"]
print(pd.DataFrame({"N": gg.size(), "avgR": gg.mean().round(3), "totR": gg.sum().round(1)}).to_string())

# ---------- Q3: what would a mid-band 0.5x throttle have done? ----------
print("\n=== Q3: replay of a 0.5x throttle on 21<=frag<44 (in-sample bookkeeping, not validation) ===")
ovs["mult"] = np.where((ovs.frag >= 21) & (ovs.frag < 44), 0.5, 1.0)
ovs["R_adj"] = ovs.R_Multiple * ovs.mult
print(f"totR {ovs.R_Multiple.sum():+.1f} -> {ovs.R_adj.sum():+.1f}; "
      f"avgR/unit-risk {ovs.R_Multiple.mean():+.4f} -> {ovs.R_adj.sum()/ovs.mult.sum():+.4f}; "
      f"risk-weighted N {ovs.mult.sum():.0f} vs {len(ovs)}")
for label, col in [("baseline", "R_Multiple"), ("throttle", "R_adj")]:
    s = ovs.sort_values("Exit Date").groupby("Exit Date")[col].sum().cumsum()
    dd = (s - s.cummax()).min()
    print(f"worst R drawdown ({label}): {dd:+.1f}")

# episode structure: distinct months in the mid band
print("\n21-44 band: trades per month (top 15):")
print(mid.groupby("ym").size().sort_values(ascending=False).head(15).to_string())

# Spearman within band ranges
rho, p = stats.spearmanr(ovs.frag, ovs.R_Multiple)
print(f"\nSpearman(frag, R) OVS all: rho={rho:+.4f} p={p:.3f} N={len(ovs)}")
