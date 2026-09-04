"""Follow-ups: midterm confound in the 21-44 band, 0-3 peak significance,
earnings-data composition, 2026 concentration, and double-throttle accounting."""
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
f = frag_ma.rename("frag").reset_index(); f.columns = ["Date", "frag"]
ovs["frag"] = pd.merge_asof(ovs[["Signal Date"]], f, left_on="Signal Date",
                            right_on="Date", tolerance=pd.Timedelta(days=5))["frag"].values
ovs = ovs.dropna(subset=["frag", "R_Multiple"]).copy()
ovs["gap_atr"] = (ovs["T+1 Open"] - ovs["Signal Close"]) / ovs["ATR"]
ovs["path"] = np.where(ovs.gap_atr > 0.25, "P1", "P2")
ovs["yr"] = ovs["Signal Date"].dt.year
ovs["ym"] = ovs["Signal Date"].dt.to_period("M")
ovs["midterm"] = (ovs.yr % 4 == 2)
EDGES = [0, 3, 21, 44, 55, 100.001]
LABELS = ["0-3", "3-21", "21-44", "44-55", "55+"]
ovs["band"] = pd.cut(ovs.frag, bins=EDGES, labels=LABELS, include_lowest=True)


def cluster_t(a, b, la, lb):
    am = a.groupby("ym")["R_Multiple"].mean()
    bm = b.groupby("ym")["R_Multiple"].mean()
    tt = stats.ttest_ind(am, bm, equal_var=False)
    print(f"{la} {am.mean():+.3f} ({len(am)} mo, {len(a)} tr) vs {lb} {bm.mean():+.3f} "
          f"({len(bm)} mo, {len(b)} tr)  t={tt.statistic:+.2f} p={tt.pvalue:.3f}")


nm = ovs[~ovs.midterm]
mt = ovs[ovs.midterm]

print("=== midterm confound: mid-band dip within non-midterm years only ===")
cluster_t(nm[nm.band == "21-44"], nm[nm.frag < 21], "NM 21-44", "NM frag<21")
print("\n=== and within midterm years only ===")
cluster_t(mt[mt.band == "21-44"], mt[mt.frag < 21], "MT 21-44", "MT frag<21")

print("\nband composition: % midterm trades per band")
print((ovs.groupby("band", observed=False).midterm.mean() * 100).round(1).to_string())
print(f"overall midterm share: {ovs.midterm.mean()*100:.1f}%")

print("\n=== is the 0-3 peak itself significant? (the case FOR full size at calm) ===")
cluster_t(ovs[ovs.band == "0-3"], ovs[ovs.band != "0-3"], "0-3", "rest")
cluster_t(ovs[(ovs.band == "0-3") & (ovs.path == "P1")],
          ovs[(ovs.band != "0-3") & (ovs.path == "P1")], "0-3 P1", "rest P1")

print("\n=== P2 overall check (live retired P2) ===")
for p in ["P1", "P2"]:
    sub = ovs[ovs.path == p]
    print(f"{p}: N={len(sub)} avgR={sub.R_Multiple.mean():+.3f} totR={sub.R_Multiple.sum():+.1f} "
          f"win%={(sub.R_Multiple>0).mean()*100:.1f}")
cluster_t(ovs[ovs.path == "P1"], ovs[ovs.path == "P2"], "P1", "P2")

print("\n=== 2026 concentration in the 21-44 band ===")
mid = ovs[ovs.band == "21-44"]
print(f"2026 share of mid-band trades: {len(mid[mid.yr==2026])}/{len(mid)} "
      f"({len(mid[mid.yr==2026])/len(mid)*100:.0f}%)")
cluster_t(mid[mid.yr != 2026], ovs[(ovs.frag < 21) & (ovs.yr != 2026)],
          "21-44 ex-2026", "frag<21 ex-2026")

print("\n=== earnings-data composition (blackout already applied upstream; ===")
print("=== remaining trades near earnings = tickers with NO earnings data) ===")
try:
    ec = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet")
    tick_col = [c for c in ec.columns if c.lower() in ("ticker", "symbol")][0]
    has_earn = set(ec[tick_col].unique())
    ovs["has_earn_data"] = ovs.Ticker.isin(has_earn)
    print("share of trades on tickers WITH earnings data, per band:")
    print((ovs.groupby("band", observed=False).has_earn_data.mean() * 100).round(1).to_string())
    g = ovs.groupby(["band", "has_earn_data"], observed=False)["R_Multiple"]
    print(pd.DataFrame({"N": g.size(), "avgR": g.mean().round(3)}).to_string())
except Exception as e:
    print("earnings calendar check failed:", e)

print("\n=== double-throttle accounting: what a 0.5x mid-band mult would do ===")
print("=== ON TOP of the existing 0.75x midterm tilt ===")
mid_mt = mid[mid.midterm]
print(f"mid-band midterm trades: {len(mid_mt)} of {len(mid)} — these already run 0.75x live;")
print(f"a naive 0.5x band mult would put them at 0.375x.")

print("\n=== half-sample split (2016-2021 vs 2022-2026) of the U-shape ===")
for label, sub in [("2016-2021", ovs[ovs.yr <= 2021]), ("2022-2026", ovs[ovs.yr >= 2022])]:
    g = sub.groupby("band", observed=False)["R_Multiple"]
    print(f"-- {label} --")
    print(pd.DataFrame({"N": g.size(), "avgR": g.mean().round(3)}).to_string(), "\n")

print("=== distinct-episode view: mid-band months with >=5 trades ===")
heavy = mid.groupby("ym").agg(N=("R_Multiple", "size"), avgR=("R_Multiple", "mean")).query("N>=5")
print(heavy.round(3).to_string())
print(f"\nmid-band trades in heavy months: {heavy.N.sum()} of {len(mid)}")
light = mid.groupby("ym").agg(N=("R_Multiple", "size"), avgR=("R_Multiple", "mean")).query("N<5")
print(f"light-month (<5 tr) mid-band: {light.N.sum()} tr, trade-wtd avgR "
      f"{mid[mid.ym.isin(light.index)].R_Multiple.mean():+.3f}")
