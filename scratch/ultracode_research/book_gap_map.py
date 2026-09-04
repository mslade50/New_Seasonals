"""Book gap map — integration-analyst factual baseline for any new sleeve.

Builds from data/backtest_trades_full.parquet (flat $750k basis) +
data/rd2_fragility.parquet (63d col, 10d MA = live sizing basis) +
data/master_prices.parquet (SPY):

1. Monthly realized R + PnL_flat series (by EXIT month); 10 worst months with
   regime context (fragility level, SPY drawdown state, SPY month return).
2. Exposure cadence: daily concurrent open trades / gross notional / open risk,
   grouped by fragility band. New-signal cadence by band.
3. Seasonality: book PnL by calendar month and by presidential cycle year.
4. Concentration: PnL share by strategy, tier, ticker family; top tickers.
5. Quant spec for a new sleeve: book monthly Sharpe/vol/maxDD, high-fragility
   month performance (63d MA10 >= 50, 2016+), and a combined-Sharpe grid over
   (sleeve Sharpe, correlation, risk share).

All stats are GROSS of the live fragility multiplier (ledger is flat-sized) —
this is the natural signal cadence of the book, not the throttled live book.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ACCOUNT = 750_000.0

pd.set_option("display.width", 200)

trades = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")

frag_ma = frag["63d"].dropna().rolling(10, min_periods=1).mean()
frag_ma.index = pd.to_datetime(frag_ma.index).normalize()

for c in ["Signal Date", "Entry Date", "Exit Date"]:
    trades[c] = pd.to_datetime(trades[c]).dt.normalize()

# SPY daily closes for drawdown context
mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
spy = (mp[mp["ticker"] == "SPY"].set_index("date")["Close"].sort_index())
spy.index = pd.to_datetime(spy.index).normalize()
spy_dd = spy / spy.cummax() - 1.0

# ---------------------------------------------------------------- 1. monthly series
trades["exit_ym"] = trades["Exit Date"].dt.to_period("M")
m_pnl = trades.groupby("exit_ym")["PnL_flat_750k"].sum()
m_r = trades.groupby("exit_ym")["R_Multiple"].sum()
m_n = trades.groupby("exit_ym").size()
# reindex to full month range so quiet months = 0
full_idx = pd.period_range(m_pnl.index.min(), m_pnl.index.max(), freq="M")
m_pnl = m_pnl.reindex(full_idx, fill_value=0.0)
m_r = m_r.reindex(full_idx, fill_value=0.0)
m_n = m_n.reindex(full_idx, fill_value=0)
m_ret = m_pnl / ACCOUNT

frag_m = frag_ma.groupby(frag_ma.index.to_period("M")).mean()
frag_m_max = frag_ma.groupby(frag_ma.index.to_period("M")).max()
spy_m_ret = spy.groupby(spy.index.to_period("M")).last().pct_change()
spy_m_dd = spy_dd.groupby(spy_dd.index.to_period("M")).min()

monthly = pd.DataFrame({
    "pnl": m_pnl, "ret_pct": m_ret * 100, "R": m_r, "n_exits": m_n,
    "frag_avg": frag_m.reindex(full_idx),
    "frag_max": frag_m_max.reindex(full_idx),
    "spy_ret_pct": (spy_m_ret.reindex(full_idx) * 100),
    "spy_dd_pct": (spy_m_dd.reindex(full_idx) * 100),
})

print("=" * 90)
print("1. BOOK MONTHLY SERIES (flat $750k, by exit month)")
print("=" * 90)
eq = m_pnl.cumsum()
dd_dollar = eq - eq.cummax()
n_months = len(m_ret)
mu, sd = m_ret.mean(), m_ret.std()
sharpe = mu / sd * np.sqrt(12)
print(f"months: {n_months}  ({full_idx[0]}..{full_idx[-1]})")
print(f"avg monthly PnL ${m_pnl.mean():,.0f}  ({mu*100:+.3f}% of 750k)  vol {sd*100:.3f}%/mo")
print(f"annualized: ret {mu*12*100:+.2f}%  vol {sd*np.sqrt(12)*100:.2f}%  Sharpe {sharpe:.2f} (gross, flat basis, 0% cash hurdle)")
print(f"total PnL ${m_pnl.sum():,.0f}   total R {m_r.sum():+.1f}")
print(f"max drawdown (flat $): ${dd_dollar.min():,.0f}  ({dd_dollar.min()/ACCOUNT*100:.2f}% of 750k)")
print(f"% months positive: {(m_ret > 0).mean()*100:.1f}%   worst month ${m_pnl.min():,.0f}  best ${m_pnl.max():,.0f}")

print("\n--- 10 WORST MONTHS (with regime context) ---")
worst = monthly.nsmallest(10, "pnl").copy()
worst["pnl"] = worst["pnl"].round(0)
print(worst.round(2).to_string())

print("\n--- 10 BEST MONTHS ---")
print(monthly.nlargest(10, "pnl").round(2).to_string())

# per-year table
print("\n--- PER-YEAR TABLE ---")
yr = pd.DataFrame({
    "pnl": m_pnl.groupby(m_pnl.index.year).sum(),
    "ret_pct": m_ret.groupby(m_ret.index.year).sum() * 100,
    "R": m_r.groupby(m_r.index.year).sum(),
    "n_trades": m_n.groupby(m_n.index.year).sum(),
    "worst_mo": m_pnl.groupby(m_pnl.index.year).min(),
    "pos_mo_frac": m_ret.groupby(m_ret.index.year).apply(lambda s: (s > 0).mean()),
})
print(yr.round(2).to_string())

# ---------------------------------------------------------------- 2. exposure cadence
print("\n" + "=" * 90)
print("2. EXPOSURE CADENCE BY FRAGILITY BAND (2016-07+, daily)")
print("=" * 90)

# daily expansion: trade open Entry Date..Exit Date inclusive
trades["notional"] = trades["Shares_flat"].abs() * trades["Entry Price"]
days_idx = pd.bdate_range(trades["Entry Date"].min(), trades["Exit Date"].max())
open_ct = pd.Series(0, index=days_idx, dtype=float)
open_notional = pd.Series(0.0, index=days_idx)
open_risk = pd.Series(0.0, index=days_idx)
for _, tr in trades.iterrows():
    sl = slice(tr["Entry Date"], tr["Exit Date"])
    open_ct.loc[sl] += 1
    open_notional.loc[sl] += tr["notional"]
    open_risk.loc[sl] += tr["Risk_flat_750k"]

sig_ct = trades.groupby("Signal Date").size().reindex(days_idx, fill_value=0)

daily = pd.DataFrame({
    "open_ct": open_ct, "gross_pct": open_notional / ACCOUNT * 100,
    "risk_pct": open_risk / ACCOUNT * 100, "new_signals": sig_ct,
})
daily["frag"] = frag_ma.reindex(days_idx).ffill(limit=5)
d16 = daily.dropna(subset=["frag"])

BANDS = [0, 25, 44, 50, 55, 100]
LBL = ["0-25", "25-44", "44-50", "50-55", "55+"]
d16 = d16.copy()
d16["band"] = pd.cut(d16["frag"], BANDS, labels=LBL, include_lowest=True)
g = d16.groupby("band", observed=False)
exp_tab = pd.DataFrame({
    "days": g.size(),
    "avg_open_trades": g["open_ct"].mean(),
    "p90_open_trades": g["open_ct"].quantile(0.9),
    "avg_gross_pct": g["gross_pct"].mean(),
    "p90_gross_pct": g["gross_pct"].quantile(0.9),
    "avg_open_risk_pct": g["risk_pct"].mean(),
    "new_sigs_per_day": g["new_signals"].mean(),
})
print(exp_tab.round(2).to_string())
print(f"\nfull-sample gross exposure: avg {daily['gross_pct'].mean():.1f}%  "
      f"p90 {daily['gross_pct'].quantile(.9):.1f}%  max {daily['gross_pct'].max():.1f}%")
print(f"full-sample open trades: avg {daily['open_ct'].mean():.1f}  max {daily['open_ct'].max():.0f}")
print(f"share of days with ZERO open trades: {(daily['open_ct']==0).mean()*100:.1f}%  "
      f"(2016+: {(d16['open_ct']==0).mean()*100:.1f}%)")

# non-OVS split (OVS is exempt from the live throttle and behaves differently)
is_ovs = trades["Strategy"].str.contains("Overbot Vol", na=False)
open_ct_no = pd.Series(0.0, index=days_idx)
for _, tr in trades[~is_ovs].iterrows():
    open_ct_no.loc[tr["Entry Date"]:tr["Exit Date"]] += 1
d16["open_ct_nonovs"] = open_ct_no.reindex(d16.index)
print("\nnon-OVS avg open trades by band:")
print(d16.groupby("band", observed=False)["open_ct_nonovs"].mean().round(2).to_string())

# ---------------------------------------------------------------- 3. seasonality
print("\n" + "=" * 90)
print("3. SEASONALITY")
print("=" * 90)
cal = pd.DataFrame({
    "avg_pnl": m_pnl.groupby(m_pnl.index.month).mean(),
    "med_pnl": m_pnl.groupby(m_pnl.index.month).median(),
    "tot_pnl": m_pnl.groupby(m_pnl.index.month).sum(),
    "pos_frac": m_ret.groupby(m_ret.index.month).apply(lambda s: (s > 0).mean()),
    "n": m_pnl.groupby(m_pnl.index.month).size(),
})
print("--- by calendar month ---")
print(cal.round(2).to_string())

CYCLE = {0: "election", 1: "post-elec", 2: "midterm", 3: "pre-elec"}
cyc_key = pd.Series(m_pnl.index.year % 4, index=m_pnl.index).map(CYCLE)
cyc = pd.DataFrame({
    "avg_mo_pnl": m_pnl.groupby(cyc_key.values).mean(),
    "tot_pnl": m_pnl.groupby(cyc_key.values).sum(),
    "avg_mo_R": m_r.groupby(cyc_key.values).mean(),
    "n_months": m_pnl.groupby(cyc_key.values).size(),
    "pos_frac": m_ret.groupby(cyc_key.values).apply(lambda s: (s > 0).mean()),
})
print("\n--- by presidential cycle year (year%4; 2=midterm) ---")
print(cyc.round(2).to_string())

# ---------------------------------------------------------------- 4. concentration
print("\n" + "=" * 90)
print("4. CONCENTRATION")
print("=" * 90)
strat = trades.groupby("Strategy").agg(
    n=("R_Multiple", "size"), totR=("R_Multiple", "sum"),
    avgR=("R_Multiple", "mean"), pnl=("PnL_flat_750k", "sum"))
strat["pnl_share_pct"] = strat["pnl"] / strat["pnl"].sum() * 100
print("--- by strategy ---")
print(strat.sort_values("pnl", ascending=False).round(2).to_string())

print("\n--- by tier ---")
print(trades.groupby("Tier")["PnL_flat_750k"].agg(["count", "sum"]).to_string())
print("\n--- by direction ---")
print(trades.groupby("Direction")["PnL_flat_750k"].agg(["count", "sum"]).to_string())

# ticker families
LEV3X = set(trades.loc[trades["Strategy"] == "3x ETF Overbot Fade", "Ticker"].unique())
INDEX_CORE = {"SPY", "QQQ", "^GSPC", "^NDX", "IWM", "DIA", "MDY", "^RUT", "^DJI", "RSP", "^MID"}
SECTOR = {"XLF", "XLK", "XLE", "XLV", "XLI", "XLB", "XLU", "XLP", "XLY", "XLC", "XLRE",
          "SMH", "SOXX", "IBB", "XBI", "XHB", "ITB", "XRT", "KRE", "KBE", "IYR", "IYT",
          "OIH", "XOP", "XME", "GDX", "GDXJ", "KWEB", "TAN", "JETS", "ARKK", "IGV", "HACK"}
CMDTY_BOND_FX = {"GLD", "SLV", "USO", "UNG", "DBC", "DBA", "TLT", "IEF", "HYG", "LQD",
                 "SHY", "UUP", "FXE", "FXY", "CPER", "PPLT", "PALL", "CORN", "WEAT", "SOYB", "URA", "SLX"}
INTL = {t for t in trades["Ticker"].unique() if t.startswith("EW")} | {
    "EEM", "EFA", "FXI", "INDA", "EPOL", "TUR", "GREK", "ARGT", "NORW", "EZA", "ILF", "VNM", "THD", "PIN"}


def family(tk: str) -> str:
    if tk in INDEX_CORE:
        return "SPY/QQQ/index-core"
    if tk in LEV3X:
        return "3x levered ETF"
    if tk in SECTOR:
        return "sector/industry ETF"
    if tk in CMDTY_BOND_FX:
        return "commodity/bond/FX ETF"
    if tk in INTL:
        return "intl ETF"
    return "single stock"


trades["family"] = trades["Ticker"].map(family)
fam = trades.groupby("family").agg(
    n=("R_Multiple", "size"), pnl=("PnL_flat_750k", "sum"),
    avgR=("R_Multiple", "mean"), n_tickers=("Ticker", "nunique"))
fam["pnl_share_pct"] = fam["pnl"] / fam["pnl"].sum() * 100
print("\n--- by ticker family ---")
print(fam.sort_values("pnl", ascending=False).round(2).to_string())

top = trades.groupby("Ticker")["PnL_flat_750k"].sum().sort_values(ascending=False)
print(f"\ntop-4 tickers (SPY QQQ ^NDX ^GSPC) PnL share: "
      f"{top[['SPY','QQQ','^NDX','^GSPC']].sum()/top.sum()*100:.1f}%")
print(f"top-10 tickers PnL share: {top.head(10).sum()/top.sum()*100:.1f}%")
hh = ((top / top.sum()) ** 2).sum()
print(f"ticker HHI on PnL share: {hh:.4f}  (effective N ~ {1/hh:.0f})")

# ---------------------------------------------------------------- 5. sleeve spec inputs
print("\n" + "=" * 90)
print("5. SLEEVE SPEC INPUTS")
print("=" * 90)

# high-fragility months (2016+), monthly mean frag MA10 >= 50
mm = monthly.dropna(subset=["frag_avg"]).copy()
hi = mm[mm["frag_avg"] >= 50]
lo = mm[mm["frag_avg"] < 50]
print(f"high-frag months (avg frag>=50): {len(hi)} of {len(mm)} since 2016-07")
print(f"  book in high-frag months: avg PnL ${hi['pnl'].mean():,.0f}/mo  avg R {hi['R'].mean():+.2f}  "
      f"pos frac {(hi['pnl']>0).mean()*100:.0f}%")
print(f"  book in other months:     avg PnL ${lo['pnl'].mean():,.0f}/mo  avg R {lo['R'].mean():+.2f}  "
      f"pos frac {(lo['pnl']>0).mean()*100:.0f}%")
print("\nhigh-frag month list:")
print(hi[["pnl", "R", "n_exits", "frag_avg", "spy_ret_pct", "spy_dd_pct"]].round(2).to_string())

# frag>=44 (the decile-band knee) for a larger sample
hi44 = mm[mm["frag_avg"] >= 44]
print(f"\nfrag>=44 months: {len(hi44)}  avg PnL ${hi44['pnl'].mean():,.0f}  avg R {hi44['R'].mean():+.2f}  "
      f"pos frac {(hi44['pnl']>0).mean()*100:.0f}%")

# book monthly stats 2016+ (the era the fragility overlay exists)
m16 = m_ret[m_ret.index >= pd.Period("2016-07", "M")]
print(f"\nbook 2016-07+: Sharpe {m16.mean()/m16.std()*np.sqrt(12):.2f}  "
      f"avg mo {m16.mean()*100:+.3f}%  vol {m16.std()*100:.3f}%/mo")

# combined-Sharpe grid: sleeve at risk fraction f of book monthly vol
S_b = sharpe
print(f"\ncombined-Sharpe grid (book Sharpe {S_b:.2f} full-sample; sleeve vol = f x book vol):")
rows = []
for S_n in [0.3, 0.5, 0.8, 1.0]:
    for rho in [-0.3, 0.0, 0.3, 0.5]:
        for f in [0.25, 0.5, 1.0]:
            S_c = (S_b + f * S_n) / np.sqrt(1 + f**2 + 2 * rho * f)
            rows.append({"sleeve_S": S_n, "rho": rho, "vol_frac": f,
                         "combined_S": round(S_c, 3), "delta": round(S_c - S_b, 3)})
grid = pd.DataFrame(rows).pivot_table(index=["sleeve_S", "rho"], columns="vol_frac", values="combined_S")
print(grid.round(3).to_string())

# save monthly series for other tracks to consume
monthly.to_csv(ROOT / "scratch" / "ultracode_research" / "book_monthly_series.csv")
print("\nwrote book_monthly_series.csv (index=exit month; pnl/ret_pct/R/n_exits/frag/spy cols)")
