"""Follow-ups for the gap map: book beta to SPY, worst-month attribution,
OVS vs non-OVS in high-frag months, partial-month check."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
ACCOUNT = 750_000.0
pd.set_option("display.width", 200)

trades = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
for c in ["Signal Date", "Entry Date", "Exit Date"]:
    trades[c] = pd.to_datetime(trades[c]).dt.normalize()
trades["exit_ym"] = trades["Exit Date"].dt.to_period("M")

monthly = pd.read_csv(ROOT / "scratch" / "ultracode_research" / "book_monthly_series.csv",
                      index_col=0)
monthly.index = pd.PeriodIndex(monthly.index, freq="M")

# ---- 1. book vs SPY monthly ----
m = monthly.dropna(subset=["spy_ret_pct"]).copy()
r_book, r_spy = m["ret_pct"], m["spy_ret_pct"]
rho = r_book.corr(r_spy)
beta = np.polyfit(r_spy, r_book, 1)[0]
up, dn = m[r_spy > 0], m[r_spy <= 0]
print(f"book vs SPY monthly: corr {rho:+.3f}  beta {beta:+.3f}  N={len(m)}")
print(f"  SPY-up months  (N={len(up)}): book avg {up['ret_pct'].mean():+.2f}%  pos frac {(up['ret_pct']>0).mean()*100:.0f}%")
print(f"  SPY-down months(N={len(dn)}): book avg {dn['ret_pct'].mean():+.2f}%  pos frac {(dn['ret_pct']>0).mean()*100:.0f}%")
dn5 = m[m["spy_ret_pct"] <= -4]
print(f"  SPY <= -4% months (N={len(dn5)}): book avg {dn5['ret_pct'].mean():+.2f}%  "
      f"pos frac {(dn5['ret_pct']>0).mean()*100:.0f}%  list:")
print(dn5[["pnl", "ret_pct", "spy_ret_pct", "frag_avg"]].round(2).to_string())
dn_corr = dn["ret_pct"].corr(dn["spy_ret_pct"])
print(f"  downside-only corr (SPY<=0 months): {dn_corr:+.3f}")

# ---- 2. worst-month strategy attribution ----
worst_months = monthly.nsmallest(10, "pnl").index
print("\n--- strategy attribution in the 10 worst months ---")
w = trades[trades["exit_ym"].isin(worst_months)]
att = w.pivot_table(index="Strategy", columns="exit_ym", values="PnL_flat_750k",
                    aggfunc="sum", fill_value=0)
att["TOTAL"] = att.sum(axis=1)
print(att.round(0).sort_values("TOTAL").to_string())

# ---- 3. OVS vs non-OVS monthly in high-frag months ----
is_ovs = trades["Strategy"].str.contains("Overbot Vol", na=False)
mo_ovs = trades[is_ovs].groupby("exit_ym")["PnL_flat_750k"].sum()
mo_non = trades[~is_ovs].groupby("exit_ym")["PnL_flat_750k"].sum()
hi_idx = monthly[(monthly["frag_avg"] >= 50)].index
lo_idx = monthly[(monthly["frag_avg"] < 50)].index
print("\n--- OVS vs non-OVS avg monthly PnL, 2016-07+ (reindexed 0-fill) ---")
all16 = monthly.dropna(subset=["frag_avg"]).index
mo_ovs = mo_ovs.reindex(all16, fill_value=0)
mo_non = mo_non.reindex(all16, fill_value=0)
for lbl, idx in [("high-frag (>=50)", hi_idx), ("other (<50)", lo_idx)]:
    print(f"{lbl:20s} N={len(idx):3d}  OVS ${mo_ovs.reindex(idx).mean():>8,.0f}/mo  "
          f"non-OVS ${mo_non.reindex(idx).mean():>8,.0f}/mo  "
          f"non-OVS pos frac {(mo_non.reindex(idx)>0).mean()*100:.0f}%")

# ---- 4. partial-month check on 2026-07 & recent months detail ----
jul = trades[trades["exit_ym"] == pd.Period("2026-07", "M")]
print(f"\n2026-07 exits so far: {len(jul)} trades, PnL ${jul['PnL_flat_750k'].sum():,.0f} "
      f"(exit dates {jul['Exit Date'].min().date()}..{jul['Exit Date'].max().date()}) — PARTIAL MONTH")
print(jul.groupby("Strategy")["PnL_flat_750k"].agg(["count", "sum"]).to_string())

# ---- 5. hold-length / horizon profile (the 'no slow sleeve' claim, quantified) ----
trades["hold_bd"] = np.busday_count(
    trades["Entry Date"].values.astype("datetime64[D]"),
    trades["Exit Date"].values.astype("datetime64[D]"))
print("\n--- holding period distribution (business days) ---")
print(trades["hold_bd"].describe(percentiles=[.25, .5, .75, .9, .95]).round(1).to_string())
print("share of trades held > 15 bd:", f"{(trades['hold_bd']>15).mean()*100:.1f}%")
print("share of PnL from trades held > 15 bd:",
      f"{trades.loc[trades['hold_bd']>15,'PnL_flat_750k'].sum()/trades['PnL_flat_750k'].sum()*100:.1f}%")

# annualized turnover: total notional traded / account
trades["notional"] = trades["Shares_flat"].abs() * trades["Entry Price"]
yrs = (trades["Exit Date"].max() - trades["Entry Date"].min()).days / 365.25
print(f"\nround-trip notional/yr: ${trades['notional'].sum()*2/yrs:,.0f}  "
      f"= {trades['notional'].sum()*2/yrs/ACCOUNT:.1f}x account/yr")

# ---- 6. calm->shock transition months: frag rising fast vs book PnL ----
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
fma = frag["63d"].dropna().rolling(10, min_periods=1).mean()
fma.index = pd.to_datetime(fma.index).normalize()
f_mo = fma.groupby(fma.index.to_period("M")).mean()
f_chg = f_mo.diff()
mm = monthly.dropna(subset=["frag_avg"]).copy()
mm["frag_chg"] = f_chg.reindex(mm.index)
mm = mm.dropna(subset=["frag_chg"])
rising = mm[mm["frag_chg"] >= 10]
falling = mm[mm["frag_chg"] <= -10]
flat = mm[mm["frag_chg"].abs() < 10]
print("\n--- book by monthly fragility CHANGE (2016-08+) ---")
for lbl, g in [("frag rising >=10", rising), ("frag falling <=-10", falling), ("|chg|<10", flat)]:
    print(f"{lbl:20s} N={len(g):3d}  avg PnL ${g['pnl'].mean():>8,.0f}  "
          f"avg R {g['R'].mean():+6.2f}  pos frac {(g['pnl']>0).mean()*100:.0f}%")
tt = stats.ttest_ind(rising["ret_pct"], mm.loc[~mm.index.isin(rising.index), "ret_pct"], equal_var=False)
print(f"rising vs rest: t={tt.statistic:+.2f} p={tt.pvalue:.3f} (monthly obs, no extra clustering needed)")

# ---- 7. worst-decile book months: what frag/spy state? ----
q10 = monthly["ret_pct"].quantile(0.1)
bad = monthly[monthly["ret_pct"] <= q10]
bad16 = bad.dropna(subset=["frag_avg"])
print(f"\nworst-decile months (ret<= {q10:.2f}%): N={len(bad)} total, {len(bad16)} since 2016-07")
print(f"  their frag_avg distribution: {bad16['frag_avg'].describe().round(1).to_dict()}")
print(f"  their SPY ret: mean {bad['spy_ret_pct'].mean():+.2f}%  "
      f"({(bad['spy_ret_pct']<0).mean()*100:.0f}% negative)")
all16_m = monthly.dropna(subset=["frag_avg"])
print(f"  base-rate frag_avg all 2016+ months: median {all16_m['frag_avg'].median():.1f}")
