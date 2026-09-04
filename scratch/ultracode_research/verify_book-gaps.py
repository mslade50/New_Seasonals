"""Adversarial recompute of the book-gaps track's decisive claims.

Independent implementation — nothing imported from book_gap_map.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")
BASE = 750_000.0

led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led["exit_month"] = led["Exit Date"].dt.to_period("M")
led["notional"] = led["Entry Price"] * led["Shares_flat"]

# ---------- Claim 1: monthly stats ----------
mo = led.groupby("exit_month").agg(
    pnl=("PnL_flat_750k", "sum"), R=("R_Multiple", "sum"), n=("trade_id", "count")
)
mo["ret_pct"] = mo["pnl"] / BASE * 100
n_months = len(mo)
avg_pnl = mo["pnl"].mean()
vol_pct = mo["ret_pct"].std(ddof=1)
sharpe = (mo["ret_pct"].mean() / vol_pct) * np.sqrt(12)
cum = mo["pnl"].cumsum()
dd = cum - cum.cummax()
maxdd = dd.min()
pos_frac = (mo["pnl"] > 0).mean()
mo2016 = mo[mo.index >= pd.Period("2016-07", "M")]
sharpe16 = (mo2016["ret_pct"].mean() / mo2016["ret_pct"].std(ddof=1)) * np.sqrt(12)
print("== Claim 1: monthly stats ==")
print(f"months={n_months} avg=${avg_pnl:,.0f}/mo (${avg_pnl*12:,.0f}/yr) "
      f"vol={vol_pct:.2f}%/mo Sharpe={sharpe:.2f} (2016+={sharpe16:.2f})")
print(f"maxDD=${maxdd:,.0f} ({maxdd/BASE*100:.2f}%) pos={pos_frac:.1%} "
      f"total=${mo['pnl'].sum():,.0f} totR={mo['R'].sum():,.0f}")

# ---------- fragility live basis ----------
frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
ma10 = frag["63d"].rolling(10, min_periods=1).mean()
frag_mo = ma10.groupby(ma10.index.to_period("M")).agg(["mean", "max"])
frag_mo.columns = ["frag_avg", "frag_max"]

moj = mo.join(frag_mo, how="left")

# ---------- Claim 2: high-frag months ----------
hf = moj[moj["frag_avg"] >= 50]
print("\n== Claim 2: high-frag months (frag_avg>=50) ==")
print(f"months with frag data: {moj['frag_avg'].notna().sum()}")
print(f"N={len(hf)} avg=${hf['pnl'].mean():,.0f} pos={(hf['pnl']>0).mean():.1%}")
led_f = led.merge(frag_mo, left_on="exit_month", right_index=True, how="left")
nonovs = led_f[led_f["Strategy"] != "Overbot Vol Spike"]
nvmo = nonovs.groupby("exit_month")["PnL_flat_750k"].sum().to_frame("pnl").join(frag_mo)
nvhf = nvmo[nvmo["frag_avg"] >= 50]
print(f"non-OVS in those months: N={len(nvhf)} avg=${nvhf['pnl'].mean():,.0f} "
      f"pos={(nvhf['pnl']>0).mean():.1%}")

# ---------- Claim 3: worst-decile months are low-frag ----------
thr = mo["ret_pct"].quantile(0.10)
worst_dec = moj[moj["ret_pct"] <= thr]
scored = worst_dec["frag_avg"].dropna()
print("\n== Claim 3: worst-decile months vs fragility ==")
print(f"decile threshold={thr:.2f}% N_worst={len(worst_dec)} "
      f"(scored: {len(scored)}) median frag of worst={scored.median():.1f} "
      f"base median frag={moj['frag_avg'].median():.1f}")
w10 = moj.nsmallest(10, "pnl")
print("10 worst months:")
print(w10[["pnl", "R", "frag_avg"]].to_string(
    float_format=lambda x: f"{x:,.1f}"))
print(f"worst-10 above frag 44: {(w10['frag_avg'] > 44).sum()} "
      f"(unscored: {w10['frag_avg'].isna().sum()})")

# ---------- Claim 4: SPY relationship ----------
mp = pd.read_parquet(ROOT / "data/master_prices.parquet",
                     columns=["ticker", "date", "Close"])
spy = mp[mp["ticker"] == "SPY"].set_index("date")["Close"].sort_index()
spy_mo = spy.groupby(spy.index.to_period("M")).last()
spy_ret = spy_mo.pct_change() * 100
both = moj.join(spy_ret.rename("spy_ret"), how="inner").dropna(subset=["spy_ret"])
corr = both["ret_pct"].corr(both["spy_ret"])
dn = both[both["spy_ret"] <= 0]
dn_corr = dn["ret_pct"].corr(dn["spy_ret"])
crash = both[both["spy_ret"] <= -4]
print("\n== Claim 4: SPY relationship ==")
print(f"corr={corr:.3f} downside(SPY<=0, N={len(dn)}) corr={dn_corr:.3f}")
print(f"SPY<=-4% months: N={len(crash)} book avg={crash['ret_pct'].mean():+.2f}% "
      f"pos={(crash['ret_pct']>0).mean():.1%}")
beta = np.polyfit(both["spy_ret"], both["ret_pct"], 1)[0]
print(f"beta={beta:.2f}")

# ---------- Claim 5: worst-10 attribution ----------
w10_months = set(w10.index)
in_w10 = led[led["exit_month"].isin(w10_months)]
attr = in_w10.groupby("Strategy")["PnL_flat_750k"].sum().sort_values()
print("\n== Claim 5: strategy attribution in 10 worst months ==")
print(attr.to_string(float_format=lambda x: f"${x:,.0f}"))

# ---------- Claim 6: exposure by frag band ----------
days = ma10.index  # business days 2016-07-05..2026-07-01
ent = led["Entry Date"].values.astype("datetime64[ns]")
exi = led["Exit Date"].values.astype("datetime64[ns]")
noti = led["notional"].values
risk = led["Risk_flat_750k"].values
is_ovs = (led["Strategy"] == "Overbot Vol Spike").values
sig_counts = led.groupby("Signal Date").size()

rows = []
for d in days:
    d64 = np.datetime64(d)
    open_mask = (ent <= d64) & (exi >= d64)
    rows.append({
        "date": d,
        "n_open": open_mask.sum(),
        "gross_pct": noti[open_mask].sum() / BASE * 100,
        "risk_pct": risk[open_mask].sum() / BASE * 100,
        "n_open_nonovs": (open_mask & ~is_ovs).sum(),
        "new_sigs": sig_counts.get(d, 0),
    })
panel = pd.DataFrame(rows).set_index("date")
panel["frag"] = ma10
bands = [(0, 25), (25, 44), (44, 50), (50, 55), (55, 999)]
print("\n== Claim 6: exposure by frag band (open incl. exit day) ==")
for lo, hi in bands:
    b = panel[(panel["frag"] >= lo) & (panel["frag"] < hi)]
    print(f"{lo}-{hi if hi<999 else '+'}: days={len(b)} open={b['n_open'].mean():.2f} "
          f"gross={b['gross_pct'].mean():.1f}% p90={b['gross_pct'].quantile(.9):.1f}% "
          f"risk={b['risk_pct'].mean():.2f}% sigs/d={b['new_sigs'].mean():.2f} "
          f"nonOVSopen={b['n_open_nonovs'].mean():.2f}")
# sensitivity: exclude exit day
rows2 = []
for d in days:
    d64 = np.datetime64(d)
    m = (ent <= d64) & (exi > d64)
    rows2.append(noti[m].sum() / BASE * 100)
panel["gross_excl_exit"] = rows2
for lo, hi in [(0, 25), (50, 55)]:
    b = panel[(panel["frag"] >= lo) & (panel["frag"] < hi)]
    print(f"  sens excl-exit-day {lo}-{hi}: gross={b['gross_excl_exit'].mean():.1f}% "
          f"p90={b['gross_excl_exit'].quantile(.9):.1f}%")
print(f"full-sample (2016+) avg gross={panel['gross_pct'].mean():.1f}% "
      f"p90={panel['gross_pct'].quantile(.9):.1f}% max={panel['gross_pct'].max():.0f}% "
      f"zero-open days={(panel['n_open']==0).mean():.1%}")

# ---------- Claim 7: seasonality ----------
mo_cal = mo.copy()
mo_cal["cal_m"] = mo_cal.index.month
mo_cal["year"] = mo_cal.index.year
jas = mo_cal[mo_cal["cal_m"].isin([7, 8, 9])]
rest = mo_cal[~mo_cal["cal_m"].isin([7, 8, 9])]
print("\n== Claim 7: seasonality ==")
print(f"Jul-Sep avg=${jas['pnl'].mean():,.0f} (N={len(jas)}) "
      f"other=${rest['pnl'].mean():,.0f} (N={len(rest)})")
cyc_name = {0: "election", 1: "post-election", 2: "midterm", 3: "pre-election"}
for c in [1, 0, 3, 2]:
    g = mo_cal[mo_cal["year"] % 4 == c]
    print(f"{cyc_name[c]}: avg=${g['pnl'].mean():,.0f} pos={(g['pnl']>0).mean():.2f} "
          f"N={len(g)}")

# ---------- Claim 8: horizon / turnover / asset mix ----------
hold = np.busday_count(led["Entry Date"].values.astype("datetime64[D]"),
                       led["Exit Date"].values.astype("datetime64[D]"))
led["hold_bd"] = hold
long_holds = led[led["hold_bd"] > 15]
tot_pnl = led["PnL_flat_750k"].sum()
years = (led["Exit Date"].max() - led["Entry Date"].min()).days / 365.25
oneway = led["notional"].sum()
print("\n== Claim 8: horizon / turnover ==")
print(f"median hold={np.median(hold):.0f}bd p90={np.percentile(hold,90):.0f} "
      f">15bd: {len(long_holds)/len(led):.1%} of trades, "
      f"{long_holds['PnL_flat_750k'].sum()/tot_pnl:.1%} of PnL")
print(f"years={years:.1f} one-way notional=${oneway/1e6:,.0f}M "
      f"round-trip/yr=${2*oneway/years/1e6:,.1f}M = {2*oneway/years/BASE:.0f}x account")
# asset-class mix: classify commodity/bond/FX ETFs
CBFX = {"GLD", "SLV", "GDX", "GDXJ", "USO", "UNG", "DBC", "DBA", "DBO", "UGA",
        "CORN", "WEAT", "SOYB", "CPER", "PALL", "PPLT", "SIL", "URA", "COPX",
        "TLT", "IEF", "SHY", "AGG", "BND", "LQD", "HYG", "JNK", "TBT", "TIP",
        "EMB", "MUB", "ZROZ", "EDV", "TMF", "TMV",
        "UUP", "UDN", "FXE", "FXY", "FXB", "FXA", "FXC", "FXF", "CYB"}
led["is_cbfx"] = led["Ticker"].isin(CBFX)
cbfx_pnl = led.loc[led["is_cbfx"], "PnL_flat_750k"].sum()
print(f"commodity/bond/FX tickers hit: "
      f"{sorted(led.loc[led['is_cbfx'],'Ticker'].unique())}")
print(f"CBFX PnL share={cbfx_pnl/tot_pnl:.1%} (${cbfx_pnl:,.0f})")

# ---------- Claim 9: marginal Sharpe arithmetic ----------
print("\n== Claim 9: marginal Sharpe ==")
Sb = sharpe
for rho, f, Ss in [(0.0, 0.25, 0.5), (0.0, 0.50, 0.5), (0.3, 0.25, 0.5),
                   (0.3, 0.0001, 0.65)]:
    comb = (Sb + Ss * f) / np.sqrt(1 + f**2 + 2 * rho * f)
    print(f"rho={rho} f={f} Ss={Ss}: combined={comb:.3f} (book {Sb:.3f})")
print(f"hurdle at rho=0.3: S > {0.3*Sb:.2f}; at rho=0.2: S > {0.2*Sb:.2f}")

# ---------- extra: does 2026-07 partial month distort? ----------
print("\n== sensitivity: drop 2026-07 partial month ==")
mo_x = mo[mo.index < pd.Period("2026-07", "M")]
print(f"avg=${mo_x['pnl'].mean():,.0f} "
      f"Sharpe={(mo_x['ret_pct'].mean()/mo_x['ret_pct'].std(ddof=1))*np.sqrt(12):.2f}")
thr_x = mo_x["ret_pct"].quantile(0.10)
print(f"worst-decile thr w/o 2026-07: {thr_x:.2f}%")
