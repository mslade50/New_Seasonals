"""Rung checks: (1) does avgR depend on ATR%% (Rung 1 sizing assumption),
(2) OLV stack correlation + marginal-leg edge (Rung 4/5),
(3) dip-buy family same-day clustering (Rung 3 across strategies),
(4) shrunk Kelly-proportional bps allocation vs current."""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
NAV = 750_000.0
pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:,.2f}".format)
df = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
df = df[df["PnL_flat_750k"].notna()].copy()
df["yr"] = df["Exit Date"].dt.year

# ---------- (1) avgR by ATR%% of price ----------
print("=== (1) avgR by ATR%% (Range %%) quintile, pooled and by strategy (2010+) ===")
d = df[df["yr"] >= 2010].copy()
d["q"] = pd.qcut(d["Range %"], 5, labels=["q1 lowvol", "q2", "q3", "q4", "q5 hivol"])
print(d.groupby("q", observed=True)["R_Multiple"].agg(["mean", "std", "count"]).to_string())
# within-strategy quintiles (so strategy mix doesn't confound)
d["qs"] = d.groupby("Strategy")["Range %"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["lo", "mid", "hi"]))
print("\nwithin-strategy vol terciles:")
t = d.pivot_table(index="Strategy", columns="qs", values="R_Multiple", aggfunc="mean", observed=True)
t["N"] = d.groupby("Strategy").size()
print(t.to_string())
print("\npooled within-strategy terciles:", d.groupby("qs", observed=True)["R_Multiple"].agg(["mean", "std", "count"]).round(3).to_dict())

# ---------- (2) OLV stack correlation ----------
print("\n=== (2) OLV: new leg vs already-open legs, trailing-63d return corr ===")
olv = df[(df.Strategy == "Oversold Low Volume") & (df["Entry Date"] >= "2016-01-01")].copy()
ticks = sorted(set(olv.Ticker))
tbl = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Close"],
                    filters=[("ticker", "in", ticks), ("date", ">=", pd.Timestamp("2015-06-01"))])
px = tbl.to_pandas().pivot(index="date", columns="ticker", values="Close").sort_index()
ret = px.pct_change()
rows = []
for _, r in olv.sort_values("Entry Date").iterrows():
    ed = r["Entry Date"]
    open_legs = olv[(olv["Entry Date"] < ed) & (olv["Exit Date"] >= ed) & (olv.Ticker != r.Ticker)]
    n_open_names = open_legs.Ticker.nunique()
    n_open_legs = len(open_legs)
    rho = np.nan
    if n_open_names:
        w = ret.loc[:ed].tail(63)
        cs = [w[r.Ticker].corr(w[t]) for t in open_legs.Ticker.unique() if t in w]
        rho = float(np.nanmean(cs)) if cs else np.nan
    rows.append(dict(entry=ed, ticker=r.Ticker, R=r.R_Multiple, pnl=r.PnL_flat_750k, n_names=n_open_names,
                     n_legs=n_open_legs, rho=rho, size_mult=r.Size_Mult))
o = pd.DataFrame(rows)
o["nb"] = pd.cut(o.n_names, [-1, 0, 2, 5, 99], labels=["0 open", "1-2", "3-5", "6+"])
print(o.groupby("nb", observed=True).agg(N=("R", "size"), avgR=("R", "mean"), sdR=("R", "std"),
                                         rho=("rho", "mean"), pnl=("pnl", "sum")).to_string())
o["rb"] = pd.cut(o.rho, [-1, 0.2, 0.4, 0.6, 1.01], labels=["rho<.2", ".2-.4", ".4-.6", ">.6"])
print("\nby avg corr with open legs (n_names>=1):")
print(o[o.n_names >= 1].groupby("rb", observed=True).agg(N=("R", "size"), avgR=("R", "mean"), sdR=("R", "std"),
                                                          n_names=("n_names", "mean"), pnl=("pnl", "sum")).to_string())
# Rung-5 marginal value: mu / (1 + (n-1) rho) relative to solo
o["kelly_marg"] = o.R / (1 + o.n_names * o.rho.fillna(0))
print("\nmean 'correlation-taxed' R by bucket (R / (1 + n*rho)):")
print(o.groupby("nb", observed=True)["kelly_marg"].mean().round(3).to_string())
# concurrent stack daily PnL vol: realized-at-exit cluster losses
worst = o.sort_values("pnl").head(8)[["entry", "ticker", "R", "pnl", "n_names", "rho"]]
print("\nworst 8 OLV legs (2016+):\n", worst.to_string(index=False))

# ---------- (3) dip-buy family same-day clustering ----------
print("\n=== (3) dip-buy family same signal-day clustering (2010+) ===")
FAM = ["Weak Close Decent Sznls", "SPY QQQ MonFri Reversion", "Monday Dip", "Indices Oversold Bounce",
       "Monthly Weak Close", "3x Bear ETF Overbot Fade", "LT Trend ST OS", "St OS Sznl"]
f = df[(df.Strategy.isin(FAM)) & (df["yr"] >= 2010)].copy()
per_day = f.groupby("Signal Date").agg(n_strats=("Strategy", "nunique"), n_trades=("Strategy", "size"),
                                       risk=("Risk_flat_750k", "sum"), pnl=("PnL_flat_750k", "sum"),
                                       avgR=("R_Multiple", "mean"))
per_day["nb"] = pd.cut(per_day.n_strats, [0, 1, 2, 3, 99], labels=["1 strat", "2", "3", "4+"])
print(per_day.groupby("nb", observed=True).agg(days=("pnl", "size"), avgR=("avgR", "mean"),
                                               risk_bps=("risk", lambda s: s.mean() / NAV * 1e4),
                                               pnl_day=("pnl", "mean"), pnl_sd=("pnl", "std"),
                                               pnl_tot=("pnl", "sum"), worst=("pnl", "min")).to_string())
# same-day cross-strategy outcome correlation: pairs of family trades on the same day
pairs = f.merge(f, on="Signal Date", suffixes=("_a", "_b"))
pairs = pairs[pairs.Strategy_a < pairs.Strategy_b]
print(f"same-day cross-strategy R corr (pairs N={len(pairs)}): {pairs.R_Multiple_a.corr(pairs.R_Multiple_b):.2f}")
same = pairs[pairs.Ticker_a == pairs.Ticker_b]
print(f"  of which SAME ticker: N={len(same)}, corr {same.R_Multiple_a.corr(same.R_Multiple_b):.2f}")
print("  same-ticker pairs by strategy pair:\n", same.groupby(["Strategy_a", "Strategy_b"]).size().sort_values(ascending=False).head(8).to_string())

# ---------- (4) shrunk Kelly-proportional allocation ----------
print("\n=== (4) shrunk Kelly-proportional bps vs current (total risk/yr held fixed, 2010+ stats) ===")
g = df[df["yr"] >= 2010].groupby("Strategy")
st = g["R_Multiple"].agg(N="size", mu="mean", m2=lambda s: (s**2).mean())
st["cur_bps"] = g["Risk_flat_750k"].mean() / NAV * 1e4
st["risk_yr"] = g["Risk_flat_750k"].sum()
st["trades_yr"] = g.size() / g["yr"].nunique()
book_mu, book_m2 = df[df["yr"] >= 2010].R_Multiple.mean(), (df[df["yr"] >= 2010].R_Multiple**2).mean()
N0 = 100
st["mu_shr"] = (st.N * st.mu + N0 * book_mu) / (st.N + N0)
st["m2_shr"] = (st.N * st.m2 + N0 * book_m2) / (st.N + N0)
st["f_raw"] = st.mu / st.m2
st["f_shr"] = st.mu_shr / st.m2_shr
# allocate: bps_s = c * f_shr_s ; choose c so that sum(bps_s * trades_yr) equals current sum
c = (st.cur_bps * st.trades_yr).sum() / (st.f_shr * st.trades_yr).sum()
st["kelly_bps"] = c * st.f_shr
st["ratio"] = st.kelly_bps / st.cur_bps
print(st[["N", "mu", "m2", "f_raw", "f_shr", "cur_bps", "kelly_bps", "ratio", "trades_yr"]].sort_values("ratio").to_string())
