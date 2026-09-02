"""Kelly-style read of the ledger: per-strategy edge/vol, implied fractions,
book-level Kelly multiple, strategy correlation + effective N, and OLV
concurrency. Flat $750k basis throughout."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
NAV = 750_000.0
df = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
df = df[df["PnL_flat_750k"].notna()].copy()
df["yr"] = df["Exit Date"].dt.year
years = df["yr"].max() - df["yr"].min() + 1

# ---------- per-strategy edge / vol / implied Kelly ----------
g = df.groupby("Strategy")
rows = []
for name, d in g:
    n = len(d)
    r = d["R_Multiple"].astype(float)
    risk = d["Risk_flat_750k"].astype(float)
    pnl = d["PnL_flat_750k"].astype(float)
    # per-trade fraction of NAV risked (effective)
    f_act = (risk / NAV).mean()
    # Kelly in units of "fraction of NAV at 1R": f* = E[R]/E[R^2] (exact for growth-optimal with small f)
    mu_r, m2 = r.mean(), (r**2).mean()
    f_kelly = mu_r / m2 if m2 > 0 else np.nan
    yrs_active = d["yr"].nunique()
    rows.append(dict(
        Strategy=name, N=n, per_yr=n / max(yrs_active, 1), avgR=mu_r, sdR=r.std(),
        win=(r > 0).mean(), risk_bps_eff=f_act * 1e4,
        f_kelly_bps=f_kelly * 1e4, half_kelly_bps=f_kelly * 5e3,
        act_over_halfK=(f_act) / (f_kelly / 2) if f_kelly > 0 else np.nan,
        pnl_yr=pnl.sum() / years, risk_dep_yr=risk.sum() / years,
        first=d["yr"].min(),
    ))
tab = pd.DataFrame(rows).sort_values("pnl_yr", ascending=False)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:,.2f}".format)
print("=== per-strategy (effective bps of NAV; f_kelly = E[R]/E[R^2]) ===")
print(tab.to_string(index=False))

# ---------- book-level Kelly multiple from daily MTM pnl ----------
dp = pd.read_parquet(ROOT / "data/backtest_daily_pnl.parquet").set_index("date")["pnl_flat"]
for label, s in [("full 2003+", dp), ("2016+", dp[dp.index >= "2016-01-01"]), ("2021+", dp[dp.index >= "2021-01-01"])]:
    mu, sd = s.mean() / NAV, s.std() / NAV
    kelly_mult = mu / sd**2          # multiple of CURRENT sizing that is full Kelly
    g_now = mu * 252 - 0.5 * (sd**2) * 252
    print(f"\n=== book daily MTM {label}: ann ret {mu*252:.1%} ann vol {sd*np.sqrt(252):.1%} "
          f"sharpe {mu/sd*np.sqrt(252):.2f} | full-Kelly = {kelly_mult:.2f}x current, half-Kelly = {kelly_mult/2:.2f}x, "
          f"growth now {g_now:.1%}/yr, at half-K {(kelly_mult/2)*mu*252 - 0.5*(kelly_mult/2)**2*sd**2*252:.1%}")

# ---------- strategy correlation (monthly realized-at-exit PnL) ----------
m = df.assign(month=df["Exit Date"].dt.to_period("M")).pivot_table(
    index="month", columns="Strategy", values="PnL_flat_750k", aggfunc="sum").fillna(0.0)
m = m[m.index >= pd.Period("2016-01", "M")]
c = m.corr()
w = m.std()
# effective N: (sum w)^2 / (w' C w)
wn = w / w.sum()
eff_n = 1.0 / float(wn.values @ c.values @ wn.values)
print(f"\n=== strategy monthly PnL corr 2016+: {len(c)} strats, vol-weighted effective N = {eff_n:.1f}")
avg_off = (c.values[np.triu_indices(len(c), 1)]).mean()
print(f"avg pairwise corr {avg_off:.2f}")
pairs = c.where(np.triu(np.ones(c.shape), 1).astype(bool)).stack().sort_values()
print("most correlated pairs:\n", pairs.tail(6).to_string())
print("most negative pairs:\n", pairs.head(4).to_string())
print("corr with book total:\n", m.corrwith(m.sum(axis=1)).sort_values(ascending=False).to_string())

# ---------- open-notional concentration (Rung 4 / gross as output) ----------
df["notional"] = df["Entry Price"] * df["Shares_flat"]
days = pd.bdate_range(df["Entry Date"].min(), df["Exit Date"].max())
# quick open-notional by strategy via cumulative add/remove
ev = pd.concat([
    df[["Entry Date", "Strategy", "notional"]].rename(columns={"Entry Date": "d"}),
    df[["Exit Date", "Strategy", "notional"]].rename(columns={"Exit Date": "d"}).assign(notional=lambda x: -x.notional),
])
ev = ev[ev["d"] >= "2016-01-01"]
open_n = ev.pivot_table(index="d", columns="Strategy", values="notional", aggfunc="sum").fillna(0).cumsum()
open_n = open_n.reindex(pd.bdate_range(open_n.index.min(), open_n.index.max())).ffill().fillna(0)
tot = open_n.sum(axis=1)
print(f"\n=== open notional 2016+ (% NAV): book mean {tot.mean()/NAV:.0%}, p95 {tot.quantile(.95)/NAV:.0%}, max {tot.max()/NAV:.0%}")
share = (open_n.clip(lower=0).mean() / open_n.clip(lower=0).mean().sum()).sort_values(ascending=False)
print("mean open-notional share by strategy:\n", share.head(6).to_string())
olv = open_n.get("Oversold Low Volume")
if olv is not None:
    print(f"OLV open notional %NAV: mean {olv.mean()/NAV:.0%}, p95 {olv.quantile(.95)/NAV:.0%}, max {olv.max()/NAV:.0%}")
    cnt = ev[ev.Strategy == "Oversold Low Volume"].assign(k=lambda x: np.sign(x.notional)).groupby("d")["k"].sum().cumsum()
    print(f"OLV concurrent legs: mean {cnt.mean():.1f}, p95 {cnt.quantile(.95):.0f}, max {cnt.max():.0f}")

# ---------- how much of a year's PnL is one strategy / worst-year concentration ----------
py = df.pivot_table(index="yr", columns="Strategy", values="PnL_flat_750k", aggfunc="sum").fillna(0)
py = py[py.index >= 2016]
print("\n=== per-year PnL share of top strategy 2016+ ===")
print((py.max(axis=1) / py.clip(lower=0).sum(axis=1)).round(2).to_string())
