from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
NAV = 750_000.0
pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:,.2f}".format)
df = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
df = df[df["PnL_flat_750k"].notna()].copy()

# ---- same-ticker OLV stacking (rho = 1 line items) ----
olv = df[(df.Strategy == "Oversold Low Volume") & (df["Entry Date"] >= "2010-01-01")].copy().sort_values("Entry Date")
n_same, ep = [], []
for _, r in olv.iterrows():
    o = olv[(olv.Ticker == r.Ticker) & (olv["Entry Date"] < r["Entry Date"]) & (olv["Exit Date"] >= r["Entry Date"])]
    n_same.append(len(o))
olv["n_same"] = n_same
print("=== OLV legs by # already-open legs in the SAME ticker (2010+) ===")
print(olv.groupby("n_same").agg(N=("R_Multiple", "size"), avgR=("R_Multiple", "mean"), sdR=("R_Multiple", "std"),
                                win=("R_Multiple", lambda s: (s > 0).mean()), pnl=("PnL_flat_750k", "sum"),
                                risk_bps=("Risk_flat_750k", lambda s: s.mean() / NAV * 1e4)).to_string())
# episode = chain of overlapping legs in one ticker
olv["ep"] = ((olv.n_same == 0).cumsum())
ep = olv.groupby(["Ticker", "ep"]).agg(legs=("R_Multiple", "size"), pnl=("PnL_flat_750k", "sum"),
                                       risk=("Risk_flat_750k", "sum"), start=("Entry Date", "min"))
ep["R_stack"] = ep.pnl / (ep.risk / ep.legs)
print("\nticker-episodes by stack depth:")
print(ep.groupby("legs").agg(N=("pnl", "size"), pnl_mean=("pnl", "mean"), pnl_sd=("pnl", "std"),
                             worst=("pnl", "min"), tot=("pnl", "sum")).to_string())
print("\nworst 8 ticker-episodes:\n", ep.sort_values("pnl").head(8).to_string())
print("\nsingle-leg episode pnl sd:", ep[ep.legs == 1].pnl.std().round(0),
      "| 3+-leg episode pnl sd:", ep[ep.legs >= 3].pnl.std().round(0))

# ---- exact log-growth optimum per strategy: f* = argmax E[log(1 + f R)] ----
print("\n=== exact growth-optimal f* (fraction of equity per 1R) vs quadratic mu/E[R^2], 2010+ ===")
d = df[df["Exit Date"].dt.year >= 2010]
fs = np.linspace(0.005, 0.95, 400)
rows = []
for name, g in d.groupby("Strategy"):
    r = g.R_Multiple.values
    r = r[r > -1 / 0.95]  # guard
    growth = [np.mean(np.log1p(f * r)) if (1 + f * r).min() > 0 else -np.inf for f in fs]
    fstar = fs[int(np.argmax(growth))]
    rows.append(dict(Strategy=name, N=len(r), f_exact=fstar, f_quad=r.mean() / (r**2).mean(),
                     skew=pd.Series(r).skew(), cur_frac=g.Risk_flat_750k.mean() / NAV))
t = pd.DataFrame(rows)
t["exact_over_quad"] = t.f_exact / t.f_quad
t["cur_over_halfexact"] = t.cur_frac / (t.f_exact / 2)
print(t.sort_values("f_exact").to_string(index=False))
