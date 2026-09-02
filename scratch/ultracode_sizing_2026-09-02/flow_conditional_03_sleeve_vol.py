"""Flow-conditional sizing, part 2: does family / book signal flow forecast the sleeve's
NEXT-21d realized vol and mean (daily MTM from dist/data/strategy_daily.json)?

Usage: python flow_conditional_03_sleeve_vol.py [fills|candidates]
Outputs flow_conditional_sleeve_<src>.json
"""
from __future__ import annotations
import json
import sys
import numpy as np
import pandas as pd
from flow_conditional_lib import (load_ledger, load_candidates, load_strategy_daily, family_daily, daily_counts, trailing,
                                  spearman, FAMILIES, FAMILY, OUT_DIR, NAV)

pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
SRC = sys.argv[1] if len(sys.argv) > 1 else "fills"
OUT: dict = {"source": SRC}

strat, tot = load_strategy_daily()
fam = family_daily(strat) / NAV                      # daily return on NAV per family sleeve + book
if SRC == "candidates":
    sig = load_candidates().rename(columns={"strategy": "strategy"})
else:
    sig = load_ledger().rename(columns={"Signal Date": "signal_date", "Strategy": "strategy"})
sig["family"] = sig["strategy"].map(FAMILY)
cf = daily_counts(sig, "family")
cf["book"] = cf.sum(axis=1)

rows = []
per_year = []
for f in FAMILIES + ["book"]:
    r = fam[f].reindex(cf.index).dropna()
    r = r[r.index >= "2006-01-01"]
    fwd_vol = r.rolling(21).std().shift(-21) * np.sqrt(252)
    fwd_mean = r.rolling(21).mean().shift(-21) * 252
    fwd_sharpe = fwd_mean / fwd_vol
    trail_vol = r.rolling(21).std() * np.sqrt(252)
    for k in [5, 21, 63]:
        flow = trailing(cf, k)[f].reindex(r.index)
        flow_rel = flow / trailing(cf, k)[f].rolling(252, min_periods=126).mean().shift(k).reindex(r.index).replace(0, np.nan)
        for lab, x in [(f"flow{k}", flow), (f"flow{k}_rel", flow_rel)]:
            m = pd.DataFrame({"x": x, "v": fwd_vol, "mu": fwd_mean, "tv": trail_vol, "sh": fwd_sharpe}).dropna()
            # partial: does flow add to trailing vol for forecasting vol?
            X = np.column_stack([np.ones(len(m)), m.tv.rank() / len(m), m.x.rank() / len(m)])
            beta, *_ = np.linalg.lstsq(X, m.v.rank() / len(m), rcond=None)
            rows.append(dict(sleeve=f, var=lab, N=len(m), rho_vol=spearman(m.x, m.v), rho_mean=spearman(m.x, m.mu),
                             rho_sharpe=spearman(m.x, m.sh), rho_trailvol_vol=spearman(m.tv, m.v),
                             partial_flow_on_vol_given_trailvol=float(beta[2]),
                             hi_flow_fwd_vol=float(m.v[m.x >= m.x.quantile(2 / 3)].mean()), lo_flow_fwd_vol=float(m.v[m.x <= m.x.quantile(1 / 3)].mean()),
                             hi_flow_fwd_mean=float(m.mu[m.x >= m.x.quantile(2 / 3)].mean()), lo_flow_fwd_mean=float(m.mu[m.x <= m.x.quantile(1 / 3)].mean()),
                             hi_flow_fwd_sharpe=float(m.mu[m.x >= m.x.quantile(2 / 3)].mean() / m.v[m.x >= m.x.quantile(2 / 3)].mean()),
                             lo_flow_fwd_sharpe=float(m.mu[m.x <= m.x.quantile(1 / 3)].mean() / m.v[m.x <= m.x.quantile(1 / 3)].mean())))
            for y, my in m.groupby(m.index.year):
                if len(my) >= 120:
                    per_year.append(dict(sleeve=f, var=lab, year=int(y), rho_vol=spearman(my.x, my.v), rho_mean=spearman(my.x, my.mu)))
R = pd.DataFrame(rows)
print("=== next-21d realized vol / mean vs trailing flow (Spearman), by sleeve ===")
print(R[["sleeve", "var", "N", "rho_vol", "rho_mean", "rho_sharpe", "rho_trailvol_vol", "partial_flow_on_vol_given_trailvol",
         "lo_flow_fwd_vol", "hi_flow_fwd_vol", "lo_flow_fwd_mean", "hi_flow_fwd_mean", "lo_flow_fwd_sharpe", "hi_flow_fwd_sharpe"]].round(3).to_string(index=False))
Y = pd.DataFrame(per_year)
ys = Y.groupby(["sleeve", "var"]).agg(years=("year", "size"), pos_vol=("rho_vol", lambda s: (s > 0).mean()), pos_mean=("rho_mean", lambda s: (s > 0).mean()),
                                      mean_rho_vol=("rho_vol", "mean"), mean_rho_mean=("rho_mean", "mean")).reset_index()
print("\nper-year sign stability (share of years with positive rho):")
print(ys.round(3).to_string(index=False))
OUT["forecast"] = R.round(4).to_dict("records")
OUT["per_year"] = ys.round(4).to_dict("records")

# Kelly-relevant: forward mu/sigma^2 by flow tercile (does the growth-optimal size rise with flow?)
print("\n=== forward mu/var ratio (relative to sleeve unconditional) by flow5 tercile ===")
rows = []
for f in FAMILIES + ["book"]:
    r = fam[f].reindex(cf.index).dropna(); r = r[r.index >= "2006-01-01"]
    fwd_mean = r.rolling(21).mean().shift(-21); fwd_var = r.rolling(21).var().shift(-21)
    for k in [5, 21]:
        flow = trailing(cf, k)[f].reindex(r.index)
        m = pd.DataFrame({"x": flow, "mu": fwd_mean, "v": fwd_var}).dropna()
        base = m.mu.mean() / m.v.mean()
        q1, q2 = m.x.quantile(1 / 3), m.x.quantile(2 / 3)
        for lab, mm in [("lo", m[m.x <= q1]), ("mid", m[(m.x > q1) & (m.x <= q2)]), ("hi", m[m.x > q2])]:
            rows.append(dict(sleeve=f, k=k, bucket=lab, days=len(mm), fwd_mean_ann=mm.mu.mean() * 252 * 100, fwd_vol_ann=np.sqrt(mm.v.mean() * 252) * 100,
                             kelly_rel=(mm.mu.mean() / mm.v.mean()) / base if base else np.nan))
K = pd.DataFrame(rows)
print(K.round(3).to_string(index=False))
OUT["kelly_by_flow"] = K.round(4).to_dict("records")
json.dump(OUT, open(OUT_DIR / f"flow_conditional_sleeve_{SRC}.json", "w"), indent=1, default=float)
print(f"wrote flow_conditional_sleeve_{SRC}.json")
