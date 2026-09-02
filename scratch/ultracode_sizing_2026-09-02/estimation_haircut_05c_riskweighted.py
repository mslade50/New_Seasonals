"""Risk-weighted (dollar) view of the in-sample vs 2026 comparison.

The trade-level avgR is dominated by OVS rows (2,483 of 4,696 rows, 53%, but
only ~15% of PnL: scale-out tranches and 8-bps P2 rows count one each). The
quantity sizing cares about is PnL per dollar of risk = sum(PnL)/sum(Risk),
i.e. a risk-weighted avgR. Recomputes:
  (1) ledger PnL and R-per-risk by year, book and per strategy; reconciles the
      ledger's yearly PnL to dist/data/strategy_daily.json;
  (2) the 2026 ratio vs trailing-5y at the 2026 strategy RISK mix;
  (3) the historical distribution of that yearly ratio;
  (4) per-strategy 2026 vs 2021-25 on the risk-weighted basis;
  (5) 2026 vs prior midterm years on the same basis;
  (6) signal flow: trades per month by year (the other half of dollar PnL).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
OUT = ROOT / "scratch/ultracode_sizing_2026-09-02"
res: dict = {}
led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led = led[led["PnL_flat_750k"].notna()].copy()
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
led["Exit Date"] = pd.to_datetime(led["Exit Date"])
led["year"] = led["Signal Date"].dt.year
led["month"] = led["Signal Date"].dt.month
led["risk"] = led["Risk_flat_750k"].astype(float)
led["pnl"] = led["PnL_flat_750k"].astype(float)

# (1) yearly book PnL and R per risk
yr = led.groupby("year").agg(N=("pnl", "size"), pnl=("pnl", "sum"), risk=("risk", "sum"), sumR=("R_Multiple", "sum"), avgR=("R_Multiple", "mean"))
yr["R_per_risk"] = yr["pnl"] / yr["risk"]
yr["risk_per_trade"] = yr["risk"] / yr["N"]
sd_ = json.load(open(ROOT / "dist/data/strategy_daily.json"))
tot = pd.Series(sd_["total_flat"], index=pd.to_datetime(sd_["dates"]), dtype=float)
yr["daily_series_pnl"] = tot.groupby(tot.index.year).sum()
# exit-year attribution for the reconciliation (daily series books PnL when marked)
yr["pnl_by_exit_year"] = led.groupby(led["Exit Date"].dt.year)["pnl"].sum()
print(yr.round(3).to_string())
res["book_by_year"] = yr.round(4).reset_index().to_dict("records")
res["pnl_share_by_strategy"] = (led.groupby("Strategy")["pnl"].sum() / led["pnl"].sum()).round(4).to_dict()
res["row_share_by_strategy"] = (led.groupby("Strategy").size() / len(led)).round(4).to_dict()

# (2)+(3) yearly ratio vs trailing-5y at the year's strategy RISK mix, risk-weighted
rows = []
for y in range(2008, 2027):
    cur = led[led["year"] == y]
    prev = led[(led["year"] >= y - 5) & (led["year"] < y)]
    if y == 2026:
        prev = prev[prev["month"] <= 8]
    mix = cur.groupby("Strategy")["risk"].sum()
    pm = (prev.groupby("Strategy")["pnl"].sum() / prev.groupby("Strategy")["risk"].sum()).reindex(mix.index)
    ok = pm.notna()
    exp = float((mix[ok] * pm[ok]).sum() / mix[ok].sum())
    c = cur[cur["Strategy"].isin(mix.index[ok])]
    act = float(c["pnl"].sum() / c["risk"].sum())
    rows.append({"year": y, "N": int(len(c)), "risk_$": float(c["risk"].sum()), "pnl_$": float(c["pnl"].sum()), "R_per_risk": act, "trailing5_at_risk_mix": exp, "ratio": act / exp if exp > 0 else None})
t = pd.DataFrame(rows)
print("\nrisk-weighted yearly ratio:\n", t.round(3).to_string(index=False))
res["yearly_ratio_riskweighted"] = t.to_dict("records")
h = t[t["year"] < 2026]["ratio"].dropna()
res["yearly_ratio_riskweighted_summary"] = {"mean": float(h.mean()), "median": float(h.median()), "sd": float(h.std()), "min": float(h.min()),
                                            "share_below_0.5": float((h < 0.5).mean()), "share_below_0.7": float((h < 0.7).mean()),
                                            "ratio_2026": float(t.loc[t["year"] == 2026, "ratio"].iloc[0]), "rank_2026_of_19": int((h < t.loc[t["year"] == 2026, "ratio"].iloc[0]).sum() + 1)}
print(res["yearly_ratio_riskweighted_summary"])

# (4) per strategy 2026 vs 2021-25 risk-weighted
per = {}
for s, g in led.groupby("Strategy"):
    c = g[g["year"] == 2026]; p = g[(g["year"] >= 2021) & (g["year"] < 2026)]
    if len(c) == 0 or p["risk"].sum() == 0:
        continue
    rc = float(c["pnl"].sum() / c["risk"].sum()); rp = float(p["pnl"].sum() / p["risk"].sum())
    per[s] = {"N_2026": int(len(c)), "Rrisk_2026": rc, "Rrisk_2021_25": rp, "N_2021_25": int(len(p)), "ratio": rc / rp if rp > 0 else None,
              "pnl_2026": float(c["pnl"].sum()), "risk_2026": float(c["risk"].sum()), "avgR_2026_unweighted": float(c["R_Multiple"].mean())}
res["per_strategy_2026_vs_2021_25_riskweighted"] = per
print("\nper-strategy risk-weighted:\n", pd.DataFrame(per).T.round(3).to_string())

# (5) midterm years Jan-Aug, risk-weighted
ja = led[led["month"] <= 8]
mid = {}
for y in (2006, 2010, 2014, 2018, 2022, 2026):
    c = ja[ja["year"] == y]
    mid[y] = {"N": int(len(c)), "R_per_risk": float(c["pnl"].sum() / c["risk"].sum()), "pnl": float(c["pnl"].sum())}
non = ja[(ja["year"] % 4 != 2)]
res["midterm_riskweighted"] = {"by_year": mid, "nonmidterm_JanAug_R_per_risk": float(non["pnl"].sum() / non["risk"].sum()),
                               "midterm_pre2026_JanAug_R_per_risk": float(ja[(ja["year"] % 4 == 2) & (ja["year"] < 2026)]["pnl"].sum() / ja[(ja["year"] % 4 == 2) & (ja["year"] < 2026)]["risk"].sum())}
print("\nmidterm risk-weighted:", json.dumps(res["midterm_riskweighted"], indent=1))
# OVS specifically
o = ja[ja["Strategy"] == "Overbot Vol Spike"]
res["OVS_midterm_riskweighted"] = {int(y): float(g["pnl"].sum() / g["risk"].sum()) for y, g in o[o["year"] % 4 == 2].groupby("year")}
res["OVS_nonmidterm_riskweighted"] = float(o[o["year"] % 4 != 2]["pnl"].sum() / o[o["year"] % 4 != 2]["risk"].sum())
print("OVS midterm R/risk by year:", res["OVS_midterm_riskweighted"], "non-midterm:", round(res["OVS_nonmidterm_riskweighted"], 3))

# (6) signal flow
flow = led.groupby("year").size()
months = led.groupby("year")["month"].nunique()
res["trades_per_month_by_year"] = (flow / months).round(2).to_dict()
print("\ntrades per month by year:", res["trades_per_month_by_year"])
(OUT / "estimation_haircut_riskweighted.json").write_text(json.dumps(res, indent=1, default=str))
