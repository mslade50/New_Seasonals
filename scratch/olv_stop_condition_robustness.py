"""Robustness checks for the OLV stop-variant study (run after
olv_stop_condition_study.py — reads its per-variant parquets)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCR = ROOT / "scratch"

VARS = ["prod", "no_stop", "eod_stop", "vol_stop_10", "vol_stop_15",
        "vol_stop_20", "vol15_dis25", "no_stop_dis25", "wide_20"]
res = {v: pd.read_parquet(SCR / f"olv_stopvar_{v}.parquet") for v in VARS}

ledger = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
olv = ledger[ledger["Strategy"] == "Oversold Low Volume"].copy()
olv["Signal Date"] = pd.to_datetime(olv["Signal Date"])
tier = olv.set_index(["Ticker", "Signal Date"])["Tier"]

key = ["Ticker", "Signal Date"]
base = res["prod"].set_index(key)
for v in VARS:
    res[v] = res[v].set_index(key)

# ---- yearly diff: vol_stop_15 and no_stop vs prod ----
print("--- yearly R diff vs prod (same entries, exit rule only) ---")
years = base.index.get_level_values("Signal Date").year
rows = []
for v in ["no_stop", "eod_stop", "vol_stop_15", "vol15_dis25"]:
    d = (res[v]["R"] - base["R"])
    yr = d.groupby(years).sum()
    rows.append(yr.rename(v))
yt = pd.concat(rows, axis=1).round(2)
yt["n_trades"] = base.groupby(years).size()
print(yt.to_string())

print("\n--- LOYO (leave-one-year-out) on the vol_stop_15 - prod diff ---")
d15 = (res["vol_stop_15"]["R"] - base["R"])
tot = d15.sum()
loyo = {y: tot - d15[years == y].sum() for y in sorted(set(years))}
print(f"full diff {tot:+.1f}R; LOYO min {min(loyo.values()):+.1f}R "
      f"(drop {min(loyo, key=loyo.get)}), max {max(loyo.values()):+.1f}R")
nz = d15[d15.abs() > 1e-9]
print(f"trades where the rule changes the outcome: {len(nz)} "
      f"(pos {(nz>0).sum()}, neg {(nz<0).sum()}, avg {nz.mean():+.2f}R)")

# episode clustering: group changed trades into ticker-chains, t-stat on chain sums
ch = nz.reset_index().sort_values(["Ticker", "Signal Date"])
ch["chain"] = (ch.groupby("Ticker")["Signal Date"].diff().dt.days.fillna(99) > 15).cumsum().astype(str) + "_" + ch["Ticker"]
chain_sums = ch.groupby("chain")["R"].sum()
t = chain_sums.mean() / (chain_sums.std(ddof=1) / np.sqrt(len(chain_sums)))
print(f"episode-clustered: {len(chain_sums)} chains, mean {chain_sums.mean():+.2f}R, t = {t:.2f}")
print(f"biggest single chain contribution: {chain_sums.max():+.1f}R / {chain_sums.min():+.1f}R")
print(f"drop-best-chain diff: {tot - chain_sums.max():+.1f}R")

# ---- tier split ----
print("\n--- tier split (survivorship flatters Overflow — diff shown per tier) ---")
tiers = base.join(tier).Tier
for v in ["no_stop", "vol_stop_15"]:
    d = (res[v]["R"] - base["R"])
    for t_ in ["Liquid", "Overflow"]:
        m = tiers == t_
        print(f"  {v:<13} {t_:<9} diff {d[m].sum():+7.1f}R over {m.sum()} trades "
              f"(prod totR {base.loc[m,'R'].sum():.1f})")

# ---- 2026 pain episodes ----
print("\n--- 2026 clusters under each rule (cluster R) ---")
for tkr in ["LMT", "LYB", "PBR", "DBC", "KT"]:
    m = base.index.get_level_values("Ticker") == tkr
    m26 = m & (years >= 2026)
    if not m26.any():
        continue
    line = f"  {tkr:<5}"
    for v in ["prod", "no_stop", "eod_stop", "vol_stop_15", "vol15_dis25"]:
        line += f" {v}={res[v].loc[m26,'R'].sum():+.2f}R"
    print(line)

# ---- stop->rebuy churn removal ----
print("\n--- same-day stop+rebuy churn ---")
for v in ["prod", "eod_stop", "vol_stop_15"]:
    d = res[v].reset_index()
    d = d.sort_values(["Ticker", "Entry Date"])
    n_churn = 0
    stops = d[d["Exit Type"].isin(["Stop", "EODStop"])]
    for _, r in stops.iterrows():
        g = d[(d["Ticker"] == r["Ticker"]) & (d["Entry Date"] >= r["Exit Date"]) &
              (d["Signal Date"] != r["Signal Date"])]
        if len(g) and (g["Entry Date"].min() - r["Exit Date"]).days <= 1:
            n_churn += 1
    print(f"  {v:<13} stops {len(stops):>4}  stop-then-rebuy-within-1-day {n_churn}")

# ---- smaller-size-no-stop arithmetic ----
print("\n--- size-down-no-stop vs volume-stop (flat-$ comparison) ---")
prod_d = (base["R"] * base["risk_$"]).sum()
for v in ["no_stop", "vol_stop_15"]:
    d = res[v]
    tot_d = (d["R"] * d["risk_$"]).sum()
    # worst same-ticker chain in $
    dd = d.reset_index().sort_values(["Ticker", "Entry Date"])
    worst_pnl = np.inf
    for tkr, g in dd.groupby("Ticker"):
        g = g.reset_index(drop=True)
        cur = [0]
        chains = []
        for i in range(1, len(g)):
            if g.loc[i, "Entry Date"] <= g.loc[cur, "Exit Date"].max() + pd.tseries.offsets.BDay(3):
                cur.append(i)
            else:
                chains.append(cur); cur = [i]
        chains.append(cur)
        for chn in chains:
            worst_pnl = min(worst_pnl, (g.loc[chn, "R"] * g.loc[chn, "risk_$"]).sum())
    print(f"  {v:<13} $PnL {tot_d:,.0f}  worst-chain ${worst_pnl:,.0f}")
prod_worst = -8841.0
ns_worst = None
print(f"  prod          $PnL {prod_d:,.0f}  worst-chain ${prod_worst:,.0f}")
ns = res["no_stop"]
ns_d = (ns["R"] * ns["risk_$"]).sum()
scale = prod_worst / -22518.0
print(f"  no_stop scaled to match prod worst-chain (x{scale:.2f}): "
      f"$PnL {ns_d*scale:,.0f} vs prod {prod_d:,.0f}")
