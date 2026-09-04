"""Robustness pass on the OVS hold-extension deltas: yearly totals + t-stats
clustered by signal week (extensions on the same violent week are one bet)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ledger = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
ovs = ledger[ledger["Strategy"] == "Overbot Vol Spike"].copy()
for c in ("Signal Date", "Entry Date", "Exit Date"):
    ovs[c] = pd.to_datetime(ovs[c])

prices = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
prices["date"] = pd.to_datetime(prices["date"])
prices = prices[prices["ticker"].isin(ovs["Ticker"].unique())]
px = {t: g.set_index("date").sort_index() for t, g in prices.groupby("ticker")}

rank252 = {}
for t, df in px.items():
    ret = df["Close"].pct_change(252, fill_method=None)
    rank252[t] = ret.expanding(min_periods=252).rank(pct=True) * 100.0
ovs["rank252"] = [
    rank252[t].loc[d] if (t in rank252 and d in rank252[t].index) else np.nan
    for t, d in zip(ovs["Ticker"], ovs["Signal Date"])
]
ovs["atr_pct"] = ovs["ATR"] / ovs["Signal Close"] * 100.0

affected = (ovs["Exit Type"] == "Time") & (ovs["R_Multiple"] < 0)
delta = pd.Series(np.nan, index=ovs.index)
for idx, row in ovs[affected].iterrows():
    df = px.get(row["Ticker"])
    if df is None or row["Exit Date"] not in df.index:
        continue
    pos = df.index.get_loc(row["Exit Date"])
    ext = df.iloc[pos + 1 : pos + 4]
    if len(ext) < 3:
        continue
    tgt = row["Entry Price"] - row["tgt_atr"] * row["ATR"]
    new_exit = ext["Close"].iloc[-1]
    for _, day in ext.iterrows():
        if day["Low"] <= tgt:
            new_exit = tgt
            break
    delta.loc[idx] = (df["Close"].iloc[pos] - new_exit) / row["ATR"]

ext = ovs[delta.notna()].assign(d=delta[delta.notna()])
ext["week"] = ext["Signal Date"].dt.to_period("W")
ext["year"] = ext["Signal Date"].dt.year


def cluster_t(sub: pd.DataFrame) -> tuple[float, float, int]:
    cl = sub.groupby("week")["d"].sum()
    n = len(cl)
    if n < 3:
        return np.nan, np.nan, n
    t = cl.mean() / (cl.std(ddof=1) / np.sqrt(n))
    return t, cl.mean(), n


def loyo(sub: pd.DataFrame) -> float:
    worst = np.inf
    for y in sub["year"].unique():
        t, _, _ = cluster_t(sub[sub["year"] != y])
        worst = min(worst, t)
    return worst


for name, mask in [
    ("ALL", pd.Series(True, index=ext.index)),
    ("rank252 < 65", ext["rank252"] < 65),
    ("atr% >= 3", ext["atr_pct"] >= 3),
    ("rank<65 & atr%>=3", (ext["rank252"] < 65) & (ext["atr_pct"] >= 3)),
]:
    sub = ext[mask]
    t, m, n = cluster_t(sub)
    print(f"{name:22s} n={len(sub):3d} weeks={n:3d} sumR={sub['d'].sum():+7.1f} "
          f"t(week-clustered)={t:+.2f} LOYO-floor={loyo(sub):+.2f}")

print("\nYearly delta, ALL extensions:")
yr = ext.groupby("year")["d"].agg(["count", "sum"])
for y, r in yr.iterrows():
    print(f"  {y}: {r['sum']:+6.1f}R  ({int(r['count'])} trades)")

print("\nYearly delta, atr% >= 3 slice:")
yr = ext[ext["atr_pct"] >= 3].groupby("year")["d"].agg(["count", "sum"])
for y, r in yr.iterrows():
    print(f"  {y}: {r['sum']:+6.1f}R  ({int(r['count'])} trades)")
