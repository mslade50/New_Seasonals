"""OVS hold extension restricted to Path-1 (decisive gap) entries, cleaned
universe. Delta stats + windowed daily-MTM Sharpe. P1 = T+1 Open > Signal
Close + 0.25 ATR (verified 1:1 against Size_Mult in the ledger)."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASIS = 750_000.0
ASOF = pd.Timestamp("2026-07-10")

ledger = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
drop = ledger["Ticker"].str.endswith("-USD") | (
    ledger["Ticker"].str.startswith("^") & ~ledger["Ticker"].isin(["^GSPC", "^NDX"]))
ovs = ledger[(ledger["Strategy"] == "Overbot Vol Spike") & ~drop].copy()
for c in ("Signal Date", "Entry Date", "Exit Date"):
    ovs[c] = pd.to_datetime(ovs[c])
ovs["atr_pct"] = ovs["ATR"] / ovs["Signal Close"] * 100.0
ovs["p1"] = ovs["T+1 Open"] > ovs["Signal Close"] + 0.25 * ovs["ATR"]

prices = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
prices["date"] = pd.to_datetime(prices["date"])
need = set(ovs["Ticker"].unique()) | {"SPY"}
prices = prices[prices["ticker"].isin(need)]
px = {t: g.set_index("date").sort_index() for t, g in prices.groupby("ticker")}

affected = (ovs["Exit Type"] == "Time") & (ovs["R_Multiple"] < 0)
new_exit, delta = {}, pd.Series(np.nan, index=ovs.index)
for idx, row in ovs[affected].iterrows():
    df = px.get(row["Ticker"])
    if df is None or row["Exit Date"] not in df.index:
        continue
    pos = df.index.get_loc(row["Exit Date"])
    if abs(df["Close"].iloc[pos] / row["Exit Price"] - 1) > 0.01:
        continue
    ext = df.iloc[pos + 1 : pos + 4]
    if len(ext) < 3:
        continue
    tgt = row["Entry Price"] - row["tgt_atr"] * row["ATR"]
    ex_d, ex_p = ext.index[-1], ext["Close"].iloc[-1]
    for d, day in ext.iterrows():
        if day["Low"] <= tgt:
            ex_d, ex_p = d, tgt
            break
    new_exit[idx] = (ex_d, ex_p)
    delta.loc[idx] = (df["Close"].iloc[pos] - ex_p) / row["ATR"]

ovs["week"] = ovs["Signal Date"].dt.to_period("W")
ovs["year"] = ovs["Signal Date"].dt.year


def cluster_t(d, w):
    cl = d.groupby(w).sum()
    return cl.mean() / (cl.std(ddof=1) / np.sqrt(len(cl)))


def loyo(d, w, y):
    return min(cluster_t(d[y != yy], w[y != yy]) for yy in y.unique())


rep = delta.notna()
print(f"P1 trades: {ovs['p1'].sum()} | P2: {(~ovs['p1']).sum()}"
      f" | extensions on P1: {(rep & ovs['p1']).sum()} | on P2: {(rep & ~ovs['p1']).sum()}")
for name, m in [
    ("P1 only", rep & ovs["p1"]),
    ("P2 only (contrast)", rep & ~ovs["p1"]),
    ("P1 & ATR%>=3", rep & ovs["p1"] & (ovs["atr_pct"] >= 3)),
    ("P1 & ATR%<3", rep & ovs["p1"] & (ovs["atr_pct"] < 3)),
]:
    d, w, y = delta[m], ovs.loc[m, "week"], ovs.loc[m, "year"]
    print(f"{name:20s} ext={m.sum():3d} sumR={d.sum():+.1f} mean={d.mean():+.3f} "
          f"improved={(d > 0).mean() * 100:.0f}% t={cluster_t(d, w):+.2f} "
          f"LOYO={loyo(d, w, y):+.2f} minD={d.min():+.2f}")


def build_curve(extend_mask):
    daily = defaultdict(float)
    for idx, row in ovs.iterrows():
        if extend_mask.loc[idx] and idx in new_exit:
            ex_d, ex_p = new_exit[idx]
        else:
            ex_d, ex_p = row["Exit Date"], row["Exit Price"]
        df = px[row["Ticker"]]
        sh, prev = row["Shares_flat"], row["Entry Price"]
        for d, day in df.loc[row["Entry Date"]:ex_d].iterrows():
            daily[d] += sh * (prev - (ex_p if d == ex_d else day["Close"]))
            prev = ex_p if d == ex_d else day["Close"]
    return pd.Series(daily).sort_index()


cal = px["SPY"].index
cal = cal[(cal >= ovs["Entry Date"].min()) & (cal <= ASOF)]
variants = {
    "baseline": pd.Series(False, index=ovs.index),
    "extend P1 only": ovs["p1"],
    "extend P1 & ATR%>=3": ovs["p1"] & (ovs["atr_pct"] >= 3),
}
curves = {n: build_curve(m).reindex(cal, fill_value=0.0) for n, m in variants.items()}
for wname, start in [("full (2003-)", None), ("last 5y", ASOF - pd.DateOffset(years=5)),
                     ("last 3y", ASOF - pd.DateOffset(years=3))]:
    print(f"\n--- {wname} ---")
    print(f"{'variant':20s} {'PnL$':>10} {'Sharpe':>7} {'vol%ann':>8} {'maxDD$':>9}")
    for n, pnl in curves.items():
        p = pnl if start is None else pnl[pnl.index >= start]
        r = p / BASIS
        curve = p.cumsum()
        dd = (curve - curve.cummax()).min()
        print(f"{n:20s} {p.sum():>10,.0f} {r.mean() / r.std() * np.sqrt(252):>7.2f} "
              f"{r.std() * np.sqrt(252) * 100:>8.2f} {dd:>9,.0f}")
