"""Daily-MTM Sharpe for the OVS hold extension on the CLEANED universe
(ex-crypto, ex-non-aliased-carets), full period plus trailing 3y/5y windows."""
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

prices = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
prices["date"] = pd.to_datetime(prices["date"])
need = set(ovs["Ticker"].unique()) | {"SPY"}
prices = prices[prices["ticker"].isin(need)]
px = {t: g.set_index("date").sort_index() for t, g in prices.groupby("ticker")}

affected = (ovs["Exit Type"] == "Time") & (ovs["R_Multiple"] < 0)
new_exit = {}
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


def build_curve(extend_mask: pd.Series) -> pd.Series:
    daily = defaultdict(float)
    for idx, row in ovs.iterrows():
        if extend_mask.loc[idx] and idx in new_exit:
            ex_d, ex_p = new_exit[idx]
        else:
            ex_d, ex_p = row["Exit Date"], row["Exit Price"]
        df = px[row["Ticker"]]
        sh, prev = row["Shares_flat"], row["Entry Price"]
        for d, day in df.loc[row["Entry Date"]:ex_d].iterrows():
            mark = ex_p if d == ex_d else day["Close"]
            daily[d] += sh * (prev - mark)
            prev = mark
    return pd.Series(daily).sort_index()


cal = px["SPY"].index
cal = cal[(cal >= ovs["Entry Date"].min()) & (cal <= ASOF)]
variants = {
    "baseline": pd.Series(False, index=ovs.index),
    "extend ALL": pd.Series(True, index=ovs.index),
    "extend ATR%>=3": ovs["atr_pct"] >= 3,
}
windows = {"full (2003-)": None, "last 5y": ASOF - pd.DateOffset(years=5),
           "last 3y": ASOF - pd.DateOffset(years=3)}

curves = {n: build_curve(m).reindex(cal, fill_value=0.0) for n, m in variants.items()}
for wname, start in windows.items():
    print(f"\n--- {wname} ---")
    print(f"{'variant':16s} {'PnL$':>10} {'Sharpe':>7} {'vol%ann':>8} {'maxDD$':>9}")
    for n, pnl in curves.items():
        p = pnl if start is None else pnl[pnl.index >= start]
        r = p / BASIS
        curve = p.cumsum()
        dd = (curve - curve.cummax()).min()
        print(f"{n:16s} {p.sum():>10,.0f} {r.mean()/r.std()*np.sqrt(252):>7.2f} "
              f"{r.std()*np.sqrt(252)*100:>8.2f} {dd:>9,.0f}")
