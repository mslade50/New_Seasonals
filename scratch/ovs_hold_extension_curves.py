"""Export daily-MTM equity curves (flat $750k) for OVS baseline vs
extend-all-losers-to-T+5 (no filters), cleaned universe -> JSON for the chart."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(r"C:\Users\McKinley Slade\AppData\Local\Temp\claude\C--Users-McKinley-Slade-dev-New-Seasonals\87012aa5-7cfc-4fa8-be64-f7f63ea04c23\scratchpad") / "ovs_curves.json"
BASIS = 750_000.0
ASOF = pd.Timestamp("2026-07-10")

ledger = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
drop = ledger["Ticker"].str.endswith("-USD") | (
    ledger["Ticker"].str.startswith("^") & ~ledger["Ticker"].isin(["^GSPC", "^NDX"]))
ovs = ledger[(ledger["Strategy"] == "Overbot Vol Spike") & ~drop].copy()
for c in ("Signal Date", "Entry Date", "Exit Date"):
    ovs[c] = pd.to_datetime(ovs[c])

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


def build_curve(extend: bool) -> pd.Series:
    daily = defaultdict(float)
    for idx, row in ovs.iterrows():
        if extend and idx in new_exit:
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
base = build_curve(False).reindex(cal, fill_value=0.0)
ext = build_curve(True).reindex(cal, fill_value=0.0)


def stats(pnl: pd.Series) -> dict:
    r = pnl / BASIS
    cum = pnl.cumsum()
    dd = cum - cum.cummax()
    return {"pnl": round(pnl.sum()), "sharpe": round(r.mean() / r.std() * np.sqrt(252), 2),
            "maxdd": round(dd.min()), "worst_day": round(pnl.min())}


payload = {
    "dates": [d.strftime("%Y-%m-%d") for d in cal],
    "base": [round(v) for v in base.cumsum()],
    "ext": [round(v) for v in ext.cumsum()],
    "base_dd": [round(v) for v in (base.cumsum() - base.cumsum().cummax())],
    "ext_dd": [round(v) for v in (ext.cumsum() - ext.cumsum().cummax())],
    "stats": {"base": stats(base), "ext": stats(ext)},
    "n_extensions": len(new_exit),
    "n_trades": len(ovs),
}
OUT.write_text(json.dumps(payload))
print(f"wrote {OUT} | {len(cal)} days | base {payload['stats']['base']} | ext {payload['stats']['ext']}")
