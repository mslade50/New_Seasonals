"""Shared loaders for the risk-architect sizing lens (2026-09-02).

Basis: flat $750k. Daily per-strategy MTM from dist/data/strategy_daily.json
(2026-08-07 site build, tiers collapsed); dial = 10d MA of the 63d column of
data/rd2_fragility.parquet (rows before 2026-07-02 are the recompute vintage);
PIT dial from the sibling cross_strategy_regime_pit_dial.parquet (column 'pit').
Nothing here writes into the repo.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
NAV = 750_000.0

THEMES: dict[str, list[str]] = {
    "dip_buy_index": ["SPY QQQ MonFri Reversion", "Indices Oversold Bounce", "Monday Dip", "Monthly Weak Close"],
    "dip_buy_stock": ["Weak Close Decent Sznls", "St OS Sznl"],
    "oversold_hold": ["Oversold Low Volume", "LT Trend ST OS"],
    "short_fade": ["Overbot Vol Spike", "3x ETF Overbot Fade", "ATR Extended Gap Up"],
    "bear_etf_fade": ["3x Bear ETF Overbot Fade", "3x Leader Gap Fade"],
    "breakout": ["52wh Breakout", "Sector BO"],
}
STRAT_TO_THEME = {s: t for t, ss in THEMES.items() for s in ss}


def load_strategy_daily() -> tuple[pd.DataFrame, pd.Series]:
    sd = json.load(open(ROOT / "dist/data/strategy_daily.json"))
    dates = pd.to_datetime(sd["dates"])
    S = pd.DataFrame(sd["series"], index=dates).fillna(0.0)
    strat = S.T.groupby(S.columns.str.split("||", regex=False).str[0]).sum().T
    total = pd.Series(sd["total_flat"], index=dates).fillna(0.0)
    return strat, total


def load_spy() -> pd.Series:
    px = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Close"],
                       filters=[("ticker", "in", ["SPY"])]).to_pandas()
    px = px.set_index("date")["Close"].sort_index()
    px.index = pd.to_datetime(px.index)
    return px


def load_prices(tickers: list[str], cols=("Close",)) -> pd.DataFrame:
    t = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", *cols],
                      filters=[("ticker", "in", list(tickers))]).to_pandas()
    t["date"] = pd.to_datetime(t["date"])
    return t


def load_dial(kind: str = "live") -> pd.Series:
    """10d MA of the 63d dial column, tz-naive daily index. kind = live | pit."""
    if kind == "live":
        f = pq.read_table(ROOT / "data/rd2_fragility.parquet").to_pandas()
        f.index = pd.to_datetime(f.index).tz_localize(None) if getattr(f.index, "tz", None) else pd.to_datetime(f.index)
        return f["63d"].rolling(10).mean().rename("dial")
    p = pq.read_table(OUT / "cross_strategy_regime_pit_dial.parquet").to_pandas()
    p.index = pd.to_datetime(p.index)
    return p["pit"].rename("dial")


def load_ledger() -> pd.DataFrame:
    df = pq.read_table(ROOT / "data/backtest_trades_full.parquet").to_pandas()
    for c in ("Signal Date", "Entry Date", "Exit Date"):
        df[c] = pd.to_datetime(df[c])
    df["theme"] = df["Strategy"].map(STRAT_TO_THEME)
    return df


def sessions(strat: pd.DataFrame, spy: pd.Series) -> pd.DatetimeIndex:
    return strat.index.intersection(spy.index)


def dial_bucket(x: pd.Series) -> pd.Series:
    return pd.cut(x, [-1, 30, 50, 65, 200], labels=["<30", "30-50", "50-65", "65+"])


def ann_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return dict(n=int(len(r)), mean_bps=float("nan"), sd_bps=float("nan"), sharpe=float("nan"))
    return dict(n=int(len(r)), mean_bps=float(r.mean() * 1e4), sd_bps=float(r.std() * 1e4),
                sharpe=float(r.mean() / r.std() * np.sqrt(252)))


def max_dd(r: pd.Series) -> float:
    eq = r.cumsum()
    return float((eq - eq.cummax()).min())


def dump(obj, name: str) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=1, default=_ser), encoding="utf-8")


def _ser(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.strftime("%Y-%m-%d")
    if isinstance(o, (np.ndarray, pd.Series)):
        return [float(x) for x in o]
    return str(o)
