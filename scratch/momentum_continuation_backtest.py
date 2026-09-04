"""Standalone scratch backtest: long momentum-continuation.

v1 idea: buy strength in an established uptrend, hold a few sessions, exit on
time. Trades the full CSV_UNIVERSE (liquid + overflow) off the local
master_prices.parquet cache. Pure signal-quality study first -- no stops/
targets -- benchmarked against the unconditional H-day forward return so we can
see whether the setup actually adds edge over a random entry.

Run:  python scratch/momentum_continuation_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from strategy_config import CSV_UNIVERSE, LIQUID_PLUS_COMMODITIES  # noqa: E402

PRICES_PATH = ROOT / "data" / "master_prices.parquet"

# ---- strategy params --------------------------------------------------------
BREAKOUT_LOOKBACK = 20      # new N-day closing high = continuation trigger
MOM_LOOKBACK = 20           # require positive N-day return (performing well)
MIN_PRICE = 5.0
MIN_DOLLAR_VOL = 1_000_000  # 20d avg dollar volume
STOP_ATR_MULT = 1.5         # R unit for avgR reporting (no stop applied to PnL)
HOLD_DAYS = [3, 5, 10]      # sweep
START_DATE = "2005-01-01"   # skip the thin early-2000s history


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    c, h, l = df["Close"], df["High"], df["Low"]
    df["SMA50"] = c.rolling(50).mean()
    df["SMA200"] = c.rolling(200).mean()
    df["SMA200_rising"] = df["SMA200"] > df["SMA200"].shift(1)

    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["ATR_pct"] = df["ATR"] / c * 100

    df["ret_mom"] = c / c.shift(MOM_LOOKBACK) - 1.0
    df["high_brk"] = c.rolling(BREAKOUT_LOOKBACK).max()
    df["is_new_high"] = c >= df["high_brk"]

    df["dollar_vol"] = (c * df["Volume"]).rolling(20).mean()

    # forward exits: enter NEXT open, exit close after H sessions
    df["entry_open"] = df["Open"].shift(-1)
    for H in HOLD_DAYS:
        df[f"exit_close_{H}"] = c.shift(-H)
    return df


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    sig = (
        (df["Close"] > df["SMA50"])
        & (df["SMA50"] > df["SMA200"])
        & df["SMA200_rising"]
        & df["is_new_high"]
        & (df["ret_mom"] > 0)
        & (df["Close"] >= MIN_PRICE)
        & (df["dollar_vol"] >= MIN_DOLLAR_VOL)
        & df["entry_open"].notna()
        & df["ATR"].notna()
        & (df["ATR"] > 0)
    )
    return df[sig]


def main() -> None:
    print(f"Loading {PRICES_PATH.name} ...", flush=True)
    raw = pd.read_parquet(PRICES_PATH)
    raw = raw[raw["ticker"].isin(set(CSV_UNIVERSE))].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[raw["date"] >= pd.Timestamp(START_DATE)]
    print(f"  {raw['ticker'].nunique()} tickers, {len(raw):,} rows from {START_DATE}", flush=True)

    liquid_set = set(LIQUID_PLUS_COMMODITIES)

    frames = []
    for tkr, g in raw.groupby("ticker", sort=False):
        if len(g) < 260:
            continue
        frames.append(add_indicators(g).assign(ticker=tkr))
    full = pd.concat(frames, ignore_index=True)

    sigs = build_signals(full)
    sigs = sigs.assign(tier=np.where(sigs["ticker"].isin(liquid_set), "Liquid", "Overflow"))
    print(f"\n{len(sigs):,} raw signals "
          f"({(sigs['tier']=='Liquid').sum():,} liquid / {(sigs['tier']=='Overflow').sum():,} overflow)\n")

    # ---- per-H stats --------------------------------------------------------
    for H in HOLD_DAYS:
        exit_col = f"exit_close_{H}"
        s = sigs[sigs[exit_col].notna()].copy()
        s["ret"] = s[exit_col] / s["entry_open"] - 1.0
        s["R"] = (s[exit_col] - s["entry_open"]) / (STOP_ATR_MULT * s["ATR"])

        # unconditional baseline over same horizon: all ticker-days, same exit math
        base = full[full[exit_col].notna() & full["entry_open"].notna()]
        base_r = (base[exit_col] / base["entry_open"] - 1.0)

        print(f"=== HOLD {H} sessions ===")
        print(f"  trades            {len(s):,}")
        print(f"  avg return        {s['ret'].mean()*100:+.3f}%   (baseline {base_r.mean()*100:+.3f}%  edge {(s['ret'].mean()-base_r.mean())*100:+.3f}%)")
        print(f"  median return     {s['ret'].median()*100:+.3f}%")
        print(f"  hit rate          {(s['ret']>0).mean()*100:.1f}%   (baseline {(base_r>0).mean()*100:.1f}%)")
        print(f"  avgR (R=1.5ATR)   {s['R'].mean():+.3f}   total {s['R'].sum():+.0f}R")
        print(f"  by tier  Liquid avg {s.loc[s['tier']=='Liquid','ret'].mean()*100:+.3f}%  "
              f"Overflow avg {s.loc[s['tier']=='Overflow','ret'].mean()*100:+.3f}%")
        print()

    # ---- yearly breakdown for the default H=5 ------------------------------
    H = 5
    s = sigs[sigs[f"exit_close_{H}"].notna()].copy()
    s["ret"] = s[f"exit_close_{H}"] / s["entry_open"] - 1.0
    s["year"] = pd.to_datetime(s["date"]).dt.year
    yr = s.groupby("year")["ret"].agg(["count", "mean"])
    yr["mean"] = (yr["mean"] * 100).round(3)
    print("=== H=5 by year (avg trade %) ===")
    print(yr.to_string())


if __name__ == "__main__":
    main()
