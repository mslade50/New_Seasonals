"""v2 entry-variant bake-off on the same plumbing as v1.

v1 finding: buying fresh 20d highs and holding a few days UNDERPERFORMS a
random entry (short-term reversal). Here we hold the trend context fixed
(Close>SMA50>SMA200, SMA200 rising, same liquidity) and swap only the *trigger*
to see which short-horizon entries actually beat the unconditional baseline.

Run:  python scratch/momentum_continuation_v2.py
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
MIN_PRICE = 5.0
MIN_DOLLAR_VOL = 1_000_000
START_DATE = "2005-01-01"
HOLDS = [5, 10]


def rsi(series: pd.Series, n: int) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    c, h, l = df["Close"], df["High"], df["Low"]
    df["SMA10"] = c.rolling(10).mean()
    df["SMA50"] = c.rolling(50).mean()
    df["SMA200"] = c.rolling(200).mean()
    df["SMA200_rising"] = df["SMA200"] > df["SMA200"].shift(1)

    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    df["RSI2"] = rsi(c, 2)
    df["today_ret_atr"] = (c - prev_c) / df["ATR"]
    df["down_day"] = c < prev_c
    df["new_high_20"] = c >= c.rolling(20).max()
    # 6-month momentum skipping the most recent 10 sessions (avoid short-term spike)
    df["mom_126_skip10"] = c.shift(10) / c.shift(126) - 1.0
    df["dollar_vol"] = (c * df["Volume"]).rolling(20).mean()

    df["entry_open"] = df["Open"].shift(-1)
    for H in HOLDS:
        df[f"exit_{H}"] = c.shift(-H)
    return df


def main() -> None:
    print(f"Loading {PRICES_PATH.name} ...", flush=True)
    raw = pd.read_parquet(PRICES_PATH)
    raw = raw[raw["ticker"].isin(set(CSV_UNIVERSE))].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[raw["date"] >= pd.Timestamp(START_DATE)]

    frames = [add_indicators(g) for _, g in raw.groupby("ticker", sort=False) if len(g) >= 260]
    full = pd.concat(frames, ignore_index=True)

    trend = (
        (full["Close"] > full["SMA50"]) & (full["SMA50"] > full["SMA200"]) & full["SMA200_rising"]
        & (full["Close"] >= MIN_PRICE) & (full["dollar_vol"] >= MIN_DOLLAR_VOL)
        & full["entry_open"].notna() & (full["ATR"] > 0)
    )

    variants = {
        "A breakout new-20d-high (v1)": full["new_high_20"] & (full["Close"] / full["Close"].shift(20) - 1 > 0),
        "B dip: down day & Close<SMA10": full["down_day"] & (full["Close"] < full["SMA10"]),
        "C oversold pullback RSI2<10":  full["RSI2"] < 10,
        "D big up day (>1 ATR)":        full["today_ret_atr"] > 1.0,
        "E 6mo-mom rank, not at high":  (full["mom_126_skip10"] > 0.15) & (~full["new_high_20"]),
        "F mom-leader(>15%) + RSI2<10":  (full["mom_126_skip10"] > 0.15) & (full["RSI2"] < 10),
        "G mom-leader(>25%) + RSI2<5":   (full["mom_126_skip10"] > 0.25) & (full["RSI2"] < 5),
    }

    # baselines per horizon (unconditional, all ticker-days)
    base = {}
    for H in HOLDS:
        m = full[f"exit_{H}"].notna() & full["entry_open"].notna()
        base[H] = (full.loc[m, f"exit_{H}"] / full.loc[m, "entry_open"] - 1).mean()

    print(f"\nbaseline avg fwd return:  " + "   ".join(f"H{H} {base[H]*100:+.3f}%" for H in HOLDS))
    print("=" * 92)
    hdr = f"{'variant':<32}{'N':>9}"
    for H in HOLDS:
        hdr += f"{'avg%':>9}{'edge%':>8}{'hit%':>7}"
    print(hdr)
    print("-" * 92)

    for name, mask in variants.items():
        sig = full[trend & mask.fillna(False)]
        row = f"{name:<32}{len(sig):>9,}"
        for H in HOLDS:
            s = sig[sig[f"exit_{H}"].notna()]
            r = s[f"exit_{H}"] / s["entry_open"] - 1
            row += f"{r.mean()*100:>+9.3f}{(r.mean()-base[H])*100:>+8.3f}{(r>0).mean()*100:>6.1f}%"
        print(row)

    print("\n(edge = avg trade return minus unconditional baseline at same horizon)")


if __name__ == "__main__":
    main()
