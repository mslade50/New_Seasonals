"""True (longer-hold) momentum test: does buying strength pay at 1-3 month holds?

v1/v2 finding: at a few-days hold, buying strength loses to baseline (short-term
reversal). Classic momentum is a weeks-to-months edge. Here we test absolute and
cross-sectional momentum (12-1 and 6-1 month, standard skip-recent-month) held
21/42/63 sessions, plus the long-short (winners - losers) spread that defines the
momentum premium.

Run:  python scratch/momentum_longhold.py
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
HOLDS = [21, 42, 63]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    c = df["Close"]
    df["SMA200"] = c.rolling(200).mean()
    df["above_200"] = c > df["SMA200"]
    # standard momentum: total return over lookback, SKIP most recent 21 sessions
    df["mom_12_1"] = c.shift(21) / c.shift(252) - 1.0
    df["mom_6_1"] = c.shift(21) / c.shift(126) - 1.0
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

    frames = [add_indicators(g) for _, g in raw.groupby("ticker", sort=False) if len(g) >= 300]
    full = pd.concat(frames, ignore_index=True)

    tradable = (
        (full["Close"] >= MIN_PRICE) & (full["dollar_vol"] >= MIN_DOLLAR_VOL)
        & full["entry_open"].notna()
    )

    # cross-sectional daily rank among tradable names (NaN excludes a row)
    for col in ("mom_12_1", "mom_6_1"):
        m = full[col].where(tradable)
        full[f"{col}_rank"] = m.groupby(full["date"]).rank(pct=True)

    # precompute fwd returns once
    for H in HOLDS:
        full[f"ret_{H}"] = full[f"exit_{H}"] / full["entry_open"] - 1.0

    base = {H: full.loc[tradable, f"ret_{H}"].mean() for H in HOLDS}
    print("\nbaseline avg fwd return:  " + "   ".join(f"H{H} {base[H]*100:+.2f}%" for H in HOLDS))

    variants = {
        "A abs 12-1 > 20%, >200d":      tradable & (full["mom_12_1"] > 0.20) & full["above_200"],
        "B xsec top-decile 12-1, >200d": tradable & (full["mom_12_1_rank"] >= 0.90) & full["above_200"],
        "C xsec top-decile 6-1, >200d":  tradable & (full["mom_6_1_rank"] >= 0.90) & full["above_200"],
        "D xsec top-decile 12-1 (no trend)": tradable & (full["mom_12_1_rank"] >= 0.90),
        "E xsec top-5% 12-1, >200d":     tradable & (full["mom_12_1_rank"] >= 0.95) & full["above_200"],
        "F xsec BOTTOM-decile 12-1 (losers)": tradable & (full["mom_12_1_rank"] <= 0.10),
    }

    print("=" * 96)
    hdr = f"{'variant':<36}{'N':>9}"
    for H in HOLDS:
        hdr += f"{'avg%':>9}{'edge%':>8}{'hit%':>7}"
    print(hdr)
    print("-" * 96)
    rows = {}
    for name, mask in variants.items():
        sig = full[mask.fillna(False)]
        row = f"{name:<36}{len(sig):>9,}"
        avgs = {}
        for H in HOLDS:
            s = sig[sig[f"ret_{H}"].notna()]
            r = s[f"ret_{H}"]
            avgs[H] = r.mean()
            row += f"{r.mean()*100:>+9.2f}{(r.mean()-base[H])*100:>+8.2f}{(r>0).mean()*100:>6.1f}%"
        rows[name] = avgs
        print(row)

    # winners - losers spread (B top-decile vs F bottom-decile)
    print("-" * 96)
    win = rows["B xsec top-decile 12-1, >200d"]
    los = rows["F xsec BOTTOM-decile 12-1 (losers)"]
    spread = "  ".join(f"H{H} {(win[H]-los[H])*100:+.2f}%" for H in HOLDS)
    print(f"{'long-short spread (B - F)':<36}{'':>9}   {spread}")

    print("\n(edge = avg trade return minus unconditional baseline at same horizon)")
    print("(overlapping daily signals; gross, equal-weight, no costs)")


if __name__ == "__main__":
    main()
