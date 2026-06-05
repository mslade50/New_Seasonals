"""Functional check for the ISOLATED staging price path: build_overflow_universe
must screen master_prices UNION overflow_prices (staging), de-duping overlaps.
Pure: no network, no R2. Uses temp parquets + monkeypatched module paths.
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import build_overflow_universe as bou

AS_OF = pd.Timestamp("2025-01-10")
_n = 0


def check(cond, msg):
    global _n
    _n += 1
    if not cond:
        raise AssertionError(f"FAIL: {msg}")
    print(f"  ok: {msg}")


def _mk(ticker, n_bars, price, volume, atr_frac=0.04, last_date=AS_OF):
    dates = pd.bdate_range(end=last_date, periods=n_bars)
    close = np.full(n_bars, float(price))
    return pd.DataFrame({
        "ticker": ticker, "date": dates,
        "High": close * (1 + atr_frac / 2), "Low": close * (1 - atr_frac / 2),
        "Close": close, "Volume": np.full(n_bars, float(volume)),
    })


def main():
    with tempfile.TemporaryDirectory() as td:
        master_p = os.path.join(td, "master_prices.parquet")
        overflow_p = os.path.join(td, "overflow_prices.parquet")

        # master: one liquid name + one already-known overflow name
        pd.concat([
            _mk("AAPL", 300, 200.0, 5_000_000),     # liquid (excluded by screen)
            _mk("OLDOF", 300, 40.0, 300_000),        # known overflow, $12MM ADDV
        ], ignore_index=True).to_parquet(master_p, index=False)

        # staging: two genuinely-new names
        pd.concat([
            _mk("NEWBIG", 300, 60.0, 200_000),       # $12MM ADDV -> pass
            _mk("NEWTHIN", 300, 5.0, 50_000),        # $0.25MM ADDV -> fail liquidity
        ], ignore_index=True).to_parquet(overflow_p, index=False)

        bou.MASTER_PRICES_PATH = master_p
        bou.OVERFLOW_PRICES_PATH = overflow_p

        prices = bou._load_prices()
        tickers = set(prices["ticker"].unique())
        check(tickers == {"AAPL", "OLDOF", "NEWBIG", "NEWTHIN"}, f"union of both caches (got {tickers})")

        out = bou.screen_universe(prices, {"AAPL"}, as_of=AS_OF)
        passed = set(out["ticker"])
        check(passed == {"OLDOF", "NEWBIG"}, f"screen master+staging, excl liquid+thin (got {passed})")

        # staging absent -> master only, no crash
        bou.OVERFLOW_PRICES_PATH = os.path.join(td, "nope.parquet")
        prices2 = bou._load_prices()
        check(set(prices2["ticker"].unique()) == {"AAPL", "OLDOF"}, "master-only when staging absent")

    print(f"\nALL {_n} STAGING CHECKS PASSED")


if __name__ == "__main__":
    main()
