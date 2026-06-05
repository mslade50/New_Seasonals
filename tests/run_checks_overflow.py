"""Standalone (no-pytest) verification of the overflow universe logic.

Pure: no network, no R2, no Sheets. Run: python tests/run_checks_overflow.py
"""
import math
import os
import sys
import tempfile

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import overflow_universe as ou
from build_overflow_universe import screen_universe, _atr_pct_series

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


def universe():
    return pd.concat([
        _mk("GOODBIG", 300, 50.0, 250_000),
        _mk("THINLOW", 300, 50.0, 10_000),
        _mk("DEADCALM", 300, 50.0, 250_000, atr_frac=0.0),
        _mk("SHORTHIST", 100, 50.0, 250_000),
        _mk("PENNY", 300, 1.0, 9_000_000),
        _mk("STALE", 300, 50.0, 250_000, last_date=AS_OF - pd.Timedelta(days=120)),
        _mk("AAPL", 300, 200.0, 5_000_000),
    ], ignore_index=True)


def main():
    out = screen_universe(universe(), {"AAPL"}, as_of=AS_OF)
    check(set(out["ticker"]) == {"GOODBIG"}, f"screen passes only GOODBIG (got {set(out['ticker'])})")

    row = out[out["ticker"] == "GOODBIG"].iloc[0]
    check(math.isclose(row["addv_63d"], 12_500_000, rel_tol=1e-6), "GOODBIG ADDV = 12.5MM")
    check(math.isclose(row["atr_pct_63d"], 4.0, rel_tol=0.05), f"GOODBIG ATR% ~4 (got {row['atr_pct_63d']:.3f})")
    check(row["n_bars"] == 300, "GOODBIG n_bars=300")

    s = _atr_pct_series(_mk("X", 50, 100.0, 1000, atr_frac=0.02).set_index("date"), 14)
    check(math.isclose(s.dropna().iloc[-1], 2.0, rel_tol=0.05), "ATR% series ~2 for 2% range")

    out2 = screen_universe(universe(), {"GOODBIG"}, as_of=AS_OF)
    check("GOODBIG" not in set(out2["ticker"]), "liquid members excluded")

    with tempfile.TemporaryDirectory() as td:
        missing = os.path.join(td, "nope.parquet")
        check(ou.load_overflow_universe(fallback=["FOO", "BAR"], path=missing) == ["FOO", "BAR"], "fallback used verbatim when missing")
        check(ou.load_overflow_universe(fallback=None, path=missing) == [], "empty when no fallback")

        p = os.path.join(td, "u.parquet")
        pd.DataFrame({"ticker": ["brk.b", "aaa", "AAA"], "addv_63d": [9e6, 5e6, 5e6]}).to_parquet(p)
        check(ou.load_overflow_universe(fallback=["ZZZ"], path=p, respect_active=False) == ["AAA", "BRK-B"], "parquet normalized/deduped/sorted")
        meta = ou.load_overflow_meta(path=p, respect_active=False)
        check(math.isclose(meta["AAA"]["addv_63d"], 5e6), "meta addv read")
        check(ou.load_overflow_meta(path=os.path.join(td, "x.parquet"), respect_active=False) == {}, "meta empty when missing")

    meta = {"BIG": {"addv_63d": 12e6}, "MID": {"addv_63d": 6e6}, "SMALL": {"addv_63d": 3.5e6}}
    tk = ["BIG", "MID", "SMALL"]
    check(ou.filter_by_addv(tk, "Overbot Vol Spike", meta) == ["BIG"], "OVS floor 10MM")
    check(ou.filter_by_addv(tk, "52wh Breakout", meta) == ["BIG", "MID"], "52wh floor 5MM")
    check(ou.filter_by_addv(tk, "Oversold Low Volume", meta) == tk, "base floor 3MM")
    check(ou.filter_by_addv(tk, "Overbot Vol Spike", {}) == tk, "no-op when meta empty")
    check(ou.filter_by_addv(["BIG", "UNK"], "Overbot Vol Spike", {"BIG": {"addv_63d": 12e6}}) == ["BIG", "UNK"], "keep names missing from meta")

    check(ou.adv_share_cap(10e6, 50.0, 0.02) == 4000, "ADV cap 2% of 10MM @ $50 = 4000 sh")
    check(ou.adv_share_cap(None, 50.0) is None, "ADV cap None on bad addv")
    check(ou.adv_share_cap(10e6, 0.0) is None, "ADV cap None on bad price")
    check(ou.adv_share_cap(float("nan"), 50.0) is None, "ADV cap None on NaN addv")
    check(ou.min_addv_for("Overbot Vol Spike") == 10e6, "min_addv OVS=10MM")
    check(ou.min_addv_for("Unknown") == ou.MIN_ADDV_BASE, "min_addv default=base")

    # Activation gate: parquet present but gate OFF → fallback; gate ON → parquet.
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "u.parquet")
        pd.DataFrame({"ticker": ["NEW1", "NEW2"], "addv_63d": [9e6, 8e6]}).to_parquet(p)
        os.environ.pop(ou.OVERFLOW_UNIVERSE_ACTIVE_ENV, None)
        check(ou.is_active() is False, "gate OFF by default")
        check(ou.load_overflow_universe(fallback=["OLD"], path=p) == ["OLD"], "gate OFF -> fallback (ignores parquet)")
        check(ou.load_overflow_meta(path=p) == {}, "gate OFF -> empty meta")
        check(ou.load_overflow_universe(fallback=["OLD"], path=p, respect_active=False) == ["NEW1", "NEW2"], "respect_active=False bypasses gate")
        os.environ[ou.OVERFLOW_UNIVERSE_ACTIVE_ENV] = "1"
        check(ou.is_active() is True, "gate ON when env=1")
        check(ou.load_overflow_universe(fallback=["OLD"], path=p) == ["NEW1", "NEW2"], "gate ON -> parquet used")
        check(set(ou.load_overflow_meta(path=p)) == {"NEW1", "NEW2"}, "gate ON -> meta populated")
        os.environ.pop(ou.OVERFLOW_UNIVERSE_ACTIVE_ENV, None)

    print(f"\nALL {_n} CHECKS PASSED")


if __name__ == "__main__":
    main()
