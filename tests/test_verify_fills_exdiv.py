"""Regression test for the ex-dividend phantom-fill bug in verify_fills.py.

Background
----------
On 2026-06-08 an OLV "Limit Order -0.25 ATR (Persistent)" long order on EWZ was
staged with Entry=33.69, ATR=0.73 -> Limit_Price=33.51, frozen into the sheet in
as-traded (pre-ex) dollars. EWZ's raw forward lows (6/9=33.58, 6/10=33.63) never
reached 33.51, so it correctly never filled live. EWZ then went ex-dividend
($0.331) on 2026-06-15; yfinance auto_adjust=True back-scaled every pre-ex bar by
f~0.9906, dropping the 6/9 low to ~33.26. verify_fills used to re-pull bars with
auto_adjust=True and compare them to the FROZEN pre-ex limit 33.51 -> phantom FILL.

The fix: fetch_price_data pulls RAW (auto_adjust=False) bars so the frozen dollar
limit is compared against bars in the same basis it was minted in.

These tests assert:
  1. check_fill on RAW bars -> NOT FILLED (the fixed behavior, == live truth).
  2. check_fill on ADJUSTED bars -> FILLED (documents the bug the fix avoids).
  3. fetch_price_data passes auto_adjust=False to yfinance (locks the fix in place).
  4. The engine's RELATIVE-limit fill rule is scale-invariant (codifies WHY the
     backtest engines can stay on adjusted prices; the guard for the book-wide
     dividend-adjustment invariant in CLAUDE.md).

Plain-assert style + main() runner, mirroring tests/test_eod_dd.py. Also
discoverable by pytest (functions named test_*).
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import verify_fills
from verify_fills import check_fill

# Dividend factor EWZ would impose post-ex (33.6001 / 33.92).
F = 0.990570

# Frozen, pre-ex sheet values for the staged EWZ OLV order.
SIGNAL_CLOSE = 33.69
ATR = 0.73
OFFSET = 0.25                      # REL_CLOSE limit = close - 0.25*ATR = 33.51
SIGNAL_DATE = "2026-06-08"         # Monday; T+1 = 2026-06-09 (Tue)
EXIT_DATE = "2026-06-12"           # GTC window end (Fri), comfortably in the past


def _raw_bars() -> pd.DataFrame:
    """As-traded (pre-ex) EWZ-like bars. Forward lows stay ABOVE the 33.51 limit."""
    idx = pd.to_datetime(["2026-06-09", "2026-06-10", "2026-06-11", "2026-06-12"])
    #                       Open    High    Low     Close
    data = np.array([
        [33.80, 33.95, 33.58, 33.75],
        [33.78, 33.90, 33.63, 33.70],
        [33.85, 33.99, 33.70, 33.88],
        [33.90, 34.10, 33.80, 34.00],
    ])
    return pd.DataFrame(data, index=idx, columns=["Open", "High", "Low", "Close"])


def _adjusted_bars() -> pd.DataFrame:
    """Post-ex adjusted bars: every pre-ex bar back-scaled by f (~0.99057).

    This drops the 6/9 low to ~33.26, BELOW the frozen 33.51 limit — the exact
    condition that produced the phantom fill before the fix.
    """
    return _raw_bars() * F


def test_relclose_raw_bars_no_phantom_fill():
    """RAW bars (the fix): frozen 33.51 limit vs raw lows -> order does NOT fill."""
    status, fill_date, fill_px = check_fill(
        order_class="REL_CLOSE", action="Long",
        signal_close=SIGNAL_CLOSE, atr=ATR, offset=OFFSET,
        limit_price_override=None,
        ticker_prices=_raw_bars(),
        signal_date=SIGNAL_DATE, exit_date=EXIT_DATE, tif="GTC",
    )
    assert status != "FILLED", (
        f"RAW bars must not fill (live truth was no-fill), got {status} @ {fill_px}"
    )


def test_relclose_adjusted_bars_would_phantom_fill():
    """ADJUSTED bars (the old bug): frozen 33.51 vs back-scaled lows -> phantom FILL.

    This is the behavior the fix removes by switching fetch_price_data to raw bars.
    Kept as a guard so the phantom mechanism stays understood and documented.
    """
    status, fill_date, fill_px = check_fill(
        order_class="REL_CLOSE", action="Long",
        signal_close=SIGNAL_CLOSE, atr=ATR, offset=OFFSET,
        limit_price_override=None,
        ticker_prices=_adjusted_bars(),
        signal_date=SIGNAL_DATE, exit_date=EXIT_DATE, tif="GTC",
    )
    assert status == "FILLED", (
        "ADJUSTED bars reproduce the phantom fill (this is the bug); "
        f"got {status}. If this no longer fills, the repro is stale."
    )
    assert fill_date == pd.Timestamp("2026-06-09").date()


def test_loc_stored_limit_raw_vs_adjusted():
    """LOC branch reads the stored raw Limit_Price directly (the more-exposed path).

    Raw closes stay above the 33.51 override (no fill); adjusted closes get
    back-scaled below it (phantom fill). The fix's raw bars give the correct no-fill.
    """
    override = 33.51

    raw_status, _, _ = check_fill(
        order_class="LOC", action="Long",
        signal_close=SIGNAL_CLOSE, atr=ATR, offset=0.0,
        limit_price_override=override,
        ticker_prices=_raw_bars(),
        signal_date=SIGNAL_DATE, exit_date=EXIT_DATE, tif="GTC",
    )
    assert raw_status != "FILLED", f"RAW LOC must not fill, got {raw_status}"

    adj_status, _, _ = check_fill(
        order_class="LOC", action="Long",
        signal_close=SIGNAL_CLOSE, atr=ATR, offset=0.0,
        limit_price_override=override,
        ticker_prices=_adjusted_bars(),
        signal_date=SIGNAL_DATE, exit_date=EXIT_DATE, tif="GTC",
    )
    assert adj_status == "FILLED", f"ADJUSTED LOC reproduces the phantom, got {adj_status}"


def test_fetch_price_data_uses_raw_bars():
    """Lock the one-line fix: fetch_price_data must pass auto_adjust=False to yfinance,
    and must still normalize OHLC columns despite the extra 'Adj Close' column raw
    downloads return."""
    captured = {}

    def fake_download(tickers, *args, **kwargs):
        captured["auto_adjust"] = kwargs.get("auto_adjust", "MISSING")
        idx = pd.to_datetime(["2026-06-09", "2026-06-10"])
        fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        cols = pd.MultiIndex.from_product([fields, ["EWZ"]])
        data = np.array([
            [33.80, 33.95, 33.58, 33.75, 33.43, 1_000_000.0],
            [33.78, 33.90, 33.63, 33.70, 33.38, 1_100_000.0],
        ])
        return pd.DataFrame(data, index=idx, columns=cols)

    orig_download = verify_fills.yf.download
    orig_sleep = verify_fills.time.sleep
    try:
        verify_fills.yf.download = fake_download
        verify_fills.time.sleep = lambda *a, **k: None
        res = verify_fills.fetch_price_data(["EWZ"], "2026-06-08", "2026-06-12")
    finally:
        verify_fills.yf.download = orig_download
        verify_fills.time.sleep = orig_sleep

    assert captured.get("auto_adjust") is False, (
        f"fetch_price_data must request RAW bars (auto_adjust=False), "
        f"got {captured.get('auto_adjust')!r}"
    )
    assert "EWZ" in res, "EWZ frame missing from fetch_price_data output"
    assert {"Open", "High", "Low", "Close"}.issubset(set(res["EWZ"].columns)), (
        f"OHLC columns missing after normalization: {list(res['EWZ'].columns)}"
    )


def _engine_long_fill(close, atr, low, k):
    """Canonical engine relative-limit rule (long): fills iff the bar's low reaches
    limit = close - k*ATR. NO rounding, matching backtester.py:2090-2101 and
    strat_backtester.py:1653-1659 (rounding is the one thing that could break
    invariance; the engines don't round)."""
    return low <= (close - k * atr)


def test_relative_limit_fill_is_scale_invariant():
    """Codifies WHY the backtest engines can stay on adjusted prices.

    The engines recompute the limit from the same series each run and compare to
    that series' forward bars. Under uniform dividend adjustment every price scales
    by the same factor f, so both sides of `low <= close - k*ATR` scale by f and the
    fill DECISION is unchanged. This asserts that invariance directly across a range
    of factors (incl. the EWZ f~0.99057 and aggressive split-like factors).

    If a future change puts an ABSOLUTE dollar level into the engine fill path (a
    hard limit, a $-pivot, a fixed stop), this property no longer holds and that
    change must instead follow the frozen-level rule (raw bars), like verify_fills.
    See the dividend-adjustment-basis rule in CLAUDE.md.
    """
    # (close, atr, low, k) — clear fills and no-fills, each well off the boundary
    # so float error can't flip the decision under scaling.
    cases = [
        (100.00, 2.00, 98.90, 0.5),   # limit 99.00, low 98.90 -> fill
        (100.00, 2.00, 99.10, 0.5),   # limit 99.00, low 99.10 -> no fill
        (33.69, 0.73, 33.58, 0.25),   # EWZ-like: limit 33.5075, low 33.58 -> no fill
        (50.00, 4.00, 45.50, 1.0),    # limit 46.00, low 45.50 -> fill
        (50.00, 4.00, 47.00, 1.0),    # limit 46.00, low 47.00 -> no fill
    ]
    factors = [0.5, 0.8, 0.990570, 1.0, 1.5, 3.2]
    for close, atr, low, k in cases:
        base = _engine_long_fill(close, atr, low, k)
        for f in factors:
            scaled = _engine_long_fill(close * f, atr * f, low * f, k)
            assert scaled == base, (
                f"scale-invariance broken: close={close} atr={atr} low={low} k={k} "
                f"f={f} -> base={base}, scaled={scaled}"
            )


def main():
    test_relclose_raw_bars_no_phantom_fill()
    print("PASS: REL_CLOSE raw bars -> no fill (matches live truth)")
    test_relclose_adjusted_bars_would_phantom_fill()
    print("PASS: REL_CLOSE adjusted bars -> phantom fill reproduced (the bug)")
    test_loc_stored_limit_raw_vs_adjusted()
    print("PASS: LOC stored-limit raw=no-fill, adjusted=phantom")
    test_fetch_price_data_uses_raw_bars()
    print("PASS: fetch_price_data uses auto_adjust=False and normalizes OHLC")
    test_relative_limit_fill_is_scale_invariant()
    print("PASS: engine relative-limit fill rule is scale-invariant")
    print("\nAll verify_fills ex-div phantom-fill tests passed.")


if __name__ == "__main__":
    main()
