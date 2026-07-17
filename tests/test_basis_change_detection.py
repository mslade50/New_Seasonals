"""Basis-change detection in update_master_prices (2026-07-16).

The nightly refresh re-adjusts only the trailing 120d window; a reverse
split or large special dividend rescales yfinance's entire adjusted history,
leaving a permanent cliff at the window boundary that corrupts every >120d
lookback the live scan computes (252d rank, SMA200, 52wh — the 3x fades gate
on exactly these). detect_basis_changes flags tickers whose re-fetched
overlap bars diverge from cached bars so the updater re-pulls their full
history on one basis.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.update_master_prices import detect_basis_changes, novel_cliff_dates


def _frame(ticker, closes, start="2026-06-01"):
    dates = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame({"ticker": ticker, "date": dates, "Close": closes})


def test_reverse_split_flagged():
    # cached at old basis; refetch shows the whole overlap at 1/4 the level
    master = _frame("TQQQ", [80.0, 81.0, 79.0, 80.5, 82.0])
    fresh = _frame("TQQQ", [20.0, 20.25, 19.75, 20.125, 20.5])
    assert detect_basis_changes(master, fresh) == ["TQQQ"]


def test_normal_dividend_drift_not_flagged():
    # regular small-dividend re-adjustment shifts the overlap well under 2%
    master = _frame("SPY", [500.0, 501.0, 499.0, 502.0, 503.0])
    fresh = _frame("SPY", [c * 0.997 for c in [500.0, 501.0, 499.0, 502.0, 503.0]])
    assert detect_basis_changes(master, fresh) == []


def test_single_bad_bar_not_flagged():
    # median across the overlap absorbs one bogus bar
    master = _frame("XLE", [90.0, 91.0, 90.5, 89.0, 90.0])
    fresh = _frame("XLE", [90.0, 91.0, 90.5, 89.0, 45.0])  # one spike only
    assert detect_basis_changes(master, fresh) == []


def test_special_dividend_flagged():
    # 5% special div going ex rescales prior history beyond the window
    master = _frame("KMB", [100.0, 101.0, 100.5, 99.5, 100.0])
    fresh = _frame("KMB", [c * 0.95 for c in [100.0, 101.0, 100.5, 99.5, 100.0]])
    assert detect_basis_changes(master, fresh) == ["KMB"]


def test_no_overlap_is_quiet():
    master = _frame("NEW", [10.0, 10.5], start="2026-01-05")
    fresh = _frame("NEW", [11.0, 11.5], start="2026-06-01")
    assert detect_basis_changes(master, fresh) == []


def test_multiple_tickers_only_diverged_flagged():
    master = pd.concat([
        _frame("AAA", [50.0, 51.0, 50.5]),
        _frame("BBB", [30.0, 31.0, 30.5]),
    ])
    fresh = pd.concat([
        _frame("AAA", [25.0, 25.5, 25.25]),   # 2:1 split
        _frame("BBB", [30.0, 31.0, 30.5]),    # unchanged
    ])
    assert detect_basis_changes(master, fresh) == ["AAA"]


# ---- novel_cliff_dates (vendor-series sanity on basis re-pulls, 2026-07-17) ----

def _series(closes, start="2026-01-05"):
    return pd.Series(closes, index=pd.date_range(start, periods=len(closes), freq="B"))


def test_broken_vendor_series_rejected():
    # fresh has a spurious 15x cliff mid-history; cache (repaired) is smooth
    cached = _series([100.0, 101.0, 99.0, 100.0, 98.0, 99.0])
    fresh = _series([1500.0, 1515.0, 1485.0, 100.0, 98.0, 99.0])
    novel = novel_cliff_dates(fresh, cached)
    assert novel == [fresh.index[3]]


def test_matching_real_crash_accepted():
    # a genuine -56% day exists in BOTH series -> not novel
    cached = _series([100.0, 44.0, 45.0, 46.0])
    fresh = _series([200.0, 88.0, 90.0, 92.0])  # same shape, new basis
    assert novel_cliff_dates(fresh, cached) == []


def test_fresh_only_date_never_blocks():
    # cliff at a date the cache doesn't have (brand-new crash bar) is ignored
    cached = _series([100.0, 101.0, 100.5])
    fresh = pd.concat([_series([100.0, 101.0, 100.5]),
                       _series([40.0], start="2026-01-08")])
    assert novel_cliff_dates(fresh, cached) == []


def test_smooth_fresh_series_accepted():
    # the intended basis-change path: fresh is the same series on a new basis
    cached = _series([80.0, 81.0, 79.0, 80.5])
    fresh = _series([20.0, 20.25, 19.75, 20.125])
    assert novel_cliff_dates(fresh, cached) == []
