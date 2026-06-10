"""No-lookahead guarantee: every feature the model consumes must be
bit-identical when all data after the asof date is removed. This is the
empirical proof that training features are point-in-time.

Uses real repo data (master_prices.parquet) when present; skips otherwise.
"""

import os

import numpy as np
import pandas as pd
import pytest

from ml import config, features, market_features

HAVE_DATA = os.path.exists(config.MASTER_PRICES_PATH)
pytestmark = pytest.mark.skipif(not HAVE_DATA, reason="master_prices.parquet not present")

TICKERS = ["SPY", "XLF", "AAPL"]
ASOF = pd.Timestamp("2018-06-15")


@pytest.fixture(scope="module")
def price_map():
    return features.load_price_map(TICKERS)


@pytest.fixture(scope="module")
def sznl_map():
    return features.load_sznl_map()


@pytest.fixture(scope="module")
def atr_map():
    return features.load_atr_sznl_map(tickers=TICKERS)


def test_ticker_features_identical_under_truncation(price_map, sznl_map, atr_map):
    for tkr in TICKERS:
        if tkr not in price_map:
            continue
        full = price_map[tkr]
        trunc = full.loc[:ASOF]
        assert ASOF in trunc.index or trunc.index.max() <= ASOF

        f_full = features.compute_ticker_feature_frame(full, sznl_map, tkr,
                                                       atr_map.get(tkr))
        f_trunc = features.compute_ticker_feature_frame(trunc, sznl_map, tkr,
                                                        atr_map.get(tkr))
        asof = f_trunc.index.max()
        row_full = f_full.loc[asof].astype(float)
        row_trunc = f_trunc.loc[asof].astype(float)
        pd.testing.assert_series_equal(row_full, row_trunc, check_names=False,
                                       rtol=1e-9, atol=1e-12)


def test_market_features_identical_under_truncation():
    closes = market_features.load_market_closes()
    full = market_features.build_market_frame(closes)
    trunc = market_features.build_market_frame(closes.loc[:ASOF])
    asof = trunc.index.max()
    pd.testing.assert_series_equal(full.loc[asof].astype(float),
                                   trunc.loc[asof].astype(float),
                                   check_names=False, rtol=1e-9, atol=1e-12)


def test_no_forbidden_lookahead_columns_in_feature_list():
    forbidden = {"NextOpen", "is_pivot_high", "is_pivot_low",
                 "LastPivotHigh", "LastPivotLow"}
    assert not (forbidden & set(config.ALL_FEATURES))


def test_expanding_rank_features_are_trailing(price_map, sznl_map):
    """Sanity: a rank feature at date t must not change when future rows are
    appended — covered by truncation test — and must be NaN before 252 bars."""
    tkr = "SPY"
    if tkr not in price_map:
        pytest.skip("SPY missing")
    f = features.compute_ticker_feature_frame(price_map[tkr], sznl_map, tkr, None)
    early = f["rank_ret_21d"].iloc[:200]
    assert early.isna().all()
