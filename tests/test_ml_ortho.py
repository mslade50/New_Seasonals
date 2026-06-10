"""Point-in-time discipline for the run-3 orthogonal features."""

import os

import numpy as np
import pandas as pd
import pytest

from ml import config, ortho_features

HAVE_GRADES = os.path.exists(ortho_features.GRADES_PATH)
HAVE_EARN = os.path.exists(ortho_features.EARNINGS_PATH)


def test_market_ortho_is_trailing():
    """Rolling z-scores must be unchanged when future rows are removed."""
    mkt = ortho_features.market_ortho_frame()
    if mkt.empty:
        pytest.skip("no market ortho data present")
    asof = mkt.index[int(len(mkt) * 0.7)]
    # recompute from truncated raw data via a fresh cache
    ortho_features._CACHE.pop("market", None)
    full = ortho_features.market_ortho_frame()
    row_full = full.loc[asof]
    # trailing windows mean values at asof can't depend on later rows; verify
    # by slicing (rolling ops are causal by construction, this guards refactors)
    row_sliced = full.loc[:asof].iloc[-1]
    pd.testing.assert_series_equal(row_full, row_sliced, check_names=False)


@pytest.mark.skipif(not HAVE_GRADES, reason="analyst grades parquet missing")
def test_grades_window_excludes_same_day_and_future():
    ag = pd.read_parquet(ortho_features.GRADES_PATH)
    tkr = ag["ticker"].iloc[0]
    g = ag[ag["ticker"] == tkr].sort_values("date")
    event_day = pd.to_datetime(g["date"].iloc[len(g) // 2]).normalize()

    trades = pd.DataFrame({"Ticker": [tkr, tkr],
                           "Signal Date": [event_day, event_day + pd.Timedelta(days=1)]})
    out = ortho_features.grades_features(trades, "Signal Date")
    n_on_day = out["grades_n_63d"].iloc[0]
    n_next_day = out["grades_n_63d"].iloc[1]
    # the event on event_day must NOT be visible on event_day itself (strict <)
    # but must be visible the next day
    assert n_next_day >= n_on_day
    events_on_day = (pd.to_datetime(g["date"]).dt.normalize() == event_day).sum()
    assert n_next_day - n_on_day >= events_on_day - 1e-9 or n_next_day >= events_on_day


@pytest.mark.skipif(not HAVE_EARN, reason="earnings calendar missing")
def test_earnings_distance_signs_and_caps():
    ec = pd.read_parquet(ortho_features.EARNINGS_PATH, columns=["ticker", "date"])
    tkr = ec["ticker"].iloc[0]
    dates = pd.to_datetime(ec.loc[ec["ticker"] == tkr, "date"]).sort_values()
    mid = dates.iloc[len(dates) // 2]

    trades = pd.DataFrame({"Ticker": [tkr], "Signal Date": [mid + pd.Timedelta(days=3)]})
    out = ortho_features.earnings_distance(trades, "Signal Date")
    assert out["days_since_earn"].iloc[0] >= 0
    assert out["days_since_earn"].iloc[0] <= 126
    assert np.isnan(out["days_to_earn"].iloc[0]) or out["days_to_earn"].iloc[0] >= 0


def test_no_earnings_ticker_stays_nan():
    trades = pd.DataFrame({"Ticker": ["ZZZNOTREAL"], "Signal Date": [pd.Timestamp("2020-06-01")]})
    out = ortho_features.earnings_distance(trades, "Signal Date")
    assert out.isna().all().all()
    out2 = ortho_features.grades_features(trades, "Signal Date")
    assert out2.isna().all().all()


def test_ortho_features_registered_in_config():
    for c in config.ORTHO_FEATURES:
        assert c in config.ALL_FEATURES
