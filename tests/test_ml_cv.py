"""Purge/embargo correctness for ml.cv.walk_forward_splits."""

import pandas as pd
import pytest
from pandas.tseries.offsets import BDay

from ml import config
from ml.cv import walk_forward_splits


@pytest.fixture
def synthetic_trades():
    # Trades every ~10 days 2008-2015, holds of 2-70 calendar days, including
    # year-end stragglers whose exits land inside the next year.
    sig = pd.date_range("2008-01-02", "2015-12-28", freq="10D")
    df = pd.DataFrame({"sig": sig})
    df["exit"] = df["sig"] + pd.to_timedelta((df.index % 7 + 1) * 10, unit="D")
    return df


def test_no_train_exit_after_purge_cutoff(synthetic_trades):
    df = synthetic_trades
    for tr_idx, te_idx, year in walk_forward_splits(
            df["sig"], df["exit"], first_test_year=2012):
        cutoff = pd.Timestamp(year=year, month=1, day=1) - BDay(config.EMBARGO_TRADING_DAYS)
        assert (df.loc[tr_idx, "exit"] < cutoff).all(), (
            f"fold {year}: train trade exits at/after purge cutoff {cutoff}")


def test_test_fold_is_exactly_the_calendar_year(synthetic_trades):
    df = synthetic_trades
    for tr_idx, te_idx, year in walk_forward_splits(
            df["sig"], df["exit"], first_test_year=2012):
        assert (df.loc[te_idx, "sig"].dt.year == year).all()


def test_train_expands_and_precedes_test(synthetic_trades):
    df = synthetic_trades
    sizes = []
    for tr_idx, te_idx, year in walk_forward_splits(
            df["sig"], df["exit"], first_test_year=2012):
        assert df.loc[tr_idx, "sig"].max() < df.loc[te_idx, "sig"].min()
        sizes.append(len(tr_idx))
    assert sizes == sorted(sizes), "training window should be expanding"
    assert len(sizes) >= 3


def test_nat_exit_never_in_train(synthetic_trades):
    df = synthetic_trades.copy()
    df.loc[df.index[-5:], "exit"] = pd.NaT  # open trades
    open_idx = set(df.index[-5:])
    for tr_idx, te_idx, year in walk_forward_splits(
            df["sig"], df["exit"], first_test_year=2012):
        assert not (set(tr_idx) & open_idx)
