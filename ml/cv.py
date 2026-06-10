"""
ml/cv.py — Purged, embargoed, expanding walk-forward splits by calendar year.

For test year Y:
  test  = trades with Signal Date inside [Y-01-01, Y-12-31]
  train = trades with Signal Date  < Y-01-01
          AND Exit Date < (Y-01-01 minus EMBARGO trading days)

The Exit-Date purge removes label leakage from trades whose outcome was
resolved by prices inside (or near) the test window — this matters for the
63-day-hold breakout strategies that straddle year-ends. The embargo adds a
buffer for serial correlation right at the boundary.
"""

import pandas as pd
from pandas.tseries.offsets import BDay

from ml import config


def walk_forward_splits(signal_dates: pd.Series, exit_dates: pd.Series,
                        first_test_year: int = None, embargo_td: int = None,
                        last_test_year: int = None):
    """Yield (train_index, test_index, test_year) tuples.

    signal_dates / exit_dates: datetime Series sharing one index (the rows of
    the dataset). Rows with NaT exit dates are never placed in train.
    """
    first_test_year = first_test_year or config.FIRST_TEST_YEAR
    embargo_td = config.EMBARGO_TRADING_DAYS if embargo_td is None else embargo_td

    signal_dates = pd.to_datetime(signal_dates)
    exit_dates = pd.to_datetime(exit_dates)
    years = sorted(signal_dates.dt.year.unique())
    if last_test_year is not None:
        years = [y for y in years if y <= last_test_year]

    for year in years:
        if year < first_test_year:
            continue
        test_start = pd.Timestamp(year=year, month=1, day=1)
        test_end = pd.Timestamp(year=year, month=12, day=31)
        purge_cutoff = test_start - BDay(embargo_td)

        test_mask = (signal_dates >= test_start) & (signal_dates <= test_end)
        train_mask = (
            (signal_dates < test_start)
            & exit_dates.notna()
            & (exit_dates < purge_cutoff)
        )
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        yield signal_dates.index[train_mask], signal_dates.index[test_mask], year
