"""Point-in-time guards for ATR-normalized seasonal ranks.

The target year's prices are out of sample. Changing any of them must leave
the complete rank surface for that target year unchanged at every horizon.
"""

import numpy as np
import pandas as pd
import pandas.testing as pdt

from build_atr_seasonal_ranks import (
    FWD_WINDOWS,
    compute_ranks_for_year,
    prepare_ticker_data,
)


def _prices() -> pd.DataFrame:
    dates = pd.bdate_range("2014-01-02", "2021-12-31")
    x = np.arange(len(dates), dtype=float)
    close = 80.0 + 0.025 * x + 2.5 * np.sin(x / 17.0) + 0.7 * np.cos(x / 41.0)
    return pd.DataFrame(
        {
            "Open": close * (1.0 + 0.001 * np.sin(x / 7.0)),
            "High": close + 1.1 + 0.1 * np.cos(x / 9.0),
            "Low": close - 1.0 - 0.1 * np.sin(x / 11.0),
            "Close": close,
            "Volume": 1_000_000.0 + x,
        },
        index=dates,
    )


def test_target_year_price_mutation_cannot_change_any_rank_horizon():
    original = _prices()
    mutated = original.copy()
    in_target = mutated.index.year == 2021

    # Before the fix, late-2020 origins referenced these perturbed prices and
    # moved every target-year rank family.
    mutated.loc[in_target, ["Open", "High", "Low", "Close"]] *= 7.0

    ranks_original = compute_ranks_for_year(prepare_ticker_data(original), 2021)
    ranks_mutated = compute_ranks_for_year(prepare_ticker_data(mutated), 2021)

    assert ranks_original is not None
    assert list(ranks_original.columns) == [f"atr_sznl_{w}d" for w in FWD_WINDOWS]
    pdt.assert_frame_equal(ranks_original, ranks_mutated, check_exact=True)
