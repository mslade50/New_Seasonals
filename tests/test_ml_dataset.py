"""Dataset/label integrity for the ML layer (uses real repo data; skips when
the ledger or price cache is absent)."""

import os

import numpy as np
import pandas as pd
import pytest

from ml import config
from ml.dataset import build_labels, load_ledger

HAVE_LEDGER = os.path.exists(config.LEDGER_PATH)
pytestmark = pytest.mark.skipif(not HAVE_LEDGER, reason="trade ledger not present")


@pytest.fixture(scope="module")
def ledger():
    return load_ledger()


def test_labels_match_ledger_arithmetic(ledger):
    y = build_labels(ledger)
    risk = ledger["Risk_flat_750k"].replace(0, np.nan)
    expected = (ledger["PnL_flat_750k"] / risk).fillna(ledger["R_Multiple"])
    pd.testing.assert_series_equal(y["y_R"], expected, check_names=False)
    assert ((y["y_win"] == 1) == (y["y_R"] > 0)).all()


def test_label_sign_consistent_with_pnl(ledger):
    y = build_labels(ledger)
    valid = ledger["Risk_flat_750k"] > 0
    pnl_sign = np.sign(ledger.loc[valid, "PnL_flat_750k"])
    label_sign = np.sign(y.loc[valid, "y_R"])
    agree = (pnl_sign == label_sign) | (pnl_sign == 0)
    assert agree.mean() > 0.999


def test_winsorization_bounds(ledger):
    y = build_labels(ledger)
    assert y["y_R_winsor"].max() <= config.R_WINSOR_HI + 1e-9
    assert y["y_R_winsor"].min() >= config.R_WINSOR_LO - 1e-9


def test_r_aligns_with_ledger_r_multiple(ledger):
    """PnL/Risk should track the ledger's own R_Multiple closely."""
    y = build_labels(ledger)
    valid = ledger["Risk_flat_750k"] > 0
    diff = (y.loc[valid, "y_R"] - ledger.loc[valid, "R_Multiple"]).abs()
    assert diff.median() < 0.05


@pytest.mark.skipif(not os.path.exists(config.DATASET_PATH),
                    reason="dataset not built yet (python -m ml.dataset)")
def test_built_dataset_quality():
    ds = pd.read_parquet(config.DATASET_PATH)
    # every configured feature column exists
    missing = [c for c in config.ALL_FEATURES if c not in ds.columns]
    assert not missing, f"missing feature columns: {missing}"
    # no feature should be entirely NaN
    dead = [c for c in config.ALL_FEATURES
            if c not in config.CATEGORICAL_FEATURES and ds[c].isna().all()]
    assert not dead, f"all-NaN features: {dead}"
    # ticker features should be populated for the vast majority of trades
    coverage = ds["ATR_Pct"].notna().mean()
    assert coverage > 0.90, f"ATR_Pct coverage only {coverage:.1%}"
    # rank features stay in [0, 100]
    for c in ["rank_ret_21d", "Sznl"]:
        vals = ds[c].dropna()
        assert vals.between(-1e-9, 100 + 1e-9).all(), f"{c} out of range"
