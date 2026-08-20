import pandas as pd
import pytest

from atr_seasonal_contract import (
    RANK_METHOD_COLUMN,
    RANK_METHOD_VERSION,
    rank_artifact_version_error,
)


def _frame(*versions):
    return pd.DataFrame({RANK_METHOD_COLUMN: list(versions)})


def test_current_rank_artifact_version_is_accepted():
    assert rank_artifact_version_error(_frame(RANK_METHOD_VERSION)) is None


def test_missing_rank_artifact_version_is_rejected():
    error = rank_artifact_version_error(pd.DataFrame({"ticker": ["SPY"]}))

    assert "missing rank_method_version" in error


def test_null_or_mixed_rank_artifact_versions_are_rejected():
    null_error = rank_artifact_version_error(_frame(RANK_METHOD_VERSION, None))
    mixed_error = rank_artifact_version_error(_frame(RANK_METHOD_VERSION, "legacy-v1"))

    assert "null/mixed-version rows" in null_error
    assert "unexpected rank method versions" in mixed_error


def test_wrong_rank_artifact_version_is_rejected():
    error = rank_artifact_version_error(_frame("legacy-v1"))

    assert RANK_METHOD_VERSION in error


def test_secondary_seasonal_loader_rejects_legacy_artifact(tmp_path):
    from scripts import seasonal_edge

    path = tmp_path / "legacy.parquet"
    pd.DataFrame({
        "Date": [pd.Timestamp("2026-01-02")],
        "ticker": ["SPY"],
        "atr_sznl_5d": [50.0],
    }).to_parquet(path, index=False)
    seasonal_edge.load_seasonal_ranks.cache_clear()

    with pytest.raises(ValueError, match="rank artifact rejected"):
        seasonal_edge.load_seasonal_ranks(str(path))

    seasonal_edge.load_seasonal_ranks.cache_clear()
