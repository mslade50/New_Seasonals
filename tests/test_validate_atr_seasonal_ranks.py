from pathlib import Path

import pandas as pd
import pytest

from scripts.validate_atr_seasonal_ranks import RANK_COLUMNS, validate
from atr_seasonal_contract import (
    RANK_METHOD_COLUMN,
    RANK_METHOD_VERSION,
    RANK_RETIRED_TICKERS,
)
from trading_calendar import TRADING_DAY


def _write(path: Path, tickers=("AAA", "BBB"), bad_rank=None) -> None:
    rows = []
    for year in (2020, 2021):
        for ticker in tickers:
            for date in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq=TRADING_DAY):
                row = {"Date": date, "ticker": ticker}
                row[RANK_METHOD_COLUMN] = RANK_METHOD_VERSION
                row.update({column: 50.0 for column in RANK_COLUMNS})
                rows.append(row)
    if bad_rank is not None:
        rows[0][RANK_COLUMNS[0]] = bad_rank
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_validated_rebuild_preserves_baseline_universe(tmp_path):
    artifact = tmp_path / "ranks.parquet"
    baseline = tmp_path / "baseline.parquet"
    _write(artifact)
    _write(baseline, tickers=("AAA",))

    manifest = validate(artifact, baseline, 2020, 2021)

    assert manifest["method_version"] == "target-year-truncated-nyse-v3"
    assert manifest["tickers"] == 2
    assert manifest["baseline_tickers"] == 1
    assert len(manifest["artifact_sha256"]) == 64


def test_out_of_range_rank_fails_closed(tmp_path):
    artifact = tmp_path / "ranks.parquet"
    _write(artifact, bad_rank=100.1)

    with pytest.raises(ValueError, match=r"outside \[0, 100\]"):
        validate(artifact, None, 2020, 2021)


def test_lost_baseline_ticker_fails_closed(tmp_path):
    artifact = tmp_path / "ranks.parquet"
    baseline = tmp_path / "baseline.parquet"
    _write(artifact, tickers=("AAA",))
    _write(baseline, tickers=("AAA", "BBB"))

    with pytest.raises(ValueError, match="lost 1 baseline tickers"):
        validate(artifact, baseline, 2020, 2021)


def test_lost_baseline_ticker_year_fails_closed(tmp_path):
    artifact = tmp_path / "ranks.parquet"
    baseline = tmp_path / "baseline.parquet"
    _write(artifact, tickers=("AAA",))
    _write(baseline, tickers=("AAA",))
    frame = pd.read_parquet(artifact)
    frame = frame[frame["Date"].dt.year != 2020]
    # Retain global 2020 coverage with a new ticker, so the assertion proves
    # preservation is checked per ticker/year rather than only book-wide.
    replacement = pd.read_parquet(baseline)
    replacement = replacement[replacement["Date"].dt.year == 2020].assign(ticker="NEW")
    pd.concat([frame, replacement], ignore_index=True).to_parquet(artifact, index=False)

    with pytest.raises(ValueError, match="lost 1 baseline ticker/year pairs"):
        validate(artifact, baseline, 2020, 2021)


def test_sparse_ticker_year_fails_complete_session_coverage(tmp_path):
    artifact = tmp_path / "ranks.parquet"
    _write(artifact, tickers=("AAA",))
    frame = pd.read_parquet(artifact)
    sparse = frame[~((frame["ticker"] == "AAA") & (frame["Date"].dt.year == 2020))]
    sparse = pd.concat(
        [sparse, frame[(frame["ticker"] == "AAA") & (frame["Date"].dt.year == 2020)].head(1)],
        ignore_index=True,
    )
    sparse.to_parquet(artifact, index=False)

    with pytest.raises(ValueError, match="incomplete ticker/year session coverage"):
        validate(artifact, None, 2020, 2021)


def test_corrected_calendar_can_remove_a_legacy_non_session_row(tmp_path):
    artifact = tmp_path / "ranks.parquet"
    baseline = tmp_path / "baseline.parquet"
    _write(artifact, tickers=("AAA",))
    _write(baseline, tickers=("AAA",))
    baseline_frame = pd.read_parquet(baseline)
    legacy_extra = baseline_frame.head(1).copy()
    legacy_extra["Date"] = pd.Timestamp("2020-01-04")
    pd.concat([baseline_frame, legacy_extra], ignore_index=True).to_parquet(
        baseline, index=False
    )

    manifest = validate(artifact, baseline, 2020, 2021)

    assert manifest["baseline_rows"] == len(baseline_frame) + 1


def test_reviewed_retired_baseline_ticker_is_removed_and_manifested(tmp_path):
    artifact = tmp_path / "ranks.parquet"
    baseline = tmp_path / "baseline.parquet"
    _write(artifact, tickers=("AAA",))
    _write(baseline, tickers=("AAA", "THS"))

    manifest = validate(artifact, baseline, 2020, 2021)

    assert manifest["retired_baseline_tickers"] == ["THS"]
    assert manifest["retired_baseline_rows"] > 0
    assert manifest["retirement_reasons"]["THS"] == RANK_RETIRED_TICKERS["THS"]
    assert manifest["baseline_tickers"] == 1


def test_missing_required_current_ticker_fails_closed(tmp_path):
    artifact = tmp_path / "ranks.parquet"
    _write(artifact, tickers=("AAA",))

    with pytest.raises(ValueError, match="missing 1 required current-universe tickers"):
        validate(
            artifact,
            None,
            2020,
            2021,
            required_tickers={"AAA", "BBB"},
        )


def test_retired_ticker_cannot_be_required_by_current_universe(tmp_path):
    artifact = tmp_path / "ranks.parquet"
    _write(artifact, tickers=("AAA",))

    with pytest.raises(ValueError, match="retired tickers are still reachable"):
        validate(
            artifact,
            None,
            2020,
            2021,
            required_tickers={"AAA", "THS"},
        )
