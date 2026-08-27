import pandas as pd
import pytest

from scripts import build_earnings_calendar as builder


def _earnings_row() -> dict[str, object]:
    return {
        "date": "2026-08-26",
        "epsActual": 1.0,
        "epsEstimated": 0.9,
        "revenueActual": 100.0,
        "revenueEstimated": 95.0,
        "lastUpdated": "2026-08-26",
    }


def test_build_calendar_no_upload_keeps_refresh_local(tmp_path, monkeypatch) -> None:
    output = tmp_path / "earnings.parquet"
    prior = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "date": [pd.Timestamp("2026-05-01")],
            "eps_actual": [0.8],
            "eps_est": [0.7],
            "revenue_actual": [90.0],
            "revenue_est": [85.0],
            "last_updated": [pd.Timestamp("2026-05-01")],
        }
    )
    prior.to_parquet(output, index=False)
    uploads: list[tuple[object, object]] = []
    monkeypatch.setattr(builder, "fetch_ticker", lambda *_: [_earnings_row()])
    monkeypatch.setattr(builder, "SLEEP_BETWEEN_CALLS", 0)
    monkeypatch.setattr(
        builder,
        "upload_to_r2",
        lambda path, key="earnings_calendar.parquet": uploads.append((path, key)),
    )

    builder.build_calendar(
        ["AAA"],
        "test-key",
        str(output),
        r2_key="earnings_calendar_overflow.parquet",
        upload=False,
    )

    assert uploads == []
    refreshed = pd.read_parquet(output)
    assert refreshed["ticker"].tolist() == ["AAA"]
    assert refreshed["date"].dt.date.tolist() == [pd.Timestamp("2026-08-26").date()]


def test_local_refresh_can_replace_a_superseded_prior_universe(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "earnings.parquet"
    prior = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "date": [pd.Timestamp("2026-05-01")] * 2,
        }
    )
    prior.to_parquet(output, index=False)
    monkeypatch.setattr(builder, "fetch_ticker", lambda *_: [_earnings_row()])
    monkeypatch.setattr(builder, "SLEEP_BETWEEN_CALLS", 0)

    builder.build_calendar(
        ["AAA"],
        "test-key",
        str(output),
        r2_key="earnings_calendar_overflow.parquet",
        upload=False,
        fail_on_fetch_errors=True,
    )

    assert pd.read_parquet(output)["ticker"].tolist() == ["AAA"]


def test_upload_still_refuses_a_degraded_prior_universe(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "earnings.parquet"
    prior = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "date": [pd.Timestamp("2026-05-01")] * 2,
        }
    )
    prior.to_parquet(output, index=False)
    original_bytes = output.read_bytes()
    uploads: list[tuple[object, object]] = []
    monkeypatch.setattr(builder, "fetch_ticker", lambda *_: [_earnings_row()])
    monkeypatch.setattr(builder, "SLEEP_BETWEEN_CALLS", 0)
    monkeypatch.setattr(
        builder,
        "upload_to_r2",
        lambda path, key="earnings_calendar.parquet": uploads.append((path, key)),
    )

    with pytest.raises(SystemExit, match="coverage dropped beyond 2%"):
        builder.build_calendar(
            ["AAA"],
            "test-key",
            str(output),
            upload=True,
        )

    assert output.read_bytes() == original_bytes
    assert uploads == []


def test_strict_fetch_failure_preserves_prior_file_and_skips_upload(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "earnings.parquet"
    prior = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "date": [pd.Timestamp("2026-05-01")] * 2,
            "eps_actual": [0.8, 0.7],
            "eps_est": [0.7, 0.6],
            "revenue_actual": [90.0, 80.0],
            "revenue_est": [85.0, 75.0],
            "last_updated": [pd.Timestamp("2026-05-01")] * 2,
        }
    )
    prior.to_parquet(output, index=False)
    original_bytes = output.read_bytes()
    uploads: list[tuple[object, object]] = []
    monkeypatch.setattr(
        builder,
        "fetch_ticker",
        lambda ticker, *_: [_earnings_row()] if ticker == "AAA" else None,
    )
    monkeypatch.setattr(builder, "SLEEP_BETWEEN_CALLS", 0)
    monkeypatch.setattr(
        builder,
        "upload_to_r2",
        lambda path, key="earnings_calendar.parquet": uploads.append((path, key)),
    )

    with pytest.raises(SystemExit, match="1 hard ticker fetch failure"):
        builder.build_calendar(
            ["AAA", "BBB"],
            "test-key",
            str(output),
            r2_key="earnings_calendar_overflow.parquet",
            upload=False,
            fail_on_fetch_errors=True,
        )

    assert output.read_bytes() == original_bytes
    assert uploads == []
