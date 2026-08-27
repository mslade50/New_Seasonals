import pandas as pd

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
