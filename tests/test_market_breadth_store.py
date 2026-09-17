import json
import pandas as pd
import pytest
from scripts.maintain_market_breadth import connect, export_history, import_wsj, validate_wsj, URL


def observation():
    return {"source_url": URL, "column": "Latest Close", "date": "2026-09-17",
            "observed_at": "2026-09-17T22:00:00Z", "nyse_highs": 46, "nyse_lows": 204,
            "nasdaq_highs": 84, "nasdaq_lows": 396,
            "visible_evidence": "Diaries Thursday, September 17, 2026; NYSE 46/204; NASDAQ 84/396 (synthetic test date)"}


@pytest.mark.parametrize("patch", [{"nyse_highs": None}, {"nyse_lows": -1},
    {"nyse_highs": True}, {"nyse_highs": 0, "nyse_lows": 0},
    {"date": "2026-09-16"}, {"date": "2026-09-19"},
    {"column": "Week Ago"}, {"observed_at": "2026-09-17T15:00:00Z"}])
def test_reject_bad_or_stale_observations(patch):
    data = observation() | patch
    with pytest.raises(ValueError):
        validate_wsj(data, "2026-09-17T22:05:00Z")


def test_idempotent_and_revisions_preserved(tmp_path):
    with connect(tmp_path / "history.sqlite") as db:
        data = observation()
        import_wsj(db, data, "2026-09-17T22:05:00Z")
        import_wsj(db, data, "2026-09-17T22:05:00Z")
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1
        data = data | {"nyse_highs": 47, "observed_at": "2026-09-17T23:00:00Z"}
        import_wsj(db, data, "2026-09-17T23:05:00Z")
        result = export_history(db, tmp_path / "breadth.parquet")
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 2
        assert result.nyse_net.iloc[-1] == -157
        assert result.nasdaq_net.iloc[-1] == -312
