"""Safety and comparison behavior for the independent earnings observation tool."""
import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import compare_earnings_shadow as shadow

HEADER = "symbol,name,reportDate,fiscalDateEnding,estimate,currency\n"


def alpha(rows):
    return shadow.parse_alpha_csv(HEADER + rows)


def fmp(rows):
    return shadow.normalize_fmp(pd.DataFrame(rows, columns=["ticker", "date", "eps_est"]))


@pytest.mark.parametrize("payload", [
    '{"Information":"quota exceeded"}', '{"Error Message":"bad key"}',
    "<html>bad gateway</html>", HEADER, "wrong,columns\n1,2\n",
    HEADER + "A,A,bad-date,2026-06-30,1,USD\n",
    HEADER + "A,A,2026-09-18,2026-06-30,oops,USD\n",
    HEADER + "A,A,2026-09-18,2026-06-30,1,USD\nA,A,2026-09-18,2026-06-30,2,USD\n",
])
def test_rejects_empty_error_and_malformed_responses(payload):
    with pytest.raises(shadow.ShadowError):
        shadow.parse_alpha_csv(payload)


def test_preserves_na_ticker_and_zero_negative_missing_estimates():
    parsed = alpha("NA,NA,2026-09-18,2026-06-30,0,USD\nB,B,2026-09-18,2026-06-30,-1,USD\nC,C,2026-09-18,2026-06-30,,USD\n")
    assert set(parsed.ticker) == {"NA", "B", "C"}
    assert parsed.set_index("ticker").loc["NA", "eps_est"] == 0
    assert parsed.set_index("ticker").loc["B", "eps_est"] == -1
    assert pd.isna(parsed.set_index("ticker").loc["C", "eps_est"])


def test_missing_events_are_not_hidden_by_high_ticker_agreement():
    baseline = fmp([("A", "2026-09-18", 1), ("B", "2026-09-18", 2),
                    ("C", "2026-09-18", 3), ("ETF", "2020-01-01", None)])
    candidate = alpha("A,A,2026-09-18,2026-06-30,1,USD\nC,C,2026-09-21,2026-06-30,3,USD\nD,D,2026-09-18,2026-06-30,4,USD\n")
    details, summary = shadow.compare(baseline, candidate, {"A", "B", "C", "D", "ETF"}, pd.Timestamp("2026-09-16"))
    statuses = details.set_index("ticker").status.to_dict()
    assert statuses == {"A": "exact_dates", "B": "missing_in_alpha", "C": "date_disagreement", "D": "alpha_only", "ETF": "no_upcoming_in_either"}
    assert summary["fmp_event_recall"] == pytest.approx(1 / 3)
    assert summary["blackout_disagreement_tickers"] == 3
    assert summary["switch_approved"] is False


def test_multiple_events_and_shared_past_are_preserved():
    baseline = fmp([("A", "2026-09-15", 1), ("A", "2026-09-18", 1), ("A", "2026-10-20", 2)])
    candidate = alpha("A,A,2026-09-18,2026-06-30,1.2,USD\n")
    details, summary = shadow.compare(baseline, candidate, {"A"}, pd.Timestamp("2026-09-16"))
    assert summary["fmp_events"] == 2
    assert summary["exact_events"] == 1
    assert details.iloc[0].status == "date_disagreement"
    assert summary["eps_differences_over_1_cent"] == 1


def test_conflicting_fmp_estimates_excluded_without_losing_blackout_date():
    baseline = fmp([("A", "2026-09-18", 1), ("A", "2026-09-18", 2)])
    candidate = alpha("A,A,2026-09-18,2026-06-30,1,USD\n")
    _, summary = shadow.compare(baseline, candidate, {"A"}, pd.Timestamp("2026-09-16"))
    assert summary["fmp_events"] == summary["exact_events"] == 1
    assert summary["eps_pairs"] == 0
    assert summary["fmp_conflicting_estimate_events"] == 1


def test_revisions_include_changed_and_disappeared_events():
    before = alpha("A,A,2026-09-18,2026-06-30,1,USD\nB,B,2026-09-20,2026-06-30,2,USD\n")
    after = alpha("A,A,2026-09-21,2026-06-30,1,USD\n")
    changes = shadow.revisions(before, after).set_index("ticker")
    assert changes.loc["A", "date_now"] == pd.Timestamp("2026-09-21")
    assert changes.loc["B", "_merge"] == "left_only"


def test_network_errors_do_not_leak_key(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.ConnectionError("https://example.com/?apikey=SECRET")
    monkeypatch.setattr(shadow.requests, "get", fail)
    with pytest.raises(shadow.ShadowError) as exc:
        shadow.fetch_alpha("SECRET")
    assert "SECRET" not in str(exc.value)


def test_offline_run_does_not_change_baseline_or_use_network(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    baseline = tmp_path / "data" / "earnings_calendar.parquet"
    fmp([("A", "2026-09-18", 1)]).to_parquet(baseline)
    original = baseline.read_bytes()
    csv = tmp_path / "input.csv"
    csv.write_text(HEADER + "A,A,2026-09-18,2026-06-30,1,USD\n")
    monkeypatch.setattr(shadow, "CSV_UNIVERSE", ["A"])
    monkeypatch.setattr(shadow.requests, "get", lambda *a, **kw: pytest.fail("unexpected network"))
    args = ["--config-root", str(tmp_path), "--alpha-csv", str(csv), "--as-of", "2026-09-16"]
    assert shadow.main(args) == 0
    assert shadow.main(args) == 0
    runs = list((tmp_path / "artifacts" / "earnings_shadow" / "offline").iterdir())
    assert len(runs) == 2
    assert baseline.read_bytes() == original
    for run in runs:
        summary = json.loads((run / "summary.json").read_text())
        assert summary["exact_events"] == 1
        assert not summary["switch_approved"]
        assert (run / "fmp_snapshot.parquet").exists()


def test_missing_key_does_not_request_or_approve_switch(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    fmp([("A", "2026-09-18", 1)]).to_parquet(tmp_path / "data" / "earnings_calendar.parquet")
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    monkeypatch.setattr(shadow.requests, "get", lambda *a, **kw: pytest.fail("unexpected network"))
    assert shadow.main(["--config-root", str(tmp_path)]) == 1
    failures = list((tmp_path / "artifacts").rglob("failure.json"))
    assert len(failures) == 1
    assert "no API call made" in failures[0].read_text()
