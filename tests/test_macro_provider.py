import json
from pathlib import Path
import pandas as pd
import pytest
from official_macro_releases import observation
from scripts import refresh_macro_releases as runner


def row(event="initial_jobless_claims", value=210, date="2026-09-17"):
    return observation(event, value, "2026-09-12", date+"T12:30:00Z",
                       source="https://www.dol.gov/ui/data.pdf", fetched_at="2026-09-24T15:00:00Z",
                       digest="abc", unit="K")


def test_history_preserved_forecast_not_copied_new_release_added():
    old = row(); old.update(source="fmp:economic-calendar", consensus=211, surprise=-1)
    prior = pd.DataFrame([old])
    fresh = pd.DataFrame([row(value=212), row(value=209, date="2026-09-24")])
    result = runner.candidate_history(prior, fresh)
    assert len(result) == 2
    assert result.iloc[0].actual == 210 and result.iloc[0].consensus == 211
    assert result.iloc[1].actual == 209 and pd.isna(result.iloc[1].consensus)


def test_gdp_stage_does_not_duplicate_legacy_event():
    prior = pd.DataFrame([row("gdp_qoq", 1.5)])
    fresh = pd.DataFrame([row("gdp_qoq_second_estimate", 1.6)])
    result = runner.candidate_history(prior, fresh)
    assert len(result) == 1 and result.iloc[0].actual == 1.5
    result = runner.candidate_history(prior, pd.DataFrame([row("gdp_qoq_third_estimate", 1.7, "2026-09-24")]))
    assert result.iloc[1].event_id == "gdp_qoq_third_estimate" and result.iloc[1].estimate_stage == "third"


def test_duplicate_history_blocks_publication():
    prior = pd.DataFrame([row(), row()])
    with pytest.raises(ValueError):
        runner.candidate_history(prior, pd.DataFrame([row()]))


@pytest.mark.parametrize("outcome", ["conflict", "bad_readback", "success"])
def test_conditional_publish_and_verified_local_replacement(tmp_path, monkeypatch, outcome):
    import cache_io
    candidate = tmp_path / "candidate"; candidate.write_bytes(b"candidate")
    local = tmp_path / "local"; local.write_bytes(b"prior")
    def upload(path, key, **kwargs):
        assert kwargs["expected_etag"] == '\"baseline\"'
        return ("conflict" if outcome == "conflict" else "uploaded"), '\"new\"'
    def download(key, path):
        Path(path).write_bytes(b"bad" if outcome == "bad_readback" else b"candidate")
        return True
    monkeypatch.setattr(cache_io, "conditional_upload_from_local", upload)
    monkeypatch.setattr(cache_io, "download_to_local", download)
    if outcome == "success":
        runner.publish(candidate, '\"baseline\"', local, tmp_path)
        assert local.read_bytes() == b"candidate"
    else:
        with pytest.raises(ValueError):
            runner.publish(candidate, '\"baseline\"', local, tmp_path)
        assert local.read_bytes() == b"prior"
        state = json.loads((tmp_path / "publication.json").read_text())["publication_state"]
        assert state == ("remote_written_unverified" if outcome == "bad_readback" else "write_outcome_unknown")


def test_failed_collector_never_publishes(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    base = tmp_path / "baseline.parquet"
    pd.DataFrame([row()]).to_parquet(base)
    monkeypatch.setattr(runner, "collect", lambda *a, **k: {"core_data_pass": False})
    monkeypatch.setattr(runner, "publish", lambda *a: pytest.fail("failed collector published"))
    out = tmp_path / "artifacts/run"
    assert runner.main(["--no-upload", "--output-dir", str(out), "--baseline", str(base)]) == 1
    assert not (tmp_path / "data").exists()
    assert json.loads((out / "failure.json").read_text())["published"] is False


def test_forecast_only_legacy_observation_retained_separately():
    old = row(); old.update(actual=None, consensus=211, source="fmp", surprise=None)
    result = runner.candidate_history(pd.DataFrame([old]), pd.DataFrame([row(value=209)]))
    assert len(result) == 1 and pd.isna(result.iloc[0].consensus)
    evidence = json.loads(result.iloc[0].superseded_provider_observation)
    assert evidence["consensus"] == 211 and evidence["source"] == "fmp"


def test_core_data_alone_does_not_authorize_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    base = tmp_path / "baseline.parquet"; pd.DataFrame([row()]).to_parquet(base)
    monkeypatch.setattr(runner, "collect", lambda *a, **k: {"core_data_pass": True, "publication_eligible": False})
    monkeypatch.setattr(runner, "publish", lambda *a: pytest.fail("ineligible capture published"))
    assert runner.main(["--no-upload", "--output-dir", str(tmp_path / "artifacts/run"), "--baseline", str(base)]) == 1
