from scripts import pull_scan_caches


def test_pitch_cache_set_uses_r2_for_local_primary_state():
    required, _optional = pull_scan_caches.SETS["pitch"]
    by_key = dict(required)
    assert by_key["rd2_fragility.parquet"] == "data/rd2_fragility.parquet"
    assert by_key["rd2_environment.json"] == "data/rd2_environment.json"
    assert by_key["exposure_state.json"] == "data/exposure_state.json"
    assert by_key["cboe_putcall.parquet"] == "data/cboe_putcall.parquet"


def test_daily_pitch_no_longer_restores_generated_state_from_git():
    from pathlib import Path

    batch = (Path(__file__).resolve().parents[1] / "scripts" / "run_daily_pitch.bat")
    text = batch.read_text(encoding="utf-8")
    assert "pull_scan_caches.py\" --set pitch" in text
    assert "git restore --source=origin/main" not in text
