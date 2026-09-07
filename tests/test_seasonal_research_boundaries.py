"""Regression coverage for causal delayed entry and worktree-local research IO."""
import importlib
from pathlib import Path

import pandas as pd
import pytest

from scripts.seasonal_ticket_sim import simulate_ticket


def bars(forward):
    rows = [[100, 102, 98, 100]] * 20 + forward
    return pd.DataFrame(rows, index=pd.bdate_range("2026-01-01", periods=len(rows)),
                        columns=["Open", "High", "Low", "Close"])


@pytest.mark.parametrize("direction", ["long", "short"])
def test_delayed_limit_does_not_retroactively_enter_when_future_limit_misses(direction):
    px = bars([[100, 100.2, 99.8, 100.1]] * 4)
    tk = {"direction": direction, "entry": 100, "stop": 90 if direction == "long" else 110,
          "target": 110 if direction == "long" else 90, "time_stop_days": 4}
    out = simulate_ticket(tk, px, px.index[19], entry_mode="delayed_limit", entry_window=1)
    assert out["filled"] is False and out["exit_type"] == "NoFill"
    assert out["bars_held"] == 0


def test_delayed_limit_waits_for_remaining_resting_window():
    px = bars([[100, 100.2, 99.8, 100.1]] * 2)
    tk = {"direction": "long", "entry": 100, "stop": 90, "target": 110, "time_stop_days": 4}
    assert simulate_ticket(tk, px, px.index[19], entry_mode="delayed_limit", entry_window=1) is None


@pytest.mark.parametrize("mode", ["delayed", "delayed_close", "delayed_limit"])
def test_delay_does_not_move_entry_earlier_when_future_bars_are_missing(mode):
    px = bars([[100, 111, 95, 100.1]] * 2)
    tk = {"direction": "long", "entry": 100, "stop": 90, "target": 110, "time_stop_days": 5}
    assert simulate_ticket(tk, px, px.index[19], entry_mode=mode, entry_window=3) is None


def test_delayed_limit_enters_first_actual_touch_after_planned_delay():
    px = bars([[100, 101, 97, 100], [100, 100.2, 99.8, 100], [100, 101, 98, 100], [100, 101, 99, 100]])
    tk = {"direction": "long", "entry": 100, "stop": 90, "target": 110, "time_stop_days": 4}
    out = simulate_ticket(tk, px, px.index[19], entry_mode="delayed_limit", entry_window=1)
    assert out["filled"] is True and out["entry_date"] == px.index[22]


@pytest.mark.parametrize("name", ["resim_seasonal_entry", "seasonal_time_in_market", "enrich_seasonal_trades"])
def test_research_script_root_is_its_checkout(name):
    module = importlib.import_module("scripts." + name)
    assert Path(module.ROOT).resolve() == Path(__file__).resolve().parents[1]


def test_enrichment_default_output_is_in_ignored_workspace_artifacts():
    module = importlib.import_module("scripts.enrich_seasonal_trades")
    assert Path(module.OUT).resolve().is_relative_to(Path(module.ROOT).resolve() / "artifacts")


def test_enrichment_requires_nonexisting_isolated_output_before_input_read(tmp_path, monkeypatch):
    module = importlib.import_module("scripts.enrich_seasonal_trades")
    monkeypatch.setattr(module, "ROOT", tmp_path / "workspace")
    source = tmp_path / "in.parquet"
    with pytest.raises(ValueError, match="artifacts"):
        module.enrich(input_path=source, output_path=tmp_path / "out.parquet")
    output = Path(module.ROOT) / "artifacts" / "seasonal-research-tests" / "existing.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"reviewed evidence")
    with pytest.raises(FileExistsError):
        module.enrich(input_path=source, output_path=output)
    assert output.read_bytes() == b"reviewed evidence"


def test_enrichment_writes_only_requested_new_artifact(tmp_path, monkeypatch):
    module = importlib.import_module("scripts.enrich_seasonal_trades")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module.se, "load_prices", lambda *a, **kw: {})
    source = tmp_path / "fixture.parquet"
    pd.DataFrame([{"ticker": "SPY", "direction": "long", "asof": "2026-01-01",
        "entry_date": "2026-01-02", "exit_date": "2026-01-05", "channel": "detect_seasonal"}]).to_parquet(source)
    before = source.read_bytes()
    output = tmp_path / "artifacts" / "result.parquet"
    result = module.enrich(input_path=source, output_path=output)
    assert source.read_bytes() == before
    assert len(pd.read_parquet(output)) == len(result) == 1
    assert "p_value" in result
