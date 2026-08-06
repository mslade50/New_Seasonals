import json
from pathlib import Path

from scripts.validate_site_freshness import validate_site


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _site(tmp_path: Path):
    data = tmp_path / "data"
    flags = {
        "strategy_daily": True,
        "positions": True,
        "exposure": True,
        "trade_mtm": True,
        "ideas": True,
        "signals": True,
        "health": True,
    }
    _write(data / "meta.json", {"built_at": "2026-08-06 12:00 UTC", "payloads": flags})
    _write(data / "health.json", {
        "built_at": "2026-08-06 12:00 UTC",
        "prev_td": "2026-08-05",
        "artifacts": {
            "ledger": {"status": "fresh"},
            "master_prices": {"status": "fresh"},
            "ideas": {"status": "fresh", "asof": "2026-08-05"},
            "signals": {"status": "fresh", "fetched_at": "2026-08-06 12:00 UTC"},
        },
    })
    _write(data / "positions.json", {
        "asof": "2026-08-06",
        "positions": [{"Ticker": "SPY", "Days_To_Time_Stop": 2}],
    })
    _write(data / "ideas.json", {
        "meta": {"asof": "2026-08-05"},
        "candidates": [],
    })
    _write(data / "signals.json", {
        "fetched_at": "2026-08-06 12:00 UTC",
        "tabs": {"Order_Staging": [], "Overflow": []},
    })
    for name in ("trades.json", "strategy_daily.json", "exposure.json", "trade_mtm.json"):
        _write(data / name, {})
    return data


def test_fresh_complete_site_passes(tmp_path):
    _site(tmp_path)
    assert validate_site(str(tmp_path)) == []


def test_stale_ledger_and_expired_open_position_block_deploy(tmp_path):
    data = _site(tmp_path)
    health = json.loads((data / "health.json").read_text(encoding="utf-8"))
    health["artifacts"]["ledger"]["status"] = "stale"
    _write(data / "health.json", health)
    _write(data / "positions.json", {
        "asof": "2026-08-06",
        "positions": [{"Ticker": "SPY", "Days_To_Time_Stop": -1}],
    })

    problems = validate_site(str(tmp_path))
    assert any("ledger health is stale" in problem for problem in problems)
    assert any("past their time stop" in problem for problem in problems)


def test_stale_ideas_or_missing_current_signals_block_deploy(tmp_path):
    data = _site(tmp_path)
    meta = json.loads((data / "meta.json").read_text(encoding="utf-8"))
    meta["payloads"]["signals"] = False
    _write(data / "meta.json", meta)
    health = json.loads((data / "health.json").read_text(encoding="utf-8"))
    health["artifacts"]["ideas"]["status"] = "stale"
    health["artifacts"]["signals"]["status"] = "missing"
    _write(data / "health.json", health)
    _write(data / "ideas.json", {
        "meta": {"asof": "2026-06-08", "unavailable": True},
        "candidates": [],
    })

    problems = validate_site(str(tmp_path))
    assert any("Seasonal ideas health is stale" in problem for problem in problems)
    assert any("current staged-signals payload is unavailable" in problem for problem in problems)
