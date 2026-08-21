import json
from pathlib import Path

from scripts.validate_site_freshness import validate_site


SITE_BUILD_ID = "123"


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _stamped(value):
    return {"_site_build_id": SITE_BUILD_ID, **value}


def _site(tmp_path: Path):
    data = tmp_path / "data"
    flags = {
        "strategy_daily": True,
        "positions": True,
        "exposure": True,
        "trade_mtm": True,
        "ideas": True,
        "signals": True,
        "risk": True,
        "seasonality": True,
        "macro_sznl": True,
        "fundamentals": True,
        "health": True,
    }
    health = _stamped({
        "build_id": SITE_BUILD_ID,
        "built_at": "2026-08-06T12:00:00Z",
        "prev_td": "2026-08-05",
        "artifacts": {
            "ledger": {"status": "fresh"},
            "master_prices": {"status": "fresh"},
            "ideas": {"status": "fresh", "asof": "2026-08-05"},
            "signals": {"status": "fresh", "fetched_at": "2026-08-06 12:00 UTC"},
        },
    })
    _write(data / "meta.json", _stamped({
        "build_id": SITE_BUILD_ID,
        "built_at": "2026-08-06T12:00:00Z",
        "freshness": health,
        "payloads": flags,
    }))
    _write(data / "health.json", health)
    _write(data / "positions.json", _stamped({
        "asof": "2026-08-06",
        "positions": [{"Ticker": "SPY", "Days_To_Time_Stop": 2}],
    }))
    _write(data / "ideas.json", _stamped({
        "meta": {"asof": "2026-08-05"},
        "candidates": [],
    }))
    _write(data / "signals.json", _stamped({
        "fetched_at": "2026-08-06 12:00 UTC",
        "tabs": {"Order_Staging": [], "Overflow": []},
    }))
    _write(data / "risk.json", {
        "built_at": "2026-08-06 12:00 UTC",
        "asof": "2026-08-05",
        "sizing_state": {"asof": "2026-08-05"},
        "signal_detail": {
            "Defensive Leadership": {"current": {"value": -1.2}},
            "Seasonal Rank Divergence": {"current": {"value": 2.4}},
        },
        "trade_console": {"state": "ok"},
    })
    _write(data / "seasonality" / "manifest.json", {
        "asof": "2026-08-05",
        "tickers": {"SPY": {"file": "t/U1BZ.bin"}},
    })
    (data / "seasonality" / "t").mkdir(parents=True, exist_ok=True)
    (data / "seasonality" / "t" / "U1BZ.bin").write_bytes(b"SLB2")
    _write(data / "seasonality" / "theses.json", {"asof": "2026-08-05"})
    _write(data / "seasonality" / "macro.json", {
        "asof": "2026-08-05",
        "sznl_available": True,
        "rows": [{"ticker": "SPY", "price": 100.0}],
    })
    _write(data / "fundamentals.json", {
        "as_of": "2026-08-05",
        "status": "NO_REVIEW",
        "reviews": [],
        "active_research": [],
        "live_actions_enabled": False,
    })
    for name in ("trades.json", "strategy_daily.json", "exposure.json", "trade_mtm.json"):
        _write(data / name, _stamped({}))
    return data


def test_fresh_complete_site_passes(tmp_path):
    _site(tmp_path)
    assert validate_site(str(tmp_path)) == []


def test_embedded_freshness_must_exactly_match_standalone_health(tmp_path):
    data = _site(tmp_path)
    meta = json.loads((data / "meta.json").read_text(encoding="utf-8"))
    meta["freshness"]["build_id"] = "adjacent-deployment"
    _write(data / "meta.json", meta)

    problems = validate_site(str(tmp_path))
    assert any("embed the exact health.json" in problem for problem in problems)


def test_equal_timestamps_do_not_hide_cross_build_mix(tmp_path):
    data = _site(tmp_path)
    health = json.loads((data / "health.json").read_text(encoding="utf-8"))
    health["build_id"] = "adjacent-deployment"
    health["_site_build_id"] = "adjacent-deployment"
    _write(data / "health.json", health)

    problems = validate_site(str(tmp_path))
    assert any("different build identities" in problem for problem in problems)


def test_page_payload_identity_must_match_meta(tmp_path):
    data = _site(tmp_path)
    positions = json.loads((data / "positions.json").read_text(encoding="utf-8"))
    positions["_site_build_id"] = "adjacent-deployment"
    _write(data / "positions.json", positions)

    problems = validate_site(str(tmp_path))
    assert any("data/positions.json belongs to a different site build" in p for p in problems)


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


def test_stale_or_degraded_risk_payload_blocks_deploy(tmp_path):
    data = _site(tmp_path)
    risk = json.loads((data / "risk.json").read_text(encoding="utf-8"))
    risk["asof"] = "2026-08-01"
    risk["signal_detail"]["Defensive Leadership"]["current"]["value"] = None
    risk.pop("trade_console")
    _write(data / "risk.json", risk)

    problems = validate_site(str(tmp_path))
    assert any("Risk payload as-of 2026-08-01" in problem for problem in problems)
    assert any("Defensive Leadership" in problem for problem in problems)
    assert any("trade console is unavailable" in problem for problem in problems)


def test_incomplete_seasonality_payloads_block_deploy(tmp_path):
    data = _site(tmp_path)
    macro = json.loads((data / "seasonality/macro.json").read_text(encoding="utf-8"))
    macro["sznl_available"] = False
    macro["rows"].append({"ticker": "TIP", "price": None})
    _write(data / "seasonality/macro.json", macro)
    (data / "seasonality/t/U1BZ.bin").rename(
        data / "seasonality/t/U1BZ.missing")

    problems = validate_site(str(tmp_path))
    assert any("Seasonality Lab is missing 1 ticker payload" in problem for problem in problems)
    assert any("Macro seasonal ranks are unavailable" in problem for problem in problems)
    assert any("Macro Seasonality is missing prices" in problem for problem in problems)


def test_missing_or_stale_fundamentals_do_not_block_deploy(tmp_path):
    data = _site(tmp_path)
    meta = json.loads((data / "meta.json").read_text(encoding="utf-8"))
    meta["payloads"]["fundamentals"] = False
    _write(data / "meta.json", meta)
    _write(data / "fundamentals.json", {
        "as_of": "2026-08-01",
        "status": "NO_REVIEW",
        "live_actions_enabled": False,
    })

    assert validate_site(str(tmp_path)) == []

    (data / "fundamentals.json").rename(data / "fundamentals.disabled.json")
    assert validate_site(str(tmp_path)) == []


def test_fundamentals_contents_do_not_participate_in_deploy_gate(tmp_path):
    data = _site(tmp_path)
    payload = json.loads((data / "fundamentals.json").read_text(encoding="utf-8"))
    payload["live_actions_enabled"] = True
    _write(data / "fundamentals.json", payload)

    assert validate_site(str(tmp_path)) == []


def test_production_gate_requires_matching_r2_provenance(tmp_path):
    data = _site(tmp_path)
    meta = json.loads((data / "meta.json").read_text(encoding="utf-8"))
    meta["data_provenance"] = {
        "mode": "r2-only",
        "run_id": "123",
        "source_sha": "abc",
    }
    _write(data / "meta.json", meta)
    _write(data / "provenance.json", {
        "mode": "r2-only",
        "run_id": "123",
        "source_sha": "abc",
        "entries": [{"name": "master_prices"}],
    })
    assert validate_site(str(tmp_path), require_r2_provenance=True) == []

    meta["data_provenance"]["run_id"] = "other"
    _write(data / "meta.json", meta)
    problems = validate_site(str(tmp_path), require_r2_provenance=True)
    assert any("identify different builds" in problem for problem in problems)
