"""Guards for the shared Denali site's redacted risk payload.

The contract: the shared site gets the market regime and nothing about the
book. ``assert_shared_payload_clean`` is the fail-closed gate both the writer
(build_risk_json) and the shared-site builder run.
"""
import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_risk_json import (
    SHARED_BANNED_PHRASES,
    SHARED_BANNED_SIZING_KEYS,
    assert_shared_payload_clean,
    redact_for_shared,
    write_shared_payload,
)
from scripts.build_shared_seasonals import build_shared_site, copy_risk_payload
from strategy_config import STRATEGY_BOOK

ROOT = Path(__file__).resolve().parents[1]
LIVE_PAYLOAD = ROOT / "data" / "site_risk.json"
STRATEGY_NAMES = [s["name"] for s in STRATEGY_BOOK]


def _asof(days_old: int = 0) -> str:
    return (dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=days_old)).isoformat()


def _payload(days_old: int = 0) -> dict:
    return {
        "built_at": "2026-09-18 10:00 UTC",
        "asof": _asof(days_old),
        "spy_last": 771.95,
        "regime_mult": 1.45,
        "price_ctx": {"regime_label": "Extended uptrend", "extension_200d": 0.11},
        "fragility": {"5d": 41.0, "21d": 50.2, "63d": 62.2},
        "signals": [{"name": "Dispersion", "on": False, "badge": "OFF"}],
        "forward_returns": {"63d": {"current_score": 52.9, "n_episodes": 30}},
        "trade_console": {"action_line": "Historical read: no trade."},
        "sizing_state": {
            "asof": _asof(days_old),
            "basis": "10d MA of 63d dial, append-only PIT parquet (sizes live orders)",
            "score": 52.9,
            "raw_63d": 62.2,
            "pit_start": "2026-07-02",
            "spark": {"dates": ["2026-09-17"], "ma": [52.9], "daily": [62.2]},
            "threshold": 50.0,
            "throttle_on": True,
            "gap_to_threshold": -2.9,
            "days_in_state": 4,
            "banded_strategies": [
                {"strategy": "Monday Dip", "bands": [[50.0, 999.0, 0.25]]},
            ],
            "throttled": [{"strategy": "Monday Dip", "mult": 0.25}],
            "episodes": [["2026-07-30", "2026-08-04"]],
            "exposure": {"mult": 0.0, "active_rule": "Rule 1", "reason": "Raw 21D 53.5 > 50"},
            "sleeve": {"position": "FLAT", "n_transitions": 0},
        },
        "nuggets": [
            {"title": "Fragility: neutral", "tone": "warn", "lines": ["Main risk dial 53."]},
            {"title": "Book posture: regime multiplier 1.45x", "tone": "good",
             "lines": ["The fragility framework's core-exposure dial says run full size."]},
        ],
    }


def _prices() -> pd.DataFrame:
    rows = []
    for ticker, base in (("SPY", 100.0), ("QQQ", 200.0)):
        for index, date in enumerate(pd.bdate_range("2000-01-03", periods=20)):
            close = base + index
            rows.append({
                "ticker": ticker,
                "date": date,
                "Open": close - 0.25,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": 1_000_000,
            })
    return pd.DataFrame(rows)


def test_redaction_removes_every_banned_sizing_key():
    shared = redact_for_shared(_payload())

    sizing = shared["sizing_state"]
    for key in SHARED_BANNED_SIZING_KEYS:
        assert key not in sizing
    assert sizing["score"] == 52.9
    assert sizing["raw_63d"] == 62.2
    assert sizing["spark"]["ma"] == [52.9]


def test_redaction_never_mutates_its_input():
    original = _payload()
    snapshot = json.dumps(original, sort_keys=True)

    redact_for_shared(original)

    assert json.dumps(original, sort_keys=True) == snapshot


def test_redaction_drops_strategy_names_and_book_posture():
    shared = redact_for_shared(_payload())
    text = json.dumps(shared)

    assert not [n for n in STRATEGY_NAMES if n in text]
    assert [n["title"] for n in shared["nuggets"]] == ["Fragility: neutral"]
    assert_shared_payload_clean(shared)


def test_redaction_leaves_market_regime_blocks_untouched():
    payload = _payload()
    shared = redact_for_shared(payload)

    for key in ("asof", "built_at", "spy_last", "regime_mult", "price_ctx",
                "fragility", "signals", "forward_returns", "trade_console"):
        assert shared[key] == payload[key]
    assert shared["shared_redacted"] is True


def test_assertion_raises_on_a_real_strategy_name():
    shared = redact_for_shared(_payload())
    shared["trade_console"]["action_line"] = f"Trim {STRATEGY_NAMES[0]} by half."

    with pytest.raises(ValueError, match="names STRATEGY_BOOK strategies"):
        assert_shared_payload_clean(shared)


def test_redaction_drops_the_sizing_basis_string():
    shared = redact_for_shared(_payload())

    assert "basis" not in shared["sizing_state"]
    assert "sizes live orders" not in json.dumps(shared).lower()


@pytest.mark.parametrize("phrase", SHARED_BANNED_PHRASES)
def test_assertion_raises_on_book_machinery_vocabulary(phrase):
    shared = redact_for_shared(_payload())
    shared["price_ctx"]["regime_label"] = f"Extended uptrend, {phrase.upper()} engaged"

    with pytest.raises(ValueError, match="describes book machinery"):
        assert_shared_payload_clean(shared)


def test_assertion_raises_on_a_surviving_policy_key():
    shared = redact_for_shared(_payload())
    shared["sizing_state"]["threshold"] = 50.0

    with pytest.raises(ValueError, match="sizing policy keys"):
        assert_shared_payload_clean(shared)


def test_assertion_raises_on_a_book_posture_nugget():
    shared = redact_for_shared(_payload())
    shared["nuggets"].append({"title": "Book posture: regime multiplier 1.45x", "lines": []})

    with pytest.raises(ValueError, match="book-posture nugget"):
        assert_shared_payload_clean(shared)


def test_writer_refuses_a_payload_it_cannot_clean(tmp_path: Path, monkeypatch):
    out = tmp_path / "shared.json"
    monkeypatch.setattr(
        "scripts.build_risk_json.redact_for_shared",
        lambda payload: {"sizing_state": {"throttled": [{"strategy": STRATEGY_NAMES[0]}]}},
    )

    assert write_shared_payload(_payload(), str(out)) is False
    assert not out.exists()


def test_writer_emits_a_clean_payload(tmp_path: Path):
    out = tmp_path / "shared.json"

    assert write_shared_payload(_payload(), str(out)) is True
    assert_shared_payload_clean(json.loads(out.read_text(encoding="utf-8")))


def test_builder_ships_a_fresh_payload_as_risk_json(tmp_path: Path):
    output = tmp_path / "shared"
    output.mkdir()
    source = tmp_path / "site_risk_shared.json"
    source.write_text(json.dumps(redact_for_shared(_payload())), encoding="utf-8")

    assert copy_risk_payload(output, source) is True
    shipped = json.loads((output / "data" / "risk.json").read_text(encoding="utf-8"))
    assert shipped["sizing_state"]["score"] == 52.9


def test_builder_declines_a_stale_payload(tmp_path: Path):
    output = tmp_path / "shared"
    output.mkdir()
    source = tmp_path / "site_risk_shared.json"
    source.write_text(json.dumps(redact_for_shared(_payload(days_old=9))), encoding="utf-8")

    assert copy_risk_payload(output, source) is False
    assert not (output / "data" / "risk.json").exists()


def test_builder_refuses_a_dirty_payload(tmp_path: Path):
    output = tmp_path / "shared"
    output.mkdir()
    source = tmp_path / "site_risk_shared.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")

    with pytest.raises(ValueError, match="names STRATEGY_BOOK strategies"):
        copy_risk_payload(output, source)
    assert not (output / "data" / "risk.json").exists()


def test_shared_build_carries_the_payload_and_validates(tmp_path: Path):
    prices = tmp_path / "prices.parquet"
    _prices().to_parquet(prices, index=False)
    source = tmp_path / "site_risk_shared.json"
    source.write_text(json.dumps(redact_for_shared(_payload())), encoding="utf-8")

    manifest = build_shared_site(
        prices, tmp_path / "shared", risk_payload=source, ranks=tmp_path / "absent.parquet"
    )

    assert manifest["risk_payload"] is True
    assert (tmp_path / "shared" / "data" / "risk.json").is_file()


def test_shared_build_fails_closed_on_a_dirty_payload(tmp_path: Path):
    prices = tmp_path / "prices.parquet"
    _prices().to_parquet(prices, index=False)
    source = tmp_path / "site_risk_shared.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")

    with pytest.raises(ValueError, match="names STRATEGY_BOOK strategies"):
        build_shared_site(
            prices, tmp_path / "shared", risk_payload=source, ranks=tmp_path / "absent.parquet"
        )


@pytest.mark.skipif(not LIVE_PAYLOAD.is_file(), reason="no local data/site_risk.json")
def test_live_payload_shape_redacts_clean():
    live = json.loads(LIVE_PAYLOAD.read_text(encoding="utf-8"))

    assert [n for n in STRATEGY_NAMES if n in json.dumps(live)], "fixture no longer exercises the leak"
    assert_shared_payload_clean(redact_for_shared(live))
