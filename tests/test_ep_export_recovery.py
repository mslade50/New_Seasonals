"""Count races can seed IBKR, never directly authorize ATR or news research."""

import json

import pytest

from episodic_pivot.tradingview import TradingViewImportError, import_tradingview_csv
from scripts.capture_ep_daily_yfinance import _load_discovery_inputs
from scripts.capture_ep_premarket_ibkr import _load_target_rows_many_with_provenance

HEADER = "Symbol,Name,Exchange,Pre-market Price,Pre-market Change,Pre-market Change %,Pre-market Volume\n"


def make_import(tmp_path, count=3, **kwargs):
    source = tmp_path / "export.csv"
    source.write_text(
        HEADER
        + "".join(
            f"TEST{i},Test Company {i},NYSE,11,1,10%,200000\n" for i in range(count)
        )
    )
    args = {
        "session": "premarket",
        "captured_at": "2026-09-09T08:24:06-04:00",
        "saved_screen_id": "yftOvM3e",
        "reported_result_count": count - 1,
        "post_download_result_count": count - 1,
        "allow_count_mismatch_for_ibkr": True,
    }
    args.update(kwargs)
    result = import_tradingview_csv(source, **args)
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(result.to_dict()))
    return result, path


@pytest.mark.parametrize("count,before", [(146, 145), (148, 146)])
def test_actual_failure_shapes_can_only_seed_independent_ibkr(tmp_path, count, before):
    result, path = make_import(
        tmp_path, count, reported_result_count=before, post_download_result_count=before
    )
    assert not result.result_count_verified
    assert result.result_count_verification == "COUNT_MISMATCH_IBKR_SEED_ONLY"
    assert all(
        not s.premarket_move_verified_at and not s.tradeable for s in result.snapshots
    )
    rows, session, raw_count, inputs = _load_target_rows_many_with_provenance([path])
    assert len(rows) == raw_count == count
    assert session == "2026-09-09"
    assert (
        inputs[0]["discovery_warning"]
        == "TRADINGVIEW_COUNT_MISMATCH_IBKR_REVERIFIED_ONLY"
    )
    with pytest.raises(ValueError, match="count/provenance"):
        _load_discovery_inputs([path])


@pytest.mark.parametrize(
    "overrides",
    [
        {"reported_result_count": 4, "post_download_result_count": 4},
        {"reported_result_count": -1},
        {"post_download_result_count": 0},
        {"post_download_result_count": None},
        {"saved_screen_id": "wrong"},
        {"captured_at": "2026-09-09T10:00:00-04:00"},
        {"allow_count_mismatch_for_ibkr": False},
    ],
)
def test_recovery_keeps_existing_rejections(tmp_path, overrides):
    with pytest.raises(TradingViewImportError):
        make_import(tmp_path, **overrides)


@pytest.mark.parametrize("mutation", ["symbol", "hash", "verified", "source"])
def test_seed_loader_replays_source_and_rejects_tampering(tmp_path, mutation):
    result, path = make_import(tmp_path)
    raw = result.to_dict()
    if mutation == "symbol":
        raw["snapshots"][0]["symbol"] = "OTHER"
    elif mutation == "hash":
        raw["source_file_sha256"] = "0" * 64
    elif mutation == "verified":
        raw["result_count_verified"] = True
    else:
        (tmp_path / "export.csv").write_text(HEADER)
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        _load_target_rows_many_with_provenance([path])


def test_exact_match_keeps_independent_tradingview_path(tmp_path):
    result, path = make_import(
        tmp_path, reported_result_count=3, post_download_result_count=3
    )
    assert result.result_count_verified
    assert result.result_count_verification == "EXACT_MATCH"
    snapshots, *_ = _load_discovery_inputs([path])
    assert len(snapshots) == 3
