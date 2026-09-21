from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from episodic_pivot.daily_prices import YFINANCE_DAILY_PRICE_BASIS
from episodic_pivot.manifest import sha256_file
from episodic_pivot.schema import PremarketSnapshot
from scripts.run_episodic_pivot_shadow import _load_snapshots
from scripts.run_episodic_pivot_shadow import main as shadow_main

TARGET_DATE = "2026-08-25"
AS_OF = "2026-08-25T12:20:00Z"


def _provenance_input(tmp_path: Path) -> dict[str, str]:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    return {
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
        "record_type": "TRADINGVIEW_NORMALIZED_IMPORT",
    }


def _yfinance_wrapper(tmp_path: Path) -> dict:
    return {
        "record_type": "EP_YFINANCE_DAILY_ENRICHMENT_V1",
        "provider": "YFINANCE",
        "mode": "YFINANCE_DAILY_RESEARCH_ONLY",
        "daily_price_basis": YFINANCE_DAILY_PRICE_BASIS,
        "target_session_date": TARGET_DATE,
        "request": {
            "auto_adjust": True,
            "repair": True,
            "event_session_excluded": True,
            "local_price_cache_used": False,
        },
        "safety": {
            "research_only": True,
            "broker_contacted": False,
            "order_submission_allowed": False,
            "order_staging_performed": False,
        },
        "inputs": [_provenance_input(tmp_path)],
        "snapshots": [],
    }


def _ibkr_snapshot() -> dict:
    row = PremarketSnapshot(
        symbol="ABC",
        observed_at=AS_OF,
        previous_close=10.0,
        last=11.0,
        bid=10.99,
        ask=11.01,
        premarket_volume=100_000,
        premarket_open=10.8,
        premarket_high=11.1,
        premarket_low=10.7,
        premarket_vwap=10.95,
        prior_two_day_low=9.5,
        atr_14=0.5,
        avg_volume_20=1_000_000,
        addv_63=10_000_000,
        provider="IBKR",
        source="IBKR_TARGETED_READ_ONLY",
        session="premarket",
        target_session_date=TARGET_DATE,
        market_data_status="LIVE",
        premarket_metrics_at=AS_OF,
    ).to_dict()
    row.update(
        {
            "premarket_move_verification_status": "VERIFIED",
            "premarket_move_verification_source": "IBKR_TARGETED_READ_ONLY",
            "premarket_move_verified_at": AS_OF,
        }
    )
    return row


def _ibkr_wrapper(tmp_path: Path) -> dict:
    return {
        "record_type": "EP_IBKR_PREMARKET_CAPTURE_V1",
        "provider": "IBKR",
        "mode": "IBKR_READ_ONLY_SHADOW",
        "target_session_date": TARGET_DATE,
        "connection": {
            "port": 7496,
            "selected_port": 7496,
            "connected": True,
            "readonly_requested": True,
            "readonly": True,
        },
        "coverage": {"mode": "TARGETED_TRADINGVIEW_CANDIDATES"},
        "inputs": [_provenance_input(tmp_path)],
        "snapshots": [_ibkr_snapshot()],
    }


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _network_safety() -> dict:
    return {
        "research_only": True,
        "live_actions_enabled": False,
        "broker_route": "NONE",
        "order_submission_allowed": False,
        "order_staging_performed": False,
        "broker_contacted": False,
        "sheets_written": False,
        "r2_written": False,
        "publishing_performed": False,
        "production_deployed": False,
    }


def _refresh_chain(tmp_path: Path) -> tuple[dict, dict, Path, Path, dict]:
    run_id = f"EP-RUN-{TARGET_DATE}-network"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    target_path = _write_json(
        run_dir / "refresh_targets.json",
        {
            "schema_version": 1,
            "record_type": "EP_RESEARCH_QUOTE_REFRESH_TARGETS_V1",
            "source_run_id": run_id,
            "generated_at": AS_OF,
            "target_session_date": TARGET_DATE,
            "research_only": True,
            "broker_route": "NONE",
            "order_submission_allowed": False,
            "snapshots": [_ibkr_snapshot()],
        },
    )
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "search_provider": "GOOGLE_CSE",
        "safety": _network_safety(),
        "artifacts": {
            "refresh_targets.json": {
                "sha256": sha256_file(target_path),
                "size_bytes": target_path.stat().st_size,
            }
        },
    }
    manifest_path = _write_json(run_dir / "manifest.json", manifest)
    input_record = {
        "path": str(target_path.resolve()),
        "sha256": sha256_file(target_path),
        "record_type": "EP_RESEARCH_QUOTE_REFRESH_TARGETS_V1",
    }
    source_record = {
        "run_id": run_id,
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
    }
    return input_record, source_record, target_path, manifest_path, manifest


def _refresh_ibkr_wrapper(tmp_path: Path) -> tuple[dict, Path, Path, dict]:
    input_record, source_record, target_path, manifest_path, manifest = _refresh_chain(
        tmp_path
    )
    payload = _ibkr_wrapper(tmp_path)
    payload["inputs"] = [input_record]
    payload["source_manifests"] = [source_record]
    return payload, target_path, manifest_path, manifest


def _rewrite_manifest(payload: dict, manifest_path: Path, manifest: dict) -> None:
    _write_json(manifest_path, manifest)
    payload["source_manifests"][0]["sha256"] = sha256_file(manifest_path)


def test_research_rejects_unverified_input_even_with_yfinance_wrapper(tmp_path: Path):
    yfinance = _write_json(tmp_path / "yfinance.json", _yfinance_wrapper(tmp_path))
    unverified = _write_json(tmp_path / "raw.json", [])

    with pytest.raises(SystemExit, match="reject UNVERIFIED"):
        shadow_main(
            [
                "--snapshot",
                str(yfinance),
                "--snapshot",
                str(unverified),
                "--target-session-date",
                TARGET_DATE,
                "--as-of",
                AS_OF,
                "--run-research",
            ]
        )


def test_dry_run_preserves_legacy_unverified_input_acceptance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    yfinance = _write_json(tmp_path / "yfinance.json", _yfinance_wrapper(tmp_path))
    unverified = _write_json(tmp_path / "raw.json", [])

    assert (
        shadow_main(
            [
                "--snapshot",
                str(yfinance),
                "--snapshot",
                str(unverified),
                "--target-session-date",
                TARGET_DATE,
                "--as-of",
                AS_OF,
            ]
        )
        == 0
    )
    assert "Dry run: validated" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("connected", False),
        ("readonly_requested", False),
        ("readonly", False),
        ("selected_port", None),
        ("port", 4001),
    ],
)
def test_ibkr_wrapper_requires_connected_readonly_selected_port_evidence(
    tmp_path: Path, field: str, value: object
):
    payload = _ibkr_wrapper(tmp_path)
    payload["connection"][field] = value
    path = _write_json(tmp_path / f"bad-{field}.json", payload)

    with pytest.raises(ValueError, match="wrapper provenance is invalid"):
        _load_snapshots(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider", "TRADINGVIEW"),
        ("source", "IBKR_READ_ONLY"),
        ("session", "after_hours"),
        ("target_session_date", "2026-08-26"),
    ],
)
def test_ibkr_wrapper_requires_per_row_identity(
    tmp_path: Path, field: str, value: object
):
    payload = _ibkr_wrapper(tmp_path)
    payload["snapshots"][0][field] = value
    path = _write_json(tmp_path / f"bad-row-{field}.json", payload)

    with pytest.raises(ValueError, match="target-session|row provenance"):
        _load_snapshots(path)


def test_ibkr_wrapper_validates_frozen_premarket_evidence_when_present(
    tmp_path: Path,
):
    payload = _ibkr_wrapper(tmp_path)
    path = _write_json(tmp_path / "verified-ibkr.json", payload)

    snapshots, kind, warnings = _load_snapshots(path)

    assert kind == "IBKR"
    assert warnings == ("IBKR_PARTIAL_EXECUTION_REFRESH",)
    assert [snapshot.symbol for snapshot in snapshots] == ["ABC"]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"premarket_move_verification_status": "UNVERIFIED"},
            "evidence is invalid",
        ),
        (
            {
                "premarket_move_verification_status": "VERIFIED",
                "premarket_move_verification_source": "TRADINGVIEW_BROWSER_EXPORT",
                "premarket_move_verified_at": AS_OF,
            },
            "evidence is invalid",
        ),
        (
            {
                "premarket_move_verification_status": "VERIFIED",
                "premarket_move_verification_source": "IBKR_TARGETED_READ_ONLY",
                "premarket_move_verified_at": "2026-08-25T08:20:00-04:00",
            },
            "must be UTC",
        ),
    ],
)
def test_ibkr_wrapper_rejects_invalid_frozen_premarket_evidence(
    tmp_path: Path, updates: dict, message: str
):
    payload = deepcopy(_ibkr_wrapper(tmp_path))
    payload["snapshots"][0].update(updates)
    path = _write_json(tmp_path / "bad-verification.json", payload)

    with pytest.raises(ValueError, match=message):
        _load_snapshots(path)


def test_ibkr_wrapper_rejects_incomplete_frozen_premarket_evidence(tmp_path: Path):
    payload = _ibkr_wrapper(tmp_path)
    payload["snapshots"][0].pop("premarket_move_verified_at")
    path = _write_json(tmp_path / "incomplete-verification.json", payload)

    with pytest.raises(ValueError, match="evidence is incomplete"):
        _load_snapshots(path)


def test_night_tradingview_targeted_ibkr_wrapper_needs_no_source_manifest(
    tmp_path: Path,
):
    payload = _ibkr_wrapper(tmp_path)
    assert payload["inputs"][0]["record_type"] == "TRADINGVIEW_NORMALIZED_IMPORT"
    assert "source_manifests" not in payload

    snapshots, kind, _warnings = _load_snapshots(
        _write_json(tmp_path / "night-ibkr.json", payload)
    )

    assert kind == "IBKR"
    assert len(snapshots) == 1


def test_research_refresh_ibkr_wrapper_accepts_hashed_network_run_chain(
    tmp_path: Path,
):
    payload, _target_path, _manifest_path, _manifest = _refresh_ibkr_wrapper(tmp_path)

    snapshots, kind, _warnings = _load_snapshots(
        _write_json(tmp_path / "final-refresh.json", payload)
    )

    assert kind == "IBKR"
    assert len(snapshots) == 1


def test_research_refresh_ibkr_wrapper_requires_source_manifest_records(
    tmp_path: Path,
):
    payload, _target_path, _manifest_path, _manifest = _refresh_ibkr_wrapper(tmp_path)
    payload.pop("source_manifests")

    with pytest.raises(ValueError, match="requires source_manifests"):
        _load_snapshots(_write_json(tmp_path / "missing-manifest.json", payload))


def test_research_refresh_ibkr_wrapper_rechecks_manifest_file_digest(
    tmp_path: Path,
):
    payload, _target_path, manifest_path, manifest = _refresh_ibkr_wrapper(tmp_path)
    manifest["warnings"] = ["TAMPERED"]
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="manifest path or digest is invalid"):
        _load_snapshots(_write_json(tmp_path / "tampered-manifest.json", payload))


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("manifest", "schema_version", 1),
        ("manifest", "run_id", "EP-RUN-2026-08-25-other"),
        ("manifest", "search_provider", "OFFLINE_VERIFIED:source"),
        ("safety", "live_actions_enabled", True),
        ("safety", "broker_contacted", True),
    ],
)
def test_research_refresh_ibkr_wrapper_requires_network_manifest_identity_and_safety(
    tmp_path: Path, section: str, field: str, value: object
):
    payload, _target_path, manifest_path, manifest = _refresh_ibkr_wrapper(tmp_path)
    destination = manifest if section == "manifest" else manifest["safety"]
    destination[field] = value
    _rewrite_manifest(payload, manifest_path, manifest)

    with pytest.raises(
        ValueError, match="manifest identity, network provenance, or safety is invalid"
    ):
        _load_snapshots(_write_json(tmp_path / f"bad-manifest-{field}.json", payload))


def test_research_refresh_ibkr_wrapper_requires_manifest_run_directory_path(
    tmp_path: Path,
):
    payload, _target_path, manifest_path, _manifest = _refresh_ibkr_wrapper(tmp_path)
    copied = _write_json(
        tmp_path / "manifest-copy.json",
        json.loads(manifest_path.read_text(encoding="utf-8")),
    )
    payload["source_manifests"][0].update(
        {"path": str(copied.resolve()), "sha256": sha256_file(copied)}
    )

    with pytest.raises(ValueError, match="manifest path or digest is invalid"):
        _load_snapshots(_write_json(tmp_path / "bad-manifest-path.json", payload))


@pytest.mark.parametrize(
    ("artifact_field", "artifact_value"),
    [("sha256", "0" * 64), ("size_bytes", -1)],
)
def test_research_refresh_ibkr_wrapper_rechecks_target_artifact_hash_and_size(
    tmp_path: Path, artifact_field: str, artifact_value: object
):
    payload, _target_path, manifest_path, manifest = _refresh_ibkr_wrapper(tmp_path)
    manifest["artifacts"]["refresh_targets.json"][artifact_field] = artifact_value
    _rewrite_manifest(payload, manifest_path, manifest)

    with pytest.raises(ValueError, match="target artifact provenance is invalid"):
        _load_snapshots(
            _write_json(tmp_path / f"bad-artifact-{artifact_field}.json", payload)
        )


def test_research_refresh_ibkr_wrapper_rechecks_target_session_binding(
    tmp_path: Path,
):
    payload, target_path, manifest_path, manifest = _refresh_ibkr_wrapper(tmp_path)
    target = json.loads(target_path.read_text(encoding="utf-8"))
    target["target_session_date"] = "2026-08-26"
    _write_json(target_path, target)
    payload["inputs"][0]["sha256"] = sha256_file(target_path)
    manifest["artifacts"]["refresh_targets.json"] = {
        "sha256": sha256_file(target_path),
        "size_bytes": target_path.stat().st_size,
    }
    _rewrite_manifest(payload, manifest_path, manifest)

    with pytest.raises(ValueError, match="target identity, session, or safety"):
        _load_snapshots(_write_json(tmp_path / "wrong-target-date.json", payload))
