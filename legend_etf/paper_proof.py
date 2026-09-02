"""Validate reviewed IBKR paper evidence before live activation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .config import NY_TZ, STRATEGY_VERSION
from .ibkr_adapter import build_order_ref

PAPER_PROOF_PROTOCOL = "legend-ibkr-paper-proof-v1"
PAPER_PROOF_ATTESTATION = "I_REVIEWED_LEGEND_IBKR_PAPER_PROOF"
REQUIRED_DRILLS = frozenset(
    {
        "one_share_long",
        "one_share_short",
        "partial_fill",
        "restart_recovery",
        "disconnect_recovery",
        "order_ref_open_order_echo",
        "order_ref_execution_echo",
        "oca_type_2_time_exit",
        "ioc_parent_partial_children_active",
    }
)


def _read_once(path: Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    try:
        raw = source.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"IBKR paper proof is missing/invalid: {source}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(  # noqa: TRY004 - invalid external proof artifact
            "IBKR paper proof is not an object"
        )
    return payload, hashlib.sha256(raw).hexdigest()


def validate_paper_proof(
    path: Path,
    *,
    expected_sha256: str,
    expected_manifest_sha256: str,
    now: pd.Timestamp | None = None,
    max_age_days: int = 45,
) -> dict[str, Any]:
    """Require exact-build, reviewed paper evidence for broker semantics."""

    payload, digest = _read_once(path)
    expected_digest = str(expected_sha256).strip().lower()
    if len(expected_digest) != 64 or digest != expected_digest:
        raise RuntimeError("IBKR paper proof hash differs from runtime.env")
    allowed = {
        "protocol",
        "strategy_version",
        "source_manifest_sha256",
        "created_at",
        "entry_date",
        "paper_account",
        "paper_endpoint",
        "order_ref_echoes",
        "entry_parent_tif",
        "oca_type",
        "target_revision_clocks",
        "max_target_revision_latency_ms",
        "drills",
        "time_exit",
        "reviewed_attestation",
    }
    if set(payload) != allowed:
        raise RuntimeError("IBKR paper proof schema is invalid")
    if payload["protocol"] != PAPER_PROOF_PROTOCOL:
        raise RuntimeError("IBKR paper proof protocol mismatch")
    if payload["strategy_version"] != STRATEGY_VERSION:
        raise RuntimeError("IBKR paper proof strategy version mismatch")
    manifest_hash = str(expected_manifest_sha256).strip().lower()
    if payload["source_manifest_sha256"] != manifest_hash:
        raise RuntimeError("IBKR paper proof is not for this exact deployment")
    created = pd.Timestamp(payload["created_at"])
    if created.tz is None:
        raise RuntimeError("IBKR paper proof created_at must be timezone-aware")
    wall_clock = pd.Timestamp.now(tz=NY_TZ) if now is None else pd.Timestamp(now)
    if wall_clock.tz is None:
        raise ValueError("IBKR paper proof validation clock must be timezone-aware")
    age = wall_clock.tz_convert("UTC") - created.tz_convert("UTC")
    if age < pd.Timedelta(0) or age > pd.Timedelta(days=max_age_days):
        raise RuntimeError("IBKR paper proof is future-dated or stale")
    account = str(payload["paper_account"]).strip().upper()
    endpoint = payload["paper_endpoint"]
    if not account.startswith("DU"):
        raise RuntimeError("IBKR proof was not produced in a paper account")
    if not isinstance(endpoint, dict) or set(endpoint) != {
        "host",
        "port",
        "client_id",
    }:
        raise RuntimeError("IBKR paper proof endpoint is invalid")
    if (
        str(endpoint["host"]).strip() not in {"127.0.0.1", "localhost"}
        or int(endpoint["port"]) not in {4002, 7497}
        or int(endpoint["client_id"]) <= 0
    ):
        raise RuntimeError("IBKR proof endpoint is not a local paper endpoint")
    try:
        entry_date = pd.Timestamp(payload["entry_date"]).date().isoformat()
    except (TypeError, ValueError) as exc:
        raise RuntimeError("IBKR paper proof entry date is invalid") from exc
    if entry_date != str(payload["entry_date"]):
        raise RuntimeError("IBKR paper proof entry date must be YYYY-MM-DD")
    echoes = payload["order_ref_echoes"]
    if not isinstance(echoes, list) or len(echoes) != 2:
        raise RuntimeError("IBKR paper proof needs one long and one short orderRef")
    directions: set[int] = set()
    for echo in echoes:
        if not isinstance(echo, dict) or set(echo) != {
            "etf",
            "direction",
            "expected",
            "open_order",
            "execution",
        }:
            raise RuntimeError("IBKR paper orderRef echo schema is invalid")
        etf = str(echo["etf"]).upper()
        direction = int(echo["direction"])
        if etf not in {"SPY", "QQQ", "IWM"} or direction not in {-1, 1}:
            raise RuntimeError("IBKR paper orderRef ETF/direction is invalid")
        expected_reference = build_order_ref(etf, direction, entry_date)
        if (
            str(echo["expected"]) != expected_reference
            or str(echo["open_order"]) != expected_reference
            or str(echo["execution"]) != expected_reference
        ):
            raise RuntimeError("IBKR paper proof did not echo the exact Legend orderRef")
        directions.add(direction)
    if directions != {-1, 1}:
        raise RuntimeError("IBKR paper proof must cover both long and short orderRefs")
    if str(payload["entry_parent_tif"]).upper() != "IOC":
        raise RuntimeError("IBKR paper proof did not exercise an IOC entry parent")
    if int(payload["oca_type"]) != 2:
        raise RuntimeError("IBKR paper proof did not exercise OCA type 2")
    if payload["target_revision_clocks"] != ["09:46", "10:01", "10:16"]:
        raise RuntimeError("IBKR paper proof target-revision clocks are incomplete")
    try:
        revision_latency = float(payload["max_target_revision_latency_ms"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("IBKR paper target-revision latency is invalid") from exc
    if (
        not math.isfinite(revision_latency)
        or revision_latency < 0
        or revision_latency > 1_000
    ):
        raise RuntimeError("IBKR paper target revisions exceeded one second")
    drills = payload["drills"]
    if not isinstance(drills, dict) or set(drills) != REQUIRED_DRILLS:
        raise RuntimeError("IBKR paper proof drill set is incomplete")
    if any(value is not True for value in drills.values()):
        raise RuntimeError("one or more required IBKR paper drills failed")
    exit_proof = payload["time_exit"]
    required_exit = {
        "scheduled_at",
        "filled_at",
        "delay_seconds",
        "remaining_owned_shares",
        "working_orders_after",
        "over_exit",
        "target_terminal",
    }
    if not isinstance(exit_proof, dict) or set(exit_proof) != required_exit:
        raise RuntimeError("IBKR paper time-exit proof is invalid")
    scheduled = pd.Timestamp(exit_proof["scheduled_at"])
    filled = pd.Timestamp(exit_proof["filled_at"])
    if scheduled.tz is None or filled.tz is None:
        raise RuntimeError("IBKR paper time-exit timestamps must be timezone-aware")
    measured_delay = (filled.tz_convert("UTC") - scheduled.tz_convert("UTC")).total_seconds()
    declared_delay = float(exit_proof["delay_seconds"])
    if (
        not math.isfinite(declared_delay)
        or declared_delay < 0
        or declared_delay > 5.0
        or not math.isclose(measured_delay, declared_delay, abs_tol=0.05)
        or scheduled.tz_convert(NY_TZ).strftime("%H:%M:%S") != "10:30:00"
        or scheduled.tz_convert(NY_TZ).date().isoformat() != entry_date
        or int(exit_proof["remaining_owned_shares"]) != 0
        or int(exit_proof["working_orders_after"]) != 0
        or exit_proof["over_exit"] is not False
        or exit_proof["target_terminal"] is not True
    ):
        raise RuntimeError("IBKR paper proof did not prove the exact safe 10:30 exit")
    if payload["reviewed_attestation"] != PAPER_PROOF_ATTESTATION:
        raise RuntimeError("IBKR paper proof lacks the explicit review attestation")
    return payload
