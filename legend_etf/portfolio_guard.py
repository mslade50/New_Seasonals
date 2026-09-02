"""Fail-closed shared equity-index portfolio risk budget validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import NY_TZ
from .storage import atomic_write_json, exclusive_file_lock, utc_now_iso

PORTFOLIO_BUDGET_PROTOCOL = "legend-equity-index-risk-budget-v2"
PORTFOLIO_LOCK_NAME = "equity_index_cluster_budget.lock"


@dataclass(frozen=True)
class PortfolioCapacity:
    remaining_long_bps: float
    remaining_short_bps: float
    remaining_gross_bps: float


@dataclass(frozen=True)
class PortfolioRequirement:
    long_bps: float
    short_bps: float
    gross_bps: float


def _capacity(value: Any, *, account: str) -> PortfolioCapacity:
    if not isinstance(value, dict):
        raise RuntimeError(  # noqa: TRY004 - invalid external risk artifact
            f"portfolio budget for {account} is not an object"
        )
    fields = {
        "remaining_long_bps",
        "remaining_short_bps",
        "remaining_gross_bps",
    }
    if set(value) != fields:
        raise RuntimeError(f"portfolio budget fields are invalid for {account}")
    numbers = {field: float(value[field]) for field in fields}
    if any(
        not math.isfinite(number) or not 0 <= number <= 10_000
        for number in numbers.values()
    ):
        raise RuntimeError(f"portfolio budget is invalid for {account}")
    return PortfolioCapacity(**numbers)


def _requirement(value: Any, *, account: str) -> PortfolioRequirement:
    if isinstance(value, PortfolioRequirement):
        result = value
    elif isinstance(value, Mapping):
        if set(value) != {"long_bps", "short_bps", "gross_bps"}:
            raise RuntimeError(f"portfolio reservation fields are invalid for {account}")
        result = PortfolioRequirement(
            long_bps=float(value["long_bps"]),
            short_bps=float(value["short_bps"]),
            gross_bps=float(value["gross_bps"]),
        )
    else:
        raise RuntimeError(  # noqa: TRY004 - invalid external risk artifact
            f"portfolio reservation for {account} is not an object"
        )
    if any(
        not math.isfinite(number) or not 0 <= number <= 10_000
        for number in (result.long_bps, result.short_bps, result.gross_bps)
    ):
        raise RuntimeError(f"portfolio reservation is invalid for {account}")
    if result.gross_bps + 1e-9 < result.long_bps + result.short_bps:
        raise RuntimeError(f"portfolio reservation gross is too small for {account}")
    return result


def portfolio_budget_lock_path(reservation_dir: Path) -> Path:
    return Path(reservation_dir) / PORTFOLIO_LOCK_NAME


def _read_once(path: Path) -> tuple[dict[str, Any], bytes, str]:
    source = Path(path)
    try:
        raw = source.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"shared portfolio budget is missing/invalid: {source}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(  # noqa: TRY004 - invalid external risk artifact
            f"shared portfolio budget is not an object: {source}"
        )
    return payload, raw, hashlib.sha256(raw).hexdigest()


def _validate_payload(
    payload: dict[str, Any],
    *,
    entry_date: str,
    account_ids: list[str],
    expected_manifest_sha256: str,
    now: pd.Timestamp | None,
) -> dict[str, PortfolioCapacity]:
    allowed = {
        "protocol",
        "entry_date",
        "generated_at",
        "expires_at",
        "source_manifest_sha256",
        "risk_basis",
        "accounts",
        "reservations",
    }
    if set(payload) != allowed:
        raise RuntimeError("shared portfolio budget schema is invalid")
    if payload["protocol"] != PORTFOLIO_BUDGET_PROTOCOL:
        raise RuntimeError("shared portfolio budget protocol mismatch")
    if payload["entry_date"] != entry_date:
        raise RuntimeError("shared portfolio budget is not for this entry date")
    if payload["risk_basis"] != "stress_atr_bps":
        raise RuntimeError("shared portfolio budget risk basis mismatch")
    expected_hash = str(expected_manifest_sha256).strip().lower()
    if payload["source_manifest_sha256"] != expected_hash:
        raise RuntimeError("portfolio budget was not produced by this deployment")
    generated = pd.Timestamp(payload["generated_at"])
    expires = pd.Timestamp(payload["expires_at"])
    if generated.tz is None or expires.tz is None:
        raise RuntimeError("portfolio budget timestamps must be timezone-aware")
    generated_et = generated.tz_convert(NY_TZ)
    if not (
        pd.Timestamp(f"{entry_date} 08:30", tz=NY_TZ)
        <= generated_et
        <= pd.Timestamp(f"{entry_date} 09:25", tz=NY_TZ)
    ):
        raise RuntimeError("portfolio budget must be finalized from 08:30-09:25 ET")
    expected_expiry = pd.Timestamp(f"{entry_date} 09:31:20", tz=NY_TZ)
    if expires.tz_convert(NY_TZ) != expected_expiry:
        raise RuntimeError("portfolio budget expiry must be exactly 09:31:20 ET")
    wall_clock = pd.Timestamp.now(tz=NY_TZ) if now is None else pd.Timestamp(now)
    if wall_clock.tz is None:
        raise ValueError("portfolio budget validation clock must be timezone-aware")
    if wall_clock.tz_convert(NY_TZ) > expected_expiry:
        raise RuntimeError("portfolio budget expired before entry transmission")
    accounts = payload["accounts"]
    if not isinstance(accounts, dict) or set(accounts) != set(account_ids):
        raise RuntimeError("portfolio budget exact account set mismatch")
    capacities = {
        account: _capacity(accounts[account], account=account)
        for account in account_ids
    }
    reservations = payload["reservations"]
    if not isinstance(reservations, dict):
        raise RuntimeError(  # noqa: TRY004 - invalid external risk artifact
            "portfolio budget reservations must be an object"
        )
    for token, reservation in reservations.items():
        if not str(token).strip() or not isinstance(reservation, dict):
            raise RuntimeError("portfolio budget contains an invalid reservation")
        if set(reservation) != {
            "owner",
            "created_at",
            "status",
            "accounts",
            "signal_ids",
        }:
            raise RuntimeError("portfolio budget reservation schema is invalid")
        if reservation["status"] != "reserved":
            raise RuntimeError("portfolio budget reservation status is invalid")
        stamp = pd.Timestamp(reservation["created_at"])
        if stamp.tz is None:
            raise RuntimeError("portfolio reservation timestamp must be timezone-aware")
        reserved_accounts = reservation["accounts"]
        if not isinstance(reserved_accounts, dict) or not set(
            reserved_accounts
        ).issubset(accounts):
            raise RuntimeError("portfolio reservation account set is invalid")
        for account, value in reserved_accounts.items():
            _requirement(value, account=account)
        signal_ids = reservation["signal_ids"]
        if not isinstance(signal_ids, list) or any(
            not isinstance(signal_id, str) or not signal_id for signal_id in signal_ids
        ):
            raise RuntimeError("portfolio reservation signal IDs are invalid")
    return capacities


def validate_portfolio_budget(
    path: Path,
    *,
    entry_date: str,
    account_ids: list[str],
    expected_manifest_sha256: str,
    now: pd.Timestamp | None = None,
) -> tuple[dict[str, PortfolioCapacity], str]:
    """Validate one finalized pre-open capacity snapshot and return its hash."""

    payload, _raw, digest = _read_once(path)
    capacities = _validate_payload(
        payload,
        entry_date=entry_date,
        account_ids=account_ids,
        expected_manifest_sha256=expected_manifest_sha256,
        now=now,
    )
    return capacities, digest


@contextmanager
def reserve_portfolio_capacity(
    path: Path,
    *,
    lock_path: Path,
    entry_date: str,
    account_requirements: Mapping[str, PortfolioRequirement],
    owner_token: str,
    signal_ids: list[str],
    expected_manifest_sha256: str,
    now: pd.Timestamp | None = None,
) -> Iterator[tuple[dict[str, PortfolioCapacity], str]]:
    """Atomically debit one full account batch and hold the lock through transmit."""

    accounts = sorted(str(account) for account in account_requirements)
    if not owner_token.strip() or not signal_ids:
        raise ValueError("portfolio reservation requires an owner token and signals")
    requirements = {
        account: _requirement(account_requirements[account], account=account)
        for account in accounts
    }
    with exclusive_file_lock(lock_path):
        payload, _raw, _digest = _read_once(path)
        capacities = _validate_payload(
            payload,
            entry_date=entry_date,
            account_ids=accounts,
            expected_manifest_sha256=expected_manifest_sha256,
            now=now,
        )
        reservations = payload["reservations"]
        existing = reservations.get(owner_token)
        expected_accounts = {
            account: {
                "long_bps": requirements[account].long_bps,
                "short_bps": requirements[account].short_bps,
                "gross_bps": requirements[account].gross_bps,
            }
            for account in accounts
        }
        if existing is not None:
            if (
                existing.get("owner") != "Legend ETF"
                or existing.get("status") != "reserved"
                or existing.get("accounts") != expected_accounts
                or sorted(existing.get("signal_ids") or []) != sorted(signal_ids)
            ):
                raise RuntimeError("portfolio reservation token was reused inconsistently")
        else:
            for account in accounts:
                capacity = capacities[account]
                requirement = requirements[account]
                if (
                    requirement.long_bps > capacity.remaining_long_bps + 1e-9
                    or requirement.short_bps > capacity.remaining_short_bps + 1e-9
                    or requirement.gross_bps > capacity.remaining_gross_bps + 1e-9
                ):
                    raise RuntimeError(
                        f"shared equity-index capacity changed before transmit for {account}"
                    )
            for account in accounts:
                capacity = capacities[account]
                requirement = requirements[account]
                payload["accounts"][account] = {
                    "remaining_long_bps": capacity.remaining_long_bps
                    - requirement.long_bps,
                    "remaining_short_bps": capacity.remaining_short_bps
                    - requirement.short_bps,
                    "remaining_gross_bps": capacity.remaining_gross_bps
                    - requirement.gross_bps,
                }
            reservations[owner_token] = {
                "owner": "Legend ETF",
                "created_at": utc_now_iso(),
                "status": "reserved",
                "accounts": expected_accounts,
                "signal_ids": sorted(signal_ids),
            }
            atomic_write_json(Path(path), payload)
        updated, _updated_raw, digest = _read_once(path)
        updated_capacities = _validate_payload(
            updated,
            entry_date=entry_date,
            account_ids=accounts,
            expected_manifest_sha256=expected_manifest_sha256,
            now=now,
        )
        yield updated_capacities, digest
