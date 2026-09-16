"""Minute-sensitive Legend ETF session orchestration.

The runner uses a read-only primary market-data connection and separate,
account-owned execution connections.  It is safe to import and defaults to a
non-mutating dry run.
"""

from __future__ import annotations

import ctypes
import math
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .calendar import require_full_entry_session, rth_bar_starts
from .config import (
    MARKETS,
    NY_TZ,
    PA_RISK,
    PRIMARY_RISK,
    RULES,
    STRATEGY_VERSION,
)
from .core import (
    ema_from_seed,
    entry_decision,
    prior_wilder_atr14,
    round_limit,
)
from .ibkr_adapter import (
    DEAD_STATUSES,
    Endpoint,
    IBKRConnection,
    LiveGate,
    PhysicalIsolationMismatch,
    PreTransmitCheckBlocked,
    build_order_ref,
    endpoints_from_env,
    entry_filled_quantity,
    owned_quantity,
    realtime_bars_to_frame,
    trade_ticks_to_frame,
    validate_correction_audit_gate,
    validate_live_gate,
    validate_recovery_gate,
)
from .paper_proof import validate_paper_proof
from .portfolio_guard import (
    PortfolioCapacity,
    PortfolioRequirement,
    portfolio_budget_lock_path,
    reserve_portfolio_capacity,
    validate_portfolio_budget,
)
from .reservations import (
    ReservationBook,
    load_attested_external_guard,
    validate_guard_manifest,
)
from .sizing import SizeRequest, SizeResult, risk_profile_from_env, size_batch
from .storage import (
    StateStore,
    atomic_write_json,
    content_hash,
    read_json,
    signal_identity,
    validate_plan,
)


def _process_started_at_utc() -> str:
    """Return the current process creation time for watchdog PID fencing."""

    if os.name != "nt":  # pragma: no cover - production host is Windows
        return datetime.now(timezone.utc).isoformat()

    class FileTime(ctypes.Structure):
        _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    get_process_times = kernel32.GetProcessTimes
    get_process_times.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
    ]
    get_process_times.restype = ctypes.c_bool
    created = FileTime()
    exited = FileTime()
    kernel = FileTime()
    user = FileTime()
    if not get_process_times(
        get_current_process(),
        ctypes.byref(created),
        ctypes.byref(exited),
        ctypes.byref(kernel),
        ctypes.byref(user),
    ):
        raise OSError(ctypes.get_last_error(), "GetProcessTimes failed")
    ticks = (int(created.high) << 32) | int(created.low)
    unix_seconds = ticks / 10_000_000 - 11_644_473_600
    return datetime.fromtimestamp(unix_seconds, tz=timezone.utc).isoformat()


@dataclass
class MarketContext:
    root: str
    etf: str
    setup_date: str
    contract: Any
    initial_ema: float
    atr14: float
    ex_dividend: bool
    dividend_detail: str
    realtime_bars: Any | None = None
    trade_ticks: Any | None = None
    direction: int | None = None
    initial_target: float | None = None
    decision_price: float | None = None
    decision_observed_at: str | None = None


@dataclass
class ManagedTrade:
    signal_id: str
    connection: IBKRConnection
    context: MarketContext
    direction: int
    order_ref: str
    target_trade: Any | None
    current_ema: float
    state: dict[str, Any]
    allow_target_updates: bool = True
    # True while a bracket is ambiguous or a flat virtual lot still has
    # attributed broker orders whose cancellation has not been proven.  Such
    # records stay under supervision because a late parent fill can recreate
    # a position after the first zero-fill snapshot.
    order_watch_only: bool = False


@dataclass
class PreparedOrder:
    label: str
    connection: IBKRConnection
    context: MarketContext
    direction: int
    sizing: SizeResult
    signal_id: str
    order_ref: str
    contract: Any
    state: dict[str, Any]


@dataclass
class SubmittedOrder:
    """One persisted bracket intent after the fast transmit phase."""

    item: PreparedOrder
    managed_trade: ManagedTrade
    trades: tuple[Any, Any, Any] | None
    transmit_error: str | None = None
    entry_cancel_requested: bool = False


@dataclass
class TargetRevisionIntent:
    trade: ManagedTrade
    activation_clock: str
    bar_clock: str
    remaining: int
    new_ema: float
    target: float
    prior_target: float | None = None
    updated_trade: Any | None = None
    transmit_error: str | None = None
    deadline_missed: bool = False


@dataclass
class FinalExitIntent:
    trade: ManagedTrade
    remaining: int
    submitted_trade: Any | None = None
    submit_error: str | None = None


def et_timestamp(entry_date: str, clock: str) -> pd.Timestamp:
    return pd.Timestamp(f"{entry_date} {clock}", tz=NY_TZ)


def aggregate_five_second_minute(
    frame: pd.DataFrame, minute_start: pd.Timestamp
) -> pd.Series:
    """Aggregate one exact completed minute from twelve 5-second bars."""

    start = pd.Timestamp(minute_start)
    if start.tz is None:
        raise ValueError("minute_start must be timezone-aware")
    start = start.tz_convert("UTC")
    expected = pd.date_range(start, periods=12, freq="5s")
    window = frame.reindex(expected)
    if window[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError(f"incomplete real-time minute {start.isoformat()}")
    _validate_realtime_ohlc(window, label=f"real-time minute {start.isoformat()}")
    return pd.Series(
        {
            "open": float(window.iloc[0]["open"]),
            "high": float(window["high"].max()),
            "low": float(window["low"].min()),
            "close": float(window.iloc[-1]["close"]),
            "volume": float(window["volume"].sum()),
        },
        name=start,
    )


def completed_fifteen_minute_close(
    frame: pd.DataFrame, bar_start: pd.Timestamp
) -> float:
    """Return the close of exactly 180 completed five-second bars."""

    start = pd.Timestamp(bar_start)
    if start.tz is None:
        raise ValueError("bar_start must be timezone-aware")
    start = start.tz_convert("UTC")
    expected = pd.date_range(start, periods=180, freq="5s")
    window = frame.reindex(expected)
    if window["close"].isna().any():
        raise ValueError(f"incomplete real-time 15-minute bar {start.isoformat()}")
    _validate_realtime_ohlc(
        window, label=f"real-time 15-minute bar {start.isoformat()}"
    )
    return float(window.iloc[-1]["close"])


def aggregate_trade_minute(
    frame: pd.DataFrame, minute_start: pd.Timestamp
) -> pd.Series:
    """Aggregate one ETF minute from the ordered IBKR Last-trade tape."""

    start = pd.Timestamp(minute_start)
    if start.tz is None:
        raise ValueError("minute_start must be timezone-aware")
    start = start.tz_convert("UTC")
    end = start + pd.Timedelta(minutes=1)
    if not {"timestamp", "price", "size"}.issubset(frame.columns):
        raise ValueError("trade tape lacks timestamp/price/size fields")
    window = frame.loc[
        (frame["timestamp"] >= start) & (frame["timestamp"] < end)
    ]
    if window.empty:
        raise ValueError(f"empty trade minute {start.isoformat()}")
    prices = pd.to_numeric(window["price"], errors="coerce")
    sizes = pd.to_numeric(window["size"], errors="coerce")
    if (
        prices.isna().any()
        or not prices.map(math.isfinite).all()
        or (prices <= 0).any()
        or sizes.isna().any()
        or not sizes.map(math.isfinite).all()
        or (sizes < 0).any()
    ):
        raise ValueError(f"invalid trade minute {start.isoformat()}")
    return pd.Series(
        {
            "open": float(prices.iloc[0]),
            "high": float(prices.max()),
            "low": float(prices.min()),
            "close": float(prices.iloc[-1]),
            "volume": float(sizes.sum()),
        },
        name=start,
    )


def first_trade_at_or_after(
    frame: pd.DataFrame, timestamp: pd.Timestamp
) -> pd.Series:
    """Return the first observed Last tick at or after an exact instant."""

    start = pd.Timestamp(timestamp)
    if start.tz is None:
        raise ValueError("timestamp must be timezone-aware")
    start = start.tz_convert("UTC")
    if not {"timestamp", "price", "size"}.issubset(frame.columns):
        raise ValueError("trade tape lacks timestamp/price/size fields")
    eligible = frame.loc[frame["timestamp"] >= start]
    if eligible.empty:
        raise ValueError(f"no trade at or after {start.isoformat()}")
    return eligible.iloc[0]


def validate_submission_tape(
    frame: pd.DataFrame,
    *,
    entry_date: str,
    direction: int,
    target: float,
    now_et: pd.Timestamp,
) -> tuple[float, str]:
    """Fail closed if the target crossed after the decision tick."""

    now = pd.Timestamp(now_et)
    if now.tz is None:
        raise ValueError("now_et must be timezone-aware")
    now = now.tz_convert(NY_TZ)
    if now > et_timestamp(entry_date, "09:31:20"):
        raise RuntimeError("entry deadline passed before broker submission")
    if frame.empty:
        raise RuntimeError("ETF Last-trade tape is empty before submission")
    latest_stamp = pd.Timestamp(frame.iloc[-1]["timestamp"])
    if latest_stamp.tz is None:
        raise RuntimeError("ETF Last-trade tape timestamps must be timezone-aware")
    latest_stamp = latest_stamp.tz_convert("UTC")
    decision_start = et_timestamp(entry_date, "09:31").tz_convert("UTC")
    now_utc = now.tz_convert("UTC")
    if latest_stamp < decision_start:
        raise RuntimeError("no 09:31 ETF trade is available before submission")
    if latest_stamp > now_utc + pd.Timedelta(seconds=2):
        raise RuntimeError("ETF Last-trade tape is ahead of the local clock")
    if now_utc - latest_stamp > pd.Timedelta(seconds=12):
        raise RuntimeError("ETF Last-trade tape is stale before submission")
    window = frame.loc[frame["timestamp"] >= decision_start]
    if window.empty:
        raise RuntimeError("no decision-window ETF trade is available")
    prices = pd.to_numeric(window["price"], errors="coerce")
    crossed = (direction > 0 and float(prices.max()) >= float(target)) or (
        direction < 0 and float(prices.min()) <= float(target)
    )
    if crossed:
        raise RuntimeError("EMA opportunity crossed before broker submission")
    return float(prices.iloc[-1]), latest_stamp.isoformat()


def _validate_realtime_ohlc(frame: pd.DataFrame, *, label: str) -> None:
    required = ["open", "high", "low", "close"]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"{label} lacks OHLC fields")
    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not math.isfinite(float(values.min())) or not math.isfinite(
        float(values.max())
    ):
        raise ValueError(f"{label} contains non-finite OHLC")
    if (values <= 0).any():
        raise ValueError(f"{label} contains non-positive OHLC")
    invalid = (
        numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)
    ) | (
        numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)
    )
    if invalid.any():
        raise ValueError(f"{label} contains malformed OHLC")


def _seed_from_history(frame: pd.DataFrame, entry_date: str, setup_date: str) -> float:
    if frame.empty:
        raise RuntimeError("IBKR ETF 15-minute history is empty")
    local = frame.copy()
    if not isinstance(local.index, pd.DatetimeIndex) or local.index.tz is None:
        raise RuntimeError("ETF EMA history requires a timezone-aware index")
    local.index = local.index.tz_convert(NY_TZ)
    start = et_timestamp(entry_date, "09:30")
    prior = local.loc[local.index < start].sort_index()
    if prior.index.has_duplicates:
        raise RuntimeError("ETF EMA history contains duplicate 15-minute bars")
    if prior.empty:
        raise RuntimeError("ETF EMA history has no prior RTH bars")
    expected_last = et_timestamp(setup_date, "15:45")
    if prior.index[-1] != expected_last:
        raise RuntimeError(
            f"ETF EMA seed ends at {prior.index[-1]}, "
            f"expected {expected_last}"
        )
    expected = rth_bar_starts(
        prior.index[0].date(), pd.Timestamp(setup_date).date()
    )
    missing = expected.difference(prior.index)
    unexpected = prior.index.difference(expected)
    if len(missing) or len(unexpected) or len(prior) != len(expected):
        raise RuntimeError(
            "ETF EMA history does not match the exact XNYS 15-minute RTH grid "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    closes = pd.to_numeric(prior["close"], errors="coerce")
    if closes.isna().any() or not closes.map(math.isfinite).all() or (closes <= 0).any():
        raise RuntimeError("ETF EMA history contains invalid closes")
    ema = closes.astype(float).ewm(
        span=RULES.ema_span, adjust=False, min_periods=RULES.ema_span
    ).mean()
    if ema.empty or not math.isfinite(float(ema.iloc[-1])):
        raise RuntimeError("ETF EMA20 seed is not ready")
    return float(ema.iloc[-1])


def _feed_endpoint(environment: dict[str, str] | None = None) -> Endpoint:
    import os

    values = os.environ if environment is None else environment
    account = str(values.get("LEGEND_ETF_PRIMARY_ACCOUNT", "")).strip()
    if not account:
        raise RuntimeError(
            "LEGEND_ETF_PRIMARY_ACCOUNT is required for the authoritative ETF feed"
        )
    return Endpoint(
        label="feed",
        host=str(values.get("LEGEND_ETF_FEED_HOST", "127.0.0.1")),
        port=int(values.get("LEGEND_ETF_FEED_PORT", "7496")),
        client_id=int(values.get("LEGEND_ETF_FEED_CLIENT_ID", "154")),
        account=account,
    )


def _wait_until(feed: IBKRConnection, target: pd.Timestamp) -> None:
    while True:
        now = pd.Timestamp.now(tz=NY_TZ)
        remaining = (target - now).total_seconds()
        if remaining <= 0:
            return
        feed.sleep(min(5.0, remaining))


def _wait_for_decision_bars(
    feed: IBKRConnection,
    contexts: list[MarketContext],
    entry_date: str,
) -> None:
    deadline = et_timestamp(entry_date, "09:31:20")
    while pd.Timestamp.now(tz=NY_TZ) <= deadline:
        ready = True
        for context in contexts:
            frame = trade_ticks_to_frame(context.trade_ticks)
            try:
                aggregate_trade_minute(
                    frame, et_timestamp(entry_date, "09:30")
                )
                first_trade_at_or_after(
                    frame, et_timestamp(entry_date, "09:31")
                )
            except ValueError:
                ready = False
                break
        if ready:
            return
        feed.sleep(0.25)
    raise RuntimeError("ETF 09:30/09:31 Last-trade tape missed the entry deadline")


def _order_ids(trades: tuple[Any, ...]) -> dict[str, int]:
    labels = ("parent", "target", "time") if len(trades) == 3 else ("target", "time")
    result: dict[str, int] = {}
    for label, trade in zip(labels, trades):
        result[f"{label}_order_id"] = int(trade.order.orderId)
        perm_id = int(getattr(trade.order, "permId", 0) or 0)
        if perm_id:
            result[f"{label}_perm_id"] = perm_id
    return result


class LegendSession:
    def __init__(
        self,
        *,
        plan_path: Path,
        state_store: StateStore,
        account_labels: list[str],
        live_requested: bool,
        shadow_through_exit: bool = False,
        reconcile_only: bool = False,
        runtime_env_path: Path | None = None,
        run_id: str | None = None,
    ):
        self.plan_path = Path(plan_path)
        self.store = state_store
        self.account_labels = account_labels
        self.live_requested = live_requested
        self.shadow_through_exit = shadow_through_exit
        self.reconcile_only = reconcile_only
        self.runtime_env_path = (
            None if runtime_env_path is None else Path(runtime_env_path)
        )
        self.run_id = str(run_id or f"manual-{os.getpid()}")
        self.lease_path = self.store.state_path.parent / "live_session_lease.json"
        self.lease_started_at = pd.Timestamp.now(tz="UTC").isoformat()
        self.process_started_at = _process_started_at_utc()
        self.entry_date = datetime.now(ZoneInfo(NY_TZ)).date().isoformat()
        self.plan: dict[str, Any] = {}
        self.gate: LiveGate | None = None
        self.endpoints: list[Endpoint] = []
        self.feed: IBKRConnection | None = None
        self.accounts: dict[str, IBKRConnection] = {}
        self.contexts: list[MarketContext] = []
        self.runtime_environment: dict[str, str] | None = None
        self.reservations: ReservationBook | None = None
        self.reservation_failures: dict[tuple[str, str], str] = {}
        self.preentry_nlv: dict[str, float] = {}
        self.preentry_profiles: dict[str, Any] = {}
        self.preentry_contracts: dict[tuple[str, str], Any] = {}
        self.preentry_blocks: dict[tuple[str, str], str] = {}
        self.preentry_short_blocks: dict[tuple[str, str], str] = {}
        self.preentry_short_detail: dict[str, str] = {}
        self.recovery_degraded_reasons: list[str] = []
        self.portfolio_capacities: dict[str, PortfolioCapacity] = {}
        self.portfolio_budget_sha256: str | None = None
        self.correction_audit_completed = False

    def _touch_live_lease(self, status: str) -> None:
        if not self.live_requested:
            return
        if status not in {"active", "finished", "failed"}:
            raise ValueError("invalid Legend live lease status")
        atomic_write_json(
            self.lease_path,
            {
                "protocol": "legend-live-session-lease-v1",
                "entry_date": self.entry_date,
                "run_id": self.run_id,
                "pid": os.getpid(),
                "process_started_at": self.process_started_at,
                "started_at": self.lease_started_at,
                "heartbeat_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "status": status,
                "reconcile_only": self.reconcile_only,
            },
        )

    def _read_live_runtime(self) -> dict[str, str]:
        if self.runtime_env_path is None or not self.runtime_env_path.is_file():
            raise RuntimeError(
                "live execution requires the machine-global Legend runtime.env"
            )
        try:
            from dotenv import dotenv_values
        except ImportError as exc:
            raise RuntimeError("python-dotenv is required for the live gate") from exc
        raw = dotenv_values(self.runtime_env_path)
        values = {str(key): str(value) for key, value in raw.items() if value is not None}
        required = {
            "LEGEND_ETF_LIVE_ENABLED",
            "LEGEND_ETF_LIVE_DATE",
            "LEGEND_ETF_STRATEGY_VERSION",
            "LEGEND_ETF_LIVE_ACCOUNTS",
            "LEGEND_ETF_ALLOW_LONGS",
            "LEGEND_ETF_ALLOW_SHORTS",
            "LEGEND_ETF_PRIMARY_ACCOUNT",
            "LEGEND_ETF_FEED_HOST",
            "LEGEND_ETF_FEED_PORT",
            "LEGEND_ETF_FEED_CLIENT_ID",
            "LEGEND_ETF_RESERVATION_DIR",
            "LEGEND_ETF_GUARD_MANIFEST",
            "LEGEND_ETF_GUARD_MANIFEST_SHA256",
            "LEGEND_ETF_EXECUTOR_ROOT",
            "LEGEND_ETF_PORTFOLIO_BUDGET",
            "LEGEND_ETF_PAPER_PROOF",
            "LEGEND_ETF_PAPER_PROOF_SHA256",
        }
        for label in self.account_labels:
            prefix = f"LEGEND_ETF_{label.upper()}"
            required.update(
                {
                    f"{prefix}_ACCOUNT",
                    f"{prefix}_HOST",
                    f"{prefix}_PORT",
                    f"{prefix}_CLIENT_ID",
                }
            )
        missing = sorted(key for key in required if not values.get(key, "").strip())
        if missing:
            raise RuntimeError(
                "live runtime.env is incomplete: " + ", ".join(missing)
            )
        return values

    def _validate_shared_guard_manifest(
        self, values: dict[str, str]
    ) -> dict[str, Any]:
        manifest_path = Path(values["LEGEND_ETF_GUARD_MANIFEST"])
        expected_hash = values["LEGEND_ETF_GUARD_MANIFEST_SHA256"].strip().lower()
        return validate_guard_manifest(
            manifest_path,
            expected_reservation_dir=Path(values["LEGEND_ETF_RESERVATION_DIR"]),
            expected_runtime_dir=self.store.state_path.parent,
            expected_legend_root=Path(__file__).resolve().parents[1],
            expected_executor_root=Path(values["LEGEND_ETF_EXECUTOR_ROOT"]),
            expected_sha256=expected_hash,
        )

    def _validate_deployment_guard(self, values: dict[str, str]) -> dict[str, Any]:
        manifest = self._validate_shared_guard_manifest(values)
        expected_hash = values["LEGEND_ETF_GUARD_MANIFEST_SHA256"].strip().lower()
        validate_paper_proof(
            Path(values["LEGEND_ETF_PAPER_PROOF"]),
            expected_sha256=values["LEGEND_ETF_PAPER_PROOF_SHA256"],
            expected_manifest_sha256=expected_hash,
        )
        return manifest

    def _reconcile_attested_external_quarantines(
        self, manifest: dict[str, Any]
    ) -> None:
        """Clear only broker-proved terminal external quarantines at startup."""

        config = manifest.get("reservation_config")
        if not isinstance(config, dict) or not str(config.get("path") or "").strip():
            raise RuntimeError("external guard reservation config is unavailable")
        config_path = Path(str(config["path"])).resolve()
        guard = load_attested_external_guard(manifest)
        uncleared: list[str] = []
        for endpoint in self.endpoints:
            connection = self.accounts.get(endpoint.label)
            if connection is None or connection.ib is None:
                raise RuntimeError(
                    f"{endpoint.label}: external quarantine reconciliation lacks "
                    "an account-owned broker connection"
                )
            results = guard.reconcile_all_external_quarantines(
                connection.broker_snapshot_facade(),
                account=endpoint.account,
                config_path=config_path,
            )
            if not isinstance(results, dict) or any(
                not isinstance(symbol, str) or not isinstance(cleared, bool)
                for symbol, cleared in results.items()
            ):
                raise RuntimeError(
                    f"{endpoint.label}: external quarantine reconciliation "
                    "returned an invalid result"
                )
            for symbol, cleared in sorted(results.items()):
                self.store.append_audit(
                    "external_quarantine_reconciliation",
                    account_label=endpoint.label,
                    account=endpoint.account,
                    symbol=symbol,
                    terminal_cleared=cleared,
                )
                if not cleared:
                    uncleared.append(f"{endpoint.label}/{symbol}")
        if uncleared:
            raise RuntimeError(
                "external account-symbol quarantines are not broker-proved terminal: "
                + ", ".join(uncleared)
            )

    def _refresh_portfolio_budget(
        self, values: dict[str, str], endpoints: list[Endpoint]
    ) -> None:
        capacities, digest = validate_portfolio_budget(
            Path(values["LEGEND_ETF_PORTFOLIO_BUDGET"]),
            entry_date=self.entry_date,
            account_ids=[endpoint.account for endpoint in endpoints],
            expected_manifest_sha256=values[
                "LEGEND_ETF_GUARD_MANIFEST_SHA256"
            ],
        )
        if (
            self.portfolio_budget_sha256 is not None
            and digest != self.portfolio_budget_sha256
        ):
            raise RuntimeError(
                "shared portfolio budget changed after preflight"
            )
        self.portfolio_capacities = capacities
        self.portfolio_budget_sha256 = digest

    def _refresh_live_gate(self) -> None:
        if not self.live_requested:
            return
        values = self._read_live_runtime()
        self._validate_deployment_guard(values)
        endpoints = endpoints_from_env(
            self.account_labels, environment=values
        )
        if endpoints != self.endpoints:
            raise RuntimeError("live runtime endpoint identity changed during the session")
        if self.feed is not None and _feed_endpoint(values) != self.feed.endpoint:
            raise RuntimeError("live runtime feed identity changed during the session")
        if self.runtime_environment is not None:
            for key in (
                "LEGEND_ETF_RESERVATION_DIR",
                "LEGEND_ETF_GUARD_MANIFEST",
                "LEGEND_ETF_GUARD_MANIFEST_SHA256",
                "LEGEND_ETF_EXECUTOR_ROOT",
                "LEGEND_ETF_PORTFOLIO_BUDGET",
                "LEGEND_ETF_PAPER_PROOF",
                "LEGEND_ETF_PAPER_PROOF_SHA256",
            ):
                if self.runtime_environment.get(key) != values.get(key):
                    raise RuntimeError(
                        "live reservation configuration changed; restart and re-preflight"
                    )
            risk_suffixes = {
                "LONG_BPS",
                "SHORT_BPS",
                "CLUSTER_BPS",
                "MAX_SHARES_PER_ROOT",
                "MAX_NOTIONAL_PCT",
            }
            for label in self.account_labels:
                prefix = f"LEGEND_ETF_{label.upper()}_"
                for suffix in risk_suffixes:
                    key = prefix + suffix
                    if self.runtime_environment.get(key) != values.get(key):
                        raise RuntimeError(
                            "live sizing configuration changed during the session; "
                            "restart and re-preflight"
                        )
        self._refresh_portfolio_budget(values, endpoints)
        self.gate = validate_live_gate(
            live_requested=True,
            account_ids=[endpoint.account for endpoint in endpoints],
            today=pd.Timestamp(self.entry_date).date(),
            environment=values,
        )
        self.runtime_environment = values

    @staticmethod
    def _broker_mutated(record: dict[str, Any]) -> bool:
        return bool(record.get("submit_intent_at")) or any(
            key.endswith("_order_id") for key in record
        )

    @staticmethod
    def _terminal_proven(record: dict[str, Any]) -> bool:
        quarantine_state_is_valid = (
            not record.get("quarantine_required")
            or (
                record.get("quarantine_active") is False
                and not record.get("quarantine_release_pending")
            )
            or (
                record.get("quarantine_active") is True
                and record.get("quarantine_release_pending") is True
                and record.get("correction_audit_required") is True
            )
        )
        return (
            record.get("state") in {"complete", "complete_emergency", "blocked"}
            and record.get("remaining_owned_shares") == 0
            and record.get("working_orders_cleared") is True
            and bool(record.get("terminal_proved_at"))
            and quarantine_state_is_valid
        )

    def _assert_recovery_scope(self, qualified: list[dict[str, Any]]) -> None:
        """Never silently ignore unresolved broker state on startup."""

        selected_accounts = {endpoint.account for endpoint in self.endpoints}
        qualified_roots = {str(item["root"]) for item in qualified}
        violations: list[str] = []
        for signal_id, record in self.store.load().get("signals", {}).items():
            state = str(record.get("state") or "")
            broker_mutated = self._broker_mutated(record)
            if state in {"dry_run", "skipped"}:
                continue
            if state in {"complete", "complete_emergency"} and not broker_mutated:
                continue
            # Same-day terminal records are revalidated for late execution
            # corrections. A prior-day record may be archived only when its
            # signed-zero and no-working-order proof was durably persisted.
            if (
                record.get("entry_date") != self.entry_date
                and self._terminal_proven(record)
            ):
                continue
            if state == "blocked" and not broker_mutated:
                continue
            reasons: list[str] = []
            if state == "critical_over_exit":
                reasons.append("critical_over_exit_requires_manual_review")
            if record.get("entry_date") != self.entry_date:
                reasons.append("prior_or_wrong_entry_date")
            if record.get("account") not in selected_accounts:
                reasons.append("account_not_selected")
            if record.get("root") not in qualified_roots:
                reasons.append("root_not_in_current_qualified_plan")
            if reasons:
                violations.append(f"{signal_id}({','.join(reasons)})")
        if violations:
            raise RuntimeError(
                "unresolved Legend broker state is outside this run's safe recovery "
                "scope: " + "; ".join(violations)
            )

    def _bootstrap_recovery(self) -> tuple[bool, list[ManagedTrade]]:
        """Recover same-day broker state without depending on fresh signal data."""

        if not self.live_requested:
            return False, []
        raw_state = self.store.recover_interrupted()
        records = raw_state.get("signals", {})
        prior_violations: list[tuple[str, dict[str, Any]]] = []
        manual_violations: list[str] = []
        same_day: list[tuple[str, dict[str, Any]]] = []
        for signal_id, record in records.items():
            if not self._broker_mutated(record):
                continue
            if record.get("state") == "critical_over_exit":
                detail = f"{signal_id}(critical_over_exit_requires_manual_review)"
                manual_violations.append(detail)
                if detail not in self.recovery_degraded_reasons:
                    self.recovery_degraded_reasons.append(detail)
                continue
            if record.get("entry_date") == self.entry_date:
                same_day.append((signal_id, record))
            elif not self._terminal_proven(record):
                prior_violations.append((signal_id, record))
        # A stale prior-day lot is never auto-managed against today's broker
        # state: it is manualized and quarantined.  It also must not prevent
        # an unrelated same-day lot from reconnecting and retaining its exits.
        for signal_id, record in prior_violations:
            reason = "unproven prior-day broker state requires manual review"
            record["state"] = "critical_prior_day_recovery"
            record["critical_reason"] = reason
            record["quarantine_required"] = True
            record["quarantine_release_pending"] = True
            record["manual_review_required"] = True
            record["manualized_at"] = pd.Timestamp.now(tz="UTC").isoformat()
            reservation_detail = "durable reservation identity unavailable"
            reservation_active = False
            reservation_dir = str(record.get("reservation_dir") or "").strip()
            account = str(record.get("account") or "").strip()
            etf = str(record.get("etf") or "").strip().upper()
            if reservation_dir and account and etf:
                prior_book: ReservationBook | None = None
                try:
                    prior_book = ReservationBook(
                        Path(reservation_dir).expanduser().resolve()
                    )
                    acquired, reservation_detail = prior_book.try_acquire(
                        account, etf, owner_token=signal_id
                    )
                    if acquired:
                        prior_book.activate_quarantine(
                            account,
                            etf,
                            owner_token=signal_id,
                            payload={
                                "signal_id": signal_id,
                                "entry_date": str(record.get("entry_date") or ""),
                                "manualized_on": self.entry_date,
                                "reason": reason,
                            },
                        )
                        reservation_active = True
                except Exception as exc:  # noqa: BLE001 - persist manual boundary
                    reservation_detail = f"{type(exc).__name__}: {exc}"
                finally:
                    if prior_book is not None:
                        prior_book.close()
            record["quarantine_active"] = reservation_active
            record["prior_day_quarantine_detail"] = reservation_detail
            self.store.put(signal_id, record)
            detail = f"{signal_id}(unproven_prior_day_manualized)"
            if detail not in self.recovery_degraded_reasons:
                self.recovery_degraded_reasons.append(detail)
            self.store.append_audit(
                "prior_day_recovery_manualized",
                signal_id=signal_id,
                quarantine_active=reservation_active,
                detail=reservation_detail,
            )
        if not same_day:
            return bool(manual_violations or prior_violations), []

        durable_by_label: dict[str, Endpoint] = {}
        reservation_dirs: set[str] = set()
        for signal_id, record in same_day:
            label = str(record.get("account_label") or "")
            raw_endpoint = record.get("execution_endpoint")
            reservation_dir = str(record.get("reservation_dir") or "").strip()
            if label not in self.account_labels:
                raise RuntimeError(
                    f"recovery account label is not selected: {label or signal_id}"
                )
            if not isinstance(raw_endpoint, dict) or not reservation_dir:
                raise RuntimeError(
                    f"recovery record {signal_id} lacks durable endpoint/reservation identity"
                )
            endpoint = Endpoint(
                label=str(raw_endpoint.get("label") or ""),
                host=str(raw_endpoint.get("host") or ""),
                port=int(raw_endpoint.get("port") or 0),
                client_id=int(raw_endpoint.get("client_id") or 0),
                account=str(raw_endpoint.get("account") or ""),
            )
            if (
                endpoint.label != label
                or not endpoint.host
                or endpoint.port <= 0
                or endpoint.client_id <= 0
                or endpoint.account != record.get("account")
            ):
                raise RuntimeError(f"invalid durable recovery endpoint for {signal_id}")
            prior = durable_by_label.get(label)
            if prior is not None and prior != endpoint:
                raise RuntimeError(f"conflicting recovery endpoints for {label}")
            durable_by_label[label] = endpoint
            reservation_dirs.add(str(Path(reservation_dir).expanduser().resolve()))
        if len(reservation_dirs) != 1:
            raise RuntimeError("recovery records disagree on the reservation directory")
        self.endpoints = [durable_by_label[label] for label in self.account_labels if label in durable_by_label]
        self.gate = LiveGate(
            True,
            False,
            False,
            frozenset(endpoint.account.upper() for endpoint in self.endpoints),
        )
        # Entry/build attestation is deliberately not an exit dependency. Any
        # mismatch is loud and makes the task fail after broker reconciliation,
        # but already-open exact lots are still protected/flattened.
        try:
            runtime_values = self._read_live_runtime()
            configured = endpoints_from_env(
                list(durable_by_label), environment=runtime_values
            )
            if configured != [durable_by_label[label] for label in durable_by_label]:
                raise RuntimeError("runtime endpoints differ from durable recovery state")
            validate_recovery_gate(
                account_ids=[endpoint.account for endpoint in self.endpoints],
                today=pd.Timestamp(self.entry_date).date(),
                environment=runtime_values,
            )
            self._validate_deployment_guard(runtime_values)
            self.runtime_environment = runtime_values
        except Exception as exc:  # noqa: BLE001 - exits remain authorized
            self.runtime_environment = None
            self.recovery_degraded_reasons.append(
                f"recovery runtime attestation failed: {type(exc).__name__}: {exc}"
            )
        def mark_unavailable(
            signal_id: str,
            record: dict[str, Any],
            *,
            state_name: str,
            reason: str,
        ) -> None:
            record["state"] = state_name
            record["critical_reason"] = reason
            record["recovery_unavailable_at"] = pd.Timestamp.now(
                tz="UTC"
            ).isoformat()
            record["recovery_unavailable_reason"] = reason
            self.store.put(signal_id, record)
            detail = f"{signal_id}: {reason}"
            if detail not in self.recovery_degraded_reasons:
                self.recovery_degraded_reasons.append(detail)
            self.store.append_audit(
                "recovery_lot_unavailable",
                signal_id=signal_id,
                state=state_name,
                detail=reason,
            )

        for endpoint in self.endpoints:
            connection = self.accounts.get(endpoint.label)
            try:
                if connection is None:
                    connection = IBKRConnection(endpoint, live=True)
                    connection.connect()
                elif connection.endpoint != endpoint:
                    raise RuntimeError(
                        "startup audit connection does not match durable endpoint"
                    )
                elif not connection.is_connected():
                    # A timed-out audit request deliberately poisons its socket.
                    # Recreate the exact durable endpoint before declaring its
                    # same-day lots unavailable.
                    try:
                        connection.reconnect()
                    except Exception:  # noqa: BLE001 - recreate exact poisoned socket
                        connection = IBKRConnection(endpoint, live=True)
                        connection.connect()
            except Exception as exc:  # noqa: BLE001 - isolate endpoint recovery
                reason = (
                    f"could not connect durable {endpoint.label} endpoint: "
                    f"{type(exc).__name__}: {exc}"
                )
                for signal_id, record in same_day:
                    if record.get("account_label") == endpoint.label:
                        mark_unavailable(
                            signal_id,
                            record,
                            state_name="critical_connection",
                            reason=reason,
                        )
                continue
            self.accounts[endpoint.label] = connection

        if self.reservations is None:
            self.reservations = ReservationBook(
                Path(next(iter(reservation_dirs)))
            )
        context_by_root: dict[str, MarketContext] = {}
        for signal_id, record in same_day:
            label = str(record.get("account_label") or "")
            connection = self.accounts.get(label)
            if connection is None:
                # The exact endpoint failure was already persisted above. A
                # healthy account/root must still be recovered and managed.
                continue
            required = {
                "root",
                "etf",
                "setup_date",
                "direction",
                "initial_ema",
                "atr14",
                "account",
                "account_label",
                "execution_endpoint",
                "execution_con_id",
                "reservation_dir",
            }
            missing = sorted(key for key in required if record.get(key) is None)
            if missing:
                mark_unavailable(
                    signal_id,
                    record,
                    state_name="critical_recovery_context",
                    reason=(
                        f"recovery record lacks {', '.join(missing)}"
                    ),
                )
                continue
            expected_endpoint = {
                "label": connection.endpoint.label,
                "host": connection.endpoint.host,
                "port": connection.endpoint.port,
                "client_id": connection.endpoint.client_id,
                "account": connection.endpoint.account,
            }
            if record["execution_endpoint"] != expected_endpoint:
                mark_unavailable(
                    signal_id,
                    record,
                    state_name="critical_recovery_context",
                    reason="recovery endpoint identity changed",
                )
                continue
            if self.recovery_degraded_reasons:
                record["degraded_execution"] = True
                record.setdefault("degraded_reasons", []).extend(
                    reason
                    for reason in self.recovery_degraded_reasons
                    if reason not in record.get("degraded_reasons", [])
                )
                self.store.put(signal_id, record)
            acquired, detail = self.reservations.try_acquire(
                str(record["account"]),
                str(record["etf"]),
                owner_token=signal_id,
            )
            self.store.append_audit(
                "recovery_symbol_reservation",
                signal_id=signal_id,
                acquired=acquired,
                detail=detail,
            )
            if not acquired:
                mark_unavailable(
                    signal_id,
                    record,
                    state_name="critical_isolation",
                    reason=detail,
                )
                continue
            else:
                # Reassert the durable quarantine before any broker
                # reconciliation. This repairs an inactive/missing marker
                # left by an older build or a crash at a state boundary.
                self.reservations.activate_quarantine(
                    str(record["account"]),
                    str(record["etf"]),
                    owner_token=signal_id,
                    payload={
                        "signal_id": signal_id,
                        "entry_date": self.entry_date,
                        "order_ref": str(record.get("order_ref") or ""),
                        "con_id": int(record["execution_con_id"]),
                        "recovery_reasserted": True,
                    },
                )
                record["quarantine_required"] = True
                record["quarantine_active"] = True
                record["quarantine_release_pending"] = True
                record["quarantine_reasserted_at"] = pd.Timestamp.now(
                    tz="UTC"
                ).isoformat()
                self.store.put(signal_id, record)
            root = str(record["root"])
            if root not in context_by_root:
                try:
                    contract = connection.stock(str(record["etf"]))
                except Exception as exc:  # noqa: BLE001 - isolate lot context
                    mark_unavailable(
                        signal_id,
                        record,
                        state_name="critical_recovery_context",
                        reason=(
                            "could not qualify durable ETF contract: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )
                    continue
                if int(contract.conId or 0) != int(record["execution_con_id"]):
                    mark_unavailable(
                        signal_id,
                        record,
                        state_name="critical_recovery_context",
                        reason="recovery ETF contract identity changed",
                    )
                    continue
                context_by_root[root] = MarketContext(
                    root=root,
                    etf=str(record["etf"]),
                    setup_date=str(record["setup_date"]),
                    contract=contract,
                    initial_ema=float(record["initial_ema"]),
                    atr14=float(record["atr14"]),
                    ex_dividend=False,
                    dividend_detail="recovery uses durable entry decision",
                    initial_target=float(
                        record.get("initial_target")
                        or round_limit(
                            float(record["initial_ema"]),
                            int(record["direction"]),
                        )
                    ),
                )
            record.pop("recovery_unavailable_at", None)
            record.pop("recovery_unavailable_reason", None)
            self.store.put(signal_id, record)
        self.contexts = list(context_by_root.values())
        return True, self._reconcile_prior_records()

    def audit_terminal_corrections(self) -> dict[str, Any]:
        """Run the read-only all-ETF audit without a calendar or signal plan."""

        if not self.live_requested:
            return {"ok": True, "live": False, "audited": False}
        if self.correction_audit_completed:
            return {"ok": True, "live": True, "audited": True}
        values = self._read_live_runtime()
        # The external reconciler is executable code. Validate its exact source
        # tree, config, and Python environment before importing it. The separate
        # paper proof remains an entry gate, not a read-only correction-audit gate.
        manifest = self._validate_shared_guard_manifest(values)
        endpoints = endpoints_from_env(
            self.account_labels, environment=values
        )
        gate = validate_correction_audit_gate(
            account_ids=[endpoint.account for endpoint in endpoints],
            environment=values,
        )
        if self.endpoints and self.endpoints != endpoints:
            raise RuntimeError("correction-audit endpoint identity changed")
        self.runtime_environment = values
        self.endpoints = endpoints
        self.gate = gate
        connection_failures: list[str] = []
        for endpoint in endpoints:
            created_connection: IBKRConnection | None = None
            try:
                existing = self.accounts.get(endpoint.label)
                if existing is not None:
                    if existing.endpoint != endpoint or not existing.is_connected():
                        raise RuntimeError(
                            "correction-audit connection changed"
                        )
                    continue
                created_connection = IBKRConnection(endpoint, live=True)
                created_connection.connect()
                created_connection.assert_server_clock()
                self.accounts[endpoint.label] = created_connection
            except Exception as exc:  # noqa: BLE001 - audit healthy peers
                if created_connection is not None:
                    try:
                        created_connection.disconnect()
                    except Exception:  # noqa: BLE001,S110 - best-effort socket cleanup
                        pass
                connection_failures.append(
                    f"{endpoint.label}: {type(exc).__name__}: {exc}"
                )
        audit_error: Exception | None = None
        try:
            self._reconcile_attested_external_quarantines(manifest)
            self._audit_recent_terminal_corrections()
        except Exception as exc:  # noqa: BLE001 - aggregate endpoint failures
            audit_error = exc
        if connection_failures or audit_error is not None:
            details = list(connection_failures)
            if audit_error is not None:
                details.append(str(audit_error))
            raise RuntimeError(
                "Legend correction audit was incomplete: " + "; ".join(details)
            ) from audit_error
        self.correction_audit_completed = True
        return {
            "ok": True,
            "live": True,
            "audited": True,
            "entry_date": self.entry_date,
            "accounts": self.account_labels,
            "symbols": [market.etf for market in MARKETS],
        }

    def preflight(self, *, connect: bool = True) -> dict[str, Any]:
        if self.live_requested and connect:
            # This must precede calendar/plan/build/entry gates. A stale or
            # missing signal plan and a disabled entry switch cannot hide a
            # prior-session broker correction.
            self.audit_terminal_corrections()
        require_full_entry_session(self.entry_date)
        plan = read_json(self.plan_path)
        if plan is None:
            raise FileNotFoundError(f"signal plan not found: {self.plan_path}")
        if not isinstance(plan, dict):
            raise TypeError(f"signal plan is not a JSON object: {self.plan_path}")
        validate_plan(plan, entry_date=self.entry_date)
        data_as_of = pd.Timestamp(plan.get("data_as_of"))
        if data_as_of.tz is None:
            raise RuntimeError("signal plan data_as_of must be timezone-aware")
        data_as_of_et = data_as_of.tz_convert(NY_TZ)
        earliest = et_timestamp(self.entry_date, "08:30")
        latest = et_timestamp(self.entry_date, "09:25")
        native_etf = plan.get("dataset") == "IBKR"
        if native_etf and pd.Timestamp(plan["created_at"]) > pd.Timestamp.now(tz="UTC"):
            raise RuntimeError("ETF plan creation timestamp is in the future")
        if self.live_requested and not native_etf:
            raise RuntimeError("This release only permits SPY/QQQ ETF-native live plans")
        if not native_etf and not (earliest <= data_as_of_et <= latest):
            raise RuntimeError(
                "signal plan must use a same-day futures contract probe between "
                "08:30 and 09:25 New York time"
            )
        self.plan = plan
        runtime_values = (
            self.runtime_environment
            if self.live_requested and self.runtime_environment is not None
            else (self._read_live_runtime() if self.live_requested else None)
        )
        endpoints = endpoints_from_env(
            self.account_labels, environment=runtime_values
        )
        if self.endpoints and self.endpoints != endpoints:
            raise RuntimeError("preflight endpoint identity differs from correction audit")
        self.runtime_environment = runtime_values
        self.endpoints = endpoints
        if self.live_requested:
            assert self.runtime_environment is not None
            self._validate_deployment_guard(self.runtime_environment)
            self._refresh_portfolio_budget(
                self.runtime_environment, self.endpoints
            )
        self.gate = validate_live_gate(
            live_requested=self.live_requested,
            account_ids=[endpoint.account for endpoint in self.endpoints],
            today=pd.Timestamp(self.entry_date).date(),
            environment=self.runtime_environment,
        )
        qualified = [item for item in plan["markets"] if item.get("qualifies")]
        self._assert_recovery_scope(qualified)
        if connect:
            if not self.live_requested:
                for endpoint in self.endpoints:
                    connection = IBKRConnection(endpoint, live=False)
                    connection.connect()
                    connection.assert_server_clock()
                    self.accounts[endpoint.label] = connection
            if qualified:
                self.feed = IBKRConnection(
                    _feed_endpoint(self.runtime_environment), live=False
                )
                self.feed.connect()
                self.feed.assert_server_clock()
        if not qualified:
            return {
                "ok": True,
                "live": self.gate.live,
                "entry_date": self.entry_date,
                "qualified": [],
                "detail": "no qualified setups; all ETF symbols audited",
            }
        return {
            "ok": True,
            "live": self.gate.live,
            "entry_date": self.entry_date,
            "qualified": [item["root"] for item in qualified],
            "accounts": self.account_labels,
            "strategy_version": STRATEGY_VERSION,
            "plan_hash": plan["plan_hash"],
        }

    def _audit_recent_terminal_corrections(self) -> None:
        """Block and re-quarantine any post-terminal ETF broker residue.

        IBKR execution corrections can arrive after the intraday quarantine is
        released. Every live startup therefore checks all three ETFs in every
        selected account, even when today's futures plan has no candidate.
        This audit never flattens a netted symbol because a residual might be
        manual or owned by another strategy.
        """

        if not (self.gate and self.gate.live):
            return
        if self.runtime_environment is None:
            raise RuntimeError("terminal-correction audit lacks live runtime")
        if self.reservations is None:
            self.reservations = ReservationBook(
                Path(self.runtime_environment["LEGEND_ETF_RESERVATION_DIR"])
            )
        records = self.store.load().get("signals", {})
        today = pd.Timestamp(self.entry_date)
        violations: list[str] = []
        for endpoint in self.endpoints:
            connection = self.accounts.get(endpoint.label)
            if connection is None:
                violations.append(
                    f"{endpoint.label}: execution endpoint was unreachable"
                )
                continue
            for market in MARKETS:
                same_day_unresolved = any(
                    record.get("account") == endpoint.account
                    and str(record.get("etf") or "").upper() == market.etf
                    and record.get("entry_date") == self.entry_date
                    and self._broker_mutated(record)
                    and not self._terminal_proven(record)
                    for record in records.values()
                )
                if same_day_unresolved:
                    # Bootstrap recovery owns this symbol. Auditing it as
                    # "clear" first would strand a legitimate open lot.
                    continue
                try:
                    contract = connection.stock(market.etf)
                except Exception as exc:  # noqa: BLE001 - audit later symbols
                    violations.append(
                        f"{endpoint.label}/{market.etf}: qualification failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                recent: list[tuple[pd.Timestamp, str, dict[str, Any]]] = []
                for signal_id, record in records.items():
                    if (
                        record.get("account") != endpoint.account
                        or str(record.get("etf") or "").upper() != market.etf
                        or not self._terminal_proven(record)
                    ):
                        continue
                    try:
                        entry_day = pd.Timestamp(record["entry_date"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    age = (today - entry_day).days
                    if 0 <= age <= 7:
                        recent.append((entry_day, signal_id, record))
                prior_terminal = (
                    max(recent, key=lambda item: item[0]) if recent else None
                )
                reservation_acquired = False
                reservation_detail = "no retained Legend terminal quarantine"
                prior_age = None
                if prior_terminal is not None:
                    prior_age = (today - prior_terminal[0]).days
                    _, prior_signal_id, _ = prior_terminal
                    reservation_acquired, reservation_detail = (
                        self.reservations.try_acquire(
                            endpoint.account,
                            market.etf,
                            owner_token=prior_signal_id,
                        )
                    )
                    if not reservation_acquired:
                        violations.append(
                            f"{endpoint.label}/{market.etf}: retained correction "
                            f"quarantine unavailable: {reservation_detail}"
                        )
                        continue
                try:
                    connection.assert_symbol_clear(contract)
                except Exception as exc:  # noqa: BLE001 - fail-closed audit
                    if prior_terminal is not None:
                        _, signal_id, record = prior_terminal
                        if reservation_acquired:
                            self.reservations.activate_quarantine(
                                endpoint.account,
                                market.etf,
                                owner_token=signal_id,
                                payload={
                                    "signal_id": signal_id,
                                    "entry_date": str(record.get("entry_date") or ""),
                                    "correction_audit_date": self.entry_date,
                                    "reason": str(exc),
                                },
                            )
                        record["state"] = "critical_terminal_correction"
                        record["critical_reason"] = (
                            "post-terminal account-symbol residue detected; "
                            f"automatic flatten prohibited: {exc}"
                        )
                        record["quarantine_required"] = True
                        record["quarantine_active"] = reservation_acquired
                        record["quarantine_release_pending"] = True
                        record["correction_audit_at"] = pd.Timestamp.now(
                            tz="UTC"
                        ).isoformat()
                        record["correction_quarantine_detail"] = reservation_detail
                        self.store.put(signal_id, record)
                    violations.append(
                        f"{endpoint.label}/{market.etf}: {type(exc).__name__}: {exc}"
                    )
                    continue
                if prior_terminal is not None and prior_age is not None and prior_age >= 1:
                    _, signal_id, record = prior_terminal
                    self.reservations.deactivate_quarantine(
                        endpoint.account,
                        market.etf,
                        owner_token=signal_id,
                    )
                    record["quarantine_active"] = False
                    record["quarantine_release_pending"] = False
                    record["correction_audit_required"] = False
                    record["correction_audit_cleared_at"] = pd.Timestamp.now(
                        tz="UTC"
                    ).isoformat()
                    self.store.put(signal_id, record)
        if violations:
            raise RuntimeError(
                "Legend ETF terminal-correction/account-symbol audit failed: "
                + "; ".join(violations)
            )

    def _acquire_symbol_reservations(self) -> None:
        """Reserve every potentially traded account/ETF before the open."""

        if not (self.gate and self.gate.live):
            return
        assert self.runtime_environment is not None
        if self.reservations is None:
            self.reservations = ReservationBook(
                Path(self.runtime_environment["LEGEND_ETF_RESERVATION_DIR"])
            )
        by_label = {endpoint.label: endpoint for endpoint in self.endpoints}
        for label in self.account_labels:
            endpoint = by_label[label]
            for context in self.contexts:
                acquired, detail = self.reservations.try_acquire(
                    endpoint.account, context.etf
                )
                if not acquired:
                    self.reservation_failures[(label, context.root)] = detail
                self.store.append_audit(
                    "symbol_reservation",
                    account_label=label,
                    account=endpoint.account,
                    root=context.root,
                    etf=context.etf,
                    acquired=acquired,
                    detail=detail,
                )

    def _preflight_accounts_before_decision(self) -> None:
        """Finish all slow account/borrow checks before the 09:31 decision."""

        assert self.feed is not None
        max_short_by_root: dict[str, int] = {context.root: 0 for context in self.contexts}
        per_account_short: dict[tuple[str, str], int] = {}
        for label in self.account_labels:
            connection = self.accounts[label]
            baseline = PRIMARY_RISK if label == "primary" else PA_RISK
            profile = risk_profile_from_env(
                label, baseline, environment=self.runtime_environment
            )
            nlv = connection.nlv()
            self.preentry_nlv[label] = nlv
            self.preentry_profiles[label] = profile
            for context in self.contexts:
                key = (label, context.root)
                if key in self.reservation_failures:
                    self.preentry_blocks[key] = self.reservation_failures[key]
                    continue
                try:
                    contract = connection.stock(context.etf)
                    connection.assert_symbol_clear(contract)
                except Exception as exc:  # noqa: BLE001 - block only this root
                    self.preentry_blocks[key] = str(exc)
                    continue
                self.preentry_contracts[key] = contract
                # This is an upper bound on actual short shares: cluster and
                # notional caps can only reduce it after the 09:31 price.
                risk_usd = nlv * profile.short_bps / 10_000.0
                stress_per_share = 1.25 * float(context.atr14)
                shares = min(
                    profile.max_shares_per_root,
                    max(0, math.floor(risk_usd / stress_per_share)),
                )
                per_account_short[key] = shares
                max_short_by_root[context.root] += shares

        # One bounded borrow probe per ETF, not once per account.
        for context in self.contexts:
            maximum = max_short_by_root[context.root]
            if maximum <= 0:
                continue
            try:
                shortable, detail = self.feed.shortability(
                    context.contract, maximum
                )
            except Exception as exc:  # noqa: BLE001 - block shorts for this root
                shortable, detail = False, str(exc)
            self.preentry_short_detail[context.root] = detail
            if not shortable:
                for label in self.account_labels:
                    self.preentry_short_blocks[(label, context.root)] = detail

        # Margin probes are account-specific but run before direction is known.
        # A later short uses no more shares than the probed upper bound.
        if self.gate and self.gate.live:
            for label in self.account_labels:
                connection = self.accounts[label]
                for context in self.contexts:
                    key = (label, context.root)
                    shares = per_account_short.get(key, 0)
                    contract = self.preentry_contracts.get(key)
                    if (
                        shares <= 0
                        or contract is None
                        or key in self.preentry_short_blocks
                    ):
                        continue
                    try:
                        connection.what_if_short(contract, shares)
                    except Exception as exc:  # noqa: BLE001 - block this short only
                        self.preentry_short_blocks[key] = str(exc)

    def _load_market_contexts(self) -> None:
        assert self.feed is not None
        for item in self.plan["markets"]:
            if not item.get("qualifies"):
                continue
            contract = self.feed.stock(item["etf"])
            history15 = self.feed.historical_bars(
                contract, duration="21 D" if self.plan.get("dataset") == "IBKR" else "20 D",
                bar_size="15 mins", use_rth=True
            )
            if self.plan.get("dataset") == "IBKR":
                from .etf_source import evaluate_etf_setup

                fresh = evaluate_etf_setup(history15, entry_date=self.entry_date)
                if (not fresh["qualifies"]
                        or fresh["history_sha256"] != item["history_sha256"]
                        or fresh["initial_ema"] != item["initial_ema"]):
                    raise RuntimeError("ETF history changed after signal preparation; prepare a new plan")
                initial_ema = fresh["initial_ema"]
            else:
                initial_ema = _seed_from_history(
                    history15, self.entry_date, item["setup_date"]
                )
            daily = self.feed.historical_bars(
                contract, duration="2 Y", bar_size="1 day", use_rth=True
            )
            atr14 = prior_wilder_atr14(
                daily,
                as_of_date=self.entry_date,
                expected_last_session=item["setup_date"],
            )
            ex_dividend, dividend_detail = self.feed.ex_dividend_status(
                contract, self.entry_date
            )
            self.contexts.append(
                MarketContext(
                    root=item["root"],
                    etf=item["etf"],
                    setup_date=item["setup_date"],
                    contract=contract,
                    initial_ema=initial_ema,
                    atr14=atr14,
                    ex_dividend=ex_dividend,
                    dividend_detail=dividend_detail,
                )
            )

    def _existing_record(self, account: str, context: MarketContext) -> dict[str, Any] | None:
        state = self.store.load()
        for record in state.get("signals", {}).values():
            if (
                record.get("account") == account
                and record.get("root") == context.root
                and record.get("entry_date") == self.entry_date
            ):
                return record
        return None

    def _decide(self) -> list[MarketContext]:
        eligible: list[MarketContext] = []
        for context in self.contexts:
            frame = trade_ticks_to_frame(context.trade_ticks)
            row0930 = aggregate_trade_minute(
                frame, et_timestamp(self.entry_date, "09:30")
            )
            decision_tick = first_trade_at_or_after(
                frame, et_timestamp(self.entry_date, "09:31")
            )
            decision_price = float(decision_tick["price"])
            decision = entry_decision(
                row0930,
                initial_ema=context.initial_ema,
                decision_price=decision_price,
                ex_dividend=context.ex_dividend,
            )
            self.store.append_audit(
                "market_decision",
                root=context.root,
                etf=context.etf,
                result=asdict(decision),
                dividend_detail=context.dividend_detail,
            )
            if decision.eligible:
                context.direction = decision.direction
                context.initial_target = decision.initial_target
                context.decision_price = decision_price
                context.decision_observed_at = pd.Timestamp(
                    decision_tick["timestamp"]
                ).isoformat()
                eligible.append(context)
        return eligible

    def _expected_con_id(self, reference: str) -> int:
        symbol = str(reference).split("|", 1)[0].upper()
        matches = [
            context for context in self.contexts if context.etf.upper() == symbol
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"could not resolve one ETF contract for orderRef {reference!r}"
            )
        con_id = int(getattr(matches[0].contract, "conId", 0) or 0)
        if con_id <= 0:
            raise RuntimeError(f"ETF contract {symbol} has no durable conId")
        return con_id

    def _working_orders(
        self, connection: IBKRConnection, reference: str
    ) -> list[Any]:
        expected_con_id = self._expected_con_id(reference)
        orders = [
            trade
            for trade in connection.attributed_orders(
                reference, expected_con_id=expected_con_id
            )
            if str(trade.orderStatus.status or "") not in DEAD_STATUSES
        ]
        foreign = [
            trade
            for trade in orders
            if int(getattr(trade.order, "clientId", -1))
            != connection.endpoint.client_id
        ]
        if foreign:
            raise RuntimeError(
                f"{connection.endpoint.label}: Legend orderRef is owned by a "
                "different IBKR client ID"
            )
        return orders

    def _proven_exit_pair(
        self,
        *,
        connection: IBKRConnection,
        reference: str,
        direction: int,
        remaining: int,
        expected_target: float | None = None,
    ) -> tuple[Any, Any] | None:
        """Return the exact broker-echoed target/time pair or fail closed."""

        working = self._working_orders(connection, reference)
        if len(working) != 2 or any(
            str(trade.orderStatus.status or "")
            not in {"PreSubmitted", "Submitted"}
            for trade in working
        ):
            return None
        targets = [
            trade
            for trade in working
            if str(trade.order.orderType or "").upper() == "LMT"
        ]
        timed = [
            trade
            for trade in working
            if str(trade.order.orderType or "").upper() == "MKT"
            and str(trade.order.goodAfterTime or "")
            == f"{self.entry_date.replace('-', '')} 10:30:00 US/Eastern"
        ]
        if len(targets) != 1 or len(timed) != 1:
            return None
        target, time_exit = targets[0], timed[0]
        expected_action = "SELL" if direction > 0 else "BUY"
        orders = [target.order, time_exit.order]
        statuses = [target.orderStatus, time_exit.orderStatus]
        oca_groups = {str(order.ocaGroup or "") for order in orders}
        parent_ids = {int(order.parentId or 0) for order in orders}
        if (
            any(str(order.account or "") != connection.endpoint.account for order in orders)
            or any(str(order.orderRef or "") != reference for order in orders)
            or any(str(order.action or "").upper() != expected_action for order in orders)
            or any(int(order.ocaType or 0) != 2 for order in orders)
            or any(str(getattr(order, "tif", "") or "").upper() != "DAY" for order in orders)
            or any(bool(getattr(order, "outsideRth", True)) for order in orders)
            or bool(str(getattr(target.order, "goodAfterTime", "") or "").strip())
            or not bool(getattr(time_exit.order, "transmit", False))
            or round(float(getattr(target.order, "totalQuantity", 0) or 0))
            < remaining
            or round(float(getattr(time_exit.order, "totalQuantity", 0) or 0))
            < remaining
            or len(oca_groups) != 1
            or "" in oca_groups
            or len(parent_ids) != 1
            or any(int(getattr(order, "permId", 0) or 0) <= 0 for order in orders)
            or any(
                round(float(getattr(status, "remaining", 0) or 0)) != remaining
                for status in statuses
            )
            or not math.isfinite(float(target.order.lmtPrice))
            or float(target.order.lmtPrice) <= 0
            or (
                expected_target is not None
                and not math.isclose(
                    float(target.order.lmtPrice),
                    float(expected_target),
                    rel_tol=0.0,
                    abs_tol=1e-7,
                )
            )
        ):
            return None
        return target, time_exit

    def _proven_exit_pair_with_reconnect(
        self,
        *,
        connection: IBKRConnection,
        reference: str,
        direction: int,
        remaining: int,
        expected_target: float | None = None,
    ) -> tuple[Any, Any] | None:
        """Retry one poisoned open-order snapshot on the exact endpoint."""

        try:
            return self._proven_exit_pair(
                connection=connection,
                reference=reference,
                direction=direction,
                remaining=remaining,
                expected_target=expected_target,
            )
        except Exception:  # noqa: BLE001 - reconnect any poisoned broker snapshot
            self._restore_connection(connection)
            return self._proven_exit_pair(
                connection=connection,
                reference=reference,
                direction=direction,
                remaining=remaining,
                expected_target=expected_target,
            )

    def _restore_connection(self, connection: IBKRConnection) -> None:
        errors: list[str] = []
        for _ in range(2):
            try:
                connection.reconnect()
                return
            except Exception as exc:  # noqa: BLE001 - bounded broker reconnect boundary
                errors.append(type(exc).__name__)
                time.sleep(1.0)
        raise RuntimeError(
            "could not restore the exact IBKR execution connection for "
            f"reconciliation ({','.join(errors)})"
        )

    def _attributed_fills(
        self, connection: IBKRConnection, reference: str
    ) -> list[Any]:
        expected_con_id = self._expected_con_id(reference)
        try:
            return connection.attributed_fills(
                reference, expected_con_id=expected_con_id
            )
        except Exception:  # noqa: BLE001 - reconnect on any IB transport failure
            self._restore_connection(connection)
            return connection.attributed_fills(
                reference, expected_con_id=expected_con_id
            )

    def _clear_reference_orders(
        self, connection: IBKRConnection, reference: str
    ) -> bool:
        for _ in range(3):
            try:
                working = self._working_orders(connection, reference)
            except Exception:
                if not connection.is_connected():
                    self._restore_connection(connection)
                    continue
                raise
            if not working:
                return True
            try:
                for item in working:
                    connection.cancel_owned_order(item)
            except Exception:
                if not connection.is_connected():
                    self._restore_connection(connection)
                    continue
                raise
            connection.sleep(0.4)
        return not self._working_orders(connection, reference)

    @staticmethod
    def _assert_physical_isolation(
        *,
        connection: IBKRConnection,
        context: MarketContext,
        reference: str,
        direction: int,
        remaining: int,
    ) -> None:
        """Apply the live adapter's account-position invariant.

        Lightweight unit fakes may omit the adapter method; every production
        connection is an ``IBKRConnection`` and implements it.
        """

        checker = getattr(connection, "assert_symbol_isolated", None)
        if checker is None:
            return
        checker(
            context.contract,
            order_ref=reference,
            direction=direction,
            virtual_quantity=remaining,
        )

    def _enforce_physical_isolation(
        self,
        *,
        signal_id: str,
        connection: IBKRConnection,
        context: MarketContext,
        reference: str,
        direction: int,
        remaining: int,
        state: dict[str, Any],
    ) -> int:
        """Retry snapshots, refresh fills, and prevent a reversing Legend exit."""

        current_remaining = int(remaining)
        mismatches: list[str] = []
        last_mismatch: PhysicalIsolationMismatch | None = None
        transport_errors: list[str] = []
        transport_failures = 0
        fill_refreshes = 0
        while len(mismatches) < 2 and transport_failures < 4 and fill_refreshes < 8:
            try:
                self._assert_physical_isolation(
                    connection=connection,
                    context=context,
                    reference=reference,
                    direction=direction,
                    remaining=current_remaining,
                )
                state["remaining_owned_shares"] = current_remaining
                return current_remaining
            except PhysicalIsolationMismatch as exc:
                # A target fill between reqExecutions and the position snapshot
                # legitimately shrinks both the virtual and physical lot. Read
                # executions again before treating a smaller physical net as a
                # foreign collision and cancelling its now-correct OCA pair.
                if not exc.foreign_working_orders:
                    try:
                        refreshed_fills = self._attributed_fills(
                            connection, reference
                        )
                        refreshed = owned_quantity(refreshed_fills, direction)
                    except Exception as refresh_exc:
                        transport_failures += 1
                        transport_errors.append(
                            f"{type(refresh_exc).__name__}: {refresh_exc}"
                        )
                        if transport_failures < 4:
                            self._restore_connection(connection)
                            continue
                        state["state"] = "critical_connection"
                        state["critical_reason"] = (
                            "could not refresh fills at a physical-isolation "
                            "boundary: " + " | ".join(transport_errors)
                        )
                        state["remaining_owned_shares"] = current_remaining
                        state["isolation_orders_preserved"] = True
                        self.store.put(signal_id, state)
                        raise RuntimeError(state["critical_reason"]) from refresh_exc
                    if refreshed < 0:
                        self._mark_over_exit(
                            signal_id=signal_id,
                            connection=connection,
                            reference=reference,
                            state=state,
                            remaining=refreshed,
                        )
                        raise RuntimeError("Legend lot was over-exited during isolation")
                    if refreshed != current_remaining:
                        fill_refreshes += 1
                        state["isolation_fill_refresh_from"] = current_remaining
                        state["isolation_fill_refresh_to"] = refreshed
                        state["remaining_owned_shares"] = refreshed
                        self.store.put(signal_id, state)
                        current_remaining = refreshed
                        mismatches.clear()
                        last_mismatch = None
                        self._restore_connection(connection)
                        # A quantity change never consumes the final proof.
                        # The next loop iteration must test physical isolation
                        # against the newly refreshed virtual lot.
                        continue
                mismatches.append(str(exc))
                last_mismatch = exc
            except Exception as exc:
                transport_failures += 1
                transport_errors.append(f"{type(exc).__name__}: {exc}")
                if transport_failures < 4:
                    self._restore_connection(connection)
                    continue
                state["state"] = "critical_connection"
                state["critical_reason"] = (
                    "could not obtain two fresh physical-isolation snapshots: "
                    + " | ".join(transport_errors)
                )
                state["remaining_owned_shares"] = current_remaining
                state["isolation_orders_preserved"] = True
                self.store.put(signal_id, state)
                raise RuntimeError(state["critical_reason"]) from exc
            if len(mismatches) < 2:
                self._restore_connection(connection)
                continue
        if last_mismatch is None or len(mismatches) < 2:
            state["state"] = "critical_connection"
            state["critical_reason"] = (
                "physical-isolation proof did not converge after bounded fresh "
                "fill/position snapshots; existing exits were preserved"
            )
            state["remaining_owned_shares"] = current_remaining
            state["isolation_orders_preserved"] = True
            self.store.put(signal_id, state)
            raise RuntimeError(state["critical_reason"])
        state["state"] = "critical_isolation"
        state["critical_reason"] = (
            "two fresh broker snapshots proved account-symbol isolation failure: "
            + " | ".join(mismatches)
        )
        state["remaining_owned_shares"] = current_remaining
        state["isolation_actual_position"] = last_mismatch.actual_position
        state["isolation_expected_position"] = last_mismatch.expected_position
        state["isolation_foreign_working_orders"] = (
            last_mismatch.foreign_working_orders
        )
        # A smaller/opposite physical lot or any foreign working order can make
        # the exact Legend exits reverse the net position. Cancel only Legend's
        # attributed orders in that case; never flatten a netted symbol. If the
        # physical net is larger on the same side, preserving the exact Legend
        # pair cannot cross zero and is the safer containment action.
        if last_mismatch.legend_exits_can_reverse_net_position:
            try:
                cleared = self._clear_reference_orders(connection, reference)
            except Exception as cancel_exc:  # noqa: BLE001 - persist ambiguity
                cleared = False
                state["isolation_cancel_error"] = (
                    f"{type(cancel_exc).__name__}: {cancel_exc}"
                )
            state["working_orders_cleared"] = cleared
            state["isolation_orders_preserved"] = not cleared
            state["isolation_collision_contained"] = cleared
        else:
            state["isolation_orders_preserved"] = True
            state["isolation_collision_contained"] = False
        self.store.put(signal_id, state)
        raise PhysicalIsolationMismatch(
            state["critical_reason"],
            actual_position=last_mismatch.actual_position,
            expected_position=last_mismatch.expected_position,
            foreign_working_orders=last_mismatch.foreign_working_orders,
        )

    def _mark_over_exit(
        self,
        *,
        signal_id: str,
        connection: IBKRConnection,
        reference: str,
        state: dict[str, Any],
        remaining: int,
    ) -> None:
        state["state"] = "critical_over_exit"
        state["remaining_owned_shares"] = remaining
        state["critical_reason"] = (
            "strategy-attributed exits exceeded entries; no ambiguous symbol-level "
            "flatten was sent"
        )
        state["working_orders_cleared"] = False
        state["over_exit_cancel_pending"] = True
        self.store.put(signal_id, state)
        try:
            cleared = self._clear_reference_orders(connection, reference)
            state["working_orders_cleared"] = cleared
            state["over_exit_cancel_pending"] = not cleared
            state.pop("over_exit_cancel_error", None)
        except Exception as exc:  # noqa: BLE001 - persist proof before best-effort cancel
            state["over_exit_cancel_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            self.store.put(signal_id, state)

    @staticmethod
    def _durable_context_contract(context: MarketContext) -> Any:
        """Return the already-qualified contract; never qualify after cancel."""

        contract = context.contract
        if int(getattr(contract, "conId", 0) or 0) <= 0:
            raise RuntimeError(
                f"{context.etf}: durable qualified contract has no conId"
            )
        return contract

    def _repair_protection(
        self,
        *,
        signal_id: str,
        connection: IBKRConnection,
        context: MarketContext,
        direction: int,
        reference: str,
        remaining: int,
        target: float,
        state: dict[str, Any],
    ) -> tuple[Any, int]:
        """Replace only an exactly attributed, positive Legend virtual lot."""

        if remaining <= 0:
            raise ValueError("repair requires a positive strategy-owned lot")
        contract = self._durable_context_contract(context)
        last_error = "replacement Legend exits were not proven at broker"
        for attempt in range(1, 3):
            if not connection.is_connected():
                self._restore_connection(connection)
            remaining = self._enforce_physical_isolation(
                signal_id=signal_id,
                connection=connection,
                context=context,
                reference=reference,
                direction=direction,
                remaining=remaining,
                state=state,
            )
            if remaining == 0:
                self._prove_zero_terminal(
                    signal_id=signal_id,
                    connection=connection,
                    context=context,
                    direction=direction,
                    reference=reference,
                    state=state,
                    terminal_state="complete_emergency",
                )
                raise RuntimeError("Legend lot became flat before exit repair")
            if not self._clear_reference_orders(connection, reference):
                raise RuntimeError("could not prove old Legend exits cancelled")
            # Cancelling a still-working parent can race a late fill. Recompute
            # the virtual lot only after every attributed order is terminal.
            fills = self._attributed_fills(connection, reference)
            remaining = owned_quantity(fills, direction)
            state["remaining_owned_shares"] = remaining
            if remaining < 0:
                self._mark_over_exit(
                    signal_id=signal_id,
                    connection=connection,
                    reference=reference,
                    state=state,
                    remaining=remaining,
                )
                raise RuntimeError("late fill produced a Legend over-exit")
            if remaining == 0:
                remaining, settled = self._prove_zero_terminal(
                    signal_id=signal_id,
                    connection=connection,
                    context=context,
                    direction=direction,
                    reference=reference,
                    state=state,
                    terminal_state="complete_emergency",
                )
                if settled:
                    raise RuntimeError(
                        "Legend lot became flat while exits were repaired"
                    )
                if remaining == 0:
                    raise RuntimeError(
                        "flat Legend lot could not obtain terminal broker proof"
                    )
            remaining = self._enforce_physical_isolation(
                signal_id=signal_id,
                connection=connection,
                context=context,
                reference=reference,
                direction=direction,
                remaining=remaining,
                state=state,
            )
            if remaining == 0:
                self._prove_zero_terminal(
                    signal_id=signal_id,
                    connection=connection,
                    context=context,
                    direction=direction,
                    reference=reference,
                    state=state,
                    terminal_state="complete_emergency",
                )
                raise RuntimeError("Legend lot became flat before OCA replacement")

            before = remaining
            try:
                replacement = connection.place_exit_oca(
                    contract=contract,
                    direction=direction,
                    shares=before,
                    target=target,
                    entry_date=self.entry_date,
                    order_ref=reference,
                )
                state.update(_order_ids(replacement))
                connection.confirm_broker_echo(
                    replacement, description="replacement Legend exit OCA"
                )
            except Exception as exc:  # noqa: BLE001 - reconcile ambiguous transmit
                last_error = f"{type(exc).__name__}: {exc}"
                # A disconnect after the transmitting leg is ambiguous. Use a
                # fresh same-client connection before inspecting broker state.
                self._restore_connection(connection)

            fills = self._attributed_fills(connection, reference)
            after = owned_quantity(fills, direction)
            state["remaining_owned_shares"] = after
            if after < 0:
                self._mark_over_exit(
                    signal_id=signal_id,
                    connection=connection,
                    reference=reference,
                    state=state,
                    remaining=after,
                )
                raise RuntimeError("replacement exits over-exited the Legend lot")
            if after == 0:
                after, settled = self._prove_zero_terminal(
                    signal_id=signal_id,
                    connection=connection,
                    context=context,
                    direction=direction,
                    reference=reference,
                    state=state,
                    terminal_state="complete_emergency",
                )
                if settled:
                    raise RuntimeError(
                        "Legend lot filled while protection was repaired"
                    )
                if after == 0:
                    raise RuntimeError(
                        "flat replacement lot could not obtain terminal broker proof"
                    )
            proven = self._proven_exit_pair_with_reconnect(
                connection=connection,
                reference=reference,
                direction=direction,
                remaining=after,
                expected_target=target,
            )
            if after == before and proven is not None:
                state["protection_repaired_at"] = pd.Timestamp.now(
                    tz="UTC"
                ).isoformat()
                state["protection_repair_attempts"] = attempt
                return proven[0], after
            last_error = (
                f"attempt {attempt}: owned quantity changed {before}->{after} "
                "or exact exit pair was absent"
            )
        raise RuntimeError(last_error)

    def _proven_emergency_order(
        self,
        *,
        connection: IBKRConnection,
        reference: str,
        direction: int,
        remaining: int,
    ) -> Any | None:
        working = self._working_orders(connection, reference)
        if len(working) != 1:
            return None
        trade = working[0]
        order = trade.order
        expected_action = "SELL" if direction > 0 else "BUY"
        if (
            str(trade.orderStatus.status or "")
            not in {"PreSubmitted", "Submitted"}
            or str(order.orderType or "").upper() != "MKT"
            or bool(str(order.goodAfterTime or "").strip())
            or str(order.action or "").upper() != expected_action
            or str(order.account or "") != connection.endpoint.account
            or str(order.orderRef or "") != reference
            or int(getattr(order, "clientId", -1))
            != connection.endpoint.client_id
            or int(getattr(order, "permId", 0) or 0) <= 0
            or round(float(getattr(trade.orderStatus, "remaining", 0) or 0))
            != remaining
        ):
            return None
        return trade

    def _emergency_flatten_owned_lot(
        self,
        *,
        signal_id: str,
        connection: IBKRConnection,
        context: MarketContext,
        direction: int,
        reference: str,
        state: dict[str, Any],
    ) -> None:
        """Flatten only a proven Legend lot after exit protection fails."""

        if not connection.is_connected():
            self._restore_connection(connection)
        contract = self._durable_context_contract(context)
        last_error = "emergency exit was not proven"
        for attempt in range(1, 3):
            fills = self._attributed_fills(connection, reference)
            remaining = owned_quantity(fills, direction)
            state["remaining_owned_shares"] = remaining
            if remaining < 0:
                self._mark_over_exit(
                    signal_id=signal_id,
                    connection=connection,
                    reference=reference,
                    state=state,
                    remaining=remaining,
                )
                raise RuntimeError("Legend lot was already over-exited")
            if remaining == 0:
                remaining, settled = self._prove_zero_terminal(
                    signal_id=signal_id,
                    connection=connection,
                    context=context,
                    direction=direction,
                    reference=reference,
                    state=state,
                    terminal_state="complete_emergency",
                )
                if settled:
                    return
                if remaining == 0:
                    raise RuntimeError(
                        "flat emergency lot could not obtain terminal broker proof"
                    )

            remaining = self._enforce_physical_isolation(
                signal_id=signal_id,
                connection=connection,
                context=context,
                reference=reference,
                direction=direction,
                remaining=remaining,
                state=state,
            )
            if remaining == 0:
                continue

            live_exit = self._proven_emergency_order(
                connection=connection,
                reference=reference,
                direction=direction,
                remaining=remaining,
            )
            if live_exit is not None:
                # Never submit a second market exit while an exact first one
                # remains active. Give it a bounded fill window, then preserve
                # it at the broker and keep the virtual lot in management.
                for _ in range(4):
                    connection.sleep(0.5)
                    fills = self._attributed_fills(connection, reference)
                    remaining = owned_quantity(fills, direction)
                    if remaining <= 0:
                        break
                    live_exit = self._proven_emergency_order(
                        connection=connection,
                        reference=reference,
                        direction=direction,
                        remaining=remaining,
                    )
                    if live_exit is None:
                        break
                if remaining == 0:
                    continue
                if remaining < 0:
                    self._mark_over_exit(
                        signal_id=signal_id,
                        connection=connection,
                        reference=reference,
                        state=state,
                        remaining=remaining,
                    )
                    raise RuntimeError("emergency exit over-exited the Legend lot")
                if live_exit is not None:
                    state["state"] = "critical_unflattened"
                    state["proven_emergency_order_id"] = int(
                        live_exit.order.orderId
                    )
                    state["critical_reason"] = (
                        "exact emergency market exit remains active at broker"
                    )
                    self.store.put(signal_id, state)
                    raise RuntimeError(state["critical_reason"])

            if not self._clear_reference_orders(connection, reference):
                state["state"] = "critical_unprotected"
                state["critical_reason"] = (
                    "could not cancel attributed orders before emergency flatten"
                )
                self.store.put(signal_id, state)
                raise RuntimeError(state["critical_reason"])
            # Re-read after cancellation before sizing the market exit.
            fills = self._attributed_fills(connection, reference)
            remaining = owned_quantity(fills, direction)
            if remaining <= 0:
                continue
            remaining = self._enforce_physical_isolation(
                signal_id=signal_id,
                connection=connection,
                context=context,
                reference=reference,
                direction=direction,
                remaining=remaining,
                state=state,
            )
            if remaining == 0:
                continue
            state["state"] = "time_exit_working"
            state["emergency_exit_intent_at"] = pd.Timestamp.now(
                tz="UTC"
            ).isoformat()
            state["emergency_exit_attempt"] = attempt
            self.store.put(signal_id, state)
            try:
                trade = connection.place_emergency_exit(
                    contract=contract,
                    direction=direction,
                    shares=remaining,
                    order_ref=reference,
                )
                state["emergency_order_id"] = int(trade.order.orderId)
                connection.confirm_broker_echo(
                    [trade], description="Legend emergency market exit"
                )
            except Exception as exc:  # noqa: BLE001 - reconcile ambiguous transmit
                last_error = f"{type(exc).__name__}: {exc}"
                self._restore_connection(connection)
            connection.sleep(1.0)
        fills = self._attributed_fills(connection, reference)
        remaining = owned_quantity(fills, direction)
        state["remaining_owned_shares"] = remaining
        if remaining < 0:
            self._mark_over_exit(
                signal_id=signal_id,
                connection=connection,
                reference=reference,
                state=state,
                remaining=remaining,
            )
            raise RuntimeError("emergency exit over-exited the Legend lot")
        if remaining == 0:
            remaining, settled = self._prove_zero_terminal(
                signal_id=signal_id,
                connection=connection,
                context=context,
                direction=direction,
                reference=reference,
                state=state,
                terminal_state="complete_emergency",
            )
            if settled:
                return
        live_exit = (
            self._proven_emergency_order(
                connection=connection,
                reference=reference,
                direction=direction,
                remaining=remaining,
            )
            if remaining > 0
            else None
        )
        if live_exit is not None:
            last_error = "exact emergency market exit remains active at broker"
            state["proven_emergency_order_id"] = int(live_exit.order.orderId)
        state["state"] = "critical_unflattened"
        state["critical_reason"] = last_error
        self.store.put(signal_id, state)
        raise RuntimeError(state["critical_reason"])

    def _reconcile_prior_records(self) -> list[ManagedTrade]:
        """Broker-check interrupted records and resume only proven-safe exits."""

        managed: list[ManagedTrade] = []
        contexts = {context.root: context for context in self.contexts}
        state = self.store.recover_interrupted()
        for signal_id, record in state.get("signals", {}).items():
            if record.get("entry_date") != self.entry_date:
                continue
            record_state = record.get("state")
            broker_mutated = self._broker_mutated(record)
            if record_state in {"dry_run", "skipped"}:
                continue
            if record_state in {"complete", "complete_emergency"} and not broker_mutated:
                continue
            if record_state == "blocked" and not broker_mutated:
                continue
            if record.get("recovery_unavailable_at"):
                # _bootstrap_recovery already persisted the precise endpoint,
                # reservation, or context failure. Do not let it prevent a
                # reachable peer lot from being protected through cutoff.
                continue
            label = record.get("account_label")
            connection = self.accounts.get(str(label))
            if connection is None:
                connection = next(
                    (
                        candidate
                        for candidate in self.accounts.values()
                        if candidate.endpoint.account == record.get("account")
                    ),
                    None,
                )
            context = contexts.get(str(record.get("root")))
            if connection is None or context is None or "direction" not in record:
                raise RuntimeError(
                    f"unresolved Legend record {signal_id} lacks recovery context"
                )
            reference = str(record.get("order_ref") or "")
            if not reference:
                raise RuntimeError(
                    f"unresolved Legend record {signal_id} has no orderRef"
                )
            reconciled = dict(record)
            direction = int(record["direction"])
            revisions = reconciled.get("target_revisions") or []
            pending_revision = reconciled.get("pending_target_revision")
            pending_deadline: pd.Timestamp | None = None
            if pending_revision is not None:
                if not isinstance(pending_revision, dict):
                    raise RuntimeError(
                        f"invalid pending target revision for {signal_id}"
                    )
                required_pending = {
                    "activation",
                    "source_bar",
                    "ema",
                    "limit",
                    "remaining",
                    "intent_at",
                }
                if not required_pending.issubset(pending_revision):
                    raise RuntimeError(
                        f"incomplete pending target revision for {signal_id}"
                    )
                allowed_pairs = {
                    ("09:46", "09:30"),
                    ("10:01", "09:45"),
                    ("10:16", "10:00"),
                }
                pair = (
                    str(pending_revision["activation"]),
                    str(pending_revision["source_bar"]),
                )
                pending_ema = float(pending_revision["ema"])
                pending_target = float(pending_revision["limit"])
                pending_remaining = int(pending_revision["remaining"])
                if (
                    pair not in allowed_pairs
                    or not math.isfinite(pending_ema)
                    or pending_ema <= 0
                    or not math.isfinite(pending_target)
                    or pending_target <= 0
                    or pending_remaining <= 0
                ):
                    raise RuntimeError(
                        f"invalid pending target revision values for {signal_id}"
                    )
                if revisions:
                    latest_pair = (
                        str(revisions[-1].get("activation")),
                        str(revisions[-1].get("source_bar")),
                    )
                    order = {"09:46": 0, "10:01": 1, "10:16": 2}
                    if order.get(latest_pair[0], -1) >= order[pair[0]]:
                        same = (
                            latest_pair == pair
                            and math.isclose(
                                float(revisions[-1]["ema"]), pending_ema
                            )
                            and math.isclose(
                                float(revisions[-1]["limit"]), pending_target
                            )
                        )
                        if not same:
                            raise RuntimeError(
                                f"pending target revision predates committed state for {signal_id}"
                            )
                        reconciled.pop("pending_target_revision", None)
                        pending_revision = None
                if pending_revision is not None:
                    phase = str(
                        pending_revision.get("phase") or "transmitting"
                    )
                    if phase not in {"prepared", "transmitting"}:
                        raise RuntimeError(
                            f"invalid pending target revision phase for {signal_id}"
                        )
                    activation_deadline = et_timestamp(
                        self.entry_date, pair[0]
                    ) + pd.Timedelta(seconds=1)
                    if phase == "prepared":
                        reason = (
                            "prepared target revision was never transmit-authorized"
                        )
                        reconciled["abandoned_pending_target_revision"] = dict(
                            pending_revision
                        )
                        reconciled.pop("pending_target_revision", None)
                        reconciled["degraded_execution"] = True
                        reconciled.setdefault("degraded_reasons", []).append(
                            reason
                        )
                        detail = f"{signal_id}({reason})"
                        if detail not in self.recovery_degraded_reasons:
                            self.recovery_degraded_reasons.append(detail)
                        pending_revision = None
                    else:
                        # Once the whole batch is durably transmit-authorized,
                        # a crash can occur before the acknowledgement write.
                        # The broker—not wall-clock age—is then the source of
                        # truth.  Recovery proves the pending target, proves the
                        # prior target, or repairs the prior target after this
                        # deadline; it never blindly applies a stale revision.
                        pending_deadline = activation_deadline
            committed_ema = float(
                revisions[-1]["ema"]
                if revisions
                else reconciled["initial_ema"]
            )
            current_ema = float(
                pending_revision["ema"]
                if pending_revision is not None
                else committed_ema
            )
            committed_target = float(
                revisions[-1].get(
                    "limit", round_limit(committed_ema, direction)
                )
                if revisions
                else reconciled.get(
                    "initial_target", round_limit(committed_ema, direction)
                )
            )
            target = float(
                pending_revision["limit"]
                if pending_revision is not None
                else committed_target
            )
            unresolved = ManagedTrade(
                signal_id=signal_id,
                connection=connection,
                context=context,
                direction=direction,
                order_ref=reference,
                target_trade=None,
                current_ema=current_ema,
                state=reconciled,
                allow_target_updates=False,
                order_watch_only=True,
            )
            try:
                fills = self._attributed_fills(connection, reference)
            except Exception as exc:  # noqa: BLE001 - retain broker supervision
                reconciled["state"] = "critical_connection"
                reconciled["critical_reason"] = str(exc)
                self.store.put(signal_id, reconciled)
                managed.append(unresolved)
                continue
            remaining = owned_quantity(fills, direction)
            reconciled["broker_fill_count"] = len(fills)
            reconciled["remaining_owned_shares"] = remaining
            if remaining < 0:
                self._mark_over_exit(
                    signal_id=signal_id,
                    connection=connection,
                    reference=reference,
                    state=reconciled,
                    remaining=remaining,
                )
                detail = f"{signal_id}(critical_over_exit_requires_manual_review)"
                if detail not in self.recovery_degraded_reasons:
                    self.recovery_degraded_reasons.append(detail)
                continue
            if remaining == 0:
                terminal_state = (
                    str(record_state)
                    if record_state in {"complete", "complete_emergency"}
                    else ("complete" if fills else "blocked")
                )
                try:
                    remaining, settled = self._settle_zero_owned(
                        unresolved, terminal_state=terminal_state
                    )
                except RuntimeError:
                    # Over-exits are manualized, but must not strand a later
                    # independent peer lot in the record loop.
                    if reconciled.get("state") == "critical_over_exit":
                        detail = (
                            f"{signal_id}(critical_over_exit_requires_manual_review)"
                        )
                        if detail not in self.recovery_degraded_reasons:
                            self.recovery_degraded_reasons.append(detail)
                        continue
                    managed.append(unresolved)
                    continue
                if settled:
                    continue
                if remaining == 0:
                    managed.append(unresolved)
                    continue

            try:
                proven = None
                if pending_revision is not None:
                    proven = self._proven_exit_pair_with_reconnect(
                        connection=connection,
                        reference=reference,
                        direction=direction,
                        remaining=remaining,
                        expected_target=target,
                    )
                    if proven is None:
                        prior_proven = self._proven_exit_pair_with_reconnect(
                            connection=connection,
                            reference=reference,
                            direction=direction,
                            remaining=remaining,
                            expected_target=committed_target,
                        )
                        deadline_missed = (
                            pending_deadline is not None
                            and pd.Timestamp.now(tz=NY_TZ) > pending_deadline
                        )
                        if prior_proven is not None or deadline_missed:
                            reason = (
                                "broker retained the prior target after restart"
                                if prior_proven is not None
                                else (
                                    "pending target was not broker-proven before its "
                                    "deadline; prior protection was restored"
                                )
                            )
                            reconciled["abandoned_pending_target_revision"] = dict(
                                pending_revision
                            )
                            reconciled.pop("pending_target_revision", None)
                            reconciled["degraded_execution"] = True
                            reconciled.setdefault("degraded_reasons", []).append(
                                reason
                            )
                            detail = f"{signal_id}({reason})"
                            if detail not in self.recovery_degraded_reasons:
                                self.recovery_degraded_reasons.append(detail)
                            pending_revision = None
                            target = committed_target
                            current_ema = committed_ema
                            unresolved.current_ema = committed_ema
                            proven = prior_proven
                else:
                    proven = self._proven_exit_pair_with_reconnect(
                        connection=connection,
                        reference=reference,
                        direction=direction,
                        remaining=remaining,
                        expected_target=target,
                    )
                if proven is None:
                    target_trade, remaining = self._repair_protection(
                        signal_id=signal_id,
                        connection=connection,
                        context=context,
                        direction=direction,
                        reference=reference,
                        remaining=remaining,
                        target=target,
                        state=reconciled,
                    )
                    reconciled["state"] = "open_repaired"
                else:
                    target_trade = proven[0]
                    reconciled["state"] = "open_reconciled"
            except Exception as exc:  # noqa: BLE001 - emergency containment boundary
                reconciled["state"] = "critical_unprotected"
                reconciled["critical_reason"] = str(exc)
                self.store.put(signal_id, reconciled)
                managed.append(unresolved)
                try:
                    self._emergency_flatten_owned_lot(
                        signal_id=signal_id,
                        connection=connection,
                        context=context,
                        direction=direction,
                        reference=reference,
                        state=reconciled,
                    )
                except Exception as flatten_exc:  # noqa: BLE001 - preserve critical lot
                    self.store.append_audit(
                        "recovery_emergency_flatten_failed",
                        signal_id=signal_id,
                        error_type=type(flatten_exc).__name__,
                    )
                    continue
                managed.remove(unresolved)
                continue
            unresolved.target_trade = target_trade
            unresolved.order_watch_only = False
            if pending_revision is not None:
                acknowledged_at = pd.Timestamp.now(tz="UTC").isoformat()
                reconciled.setdefault("target_revisions", []).append(
                    {
                        "activation": pending_revision["activation"],
                        "source_bar": pending_revision["source_bar"],
                        "ema": pending_revision["ema"],
                        "limit": pending_revision["limit"],
                        "remaining": remaining,
                        "transmitted_at": reconciled.get(
                            "target_revision_transmitted_at"
                        ),
                        "acknowledged_at": acknowledged_at,
                        "recovered_after_restart": True,
                    }
                )
                reconciled["target_revision_acknowledged_at"] = acknowledged_at
                reconciled["target_revision_recovered_at"] = acknowledged_at
                reconciled.pop("pending_target_revision", None)
            managed.append(unresolved)
            self.store.put(signal_id, reconciled)
        return managed

    def _clear_lingering_orders(self, trade: ManagedTrade) -> bool:
        return self._clear_reference_orders(trade.connection, trade.order_ref)

    def _persist_terminal_proof(
        self,
        signal_id: str,
        state: dict[str, Any],
        *,
        terminal_state: str,
    ) -> None:
        if terminal_state not in {"complete", "complete_emergency", "blocked"}:
            raise ValueError("invalid terminal Legend state")
        state["state"] = terminal_state
        state["remaining_owned_shares"] = 0
        state["working_orders_cleared"] = True
        state["terminal_proved_at"] = pd.Timestamp.now(tz="UTC").isoformat()
        state.pop("pending_target_revision", None)
        state.pop("critical_reason", None)
        if state.get("quarantine_required"):
            state["quarantine_active"] = True
            state["quarantine_release_pending"] = True
            state["correction_audit_required"] = True
        # Persist the terminal broker proof while the durable quarantine is
        # still active. A crash at this boundary can only leave the symbol
        # over-protected, never available to another executor too early.
        self.store.put(signal_id, state)

    def _resolve_zero_physical_boundary(
        self,
        *,
        signal_id: str,
        connection: IBKRConnection,
        context: MarketContext,
        direction: int,
        reference: str,
        state: dict[str, Any],
        phase: str,
    ) -> int:
        """Resolve a zero-fill/physical mismatch without cancelling protection."""

        try:
            self._assert_physical_isolation(
                connection=connection,
                context=context,
                reference=reference,
                direction=direction,
                remaining=0,
            )
            return 0
        except Exception as physical_exc:
            connection.sleep(0.25)
            fills = self._attributed_fills(connection, reference)
            remaining = owned_quantity(fills, direction)
            state["broker_fill_count"] = len(fills)
            state["remaining_owned_shares"] = remaining
            if remaining < 0:
                self._mark_over_exit(
                    signal_id=signal_id,
                    connection=connection,
                    reference=reference,
                    state=state,
                    remaining=remaining,
                )
                raise RuntimeError(f"Legend over-exit for {signal_id}")
            if remaining > 0:
                try:
                    self._assert_physical_isolation(
                        connection=connection,
                        context=context,
                        reference=reference,
                        direction=direction,
                        remaining=remaining,
                    )
                except Exception as confirmed_exc:
                    state["state"] = "critical_isolation"
                    state["critical_reason"] = str(confirmed_exc)
                    state["working_orders_cleared"] = False
                    self.store.put(signal_id, state)
                    raise RuntimeError(
                        "late Legend fill does not match the physical account lot"
                    ) from confirmed_exc
                state["state"] = "late_entry_fill"
                state["critical_reason"] = (
                    f"entry fill appeared during {phase} zero-lot proof"
                )
                state["working_orders_cleared"] = False
                self.store.put(signal_id, state)
                return remaining
            state["state"] = "critical_isolation"
            state["critical_reason"] = (
                f"physical account mismatch during {phase} zero-lot proof: "
                f"{physical_exc}"
            )
            state["working_orders_cleared"] = False
            self.store.put(signal_id, state)
            raise RuntimeError(state["critical_reason"]) from physical_exc

    def _prove_zero_terminal(
        self,
        *,
        signal_id: str,
        connection: IBKRConnection,
        context: MarketContext,
        direction: int,
        reference: str,
        state: dict[str, Any],
        terminal_state: str,
    ) -> tuple[int, bool]:
        """Prove signed-zero, no orders, and physical zero at one boundary."""

        late_remaining = self._resolve_zero_physical_boundary(
            signal_id=signal_id,
            connection=connection,
            context=context,
            direction=direction,
            reference=reference,
            state=state,
            phase="pre-cancel",
        )
        if late_remaining > 0:
            return late_remaining, False
        cleared = self._clear_reference_orders(connection, reference)
        connection.sleep(0.5)
        fills = self._attributed_fills(connection, reference)
        remaining = owned_quantity(fills, direction)
        state["broker_fill_count"] = len(fills)
        state["remaining_owned_shares"] = remaining
        if remaining < 0:
            self._mark_over_exit(
                signal_id=signal_id,
                connection=connection,
                reference=reference,
                state=state,
                remaining=remaining,
            )
            raise RuntimeError(f"Legend over-exit for {signal_id}")
        if remaining > 0:
            state["state"] = "late_entry_fill"
            state["critical_reason"] = (
                "entry filled while attributed orders were being cancelled"
            )
            self.store.put(signal_id, state)
            return remaining, False
        # A previously invisible, ambiguously transmitted parent can appear
        # only after the first reqAllOpenOrders snapshot. Prove cancellation a
        # second time after the settlement delay, then re-read executions once
        # more so order proof and signed-zero share the final boundary.
        cleared = self._clear_reference_orders(connection, reference) and cleared
        connection.sleep(0.25)
        fills = self._attributed_fills(connection, reference)
        remaining = owned_quantity(fills, direction)
        state["broker_fill_count"] = len(fills)
        state["remaining_owned_shares"] = remaining
        if remaining < 0:
            self._mark_over_exit(
                signal_id=signal_id,
                connection=connection,
                reference=reference,
                state=state,
                remaining=remaining,
            )
            raise RuntimeError(f"Legend over-exit for {signal_id}")
        if remaining > 0:
            state["state"] = "late_entry_fill"
            state["critical_reason"] = (
                "entry filled during final zero-lot settlement"
            )
            self.store.put(signal_id, state)
            return remaining, False
        late_remaining = self._resolve_zero_physical_boundary(
            signal_id=signal_id,
            connection=connection,
            context=context,
            direction=direction,
            reference=reference,
            state=state,
            phase="final",
        )
        if late_remaining > 0:
            return late_remaining, False
        if cleared:
            self._persist_terminal_proof(
                signal_id,
                state,
                terminal_state=terminal_state,
            )
            return 0, True
        state["state"] = "critical_working_order"
        state["critical_reason"] = (
            "zero Legend virtual lot still has an attributed working order"
        )
        state["working_orders_cleared"] = False
        self.store.put(signal_id, state)
        return 0, False

    def _settle_zero_owned(
        self, trade: ManagedTrade, *, terminal_state: str = "complete"
    ) -> tuple[int, bool]:
        """Cancel, settle, and re-read a zero virtual lot before dropping it.

        A zero fill snapshot is not terminal while a parent or exit can still
        execute.  Every caller retains ``trade`` when ``settled`` is false.
        The post-cancel execution query also catches a parent fill racing the
        cancellation.
        """

        return self._prove_zero_terminal(
            signal_id=trade.signal_id,
            connection=trade.connection,
            context=trade.context,
            direction=trade.direction,
            reference=trade.order_ref,
            state=trade.state,
            terminal_state=terminal_state,
        )

    def _prepare_account(
        self, label: str, eligible: list[MarketContext]
    ) -> list[PreparedOrder]:
        """Complete every account/root validation before any broker mutation."""

        assert self.gate is not None
        assert self.feed is not None
        connection = self.accounts[label]
        profile = self.preentry_profiles[label]
        nlv = self.preentry_nlv[label]
        sizes = size_batch(
            [
                SizeRequest(
                    context.root,
                    context.etf,
                    int(context.direction),
                    context.atr14,
                    float(context.decision_price),
                )
                for context in eligible
            ],
            nlv=nlv,
            profile=profile,
            available_long_bps=(
                self.portfolio_capacities[connection.endpoint.account]
                .remaining_long_bps
                if self.gate.live
                else None
            ),
            available_short_bps=(
                self.portfolio_capacities[connection.endpoint.account]
                .remaining_short_bps
                if self.gate.live
                else None
            ),
            available_gross_bps=(
                self.portfolio_capacities[connection.endpoint.account]
                .remaining_gross_bps
                if self.gate.live
                else None
            ),
        )
        by_root = {result.root: result for result in sizes}
        prepared: list[PreparedOrder] = []
        for context in eligible:
            direction = int(context.direction)
            sizing = by_root[context.root]
            signal_id = signal_identity(
                account=connection.endpoint.account,
                root=context.root,
                etf=context.etf,
                setup_date=context.setup_date,
                entry_date=self.entry_date,
                direction=direction,
            )
            reference = build_order_ref(context.etf, direction, self.entry_date)
            base = {
                "state": "planned",
                "account_label": label,
                "account": connection.endpoint.account,
                "execution_endpoint": {
                    "label": connection.endpoint.label,
                    "host": connection.endpoint.host,
                    "port": connection.endpoint.port,
                    "client_id": connection.endpoint.client_id,
                    "account": connection.endpoint.account,
                },
                "reservation_dir": (
                    self.runtime_environment["LEGEND_ETF_RESERVATION_DIR"]
                    if self.runtime_environment is not None
                    else ""
                ),
                "deployment_manifest_sha256": (
                    self.runtime_environment["LEGEND_ETF_GUARD_MANIFEST_SHA256"]
                    if self.runtime_environment is not None
                    else ""
                ),
                "portfolio_budget_sha256": self.portfolio_budget_sha256 or "",
                "root": context.root,
                "etf": context.etf,
                "setup_date": context.setup_date,
                "entry_date": self.entry_date,
                "direction": direction,
                "side": "long" if direction > 0 else "short",
                "order_ref": reference,
                "plan_hash": self.plan["plan_hash"],
                "signal_data_as_of": self.plan["data_as_of"],
                "signal_dataset": self.plan["dataset"],
                "nlv_frozen": nlv,
                "initial_ema": context.initial_ema,
                "initial_target": context.initial_target,
                "decision_price": context.decision_price,
                "decision_observed_at": context.decision_observed_at,
                "atr14": context.atr14,
                "sizing": sizing.to_dict(),
            }
            existing = self._existing_record(connection.endpoint.account, context)
            if existing:
                self.store.append_audit(
                    "duplicate_entry_intent_suppressed",
                    signal_id=signal_id,
                    preserved_state=existing.get("state"),
                    account=connection.endpoint.account,
                    root=context.root,
                )
                continue
            if sizing.shares <= 0:
                base["state"] = "skipped"
                base["skip_reason"] = sizing.reason
                self.store.put(signal_id, base)
                continue
            preentry_key = (label, context.root)
            preentry_block = self.preentry_blocks.get(preentry_key)
            if preentry_block:
                base["state"] = "blocked"
                base["blocked_reason"] = preentry_block
                self.store.put(signal_id, base)
                continue
            try:
                self.gate.require_side(direction)
            except RuntimeError as exc:
                # Side switches are intentional per-root controls. A disabled
                # short must not suppress an independently valid long in the
                # same account batch (or vice versa).
                base["state"] = "blocked"
                base["blocked_reason"] = str(exc)
                self.store.put(signal_id, base)
                continue
            contract = self.preentry_contracts[preentry_key]
            execution_con_id = int(getattr(contract, "conId", 0) or 0)
            context_con_id = int(getattr(context.contract, "conId", 0) or 0)
            if execution_con_id <= 0 or execution_con_id != context_con_id:
                base["state"] = "blocked"
                base["blocked_reason"] = "ETF contract conId mismatch across IBKR clients"
                self.store.put(signal_id, base)
                continue
            base["execution_con_id"] = execution_con_id
            if direction < 0:
                short_block = self.preentry_short_blocks.get(preentry_key)
                if short_block:
                    base["state"] = "blocked"
                    base["blocked_reason"] = short_block
                    self.store.put(signal_id, base)
                    continue
                base["shortability"] = self.preentry_short_detail.get(
                    context.root, "pre-open borrow probe passed"
                )

            if not self.gate.live:
                base["state"] = "dry_run"
                self.store.put(signal_id, base)
                continue
            prepared.append(
                PreparedOrder(
                    label=label,
                    connection=connection,
                    context=context,
                    direction=direction,
                    sizing=sizing,
                    signal_id=signal_id,
                    order_ref=reference,
                    contract=contract,
                    state=base,
                )
            )
        return prepared

    @contextmanager
    def _reserve_portfolio_batch(
        self, prepared: list[PreparedOrder]
    ) -> Any:
        """Debit all account/root stress capacity atomically before transmission."""

        if not (self.gate and self.gate.live) or not prepared:
            yield
            return
        if self.runtime_environment is None:
            raise RuntimeError("live portfolio reservation lacks runtime configuration")
        requirements = {
            endpoint.account: PortfolioRequirement(0.0, 0.0, 0.0)
            for endpoint in self.endpoints
        }
        totals = {
            endpoint.account: {"long": 0.0, "short": 0.0, "gross": 0.0}
            for endpoint in self.endpoints
        }
        for item in prepared:
            account = item.connection.endpoint.account
            nlv = float(item.state["nlv_frozen"])
            actual_bps = item.sizing.stress_risk_usd / nlv * 10_000.0
            side = "long" if item.direction > 0 else "short"
            totals[account][side] += actual_bps
            totals[account]["gross"] += actual_bps
        requirements = {
            account: PortfolioRequirement(
                long_bps=values["long"],
                short_bps=values["short"],
                gross_bps=values["gross"],
            )
            for account, values in totals.items()
        }
        signal_ids = sorted(item.signal_id for item in prepared)
        owner_token = "legend-" + content_hash(
            {
                "strategy_version": STRATEGY_VERSION,
                "entry_date": self.entry_date,
                "plan_hash": self.plan["plan_hash"],
                "accounts": sorted(requirements),
                "signals": signal_ids,
            }
        )
        budget_path = Path(
            self.runtime_environment["LEGEND_ETF_PORTFOLIO_BUDGET"]
        )
        reservation_dir = Path(
            self.runtime_environment["LEGEND_ETF_RESERVATION_DIR"]
        )
        with reserve_portfolio_capacity(
            budget_path,
            lock_path=portfolio_budget_lock_path(reservation_dir),
            entry_date=self.entry_date,
            account_requirements=requirements,
            owner_token=owner_token,
            signal_ids=signal_ids,
            expected_manifest_sha256=self.runtime_environment[
                "LEGEND_ETF_GUARD_MANIFEST_SHA256"
            ],
        ) as (capacities, digest):
            self.portfolio_capacities = capacities
            self.portfolio_budget_sha256 = digest
            for item in prepared:
                item.state["portfolio_reservation_token"] = owner_token
                item.state["portfolio_budget_sha256"] = digest
                item.state["portfolio_reserved_at"] = pd.Timestamp.now(
                    tz="UTC"
                ).isoformat()
                self.store.put(item.signal_id, item.state)
            self.store.append_audit(
                "portfolio_batch_reserved",
                owner_token=owner_token,
                signal_ids=signal_ids,
                accounts={
                    account: {
                        "long_bps": requirement.long_bps,
                        "short_bps": requirement.short_bps,
                        "gross_bps": requirement.gross_bps,
                    }
                    for account, requirement in requirements.items()
                },
                budget_sha256=digest,
            )
            # Keep the cross-executor cluster mutex until every staged bracket
            # has either crossed the wire or been causally blocked.
            yield

    def _submission_market_recheck(self, item: PreparedOrder) -> tuple[float, str]:
        now_et = pd.Timestamp.now(tz=NY_TZ)
        frame = trade_ticks_to_frame(item.context.trade_ticks)
        return validate_submission_tape(
            frame,
            entry_date=self.entry_date,
            direction=item.direction,
            target=float(item.context.initial_target),
            now_et=now_et,
        )

    def _record_submission_market_recheck(
        self,
        item: PreparedOrder,
        result: list[tuple[float, str]],
    ) -> None:
        if self.feed is None:
            raise RuntimeError("market-data feed is absent during final entry proof")
        baseline = len(getattr(item.context.trade_ticks, "tickByTicks", ()))
        self.feed.wait_for_new_trade_tick(
            item.context.trade_ticks,
            after_count=baseline,
        )
        result[0] = self._submission_market_recheck(item)

    def _transmit_batch(
        self, prepared: list[PreparedOrder], managed: list[ManagedTrade]
    ) -> list[SubmittedOrder]:
        """Persist and transmit every eligible bracket before any ack wait.

        Overlap days can contain six account/root orders. Broker echo and fill
        reconciliation are intentionally deferred until all eligible brackets
        have crossed the wire inside the 09:31:20 entry deadline.
        """

        submitted: list[SubmittedOrder] = []
        for item in prepared:
            connection = item.connection
            base = item.state
            try:
                if not (
                    self.reservations
                    and self.reservations.holds(
                        connection.endpoint.account, item.context.etf
                    )
                ):
                    raise RuntimeError("account-symbol reservation is not held")
                # Reservation first closes the cross-executor race. Broker
                # state can then be checked before the *final* causal tape
                # check, which must be immediately adjacent to transmission.
                connection.assert_symbol_clear(item.contract)
                self._refresh_live_gate()
                assert self.gate is not None
                self.gate.require_side(item.direction)
                # Fast rejection before any durable intent/quarantine writes.
                # A second tape proof is taken after those writes, literally
                # adjacent to the broker call below.
                self._submission_market_recheck(item)
            except RuntimeError as exc:
                base["state"] = "blocked"
                base["blocked_reason"] = str(exc)
                base["submission_check_at"] = pd.Timestamp.now(
                    tz="UTC"
                ).isoformat()
                self.store.put(item.signal_id, base)
                continue
            base["state"] = "submitting"
            base["submit_intent_at"] = pd.Timestamp.now(tz="UTC").isoformat()
            base["quarantine_required"] = True
            self.store.put(item.signal_id, base)
            assert self.reservations is not None
            self.reservations.activate_quarantine(
                connection.endpoint.account,
                item.context.etf,
                owner_token=item.signal_id,
                payload={
                    "signal_id": item.signal_id,
                    "entry_date": self.entry_date,
                    "order_ref": item.order_ref,
                    "con_id": int(item.contract.conId),
                },
            )
            base["quarantine_active"] = True
            self.store.put(item.signal_id, base)
            watched = ManagedTrade(
                signal_id=item.signal_id,
                connection=connection,
                context=item.context,
                direction=item.direction,
                order_ref=item.order_ref,
                target_trade=None,
                current_ema=item.context.initial_ema,
                state=base,
                allow_target_updates=False,
                order_watch_only=True,
            )
            # Add the watch before the first placeOrder. A transport exception
            # may occur after IBKR accepted the transmitting leg.
            managed.append(watched)
            try:
                # Nothing that can block on disk or network is allowed between
                # this causal tape proof and placeOrder. In particular, state
                # and quarantine fsyncs have already completed above.
                recheck_price, recheck_stamp = self._submission_market_recheck(item)
            except RuntimeError as exc:
                # No broker method has been called. Clear the durable
                # quarantine first; if that I/O fails, the prior ambiguous
                # intent remains recoverable instead of being mislabeled safe.
                self.reservations.deactivate_quarantine(
                    connection.endpoint.account,
                    item.context.etf,
                    owner_token=item.signal_id,
                )
                base.pop("submit_intent_at", None)
                base["state"] = "blocked"
                base["blocked_reason"] = str(exc)
                base["submission_aborted_before_broker_call"] = True
                base["submission_check_at"] = pd.Timestamp.now(
                    tz="UTC"
                ).isoformat()
                base["quarantine_required"] = False
                base["quarantine_active"] = False
                base["quarantine_release_pending"] = False
                base["quarantine_released_at"] = pd.Timestamp.now(
                    tz="UTC"
                ).isoformat()
                self.store.put(item.signal_id, base)
                managed.remove(watched)
                continue
            final_check: list[tuple[float, str]] = [(recheck_price, recheck_stamp)]

            final_transmit_check = partial(
                self._record_submission_market_recheck, item, final_check
            )

            try:
                trades = connection.place_bracket(
                    contract=item.contract,
                    direction=item.direction,
                    shares=item.sizing.shares,
                    target=float(item.context.initial_target),
                    entry_date=self.entry_date,
                    final_transmit_check=final_transmit_check,
                )
            except PreTransmitCheckBlocked as exc:
                # Adapter proved the two transmit=False staged legs cancelled
                # and never called the transmitting third placeOrder.
                self.reservations.deactivate_quarantine(
                    connection.endpoint.account,
                    item.context.etf,
                    owner_token=item.signal_id,
                )
                base.pop("submit_intent_at", None)
                base["state"] = "blocked"
                base["blocked_reason"] = str(exc)
                base["submission_aborted_before_transmitting_leg"] = True
                base["submission_check_at"] = pd.Timestamp.now(
                    tz="UTC"
                ).isoformat()
                base["quarantine_required"] = False
                base["quarantine_active"] = False
                base["quarantine_release_pending"] = False
                base["quarantine_released_at"] = pd.Timestamp.now(
                    tz="UTC"
                ).isoformat()
                self.store.put(item.signal_id, base)
                managed.remove(watched)
                continue
            except Exception as exc:  # noqa: BLE001 - ambiguous transmit boundary
                base["state"] = "unknown"
                base["unknown_reason"] = (
                    f"bracket_call_failed:{type(exc).__name__}"
                )
                self.store.put(item.signal_id, base)
                submitted.append(
                    SubmittedOrder(item, watched, None, str(exc))
                )
                continue
            recheck_price, recheck_stamp = final_check[0]
            base["submission_market_price"] = recheck_price
            base["submission_market_bar_at"] = recheck_stamp
            base.update(_order_ids(trades))
            base["state"] = "entry_working"
            base["submitted_at"] = pd.Timestamp.now(tz="UTC").isoformat()
            self.store.put(item.signal_id, base)
            watched.target_trade = trades[1]
            watched.order_watch_only = False
            submitted.append(SubmittedOrder(item, watched, trades))
        return submitted

    def _reconcile_submission(
        self, submission: SubmittedOrder, managed: list[ManagedTrade]
    ) -> None:
        """Reconcile one already-transmitted intent without dropping its watch."""

        item = submission.item
        trade = submission.managed_trade
        base = trade.state
        connection = item.connection
        acknowledgement_error: str | None = submission.transmit_error
        if submission.trades is None:
            try:
                self._restore_connection(connection)
            except RuntimeError as exc:
                base["state"] = "critical_connection"
                base["critical_reason"] = str(exc)
                self.store.put(item.signal_id, base)
                raise RuntimeError(
                    "ambiguous bracket submission could not reconnect for "
                    "immediate reconciliation"
                ) from exc
        if submission.trades is not None and not submission.entry_cancel_requested:
            try:
                connection.confirm_broker_echo(
                    submission.trades, description="Legend entry bracket"
                )
            except RuntimeError as exc:
                acknowledgement_error = str(exc)
                base["broker_ack_error"] = acknowledgement_error
                self.store.put(item.signal_id, base)
        # The echo wait can last several seconds. A parent or target can fill
        # during it, so quantities must come from a fresh execution snapshot
        # taken after the wait/error boundary, never the pre-wait snapshot.
        try:
            fills = self._attributed_fills(connection, item.order_ref)
        except Exception as exc:
            base["state"] = "critical_connection"
            base["critical_reason"] = str(exc)
            self.store.put(item.signal_id, base)
            raise RuntimeError(
                "ambiguous bracket submission lacks a fresh execution snapshot"
            ) from exc
        entered = entry_filled_quantity(fills, item.direction)
        remaining = owned_quantity(fills, item.direction)
        if remaining < 0:
            self._mark_over_exit(
                signal_id=item.signal_id,
                connection=connection,
                reference=item.order_ref,
                state=base,
                remaining=remaining,
            )
            raise RuntimeError(f"Legend over-exit for {item.signal_id}")
        if remaining == 0:
            base["filled_shares"] = entered
            base["blocked_reason"] = (
                "bracket_submission_failed_no_fill"
                if submission.trades is None
                else "entry_market_order_not_filled_in_window"
            )
            remaining, settled = self._settle_zero_owned(
                trade, terminal_state="complete" if entered else "blocked"
            )
            if settled:
                managed.remove(trade)
                return
            if remaining == 0:
                trade.order_watch_only = True
                trade.allow_target_updates = False
                trade.target_trade = None
                raise RuntimeError(
                    "zero virtual lot lacks terminal order proof after submission"
                )

        target = float(item.context.initial_target)
        remaining = self._enforce_physical_isolation(
            signal_id=item.signal_id,
            connection=connection,
            context=item.context,
            reference=item.order_ref,
            direction=item.direction,
            remaining=remaining,
            state=base,
        )
        if remaining == 0:
            _, settled = self._settle_zero_owned(
                trade, terminal_state="complete" if entered else "blocked"
            )
            if settled:
                managed.remove(trade)
            return
        proven = self._proven_exit_pair_with_reconnect(
            connection=connection,
            reference=item.order_ref,
            direction=item.direction,
            remaining=remaining,
            expected_target=target,
        )
        if proven is None:
            try:
                target_trade, remaining = self._repair_protection(
                    signal_id=item.signal_id,
                    connection=connection,
                    context=item.context,
                    direction=item.direction,
                    reference=item.order_ref,
                    remaining=remaining,
                    target=target,
                    state=base,
                )
            except Exception as exc:  # noqa: BLE001 - emergency containment boundary
                base["state"] = "critical_unprotected"
                base["critical_reason"] = str(exc)
                self.store.put(item.signal_id, base)
                trade.allow_target_updates = False
                trade.order_watch_only = True
                trade.target_trade = None
                try:
                    self._emergency_flatten_owned_lot(
                        signal_id=item.signal_id,
                        connection=connection,
                        context=item.context,
                        direction=item.direction,
                        reference=item.order_ref,
                        state=base,
                    )
                except Exception as flatten_exc:  # noqa: BLE001 - retain watch
                    self.store.append_audit(
                        "submission_emergency_flatten_failed",
                        signal_id=item.signal_id,
                        error_type=type(flatten_exc).__name__,
                    )
                    return
                managed.remove(trade)
                return
            base["entry_bracket_reprotected"] = True
        else:
            target_trade = proven[0]
            if acknowledgement_error is not None:
                self.store.append_audit(
                    "broker_ack_timeout_but_exact_pair_proven",
                    signal_id=item.signal_id,
                    detail=acknowledgement_error,
                )
        base["state"] = "open"
        base["filled_shares"] = entered
        base["remaining_owned_shares"] = remaining
        entry_action = "BOT" if item.direction > 0 else "SLD"
        entry_times = [
            fill.timestamp for fill in fills if fill.action.upper() == entry_action
        ]
        if entry_times:
            base["first_entry_fill_at"] = min(entry_times)
        self.store.put(item.signal_id, base)
        trade.target_trade = target_trade
        trade.allow_target_updates = True
        trade.order_watch_only = False

    def _recover_failed_submission_reconciliation(
        self,
        submission: SubmittedOrder,
        managed: list[ManagedTrade],
        *,
        cause: Exception,
    ) -> None:
        """Immediately prove, repair, or flatten a lot after reconcile failure."""

        item = submission.item
        trade = submission.managed_trade
        connection = item.connection
        state = trade.state
        trade.allow_target_updates = False
        trade.order_watch_only = True
        trade.target_trade = None
        state["state"] = "critical_reconciliation"
        state["critical_reason"] = str(cause)
        state["degraded_execution"] = True
        state.setdefault("degraded_reasons", []).append(
            "entry submission reconciliation required immediate recovery"
        )
        self.store.put(item.signal_id, state)
        try:
            fills = self._attributed_fills(connection, item.order_ref)
            remaining = owned_quantity(fills, item.direction)
            state["remaining_owned_shares"] = remaining
            if remaining < 0:
                self._mark_over_exit(
                    signal_id=item.signal_id,
                    connection=connection,
                    reference=item.order_ref,
                    state=state,
                    remaining=remaining,
                )
                return
            if remaining == 0:
                remaining, settled = self._settle_zero_owned(
                    trade,
                    terminal_state=(
                        "complete"
                        if entry_filled_quantity(fills, item.direction)
                        else "blocked"
                    ),
                )
                if settled:
                    managed.remove(trade)
                    return
                if remaining == 0:
                    return
            remaining = self._enforce_physical_isolation(
                signal_id=item.signal_id,
                connection=connection,
                context=item.context,
                reference=item.order_ref,
                direction=item.direction,
                remaining=remaining,
                state=state,
            )
            if remaining == 0:
                _, settled = self._settle_zero_owned(trade)
                if settled:
                    managed.remove(trade)
                return
            target = round_limit(trade.current_ema, trade.direction)
            try:
                proven = self._proven_exit_pair_with_reconnect(
                    connection=connection,
                    reference=item.order_ref,
                    direction=item.direction,
                    remaining=remaining,
                    expected_target=target,
                )
            except Exception:  # noqa: BLE001 - discard a poisoned broker snapshot
                self._restore_connection(connection)
                proven = self._proven_exit_pair(
                    connection=connection,
                    reference=item.order_ref,
                    direction=item.direction,
                    remaining=remaining,
                    expected_target=target,
                )
            if proven is None:
                target_trade, remaining = self._repair_protection(
                    signal_id=item.signal_id,
                    connection=connection,
                    context=item.context,
                    direction=item.direction,
                    reference=item.order_ref,
                    remaining=remaining,
                    target=target,
                    state=state,
                )
            else:
                target_trade = proven[0]
            state["state"] = "open_reconciled_after_error"
            state["remaining_owned_shares"] = remaining
            state["reconciled_after_error_at"] = pd.Timestamp.now(
                tz="UTC"
            ).isoformat()
            self.store.put(item.signal_id, state)
            trade.target_trade = target_trade
            trade.order_watch_only = False
        except Exception as recovery_exc:  # noqa: BLE001 - immediate containment
            if self._contain_unprotected_trade(
                trade,
                reason=(
                    f"submission reconciliation failed ({type(cause).__name__}); "
                    f"recovery failed: {recovery_exc}"
                ),
            ):
                managed.remove(trade)

    def _reconcile_submissions(
        self, submissions: list[SubmittedOrder], managed: list[ManagedTrade]
    ) -> None:
        """Reconcile every transmitted bracket, isolating failures per lot."""

        # One bounded grace period for the whole batch, never one sleep per
        # root. All entry brackets have already been transmitted at this point.
        if submissions:
            self._wait_management_until(
                pd.Timestamp.now(tz=NY_TZ) + pd.Timedelta(seconds=0.4)
            )
        # Cancel every known parent handle before any execution/open-order
        # request. Those broker snapshots can each time out; they must never
        # extend a later root's market-entry fill window.
        cancel_groups: dict[IBKRConnection, list[SubmittedOrder]] = {}
        for submission in submissions:
            if submission.trades is None:
                continue
            cancel_groups.setdefault(submission.item.connection, []).append(
                submission
            )
        for connection, group in cancel_groups.items():
            for submission in group:
                try:
                    connection.request_cancel_orders([submission.trades[0]])
                    submission.entry_cancel_requested = True
                except Exception as first_exc:  # noqa: BLE001 - isolate each parent
                    try:
                        self._restore_connection(connection)
                        connection.request_cancel_orders([submission.trades[0]])
                        submission.entry_cancel_requested = True
                    except Exception as retry_exc:  # noqa: BLE001 - peers still cancel
                        self.store.append_audit(
                            "entry_parent_cancel_request_failed",
                            signal_id=submission.item.signal_id,
                            first_error_type=type(first_exc).__name__,
                            retry_error_type=type(retry_exc).__name__,
                        )
        if any(item.entry_cancel_requested for item in submissions):
            self._wait_management_until(
                pd.Timestamp.now(tz=NY_TZ) + pd.Timedelta(seconds=0.5)
            )
        for submission in submissions:
            try:
                self._reconcile_submission(submission, managed)
            except Exception as exc:  # noqa: BLE001 - continue supervising peers
                self.store.append_audit(
                    "submission_reconcile_failed",
                    signal_id=submission.item.signal_id,
                    error_type=type(exc).__name__,
                )
                self._recover_failed_submission_reconciliation(
                    submission, managed, cause=exc
                )

    def _wait_management_until(self, target: pd.Timestamp) -> None:
        """Advance the IB event loop without depending on the market-data feed."""

        while True:
            self._touch_live_lease("active")
            remaining = (target - pd.Timestamp.now(tz=NY_TZ)).total_seconds()
            if remaining <= 0:
                return
            delay = min(1.0, remaining)
            waiters = [self.feed, *self.accounts.values()]
            advanced = False
            for waiter in waiters:
                if waiter is None:
                    continue
                try:
                    connected = getattr(waiter, "is_connected", lambda: True)()
                    if not connected:
                        continue
                    waiter.sleep(delay)
                    advanced = True
                    break
                except Exception:  # noqa: BLE001,S112 - try another client
                    continue
            if not advanced:
                time.sleep(delay)

    def _contain_unprotected_trade(
        self, trade: ManagedTrade, *, reason: str
    ) -> bool:
        """Emergency-flatten a proven virtual lot; retain the watch on failure."""

        trade.state["state"] = "critical_unprotected"
        trade.state["critical_reason"] = reason
        self.store.put(trade.signal_id, trade.state)
        trade.allow_target_updates = False
        trade.order_watch_only = True
        trade.target_trade = None
        try:
            self._emergency_flatten_owned_lot(
                signal_id=trade.signal_id,
                connection=trade.connection,
                context=trade.context,
                direction=trade.direction,
                reference=trade.order_ref,
                state=trade.state,
            )
        except Exception as exc:  # noqa: BLE001 - caller must retain the watch
            self.store.append_audit(
                "emergency_flatten_failed",
                signal_id=trade.signal_id,
                error_type=type(exc).__name__,
            )
            return False
        return True

    def _persist_target_revision_intents(
        self, intents: list[TargetRevisionIntent]
    ) -> None:
        """Durably prepare the whole revision batch before activation."""

        if not intents:
            return
        prepared_at = pd.Timestamp.now(tz="UTC").isoformat()
        updates: dict[str, dict[str, Any]] = {}
        for intent in intents:
            trade = intent.trade
            try:
                intent.prior_target = float(trade.target_trade.order.lmtPrice)
            except (AttributeError, TypeError, ValueError):
                intent.prior_target = round_limit(
                    trade.current_ema, trade.direction
                )
            trade.state["state"] = "exit_modifying"
            trade.state["target_revision_intent_at"] = prepared_at
            trade.state["pending_target_revision"] = {
                "activation": intent.activation_clock,
                "source_bar": intent.bar_clock,
                "ema": intent.new_ema,
                "limit": intent.target,
                "prior_limit": intent.prior_target,
                "remaining": intent.remaining,
                "intent_at": prepared_at,
                "phase": "prepared",
            }
            updates[trade.signal_id] = trade.state
        self.store.put_many(updates)

    def _execute_target_revision_batch(
        self,
        intents: list[TargetRevisionIntent],
        managed: list[ManagedTrade],
        critical: set[str],
    ) -> None:
        """Transmit all target prices first, then prove/repair every lot."""

        if intents:
            authorized_at = pd.Timestamp.now(tz="UTC").isoformat()
            updates: dict[str, dict[str, Any]] = {}
            for intent in intents:
                pending = intent.trade.state.get("pending_target_revision")
                if not isinstance(pending, dict) or pending.get("phase") != "prepared":
                    raise RuntimeError(
                        "target revision batch was not durably prepared before activation"
                    )
                pending["phase"] = "transmitting"
                pending["transmit_authorized_at"] = authorized_at
                updates[intent.trade.signal_id] = intent.trade.state
            # One atomic state fsync for the entire batch. From this boundary,
            # recovery treats every intent as potentially transmitted and
            # reconciles broker state rather than blindly retrying.
            self.store.put_many(updates)

        for intent in intents:
            trade = intent.trade
            try:
                activation = et_timestamp(
                    self.entry_date, intent.activation_clock
                ).tz_convert("UTC")
                transmit_deadline = activation + pd.Timedelta(seconds=1)
                if pd.Timestamp.now(tz="UTC") > transmit_deadline:
                    intent.deadline_missed = True
                    intent.transmit_error = (
                        "target revision activation deadline was missed; "
                        "prior broker-held protection was preserved"
                    )
                    trade.state["target_revision_deadline_missed_at"] = (
                        pd.Timestamp.now(tz="UTC").isoformat()
                    )
                    continue
                broker_remaining = round(
                    float(getattr(trade.target_trade.orderStatus, "remaining", 0) or 0)
                )
                if broker_remaining > 0:
                    intent.remaining = min(intent.remaining, broker_remaining)
                intent.updated_trade = trade.connection.request_target_modification(
                    trade.target_trade,
                    new_target=intent.target,
                    remaining=intent.remaining,
                )
                trade.state["target_revision_transmitted_at"] = pd.Timestamp.now(
                    tz="UTC"
                ).isoformat()
                transmitted = pd.Timestamp(
                    trade.state["target_revision_transmitted_at"]
                ).tz_convert("UTC")
                trade.state["target_revision_latency_ms"] = round(
                    max(0.0, (transmitted - activation).total_seconds() * 1_000),
                    3,
                )
            except Exception as exc:  # noqa: BLE001 - batch continues transmitting
                intent.transmit_error = str(exc)
        if intents:
            self._wait_management_until(
                pd.Timestamp.now(tz=NY_TZ) + pd.Timedelta(seconds=0.5)
            )

        for intent in intents:
            trade = intent.trade
            if trade not in managed:
                continue
            remaining = intent.remaining
            if intent.deadline_missed:
                trade.state["state"] = "open"
                trade.state["degraded_execution"] = True
                reason = str(intent.transmit_error)
                trade.state.setdefault("degraded_reasons", []).append(reason)
                trade.state.setdefault("target_revision_skips", []).append(
                    {
                        "activation": intent.activation_clock,
                        "source_bar": intent.bar_clock,
                        "reason": reason,
                    }
                )
                trade.state.pop("pending_target_revision", None)
                self.store.put(trade.signal_id, trade.state)
                continue
            try:
                if intent.updated_trade is None:
                    raise RuntimeError(
                        intent.transmit_error or "target revision transmit failed"
                    )
                trade.connection.confirm_target_modification(
                    intent.updated_trade,
                    expected_target=intent.target,
                    expected_con_id=self._expected_con_id(trade.order_ref),
                    order_ref=trade.order_ref,
                )
                fills = self._attributed_fills(
                    trade.connection, trade.order_ref
                )
                remaining = owned_quantity(fills, trade.direction)
                if remaining < 0:
                    self._mark_over_exit(
                        signal_id=trade.signal_id,
                        connection=trade.connection,
                        reference=trade.order_ref,
                        state=trade.state,
                        remaining=remaining,
                    )
                    raise RuntimeError("target revision over-exited the Legend lot")
                if remaining == 0:
                    remaining, settled = self._settle_zero_owned(trade)
                    if settled:
                        managed.remove(trade)
                        critical.discard(trade.signal_id)
                        continue
                    if remaining == 0:
                        critical.add(trade.signal_id)
                        continue
                remaining = self._enforce_physical_isolation(
                    signal_id=trade.signal_id,
                    connection=trade.connection,
                    context=trade.context,
                    reference=trade.order_ref,
                    direction=trade.direction,
                    remaining=remaining,
                    state=trade.state,
                )
                if remaining == 0:
                    _, settled = self._settle_zero_owned(trade)
                    if settled:
                        managed.remove(trade)
                        critical.discard(trade.signal_id)
                    continue
                proven = self._proven_exit_pair(
                    connection=trade.connection,
                    reference=trade.order_ref,
                    direction=trade.direction,
                    remaining=remaining,
                    expected_target=intent.target,
                )
                if proven is None:
                    raise RuntimeError(
                        "target revision left no exact broker-proven exit pair"
                    )
                trade.target_trade = proven[0]
                trade.state["target_revision_acknowledged_at"] = pd.Timestamp.now(
                    tz="UTC"
                ).isoformat()
            except Exception as modify_exc:  # noqa: BLE001 - repair ambiguous modify
                trade.state["target_modify_error"] = str(modify_exc)
                try:
                    activation = et_timestamp(
                        self.entry_date, intent.activation_clock
                    ).tz_convert("UTC")
                    deadline = activation + pd.Timedelta(seconds=1)
                    if not trade.connection.is_connected():
                        self._restore_connection(trade.connection)
                    fills = self._attributed_fills(
                        trade.connection, trade.order_ref
                    )
                    remaining = owned_quantity(fills, trade.direction)
                    if remaining < 0:
                        self._mark_over_exit(
                            signal_id=trade.signal_id,
                            connection=trade.connection,
                            reference=trade.order_ref,
                            state=trade.state,
                            remaining=remaining,
                        )
                        raise RuntimeError(
                            "target revision reconciliation found an over-exit"
                        )
                    if remaining == 0:
                        _, settled = self._settle_zero_owned(trade)
                        if settled:
                            managed.remove(trade)
                            critical.discard(trade.signal_id)
                        continue
                    # First reconcile the exact broker pair. A transport error
                    # may mean the requested target is already live; blindly
                    # cancelling it would create a new protection gap.
                    proven = self._proven_exit_pair_with_reconnect(
                        connection=trade.connection,
                        reference=trade.order_ref,
                        direction=trade.direction,
                        remaining=remaining,
                        expected_target=None,
                    )
                    actual_target = (
                        None
                        if proven is None
                        else float(proven[0].order.lmtPrice)
                    )
                    if actual_target is not None and math.isclose(
                        actual_target,
                        intent.target,
                        rel_tol=0.0,
                        abs_tol=1e-7,
                    ):
                        trade.target_trade = proven[0]
                        trade.state["target_revision_reconciled_at"] = (
                            pd.Timestamp.now(tz="UTC").isoformat()
                        )
                    else:
                        deadline_elapsed = pd.Timestamp.now(tz="UTC") > deadline
                        prior_target = float(intent.prior_target)
                        if (
                            deadline_elapsed
                            and actual_target is not None
                            and math.isclose(
                                actual_target,
                                prior_target,
                                rel_tol=0.0,
                                abs_tol=1e-7,
                            )
                        ):
                            trade.target_trade = proven[0]
                        else:
                            repair_target = (
                                prior_target if deadline_elapsed else intent.target
                            )
                            trade.target_trade, remaining = self._repair_protection(
                                signal_id=trade.signal_id,
                                connection=trade.connection,
                                context=trade.context,
                                direction=trade.direction,
                                reference=trade.order_ref,
                                remaining=remaining,
                                target=repair_target,
                                state=trade.state,
                            )
                            trade.state["target_revision_reprotected"] = True
                        if deadline_elapsed:
                            intent.deadline_missed = True
                            intent.transmit_error = (
                                "target revision could not be proven before the "
                                "activation deadline; prior protection was preserved"
                            )
                            trade.state["state"] = "open"
                            trade.state["degraded_execution"] = True
                            trade.state.setdefault("degraded_reasons", []).append(
                                intent.transmit_error
                            )
                            trade.state.setdefault(
                                "target_revision_skips", []
                            ).append(
                                {
                                    "activation": intent.activation_clock,
                                    "source_bar": intent.bar_clock,
                                    "reason": intent.transmit_error,
                                }
                            )
                            trade.state.pop("pending_target_revision", None)
                            self.store.put(trade.signal_id, trade.state)
                            continue
                except Exception as repair_exc:  # noqa: BLE001 - contain
                    if self._contain_unprotected_trade(
                        trade, reason=str(repair_exc)
                    ):
                        managed.remove(trade)
                        critical.discard(trade.signal_id)
                    else:
                        critical.add(trade.signal_id)
                    continue
            trade.current_ema = intent.new_ema
            trade.state["state"] = "open"
            trade.state.setdefault("target_revisions", []).append(
                {
                    "activation": intent.activation_clock,
                    "source_bar": intent.bar_clock,
                    "ema": intent.new_ema,
                    "limit": intent.target,
                    "remaining": remaining,
                    "transmitted_at": trade.state.get(
                        "target_revision_transmitted_at"
                    ),
                    "acknowledged_at": trade.state.get(
                        "target_revision_acknowledged_at"
                    ),
                }
            )
            trade.state.pop("pending_target_revision", None)
            self.store.put(trade.signal_id, trade.state)
            critical.discard(trade.signal_id)

    def _final_account_snapshots(
        self, trades: list[ManagedTrade]
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Snapshot each account once, isolating a poisoned endpoint.

        This intentionally does not reconnect inside the fast cutoff phase.
        A failed account is retried only after every healthy account has had
        any required emergency exit transmitted.
        """

        groups: dict[IBKRConnection, list[ManagedTrade]] = {}
        for trade in trades:
            groups.setdefault(trade.connection, []).append(trade)
        snapshots: dict[str, Any] = {}
        errors: dict[str, str] = {}
        for connection, group in groups.items():
            expected = {
                trade.order_ref: self._expected_con_id(trade.order_ref)
                for trade in group
            }
            try:
                snapshotter = getattr(connection, "legend_exit_snapshot", None)
                if snapshotter is None:
                    raise RuntimeError(
                        "execution adapter lacks account-wide final snapshot"
                    )
                snapshot = snapshotter(
                    expected, request_timeout_seconds=0.3
                )
                for trade in group:
                    if trade.order_ref not in snapshot.fills_by_reference:
                        raise RuntimeError(
                            f"final snapshot omitted {trade.order_ref}"
                        )
                    snapshots[trade.signal_id] = snapshot
            except Exception as exc:  # noqa: BLE001 - later accounts must proceed
                detail = f"{type(exc).__name__}: {exc}"
                for trade in group:
                    errors[trade.signal_id] = detail
        return snapshots, errors

    @staticmethod
    def _is_exact_immediate_market_exit(
        order_trade: Any, trade: ManagedTrade, remaining: int
    ) -> bool:
        order = order_trade.order
        expected_action = "SELL" if trade.direction > 0 else "BUY"
        return (
            remaining > 0
            and str(order_trade.orderStatus.status or "")
            in {"PreSubmitted", "Submitted"}
            and str(order.orderType or "").upper() == "MKT"
            and not str(order.goodAfterTime or "").strip()
            and str(order.action or "").upper() == expected_action
            and str(order.account or "") == trade.connection.endpoint.account
            and str(order.orderRef or "") == trade.order_ref
            and int(getattr(order, "clientId", -1))
            == trade.connection.endpoint.client_id
            and int(getattr(order_trade.contract, "conId", 0) or 0)
            == int(getattr(trade.context.contract, "conId", 0) or 0)
            and int(getattr(order, "permId", 0) or 0) > 0
            and round(
                float(getattr(order_trade.orderStatus, "remaining", 0) or 0)
            )
            == remaining
        )

    def _transmit_final_exit_batch(
        self, managed: list[ManagedTrade], critical: set[str]
    ) -> set[str]:
        """Classify all lots and transmit safe residual exits before slow proof."""

        snapshots, snapshot_errors = self._final_account_snapshots(list(managed))
        cancel_by_connection: dict[IBKRConnection, list[Any]] = {}
        state_updates: dict[str, dict[str, Any]] = {}
        for trade in list(managed):
            snapshot = snapshots.get(trade.signal_id)
            if snapshot is None:
                trade.state["state"] = "critical_connection"
                trade.state["critical_reason"] = (
                    "account-wide 10:30 snapshot failed: "
                    + snapshot_errors.get(trade.signal_id, "unknown snapshot failure")
                )
                trade.state["final_exit_fast_snapshot_failed"] = True
                state_updates[trade.signal_id] = trade.state
                critical.add(trade.signal_id)
                continue
            fills = snapshot.fills_by_reference[trade.order_ref]
            remaining = owned_quantity(fills, trade.direction)
            trade.state["remaining_owned_shares"] = remaining
            trade.state["final_exit_snapshot_at"] = pd.Timestamp.now(
                tz="UTC"
            ).isoformat()
            working = list(
                snapshot.working_orders_by_reference.get(trade.order_ref, [])
            )
            immediate = [
                item
                for item in working
                if self._is_exact_immediate_market_exit(item, trade, remaining)
            ]
            # Preserve an already-working immediate emergency market exit for
            # a positive lot. Everything else is cancelled in one account
            # batch, including any stale target/GAT order for a zero/over-exit.
            if len(immediate) > 1:
                # Duplicate exact MKT exits can both fill while the next
                # account snapshot is running. Persist the block and cancel
                # every duplicate before any refresh or peer replacement.
                cancellable = working
                trade.state["final_exit_fast_blocked"] = True
                trade.state["final_exit_duplicate_detected"] = True
                trade.state["critical_reason"] = (
                    "duplicate immediate market exits detected in first final snapshot"
                )
                critical.add(trade.signal_id)
            else:
                cancellable = [
                    item
                    for item in working
                    if remaining <= 0 or item not in immediate
                ]
            if cancellable:
                trade.state["state"] = "final_exit_preparing"
                trade.state["final_exit_cancel_intent_at"] = pd.Timestamp.now(
                    tz="UTC"
                ).isoformat()
                cancel_by_connection.setdefault(trade.connection, []).extend(
                    cancellable
                )
            state_updates[trade.signal_id] = trade.state
        if state_updates:
            self.store.put_many(state_updates)

        for connection, orders in cancel_by_connection.items():
            try:
                connection.request_cancel_orders(orders)
            except Exception as exc:  # noqa: BLE001 - peers still cancel/exit
                detail = f"{type(exc).__name__}: {exc}"
                self.store.append_audit(
                    "final_exit_cancel_request_failed",
                    account_label=connection.endpoint.label,
                    detail=detail,
                )
                for trade in managed:
                    if trade.connection is connection:
                        trade.state["final_exit_cancel_request_error"] = detail
        if cancel_by_connection:
            self._wait_management_until(
                pd.Timestamp.now(tz=NY_TZ) + pd.Timedelta(seconds=0.5)
            )

        # Re-read only accounts that survived the first snapshot. A target or
        # GAT can fill while cancellation is in flight, so the first quantities
        # are never used to size a new market order.
        refresh_trades = [
            trade for trade in managed if trade.signal_id in snapshots
        ]
        refreshed, refresh_errors = self._final_account_snapshots(refresh_trades)
        intents: list[FinalExitIntent] = []
        intent_updates: dict[str, dict[str, Any]] = {}
        unsafe_cancel_by_connection: dict[IBKRConnection, list[Any]] = {}
        for trade in refresh_trades:
            snapshot = refreshed.get(trade.signal_id)
            if snapshot is None:
                trade.state["state"] = "critical_connection"
                trade.state["critical_reason"] = (
                    "post-cancel 10:30 snapshot failed: "
                    + refresh_errors.get(trade.signal_id, "unknown snapshot failure")
                )
                trade.state["final_exit_fast_snapshot_failed"] = True
                intent_updates[trade.signal_id] = trade.state
                critical.add(trade.signal_id)
                continue
            fills = snapshot.fills_by_reference[trade.order_ref]
            remaining = owned_quantity(fills, trade.direction)
            trade.state["remaining_owned_shares"] = remaining
            working = list(
                snapshot.working_orders_by_reference.get(trade.order_ref, [])
            )
            con_id = self._expected_con_id(trade.order_ref)
            physical = float(
                snapshot.physical_positions_by_con_id.get(con_id, 0.0)
            )
            expected = float(trade.direction * max(remaining, 0))
            foreign = con_id in snapshot.foreign_working_con_ids
            if remaining <= 0:
                # Slow terminal/over-exit proof follows only after all positive
                # peers have received their containment mutations.
                intent_updates[trade.signal_id] = trade.state
                continue
            if trade.state.pop("final_exit_duplicate_detected", False):
                # The post-cancel snapshot is authoritative. Clear the initial
                # duplicate block only if the branch below now proves zero or
                # one exact residual exit.
                trade.state.pop("final_exit_fast_blocked", None)
            immediate = [
                item
                for item in working
                if self._is_exact_immediate_market_exit(item, trade, remaining)
            ]
            non_immediate = [item for item in working if item not in immediate]
            isolated = math.isclose(
                physical, expected, rel_tol=0.0, abs_tol=1e-7
            ) and not foreign
            if not isolated:
                trade.state["state"] = "critical_isolation"
                trade.state["critical_reason"] = (
                    "final exit blocked: account-symbol isolation changed before "
                    f"containment (physical={physical:g}, expected={expected:g}, "
                    f"foreign_orders={foreign})"
                )
                trade.state["final_exit_fast_blocked"] = True
                trade.state["isolation_actual_position"] = physical
                trade.state["isolation_expected_position"] = expected
                trade.state["isolation_foreign_working_orders"] = foreign
                unsafe_cancel_by_connection.setdefault(trade.connection, []).extend(
                    working
                )
                critical.add(trade.signal_id)
            elif len(immediate) > 1:
                trade.state["state"] = "critical_unflattened"
                trade.state["critical_reason"] = (
                    "multiple immediate Legend market exits are working; all "
                    "duplicates were queued for exact-reference cancellation"
                )
                trade.state["final_exit_fast_blocked"] = True
                unsafe_cancel_by_connection.setdefault(trade.connection, []).extend(
                    working
                )
                critical.add(trade.signal_id)
            elif non_immediate:
                # Preserve the one exact residual MKT, but immediately cancel
                # only the stale target/GAT legs that could fill alongside it.
                unsafe_cancel_by_connection.setdefault(trade.connection, []).extend(
                    non_immediate
                )
                trade.state["state"] = "time_exit_working"
                trade.state["critical_reason"] = (
                    "one exact market exit coexists with lingering target/GAT "
                    "legs; stale legs were queued for cancellation"
                )
                trade.state["final_exit_already_working"] = bool(immediate)
                trade.state["final_exit_lingering_cancel_pending"] = True
                critical.add(trade.signal_id)
            elif len(immediate) == 1:
                trade.state["state"] = "time_exit_working"
                trade.state["final_exit_already_working"] = True
            else:
                trade.state["state"] = "time_exit_working"
                trade.state["emergency_exit_intent_at"] = pd.Timestamp.now(
                    tz="UTC"
                ).isoformat()
                trade.state["emergency_exit_attempt"] = "10:30-batch"
                intents.append(FinalExitIntent(trade=trade, remaining=remaining))
            intent_updates[trade.signal_id] = trade.state
        if intent_updates:
            self.store.put_many(intent_updates)

        # Cancel any exact Legend order that could reverse a collided physical
        # net. These requests are issued before healthy peers' market exits,
        # without waiting for per-lot acknowledgement.
        unsafe_cancel_requested = False
        for connection, orders in unsafe_cancel_by_connection.items():
            try:
                connection.request_cancel_orders(orders)
                unsafe_cancel_requested = True
            except Exception as exc:  # noqa: BLE001 - persist during slow proof
                for intent_trade in refresh_trades:
                    if intent_trade.connection is connection:
                        intent_trade.state["final_exit_collision_cancel_error"] = (
                            f"{type(exc).__name__}: {exc}"
                        )

        # Every intent above is durably recorded before the first mutation.
        # Transmit them all without acknowledgement waits; exact proof follows.
        for intent in intents:
            trade = intent.trade
            try:
                intent.submitted_trade = trade.connection.place_emergency_exit(
                    contract=self._durable_context_contract(trade.context),
                    direction=trade.direction,
                    shares=intent.remaining,
                    order_ref=trade.order_ref,
                )
                trade.state["emergency_order_id"] = int(
                    intent.submitted_trade.order.orderId
                )
                trade.state["emergency_exit_submitted_at"] = pd.Timestamp.now(
                    tz="UTC"
                ).isoformat()
            except Exception as exc:  # noqa: BLE001 - later peers still transmit
                intent.submit_error = f"{type(exc).__name__}: {exc}"
                trade.state["emergency_exit_submit_error"] = intent.submit_error
                critical.add(trade.signal_id)
        if intents:
            self.store.put_many(
                {intent.trade.signal_id: intent.trade.state for intent in intents}
            )
            self._wait_management_until(
                pd.Timestamp.now(tz=NY_TZ) + pd.Timedelta(seconds=0.5)
            )
        elif unsafe_cancel_requested:
            self._wait_management_until(
                pd.Timestamp.now(tz=NY_TZ) + pd.Timedelta(seconds=0.3)
            )
        return {intent.trade.signal_id for intent in intents}

    def _manage_live(self, managed: list[ManagedTrade]) -> None:
        critical: set[str] = set()
        degraded: set[str] = {
            trade.signal_id
            for trade in managed
            if trade.state.get("degraded_execution")
        }
        updates = (("09:46", "09:30"), ("10:01", "09:45"), ("10:16", "10:00"))
        for activation_clock, bar_clock in updates:
            source_ready = et_timestamp(self.entry_date, bar_clock) + pd.Timedelta(
                minutes=15, seconds=2
            )
            self._wait_management_until(
                source_ready
            )
            feed_connected = bool(
                self.feed is not None
                and getattr(self.feed, "is_connected", lambda: True)()
            )
            revision_intents: list[TargetRevisionIntent] = []
            for trade in list(managed):
                try:
                    fills = self._attributed_fills(
                        trade.connection, trade.order_ref
                    )
                    remaining = owned_quantity(fills, trade.direction)
                    if remaining < 0:
                        self._mark_over_exit(
                            signal_id=trade.signal_id,
                            connection=trade.connection,
                            reference=trade.order_ref,
                            state=trade.state,
                            remaining=remaining,
                        )
                        critical.add(trade.signal_id)
                        continue
                    if remaining == 0:
                        remaining, settled = self._settle_zero_owned(trade)
                        if settled:
                            managed.remove(trade)
                            critical.discard(trade.signal_id)
                            continue
                        if remaining == 0:
                            critical.add(trade.signal_id)
                            continue

                    # A parent may fill while an ambiguous/flat record is being
                    # cancelled. Protect that newly positive virtual lot before
                    # considering any discretionary EMA revision.
                    target = round_limit(trade.current_ema, trade.direction)
                    try:
                        try:
                            proven = self._proven_exit_pair(
                                connection=trade.connection,
                                reference=trade.order_ref,
                                direction=trade.direction,
                                remaining=remaining,
                                expected_target=target,
                            )
                        except Exception:  # noqa: BLE001 - refresh poisoned snapshot
                            self._restore_connection(trade.connection)
                            proven = self._proven_exit_pair(
                                connection=trade.connection,
                                reference=trade.order_ref,
                                direction=trade.direction,
                                remaining=remaining,
                                expected_target=target,
                            )
                        if proven is None:
                            trade.target_trade, remaining = self._repair_protection(
                                signal_id=trade.signal_id,
                                connection=trade.connection,
                                context=trade.context,
                                direction=trade.direction,
                                reference=trade.order_ref,
                                remaining=remaining,
                                target=target,
                                state=trade.state,
                            )
                        else:
                            trade.target_trade = proven[0]
                        trade.order_watch_only = False
                        if trade.state.get("state") in {
                            "unknown",
                            "critical_connection",
                            "critical_reconciliation",
                            "critical_unprotected",
                        }:
                            trade.state["state"] = "open_repaired_late_fill"
                        self.store.put(trade.signal_id, trade.state)
                    except Exception as exc:  # noqa: BLE001 - emergency containment
                        if self._contain_unprotected_trade(
                            trade, reason=str(exc)
                        ):
                            managed.remove(trade)
                            critical.discard(trade.signal_id)
                        else:
                            critical.add(trade.signal_id)
                        continue

                    remaining = self._enforce_physical_isolation(
                        signal_id=trade.signal_id,
                        connection=trade.connection,
                        context=trade.context,
                        reference=trade.order_ref,
                        direction=trade.direction,
                        remaining=remaining,
                        state=trade.state,
                    )
                    if remaining == 0:
                        _, settled = self._settle_zero_owned(trade)
                        if settled:
                            managed.remove(trade)
                            critical.discard(trade.signal_id)
                        else:
                            critical.add(trade.signal_id)
                        continue

                    if not feed_connected:
                        if trade.allow_target_updates:
                            self.store.append_audit(
                                "target_updates_disabled",
                                signal_id=trade.signal_id,
                                reason="market-data feed disconnected",
                            )
                        trade.state["degraded_execution"] = True
                        trade.state.setdefault("degraded_reasons", []).append(
                            "market-data feed disconnected before EMA revision"
                        )
                        self.store.put(trade.signal_id, trade.state)
                        degraded.add(trade.signal_id)
                        trade.allow_target_updates = False
                    if not trade.allow_target_updates:
                        if not trade.state.get("degraded_execution"):
                            trade.state["degraded_execution"] = True
                            trade.state.setdefault("degraded_reasons", []).append(
                                "target revisions disabled after restart or recovery"
                            )
                            self.store.put(trade.signal_id, trade.state)
                            degraded.add(trade.signal_id)
                        self.store.append_audit(
                            "target_update_skipped",
                            signal_id=trade.signal_id,
                            reason="feed unavailable or restart preserves last target",
                        )
                        continue

                    frame = realtime_bars_to_frame(trade.context.realtime_bars)
                    try:
                        close = completed_fifteen_minute_close(
                            frame, et_timestamp(self.entry_date, bar_clock)
                        )
                    except ValueError as exc:
                        self.store.append_audit(
                            "target_update_skipped",
                            signal_id=trade.signal_id,
                            reason=str(exc),
                        )
                        trade.allow_target_updates = False
                        trade.state["degraded_execution"] = True
                        trade.state.setdefault("degraded_reasons", []).append(
                            str(exc)
                        )
                        self.store.put(trade.signal_id, trade.state)
                        degraded.add(trade.signal_id)
                        continue
                    new_ema = float(
                        ema_from_seed(
                            pd.Series([close]), trade.current_ema
                        ).iloc[0]
                    )
                    target = round_limit(new_ema, trade.direction)
                    revision_intents.append(
                        TargetRevisionIntent(
                            trade=trade,
                            activation_clock=activation_clock,
                            bar_clock=bar_clock,
                            remaining=remaining,
                            new_ema=new_ema,
                            target=target,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - isolate every managed lot
                    if trade.state.get("state") != "critical_over_exit":
                        trade.state["state"] = "critical_reconciliation"
                        trade.state["critical_reason"] = str(exc)
                        self.store.put(trade.signal_id, trade.state)
                    trade.allow_target_updates = False
                    critical.add(trade.signal_id)
            # Slow fill/order/position proof happens during the minute between
            # source-bar completion and activation. Persist the entire batch
            # now, then use one batch state transition at activation so disk
            # fsyncs cannot serialize the broker modifications.
            self._persist_target_revision_intents(revision_intents)
            self._wait_management_until(
                et_timestamp(self.entry_date, activation_clock)
            )
            self._execute_target_revision_batch(
                revision_intents, managed, critical
            )
            for intent in revision_intents:
                if intent.deadline_missed:
                    degraded.add(intent.trade.signal_id)
                    continue
                latency = intent.trade.state.get("target_revision_latency_ms")
                if latency is not None and float(latency) > 1_000:
                    intent.trade.state["degraded_execution"] = True
                    intent.trade.state.setdefault("degraded_reasons", []).append(
                        f"target revision {activation_clock} transmitted {latency}ms late"
                    )
                    self.store.put(intent.trade.signal_id, intent.trade.state)
                    degraded.add(intent.trade.signal_id)

        # The safety clock is independent of the market-data connection. Every
        # execution account is classified before any slow per-lot proof. Safe
        # residual market exits are transmitted as a batch first, so a poisoned
        # endpoint cannot delay a healthy peer's missing/rejected GAT fallback.
        # The reviewed broker proof permits the scheduled GAT sibling up to
        # five seconds to fill. Do not cancel/replace a still-valid order inside
        # that attested window; fallback begins immediately after it expires.
        self._wait_management_until(et_timestamp(self.entry_date, "10:30:05"))
        self._transmit_final_exit_batch(managed, critical)
        for trade in list(managed):
            try:
                fills = self._attributed_fills(
                    trade.connection, trade.order_ref
                )
                remaining = owned_quantity(fills, trade.direction)
                trade.state["remaining_owned_shares"] = remaining
                if remaining < 0:
                    self._mark_over_exit(
                        signal_id=trade.signal_id,
                        connection=trade.connection,
                        reference=trade.order_ref,
                        state=trade.state,
                        remaining=remaining,
                    )
                    critical.add(trade.signal_id)
                    continue
                if remaining == 0:
                    remaining, settled = self._settle_zero_owned(trade)
                    if settled:
                        managed.remove(trade)
                        critical.discard(trade.signal_id)
                        continue
                    if remaining == 0:
                        critical.add(trade.signal_id)
                        continue
                if trade.state.get("final_exit_fast_blocked"):
                    try:
                        remaining = self._enforce_physical_isolation(
                            signal_id=trade.signal_id,
                            connection=trade.connection,
                            context=trade.context,
                            reference=trade.order_ref,
                            direction=trade.direction,
                            remaining=remaining,
                            state=trade.state,
                        )
                    except Exception as isolation_exc:  # noqa: BLE001 - contain
                        # Healthy-peer containment has already been transmitted.
                        # Now prove/retry cancellation of only this collided
                        # Legend reference; never flatten the netted symbol.
                        try:
                            cleared = self._clear_reference_orders(
                                trade.connection, trade.order_ref
                            )
                        except Exception as cancel_exc:  # noqa: BLE001 - persist
                            cleared = False
                            trade.state["final_exit_collision_cancel_error"] = (
                                f"{type(cancel_exc).__name__}: {cancel_exc}"
                            )
                        trade.state["working_orders_cleared"] = cleared
                        trade.state["critical_reason"] = (
                            "fresh final isolation proof failed; exact Legend "
                            f"orders cancellation_proved={cleared}: {isolation_exc}"
                        )
                        trade.state["remaining_owned_shares"] = remaining
                        self.store.put(trade.signal_id, trade.state)
                        critical.add(trade.signal_id)
                        continue
                    trade.state.pop("final_exit_fast_blocked", None)
                trade.state["state"] = "critical_unflattened"
                trade.state["critical_reason"] = (
                    "broker-held 10:30 exit did not prove the Legend lot flat"
                )
                self.store.put(trade.signal_id, trade.state)
                if self._contain_unprotected_trade(
                    trade, reason=trade.state["critical_reason"]
                ):
                    managed.remove(trade)
                    critical.discard(trade.signal_id)
                else:
                    critical.add(trade.signal_id)
            except Exception as exc:  # noqa: BLE001 - always continue final pass
                if trade.state.get("state") != "critical_over_exit":
                    trade.state["state"] = "critical_reconciliation"
                    trade.state["critical_reason"] = str(exc)
                    self.store.put(trade.signal_id, trade.state)
                critical.add(trade.signal_id)
        if critical or degraded:
            details: list[str] = []
            if critical:
                details.append("critical=" + ",".join(sorted(critical)))
            if degraded:
                details.append("degraded=" + ",".join(sorted(degraded)))
            raise RuntimeError(
                "Legend live session did not complete cleanly: " + "; ".join(details)
            )

    def _manage_shadow(self, eligible: list[MarketContext]) -> None:
        """Exercise every causal update clock without any broker mutation."""

        assert self.feed is not None
        current_ema = {context.root: context.initial_ema for context in eligible}
        causal = {context.root: True for context in eligible}
        updates = (("09:46", "09:30"), ("10:01", "09:45"), ("10:16", "10:00"))
        for activation_clock, bar_clock in updates:
            _wait_until(self.feed, et_timestamp(self.entry_date, activation_clock))
            for context in eligible:
                if not causal[context.root]:
                    continue
                frame = realtime_bars_to_frame(context.realtime_bars)
                try:
                    close = completed_fifteen_minute_close(
                        frame, et_timestamp(self.entry_date, bar_clock)
                    )
                except ValueError as exc:
                    causal[context.root] = False
                    self.store.append_audit(
                        "shadow_target_update_failed",
                        root=context.root,
                        activation=activation_clock,
                        reason=str(exc),
                    )
                    continue
                ema = float(
                    ema_from_seed(pd.Series([close]), current_ema[context.root]).iloc[0]
                )
                current_ema[context.root] = ema
                self.store.append_audit(
                    "shadow_target_revision",
                    root=context.root,
                    activation=activation_clock,
                    source_bar=bar_clock,
                    ema=ema,
                    limit=round_limit(ema, int(context.direction)),
                )
        _wait_until(self.feed, et_timestamp(self.entry_date, "10:30:15"))
        for context in eligible:
            frame = realtime_bars_to_frame(context.realtime_bars)
            cutoff = et_timestamp(self.entry_date, "10:30").tz_convert("UTC")
            first_cutoff_bar = frame.loc[cutoff, "open"] if cutoff in frame.index else None
            self.store.append_audit(
                "shadow_time_exit_observed",
                root=context.root,
                causal_updates_complete=causal[context.root],
                first_1030_trade=first_cutoff_bar,
            )

    def run(self) -> dict[str, Any]:
        # A lock loser must never overwrite the real owner's lease. Heartbeats
        # and the final state are therefore written only while this process
        # owns the cross-worktree machine mutex.
        with self.store.exclusive_session():
            self._touch_live_lease("active")
            try:
                result = self._run_locked()
            except BaseException:
                self._touch_live_lease("failed")
                raise
            self._touch_live_lease("finished")
            return result

    def _run_locked(self) -> dict[str, Any]:
        audit_error: Exception | None = None
        if self.live_requested:
            try:
                self.audit_terminal_corrections()
            except Exception as exc:  # noqa: BLE001 - peers still need recovery
                audit_error = exc
                detail = (
                    "startup terminal-correction audit failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                if detail not in self.recovery_degraded_reasons:
                    self.recovery_degraded_reasons.append(detail)
        recovery_found, recovered = self._bootstrap_recovery()
        if recovery_found:
            if recovered:
                self._manage_live(recovered)
            if self.recovery_degraded_reasons:
                raise RuntimeError(
                    "Legend recovery completed broker reconciliation but runtime "
                    "attestation is degraded: "
                    + "; ".join(self.recovery_degraded_reasons)
                )
            return {
                "ok": True,
                "live": True,
                "entry_date": self.entry_date,
                "eligible": [],
                "managed_live": len(recovered),
                "recovery_only": True,
                "state_path": str(self.store.state_path),
            }
        if audit_error is not None:
            raise RuntimeError(
                "Legend startup correction audit failed before new-entry gates"
            ) from audit_error
        if self.reconcile_only:
            return {
                "ok": True,
                "live": self.live_requested,
                "entry_date": self.entry_date,
                "eligible": [],
                "managed_live": 0,
                "recovery_only": True,
                "detail": "no same-day broker-mutated Legend records",
                "state_path": str(self.store.state_path),
            }
        deadline = et_timestamp(self.entry_date, "09:31:20")
        if pd.Timestamp.now(tz=NY_TZ) > deadline:
            raise RuntimeError(
                "late start: new Legend entries are prohibited after 09:31:20"
            )
        summary = self.preflight(connect=True)
        if not summary["qualified"]:
            return summary
        assert self.feed is not None
        self._load_market_contexts()
        self._acquire_symbol_reservations()
        for context in self.contexts:
            context.realtime_bars = self.feed.subscribe_realtime_bars(context.contract)
            context.trade_ticks = self.feed.subscribe_trade_ticks(context.contract)
        try:
            managed = self._reconcile_prior_records()
            self._preflight_accounts_before_decision()
            if pd.Timestamp.now(tz=NY_TZ) > deadline:
                raise RuntimeError(
                    "slow pre-entry validation missed the 09:31:20 entry deadline"
                )
            self.feed.assert_server_clock()
            _wait_until(self.feed, et_timestamp(self.entry_date, "09:31"))
            _wait_for_decision_bars(self.feed, self.contexts, self.entry_date)
            eligible = self._decide()
            # Validate every selected account/root before the first live order.
            prepared_batches = [
                self._prepare_account(label, eligible)
                for label in self.account_labels
            ]
            try:
                prepared = [
                    item for account_batch in prepared_batches for item in account_batch
                ]
                with self._reserve_portfolio_batch(prepared):
                    submissions = self._transmit_batch(prepared, managed)
                self._reconcile_submissions(submissions, managed)
            except Exception as submit_exc:
                self.store.append_audit(
                    "submit_phase_failed",
                    error_type=type(submit_exc).__name__,
                    managed_signal_ids=[trade.signal_id for trade in managed],
                )
                # Already-transmitted positions retain broker-held exits. Keep
                # managing every proven-safe lot through the 10:30 deadline
                # before surfacing the submission failure.
                if self.gate and self.gate.live and managed:
                    try:
                        self._manage_live(managed)
                    except Exception as manage_exc:
                        raise RuntimeError(
                            "Legend submission failed and one or more prior "
                            "submitted lots also failed reconciliation"
                        ) from manage_exc
                raise
            if self.gate and self.gate.live:
                self._manage_live(managed)
            elif self.shadow_through_exit:
                self._manage_shadow(eligible)
            return {
                **summary,
                "eligible": [context.root for context in eligible],
                "managed_live": len(managed),
                "state_path": str(self.store.state_path),
            }
        finally:
            for context in self.contexts:
                if context.realtime_bars is not None:
                    try:
                        self.feed.cancel_realtime_bars(context.realtime_bars)
                    except Exception as exc:  # noqa: BLE001 - cleanup must not mask run
                        self.store.append_audit(
                            "realtime_cancel_failed",
                            root=context.root,
                            error_type=type(exc).__name__,
                        )
                if context.trade_ticks is not None:
                    try:
                        self.feed.cancel_trade_ticks(context.contract)
                    except Exception as exc:  # noqa: BLE001 - cleanup must not mask run
                        self.store.append_audit(
                            "trade_tick_cancel_failed",
                            root=context.root,
                            error_type=type(exc).__name__,
                        )
            self.close()

    def close(self) -> None:
        for connection in self.accounts.values():
            try:
                connection.disconnect()
            except Exception as exc:  # noqa: BLE001 - best-effort disconnect
                self.store.append_audit(
                    "account_disconnect_failed",
                    account_label=connection.endpoint.label,
                    error_type=type(exc).__name__,
                )
        if self.feed is not None:
            try:
                self.feed.disconnect()
            except Exception as exc:  # noqa: BLE001 - best-effort disconnect
                self.store.append_audit(
                    "feed_disconnect_failed", error_type=type(exc).__name__
                )
        if self.reservations is not None:
            try:
                self.reservations.close()
            except Exception as exc:  # noqa: BLE001 - audit cleanup failure
                self.store.append_audit(
                    "reservation_release_failed", error_type=type(exc).__name__
                )
            finally:
                self.reservations = None
