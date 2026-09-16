"""Dedicated IBKR adapter for the Legend ETF sleeve.

This module is intentionally not wired to the generic execution bridge.  A
dry-run connection is read-only; a live connection is possible only after the
independent, dated Legend gate passes.
"""

from __future__ import annotations

import asyncio
import copy
import math
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from types import SimpleNamespace
from typing import Any

import pandas as pd

from .config import NY_TZ, STRATEGY_NAME, STRATEGY_VERSION

WORKING_STATUSES = {
    "PendingSubmit",
    "PreSubmitted",
    "Submitted",
    "PendingCancel",
}
DEAD_STATUSES = {"Cancelled", "ApiCancelled", "Inactive", "Filled"}
REJECTED_STATUSES = {"ApiCancelled", "Inactive"}
ACKNOWLEDGED_STATUSES = {
    "PreSubmitted",
    "Submitted",
    "PendingCancel",
    "Cancelled",
    "Filled",
}
_RAW_OPEN_ORDER_LOCK = threading.RLock()


class PreTransmitCheckBlocked(RuntimeError):
    """Final causal proof blocked after staged legs were proven cancelled."""


class PhysicalIsolationMismatch(RuntimeError):
    """A fresh broker snapshot proved a physical/foreign-symbol mismatch."""

    def __init__(
        self,
        message: str,
        *,
        actual_position: float,
        expected_position: float,
        foreign_working_orders: bool = False,
    ) -> None:
        super().__init__(message)
        self.actual_position = float(actual_position)
        self.expected_position = float(expected_position)
        self.foreign_working_orders = bool(foreign_working_orders)

    @property
    def legend_exits_can_reverse_net_position(self) -> bool:
        """Whether exiting the virtual lot could cross the physical net through zero."""

        expected = self.expected_position
        actual = self.actual_position
        if self.foreign_working_orders:
            return True
        if math.isclose(expected, 0.0, rel_tol=0.0, abs_tol=1e-7):
            return not math.isclose(actual, 0.0, rel_tol=0.0, abs_tol=1e-7)
        return actual * expected <= 0 or abs(actual) + 1e-7 < abs(expected)


BROKER_REQUEST_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class Endpoint:
    label: str
    host: str
    port: int
    client_id: int
    account: str


@dataclass(frozen=True)
class LiveGate:
    live: bool
    allow_longs: bool
    allow_shorts: bool
    allowed_accounts: frozenset[str]

    def require_side(self, direction: int) -> None:
        if direction > 0 and not self.allow_longs:
            raise RuntimeError("Legend long entries are not enabled")
        if direction < 0 and not self.allow_shorts:
            raise RuntimeError("Legend short entries are not enabled")


def _truthy_from(environment: Mapping[str, str], name: str) -> bool:
    return str(environment.get(name, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def validate_live_gate(
    *,
    live_requested: bool,
    account_ids: Iterable[str],
    today: date,
    environment: Mapping[str, str] | None = None,
) -> LiveGate:
    """Validate a daily-expiring live gate; dry-run needs no environment flags."""

    requested = frozenset(str(account).strip().upper() for account in account_ids)
    if not requested or "" in requested:
        raise ValueError("at least one exact IBKR account ID is required")
    if not live_requested:
        return LiveGate(False, True, True, requested)
    values = os.environ if environment is None else environment
    errors: list[str] = []
    if not _truthy_from(values, "LEGEND_ETF_LIVE_ENABLED"):
        errors.append("LEGEND_ETF_LIVE_ENABLED is not 1")
    if str(values.get("LEGEND_ETF_LIVE_DATE", "")).strip() != today.isoformat():
        errors.append("LEGEND_ETF_LIVE_DATE must equal today's New York date")
    if str(values.get("LEGEND_ETF_STRATEGY_VERSION", "")).strip() != STRATEGY_VERSION:
        errors.append("LEGEND_ETF_STRATEGY_VERSION does not match this build")
    allowed = frozenset(
        value.strip().upper()
        for value in str(values.get("LEGEND_ETF_LIVE_ACCOUNTS", "")).split(",")
        if value.strip()
    )
    if not requested.issubset(allowed):
        errors.append(
            "requested exact IBKR account ID is absent from "
            "LEGEND_ETF_LIVE_ACCOUNTS"
        )
    allow_longs = _truthy_from(values, "LEGEND_ETF_ALLOW_LONGS")
    allow_shorts = _truthy_from(values, "LEGEND_ETF_ALLOW_SHORTS")
    if not allow_longs and not allow_shorts:
        errors.append("neither Legend longs nor shorts are enabled")
    if errors:
        raise RuntimeError("Live gate refused: " + "; ".join(errors))
    return LiveGate(True, allow_longs, allow_shorts, allowed)


def validate_recovery_gate(
    *,
    account_ids: Iterable[str],
    today: date,
    environment: Mapping[str, str] | None = None,
) -> LiveGate:
    """Authorize management of existing lots without authorizing new entries.

    The operator's halt switch and side switches are entry controls. Turning
    them off must not strand a previously submitted lot after a process restart.
    Recovery still requires the dated build identity and exact account allowlist.
    """

    requested = frozenset(str(account).strip().upper() for account in account_ids)
    if not requested or "" in requested:
        raise ValueError("at least one exact IBKR account ID is required")
    values = os.environ if environment is None else environment
    errors: list[str] = []
    if str(values.get("LEGEND_ETF_LIVE_DATE", "")).strip() != today.isoformat():
        errors.append("LEGEND_ETF_LIVE_DATE must equal today's New York date")
    if str(values.get("LEGEND_ETF_STRATEGY_VERSION", "")).strip() != STRATEGY_VERSION:
        errors.append("LEGEND_ETF_STRATEGY_VERSION does not match this build")
    allowed = frozenset(
        value.strip().upper()
        for value in str(values.get("LEGEND_ETF_LIVE_ACCOUNTS", "")).split(",")
        if value.strip()
    )
    if not requested.issubset(allowed):
        errors.append(
            "recovery account is absent from LEGEND_ETF_LIVE_ACCOUNTS"
        )
    if errors:
        raise RuntimeError("Recovery gate refused: " + "; ".join(errors))
    # Both side permissions stay false: this gate may manage/exit only and is
    # never valid for the new-entry path, which calls require_side().
    return LiveGate(True, False, False, allowed)


def validate_correction_audit_gate(
    *,
    account_ids: Iterable[str],
    environment: Mapping[str, str] | None = None,
) -> LiveGate:
    """Authorize read-only all-ETF residue checks while entry is halted.

    This intentionally ignores the live/date/side switches: a next-session
    broker correction must remain detectable after the operator disarms entry.
    It authorizes no order mutation and requires the exact build identity and
    account allowlist.
    """

    requested = frozenset(str(account).strip().upper() for account in account_ids)
    if not requested or "" in requested:
        raise ValueError("at least one exact IBKR account ID is required")
    values = os.environ if environment is None else environment
    errors: list[str] = []
    if str(values.get("LEGEND_ETF_STRATEGY_VERSION", "")).strip() != STRATEGY_VERSION:
        errors.append("LEGEND_ETF_STRATEGY_VERSION does not match this build")
    allowed = frozenset(
        value.strip().upper()
        for value in str(values.get("LEGEND_ETF_LIVE_ACCOUNTS", "")).split(",")
        if value.strip()
    )
    if not requested.issubset(allowed):
        errors.append("audit account is absent from LEGEND_ETF_LIVE_ACCOUNTS")
    if errors:
        raise RuntimeError("Correction audit gate refused: " + "; ".join(errors))
    return LiveGate(True, False, False, allowed)


def endpoints_from_env(
    labels: Iterable[str], *, environment: Mapping[str, str] | None = None
) -> list[Endpoint]:
    defaults = {
        "primary": (7496, 155),
        "pa": (4001, 156),
    }
    endpoints: list[Endpoint] = []
    values = os.environ if environment is None else environment
    normalized_labels = [str(raw_label).lower() for raw_label in labels]
    if len(normalized_labels) != len(set(normalized_labels)):
        raise ValueError("account labels must be unique")
    for label in normalized_labels:
        if label not in defaults:
            raise ValueError(f"unknown account label {label!r}")
        prefix = f"LEGEND_ETF_{label.upper()}"
        account = str(values.get(f"{prefix}_ACCOUNT", "")).strip()
        if not account:
            raise RuntimeError(f"{prefix}_ACCOUNT must name the exact IBKR account")
        default_port, default_client = defaults[label]
        endpoints.append(
            Endpoint(
                label=label,
                host=str(values.get(f"{prefix}_HOST", "127.0.0.1")).strip(),
                port=int(values.get(f"{prefix}_PORT", str(default_port))),
                client_id=int(
                    values.get(f"{prefix}_CLIENT_ID", str(default_client))
                ),
                account=account,
            )
        )
    identities = [(item.host, item.port, item.client_id) for item in endpoints]
    if len(identities) != len(set(identities)):
        raise ValueError("each account needs a unique IBKR host/port/client ID")
    accounts = [item.account.upper() for item in endpoints]
    if len(accounts) != len(set(accounts)):
        raise ValueError("each execution label must resolve to a different IBKR account")
    return endpoints


def build_order_ref(symbol: str, direction: int, entry_date: str) -> str:
    action = "BUY" if direction > 0 else "SELL"
    reference = f"{symbol.upper()}|{action}|{STRATEGY_NAME}|{entry_date}"
    if len(reference) > 128 or "|" in STRATEGY_NAME:
        raise ValueError("invalid Legend orderRef")
    return reference


@dataclass(frozen=True)
class FillEvent:
    exec_id: str
    order_id: int
    action: str
    shares: int
    price: float
    timestamp: str
    client_id: int = 0
    con_id: int = 0


@dataclass
class LegendAccountSnapshot:
    """One bounded account-wide snapshot for the final exit boundary."""

    fills_by_reference: dict[str, list[FillEvent]]
    working_orders_by_reference: dict[str, list[Any]]
    physical_positions_by_con_id: dict[int, float]
    foreign_working_con_ids: frozenset[int]


def owned_quantity(fills: Iterable[FillEvent], entry_direction: int) -> int:
    """Return the signed virtual Legend lot, idempotent to duplicate callbacks.

    A negative value is an over-exit/reversal and must never be called flat.
    """

    # IBKR corrections retain the execution-ID stem and increment the final
    # numeric revision. Count only the newest revision of each execution.
    unique: dict[str, tuple[int, FillEvent]] = {}
    for fill in fills:
        stem, separator, revision = fill.exec_id.rpartition(".")
        if separator and revision.isdigit():
            key = stem
            rank = int(revision)
        else:
            key = fill.exec_id
            rank = 0
        current = unique.get(key)
        if current is None or rank >= current[0]:
            unique[key] = (rank, fill)
    entry_action = "BOT" if entry_direction > 0 else "SLD"
    quantity = 0
    for _, fill in unique.values():
        sign = 1 if fill.action.upper() == entry_action else -1
        quantity += sign * int(fill.shares)
    return quantity


def entry_filled_quantity(fills: Iterable[FillEvent], entry_direction: int) -> int:
    unique: dict[str, tuple[int, FillEvent]] = {}
    for fill in fills:
        stem, separator, revision = fill.exec_id.rpartition(".")
        if separator and revision.isdigit():
            key = stem
            rank = int(revision)
        else:
            key = fill.exec_id
            rank = 0
        current = unique.get(key)
        if current is None or rank >= current[0]:
            unique[key] = (rank, fill)
    entry_action = "BOT" if entry_direction > 0 else "SLD"
    return sum(
        int(fill.shares)
        for _, fill in unique.values()
        if fill.action.upper() == entry_action
    )


def _finite_price(value: Any) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if math.isfinite(price) and 0 < price < 1e9 else None


class TradeTickTape:
    """Retain Last ticks across ib_insync's per-packet Ticker resets.

    Never discard the opening tape to make room: overflow invalidates entry
    evidence. The bound also limits memory if a shadow runs through 10:30.
    """

    def __init__(self, *, max_ticks: int = 500_000):
        if max_ticks < 1:
            raise ValueError("trade tape capacity must be positive")
        self.max_ticks = max_ticks
        self.tickByTicks: list[Any] = []
        self.overflowed = False

    def capture(self, ticker: Any) -> None:
        batch = ticker.tickByTicks
        if self.overflowed or len(self.tickByTicks) + len(batch) > self.max_ticks:
            self.overflowed = True
            return
        self.tickByTicks.extend(batch)

    def assert_complete(self) -> None:
        if self.overflowed:
            raise RuntimeError("ETF Last-trade tape overflowed; opening evidence is incomplete")


class IBKRConnection:
    """Thin synchronous wrapper around one account-owned ib_insync client."""

    def __init__(self, endpoint: Endpoint, *, live: bool):
        self.endpoint = endpoint
        self.live = live
        self.ib: Any | None = None
        self._trade_tapes: dict[int, tuple[Any, TradeTickTape]] = {}

    def connect(self) -> None:
        try:
            from ib_insync import IB
        except ImportError as exc:  # pragma: no cover - production dependency guard
            raise RuntimeError("install requirements-legend-etf.txt") from exc
        ib = IB()
        # ib_insync otherwise defaults synchronous API requests to no timeout.
        # A hung request must not monopolize the only exit-management thread.
        ib.RequestTimeout = BROKER_REQUEST_TIMEOUT_SECONDS
        ib.RaiseRequestErrors = True
        ib.connect(
            self.endpoint.host,
            self.endpoint.port,
            clientId=self.endpoint.client_id,
            timeout=10,
            readonly=not self.live,
            account=self.endpoint.account,
        )
        managed = set(ib.managedAccounts())
        if self.endpoint.account not in managed:
            ib.disconnect()
            raise RuntimeError(
                f"{self.endpoint.label}: configured account is not managed by this session"
            )
        self.ib = ib

    def assert_server_clock(
        self,
        *,
        max_skew_seconds: float = 2.0,
        max_round_trip_seconds: float = 2.0,
    ) -> float:
        """Prove the local clock is close enough to IBKR server time."""

        if max_skew_seconds <= 0 or max_round_trip_seconds <= 0:
            raise ValueError("clock bounds must be positive")
        ib = self._require()
        before = pd.Timestamp.now(tz="UTC")
        server = pd.Timestamp(self._blocking_request(ib.reqCurrentTime))
        after = pd.Timestamp.now(tz="UTC")
        if server.tz is None:
            server = server.tz_localize("UTC")
        else:
            server = server.tz_convert("UTC")
        round_trip = (after - before).total_seconds()
        midpoint = before + (after - before) / 2
        skew = abs((server - midpoint).total_seconds())
        if round_trip > max_round_trip_seconds:
            raise RuntimeError(
                f"{self.endpoint.label}: IBKR clock probe took {round_trip:.3f}s"
            )
        if skew > max_skew_seconds:
            raise RuntimeError(
                f"{self.endpoint.label}: local/IBKR clock skew is {skew:.3f}s"
            )
        return skew

    def disconnect(self) -> None:
        for ticker, tape in self._trade_tapes.values():
            ticker.updateEvent -= tape.capture
        self._trade_tapes.clear()
        if self.ib is not None:
            self.ib.disconnect()
            self.ib = None

    def is_connected(self) -> bool:
        return self.ib is not None and bool(self.ib.isConnected())

    def reconnect(self) -> None:
        """Reconnect with the same exact endpoint/account/client identity."""

        try:
            self.disconnect()
        finally:
            self.ib = None
        self.connect()

    def _require(self) -> Any:
        if self.ib is None or not self.ib.isConnected():
            raise RuntimeError(f"{self.endpoint.label}: IBKR is not connected")
        return self.ib

    def _blocking_request(
        self,
        method: Any,
        *args: Any,
        _timeout_seconds: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Discard a connection whose synchronous IB request timed out.

        ib_insync reuses request keys for some blocking calls. Reusing the
        same socket after a timeout risks a late callback completing a newer
        request, so recovery must reconnect the exact endpoint first.
        """

        ib = self._require()
        had_timeout = hasattr(ib, "RequestTimeout")
        prior_timeout = float(
            getattr(ib, "RequestTimeout", BROKER_REQUEST_TIMEOUT_SECONDS)
        )
        if _timeout_seconds is not None:
            if not math.isfinite(_timeout_seconds) or _timeout_seconds <= 0:
                raise ValueError("blocking request timeout must be positive")
            ib.RequestTimeout = min(prior_timeout, float(_timeout_seconds))
        try:
            return method(*args, **kwargs)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            try:
                ib.disconnect()
            finally:
                self.ib = None
            raise RuntimeError(
                f"{self.endpoint.label}: IBKR blocking request timed out; "
                "connection discarded before reconciliation"
            ) from exc
        finally:
            if had_timeout:
                ib.RequestTimeout = prior_timeout
            elif hasattr(ib, "RequestTimeout"):
                delattr(ib, "RequestTimeout")

    def broker_snapshot_facade(self) -> Any:
        """Expose only timeout-fenced, read-only broker snapshot operations."""

        connection = self

        class ReadOnlyBrokerSnapshot:
            def managedAccounts(self) -> list[str]:
                ib = connection._require()
                accounts = [
                    str(account).strip()
                    for account in ib.managedAccounts()
                    if str(account).strip()
                ]
                if connection.endpoint.account not in accounts:
                    raise RuntimeError(
                        "configured account disappeared from the broker session"
                    )
                return accounts

            def reqPositions(self) -> Any:
                ib = connection._require()
                return connection._blocking_request(ib.reqPositions)

            def reqAllOpenOrders(self) -> Any:
                ib = connection._require()
                return connection._blocking_request(ib.reqAllOpenOrders)

            def reqAllOpenOrdersRaw(self) -> list[Any]:
                """Capture full decoder echoes without exposing the live wrapper."""

                ib = connection._require()
                wrapper = ib.wrapper
                with _RAW_OPEN_ORDER_LOCK:
                    original = wrapper.openOrder
                    captured: list[Any] = []

                    def capture(
                        order_id: int,
                        contract: Any,
                        order: Any,
                        order_state: Any,
                    ) -> Any:
                        captured.append(
                            SimpleNamespace(
                                contract=copy.deepcopy(contract),
                                order=copy.deepcopy(order),
                                orderStatus=SimpleNamespace(
                                    orderId=order_id,
                                    status=str(
                                        getattr(order_state, "status", "") or ""
                                    ),
                                ),
                            )
                        )
                        return original(order_id, contract, order, order_state)

                    had_override = "openOrder" in getattr(wrapper, "__dict__", {})
                    prior_override = getattr(wrapper, "__dict__", {}).get("openOrder")
                    try:
                        wrapper.openOrder = capture
                        returned = list(
                            connection._blocking_request(ib.reqAllOpenOrders)
                        )
                    finally:
                        if had_override:
                            wrapper.openOrder = prior_override
                        else:
                            try:
                                delattr(wrapper, "openOrder")
                            except AttributeError:
                                pass
                if captured:
                    return captured
                if returned:
                    raise RuntimeError(
                        "all-client open orders returned without raw broker echoes"
                    )
                return []

            def sleep(self, seconds: float) -> None:
                connection._require().sleep(seconds)

        return ReadOnlyBrokerSnapshot()

    def stock(self, symbol: str) -> Any:
        from ib_insync import Stock

        ib = self._require()
        contract = Stock(symbol.upper(), "SMART", "USD")
        qualified = self._blocking_request(ib.qualifyContracts, contract)
        if len(qualified) != 1 or not int(qualified[0].conId or 0):
            raise RuntimeError(f"could not uniquely qualify {symbol}")
        return qualified[0]

    def nlv(self) -> float:
        ib = self._require()
        values = [
            row
            for row in self._blocking_request(
                ib.accountSummary, self.endpoint.account
            )
            if row.tag == "NetLiquidation" and row.currency in {"USD", "BASE"}
        ]
        if not values:
            raise RuntimeError(f"{self.endpoint.label}: NetLiquidation unavailable")
        nlv = float(values[0].value)
        if not math.isfinite(nlv) or nlv <= 0:
            raise RuntimeError(f"{self.endpoint.label}: invalid NetLiquidation")
        return nlv

    def assert_symbol_clear(self, contract: Any) -> None:
        """V1 isolation: never net Legend against an existing sleeve."""

        ib = self._require()
        # Refresh orders first, then snapshot both orders and positions. Taking
        # positions before the bounded wait can miss another order filling
        # during that wait.
        observed_orders = list(self._blocking_request(ib.reqAllOpenOrders))
        ib.sleep(0.5)
        positions = [
            position
            for position in ib.positions(self.endpoint.account)
            if int(position.contract.conId or 0) == int(contract.conId)
            and float(position.position) != 0
        ]
        orders = [
            trade
            for trade in observed_orders
            if int(trade.contract.conId or 0) == int(contract.conId)
            and str(trade.order.account or self.endpoint.account)
            == self.endpoint.account
            and trade.orderStatus.status not in DEAD_STATUSES
        ]
        if positions or orders:
            raise RuntimeError(
                f"{self.endpoint.label}/{contract.symbol}: existing position or working "
                "order; V1 will not broker-net another sleeve"
            )

    def assert_symbol_isolated(
        self,
        contract: Any,
        *,
        order_ref: str,
        direction: int,
        virtual_quantity: int,
    ) -> None:
        """Prove the account-symbol still contains only the Legend lot."""

        if direction not in (-1, 1) or virtual_quantity < 0:
            raise ValueError("invalid Legend isolation quantity")
        ib = self._require()
        observed_orders = list(self._blocking_request(ib.reqAllOpenOrders))
        ib.sleep(0.4)
        physical = sum(
            float(position.position)
            for position in ib.positions(self.endpoint.account)
            if int(position.contract.conId or 0) == int(contract.conId)
        )
        expected = float(direction * virtual_quantity)
        foreign = [
            trade
            for trade in observed_orders
            if int(trade.contract.conId or 0) == int(contract.conId)
            and str(trade.order.account or self.endpoint.account)
            == self.endpoint.account
            and str(trade.order.orderRef or "") != order_ref
            and trade.orderStatus.status not in DEAD_STATUSES
        ]
        if not math.isclose(physical, expected, rel_tol=0.0, abs_tol=1e-7):
            raise PhysicalIsolationMismatch(
                f"{self.endpoint.label}/{contract.symbol}: physical position "
                f"{physical:g} does not equal Legend virtual lot {expected:g}",
                actual_position=physical,
                expected_position=expected,
                foreign_working_orders=bool(foreign),
            )
        if foreign:
            raise PhysicalIsolationMismatch(
                f"{self.endpoint.label}/{contract.symbol}: foreign working order "
                "violates the Legend account-symbol reservation",
                actual_position=physical,
                expected_position=expected,
                foreign_working_orders=True,
            )

    def attributed_fills_batch(
        self,
        expected_con_ids: Mapping[str, int],
        *,
        request_timeout_seconds: float | None = None,
    ) -> dict[str, list[FillEvent]]:
        """Request executions once and attribute every requested Legend lot."""

        ib = self._require()
        normalized = {
            str(reference): int(con_id)
            for reference, con_id in expected_con_ids.items()
        }
        if not normalized or any(con_id <= 0 for con_id in normalized.values()):
            raise ValueError("expected ETF conIds must be positive")
        # A new API client does not inherit the prior client's in-memory fill
        # list. Explicitly request today's executions on every reconciliation.
        requested = list(
            self._blocking_request(
                ib.reqExecutions, _timeout_seconds=request_timeout_seconds
            )
        )
        all_fills = list(ib.fills()) + requested
        result: dict[str, list[FillEvent]] = {
            reference: [] for reference in normalized
        }
        seen: set[str] = set()
        for fill in all_fills:
            execution = fill.execution
            exec_id = str(execution.execId)
            if exec_id in seen:
                continue
            seen.add(exec_id)
            reference = str(execution.orderRef or "")
            expected_con_id = normalized.get(reference)
            if expected_con_id is None:
                continue
            if (
                str(execution.acctNumber) != self.endpoint.account
                or int(getattr(execution, "clientId", -1))
                != self.endpoint.client_id
                or int(getattr(fill.contract, "conId", 0) or 0)
                != int(expected_con_id)
            ):
                continue
            result[reference].append(
                FillEvent(
                    exec_id=exec_id,
                    order_id=int(execution.orderId),
                    action=str(execution.side),
                    shares=round(float(execution.shares)),
                    price=float(execution.price),
                    timestamp=pd.Timestamp(execution.time).isoformat(),
                    client_id=int(execution.clientId),
                    con_id=int(fill.contract.conId),
                )
            )
        return result

    def attributed_fills(
        self, order_ref: str, *, expected_con_id: int
    ) -> list[FillEvent]:
        return self.attributed_fills_batch({order_ref: expected_con_id})[order_ref]

    def legend_exit_snapshot(
        self,
        expected_con_ids: Mapping[str, int],
        *,
        request_timeout_seconds: float = 0.3,
    ) -> LegendAccountSnapshot:
        """Capture fills, open orders, and positions once for all tracked ETFs.

        The final 10:30 boundary uses this account-wide call so one request
        timeout costs at most one account snapshot, not one timeout per lot.
        """

        normalized = {
            str(reference): int(con_id)
            for reference, con_id in expected_con_ids.items()
        }
        tracked_con_ids = frozenset(normalized.values())
        for attempt in range(2):
            fills_before = self.attributed_fills_batch(
                normalized, request_timeout_seconds=request_timeout_seconds
            )
            ib = self._require()
            observed_orders = list(
                self._blocking_request(
                    ib.reqAllOpenOrders,
                    _timeout_seconds=request_timeout_seconds,
                )
            )
            ib.sleep(0.05)
            positions: dict[int, float] = {
                con_id: 0.0 for con_id in tracked_con_ids
            }
            for position in ib.positions(self.endpoint.account):
                con_id = int(getattr(position.contract, "conId", 0) or 0)
                if con_id in tracked_con_ids:
                    positions[con_id] += float(position.position)
            # A target/GAT fill can land between the execution and position
            # reads. Re-read executions and accept only a stable pair; the
            # caller must not classify a stale-high virtual lot as collision.
            fills_after = self.attributed_fills_batch(
                normalized, request_timeout_seconds=request_timeout_seconds
            )
            before_ids = {
                reference: tuple(sorted(fill.exec_id for fill in events))
                for reference, events in fills_before.items()
            }
            after_ids = {
                reference: tuple(sorted(fill.exec_id for fill in events))
                for reference, events in fills_after.items()
            }
            if before_ids != after_ids:
                if attempt == 0:
                    continue
                raise RuntimeError(
                    f"{self.endpoint.label}: final exit snapshot did not stabilize"
                )
            working: dict[str, list[Any]] = {
                reference: [] for reference in normalized
            }
            foreign: set[int] = set()
            for trade in observed_orders:
                status = str(trade.orderStatus.status or "")
                if status in DEAD_STATUSES:
                    continue
                con_id = int(getattr(trade.contract, "conId", 0) or 0)
                if con_id not in tracked_con_ids:
                    continue
                order = trade.order
                if (
                    str(order.account or self.endpoint.account)
                    != self.endpoint.account
                ):
                    continue
                reference = str(order.orderRef or "")
                owner = int(getattr(order, "clientId", -1))
                if (
                    normalized.get(reference) == con_id
                    and owner == self.endpoint.client_id
                ):
                    working[reference].append(trade)
                else:
                    foreign.add(con_id)
            return LegendAccountSnapshot(
                fills_by_reference=fills_after,
                working_orders_by_reference=working,
                physical_positions_by_con_id=positions,
                foreign_working_con_ids=frozenset(foreign),
            )
        raise RuntimeError(
            f"{self.endpoint.label}: final exit snapshot was unavailable"
        )

    def attributed_orders(
        self, order_ref: str, *, expected_con_id: int
    ) -> list[Any]:
        ib = self._require()
        if int(expected_con_id) <= 0:
            raise ValueError("expected ETF conId must be positive")
        observed_orders = list(self._blocking_request(ib.reqAllOpenOrders))
        ib.sleep(0.4)
        return [
            trade
            for trade in observed_orders
            if str(trade.order.account or self.endpoint.account) == self.endpoint.account
            and str(trade.order.orderRef or "") == order_ref
            and int(getattr(trade.contract, "conId", 0) or 0)
            == int(expected_con_id)
        ]

    def shortability(self, contract: Any, shares: int) -> tuple[bool, str]:
        ib = self._require()
        ib.reqMarketDataType(1)
        ticker = ib.reqMktData(contract, genericTickList="236", snapshot=False)
        try:
            available = None
            for _ in range(20):
                available = _finite_price(getattr(ticker, "shortableShares", None))
                if available is not None:
                    break
                ib.sleep(0.25)
        finally:
            ib.cancelMktData(contract)
        if available is None:
            return False, "shortable shares unavailable after bounded live probe"
        if available < shares:
            return False, f"only {int(available)} shortable shares reported"
        return True, f"{int(available)} shares reported shortable"

    def ex_dividend_status(self, contract: Any, entry_date: str) -> tuple[bool, str]:
        ib = self._require()
        ib.reqMarketDataType(1)
        ticker = ib.reqMktData(contract, genericTickList="456", snapshot=False)
        try:
            dividends = None
            raw = ""
            for _ in range(20):
                dividends = getattr(ticker, "dividends", None)
                raw = str(getattr(dividends, "nextDate", "") or "").strip()
                if raw:
                    break
                ib.sleep(0.25)
        finally:
            ib.cancelMktData(contract)
        if dividends is None:
            raise RuntimeError("IBKR dividend reference tick is unavailable")
        if not raw:
            raise RuntimeError(
                "IBKR did not provide a next ex-dividend date during the bounded probe"
            )
        parsed = pd.to_datetime(raw, errors="coerce")
        if pd.isna(parsed):
            raise RuntimeError(f"unparseable IBKR next dividend date {raw!r}")
        ex_date = pd.Timestamp(parsed).date().isoformat()
        return ex_date == entry_date, f"next ex-dividend date {ex_date}"

    def historical_bars(
        self,
        contract: Any,
        *,
        duration: str,
        bar_size: str,
        use_rth: bool = True,
    ) -> pd.DataFrame:
        ib = self._require()
        bars = self._blocking_request(
            ib.reqHistoricalData,
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=use_rth,
            formatDate=2,
            keepUpToDate=False,
        )
        return bars_to_frame(bars)

    def subscribe_realtime_bars(self, contract: Any) -> Any:
        ib = self._require()
        ib.reqMarketDataType(1)
        return ib.reqRealTimeBars(
            contract,
            barSize=5,
            whatToShow="TRADES",
            useRTH=True,
        )

    def cancel_realtime_bars(self, bars: Any) -> None:
        self._require().cancelRealTimeBars(bars)

    def subscribe_trade_ticks(self, contract: Any) -> Any:
        ib = self._require()
        con_id = int(contract.conId)
        if con_id in self._trade_tapes:
            raise RuntimeError("ETF Last-trade subscription already exists")
        ib.reqMarketDataType(1)
        ticker = ib.reqTickByTickData(
            contract,
            tickType="Last",
            numberOfTicks=0,
            ignoreSize=False,
        )
        tape = TradeTickTape()
        ticker.updateEvent += tape.capture
        self._trade_tapes[con_id] = (ticker, tape)
        return tape

    def cancel_trade_ticks(self, contract: Any) -> None:
        self._require().cancelTickByTickData(contract, "Last")
        subscription = self._trade_tapes.pop(int(contract.conId), None)
        if subscription is not None:
            ticker, tape = subscription
            ticker.updateEvent -= tape.capture

    def wait_for_new_trade_tick(
        self,
        ticker: Any,
        *,
        after_count: int,
        timeout_seconds: float = 0.75,
    ) -> None:
        """Require a newly received Last tick during a bounded causal check."""

        if after_count < 0 or timeout_seconds <= 0:
            raise ValueError("invalid new-tick wait boundary")
        ib = self._require()
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if isinstance(ticker, TradeTickTape):
                ticker.assert_complete()
            if len(getattr(ticker, "tickByTicks", ())) > after_count:
                return
            ib.sleep(0.01)
        raise RuntimeError("no fresh ETF Last tick arrived after bracket staging")

    def sleep(self, seconds: float) -> None:
        self._require().sleep(max(0.0, float(seconds)))

    def what_if_short(self, contract: Any, shares: int) -> None:
        from ib_insync import MarketOrder

        ib = self._require()
        order = MarketOrder("SELL", shares, account=self.endpoint.account)
        state = self._blocking_request(ib.whatIfOrder, contract, order)
        warning = str(getattr(state, "warningText", "") or "").strip()
        try:
            margin_change = float(getattr(state, "initMarginChange", math.nan))
        except (TypeError, ValueError):
            margin_change = math.nan
        if warning or not math.isfinite(margin_change) or abs(margin_change) >= 1e100:
            raise RuntimeError(f"short what-if rejected: {warning or 'no order state'}")

    def place_bracket(
        self,
        *,
        contract: Any,
        direction: int,
        shares: int,
        target: float,
        entry_date: str,
        final_transmit_check: Callable[[], None] | None = None,
    ) -> tuple[Any, Any, Any]:
        if not self.live:
            raise RuntimeError("dry-run connection may not place orders")
        from ib_insync import LimitOrder, MarketOrder

        ib = self._require()
        action = "BUY" if direction > 0 else "SELL"
        exit_action = "SELL" if direction > 0 else "BUY"
        reference = build_order_ref(contract.symbol, direction, entry_date)
        parent_id = ib.client.getReqId()
        target_id = ib.client.getReqId()
        time_id = ib.client.getReqId()
        oca = f"LEGEND_{self.endpoint.account}_{parent_id}"

        parent = MarketOrder(action, shares)
        parent.orderId = parent_id
        parent.account = self.endpoint.account
        # IOC bounds the market-entry fill window at the broker. The child
        # target/time pair remains DAY and is required to survive a partial
        # parent fill in the exact-build paper proof before live can arm.
        parent.tif = "IOC"
        parent.outsideRth = False
        parent.orderRef = reference
        parent.transmit = False

        limit = LimitOrder(exit_action, shares, target)
        limit.orderId = target_id
        limit.parentId = parent_id
        limit.account = self.endpoint.account
        limit.tif = "DAY"
        limit.outsideRth = False
        limit.ocaGroup = oca
        # Type 2 proportionately reduces the 10:30 peer after a partial target
        # fill; type 1 would cancel it and leave the residual lot unprotected.
        limit.ocaType = 2
        limit.orderRef = reference
        limit.transmit = False

        timed = MarketOrder(exit_action, shares)
        timed.orderId = time_id
        timed.parentId = parent_id
        timed.account = self.endpoint.account
        timed.tif = "DAY"
        timed.outsideRth = False
        timed.ocaGroup = oca
        timed.ocaType = 2
        timed.orderRef = reference
        timed.goodAfterTime = f"{entry_date.replace('-', '')} 10:30:00 US/Eastern"
        timed.transmit = True

        parent_trade = ib.placeOrder(contract, parent)
        limit_trade = ib.placeOrder(contract, limit)
        if final_transmit_check is not None:
            try:
                final_transmit_check()
            except Exception as check_exc:
                # The first two legs are transmit=False, but they are durable
                # TWS client state. Cancel and obtain a fresh exact-client
                # openOrder snapshot before declaring that no bracket crossed.
                try:
                    self.request_cancel_orders((parent_trade, limit_trade))
                    cleared = False
                    for _ in range(3):
                        observed_orders = list(
                            self._blocking_request(ib.reqOpenOrders)
                        )
                        ib.sleep(0.2)
                        staged = [
                            trade
                            for trade in observed_orders
                            if int(getattr(trade.order, "orderId", 0) or 0)
                            in {parent_id, target_id}
                            and str(
                                getattr(trade.order, "account", "")
                                or self.endpoint.account
                            )
                            == self.endpoint.account
                            and str(getattr(trade.order, "orderRef", "") or "")
                            == reference
                            and int(getattr(trade.contract, "conId", 0) or 0)
                            == int(getattr(contract, "conId", 0) or 0)
                            and str(trade.orderStatus.status or "")
                            not in DEAD_STATUSES
                        ]
                        if not staged:
                            cleared = True
                            break
                        self.request_cancel_orders(staged)
                    if not cleared:
                        raise RuntimeError(
                            "staged non-transmitting bracket legs remain open"
                        )
                except Exception as cleanup_exc:
                    raise RuntimeError(
                        "final entry proof failed and staged bracket cleanup "
                        "was not broker-proven"
                    ) from cleanup_exc
                raise PreTransmitCheckBlocked(str(check_exc)) from check_exc
        timed_trade = ib.placeOrder(contract, timed)
        return parent_trade, limit_trade, timed_trade

    def place_exit_oca(
        self,
        *,
        contract: Any,
        direction: int,
        shares: int,
        target: float,
        entry_date: str,
        order_ref: str | None = None,
    ) -> tuple[Any, Any]:
        """Attach target/time protection to an already-attributed partial fill."""

        if not self.live:
            raise RuntimeError("dry-run connection may not place orders")
        from ib_insync import LimitOrder, MarketOrder

        ib = self._require()
        exit_action = "SELL" if direction > 0 else "BUY"
        reference = order_ref or build_order_ref(contract.symbol, direction, entry_date)
        target_id = ib.client.getReqId()
        time_id = ib.client.getReqId()
        oca = f"LEGEND_EXIT_{self.endpoint.account}_{target_id}"
        limit = LimitOrder(exit_action, shares, target)
        limit.orderId = target_id
        limit.account = self.endpoint.account
        limit.tif = "DAY"
        limit.outsideRth = False
        limit.ocaGroup = oca
        limit.ocaType = 2
        limit.orderRef = reference
        limit.transmit = False
        timed = MarketOrder(exit_action, shares)
        timed.orderId = time_id
        timed.account = self.endpoint.account
        timed.tif = "DAY"
        timed.outsideRth = False
        timed.ocaGroup = oca
        timed.ocaType = 2
        timed.orderRef = reference
        timed.goodAfterTime = f"{entry_date.replace('-', '')} 10:30:00 US/Eastern"
        timed.transmit = True
        limit_trade = ib.placeOrder(contract, limit)
        timed_trade = ib.placeOrder(contract, timed)
        return limit_trade, timed_trade

    def place_emergency_exit(
        self,
        *,
        contract: Any,
        direction: int,
        shares: int,
        order_ref: str,
    ) -> Any:
        """Market-exit only a proven positive Legend virtual lot."""

        if not self.live:
            raise RuntimeError("dry-run connection may not place orders")
        if direction not in (-1, 1) or shares <= 0:
            raise ValueError("emergency exit requires direction and positive shares")
        from ib_insync import MarketOrder

        ib = self._require()
        action = "SELL" if direction > 0 else "BUY"
        order = MarketOrder(action, shares)
        order.orderId = ib.client.getReqId()
        order.account = self.endpoint.account
        order.tif = "DAY"
        order.outsideRth = False
        order.orderRef = order_ref
        order.transmit = True
        return ib.placeOrder(contract, order)

    def confirm_broker_echo(
        self, trades: Iterable[Any], *, description: str, timeout_seconds: float = 3.0
    ) -> None:
        """Require broker permIds and non-pending statuses for every order leg."""

        ib = self._require()
        items = list(trades)
        if not items:
            raise RuntimeError(f"{description}: no orders were submitted")
        loops = max(1, math.ceil(timeout_seconds / 0.2))
        statuses: list[str] = []
        perm_ids: list[int] = []
        for _ in range(loops):
            statuses = [str(item.orderStatus.status or "") for item in items]
            perm_ids = [int(getattr(item.order, "permId", 0) or 0) for item in items]
            if any(status in REJECTED_STATUSES for status in statuses):
                break
            if all(perm_ids) and all(
                status in ACKNOWLEDGED_STATUSES for status in statuses
            ):
                return
            ib.sleep(0.2)
        raise RuntimeError(
            f"{description}: broker echo not proven; statuses={statuses}, "
            f"permIds={perm_ids}"
        )

    def modify_target(self, trade: Any, *, new_target: float, remaining: int) -> Any:
        updated = self.request_target_modification(
            trade, new_target=new_target, remaining=remaining
        )
        return self.confirm_target_modification(
            updated,
            expected_target=new_target,
            expected_con_id=int(updated.contract.conId),
            order_ref=str(updated.order.orderRef),
        )

    def request_target_modification(
        self, trade: Any, *, new_target: float, remaining: int
    ) -> Any:
        """Transmit a target revision without a per-order acknowledgement wait."""

        if not self.live:
            raise RuntimeError("dry-run connection may not modify orders")
        if remaining <= 0:
            raise ValueError("remaining target quantity must be positive")
        ib = self._require()
        order = trade.order
        original = round(float(order.totalQuantity))
        broker_remaining = round(float(trade.orderStatus.remaining or 0))
        if remaining > original or (broker_remaining and remaining > broker_remaining):
            raise ValueError("target modification may not increase remaining quantity")
        # Do not rewrite totalQuantity after a partial fill: IBKR interprets it
        # as lifetime order quantity, not just the unfilled remainder. OCA type
        # 2 already reduces the time peer proportionately.
        order.lmtPrice = float(new_target)
        # The original bracket child is Transmit=False and was released by the
        # final sibling. A later modification must itself be transmitted.
        order.transmit = True
        return ib.placeOrder(trade.contract, order)

    def confirm_target_modification(
        self,
        trade: Any,
        *,
        expected_target: float,
        expected_con_id: int,
        order_ref: str,
        timeout_seconds: float = 3.0,
    ) -> Any:
        """Require a fresh server openOrder echo of the revised limit price."""

        ib = self._require()
        order_id = int(trade.order.orderId)
        perm_id = int(getattr(trade.order, "permId", 0) or 0)
        if order_id <= 0 or perm_id <= 0 or expected_con_id <= 0:
            raise RuntimeError("target revision lacks durable order/contract identity")
        loops = max(1, math.ceil(timeout_seconds / 0.3))
        for _ in range(loops):
            snapshots: list[dict[str, Any]] = []
            # ib_insync returns Trade snapshots for a blocking openOrders
            # request; it intentionally does not emit openOrderEvent while
            # that request is active.
            observed_trades = list(
                self._blocking_request(ib.reqOpenOrders)
            )
            for observed in observed_trades:
                snapshots.append(
                    {
                        "order_id": int(observed.order.orderId),
                        "perm_id": int(getattr(observed.order, "permId", 0) or 0),
                        "client_id": int(
                            getattr(observed.order, "clientId", -1)
                        ),
                        "account": str(observed.order.account or ""),
                        "order_ref": str(observed.order.orderRef or ""),
                        "order_type": str(observed.order.orderType or "").upper(),
                        "limit": float(observed.order.lmtPrice),
                        "transmit": bool(observed.order.transmit),
                        "con_id": int(getattr(observed.contract, "conId", 0) or 0),
                        "status": str(observed.orderStatus.status or ""),
                    }
                )
            for snapshot in snapshots:
                if (
                    snapshot["order_id"] == order_id
                    and snapshot["perm_id"] == perm_id
                    and snapshot["client_id"] == self.endpoint.client_id
                    and snapshot["account"] == self.endpoint.account
                    and snapshot["order_ref"] == order_ref
                    and snapshot["order_type"] == "LMT"
                    and math.isclose(
                        snapshot["limit"],
                        float(expected_target),
                        rel_tol=0.0,
                        abs_tol=1e-7,
                    )
                    and snapshot["transmit"]
                    and snapshot["con_id"] == int(expected_con_id)
                    and snapshot["status"] in ACKNOWLEDGED_STATUSES
                ):
                    return trade
            ib.sleep(0.3)
        raise RuntimeError(
            "target modification lacks a fresh matching broker openOrder echo"
        )

    def cancel_owned_order(self, trade: Any) -> None:
        if not self.live:
            raise RuntimeError("dry-run connection may not cancel orders")
        ib = self._require()
        ib.cancelOrder(trade.order)
        ib.sleep(0.4)

    def request_cancel_orders(self, trades: Iterable[Any]) -> None:
        """Issue a batch of owned cancellations without per-order sleeps."""

        if not self.live:
            raise RuntimeError("dry-run connection may not cancel orders")
        ib = self._require()
        for trade in trades:
            if str(trade.orderStatus.status or "") in DEAD_STATUSES:
                continue
            # A just-submitted ib_insync Order can still carry clientId=0
            # until the broker echo arrives. Such objects were created by this
            # connection; a nonzero mismatched client ID is foreign.
            owner = int(
                getattr(trade.order, "clientId", 0) or self.endpoint.client_id
            )
            if owner != self.endpoint.client_id:
                raise RuntimeError("refusing to cancel an order owned by another client")
            ib.cancelOrder(trade.order)


def bars_to_frame(bars: Any) -> pd.DataFrame:
    """Convert ib_insync BarDataList to the core's normalized column shape."""

    from ib_insync import util

    frame = util.df(bars)
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = frame.rename(columns={column: str(column).lower() for column in frame.columns})
    if "date" not in frame:
        raise RuntimeError("IBKR bars have no date column")
    index = pd.DatetimeIndex(pd.to_datetime(frame.pop("date"), errors="raise"))
    if index.tz is None:
        index = index.tz_localize(NY_TZ)
    frame.index = index.tz_convert("UTC")
    return frame


def realtime_bars_to_frame(bars: Any) -> pd.DataFrame:
    rows = []
    for bar in bars:
        timestamp = pd.Timestamp(bar.time)
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize("UTC")
        rows.append(
            {
                "timestamp": timestamp.tz_convert("UTC"),
                "open": float(bar.open_),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(rows).set_index("timestamp").sort_index()
    if frame.index.has_duplicates:
        raise RuntimeError("IBKR real-time bars contain duplicate timestamps")
    ohlc = frame[["open", "high", "low", "close"]]
    values = ohlc.to_numpy(dtype=float)
    if not math.isfinite(float(values.min())) or not math.isfinite(
        float(values.max())
    ):
        raise RuntimeError("IBKR real-time bars contain non-finite OHLC")
    if (values <= 0).any():
        raise RuntimeError("IBKR real-time bars contain non-positive OHLC")
    invalid = (
        ohlc["high"] < ohlc[["open", "close", "low"]].max(axis=1)
    ) | (
        ohlc["low"] > ohlc[["open", "close", "high"]].min(axis=1)
    )
    if invalid.any():
        raise RuntimeError("IBKR real-time bars contain malformed OHLC")
    volume = pd.to_numeric(frame["volume"], errors="coerce").to_numpy(dtype=float)
    if not all(math.isfinite(value) and value >= 0 for value in volume):
        raise RuntimeError("IBKR real-time bars contain invalid volume")
    return frame


def trade_ticks_to_frame(ticker: Any) -> pd.DataFrame:
    """Snapshot ordered exchange-reported Last ticks from a retained tape."""

    if isinstance(ticker, TradeTickTape):
        ticker.assert_complete()
    rows: list[dict[str, Any]] = []
    for sequence, tick in enumerate(getattr(ticker, "tickByTicks", ())):
        timestamp = pd.Timestamp(tick.time)
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize("UTC")
        rows.append(
            {
                "sequence": sequence,
                "timestamp": timestamp.tz_convert("UTC"),
                "price": float(tick.price),
                "size": float(tick.size),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["sequence", "timestamp", "price", "size"])
    frame = pd.DataFrame(rows).sort_values(
        ["timestamp", "sequence"], kind="stable"
    )
    if not all(
        math.isfinite(price) and price > 0
        for price in frame["price"].to_numpy(dtype=float)
    ):
        raise RuntimeError("IBKR trade ticks contain invalid prices")
    if not all(
        math.isfinite(size) and size >= 0
        for size in frame["size"].to_numpy(dtype=float)
    ):
        raise RuntimeError("IBKR trade ticks contain invalid sizes")
    return frame.reset_index(drop=True)
