from __future__ import annotations

import asyncio
import copy
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from legend_etf.databento_source import (
    CACHE_REFRESH_OVERLAP,
    PAID_CONFIRMATION,
    _cache_request_start,
    _daily_charge_state,
    _merge_cache_refresh,
    _read_cache,
    _write_verified_cache,
    update_rolling_cache,
)
from legend_etf.ibkr_adapter import (
    Endpoint,
    FillEvent,
    IBKRConnection,
    LegendAccountSnapshot,
    LiveGate,
    PhysicalIsolationMismatch,
    owned_quantity,
)
from legend_etf.paper_proof import validate_paper_proof
from legend_etf.portfolio_guard import (
    PortfolioRequirement,
    reserve_portfolio_capacity,
    validate_portfolio_budget,
)
from legend_etf.reservations import (
    CANDIDATE_ATR_ATOL,
    CANDIDATE_PARITY_COUNT,
    CANDIDATE_PARITY_PROTOCOL,
    CANDIDATE_PARITY_RANGE,
    CANDIDATE_PARITY_SESSION_COUNT,
    CANDIDATE_PIPELINE_FILES,
    CANDIDATE_RATIO_ATOL,
    GUARD_REQUIRED_MARKER_NAME,
    INTEGRATION_RECEIPT_PROTOCOL,
    INTEGRATION_REVIEW_TOKEN,
    REQUIRED_INTEGRATION_TESTS,
    ReservationBook,
    candidate_pipeline_attestation,
    discover_broker_mutation_files,
    discover_executor_python_files,
    file_sha256,
    load_attested_external_guard,
    quarantine_path,
    source_tree_sha256,
    validate_candidate_parity_evidence,
    validate_guard_manifest,
)
from legend_etf.session import LegendSession, ManagedTrade, MarketContext
from legend_etf.storage import (
    StateStore,
    atomic_write_json,
    content_hash,
    finalize_plan,
    read_json,
    validate_plan,
)


def _futures_cache_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "volume": 10.0,
            "instrument_id": 123,
        },
        index=index,
    )


def _input_file_record(path):
    source = path.resolve()
    stat = source.stat()
    return {
        "path": str(source),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": file_sha256(source),
    }


def test_candidate_parity_verifier_bootstraps_source_before_imports():
    root = Path(__file__).resolve().parents[1]
    verifier = root / "scripts" / "verify_legend_futures_candidate_parity.py"
    source = verifier.read_text(encoding="utf-8")
    assert source.index("EARLY_CANDIDATE_SOURCE_TREE_SHA256 =") < source.index(
        "import numpy as np"
    )
    assert source.index("import numpy as np") < source.index(
        "from legend_etf.calendar import _calendar"
    )

    spec = importlib.util.spec_from_file_location("legend_parity_verifier_test", verifier)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    attestation = candidate_pipeline_attestation(root)
    assert module.CANDIDATE_SOURCE_LABELS == CANDIDATE_PIPELINE_FILES
    assert (
        module.EARLY_CANDIDATE_SOURCE_TREE_SHA256
        == attestation["source_tree_sha256"]
    )


def _candidate_parity_evidence(root, directory):
    inputs = directory / "parity_inputs"
    archive = inputs / "archive"
    archive.mkdir(parents=True)
    engine = inputs / "engine.py"
    golden = inputs / "golden.csv"
    archive_file = archive / "2016.parquet"
    engine.write_text("# frozen engine fixture\n", encoding="utf-8")
    golden.write_text("root,setup_date,entry_date\n", encoding="utf-8")
    archive_file.write_bytes(b"frozen archive fixture")
    archive_records = [_input_file_record(archive_file)]
    return {
        "protocol": CANDIDATE_PARITY_PROTOCOL,
        "status": "pass",
        "completed_at": "2026-09-02T12:00:00Z",
        "command": "python scripts/verify_legend_futures_candidate_parity.py",
        "inputs": {
            "historical_engine": _input_file_record(engine),
            "golden": _input_file_record(golden),
            "archive": {
                "path": str(archive.resolve()),
                "metadata_manifest_sha256": content_hash(archive_records),
                "files": archive_records,
            },
        },
        "range": CANDIDATE_PARITY_RANGE,
        "full_session_count": CANDIDATE_PARITY_SESSION_COUNT,
        "counts": {
            label: CANDIDATE_PARITY_COUNT
            for label in ("golden", "research", "production")
        },
        "duplicate_key_rows": {
            label: 0 for label in ("golden", "research", "production")
        },
        "matches": {
            label: CANDIDATE_PARITY_COUNT
            for label in (
                "research_direction",
                "research_contract",
                "production_direction",
                "production_contract",
            )
        },
        "max_deltas": {
            "research_ratio": 0.0,
            "production_ratio": 0.0,
            "research_atr14": 0.0,
        },
        "tolerances": {
            "ratio_atol": CANDIDATE_RATIO_ATOL,
            "atr_atol": CANDIDATE_ATR_ATOL,
        },
        "runtime_seconds": 1.0,
        "candidate_pipeline": candidate_pipeline_attestation(root),
    }


def test_futures_cache_sidecar_round_trip_and_tamper_detection(tmp_path):
    path = tmp_path / "ES.parquet"
    frame = _futures_cache_frame(
        pd.date_range("2026-08-31 13:30", periods=3, freq="1min", tz="UTC")
    )
    _write_verified_cache(path, frame)
    pd.testing.assert_frame_equal(
        _read_cache(path), frame, check_dtype=False, check_freq=False
    )

    frame.iloc[:-1].to_parquet(path)
    with pytest.raises(RuntimeError, match="integrity mismatch"):
        _read_cache(path)


def test_futures_cache_requires_integrity_sidecar(tmp_path):
    path = tmp_path / "NQ.parquet"
    frame = _futures_cache_frame(
        pd.date_range("2026-08-31 13:30", periods=2, freq="1min", tz="UTC")
    )
    frame.to_parquet(path)
    with pytest.raises(RuntimeError, match="sidecar pair is incomplete"):
        _read_cache(path)


def test_cache_request_replays_overlap_and_never_jumps_append_gap():
    end = pd.Timestamp("2026-09-01 12:45", tz="UTC")
    fallback = end - pd.Timedelta(days=150)
    current = _futures_cache_frame(
        pd.date_range(end - pd.Timedelta(days=2), periods=2, freq="1min")
    )
    assert _cache_request_start(
        current,
        start_fallback=fallback.to_pydatetime(),
        end=end.to_pydatetime(),
    ) == (end - CACHE_REFRESH_OVERLAP).to_pydatetime()

    stale = _futures_cache_frame(
        pd.date_range(end - pd.Timedelta(days=40), periods=2, freq="1min")
    )
    assert _cache_request_start(
        stale,
        start_fallback=fallback.to_pydatetime(),
        end=end.to_pydatetime(),
    ) == (stale.index[-1] + pd.Timedelta(minutes=1)).to_pydatetime()


def test_cache_refresh_heals_additions_but_blocks_disappearing_rows():
    index = pd.date_range("2026-08-31 13:30", periods=3, freq="1min", tz="UTC")
    cached = _futures_cache_frame(index[[0, 2]])
    fresh = _futures_cache_frame(index)
    merged = _merge_cache_refresh(
        cached,
        fresh,
        request_start=index[0].to_pydatetime(),
        end=(index[-1] + pd.Timedelta(minutes=1)).to_pydatetime(),
    )
    assert merged.index.equals(index)

    truncated = fresh.drop(index=index[2])
    with pytest.raises(RuntimeError, match="omitted cached timestamp"):
        _merge_cache_refresh(
            cached,
            truncated,
            request_start=index[0].to_pydatetime(),
            end=(index[-1] + pd.Timedelta(minutes=1)).to_pydatetime(),
        )


def test_completed_databento_request_is_not_downloaded_or_billed_twice(
    tmp_path, monkeypatch
):
    class Metadata:
        @staticmethod
        def get_cost(**_kwargs):
            return 0.25

        @staticmethod
        def get_billable_size(**_kwargs):
            return 100

    client = SimpleNamespace(metadata=Metadata())
    calls = []

    def fake_fetch(_client, symbol, start, _end):
        calls.append(symbol)
        return _futures_cache_frame(pd.DatetimeIndex([pd.Timestamp(start)]))

    monkeypatch.setattr(
        "legend_etf.databento_source.fetch_market_minutes", fake_fetch
    )
    end = pd.Timestamp("2026-09-02 12:45", tz="UTC").to_pydatetime()
    start = pd.Timestamp(end) - pd.Timedelta(days=2)
    kwargs = {
        "client": client,
        "start_fallback": start.to_pydatetime(),
        "end": end,
        "cache_dir": tmp_path,
        "archive_dir": None,
        "max_cost_usd": 0.75,
        "paid_confirmation": PAID_CONFIRMATION,
    }
    update_rolling_cache(**kwargs)
    _frames, second_quotes, _request_start = update_rolling_cache(**kwargs)

    assert len(calls) == 3
    assert sum(quote.cost_usd for quote in second_quotes) == 0
    cost, ambiguous, completed = _daily_charge_state(
        tmp_path / "databento_charge_ledger.jsonl"
    )
    assert cost == pytest.approx(0.75)
    assert not ambiguous
    assert len(completed) == 3


def test_candidate_parity_evidence_is_bound_to_current_source_and_runtime(tmp_path):
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    path = tmp_path / "candidate_parity.json"
    evidence = _candidate_parity_evidence(root, tmp_path)
    atomic_write_json(path, evidence)
    validate_candidate_parity_evidence(path, legend_root=root)

    evidence["candidate_pipeline"]["source_tree_sha256"] = "0" * 64
    atomic_write_json(path, evidence)
    with pytest.raises(RuntimeError, match="stale"):
        validate_candidate_parity_evidence(path, legend_root=root)


def test_candidate_parity_evidence_rejects_synthetic_headline_only_payload(tmp_path):
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    path = tmp_path / "candidate_parity.json"
    atomic_write_json(
        path,
        {
            "protocol": CANDIDATE_PARITY_PROTOCOL,
            "status": "pass",
            "counts": {
                label: CANDIDATE_PARITY_COUNT
                for label in ("golden", "research", "production")
            },
            "candidate_pipeline": candidate_pipeline_attestation(root),
        },
    )
    with pytest.raises(RuntimeError, match="schema"):
        validate_candidate_parity_evidence(path, legend_root=root)


def _target_trade(*, price: float = 500.0, client_id: int = 155, con_id: int = 1):
    order = SimpleNamespace(
        orderId=2,
        permId=22,
        clientId=client_id,
        account="U123",
        orderRef="SPY|BUY|Legend EMA ETF|2026-09-01",
        orderType="LMT",
        lmtPrice=price,
        transmit=True,
    )
    return SimpleNamespace(
        contract=SimpleNamespace(symbol="SPY", conId=con_id),
        order=order,
        orderStatus=SimpleNamespace(status="Submitted"),
    )


class _OpenOrderIB:
    def __init__(self, snapshots):
        self.snapshots = list(snapshots)
        self.calls = 0
        self.disconnected = False

    def isConnected(self):
        return not self.disconnected

    def reqOpenOrders(self):
        self.calls += 1
        value = self.snapshots.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    def sleep(self, _seconds):
        return None

    def disconnect(self):
        self.disconnected = True


def test_target_revision_requires_fresh_blocking_open_order_snapshot():
    connection = IBKRConnection(
        Endpoint("primary", "127.0.0.1", 7496, 155, "U123"), live=True
    )
    local_trade = _target_trade(price=499.5)
    stale = _target_trade(price=500.0)
    fresh = _target_trade(price=499.5)
    connection.ib = _OpenOrderIB([[stale], [fresh]])
    result = connection.confirm_target_modification(
        local_trade,
        expected_target=499.5,
        expected_con_id=1,
        order_ref=local_trade.order.orderRef,
        timeout_seconds=0.6,
    )
    assert result is local_trade
    assert connection.ib.calls == 2


def test_timed_out_blocking_request_discards_connection_before_repair():
    connection = IBKRConnection(
        Endpoint("primary", "127.0.0.1", 7496, 155, "U123"), live=True
    )
    broker = _OpenOrderIB([asyncio.TimeoutError("late openOrderEnd")])
    connection.ib = broker
    trade = _target_trade(price=499.5)
    with pytest.raises(RuntimeError, match="connection discarded"):
        connection.confirm_target_modification(
            trade,
            expected_target=499.5,
            expected_con_id=1,
            order_ref=trade.order.orderRef,
            timeout_seconds=0.1,
        )
    assert broker.disconnected
    assert connection.ib is None


def test_broker_snapshot_facade_exposes_only_fenced_reads_and_raw_echoes():
    callback_calls: list[tuple[int, object, object, object]] = []
    contract = SimpleNamespace(symbol="SPY", conId=1)
    order = SimpleNamespace(orderId=17, account="U123", totalQuantity=5)
    state = SimpleNamespace(status="Submitted")

    def original_open_order(order_id, broker_contract, broker_order, order_state):
        callback_calls.append(
            (order_id, broker_contract, broker_order, order_state)
        )

    class SnapshotIB:
        RequestTimeout = 5.0

        def __init__(self):
            self.wrapper = SimpleNamespace(openOrder=original_open_order)

        def isConnected(self):
            return True

        def managedAccounts(self):
            return ["U123", "U999"]

        def reqPositions(self):
            return ["positions"]

        def reqAllOpenOrders(self):
            self.wrapper.openOrder(17, contract, order, state)
            return ["merged-cache-value"]

        def sleep(self, _seconds):
            return None

    connection = IBKRConnection(
        Endpoint("primary", "127.0.0.1", 7496, 155, "U123"), live=True
    )
    broker = SnapshotIB()
    connection.ib = broker
    facade = connection.broker_snapshot_facade()
    for forbidden in ("placeOrder", "cancelOrder", "reqGlobalCancel", "wrapper"):
        assert not hasattr(facade, forbidden)
    assert facade.managedAccounts() == ["U123", "U999"]
    assert facade.reqPositions() == ["positions"]
    raw = facade.reqAllOpenOrdersRaw()
    assert len(raw) == 1
    assert raw[0].contract is not contract
    assert raw[0].order is not order
    assert raw[0].order.orderId == 17
    assert raw[0].orderStatus.status == "Submitted"
    assert broker.wrapper.openOrder is original_open_order
    assert callback_calls == [(17, contract, order, state)]


def test_broker_snapshot_facade_rejects_cache_without_decoder_echo():
    class SnapshotIB:
        RequestTimeout = 5.0

        def __init__(self):
            self.wrapper = SimpleNamespace(openOrder=lambda *_args: None)

        def isConnected(self):
            return True

        def reqAllOpenOrders(self):
            return [SimpleNamespace(order=SimpleNamespace(orderId=7))]

    connection = IBKRConnection(
        Endpoint("primary", "127.0.0.1", 7496, 155, "U123"), live=True
    )
    connection.ib = SnapshotIB()
    with pytest.raises(RuntimeError, match="without raw broker echoes"):
        connection.broker_snapshot_facade().reqAllOpenOrdersRaw()


def test_durable_quarantine_survives_process_lock_and_same_owner_can_adopt(tmp_path):
    directory = tmp_path / "reservations"
    first = ReservationBook(directory)
    assert first.try_acquire("U123", "SPY") == (True, "acquired")
    first.activate_quarantine(
        "U123", "SPY", owner_token="sig", payload={"signal_id": "sig"}
    )
    first.close()

    stranger = ReservationBook(directory)
    acquired, detail = stranger.try_acquire("U123", "SPY")
    assert not acquired and "quarantine" in detail

    recovery = ReservationBook(directory)
    assert recovery.try_acquire("U123", "SPY", owner_token="sig")[0]
    recovery.deactivate_quarantine("U123", "SPY", owner_token="sig")
    recovery.close()

    successor = ReservationBook(directory)
    assert successor.try_acquire("U123", "SPY", owner_token="other")[0]
    successor.close()


def test_terminal_proof_keeps_quarantine_until_watchdog_revalidation(tmp_path):
    directory = tmp_path / "reservations"
    store = StateStore(tmp_path / "state.json", tmp_path / "audit.jsonl")
    book = ReservationBook(directory)
    assert book.try_acquire("U123", "SPY", owner_token="sig")[0]
    book.activate_quarantine(
        "U123", "SPY", owner_token="sig", payload={"signal_id": "sig"}
    )
    state = {
        "account": "U123",
        "etf": "SPY",
        "quarantine_required": True,
        "quarantine_active": True,
    }
    session = LegendSession(
        plan_path=tmp_path / "plan.json",
        state_store=store,
        account_labels=["primary"],
        live_requested=True,
    )
    session.reservations = book
    session._persist_terminal_proof("sig", state, terminal_state="complete")
    marker = read_json(quarantine_path(directory, "U123", "SPY"))
    assert marker["active"] is True
    assert store.get("sig")["quarantine_release_pending"] is True

    session.reconcile_only = True
    session._persist_terminal_proof("sig", state, terminal_state="complete")
    marker = read_json(quarantine_path(directory, "U123", "SPY"))
    assert marker["active"] is True
    assert store.get("sig")["quarantine_release_pending"] is True
    assert store.get("sig")["correction_audit_required"] is True
    book.close()


def _valid_plan() -> dict:
    markets = []
    mapping = (("ES", "ES.v.0", "SPY"), ("NQ", "NQ.v.0", "QQQ"), ("RTY", "RTY.v.0", "IWM"))
    for index, (root, future, etf) in enumerate(mapping):
        qualified = index == 0
        markets.append(
            {
                "root": root,
                "futures_symbol": future,
                "etf": etf,
                "qualifies": qualified,
                "reason": "qualified" if qualified else "trend_ratio_below_threshold",
                "setup_date": "2026-08-31",
                "entry_date": "2026-09-01",
                "instrument_id": 11 if qualified else None,
                "entry_instrument_id": 11 if qualified else None,
                "trend_direction": 1 if qualified else None,
                "trend_ratio": 0.8 if qualified else 0.5,
                "rth_bar_count": 26,
            }
        )
    return finalize_plan(
        {
            "entry_date": "2026-09-01",
            "setup_date": "2026-08-31",
            "created_at": "2026-09-01T12:45:00+00:00",
            "data_as_of": "2026-09-01T12:45:00+00:00",
            "dataset": "GLBX.MDP3",
            "schema": "ohlcv-1m",
            "request_start": "2026-04-01T00:00:00+00:00",
            "cache_dir": "C:/runtime/cache",
            "quoted_cost_usd": 0.0,
            "quoted_billable_bytes": 0,
            "markets": markets,
        }
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda plan: plan["markets"].__setitem__(1, copy.deepcopy(plan["markets"][0])), "roots"),
        (lambda plan: plan["markets"][0].__setitem__("etf", "DIA"), "mapping"),
        (lambda plan: plan["markets"][0].__setitem__("trend_ratio", float("nan")), "ratio"),
    ],
)
def test_signal_plan_semantics_reject_mutated_market_schema(mutation, match):
    plan = _valid_plan()
    validate_plan(plan, entry_date="2026-09-01")
    body = copy.deepcopy(plan)
    body.pop("plan_hash")
    mutation(body)
    changed = finalize_plan(body)
    with pytest.raises((TypeError, ValueError), match=match):
        validate_plan(changed, entry_date="2026-09-01")


def test_ibkr_clock_skew_fails_closed():
    class ClockIB:
        def isConnected(self):
            return True

        def reqCurrentTime(self):
            return pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=5)

    connection = IBKRConnection(
        Endpoint("feed", "127.0.0.1", 7496, 154, "U123"), live=False
    )
    connection.ib = ClockIB()
    with pytest.raises(RuntimeError, match="clock skew"):
        connection.assert_server_clock()


def test_shared_portfolio_budget_is_exact_date_account_and_deployment_scoped(
    tmp_path,
):
    path = tmp_path / "portfolio_budget.json"
    payload = {
        "protocol": "legend-equity-index-risk-budget-v3",
        "entry_date": "2026-09-01",
        "generated_at": "2026-09-01T08:45:00-04:00",
        "expires_at": "2026-09-02T00:00:00-04:00",
        "source_manifest_sha256": "a" * 64,
        "risk_basis": "stress_atr_bps",
        "accounts": {
            "U123": {
                "remaining_long_bps": 20.0,
                "remaining_short_bps": 10.0,
                "remaining_gross_bps": 25.0,
            }
        },
        "reservations": {},
    }
    atomic_write_json(path, payload)
    capacities, digest = validate_portfolio_budget(
        path,
        entry_date="2026-09-01",
        account_ids=["U123"],
        expected_manifest_sha256="a" * 64,
        now=pd.Timestamp("2026-09-01 15:55", tz="America/New_York"),
    )
    assert capacities["U123"].remaining_gross_bps == 25
    assert len(digest) == 64
    with pytest.raises(RuntimeError, match="account set"):
        validate_portfolio_budget(
            path,
            entry_date="2026-09-01",
            account_ids=["U999"],
            expected_manifest_sha256="a" * 64,
            now=pd.Timestamp("2026-09-01 09:30", tz="America/New_York"),
        )
    with pytest.raises(RuntimeError, match="expired"):
        validate_portfolio_budget(
            path,
            entry_date="2026-09-01",
            account_ids=["U123"],
            expected_manifest_sha256="a" * 64,
            now=pd.Timestamp("2026-09-02 00:00", tz="America/New_York"),
        )


def test_portfolio_capacity_is_atomically_debited_and_cannot_be_reused(tmp_path):
    path = tmp_path / "portfolio_budget.json"
    lock = tmp_path / "equity_index_cluster_budget.lock"
    payload = {
        "protocol": "legend-equity-index-risk-budget-v3",
        "entry_date": "2026-09-01",
        "generated_at": "2026-09-01T08:45:00-04:00",
        "expires_at": "2026-09-02T00:00:00-04:00",
        "source_manifest_sha256": "a" * 64,
        "risk_basis": "stress_atr_bps",
        "accounts": {
            "U123": {
                "remaining_long_bps": 5.0,
                "remaining_short_bps": 0.0,
                "remaining_gross_bps": 5.0,
            }
        },
        "reservations": {},
    }
    atomic_write_json(path, payload)
    kwargs = {
        "path": path,
        "lock_path": lock,
        "entry_date": "2026-09-01",
        "account_requirements": {
            "U123": PortfolioRequirement(5.0, 0.0, 5.0)
        },
        "signal_ids": ["sig"],
        "expected_manifest_sha256": "a" * 64,
        "now": pd.Timestamp("2026-09-01 09:30", tz="America/New_York"),
    }
    with reserve_portfolio_capacity(owner_token="first", **kwargs) as (
        capacities,
        _digest,
    ):
        assert capacities["U123"].remaining_gross_bps == 0
    with (
        pytest.raises(RuntimeError, match="capacity changed"),
        reserve_portfolio_capacity(
            owner_token="second",
            **{**kwargs, "signal_ids": ["sig2"]},
        ),
    ):
        pass
    stored = read_json(path)
    assert set(stored["reservations"]) == {"first"}


def _paper_proof() -> dict:
    return {
        "protocol": "legend-ibkr-paper-proof-v1",
        "strategy_version": "legend-etf-original-v1",
        "source_manifest_sha256": "a" * 64,
        "created_at": "2026-09-01T09:00:00-04:00",
        "entry_date": "2026-09-01",
        "paper_account": "DU123",
        "paper_endpoint": {"host": "127.0.0.1", "port": 4002, "client_id": 156},
        "order_ref_echoes": [
            {
                "etf": "SPY",
                "direction": 1,
                "expected": "SPY|BUY|Legend EMA ETF|2026-09-01",
                "open_order": "SPY|BUY|Legend EMA ETF|2026-09-01",
                "execution": "SPY|BUY|Legend EMA ETF|2026-09-01",
            },
            {
                "etf": "SPY",
                "direction": -1,
                "expected": "SPY|SELL|Legend EMA ETF|2026-09-01",
                "open_order": "SPY|SELL|Legend EMA ETF|2026-09-01",
                "execution": "SPY|SELL|Legend EMA ETF|2026-09-01",
            },
        ],
        "entry_parent_tif": "IOC",
        "oca_type": 2,
        "target_revision_clocks": ["09:46", "10:01", "10:16"],
        "max_target_revision_latency_ms": 250.0,
        "drills": {
            "one_share_long": True,
            "one_share_short": True,
            "partial_fill": True,
            "restart_recovery": True,
            "disconnect_recovery": True,
            "order_ref_open_order_echo": True,
            "order_ref_execution_echo": True,
            "oca_type_2_time_exit": True,
            "ioc_parent_partial_children_active": True,
        },
        "time_exit": {
            "scheduled_at": "2026-09-01T10:30:00-04:00",
            "filled_at": "2026-09-01T10:30:01-04:00",
            "delay_seconds": 1.0,
            "remaining_owned_shares": 0,
            "working_orders_after": 0,
            "over_exit": False,
            "target_terminal": True,
        },
        "reviewed_attestation": "I_REVIEWED_LEGEND_IBKR_PAPER_PROOF",
    }


def test_paper_proof_requires_exact_long_and_short_order_ref_echoes(tmp_path):
    path = tmp_path / "paper_proof.json"
    proof = _paper_proof()
    atomic_write_json(path, proof)
    validate_paper_proof(
        path,
        expected_sha256=file_sha256(path),
        expected_manifest_sha256="a" * 64,
        now=pd.Timestamp("2026-09-01 11:00", tz="America/New_York"),
    )
    proof["order_ref_echoes"][1]["execution"] = "x"
    atomic_write_json(path, proof)
    with pytest.raises(RuntimeError, match="exact Legend orderRef"):
        validate_paper_proof(
            path,
            expected_sha256=file_sha256(path),
            expected_manifest_sha256="a" * 64,
            now=pd.Timestamp("2026-09-01 11:00", tz="America/New_York"),
        )


def test_fresh_tick_wait_executes_real_monotonic_boundary():
    ticker = SimpleNamespace(tickByTicks=[])

    class TickIB:
        def isConnected(self):
            return True

        def sleep(self, _seconds):
            ticker.tickByTicks.append(SimpleNamespace(price=500.0))

    connection = IBKRConnection(
        Endpoint("feed", "127.0.0.1", 7496, 154, "U123"), live=False
    )
    connection.ib = TickIB()
    connection.wait_for_new_trade_tick(
        ticker, after_count=0, timeout_seconds=0.1
    )
    assert len(ticker.tickByTicks) == 1


def test_fresh_empty_open_order_snapshot_rejects_cached_ghost():
    ghost = _target_trade()

    class SnapshotIB:
        def isConnected(self):
            return True

        def reqAllOpenOrders(self):
            return []

        def openTrades(self):
            return [ghost]

        def sleep(self, _seconds):
            return None

    connection = IBKRConnection(
        Endpoint("primary", "127.0.0.1", 7496, 155, "U123"), live=True
    )
    connection.ib = SnapshotIB()
    assert (
        connection.attributed_orders(
            ghost.order.orderRef, expected_con_id=ghost.contract.conId
        )
        == []
    )


@pytest.mark.parametrize(
    ("actual", "foreign", "should_cancel"),
    [(0.0, False, True), (5.0, False, True), (12.0, False, False), (10.0, True, True)],
)
def test_physical_isolation_contains_only_reversal_collision(
    tmp_path, monkeypatch, actual, foreign, should_cancel
):
    store = StateStore(tmp_path / "state.json", tmp_path / "audit.jsonl")
    session = LegendSession(
        plan_path=tmp_path / "plan.json",
        state_store=store,
        account_labels=["primary"],
        live_requested=True,
    )
    context = MarketContext(
        root="ES",
        etf="SPY",
        setup_date="2026-08-31",
        contract=SimpleNamespace(symbol="SPY", conId=1),
        initial_ema=500.0,
        atr14=5.0,
        ex_dividend=False,
        dividend_detail="test",
    )

    class IsolationConnection:
        endpoint = Endpoint("primary", "127.0.0.1", 7496, 155, "U123")

        def assert_symbol_isolated(self, *_args, **_kwargs):
            raise PhysicalIsolationMismatch(
                "mismatch",
                actual_position=actual,
                expected_position=10.0,
                foreign_working_orders=foreign,
            )

        def reconnect(self):
            return None

    cleared: list[bool] = []

    def clear_orders(*_args, **_kwargs):
        cleared.append(True)
        return True

    monkeypatch.setattr(session, "_clear_reference_orders", clear_orders)
    monkeypatch.setattr(
        session,
        "_attributed_fills",
        lambda *_args, **_kwargs: [
            SimpleNamespace(exec_id="entry.0", action="BOT", shares=10)
        ],
    )
    state = {"state": "open"}
    with pytest.raises(PhysicalIsolationMismatch):
        session._enforce_physical_isolation(
            signal_id="sig",
            connection=IsolationConnection(),
            context=context,
            reference="SPY|BUY|Legend EMA ETF|2026-09-01",
            direction=1,
            remaining=10,
            state=state,
        )
    assert bool(cleared) is should_cancel
    assert state["isolation_orders_preserved"] is (not should_cancel)


def test_physical_isolation_refreshes_partial_target_fill_before_cancelling(
    tmp_path, monkeypatch
):
    store = StateStore(tmp_path / "state.json", tmp_path / "audit.jsonl")
    session = LegendSession(
        plan_path=tmp_path / "plan.json",
        state_store=store,
        account_labels=["primary"],
        live_requested=True,
    )
    context = MarketContext(
        root="ES",
        etf="SPY",
        setup_date="2026-08-31",
        contract=SimpleNamespace(symbol="SPY", conId=1),
        initial_ema=500.0,
        atr14=5.0,
        ex_dividend=False,
        dividend_detail="test",
    )

    class PartialFillConnection:
        endpoint = Endpoint("primary", "127.0.0.1", 7496, 155, "U123")

        def assert_symbol_isolated(self, *_args, **kwargs):
            expected = float(kwargs["virtual_quantity"])
            if expected != 5.0:
                raise PhysicalIsolationMismatch(
                    "target filled five shares during proof",
                    actual_position=5.0,
                    expected_position=expected,
                )

        def reconnect(self):
            return None

    monkeypatch.setattr(
        session,
        "_attributed_fills",
        lambda *_args, **_kwargs: [
            SimpleNamespace(exec_id="entry.0", action="BOT", shares=10),
            SimpleNamespace(exec_id="exit.0", action="SLD", shares=5),
        ],
    )
    monkeypatch.setattr(
        session,
        "_clear_reference_orders",
        lambda *_args, **_kwargs: pytest.fail("correct partial OCA was cancelled"),
    )
    state = {"state": "open"}
    assert (
        session._enforce_physical_isolation(
            signal_id="sig",
            connection=PartialFillConnection(),
            context=context,
            reference="SPY|BUY|Legend EMA ETF|2026-09-01",
            direction=1,
            remaining=10,
            state=state,
        )
        == 5
    )
    assert state["isolation_fill_refresh_to"] == 5


def test_next_session_terminal_correction_requarantines_without_flatten(tmp_path):
    store = StateStore(tmp_path / "state.json", tmp_path / "audit.jsonl")
    store.put(
        "old-sig",
        {
            "state": "complete",
            "account": "U123",
            "etf": "SPY",
            "entry_date": "2026-08-31",
            "remaining_owned_shares": 0,
            "working_orders_cleared": True,
            "terminal_proved_at": "2026-08-31T14:40:00Z",
            "quarantine_required": True,
            "quarantine_active": False,
            "quarantine_release_pending": False,
        },
    )
    session = LegendSession(
        plan_path=tmp_path / "plan.json",
        state_store=store,
        account_labels=["primary"],
        live_requested=True,
    )
    session.entry_date = "2026-09-01"
    session.gate = LiveGate(True, True, True, frozenset({"U123"}))
    session.runtime_environment = {
        "LEGEND_ETF_RESERVATION_DIR": str(tmp_path / "reservations")
    }
    endpoint = Endpoint("primary", "127.0.0.1", 7496, 155, "U123")
    session.endpoints = [endpoint]

    class AuditConnection:
        def stock(self, symbol):
            return SimpleNamespace(symbol=symbol, conId={"SPY": 1, "QQQ": 2, "IWM": 3}[symbol])

        def assert_symbol_clear(self, contract):
            if contract.symbol == "SPY":
                raise RuntimeError("unexpected physical residue")

    session.accounts = {"primary": AuditConnection()}
    with pytest.raises(RuntimeError, match="terminal-correction"):
        session._audit_recent_terminal_corrections()
    record = store.get("old-sig")
    assert record["state"] == "critical_terminal_correction"
    assert record["quarantine_active"] is True
    assert "automatic flatten prohibited" in record["critical_reason"]
    assert session.reservations is not None
    session.reservations.close()


def test_attested_external_quarantine_reconciliation_requires_every_true_result(
    tmp_path, monkeypatch
):
    session = LegendSession(
        plan_path=tmp_path / "plan.json",
        state_store=StateStore(tmp_path / "state.json", tmp_path / "audit.jsonl"),
        account_labels=["primary"],
        live_requested=True,
    )
    endpoint = Endpoint("primary", "127.0.0.1", 7496, 155, "U123")
    facade = object()
    connection = SimpleNamespace(
        ib=object(), broker_snapshot_facade=lambda: facade
    )
    session.endpoints = [endpoint]
    session.accounts = {"primary": connection}
    config = tmp_path / "reservation.json"
    config.write_text("{}\n", encoding="utf-8")
    calls: list[tuple[object, str, Path]] = []

    class Guard:
        @staticmethod
        def reconcile_all_external_quarantines(
            broker, *, account, config_path
        ):
            calls.append((broker, account, config_path))
            return {"SPY": False}

    monkeypatch.setattr(
        "legend_etf.session.load_attested_external_guard", lambda _manifest: Guard
    )
    with pytest.raises(RuntimeError, match="not broker-proved terminal"):
        session._reconcile_attested_external_quarantines(
            {"reservation_config": {"path": str(config)}}
        )
    assert calls == [(facade, "U123", config.resolve())]


@pytest.mark.parametrize(
    "result",
    [None, {"SPY": "yes"}, {1: True}],
)
def test_attested_external_quarantine_reconciliation_rejects_invalid_results(
    tmp_path, monkeypatch, result
):
    session = LegendSession(
        plan_path=tmp_path / "plan.json",
        state_store=StateStore(tmp_path / "state.json", tmp_path / "audit.jsonl"),
        account_labels=["primary"],
        live_requested=True,
    )
    endpoint = Endpoint("primary", "127.0.0.1", 7496, 155, "U123")
    session.endpoints = [endpoint]
    session.accounts = {
        "primary": SimpleNamespace(
            ib=object(), broker_snapshot_facade=lambda: object()
        )
    }
    config = tmp_path / "reservation.json"
    config.write_text("{}\n", encoding="utf-8")
    guard = SimpleNamespace(
        reconcile_all_external_quarantines=lambda *_args, **_kwargs: result
    )
    monkeypatch.setattr(
        "legend_etf.session.load_attested_external_guard", lambda _manifest: guard
    )
    with pytest.raises(RuntimeError, match="invalid result"):
        session._reconcile_attested_external_quarantines(
            {"reservation_config": {"path": str(config)}}
        )


def test_correction_audit_reconciles_external_guard_before_residue_audit(
    tmp_path, monkeypatch
):
    session = LegendSession(
        plan_path=tmp_path / "plan.json",
        state_store=StateStore(tmp_path / "state.json", tmp_path / "audit.jsonl"),
        account_labels=["primary"],
        live_requested=True,
    )
    endpoint = Endpoint("primary", "127.0.0.1", 7496, 155, "U123")
    connection = SimpleNamespace(
        endpoint=endpoint,
        is_connected=lambda: True,
    )
    session.accounts = {"primary": connection}
    values = {"LEGEND_ETF_GUARD_MANIFEST": "unused"}
    manifest = {"reservation_config": {"path": str(tmp_path / "config.json")}}
    monkeypatch.setattr(session, "_read_live_runtime", lambda: values)
    monkeypatch.setattr(
        session, "_validate_shared_guard_manifest", lambda _values: manifest
    )
    monkeypatch.setattr(
        "legend_etf.session.endpoints_from_env",
        lambda _labels, environment: [endpoint],
    )
    monkeypatch.setattr(
        "legend_etf.session.validate_correction_audit_gate",
        lambda **_kwargs: LiveGate(True, True, True, frozenset({"U123"})),
    )
    events: list[str] = []
    monkeypatch.setattr(
        session,
        "_reconcile_attested_external_quarantines",
        lambda _manifest: events.append("external_reconcile"),
    )
    monkeypatch.setattr(
        session,
        "_audit_recent_terminal_corrections",
        lambda: events.append("residue_audit"),
    )
    assert session.audit_terminal_corrections()["audited"] is True
    assert events == ["external_reconcile", "residue_audit"]


def test_lock_loser_cannot_overwrite_live_lease(tmp_path, monkeypatch):
    store = StateStore(tmp_path / "state.json", tmp_path / "audit.jsonl")
    session = LegendSession(
        plan_path=tmp_path / "plan.json",
        state_store=store,
        account_labels=["primary"],
        live_requested=True,
    )
    touches: list[str] = []

    @contextmanager
    def busy_lock():
        raise RuntimeError("busy")
        yield  # pragma: no cover

    monkeypatch.setattr(store, "exclusive_session", busy_lock)
    monkeypatch.setattr(session, "_touch_live_lease", touches.append)
    with pytest.raises(RuntimeError, match="busy"):
        session.run()
    assert touches == []


def test_selftest_named_live_mutator_is_in_guard_inventory(tmp_path):
    source = tmp_path / "div_adjust_selftest.py"
    source.write_text("def go(ib, c, o):\n    return ib.placeOrder(c, o)\n", encoding="utf-8")
    assert "div_adjust_selftest.py" in discover_executor_python_files(tmp_path)
    assert "div_adjust_selftest.py" in discover_broker_mutation_files(tmp_path)


def test_attested_guard_loader_uses_exact_bytes_and_restores_budget_module(
    tmp_path, monkeypatch
):
    executor = tmp_path / "executor"
    executor.mkdir()
    budget = executor / "legend_portfolio_budget.py"
    guard_path = executor / "legend_reservation_guard.py"
    budget.write_text("SENTINEL = 'attested-budget'\n", encoding="utf-8")
    guard_path.write_text(
        "from legend_portfolio_budget import SENTINEL\n"
        "def reconcile_all_external_quarantines(*_args, **_kwargs):\n"
        "    return {'SPY': SENTINEL == 'attested-budget'}\n",
        encoding="utf-8",
    )
    tree_hash = "b" * 64
    manifest = {
        "executor_root": str(executor.resolve()),
        "executor_source_tree_sha256": tree_hash,
        "executors": [
            {
                "label": path.name,
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
            for path in (guard_path, budget)
        ],
    }
    prior_budget = SimpleNamespace(SENTINEL="pre-existing-budget")
    monkeypatch.setitem(sys.modules, "legend_portfolio_budget", prior_budget)
    unique_name = f"_legend_attested_external_guard_{tree_hash[:16]}"
    try:
        loaded = load_attested_external_guard(manifest)
        assert loaded.reconcile_all_external_quarantines() == {"SPY": True}
        assert sys.modules["legend_portfolio_budget"] is prior_budget

        guard_path.write_text("# changed after attestation\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="changed before import"):
            load_attested_external_guard(manifest)
        assert sys.modules["legend_portfolio_budget"] is prior_budget
    finally:
        sys.modules.pop(unique_name, None)


def test_manifest_builder_output_validates_end_to_end(tmp_path, monkeypatch):
    from scripts import build_legend_executor_guard_manifest as builder

    executor = tmp_path / "executor"
    executor.mkdir()
    (executor / "legend_reservation_guard.py").write_text(
        "def guarded_place_order(ib, contract, order):\n"
        "    return ib.placeOrder(contract, order)\n\n"
        "def guarded_cancel_order(ib, order):\n"
        "    return ib.cancelOrder(order)\n\n"
        "def guarded_global_cancel(ib):\n"
        "    return ib.reqGlobalCancel()\n",
        encoding="utf-8",
    )
    (executor / "legend_portfolio_budget.py").write_text(
        'LEGEND_PORTFOLIO_BUDGET_PROTOCOL = "legend-equity-index-risk-budget-v3"\n\n'
        "def reserve_equity_index_capacity():\n"
        "    return None\n",
        encoding="utf-8",
    )
    (executor / "contract_reference.json").write_text("{}\n", encoding="utf-8")
    (executor / "executor.py").write_text(
        "from legend_reservation_guard import guarded_place_order\n",
        encoding="utf-8",
    )
    tree_hash = source_tree_sha256(builder.discover_executor_runtime_files(executor))
    receipt = tmp_path / "integration.json"
    atomic_write_json(
        receipt,
        {
            "protocol": INTEGRATION_RECEIPT_PROTOCOL,
            "created_at": "2026-09-01T12:00:00Z",
            "executor_root": str(executor.resolve()),
            "source_tree_sha256": tree_hash,
            "passed_tests": sorted(REQUIRED_INTEGRATION_TESTS),
            "reviewed_attestation": INTEGRATION_REVIEW_TOKEN,
        },
    )
    runtime = tmp_path / "runtime"
    reservations = tmp_path / "reservations"
    runtime.mkdir()
    reservations.mkdir()
    config = tmp_path / "reservation.json"
    manifest = tmp_path / "manifest.json"
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    parity = tmp_path / "candidate_parity.json"
    atomic_write_json(parity, _candidate_parity_evidence(root, tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_legend_executor_guard_manifest.py",
            "--reservation-dir",
            str(reservations),
            "--runtime-dir",
            str(runtime),
            "--legend-root",
            str(root),
            "--executor-root",
            str(executor),
            "--reservation-config",
            str(config),
            "--integration-receipt",
            str(receipt),
            "--candidate-parity-evidence",
            str(parity),
            "--manifest",
            str(manifest),
            "--reviewed-attestation",
            builder.REVIEW_TOKEN,
        ],
    )
    assert builder.main() == 0
    payload = validate_guard_manifest(
        manifest,
        expected_reservation_dir=reservations,
        expected_runtime_dir=runtime,
        expected_legend_root=root,
        expected_executor_root=executor,
        expected_sha256=file_sha256(manifest),
    )
    assert payload["executor_source_tree_sha256"] == tree_hash
    marker = executor / GUARD_REQUIRED_MARKER_NAME
    assert read_json(marker) == {
        "protocol": "legend-account-symbol-lock-v1",
        "executor_root": str(executor.resolve()),
    }
    atomic_write_json(marker, {"protocol": "changed"})
    with pytest.raises(RuntimeError, match="required marker"):
        validate_guard_manifest(
            manifest,
            expected_reservation_dir=reservations,
            expected_runtime_dir=runtime,
            expected_legend_root=root,
            expected_executor_root=executor,
            expected_sha256=file_sha256(manifest),
        )


def test_final_snapshot_retries_when_a_partial_fill_changes_virtual_quantity():
    reference = "SPY|BUY|Legend EMA ETF|2026-09-01"

    def raw_fill(exec_id, side, shares, order_id):
        return SimpleNamespace(
            execution=SimpleNamespace(
                execId=exec_id,
                acctNumber="U123",
                orderRef=reference,
                clientId=155,
                orderId=order_id,
                side=side,
                shares=shares,
                price=500.0,
                time="2026-09-01T14:30:00Z",
            ),
            contract=SimpleNamespace(conId=1),
        )

    entry = raw_fill("entry", "BOT", 10, 1)
    partial = raw_fill("target", "SLD", 5, 2)

    class SnapshotIB:
        RequestTimeout = 5.0

        def __init__(self):
            self.execution_snapshots = [
                [entry],
                [entry, partial],
                [entry, partial],
                [entry, partial],
            ]

        def isConnected(self):
            return True

        def reqExecutions(self):
            return self.execution_snapshots.pop(0)

        def fills(self):
            return []

        def reqAllOpenOrders(self):
            return []

        def positions(self, account):
            assert account == "U123"
            return [
                SimpleNamespace(
                    contract=SimpleNamespace(conId=1), position=5.0
                )
            ]

        def sleep(self, _seconds):
            return None

    connection = IBKRConnection(
        Endpoint("primary", "127.0.0.1", 7496, 155, "U123"), live=True
    )
    connection.ib = SnapshotIB()
    snapshot = connection.legend_exit_snapshot(
        {reference: 1}, request_timeout_seconds=0.3
    )
    assert owned_quantity(snapshot.fills_by_reference[reference], 1) == 5
    assert snapshot.physical_positions_by_con_id[1] == 5
    assert connection.ib.execution_snapshots == []


def test_exact_immediate_exit_rejects_ioc_parent_and_stale_quantity(tmp_path):
    endpoint = Endpoint("primary", "127.0.0.1", 7496, 155, "U123")
    connection = SimpleNamespace(endpoint=endpoint)
    context = MarketContext(
        root="ES",
        etf="SPY",
        setup_date="2026-08-31",
        contract=SimpleNamespace(symbol="SPY", conId=1),
        initial_ema=500.0,
        atr14=5.0,
        ex_dividend=False,
        dividend_detail="test",
    )
    trade = ManagedTrade(
        signal_id="sig",
        connection=connection,
        context=context,
        direction=1,
        order_ref="SPY|BUY|Legend EMA ETF|2026-09-01",
        target_trade=None,
        current_ema=500.0,
        state={"state": "open"},
    )

    def order_trade(action, remaining):
        return SimpleNamespace(
            contract=context.contract,
            order=SimpleNamespace(
                orderType="MKT",
                goodAfterTime="",
                action=action,
                account="U123",
                orderRef=trade.order_ref,
                clientId=155,
                permId=99,
            ),
            orderStatus=SimpleNamespace(
                status="Submitted", remaining=float(remaining)
            ),
        )

    exact = order_trade("SELL", 10)
    parent = order_trade("BUY", 10)
    stale = order_trade("SELL", 12)
    assert LegendSession._is_exact_immediate_market_exit(exact, trade, 10)
    assert not LegendSession._is_exact_immediate_market_exit(parent, trade, 10)
    assert not LegendSession._is_exact_immediate_market_exit(stale, trade, 10)


def test_final_exit_batch_transmits_healthy_peer_before_failed_account_recovery(
    tmp_path, monkeypatch
):
    events: list[str] = []
    primary_endpoint = Endpoint("primary", "127.0.0.1", 7496, 155, "U1")
    pa_endpoint = Endpoint("pa", "127.0.0.1", 4002, 156, "DU2")

    class FailedConnection:
        endpoint = primary_endpoint

        def legend_exit_snapshot(self, *_args, **_kwargs):
            events.append("primary_snapshot_failed")
            raise RuntimeError("poisoned endpoint")

    reference = "QQQ|BUY|Legend EMA ETF|2026-09-01"
    target = SimpleNamespace(
        contract=SimpleNamespace(symbol="QQQ", conId=2),
        order=SimpleNamespace(orderType="LMT", goodAfterTime=""),
        orderStatus=SimpleNamespace(status="Submitted"),
    )
    gat = SimpleNamespace(
        contract=SimpleNamespace(symbol="QQQ", conId=2),
        order=SimpleNamespace(
            orderType="MKT", goodAfterTime="20260901 10:30:00 US/Eastern"
        ),
        orderStatus=SimpleNamespace(status="Submitted"),
    )
    entry_fills = [FillEvent("entry", 1, "BOT", 10, 500.0, "t")]
    before_cancel = LegendAccountSnapshot(
        fills_by_reference={reference: entry_fills},
        working_orders_by_reference={reference: [target, gat]},
        physical_positions_by_con_id={2: 10.0},
        foreign_working_con_ids=frozenset(),
    )
    after_cancel = LegendAccountSnapshot(
        fills_by_reference={reference: entry_fills},
        working_orders_by_reference={reference: []},
        physical_positions_by_con_id={2: 10.0},
        foreign_working_con_ids=frozenset(),
    )

    class HealthyConnection:
        endpoint = pa_endpoint

        def __init__(self):
            self.snapshots = [before_cancel, after_cancel]

        def legend_exit_snapshot(self, *_args, **_kwargs):
            events.append("pa_snapshot")
            return self.snapshots.pop(0)

        def request_cancel_orders(self, orders):
            assert list(orders) == [target, gat]
            events.append("pa_cancel")

        def place_emergency_exit(self, **kwargs):
            assert kwargs["shares"] == 10
            events.append("pa_place_emergency")
            return SimpleNamespace(order=SimpleNamespace(orderId=77))

    spy = MarketContext(
        root="ES",
        etf="SPY",
        setup_date="2026-08-31",
        contract=SimpleNamespace(symbol="SPY", conId=1),
        initial_ema=500.0,
        atr14=5.0,
        ex_dividend=False,
        dividend_detail="test",
    )
    qqq = MarketContext(
        root="NQ",
        etf="QQQ",
        setup_date="2026-08-31",
        contract=SimpleNamespace(symbol="QQQ", conId=2),
        initial_ema=500.0,
        atr14=5.0,
        ex_dividend=False,
        dividend_detail="test",
    )
    failed = FailedConnection()
    healthy = HealthyConnection()
    managed = [
        ManagedTrade(
            "primary-sig",
            failed,
            spy,
            1,
            "SPY|BUY|Legend EMA ETF|2026-09-01",
            None,
            500.0,
            {"state": "open"},
        ),
        ManagedTrade(
            "pa-sig",
            healthy,
            qqq,
            1,
            reference,
            None,
            500.0,
            {"state": "open"},
        ),
    ]
    session = LegendSession(
        plan_path=tmp_path / "plan.json",
        state_store=StateStore(tmp_path / "state.json", tmp_path / "audit.jsonl"),
        account_labels=["primary", "pa"],
        live_requested=True,
    )
    session.entry_date = "2026-09-01"
    session.contexts = [spy, qqq]
    monkeypatch.setattr(session, "_wait_management_until", lambda *_args: None)
    critical: set[str] = set()
    transmitted = session._transmit_final_exit_batch(managed, critical)
    assert transmitted == {"pa-sig"}
    assert "primary-sig" in critical
    assert events[-1] == "pa_place_emergency"
    assert session.store.get("pa-sig")["emergency_order_id"] == 77


def test_final_exit_batch_cancels_duplicate_immediate_exits_before_refresh(
    tmp_path, monkeypatch
):
    reference = "SPY|BUY|Legend EMA ETF|2026-09-01"
    endpoint = Endpoint("primary", "127.0.0.1", 7496, 155, "U123")
    context = MarketContext(
        root="ES",
        etf="SPY",
        setup_date="2026-08-31",
        contract=SimpleNamespace(symbol="SPY", conId=1),
        initial_ema=500.0,
        atr14=5.0,
        ex_dividend=False,
        dividend_detail="test",
    )

    def immediate(order_id):
        return SimpleNamespace(
            contract=context.contract,
            order=SimpleNamespace(
                orderId=order_id,
                permId=order_id + 100,
                clientId=155,
                orderType="MKT",
                goodAfterTime="",
                action="SELL",
                account="U123",
                orderRef=reference,
            ),
            orderStatus=SimpleNamespace(status="Submitted", remaining=10.0),
        )

    first = immediate(10)
    second = immediate(11)
    fills = [FillEvent("entry", 1, "BOT", 10, 500.0, "t")]
    events: list[str] = []

    class Connection:
        def __init__(self, endpoint_value):
            self.endpoint = endpoint_value
            self.snapshots = [
                LegendAccountSnapshot(
                    {reference: fills},
                    {reference: [first, second]},
                    {1: 10.0},
                    frozenset(),
                ),
                LegendAccountSnapshot(
                    {reference: fills},
                    {reference: []},
                    {1: 10.0},
                    frozenset(),
                ),
            ]

        def legend_exit_snapshot(self, *_args, **_kwargs):
            events.append("snapshot")
            return self.snapshots.pop(0)

        def request_cancel_orders(self, orders):
            assert list(orders) == [first, second]
            events.append("cancel_duplicates")

        def place_emergency_exit(self, **kwargs):
            assert kwargs["shares"] == 10
            events.append("place_replacement")
            return SimpleNamespace(order=SimpleNamespace(orderId=12))

    connection = Connection(endpoint)
    trade = ManagedTrade(
        "sig",
        connection,
        context,
        1,
        reference,
        None,
        500.0,
        {"state": "open"},
    )
    session = LegendSession(
        plan_path=tmp_path / "plan.json",
        state_store=StateStore(tmp_path / "state.json", tmp_path / "audit.jsonl"),
        account_labels=["primary"],
        live_requested=True,
    )
    session.entry_date = "2026-09-01"
    session.contexts = [context]
    monkeypatch.setattr(session, "_wait_management_until", lambda *_args: None)
    session._transmit_final_exit_batch([trade], set())
    assert events == [
        "snapshot",
        "cancel_duplicates",
        "snapshot",
        "place_replacement",
    ]


def test_final_exit_batch_does_not_replace_target_that_fills_during_cancel(
    tmp_path, monkeypatch
):
    reference = "SPY|BUY|Legend EMA ETF|2026-09-01"
    endpoint = Endpoint("primary", "127.0.0.1", 7496, 155, "U123")
    context = MarketContext(
        root="ES",
        etf="SPY",
        setup_date="2026-08-31",
        contract=SimpleNamespace(symbol="SPY", conId=1),
        initial_ema=500.0,
        atr14=5.0,
        ex_dividend=False,
        dividend_detail="test",
    )
    target = SimpleNamespace(
        contract=context.contract,
        order=SimpleNamespace(orderType="LMT", goodAfterTime=""),
        orderStatus=SimpleNamespace(status="Submitted"),
    )
    gat = SimpleNamespace(
        contract=context.contract,
        order=SimpleNamespace(
            orderType="MKT", goodAfterTime="20260901 10:30:00 US/Eastern"
        ),
        orderStatus=SimpleNamespace(status="Submitted"),
    )
    entry = FillEvent("entry", 1, "BOT", 10, 500.0, "t1")
    exit_fill = FillEvent("target", 2, "SLD", 10, 500.1, "t2")

    class Connection:
        def __init__(self, endpoint_value):
            self.endpoint = endpoint_value
            self.snapshots = [
                LegendAccountSnapshot(
                    {reference: [entry]},
                    {reference: [target, gat]},
                    {1: 10.0},
                    frozenset(),
                ),
                LegendAccountSnapshot(
                    {reference: [entry, exit_fill]},
                    {reference: []},
                    {1: 0.0},
                    frozenset(),
                ),
            ]
            self.placed = False

        def legend_exit_snapshot(self, *_args, **_kwargs):
            return self.snapshots.pop(0)

        def request_cancel_orders(self, _orders):
            return None

        def place_emergency_exit(self, **_kwargs):
            self.placed = True
            raise AssertionError("flat target fill must not be replaced")

    connection = Connection(endpoint)
    trade = ManagedTrade(
        "sig",
        connection,
        context,
        1,
        reference,
        None,
        500.0,
        {"state": "open"},
    )
    session = LegendSession(
        plan_path=tmp_path / "plan.json",
        state_store=StateStore(tmp_path / "state.json", tmp_path / "audit.jsonl"),
        account_labels=["primary"],
        live_requested=True,
    )
    session.entry_date = "2026-09-01"
    session.contexts = [context]
    monkeypatch.setattr(session, "_wait_management_until", lambda *_args: None)
    assert session._transmit_final_exit_batch([trade], set()) == set()
    assert connection.placed is False
    assert session.store.get("sig")["remaining_owned_shares"] == 0
