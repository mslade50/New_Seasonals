from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from legend_etf.config import PA_RISK, PRIMARY_RISK
from legend_etf.databento_source import PAID_CONFIRMATION, Quote, enforce_cost_gate
from legend_etf.ibkr_adapter import (
    Endpoint,
    FillEvent,
    IBKRConnection,
    endpoints_from_env,
    owned_quantity,
    validate_live_gate,
)
from legend_etf.reservations import (
    CRITICAL_RUNTIME_DISTRIBUTIONS,
    PROTOCOL_VERSION,
    REQUIRED_LEGEND_RUNTIME_FILES,
    candidate_pipeline_attestation,
    file_sha256,
    quarantine_path,
)
from legend_etf.session import LegendSession, MarketContext
from legend_etf.sizing import SizeRequest, risk_profile_from_env, size_batch
from legend_etf.storage import (
    StateStore,
    atomic_write_json,
    signal_identity,
)


def _parity_input_record(path: Path) -> dict[str, object]:
    source = path.resolve()
    stat = source.stat()
    return {
        "path": str(source),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": file_sha256(source),
    }


def _write_parity_evidence(path: Path, *, root: Path) -> None:
    source = path.parent / "native_input.parquet"
    source.write_bytes(b"native archive fixture")
    atomic_write_json(path, {
        "protocol": "legend-etf-native-candidate-parity-v1", "status": "pass",
        "completed_at": "2026-09-16T12:00:00Z", "runtime_seconds": 1.0,
        "range": {"start": "2012-01-01", "end": "2026-08-28"},
        "inputs": {symbol: _parity_input_record(source) for symbol in ("SPY", "QQQ")},
        "counts": {symbol: {"evaluated": 3000, "blocked_history": 0,
                            "reference": 100, "production": 100, "mismatches": 0,
                            "max_ema_delta": 0., "max_ratio_delta": 0.}
                   for symbol in ("SPY", "QQQ")},
        "candidate_pipeline": candidate_pipeline_attestation(root),
    })


def test_primary_and_pa_size_independently_with_smaller_shorts():
    requests = [
        SizeRequest("ES", "SPY", 1, atr14=5.0, reference_price=500),
        SizeRequest("NQ", "QQQ", -1, atr14=5.0, reference_price=500),
    ]
    primary = size_batch(requests, nlv=1_000_000, profile=PRIMARY_RISK)
    pa = size_batch(requests, nlv=1_000_000, profile=PA_RISK)
    assert primary[0].shares == 160  # $1,000 / (1.25 * $5)
    assert primary[1].shares == 80
    assert pa[0].shares == 80
    assert pa[1].shares == 40
    assert primary[1].requested_bps == primary[0].requested_bps / 2


def test_cluster_cap_is_gross_pro_rata_and_never_forces_one_share():
    from legend_etf.config import RiskProfile

    profile = RiskProfile(
        long_bps=20,
        short_bps=20,
        cluster_bps=10,
        max_shares_per_root=10_000,
        max_notional_pct=1,
    )
    results = size_batch(
        [
            SizeRequest("ES", "SPY", 1, 10, 100),
            SizeRequest("NQ", "QQQ", 1, 10, 100),
        ],
        nlv=100_000,
        profile=profile,
    )
    assert all(result.scale == pytest.approx(0.25) for result in results)
    assert sum(result.stress_risk_usd for result in results) <= 100
    tiny = size_batch(
        [SizeRequest("RTY", "IWM", -1, 1000, 200)],
        nlv=1_000,
        profile=PA_RISK,
    )
    assert tiny[0].shares == 0


def test_fill_ledger_is_idempotent_and_exposes_over_exit():
    fills = [
        FillEvent("a", 1, "BOT", 10, 100, "t1"),
        FillEvent("a", 1, "BOT", 10, 100, "t1"),
        FillEvent("b", 2, "SLD", 4, 101, "t2"),
    ]
    assert owned_quantity(fills, 1) == 6
    assert owned_quantity([*fills, FillEvent("c", 2, "SLD", 20, 101, "t3")], 1) == -14


def test_fill_correction_uses_only_latest_exec_revision():
    fills = [
        FillEvent("execution.01", 1, "BOT", 10, 100, "t1"),
        FillEvent("execution.02", 1, "BOT", 7, 100, "t2"),
    ]
    assert owned_quantity(fills, 1) == 7


def test_interrupted_state_becomes_unknown_and_is_not_a_retry(tmp_path):
    store = StateStore(tmp_path / "state.json", tmp_path / "audit.jsonl")
    store.put("signal", {"state": "submitting", "root": "ES"})
    recovered = store.recover_interrupted()
    record = recovered["signals"]["signal"]
    assert record["state"] == "unknown"
    assert record["unknown_reason"] == "restart_during_submitting"
    assert "restart_marked_unknown" in (tmp_path / "audit.jsonl").read_text()


def test_identity_is_account_side_and_version_scoped():
    common = {
        "root": "ES",
        "etf": "SPY",
        "setup_date": "2026-08-31",
        "entry_date": "2026-09-01",
        "direction": 1,
    }
    primary = signal_identity(account="U1", **common)
    pa = signal_identity(account="U2", **common)
    short = signal_identity(account="U1", **{**common, "direction": -1})
    assert len({primary, pa, short}) == 3


def test_live_gate_is_dry_by_default_and_daily_expiring(monkeypatch):
    dry = validate_live_gate(
        live_requested=False, account_ids=["U123"], today=date(2026, 9, 1)
    )
    assert not dry.live
    with pytest.raises(RuntimeError, match="Live gate refused"):
        validate_live_gate(
            live_requested=True, account_ids=["U123"], today=date(2026, 9, 1)
        )
    monkeypatch.setenv("LEGEND_ETF_LIVE_ENABLED", "1")
    monkeypatch.setenv("LEGEND_ETF_LIVE_DATE", "2026-09-01")
    monkeypatch.setenv("LEGEND_ETF_STRATEGY_VERSION", "legend-etf-spy-qqq-v2")
    monkeypatch.setenv("LEGEND_ETF_LIVE_ACCOUNTS", "U123")
    monkeypatch.setenv("LEGEND_ETF_ALLOW_LONGS", "1")
    monkeypatch.setenv("LEGEND_ETF_ALLOW_SHORTS", "0")
    gate = validate_live_gate(
        live_requested=True, account_ids=["U123"], today=date(2026, 9, 1)
    )
    assert gate.live and gate.allow_longs and not gate.allow_shorts
    with pytest.raises(RuntimeError, match="short"):
        gate.require_side(-1)


def test_databento_cost_gate_defaults_to_zero_and_requires_paid_token():
    free = [Quote("ES", 0, 0), Quote("NQ", 0, 0)]
    enforce_cost_gate(free, max_cost_usd=0, paid_confirmation=None)
    paid = [Quote("ES", 0.01, 100)]
    with pytest.raises(RuntimeError, match="exceeds"):
        enforce_cost_gate(paid, max_cost_usd=0, paid_confirmation=None)
    with pytest.raises(RuntimeError, match="explicit financial approval"):
        enforce_cost_gate(paid, max_cost_usd=1, paid_confirmation=None)
    enforce_cost_gate(
        paid, max_cost_usd=1, paid_confirmation=PAID_CONFIRMATION
    )
    with pytest.raises(RuntimeError, match="daily"):
        enforce_cost_gate(
            paid,
            max_cost_usd=0.05,
            paid_confirmation=PAID_CONFIRMATION,
            prior_cost_usd=0.045,
        )
    with pytest.raises(ValueError, match="finite"):
        enforce_cost_gate(free, max_cost_usd=float("nan"), paid_confirmation=None)
    with pytest.raises(RuntimeError, match="invalid cost"):
        enforce_cost_gate(
            [Quote("ES", float("nan"), 1)],
            max_cost_usd=1,
            paid_confirmation=PAID_CONFIRMATION,
        )


def test_runtime_risk_nan_and_duplicate_execution_accounts_fail_closed():
    with pytest.raises(ValueError, match="between"):
        risk_profile_from_env(
            "primary",
            PRIMARY_RISK,
            environment={"LEGEND_ETF_PRIMARY_LONG_BPS": "nan"},
        )
    environment = {
        "LEGEND_ETF_PRIMARY_ACCOUNT": "U1",
        "LEGEND_ETF_PRIMARY_HOST": "127.0.0.1",
        "LEGEND_ETF_PRIMARY_PORT": "7496",
        "LEGEND_ETF_PRIMARY_CLIENT_ID": "155",
        "LEGEND_ETF_PA_ACCOUNT": "U1",
        "LEGEND_ETF_PA_HOST": "127.0.0.1",
        "LEGEND_ETF_PA_PORT": "4001",
        "LEGEND_ETF_PA_CLIENT_ID": "156",
    }
    with pytest.raises(ValueError, match="different IBKR account"):
        endpoints_from_env(["primary", "pa"], environment=environment)


class _FakeClient:
    def __init__(self):
        self.value = 100

    def getReqId(self):
        self.value += 1
        return self.value


class _FakeIB:
    def __init__(self):
        self.client = _FakeClient()
        self.placed = []

    def isConnected(self):
        return True

    def placeOrder(self, contract, order):
        self.placed.append(order)
        return SimpleNamespace(
            contract=contract,
            order=order,
            orderStatus=SimpleNamespace(
                status="Submitted", remaining=float(order.totalQuantity), filled=0.0
            ),
        )

    def sleep(self, _seconds):
        return None


def test_bracket_has_partial_safe_oca_and_server_held_1030_exit():
    endpoint = Endpoint("primary", "127.0.0.1", 7496, 155, "U123")
    connection = IBKRConnection(endpoint, live=True)
    connection.ib = _FakeIB()
    contract = SimpleNamespace(symbol="SPY")
    parent, target, timed = connection.place_bracket(
        contract=contract,
        direction=1,
        shares=10,
        target=501.23,
        entry_date="2026-09-01",
    )
    assert parent.order.transmit is False
    assert target.order.ocaGroup == timed.order.ocaGroup
    assert target.order.ocaType == timed.order.ocaType == 2
    assert target.order.transmit is False and timed.order.transmit is True
    assert timed.order.goodAfterTime == "20260901 10:30:00 US/Eastern"
    assert all(order.orderRef == "SPY|BUY|Legend EMA ETF|2026-09-01" for order in connection.ib.placed)


def test_target_modification_preserves_lifetime_quantity_after_partial_fill():
    endpoint = Endpoint("primary", "127.0.0.1", 7496, 155, "U123")
    connection = IBKRConnection(endpoint, live=True)
    connection.ib = _FakeIB()
    from ib_insync import LimitOrder

    order = LimitOrder("SELL", 10, 100)
    order.orderId = 2
    order.permId = 22
    order.clientId = 155
    order.account = "U123"
    order.orderRef = "SPY|BUY|Legend EMA ETF|2026-09-01"
    trade = SimpleNamespace(
        contract=SimpleNamespace(symbol="SPY", conId=1),
        order=order,
        orderStatus=SimpleNamespace(status="Submitted", remaining=6.0, filled=4.0),
    )
    connection.request_target_modification(trade, new_target=99.9, remaining=6)
    assert order.totalQuantity == 10
    assert order.lmtPrice == 99.9
    assert order.transmit is True
    with pytest.raises(ValueError, match="increase"):
        connection.request_target_modification(trade, new_target=99.8, remaining=7)


def test_restart_reconciles_proven_open_lot_without_retrying_entry(tmp_path):
    store = StateStore(tmp_path / "state.json", tmp_path / "audit.jsonl")
    store.put(
        "sig",
        {
            "state": "submitting",
            "account_label": "primary",
            "account": "U123",
            "root": "ES",
            "etf": "SPY",
            "entry_date": "2026-09-01",
            "direction": 1,
            "order_ref": "SPY|BUY|Legend EMA ETF|2026-09-01",
            "initial_ema": 500.0,
        },
    )
    target = SimpleNamespace(
        order=SimpleNamespace(
            orderId=2,
            permId=22,
            clientId=155,
            orderType="LMT",
            action="SELL",
            account="U123",
            orderRef="SPY|BUY|Legend EMA ETF|2026-09-01",
            parentId=1,
            ocaGroup="oca",
            ocaType=2,
            tif="DAY",
            outsideRth=False,
            totalQuantity=10,
            transmit=False,
            goodAfterTime="",
            lmtPrice=500.0,
        ),
        orderStatus=SimpleNamespace(status="Submitted", remaining=10.0),
    )
    timed = SimpleNamespace(
        order=SimpleNamespace(
            orderId=3,
            permId=23,
            clientId=155,
            orderType="MKT",
            action="SELL",
            account="U123",
            orderRef="SPY|BUY|Legend EMA ETF|2026-09-01",
            parentId=1,
            ocaGroup="oca",
            ocaType=2,
            tif="DAY",
            outsideRth=False,
            totalQuantity=10,
            transmit=True,
            goodAfterTime="20260901 10:30:00 US/Eastern",
        ),
        orderStatus=SimpleNamespace(status="PreSubmitted", remaining=10.0),
    )

    class ReconcileConnection:
        endpoint = Endpoint("primary", "127.0.0.1", 7496, 155, "U123")

        def attributed_fills(self, _reference, *, expected_con_id):
            assert expected_con_id == 1
            return [FillEvent("entry", 1, "BOT", 10, 499, "t")]

        def attributed_orders(self, _reference, *, expected_con_id):
            assert expected_con_id == 1
            return [target, timed]

    session = LegendSession(
        plan_path=tmp_path / "plan.json",
        state_store=store,
        account_labels=["primary"],
        live_requested=True,
    )
    session.entry_date = "2026-09-01"
    session.accounts = {"primary": ReconcileConnection()}
    session.contexts = [
        MarketContext(
            root="ES",
            etf="SPY",
            setup_date="2026-08-31",
                contract=SimpleNamespace(symbol="SPY", conId=1),
            initial_ema=500,
            atr14=5,
            ex_dividend=False,
            dividend_detail="none",
        )
    ]
    managed = session._reconcile_prior_records()
    assert len(managed) == 1
    assert managed[0].allow_target_updates is False
    assert store.get("sig")["state"] == "open_reconciled"


def _exact_exit_trade(
    *, order_id: int, order_type: str, status: str, good_after: str = ""
):
    return SimpleNamespace(
        order=SimpleNamespace(
            orderId=order_id,
            permId=order_id + 100,
            clientId=155,
            orderType=order_type,
            action="SELL",
            account="U123",
            orderRef="SPY|BUY|Legend EMA ETF|2026-09-01",
            parentId=1,
            ocaGroup="oca",
            ocaType=2,
            tif="DAY",
            outsideRth=False,
            totalQuantity=10,
            transmit=order_type == "MKT",
            goodAfterTime=good_after,
            lmtPrice=500.0 if order_type == "LMT" else 0.0,
        ),
        orderStatus=SimpleNamespace(status=status, remaining=10.0),
    )


def test_exit_proof_rejects_pending_extra_parent_and_wrong_target(tmp_path):
    target = _exact_exit_trade(order_id=2, order_type="LMT", status="Submitted")
    timed = _exact_exit_trade(
        order_id=3,
        order_type="MKT",
        status="PreSubmitted",
        good_after="20260901 10:30:00 US/Eastern",
    )

    class Connection:
        endpoint = Endpoint("primary", "127.0.0.1", 7496, 155, "U123")

        def __init__(self):
            self.orders = [target, timed]

        def attributed_orders(self, _reference, *, expected_con_id):
            assert expected_con_id == 1
            return self.orders

    connection = Connection()
    session = LegendSession(
        plan_path=tmp_path / "plan.json",
        state_store=StateStore(tmp_path / "state.json", tmp_path / "audit.jsonl"),
        account_labels=["primary"],
        live_requested=True,
    )
    session.entry_date = "2026-09-01"
    session.contexts = [
        MarketContext(
            root="ES",
            etf="SPY",
            setup_date="2026-08-31",
            contract=SimpleNamespace(symbol="SPY", conId=1),
            initial_ema=500,
            atr14=5,
            ex_dividend=False,
            dividend_detail="none",
        )
    ]
    reference = "SPY|BUY|Legend EMA ETF|2026-09-01"
    assert session._proven_exit_pair(
        connection=connection,
        reference=reference,
        direction=1,
        remaining=10,
        expected_target=500.0,
    ) == (target, timed)
    pending_parent = _exact_exit_trade(
        order_id=1, order_type="MKT", status="PendingSubmit"
    )
    pending_parent.order.action = "BUY"
    connection.orders.append(pending_parent)
    assert session._proven_exit_pair(
        connection=connection,
        reference=reference,
        direction=1,
        remaining=10,
        expected_target=500.0,
    ) is None
    connection.orders = [target, timed]
    assert session._proven_exit_pair(
        connection=connection,
        reference=reference,
        direction=1,
        remaining=10,
        expected_target=499.99,
    ) is None


def test_protection_repair_resizes_after_fill_during_cancel(tmp_path, monkeypatch):
    reference = "SPY|BUY|Legend EMA ETF|2026-09-01"
    fills_sequence = [
        [FillEvent("entry", 1, "BOT", 10, 499, "t1")],
        [
            FillEvent("entry", 1, "BOT", 10, 499, "t1"),
            FillEvent("target", 2, "SLD", 2, 500, "t2"),
        ],
        [
            FillEvent("entry", 1, "BOT", 10, 499, "t1"),
            FillEvent("target", 2, "SLD", 2, 500, "t2"),
        ],
        [
            FillEvent("entry", 1, "BOT", 10, 499, "t1"),
            FillEvent("target", 2, "SLD", 2, 500, "t2"),
        ],
    ]

    class Connection:
        endpoint = Endpoint("primary", "127.0.0.1", 7496, 155, "U123")

        def __init__(self):
            self.shares = []

        def is_connected(self):
            return True

        def attributed_fills(self, _reference, *, expected_con_id):
            assert expected_con_id == 1
            return fills_sequence.pop(0)

        def stock(self, symbol):
            return SimpleNamespace(symbol=symbol)

        def place_exit_oca(self, *, shares, **_kwargs):
            self.shares.append(shares)
            return (
                SimpleNamespace(order=SimpleNamespace(orderId=10, permId=110)),
                SimpleNamespace(order=SimpleNamespace(orderId=11, permId=111)),
            )

        def confirm_broker_echo(self, *_args, **_kwargs):
            return None

    connection = Connection()
    session = LegendSession(
        plan_path=tmp_path / "plan.json",
        state_store=StateStore(tmp_path / "state.json", tmp_path / "audit.jsonl"),
        account_labels=["primary"],
        live_requested=True,
    )
    session.entry_date = "2026-09-01"
    monkeypatch.setattr(session, "_clear_reference_orders", lambda *_args: True)
    proven_target = SimpleNamespace(order=SimpleNamespace(orderId=10))
    monkeypatch.setattr(
        session,
        "_proven_exit_pair",
        lambda **kwargs: (proven_target, object())
        if kwargs["remaining"] == 8
        else None,
    )
    context = MarketContext(
        root="ES",
        etf="SPY",
        setup_date="2026-08-31",
        contract=SimpleNamespace(symbol="SPY", conId=1),
        initial_ema=500,
        atr14=5,
        ex_dividend=False,
        dividend_detail="none",
    )
    session.contexts = [context]
    target_trade, remaining = session._repair_protection(
        signal_id="sig",
        connection=connection,
        context=context,
        direction=1,
        reference=reference,
        remaining=10,
        target=500.0,
        state={},
    )
    assert target_trade is proven_target
    assert remaining == 8
    assert connection.shares == [10, 8]


def test_file_backed_live_gate_is_hot_and_complete(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    parity_path = tmp_path / "candidate_parity.json"
    _write_parity_evidence(parity_path, root=root)
    reservation_dir = tmp_path / "locks"
    reservation_config = tmp_path / "reservation.json"
    primary_executor = tmp_path / "primary.py"
    pa_executor = tmp_path / "pa.py"
    guard_module = tmp_path / "legend_reservation_guard.py"
    budget_module = tmp_path / "legend_portfolio_budget.py"
    marker = (
        'LEGEND_ACCOUNT_SYMBOL_LOCK_PROTOCOL = '
        '"legend-account-symbol-lock-v1"\n'
    )
    primary_executor.write_text(
        marker + "ib.placeOrder(contract, order)\n", encoding="utf-8"
    )
    pa_executor.write_text(
        marker + "ib.cancelOrder(order)\n", encoding="utf-8"
    )
    guard_module.write_text(marker, encoding="utf-8")
    budget_module.write_text(marker, encoding="utf-8")
    reservation_config.write_text(
        "{\n"
        f'  "protocol": "{PROTOCOL_VERSION}",\n'
        f'  "reservation_dir": "{str(reservation_dir).replace(chr(92), chr(92) * 2)}",\n'
        f'  "legend_runtime_dir": "{str(tmp_path).replace(chr(92), chr(92) * 2)}"\n'
        "}\n",
        encoding="utf-8",
    )
    guard_manifest = tmp_path / "guard.json"
    guard_manifest.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL_VERSION,
                "reservation_dir": str(reservation_dir),
                "legend_runtime_dir": str(tmp_path),
                "executor_root": str(tmp_path),
                "reservation_config": {
                    "path": str(reservation_config),
                    "sha256": file_sha256(reservation_config),
                },
                "executors": [
                    {
                        "label": "primary.py",
                        "path": str(primary_executor),
                        "sha256": file_sha256(primary_executor),
                    },
                    {
                        "label": "pa.py",
                        "path": str(pa_executor),
                        "sha256": file_sha256(pa_executor),
                    },
                    {
                        "label": "legend_reservation_guard.py",
                        "path": str(guard_module),
                        "sha256": file_sha256(guard_module),
                    },
                    {
                        "label": "legend_portfolio_budget.py",
                        "path": str(budget_module),
                        "sha256": file_sha256(budget_module),
                    },
                ],
                "candidate_parity": {
                    "path": str(parity_path),
                    "sha256": file_sha256(parity_path),
                    "source_tree_sha256": candidate_pipeline_attestation(root)[
                        "source_tree_sha256"
                    ],
                },
                "legend_build": {
                    "strategy_version": "legend-etf-spy-qqq-v2",
                    "files": [
                        {
                            "label": label,
                            "path": str(root / Path(label)),
                            "sha256": file_sha256(root / Path(label)),
                        }
                        for label in sorted(REQUIRED_LEGEND_RUNTIME_FILES)
                    ],
                },
                "python_runtime": {
                    "executable": str(Path(sys.executable).resolve()),
                    "python_version": platform.python_version(),
                    "packages": {
                        distribution: importlib.metadata.version(distribution)
                        for distribution in sorted(
                            CRITICAL_RUNTIME_DISTRIBUTIONS
                        )
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime.env"
    values = {
        "LEGEND_ETF_LIVE_ENABLED": "1",
        "LEGEND_ETF_LIVE_DATE": "2026-09-01",
        "LEGEND_ETF_STRATEGY_VERSION": "legend-etf-spy-qqq-v2",
        "LEGEND_ETF_LIVE_ACCOUNTS": "DU123",
        "LEGEND_ETF_ALLOW_LONGS": "1",
        "LEGEND_ETF_ALLOW_SHORTS": "0",
        "LEGEND_ETF_PRIMARY_ACCOUNT": "U123",
        "LEGEND_ETF_FEED_HOST": "127.0.0.1",
        "LEGEND_ETF_FEED_PORT": "7496",
        "LEGEND_ETF_FEED_CLIENT_ID": "154",
        "LEGEND_ETF_RESERVATION_DIR": str(reservation_dir),
        "LEGEND_ETF_GUARD_MANIFEST": str(guard_manifest),
        "LEGEND_ETF_GUARD_MANIFEST_SHA256": file_sha256(guard_manifest),
        "LEGEND_ETF_EXECUTOR_ROOT": str(tmp_path),
        "LEGEND_ETF_PORTFOLIO_BUDGET": str(tmp_path / "budget.json"),
        "LEGEND_ETF_PAPER_PROOF": str(tmp_path / "paper_proof.json"),
        "LEGEND_ETF_PAPER_PROOF_SHA256": "0" * 64,
        "LEGEND_ETF_PA_ACCOUNT": "DU123",
        "LEGEND_ETF_PA_HOST": "127.0.0.1",
        "LEGEND_ETF_PA_PORT": "4001",
        "LEGEND_ETF_PA_CLIENT_ID": "156",
    }
    runtime.write_text(
        "\n".join(f"{key}={value}" for key, value in values.items()) + "\n",
        encoding="utf-8",
    )
    session = LegendSession(
        plan_path=tmp_path / "plan.json",
        state_store=StateStore(tmp_path / "state.json", tmp_path / "audit.jsonl"),
        account_labels=["pa"],
        live_requested=True,
        runtime_env_path=runtime,
    )
    session.entry_date = "2026-09-01"
    monkeypatch.setattr(session, "_refresh_portfolio_budget", lambda *_args: None)
    monkeypatch.setattr(session, "_validate_deployment_guard", lambda *_args: None)
    session.runtime_environment = session._read_live_runtime()
    session.endpoints = endpoints_from_env(
        ["pa"], environment=session.runtime_environment
    )
    session._refresh_live_gate()
    assert session.gate is not None and session.gate.live
    values["LEGEND_ETF_LIVE_ENABLED"] = "0"
    runtime.write_text(
        "\n".join(f"{key}={value}" for key, value in values.items()) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Live gate refused"):
        session._refresh_live_gate()


def test_unresolved_prior_day_or_omitted_account_blocks_startup(tmp_path):
    store = StateStore(tmp_path / "state.json", tmp_path / "audit.jsonl")
    store.put(
        "sig",
        {
            "state": "open",
            "submit_intent_at": "2026-08-31T13:31:00Z",
            "entry_date": "2026-08-31",
            "account": "DU123",
            "root": "ES",
        },
    )
    session = LegendSession(
        plan_path=tmp_path / "plan.json",
        state_store=store,
        account_labels=["primary"],
        live_requested=False,
    )
    session.entry_date = "2026-09-01"
    session.endpoints = [Endpoint("primary", "127.0.0.1", 7496, 155, "U123")]
    with pytest.raises(RuntimeError, match="outside this run's safe recovery"):
        session._assert_recovery_scope([])


def test_transmit_authorized_pending_revision_is_resolved_from_broker_after_deadline(
    tmp_path,
):
    reference = "SPY|BUY|Legend EMA ETF|2026-09-01"
    store = StateStore(tmp_path / "state.json", tmp_path / "audit.jsonl")
    store.put(
        "sig",
        {
            "state": "exit_modifying",
            "submit_intent_at": "2026-09-01T13:31:00Z",
            "account_label": "primary",
            "account": "U123",
            "root": "ES",
            "etf": "SPY",
            "entry_date": "2026-09-01",
            "direction": 1,
            "order_ref": reference,
            "initial_ema": 500.0,
            "initial_target": 500.0,
            "pending_target_revision": {
                "activation": "09:46",
                "source_bar": "09:30",
                "ema": 499.9,
                "limit": 499.89,
                "prior_limit": 500.0,
                "remaining": 10,
                "intent_at": "2026-09-01T13:45:59Z",
                "phase": "transmitting",
                "transmit_authorized_at": "2026-09-01T13:46:00Z",
            },
        },
    )

    def exit_trade(order_id, order_type, *, good_after=""):
        return SimpleNamespace(
            contract=SimpleNamespace(symbol="SPY", conId=1),
            order=SimpleNamespace(
                orderId=order_id,
                permId=order_id + 100,
                clientId=155,
                orderType=order_type,
                action="SELL",
                account="U123",
                orderRef=reference,
                parentId=1,
                ocaGroup="oca",
                ocaType=2,
                tif="DAY",
                outsideRth=False,
                totalQuantity=10,
                transmit=order_type == "MKT",
                goodAfterTime=good_after,
                lmtPrice=499.89 if order_type == "LMT" else 0.0,
            ),
            orderStatus=SimpleNamespace(status="Submitted", remaining=10.0),
        )

    target = exit_trade(2, "LMT")
    timed = exit_trade(
        3, "MKT", good_after="20260901 10:30:00 US/Eastern"
    )

    class Connection:
        endpoint = Endpoint("primary", "127.0.0.1", 7496, 155, "U123")

        def attributed_fills(self, _reference, *, expected_con_id):
            assert expected_con_id == 1
            return [FillEvent("entry", 1, "BOT", 10, 499.0, "t")]

        def attributed_orders(self, _reference, *, expected_con_id):
            assert expected_con_id == 1
            return [target, timed]

    session = LegendSession(
        plan_path=tmp_path / "plan.json",
        state_store=store,
        account_labels=["primary"],
        live_requested=True,
    )
    session.entry_date = "2026-09-01"
    session.accounts = {"primary": Connection()}
    session.contexts = [
        MarketContext(
            root="ES",
            etf="SPY",
            setup_date="2026-08-31",
            contract=SimpleNamespace(symbol="SPY", conId=1),
            initial_ema=500.0,
            atr14=5.0,
            ex_dividend=False,
            dividend_detail="test",
            initial_target=500.0,
        )
    ]
    managed = session._reconcile_prior_records()
    assert len(managed) == 1
    record = store.get("sig")
    assert record["state"] == "open_reconciled"
    assert "pending_target_revision" not in record
    assert "abandoned_pending_target_revision" not in record
    assert record["target_revisions"][-1]["limit"] == pytest.approx(499.89)
    assert record["target_revisions"][-1]["recovered_after_restart"] is True


def test_unproven_prior_day_is_manualized_and_quarantined_without_abort(
    tmp_path,
):
    store = StateStore(tmp_path / "state.json", tmp_path / "audit.jsonl")
    reservation_dir = tmp_path / "reservations"
    store.put(
        "old",
        {
            "state": "open",
            "submit_intent_at": "2026-08-31T13:31:00Z",
            "entry_date": "2026-08-31",
            "account": "U123",
            "etf": "SPY",
            "root": "ES",
            "reservation_dir": str(reservation_dir),
        },
    )
    session = LegendSession(
        plan_path=tmp_path / "plan.json",
        state_store=store,
        account_labels=["primary"],
        live_requested=True,
    )
    session.entry_date = "2026-09-01"
    recovery_found, managed = session._bootstrap_recovery()
    assert recovery_found is True
    assert managed == []
    record = store.get("old")
    assert record["state"] == "critical_prior_day_recovery"
    assert record["manual_review_required"] is True
    assert record["quarantine_active"] is True
    assert quarantine_path(reservation_dir, "U123", "SPY").is_file()
