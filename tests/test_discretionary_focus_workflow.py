from pathlib import Path

from scripts import pull_discretionary_focus_inputs as puller
from scripts.pull_discretionary_focus_inputs import OPTIONAL, REQUIRED


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "discretionary_focus.yml"


def test_focus_inputs_fail_closed_on_market_identity_and_earnings() -> None:
    assert set(REQUIRED) == {
        "master_prices.parquet",
        "overflow_prices.parquet",
        "earnings_calendar.parquet",
        "symbol_master.parquet",
    }
    assert "fundamental/current/daily_report_latest.json" in OPTIONAL
    assert "earnings_calendar_overflow.parquet" in OPTIONAL
    assert "discretionary_focus/email_receipt.json" in OPTIONAL


def test_workflow_is_research_only_and_at_most_once() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "send_discretionary_focus_email.py" in workflow
    assert "--receipt data/discretionary_focus/email_receipt.json" in workflow
    assert "--persist-receipt-r2" in workflow
    assert "publish_discretionary_focus.py" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "contents: read" in workflow

    forbidden = (
        "daily_scan.py",
        "strategy_config.py",
        "Order_Staging",
        "gspread",
        "GCP_JSON",
        "IBKR",
        "broker",
    )
    # Comments are part of the operational contract, so compare executable
    # lines only when checking imports/commands.
    executable = "\n".join(
        line for line in workflow.splitlines() if not line.lstrip().startswith("#")
    )
    for token in forbidden:
        assert token not in executable


def test_workflow_publishes_before_sending_email() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    publish = workflow.index("Publish immutable history")
    email = workflow.index("Send the at-most-once morning email")
    assert publish < email


def test_scheduled_workflow_is_gated_to_actual_nyse_sessions() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    gate = workflow.index("Gate scheduled runs to NYSE sessions")
    pull = workflow.index("Pull current research inputs")
    assert gate < pull
    assert "id: session" in workflow
    assert "check_discretionary_focus_session.py" in workflow
    assert "35 12 * * 1-5" in workflow
    assert "35 13 * * 1-5" in workflow
    assert '--scheduled-cron "${{ github.event.schedule }}"' in workflow

    condition = (
        "if: github.event_name == 'workflow_dispatch' || "
        "steps.session.outputs.should_run == 'true'"
    )
    # Pull, refresh, build, publish, and email all stop on exchange holidays;
    # an explicit workflow dispatch remains available for diagnostics.
    assert workflow.count(condition) == 4
    assert "default: dry_run" in workflow
    assert "publish_and_email" in workflow
    assert workflow.count("inputs.delivery_mode == 'publish_and_email'") == 3
    assert "timeout-minutes: 25" in workflow
    assert "Refresh isolated overflow prices" in workflow
    assert "build_overflow_prices.py" in workflow
    assert "--exclude-today" in workflow
    assert "--no-upload" in workflow
    assert "Refresh isolated overflow earnings coverage" in workflow
    assert "build_earnings_calendar.py" in workflow
    assert "--overflow-staging" in workflow
    assert "yfinance" in workflow
    assert "--delivery-window" in workflow
    assert "steps.delivery.outputs.should_run == 'true'" in workflow
    assert workflow.count("github.ref == 'refs/heads/main'") == 3


def test_receipt_download_fails_closed_except_confirmed_absence(
    tmp_path, monkeypatch
) -> None:
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(puller, "REQUIRED", {})
    monkeypatch.setattr(puller, "OPTIONAL", {puller.RECEIPT_KEY: receipt})
    monkeypatch.setattr(puller, "download_to_local", lambda *args: False)

    monkeypatch.setattr(
        puller, "last_download_error", lambda: "EndpointConnectionError: timed out"
    )
    assert puller.pull_inputs() == [
        f"{puller.RECEIPT_KEY}: EndpointConnectionError: timed out"
    ]

    monkeypatch.setattr(
        puller, "last_download_error", lambda: "ClientError: 404 Not Found"
    )
    assert puller.pull_inputs() == []

    receipt.write_text("stale local receipt", encoding="utf-8")
    assert puller.pull_inputs() == [
        f"{puller.RECEIPT_KEY}: ClientError: 404 Not Found"
    ]
