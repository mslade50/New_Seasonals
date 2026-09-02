from __future__ import annotations

import json
from pathlib import Path

import pytest

from episodic_pivot.config import DEFAULT_POLICY
from episodic_pivot.manifest import write_run_artifacts
from episodic_pivot.pipeline import run_shadow_pipeline
from episodic_pivot.premarket import nominate_candidates
from episodic_pivot.tradingview import (
    TradingViewImportError,
    import_tradingview_csv,
    target_session_date,
)
from scripts.capture_ep_premarket_ibkr import main as capture_main
from scripts.import_tradingview_ep import main as import_main
from scripts.run_episodic_pivot_shadow import main as shadow_main


PREMARKET_AT = "2026-08-25T08:30:00-04:00"
PREMARKET_HEADER = (
    "Symbol,Name,Exchange,Pre-market Price,Pre-market Change,"
    "Pre-market Change %,Pre-market Volume\n"
)


def _csv(tmp_path: Path, body: str, *, name: str = "tradingview.csv") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8-sig")
    return path


def _import(path: Path, **overrides):
    values = {
        "session": "premarket",
        "captured_at": PREMARKET_AT,
        "saved_screen_id": "yftOvM3e",
        "reported_result_count": 1,
    }
    values.update(overrides)
    return import_tradingview_csv(path, **values)


def test_official_premarket_headers_normalize_with_safe_defaults(tmp_path):
    path = _csv(
        tmp_path,
        PREMARKET_HEADER
        + "NYSE:ABC,<script>alert(1)</script>,NYSE,$10.90,$0.90,9.00%,100K\n",
    )
    result = _import(path)
    snapshot = result.snapshots[0]

    assert result.target_session_date == "2026-08-25"
    assert result.result_count_verified is True
    assert len(result.source_file_sha256) == 64
    assert snapshot.symbol == "ABC"
    assert snapshot.previous_close == pytest.approx(10.0)
    assert snapshot.premarket_volume == 100_000
    assert snapshot.market_data_status == "BROWSER_EXPORT"
    assert snapshot.tradeable is False
    assert snapshot.halt_status == "UNKNOWN"
    assert snapshot.bid == snapshot.ask == snapshot.premarket_vwap == 0
    assert snapshot.premarket_metrics_at is None
    assert snapshot.contract_con_id is None
    assert snapshot.contract_identity_status == "UNRESOLVED"
    assert snapshot.security_type == "UNKNOWN"


def test_postmarket_friday_and_good_friday_weekend_map_to_next_session():
    assert (
        target_session_date(
            "2026-08-28T17:15:00-04:00", session="after_hours"
        ).isoformat()
        == "2026-08-31"
    )
    assert (
        target_session_date(
            "2026-04-02T17:15:00-04:00", session="after_hours"
        ).isoformat()
        == "2026-04-06"
    )


def test_columbus_day_premarket_is_a_trading_session():
    assert (
        target_session_date(
            "2026-10-12T08:15:00-04:00", session="premarket"
        ).isoformat()
        == "2026-10-12"
    )


def test_after_hours_and_premarket_share_episode_identity_and_merge_latest(tmp_path):
    after_path = _csv(
        tmp_path,
        "Symbol,Name,Exchange,Post-market Price,Post-market Change,"
        "Post-market Change %,Post-market Volume\n"
        "NYSE:ABC,ABC Co,NYSE,10.90,0.90,9.00%,100K\n",
        name="after.csv",
    )
    pre_path = _csv(
        tmp_path,
        PREMARKET_HEADER + "NYSE:ABC,ABC Co,NYSE,11.20,1.20,12.00%,150K\n",
        name="pre.csv",
    )
    after_at = "2026-08-28T17:15:00-04:00"
    pre_at = "2026-08-31T08:15:00-04:00"
    after = import_tradingview_csv(
        after_path,
        session="after_hours",
        captured_at=after_at,
        saved_screen_id="Hqgnyp7Y",
        reported_result_count=1,
    )
    pre = import_tradingview_csv(
        pre_path,
        session="premarket",
        captured_at=pre_at,
        saved_screen_id="yftOvM3e",
        reported_result_count=1,
    )
    after_candidate = nominate_candidates(
        list(after.snapshots), as_of=after_at, policy=DEFAULT_POLICY
    )[0]
    pre_candidate = nominate_candidates(
        list(pre.snapshots), as_of=pre_at, policy=DEFAULT_POLICY
    )[0]
    assert after.target_session_date == pre.target_session_date == "2026-08-31"
    assert after_candidate.candidate_id == pre_candidate.candidate_id

    result = run_shadow_pipeline(
        [*after.snapshots, *pre.snapshots],
        as_of=pre_at,
        target_session_date="2026-08-31",
        policy=DEFAULT_POLICY,
        offline_documents={"ABC": []},
        offline_documents_verified=True,
    )
    assert len(result.candidates) == 1
    assert result.candidates[0].snapshot.last == pytest.approx(11.20)


def test_after_hours_research_for_next_session_is_not_marked_expired(tmp_path):
    path = _csv(
        tmp_path,
        "Symbol,Name,Exchange,Post-market Price,Post-market Change,"
        "Post-market Change %,Post-market Volume\n"
        "NYSE:ABC,ABC Co,NYSE,10.90,0.90,9.00%,100K\n",
    )
    captured = "2026-08-28T17:15:00-04:00"
    imported = import_tradingview_csv(
        path,
        session="after_hours",
        captured_at=captured,
        saved_screen_id="Hqgnyp7Y",
        reported_result_count=1,
    )
    result = run_shadow_pipeline(
        list(imported.snapshots),
        as_of=captured,
        target_session_date=imported.target_session_date,
        policy=DEFAULT_POLICY,
        offline_documents={"ABC": []},
        offline_documents_verified=True,
    )
    assert "ENTRY_WINDOW_EXPIRED" not in result.decisions[0].blockers


def test_reported_percent_or_dollar_boundary_drives_nomination(tmp_path):
    path = _csv(
        tmp_path,
        PREMARKET_HEADER
        + "NYSE:PCT,Percent Co,NYSE,10.20,0.20,2.00%,100K\n"
        + "NYSE:DLR,Dollar Co,NYSE,100.90,0.90,0.90%,100K\n",
    )
    result = _import(path, reported_result_count=2)
    candidates = nominate_candidates(
        list(result.snapshots), as_of=PREMARKET_AT, policy=DEFAULT_POLICY
    )
    assert {item.snapshot.symbol for item in candidates} == {"PCT", "DLR"}
    reasons = {item.snapshot.symbol: set(item.discovery_reasons) for item in candidates}
    assert "SESSION_PERCENT_MOVE_THRESHOLD" in reasons["PCT"]
    assert "SESSION_DOLLAR_MOVE_THRESHOLD" in reasons["DLR"]


def test_inconsistent_price_percent_and_dollar_inputs_fail_closed(tmp_path):
    path = _csv(
        tmp_path,
        PREMARKET_HEADER + "NYSE:BAD,Bad Basis,NYSE,10.90,0.90,2.00%,100K\n",
    )
    with pytest.raises(TradingViewImportError, match="inconsistent"):
        _import(path)


def test_generic_regular_session_columns_cannot_replace_extended_fields(tmp_path):
    path = _csv(
        tmp_path,
        "Symbol,Name,Price,Change,Change %,Volume\nNYSE:BAD,Bad Export,10.90,0.90,9%,100K\n",
    )
    with pytest.raises(TradingViewImportError, match="missing required"):
        _import(path)


def test_count_mismatch_duplicate_malformed_and_empty_export_behavior(tmp_path):
    one = _csv(
        tmp_path,
        PREMARKET_HEADER + "NYSE:ABC,ABC Co,NYSE,10.90,0.90,9%,100K\n",
        name="one.csv",
    )
    with pytest.raises(TradingViewImportError, match="incomplete export"):
        _import(one, reported_result_count=2)

    duplicate = _csv(
        tmp_path,
        PREMARKET_HEADER
        + "NYSE:ABC,ABC Co,NYSE,10.90,0.90,9%,100K\n"
        + "NASDAQ:ABC,ABC ADR,NASDAQ,10.90,0.90,9%,100K\n",
        name="duplicate.csv",
    )
    with pytest.raises(TradingViewImportError, match="duplicate or ambiguous"):
        _import(duplicate, reported_result_count=2)

    malformed = _csv(
        tmp_path,
        PREMARKET_HEADER + "NYSE:ABC,ABC Co,NYSE,wat,0.90,9%,100K\n",
        name="malformed.csv",
    )
    with pytest.raises(TradingViewImportError, match="invalid session price"):
        _import(malformed)

    empty = _csv(tmp_path, PREMARKET_HEADER, name="empty.csv")
    result = _import(empty, reported_result_count=0)
    assert result.snapshots == ()
    assert result.extracted_row_count == 0


@pytest.mark.parametrize(
    "captured_at,session",
    [
        ("2026-08-25T08:30:00", "premarket"),
        ("2026-08-29T08:30:00-04:00", "premarket"),
        ("2026-08-25T10:00:00-04:00", "premarket"),
        ("2026-08-25T15:59:59-04:00", "after_hours"),
        ("2041-08-25T08:30:00-04:00", "premarket"),
    ],
)
def test_invalid_capture_clock_fails_closed(captured_at, session):
    with pytest.raises((TradingViewImportError, ValueError)):
        target_session_date(captured_at, session=session)


def test_tradingview_candidate_can_research_but_never_size(tmp_path):
    path = _csv(
        tmp_path,
        PREMARKET_HEADER + "NYSE:ABC,ABC Co,NYSE,10.90,0.90,9%,100K\n",
    )
    imported = _import(path)
    result = run_shadow_pipeline(
        list(imported.snapshots),
        as_of=PREMARKET_AT,
        target_session_date=imported.target_session_date,
        policy=DEFAULT_POLICY,
        offline_documents={"ABC": []},
        offline_documents_verified=True,
    )
    assert len(result.candidates) == 1
    assert result.previews == []
    blockers = set(result.decisions[0].blockers)
    assert {
        "NON_LIVE_MARKET_DATA",
        "NOT_TRADEABLE",
        "MISSING_EXECUTABLE_QUOTE",
        "UNRESOLVED_IB_CONTRACT",
        "CATALYST_NOT_CONFIRMED",
    } <= blockers


def test_html_report_escapes_tradingview_company_name(tmp_path):
    path = _csv(
        tmp_path,
        PREMARKET_HEADER
        + "NYSE:ABC,<script>alert(1)</script>,NYSE,10.90,0.90,9%,100K\n",
    )
    imported = _import(path)
    result = run_shadow_pipeline(
        list(imported.snapshots),
        as_of=PREMARKET_AT,
        target_session_date=imported.target_session_date,
        policy=DEFAULT_POLICY,
        offline_documents={"ABC": []},
        offline_documents_verified=True,
    )
    output = write_run_artifacts(
        result, policy=DEFAULT_POLICY, output_dir=tmp_path / "run"
    )
    report = (output / "report.html").read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in report
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in report
    assert "broker route NONE" in report


def test_all_operator_clis_are_no_network_no_write_by_default(tmp_path):
    csv_path = _csv(
        tmp_path,
        PREMARKET_HEADER + "NYSE:ABC,ABC Co,NYSE,10.90,0.90,9%,100K\n",
    )
    import_output = tmp_path / "import-output.json"
    assert (
        import_main(
            [
                "--input",
                str(csv_path),
                "--session",
                "premarket",
                "--captured-at",
                PREMARKET_AT,
                "--screen-id",
                "yftOvM3e",
                "--reported-count",
                "1",
                "--output",
                str(import_output),
            ]
        )
        == 0
    )
    assert not import_output.exists()

    imported = _import(csv_path)
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(
        json.dumps(imported.to_dict()), encoding="utf-8"
    )
    report_root = tmp_path / "reports"
    assert (
        shadow_main(
            [
                "--snapshot",
                str(snapshot_path),
                "--as-of",
                PREMARKET_AT,
                "--target-session-date",
                imported.target_session_date,
                "--output-root",
                str(report_root),
            ]
        )
        == 0
    )
    assert not report_root.exists()

    capture_output = tmp_path / "ibkr.json"
    assert (
        capture_main(
            [
                "--symbols-from",
                str(snapshot_path),
                "--output",
                str(capture_output),
            ]
        )
        == 0
    )
    assert not capture_output.exists()


def test_ep_workflow_has_no_live_order_or_sheet_import_path():
    root = Path(__file__).resolve().parents[1]
    paths = [
        *sorted((root / "episodic_pivot").glob("*.py")),
        root / "scripts" / "import_tradingview_ep.py",
        root / "scripts" / "run_episodic_pivot_shadow.py",
        root / "scripts" / "capture_ep_premarket_ibkr.py",
        root / "scripts" / "capture_ep_daily_yfinance.py",
    ]
    source = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)
    for forbidden in (
        "import daily_scan",
        "from daily_scan",
        "import gspread",
        "from gspread",
        ".placeorder(",
        ".reqopenorders(",
        ".reqallopenorders(",
    ):
        assert forbidden not in source
