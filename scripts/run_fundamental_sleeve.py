"""Canonical, dry-run-first fundamental-sleeve research orchestrator.

This command coordinates local research state only.  It intentionally has no
upload, deployment, messaging, broker, order, or allocation capability.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fundamental.config import (  # noqa: E402
    BROAD_UNIVERSE_POLICY,
    CURRENT_ROOT,
    FMP_ENDPOINTS,
    REPORT_ROOT,
)
from fundamental.research_controls import load_research_controls  # noqa: E402
from fundamental.research_state import (  # noqa: E402
    EVIDENCE_STATE_PATH,
    PORTFOLIO_SNAPSHOT_PATH,
    RUN_MANIFEST_LATEST_PATH,
    TRIGGER_STATE_PATH,
    load_portfolio_snapshot,
    load_research_event_state,
)
from fundamental.run_manifest import (  # noqa: E402
    append_decision_transitions,
    decision_states,
    file_sha256,
    freeze_sources,
    git_code_state,
    write_sleeve_run_manifest,
)
from fundamental.storage import iso_utc  # noqa: E402
from fundamental.underwrite import is_surfaceable_quick_review, load_underwrite_decisions  # noqa: E402


BROAD_UNIVERSE = CURRENT_ROOT / "broad_universe_latest.parquet"
FMP_CURRENT = CURRENT_ROOT / "fmp_latest.parquet"
SEC_CURRENT = CURRENT_ROOT / "sec_latest.parquet"
CANDIDATES_CURRENT = CURRENT_ROOT / "candidates_latest.parquet"
DAILY_REPORT_CURRENT = CURRENT_ROOT / "daily_report_latest.json"
UNDERWRITE_DECISIONS = CURRENT_ROOT / "underwrite_decisions_latest.json"
SITE_STATE = CURRENT_ROOT / "site_state_latest.json"
MASTER_PRICES = ROOT / "data" / "master_prices.parquet"
OVERFLOW_PRICES = ROOT / "data" / "overflow_prices.parquet"


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _eligible_universe() -> pd.DataFrame:
    universe = _read_parquet(BROAD_UNIVERSE)
    if universe.empty:
        return universe
    if "research_eligible" in universe.columns:
        universe = universe[universe["research_eligible"].eq(True)]
    universe = universe.copy()
    universe["ticker"] = universe["ticker"].astype(str).str.upper()
    return universe.drop_duplicates("ticker").reset_index(drop=True)


def _endpoint_state(fmp: pd.DataFrame) -> tuple[dict[str, set[str]], dict[tuple[str, str], str]]:
    endpoint_sets: dict[str, set[str]] = {}
    endpoint_dates: dict[tuple[str, str], str] = {}
    if fmp.empty or "ticker" not in fmp or "endpoint" not in fmp:
        return endpoint_sets, endpoint_dates
    frame = fmp.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["endpoint"] = frame["endpoint"].astype(str)
    for ticker, group in frame.groupby("ticker"):
        endpoint_sets[ticker] = set(group["endpoint"].dropna())
        if "snapshot_as_of" in group:
            for endpoint, endpoint_group in group.groupby("endpoint"):
                dates = pd.to_datetime(endpoint_group["snapshot_as_of"], errors="coerce")
                if dates.notna().any():
                    endpoint_dates[(ticker, endpoint)] = str(dates.max().date())
    return endpoint_sets, endpoint_dates


def build_run_plan(
    *,
    as_of: str,
    batch_size: int,
    universe_refresh_days: int,
) -> dict[str, Any]:
    universe = _eligible_universe()
    eligible = set(universe.get("ticker", pd.Series(dtype=object)).astype(str).str.upper())
    fmp = _read_parquet(FMP_CURRENT)
    endpoint_sets, endpoint_dates = _endpoint_state(fmp)
    baseline_endpoints = set(FMP_ENDPOINTS[:4])
    deep_endpoints = set(FMP_ENDPOINTS)
    cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=BROAD_UNIVERSE_POLICY.refresh_after_days)

    baseline_ready: set[str] = set()
    for ticker in eligible:
        if not baseline_endpoints <= endpoint_sets.get(ticker, set()):
            continue
        dates = [pd.to_datetime(endpoint_dates.get((ticker, endpoint)), errors="coerce") for endpoint in baseline_endpoints]
        if all(pd.notna(value) and value >= cutoff for value in dates):
            baseline_ready.add(ticker)

    sec = _read_parquet(SEC_CURRENT)
    sec_ready = (
        set(sec["ticker"].astype(str).str.upper())
        if not sec.empty and "ticker" in sec else set()
    )
    deep_ready = {
        ticker for ticker in eligible if deep_endpoints <= endpoint_sets.get(ticker, set())
    } & sec_ready
    baseline_gap = sorted(eligible - baseline_ready)

    lane_depth: dict[str, dict[str, int]] = {}
    if not universe.empty and "research_lane" in universe:
        for lane, rows in universe.groupby("research_lane"):
            tickers = set(rows["ticker"])
            lane_depth[str(lane)] = {
                "eligible": len(tickers),
                "baseline_ready": len(tickers & baseline_ready),
                "deep_ready": len(tickers & deep_ready),
            }

    decisions = load_underwrite_decisions(UNDERWRITE_DECISIONS)
    review_ready = [record for record in decisions if is_surfaceable_quick_review(record, decision_as_of=as_of)]
    legacy = [record for record in decisions if record.get("schema_version") != "fundamental-underwrite.v2"]
    controls, control_health = load_research_controls(SITE_STATE, as_of=as_of)
    events = load_research_event_state(as_of=as_of)
    _, portfolio_health = load_portfolio_snapshot(as_of=as_of)

    universe_date = None
    if not universe.empty and "as_of" in universe:
        values = pd.to_datetime(universe["as_of"], errors="coerce")
        universe_date = str(values.max().date()) if values.notna().any() else None
    universe_age = (
        int((pd.Timestamp(as_of) - pd.Timestamp(universe_date)).days) if universe_date else None
    )
    universe_stale = universe_age is None or universe_age > universe_refresh_days

    price_health = {"available": False, "stale": None}
    candidates = _read_parquet(CANDIDATES_CURRENT)
    if not candidates.empty and "price_as_of" in candidates:
        dates = pd.to_datetime(candidates["price_as_of"], errors="coerce")
        ages = (pd.Timestamp(as_of) - dates.dt.normalize()).dt.days
        price_health = {
            "available": True,
            "covered": int(dates.notna().sum()),
            "missing": int(dates.isna().sum()),
            "stale": int(((ages > 7) | (ages < 0)).fillna(False).sum()),
            "oldest_as_of": str(dates.min().date()) if dates.notna().any() else None,
            "newest_as_of": str(dates.max().date()) if dates.notna().any() else None,
        }

    return {
        "as_of": as_of,
        "mode": "DRY_RUN",
        "research_only": True,
        "publishing_performed": False,
        "live_actions_enabled": False,
        "universe": {
            "eligible": len(eligible),
            "as_of": universe_date,
            "age_days": universe_age,
            "refresh_after_days": universe_refresh_days,
            "stale": universe_stale,
        },
        "coverage": {
            "baseline_ready": len(baseline_ready),
            "baseline_gap": len(baseline_gap),
            "bounded_refresh_count": min(len(baseline_gap), batch_size),
            "bounded_refresh_tickers": baseline_gap[:batch_size],
            "deep_ready": len(deep_ready),
            "decision_ready": len(review_ready),
            "sec_packages": len(sec_ready & eligible),
            "by_lane": lane_depth,
        },
        "price_health": price_health,
        "underwrites": {
            "records": len(decisions),
            "legacy_records": len(legacy),
            "decision_ready": len(review_ready),
            "review_ready_tickers": [str(record.get("ticker")) for record in review_ready[:3]],
        },
        "controls": {"records": len(controls), **control_health},
        "events": events["health"],
        "portfolio": portfolio_health,
        "recommended_actions": [
            action for action in [
                "refresh_universe" if universe_stale else None,
                "refresh_missing_or_stale_baselines" if baseline_gap else None,
                "rebuild_local_research_views",
                "validate_underwrites_and_tests",
                "inspect_changed_html_in_browser",
            ] if action
        ],
    }


def _run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _output_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "available": path.exists(),
        "sha256": file_sha256(path) if path.exists() else None,
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def execute_run(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    started_at = iso_utc()
    if args.refresh_universe or (args.refresh and plan["universe"]["stale"]):
        _run([sys.executable, "scripts/build_fundamental_universe.py", "--as-of", args.as_of])
        plan = build_run_plan(
            as_of=args.as_of,
            batch_size=args.batch_size,
            universe_refresh_days=args.universe_refresh_days,
        )
    if args.refresh and plan["coverage"]["baseline_gap"]:
        _run([
            sys.executable,
            "scripts/update_fundamentals.py",
            "--as-of", args.as_of,
            "--balanced-batch", str(args.batch_size),
            "--include-specialists",
            "--bundle-depth", "screen",
            "--refresh-after-days", str(BROAD_UNIVERSE_POLICY.refresh_after_days),
            "--with-sec",
            "--sec-limit", str(args.sec_limit),
        ])
    if args.refresh_prices:
        _run([sys.executable, "scripts/build_overflow_prices.py", "--no-upload", "--exclude-today"])

    _run([
        sys.executable,
        "scripts/build_fundamental_report.py",
        "--as-of", args.as_of,
        "--output", str(args.output),
    ])
    _run([sys.executable, "scripts/validate_fundamental_underwrites.py", "--as-of", args.as_of])

    verification = {"underwrite_validator": "PASS", "python_tests": "NOT_RUN", "javascript_tests": "NOT_RUN"}
    if args.verify:
        _run([
            sys.executable, "-m", "pytest",
            "tests/test_fundamental_sleeve.py",
            "tests/test_fundamental_site.py",
            "tests/test_fundamental_workflow.py",
            "tests/test_fundamental_v2.py",
            "-q",
        ])
        verification["python_tests"] = "PASS"
        _run(["node", "tests/js/test_fundamentals_research_actions.js"])
        verification["javascript_tests"] = "PASS"

    report_payload = json.loads(DAILY_REPORT_CURRENT.read_text(encoding="utf-8"))
    health = report_payload.get("health", {})
    decisions = report_payload.get("underwrite_decisions", [])
    states = decision_states(decisions)
    legacy_decision_count = sum(
        str(record.get("schema_version") or "legacy") != "fundamental-underwrite.v2"
        for record in decisions if isinstance(record, dict)
    )
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"fundamental-sleeve-{run_stamp}"
    transition_rows = append_decision_transitions(
        previous_manifest_path=RUN_MANIFEST_LATEST_PATH,
        current_states=states,
        run_id=run_id,
        as_of=args.as_of,
    )
    source_paths = {
        "broad_universe": BROAD_UNIVERSE,
        "fmp_current": FMP_CURRENT,
        "sec_current": SEC_CURRENT,
        "master_prices": MASTER_PRICES,
        "overflow_prices": OVERFLOW_PRICES,
        "underwrite_decisions": UNDERWRITE_DECISIONS,
        "site_research_controls": SITE_STATE,
        "trigger_ledger": TRIGGER_STATE_PATH,
        "evidence_ledger": EVIDENCE_STATE_PATH,
        "portfolio_snapshot": PORTFOLIO_SNAPSHOT_PATH,
    }
    manifest = {
        "run_id": run_id,
        "as_of": args.as_of,
        "started_at": started_at,
        "completed_at": iso_utc(),
        "completion_status": "BUILT_AWAITING_VISUAL_QA",
        "code": git_code_state(),
        "source_freeze": freeze_sources(source_paths),
        "coverage": health.get("coverage_depth", {}),
        "research_funnel": health.get("research_funnel", {}),
        "research_controls": health.get("research_control_state", {}),
        "research_events": health.get("research_event_state", {}),
        "portfolio_context": health.get("portfolio_context", {}),
        "decision_states": states,
        "underwrite_contract": {
            "records": len(states),
            "legacy_records": int(legacy_decision_count),
            "authoritative_v2_records": int(len(states) - legacy_decision_count),
            "decision_ready": int(health.get("coverage_depth", {}).get("decision_ready", 0)),
            "status": "INCOMPLETE" if legacy_decision_count else "CURRENT",
        },
        "transition_count": len(transition_rows),
        "verification": verification,
        "qa": {
            "visual_status": "PENDING_EXTERNAL_BROWSER_INSPECTION",
            "require_exact_report_digest": True,
        },
        "outputs": {
            "report": _output_record(Path(args.output)),
            "daily_report_json": _output_record(DAILY_REPORT_CURRENT),
            "candidates": _output_record(CANDIDATES_CURRENT),
        },
        "research_only": True,
        "publishing_performed": False,
        "live_actions_enabled": False,
    }
    immutable_manifest, latest_manifest = write_sleeve_run_manifest(manifest)
    result = {
        "run_id": run_id,
        "completion_status": manifest["completion_status"],
        "manifest": str(immutable_manifest),
        "latest_manifest": str(latest_manifest),
        "report": str(args.output),
        "decision_ready": int(health.get("coverage_depth", {}).get("decision_ready", 0)),
        "visual_qa_required": True,
        "publishing_performed": False,
        "live_actions_enabled": False,
    }
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or execute the research-only fundamental sleeve loop."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Inspect the run plan without writing anything (default).")
    mode.add_argument("--execute", action="store_true", help="Build local research views and a versioned run manifest.")
    parser.add_argument("--as-of", default=str(date.today()))
    parser.add_argument("--batch-size", type=int, default=BROAD_UNIVERSE_POLICY.default_enrichment_batch)
    parser.add_argument("--sec-limit", type=int, default=10)
    parser.add_argument("--universe-refresh-days", type=int, default=7)
    parser.add_argument("--refresh", action="store_true", help="Refresh only missing/stale baseline inputs before building.")
    parser.add_argument("--refresh-universe", action="store_true", help="Explicitly refresh the broad-universe screener snapshot.")
    parser.add_argument("--refresh-prices", action="store_true", help="Increment the isolated research price cache locally.")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=REPORT_ROOT / "fundamental_daily.html")
    args = parser.parse_args()
    args.as_of = str(pd.Timestamp(args.as_of).date())
    if args.batch_size < 1 or args.batch_size > BROAD_UNIVERSE_POLICY.max_enrichment_batch:
        parser.error(f"--batch-size must be 1..{BROAD_UNIVERSE_POLICY.max_enrichment_batch}")
    if args.sec_limit < 0 or args.sec_limit > args.batch_size:
        parser.error("--sec-limit must be between zero and --batch-size")
    if not args.execute and (args.refresh or args.refresh_universe or args.refresh_prices):
        parser.error("refresh flags require --execute; dry-run never fetches or writes")
    return args


def main() -> None:
    args = parse_args()
    plan = build_run_plan(
        as_of=args.as_of,
        batch_size=args.batch_size,
        universe_refresh_days=args.universe_refresh_days,
    )
    if not args.execute:
        print(json.dumps(plan, indent=2, default=str))
        return
    execute_run(args, plan)


if __name__ == "__main__":
    main()
