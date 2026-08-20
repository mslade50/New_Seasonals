"""Build the phase-one fundamental research queue and standalone HTML report."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fundamental.config import (  # noqa: E402
    CURRENT_ROOT,
    FMP_ENDPOINTS,
    POLICY_VERSION,
    REPORT_ROOT,
    RUN_MANIFEST_SCHEMA_VERSION,
    SCHEMA_VERSION,
    UNDERWRITE_SCHEMA_VERSION,
)
from fundamental.company_maps import build_company_maps_report  # noqa: E402
from fundamental.metrics import build_metric_frame, compute_trend_metrics  # noqa: E402
from fundamental.report import render_candidate_report  # noqa: E402
from fundamental.research_controls import (  # noqa: E402
    apply_research_controls,
    load_research_controls,
)
from fundamental.research_process import summarize_research_funnel  # noqa: E402
from fundamental.research_state import (  # noqa: E402
    load_portfolio_snapshot,
    load_research_event_state,
)
from fundamental.storage import (  # noqa: E402
    latest_snapshot_date,
    load_latest_snapshot_parts,
    write_current_parquet,
)
from fundamental.tearsheet import build_tearsheet_pack  # noqa: E402
from fundamental.underwrite import (  # noqa: E402
    build_underwrite_pack,
    is_surfaceable_quick_review,
    load_underwrite_decisions,
)
from fundamental.triage import score_candidates  # noqa: E402
from fundamental.universe import summarize_universe  # noqa: E402


SYMBOL_MASTER = ROOT / "data" / "symbol_master.parquet"
MASTER_PRICES = ROOT / "data" / "master_prices.parquet"
OVERFLOW_PRICES = ROOT / "data" / "overflow_prices.parquet"
BROAD_UNIVERSE = ROOT / "data" / "fundamental" / "current" / "broad_universe_latest.parquet"
UNDERWRITE_DECISIONS = (
    ROOT / "data" / "fundamental" / "current" / "underwrite_decisions_latest.json"
)
SITE_STATE = ROOT / "data" / "fundamental" / "current" / "site_state_latest.json"


def _file_as_of(path: Path) -> str:
    if not path.exists():
        return "missing"
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat()


def _load_research_prices(tickers: set[str]) -> pd.DataFrame:
    columns = ["ticker", "date", "Close", "Volume"]
    frames = []
    for path in (OVERFLOW_PRICES, MASTER_PRICES):
        if path.exists():
            frame = pd.read_parquet(path, columns=columns)
            frame["ticker"] = frame["ticker"].astype(str).str.upper()
            frames.append(frame[frame["ticker"].isin(tickers | {"SPY"})])
    if not frames:
        return pd.DataFrame(columns=columns)
    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    return prices.sort_values("date").drop_duplicates(["ticker", "date"], keep="last")


def _load_current_or_snapshots(
    kind: str, as_of: str, requested: list[str] | None
) -> pd.DataFrame:
    current_path = CURRENT_ROOT / f"{kind}_latest.parquet"
    if current_path.exists():
        frame = pd.read_parquet(current_path)
        if "snapshot_as_of" in frame.columns:
            frame = frame[frame["snapshot_as_of"].astype(str).le(as_of)]
        if requested and "ticker" in frame.columns:
            frame = frame[frame["ticker"].astype(str).str.upper().isin(requested)]
        if not frame.empty:
            return frame.reset_index(drop=True)
    return load_latest_snapshot_parts(kind, as_of, requested)


def _research_eligible_symbols(symbols: pd.DataFrame) -> pd.DataFrame:
    """Keep the report aligned to today's eligible universe.

    Current fundamental views are cumulative by design, so they can retain
    issuers that later leave the research universe.  The daily ranking should
    not silently keep scoring those stale members.
    """
    if "research_eligible" not in symbols.columns:
        return symbols.copy()
    return symbols[symbols["research_eligible"].eq(True)].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build research-only fundamental candidate report.")
    parser.add_argument("--as-of", default=str(date.today()), help="Price/report cutoff date.")
    parser.add_argument("--snapshot-date", default=None,
                        help="Fundamental snapshot date; defaults to latest on/before --as-of.")
    parser.add_argument("--tickers", nargs="+", default=None, help="Optional report ticker subset.")
    parser.add_argument("--output", default=str(REPORT_ROOT / "fundamental_daily.html"))
    parser.add_argument(
        "--underwrite-decisions", default=str(UNDERWRITE_DECISIONS),
        help="Latest completed decision records; absent means no QUICK REVIEW items.",
    )
    parser.add_argument("--upload", action="store_true",
                        help="Upload report/current support files to the existing R2 cache.")
    args = parser.parse_args()

    as_of = str(pd.Timestamp(args.as_of).date())
    snapshot_date = args.snapshot_date or latest_snapshot_date("fmp", as_of)
    if not snapshot_date:
        raise SystemExit("no FMP snapshot exists on or before the report date")
    if not SYMBOL_MASTER.exists() or not MASTER_PRICES.exists():
        raise SystemExit("symbol_master.parquet and master_prices.parquet are required")

    requested = [str(t).upper() for t in args.tickers] if args.tickers else None
    symbols = pd.read_parquet(BROAD_UNIVERSE) if BROAD_UNIVERSE.exists() else pd.read_parquet(SYMBOL_MASTER)
    symbols = _research_eligible_symbols(symbols)
    if requested:
        symbols = symbols[symbols["ticker"].astype(str).str.upper().isin(requested)]
    eligible_tickers = set(symbols["ticker"].astype(str).str.upper())
    fmp_snapshot = _load_current_or_snapshots("fmp", as_of, requested)
    fmp_snapshot = fmp_snapshot[
        fmp_snapshot["ticker"].astype(str).str.upper().isin(eligible_tickers)
    ].reset_index(drop=True)
    sec_snapshot = _load_current_or_snapshots("sec", as_of, requested)
    if not sec_snapshot.empty:
        sec_snapshot = sec_snapshot[
            sec_snapshot["ticker"].astype(str).str.upper().isin(eligible_tickers)
        ].reset_index(drop=True)
    if fmp_snapshot.empty:
        raise SystemExit(f"FMP snapshot is empty for {snapshot_date}")

    tickers = sorted(fmp_snapshot["ticker"].astype(str).str.upper().unique())
    # Read only the fields and tickers needed for this queue; SPY supplies the
    # benchmark-relative 12-1 trend calculation.  The overflow cache materially
    # expands small/mid-cap coverage without changing the strategy price cache.
    prices = _load_research_prices(set(tickers))

    metrics = build_metric_frame(fmp_snapshot, symbols, sec_snapshot)
    trend = compute_trend_metrics(prices, tickers, as_of=as_of)
    candidates = score_candidates(metrics, trend, as_of=as_of)
    research_controls, control_health = load_research_controls(SITE_STATE, as_of=as_of)
    event_state = load_research_event_state(as_of=as_of)
    candidates = apply_research_controls(
        candidates,
        research_controls,
        thesis_changed_tickers=event_state["thesis_changed_tickers"],
        fired_trigger_tickers=event_state["fired_trigger_tickers"],
    )
    research_funnel = summarize_research_funnel(candidates)
    research_funnel["controls"] = {
        "DEEPEN": int(candidates.get("research_control", pd.Series(dtype=object)).eq("DEEPEN").sum()),
        "WATCH": int(candidates.get("research_control", pd.Series(dtype=object)).eq("WATCH").sum()),
        "PASS": int(candidates.get("research_control", pd.Series(dtype=object)).eq("PASS").sum()),
        "CLEAR": int(control_health.get("action_counts", {}).get("CLEAR", 0)),
        "suppressed": int(candidates.get("research_suppressed", pd.Series(dtype=bool)).fillna(False).sum()),
    }
    portfolio_snapshot, portfolio_health = load_portfolio_snapshot(as_of=as_of)
    underwrite_decisions = load_underwrite_decisions(Path(args.underwrite_decisions))
    metrics_path = write_current_parquet(metrics, "metrics_latest.parquet")
    candidates_path = write_current_parquet(candidates, "candidates_latest.parquet")
    company_maps_path, company_maps_support_path = build_company_maps_report(
        as_of=as_of,
        output_path=REPORT_ROOT / "company_maps.html",
        candidates=candidates,
    )

    universe_summary = {}
    if BROAD_UNIVERSE.exists():
        universe = pd.read_parquet(BROAD_UNIVERSE)
        fundamental_set = set(fmp_snapshot["ticker"].astype(str).str.upper())
        sec_set = set(sec_snapshot["ticker"].astype(str).str.upper()) if not sec_snapshot.empty else set()
        universe["fundamental_covered"] = universe["ticker"].astype(str).str.upper().isin(fundamental_set)
        universe["sec_covered"] = universe["ticker"].astype(str).str.upper().isin(sec_set)
        universe_summary = summarize_universe(universe)
        universe_summary["scored_candidates"] = int(len(candidates))
        universe_summary["advance_or_watch"] = int(
            candidates["research_route"].isin({"HYPOTHESIS_TEST", "WATCH_FOR_CHANGE"}).sum()
        )

    endpoint_counts = (
        fmp_snapshot.assign(ticker=fmp_snapshot["ticker"].astype(str).str.upper())
        .groupby("ticker")["endpoint"]
        .agg(lambda values: set(values.dropna().astype(str)))
    )
    baseline_endpoints = set(FMP_ENDPOINTS[:4])
    deep_endpoints = set(FMP_ENDPOINTS)
    baseline_ready = {ticker for ticker, endpoints in endpoint_counts.items() if baseline_endpoints <= endpoints}
    sec_ready = set(sec_snapshot["ticker"].astype(str).str.upper()) if not sec_snapshot.empty else set()
    deep_ready = {ticker for ticker, endpoints in endpoint_counts.items() if deep_endpoints <= endpoints} & sec_ready
    decision_ready = [
        record for record in underwrite_decisions if is_surfaceable_quick_review(record)
    ]
    lane_depth: dict[str, dict[str, int]] = {}
    if "research_lane" in symbols.columns:
        for lane, lane_rows in symbols.groupby("research_lane"):
            lane_tickers = set(lane_rows["ticker"].astype(str).str.upper())
            lane_depth[str(lane)] = {
                "eligible": int(len(lane_tickers)),
                "baseline_ready": int(len(lane_tickers & baseline_ready)),
                "deep_ready": int(len(lane_tickers & deep_ready)),
            }

    def freshness_summary(
        values: pd.Series, max_age_days: int, *, expected_count: int | None = None
    ) -> dict:
        dated = pd.to_datetime(values, errors="coerce").dropna()
        expected = int(expected_count if expected_count is not None else len(values))
        if dated.empty:
            return {"available": 0, "missing": expected, "stale": 0}
        ages = (pd.Timestamp(as_of) - dated.dt.tz_localize(None).dt.normalize()).dt.days
        return {
            "available": int(len(dated)),
            "missing": int(max(expected - len(dated), 0)),
            "oldest_as_of": str(dated.min().date()),
            "median_as_of": str(dated.sort_values().iloc[len(dated) // 2].date()),
            "newest_as_of": str(dated.max().date()),
            "stale": int((ages > max_age_days).sum() + (ages < 0).sum()),
            "max_age_days": int(max_age_days),
        }

    fmp_ticker_dates = (
        fmp_snapshot.groupby(fmp_snapshot["ticker"].astype(str).str.upper())["snapshot_as_of"].max()
        if "snapshot_as_of" in fmp_snapshot.columns else pd.Series(dtype=object)
    )
    sec_ticker_dates = (
        sec_snapshot.groupby(sec_snapshot["ticker"].astype(str).str.upper())["snapshot_as_of"].max()
        if not sec_snapshot.empty and "snapshot_as_of" in sec_snapshot.columns
        else pd.Series(dtype=object)
    )
    freshness = {
        "price": freshness_summary(
            candidates.get("price_as_of", pd.Series(dtype=object)), 7,
            expected_count=len(candidates),
        ),
        "baseline_fundamentals": freshness_summary(
            fmp_ticker_dates, 30, expected_count=len(candidates)
        ),
        "sec_packages": freshness_summary(
            sec_ticker_dates, 550, expected_count=len(candidates)
        ),
        "note": "SEC package presence is not a line-by-line filed-fact reconciliation.",
    }

    health = {
        "versions": {
            "policy": POLICY_VERSION,
            "data_schema": SCHEMA_VERSION,
            "underwrite_schema": UNDERWRITE_SCHEMA_VERSION,
            "run_manifest_schema": RUN_MANIFEST_SCHEMA_VERSION,
        },
        "as_of": as_of,
        "snapshot_date": snapshot_date,
        "universe": universe_summary,
        "research_funnel": research_funnel,
        "research_control_state": control_health,
        "research_event_state": event_state["health"],
        "portfolio_context": portfolio_health,
        "coverage_depth": {
            "eligible": int(len(candidates)),
            "baseline_ready": int(len(baseline_ready)),
            "deep_ready": int(len(deep_ready)),
            "decision_ready": int(len(decision_ready)),
            "by_lane": lane_depth,
            "method": (
                "Deep-ready requires all seven FMP endpoints plus an SEC package. "
                "Decision-ready additionally requires a validated v2 QUICK REVIEW."
            ),
        },
        "freshness": freshness,
        "gaps": [
            "Discovery axes and routes allocate research; they cannot create QUICK REVIEW or approve capital.",
            "Historical universe and prices remain survivorship-biased until delisted-security coverage is added.",
            "Historical analyst-estimate snapshots begin with this project; prior consensus cannot be reconstructed safely.",
            "SEC package presence is not a filed-fact tie-out; a reconciliation ledger remains required before decision readiness.",
            "Trigger, evidence, transition, and manual portfolio ledgers are explicit inputs; missing ledgers fail closed and never imply no events or zero holdings.",
            "Specialist lanes remain baseline-only until dedicated financials capital/credit/book-value, REIT AFFO/NAV/maturities, and biotech pipeline/rNPV/runway scorecards are implemented.",
            "Live sleeve attribution and technical/fundamental ticker-overlap controls are not implemented.",
        ],
        "sources": [
            {
                "source": "FMP immutable endpoint snapshots",
                "as_of": (
                    f"{freshness['baseline_fundamentals'].get('oldest_as_of', 'missing')} to "
                    f"{freshness['baseline_fundamentals'].get('newest_as_of', 'missing')}"
                ),
                "posture": "Provider-standardized; accepted timestamps retained when supplied",
                "use": "Statements, ratios, metrics, profile, and consensus snapshots",
            },
            {
                "source": "SEC EDGAR companyfacts",
                "as_of": (
                    f"{freshness['sec_packages'].get('oldest_as_of', 'missing')} to "
                    f"{freshness['sec_packages'].get('newest_as_of', 'missing')}"
                    if not sec_snapshot.empty else "not loaded"
                ),
                "posture": "Filed packages present; reconciliation not yet complete" if not sec_snapshot.empty else "Missing required package",
                "use": "Accession and filing-acceptance verification; not a metric tie-out",
            },
            {
                "source": "Adjusted master price cache",
                "as_of": str(pd.to_datetime(prices["date"], errors="coerce").max().date()),
                "posture": "Derived market data",
                "use": "200-day timing, optional 200-week context, 12-1 relative trend, and liquidity",
            },
            {
                "source": "Broad FMP stock screener / current symbol master",
                "as_of": str(pd.to_datetime(symbols.get("as_of"), errors="coerce").max().date())
                    if "as_of" in symbols.columns else _file_as_of(SYMBOL_MASTER),
                "posture": "Current-universe metadata; not survivorship-free history",
                "use": "2,000-name discovery funnel, issuer metadata, and universe gates",
            },
        ],
    }
    output_path = Path(args.output)
    tearsheet_links = build_tearsheet_pack(
        candidates,
        fmp_snapshot,
        sec_snapshot,
        output_path.parent / "tearsheets",
        max_names=10,
    )
    underwrite_links = build_underwrite_pack(
        underwrite_decisions,
        candidates,
        output_path.parent / "underwrites",
    )
    tearsheet_links.update(underwrite_links)
    report_path = render_candidate_report(
        candidates,
        health,
        output_path,
        tearsheet_links=tearsheet_links,
        underwrite_decisions=underwrite_decisions,
    )
    json_path = candidates_path.with_name("daily_report_latest.json")
    json_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "underwrite_schema_version": UNDERWRITE_SCHEMA_VERSION,
        "health": health,
        "candidates": candidates.to_dict("records"),
        "underwrite_decisions": underwrite_decisions,
        "live_actions_enabled": False,
        "portfolio_snapshot_loaded": portfolio_snapshot is not None,
    }, indent=2, default=str), encoding="utf-8")

    print(f"Snapshot: {snapshot_date}; candidates: {len(candidates)}")
    print(candidates["research_priority"].value_counts().to_string())
    print(f"Report: {report_path}")
    print(f"Company maps: {company_maps_path}")

    if args.upload:
        from cache_io import upload_from_local
        uploads = [
            (metrics_path, "fundamental/current/metrics_latest.parquet"),
            (candidates_path, "fundamental/current/candidates_latest.parquet"),
            (json_path, "fundamental/current/daily_report_latest.json"),
            (company_maps_support_path, "fundamental/current/company_maps_latest.json"),
            (report_path, "fundamental/reports/fundamental_daily.html"),
            (company_maps_path, "fundamental/reports/company_maps.html"),
        ]
        if Path(args.underwrite_decisions).exists():
            uploads.append(
                (
                    Path(args.underwrite_decisions),
                    "fundamental/current/underwrite_decisions_latest.json",
                )
            )
        uploads.extend(
            (output_path.parent / link, f"fundamental/reports/{link}")
            for link in tearsheet_links.values()
        )
        for local, key in uploads:
            if not upload_from_local(str(local), key):
                raise SystemExit(f"R2 upload failed: {local}")


if __name__ == "__main__":
    main()
