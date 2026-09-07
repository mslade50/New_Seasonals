"""Deterministic worthwhile-only gate for validated strategy research.

The creative research agent writes prose and reproducible analysis artifacts.
This module decides whether any candidate is allowed into the email from the
validated discovery report and its algorithm-family comparison.  It never
reads holdings, broker state, or current positions.
"""
from __future__ import annotations

import html
import math
from typing import Any

from .contracts import ContractError, parse_timestamp, sha256_json, validate_report
from .family_fit import validate_family_fit_assessment

PACKAGE_SCHEMA = "strategy-research-email-package.v1"
DECISION_SCHEMA = "strategy-research-email-decision.v1"
REQUIRED_METRICS = {
    "sample_size",
    "gross_mean_bps",
    "net_mean_bps",
    "median_net_bps",
    "hit_rate_pct",
    "worst_trade_bps",
    "recent_net_mean_bps",
    "neighbor_positive_count",
    "neighbor_test_count",
    "bootstrap_probability_mean_le_zero",
    "active_book_daily_correlation",
    "bad_day_co_loss_pct",
    "incremental_portfolio_sharpe",
    "marginal_capital_occupancy_pct",
    "estimated_strategy_capacity_usd",
    "round_trip_cost_bps",
    "liquid_ibkr_instrument_count",
    "total_instrument_count",
    "point_in_time_universe_flag",
}
WRITEUP_FIELDS = {
    "strategy",
    "validation",
    "why_it_fits",
    "implementation",
    "risks_and_falsifiers",
}


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{path} must be nonempty text")
    return value.strip()


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContractError(f"{path} must be a finite number")
    return float(value)


def _metrics(candidate: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in candidate["internally_validated_metrics"]:
        name = str(row["name"])
        if name in result:
            raise ContractError(f"candidate {candidate['fingerprint']} has duplicate metric {name}")
        if name in REQUIRED_METRICS:
            result[name] = _number(row["value"], f"metric {name}")
    missing = REQUIRED_METRICS - result.keys()
    if missing:
        raise ContractError(
            f"candidate {candidate['fingerprint']} lacks required portfolio validation metrics: {sorted(missing)}"
        )
    for name in (
        "sample_size",
        "neighbor_positive_count",
        "neighbor_test_count",
        "liquid_ibkr_instrument_count",
        "total_instrument_count",
        "point_in_time_universe_flag",
    ):
        if not result[name].is_integer():
            raise ContractError(f"metric {name} must be integer-valued")
    for name in (
        "sample_size",
        "neighbor_positive_count",
        "neighbor_test_count",
        "estimated_strategy_capacity_usd",
        "round_trip_cost_bps",
        "liquid_ibkr_instrument_count",
        "total_instrument_count",
    ):
        if result[name] < 0:
            raise ContractError(f"metric {name} must be nonnegative")
    for name in ("hit_rate_pct", "bad_day_co_loss_pct", "marginal_capital_occupancy_pct"):
        if not 0 <= result[name] <= 100:
            raise ContractError(f"metric {name} must be in 0..100")
    if not 0 <= result["bootstrap_probability_mean_le_zero"] <= 1:
        raise ContractError("bootstrap probability metric must be in 0..1")
    if result["neighbor_positive_count"] > result["neighbor_test_count"]:
        raise ContractError("positive neighboring specifications exceed tests performed")
    if result["liquid_ibkr_instrument_count"] > result["total_instrument_count"]:
        raise ContractError("liquid IBKR instrument count exceeds total instruments")
    if result["point_in_time_universe_flag"] not in {0, 1}:
        raise ContractError("point-in-time universe flag must be zero or one")
    return result


def _validate_package(raw: Any, report: dict[str, Any], family_fit: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ContractError("research email package must be an object")
    expected = {
        "schema_version",
        "as_of",
        "discovery_run_id",
        "source_bundle_digest",
        "discovery_report_digest",
        "family_fit_digest",
        "positions_used",
        "candidates",
    }
    if set(raw) != expected or raw.get("schema_version") != PACKAGE_SCHEMA:
        raise ContractError("research email package fields/schema are invalid")
    package_as_of = parse_timestamp(raw["as_of"], "email_package.as_of")
    if package_as_of < parse_timestamp(report["as_of"], "report.as_of"):
        raise ContractError("research email package predates the discovery report")
    if raw["discovery_run_id"] != report["run_id"]:
        raise ContractError("research email package references a different discovery run")
    if (
        not isinstance(raw["source_bundle_digest"], str)
        or len(raw["source_bundle_digest"]) != 64
        or any(character not in "0123456789abcdef" for character in raw["source_bundle_digest"])
    ):
        raise ContractError("research email package source bundle digest is invalid")
    if raw["discovery_report_digest"] != sha256_json(report):
        raise ContractError("research email package report digest mismatch")
    if raw["family_fit_digest"] != sha256_json(family_fit):
        raise ContractError("research email package family-fit digest mismatch")
    if raw["positions_used"] is not False or family_fit.get("positions_used") is not False:
        raise ContractError("portfolio-fit research must not use current positions")
    rows = raw["candidates"]
    if not isinstance(rows, list):
        raise ContractError("research email candidates must be a list")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"fingerprint", "writeup"}:
            raise ContractError(f"email candidate {index} fields are invalid")
        fingerprint = _text(row["fingerprint"], f"email candidate {index}.fingerprint")
        if fingerprint in seen:
            raise ContractError("research email package contains duplicate candidates")
        seen.add(fingerprint)
        writeup = row["writeup"]
        if not isinstance(writeup, dict) or set(writeup) != WRITEUP_FIELDS:
            raise ContractError(f"email candidate {index} writeup fields are invalid")
        for key in WRITEUP_FIELDS:
            _text(writeup[key], f"email candidate {index}.writeup.{key}")
    required = {
        row["fingerprint"]
        for row in report["candidates"]
        if row["disposition"] == "NEW_RESEARCH_CANDIDATE"
        and row["actionability"] == "RESEARCH_ACTIONABLE"
    }
    if seen != required:
        raise ContractError(
            "research email package must cover every actionable discovery candidate "
            f"(missing={sorted(required - seen)}, extra={sorted(seen - required)})"
        )
    return raw


def _gate_reasons(metrics: dict[str, float], fit: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    checks = (
        (metrics["sample_size"] >= 40, "fewer than 40 costed observations"),
        (metrics["net_mean_bps"] > 0, "net average return is not positive"),
        (metrics["median_net_bps"] > 0, "net median return is not positive"),
        (
            metrics["gross_mean_bps"] >= 5 * metrics["round_trip_cost_bps"],
            "gross edge is below five times modeled round-trip costs",
        ),
        (metrics["recent_net_mean_bps"] > 0, "recent-era net edge is not positive"),
        (
            metrics["neighbor_test_count"] >= 3
            and metrics["neighbor_positive_count"] >= 2,
            "fewer than two of at least three neighboring specifications are positive",
        ),
        (
            0 <= metrics["bootstrap_probability_mean_le_zero"] <= 0.10,
            "bootstrap probability of a nonpositive mean exceeds 10%",
        ),
        (
            abs(metrics["active_book_daily_correlation"]) <= 0.60,
            "daily correlation to the active algorithm book exceeds 0.60",
        ),
        (
            0 <= metrics["bad_day_co_loss_pct"] <= 35,
            "bad-day co-loss with the active algorithm book exceeds 35%",
        ),
        (
            metrics["incremental_portfolio_sharpe"] >= 0.05,
            "incremental portfolio Sharpe improvement is below 0.05",
        ),
        (
            0 <= metrics["marginal_capital_occupancy_pct"] <= 35,
            "marginal capital occupancy exceeds 35%",
        ),
        (
            metrics["estimated_strategy_capacity_usd"] >= 750_000,
            "estimated capacity is below the current strategy-account scale",
        ),
        (
            metrics["total_instrument_count"] >= 1
            and metrics["liquid_ibkr_instrument_count"] == metrics["total_instrument_count"],
            "not every proposed instrument was verified liquid and IBKR-tradeable",
        ),
        (
            metrics["point_in_time_universe_flag"] == 1,
            "universe history is neither point-in-time nor an explicitly fixed instrument set",
        ),
    )
    failures.extend(reason for passed, reason in checks if not passed)
    if fit["fit_status"] == "ACTIVE_FAMILY_OVERLAP":
        if abs(metrics["active_book_daily_correlation"]) > 0.35:
            failures.append("active-family overlap requires correlation of 0.35 or lower")
        if metrics["incremental_portfolio_sharpe"] < 0.10:
            failures.append("active-family overlap requires at least 0.10 incremental Sharpe")
    elif fit["fit_status"] != "NO_ACTIVE_FAMILY_MATCH":
        failures.append(f"algorithm-family baseline is unresolved: {fit['fit_status']}")
    return failures


def evaluate_email_package(
    report: dict[str, Any], family_fit: dict[str, Any], package_raw: Any
) -> dict[str, Any]:
    validate_report(report)
    validate_family_fit_assessment(family_fit, report)
    package = _validate_package(package_raw, report, family_fit)
    if report["completeness"] != "COMPLETE":
        raise ContractError("email decision requires complete configured source coverage")
    if report["run_mode"] != "LIVE":
        raise ContractError("email decision requires LIVE discovery mode")
    report_by_fp = {row["fingerprint"]: row for row in report["candidates"]}
    fit_by_fp = {row["fingerprint"]: row for row in family_fit.get("candidates", [])}
    decisions = []
    for supplied in package["candidates"]:
        fingerprint = supplied["fingerprint"]
        candidate = report_by_fp.get(fingerprint)
        fit = fit_by_fp.get(fingerprint)
        if candidate is None or fit is None:
            raise ContractError(f"email candidate is absent from report/family fit: {fingerprint}")
        if candidate["lifecycle"] != "VALIDATED_RESEARCH":
            raise ContractError(
                f"candidate {fingerprint} lacks a journaled reproducible validation artifact"
            )
        if fit["fit_status"] not in {"ACTIVE_FAMILY_OVERLAP", "NO_ACTIVE_FAMILY_MATCH"}:
            raise ContractError(
                f"candidate {fingerprint} has an incomplete algorithm-family baseline: "
                f"{fit['fit_status']}"
            )
        metrics = _metrics(candidate)
        failures = _gate_reasons(metrics, fit)
        source_links = sorted({row["permalink"] for row in candidate["provenance"]})
        decisions.append(
            {
                "fingerprint": fingerprint,
                "name": candidate["name"],
                "email_eligible": not failures,
                "failed_gates": failures,
                "fit_status": fit["fit_status"],
                "active_family_matches": [row["name"] for row in fit["active_matches"]],
                "metrics": metrics,
                "source_links": source_links,
                "writeup": supplied["writeup"],
            }
        )
    eligible = [row for row in decisions if row["email_eligible"]]
    return {
        "schema_version": DECISION_SCHEMA,
        "as_of": package["as_of"],
        "discovery_run_id": report["run_id"],
        "source_bundle_digest": package["source_bundle_digest"],
        "discovery_report_digest": sha256_json(report),
        "family_fit_digest": sha256_json(family_fit),
        "positions_used": False,
        "email_required": bool(eligible),
        "eligible_count": len(eligible),
        "candidate_count": len(decisions),
        "candidates": decisions,
    }


def render_email(decision: dict[str, Any]) -> tuple[str, str]:
    eligible = [row for row in decision["candidates"] if row["email_eligible"]]
    if not eligible or not decision["email_required"]:
        raise ContractError("no worthwhile research exists; no email should be rendered")
    date = decision["as_of"][:10]
    subject = f"Strategy research worth reviewing — {date}"
    cards = []
    for row in eligible:
        writeup = row["writeup"]
        metrics = row["metrics"]
        sections = "".join(
            f"<h3>{html.escape(label)}</h3><p>{html.escape(writeup[key])}</p>"
            for key, label in (
                ("strategy", "Strategy"),
                ("validation", "What our data showed"),
                ("why_it_fits", "Why it fits the algorithm book"),
                ("implementation", "How to implement"),
                ("risks_and_falsifiers", "Risks and falsifiers"),
            )
        )
        metric_line = (
            f"N={metrics['sample_size']:.0f}; net mean={metrics['net_mean_bps']:.1f} bps; "
            f"recent={metrics['recent_net_mean_bps']:.1f} bps; book corr="
            f"{metrics['active_book_daily_correlation']:.2f}; incremental Sharpe="
            f"{metrics['incremental_portfolio_sharpe']:.2f}; capacity=$"
            f"{metrics['estimated_strategy_capacity_usd']:,.0f}."
        )
        sources = "".join(
            f'<li><a href="{html.escape(url, quote=True)}">{html.escape(url)}</a></li>'
            for url in row["source_links"]
        )
        cards.append(
            '<div style="border:1px solid #ccd3dd;border-radius:8px;padding:16px;margin:16px 0">'
            f"<h2>{html.escape(row['name'])}</h2><p><b>Gate evidence:</b> {html.escape(metric_line)}</p>"
            f"{sections}<h3>Sources</h3><ul>{sources}</ul></div>"
        )
    body = (
        '<div style="font-family:Segoe UI,Arial,sans-serif;max-width:800px">'
        "<p>These ideas passed reproducibility, cost, robustness, capacity, liquidity, and "
        "active-algorithm portfolio-fit gates. Current positions were not used.</p>"
        + "".join(cards)
        + "</div>"
    )
    return subject, body
