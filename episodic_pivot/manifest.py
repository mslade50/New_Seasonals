"""Immutable-ish local run artifacts for replay and review."""

from __future__ import annotations

import csv
import hashlib
import html
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from .config import EPPolicy
from .schema import ResearchSizingPreview, RunResult


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _csv_safe(value: Any) -> Any:
    """Prevent review CSV cells from becoming spreadsheet formulas."""

    if not isinstance(value, str):
        return value
    trimmed = value.lstrip(" \t\r\n")
    if trimmed.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _normalized_key(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _validate_research_previews(previews: list[ResearchSizingPreview]) -> None:
    """Reject order-shaped or armed objects before creating any artifact path."""

    forbidden = {
        _normalized_key(name)
        for name in {
            "Action",
            "Quantity",
            "Order_Type",
            "TIF",
            "Limit_Price",
            "Manual_Limit",
            "Strategy_Ref",
            "Trade_Direction",
            "Risk_Amt",
            "Risk_Bps",
            "Approval",
            "Execute_On",
            "Transmit",
        }
    }
    for preview in previews:
        if type(preview) is not ResearchSizingPreview:
            raise TypeError("run previews must be exact ResearchSizingPreview records")
        if (
            preview.record_type != "EP_RESEARCH_SIZING_PREVIEW_V1"
            or not preview.preview_only
            or preview.executable
            or preview.broker_route != "NONE"
            or preview.order_submission_allowed
            or not preview.human_review_required
            or preview.production_eligible
            or preview.live_actions_enabled
        ):
            raise ValueError("research sizing preview safety sentinel failed")
        collision = forbidden & {_normalized_key(key) for key in preview.to_dict()}
        if collision:
            raise ValueError(
                f"research sizing preview collides with live-order keys: {sorted(collision)}"
            )


def _report(result: RunResult, policy: EPPolicy) -> str:
    by_decision: dict[str, int] = {}
    for decision in result.decisions:
        by_decision[decision.decision] = by_decision.get(decision.decision, 0) + 1
    lines = [
        f"# Episodic Pivot shadow run — {result.run_id}",
        "",
        f"Generated: {result.generated_at}",
        f"Policy: `{policy.policy_id}`",
        "Mode: shadow research; no broker, Sheets, R2, schedule, or production write.",
        "",
        "## Counts",
        "",
        f"- Nominations: {len(result.candidates)}",
        f"- Research sizing previews: {len(result.previews)}",
    ]
    for label in sorted(by_decision):
        lines.append(f"- {label}: {by_decision[label]}")
    lines.extend(["", "## Candidate decisions", ""])
    if not result.decisions:
        lines.append("No nominations passed the broad premarket discovery screen.")
    for decision in result.decisions:
        blockers = ", ".join(decision.blockers) or "none"
        warnings = ", ".join(decision.warnings) or "none"
        lines.append(
            f"- **{decision.symbol}** — {decision.decision} / {decision.setup_type}; "
            f"catalyst={decision.catalyst.catalyst_type}; "
            f"materiality={decision.catalyst.materiality_score}/5; "
            f"blockers={blockers}; warnings={warnings}."
        )
    lines.extend(
        [
            "",
            "## Safety invariants",
            "",
            "- Every sizing object is hypothetical, deliberately non-executable, and names no broker route.",
            "- Any later implementation would need fresh quote, halt, gap, and contract validation.",
            "- `Order_Submission_Allowed` and `Production_Eligible` are false.",
            "- A fetched, timestamped source document is required; a search snippet cannot confirm an EP.",
            "- This output is a review artifact, not an instruction to trade.",
            "",
        ]
    )
    return "\n".join(lines)


def _safe_link(url: str) -> str:
    value = str(url or "").strip()
    if not value.lower().startswith("https://"):
        return ""
    return html.escape(value, quote=True)


def _html_report(result: RunResult, policy: EPPolicy) -> str:
    """Render a standalone, escaped research-triage report."""

    decisions = {item.candidate_id: item for item in result.decisions}
    previews = {item.candidate_id: item for item in result.previews}
    priority = {"RESEARCH_PREVIEW_ELIGIBLE": 0, "WATCH": 1, "REJECT": 2}
    ordered = sorted(
        result.candidates,
        key=lambda item: (
            priority.get(decisions.get(item.candidate_id).decision, 9)
            if decisions.get(item.candidate_id)
            else 9,
            -(decisions.get(item.candidate_id).catalyst.materiality_score)
            if decisions.get(item.candidate_id)
            else 0,
            -item.snapshot.premarket_dollar_volume,
            item.snapshot.symbol,
        ),
    )
    counts: dict[str, int] = {}
    for decision in result.decisions:
        counts[decision.decision] = counts.get(decision.decision, 0) + 1

    cards = [
        ("Nominations", len(result.candidates)),
        ("Preview-eligible", counts.get("RESEARCH_PREVIEW_ELIGIBLE", 0)),
        ("Watch", counts.get("WATCH", 0)),
        ("Rejected", counts.get("REJECT", 0)),
    ]
    card_html = "".join(
        f'<div class="metric"><span>{html.escape(label)}</span><strong>{value}</strong></div>'
        for label, value in cards
    )

    candidate_html: list[str] = []
    for index, candidate in enumerate(ordered, start=1):
        snap = candidate.snapshot
        decision = decisions.get(candidate.candidate_id)
        if decision is None:
            continue
        docs = result.documents_by_candidate.get(candidate.candidate_id, [])
        preview = previews.get(candidate.candidate_id)
        blockers = list(decision.blockers)
        warnings = list(decision.warnings)
        first_rejection = blockers[0] if blockers else "None"
        evidence_links = []
        for document in docs:
            url = _safe_link(document.canonical_url or document.url)
            title = html.escape(document.title or document.publisher or "Source")
            published = html.escape(document.published_at or "timestamp unavailable")
            status = html.escape(document.fetch_status)
            source = html.escape(document.source_tier)
            link = f'<a href="{url}" rel="noreferrer">{title}</a>' if url else title
            evidence_links.append(
                f"<li>{link}<small>{source} · {published} · {status}</small></li>"
            )
        if not evidence_links:
            evidence_links.append("<li>No fetched source document.</li>")

        if decision.decision == "RESEARCH_PREVIEW_ELIGIBLE":
            actionability = "Cleared deterministic research gates; human review still required."
            next_step = "Review the causal source and recapture a fresh IBKR snapshot before considering any separate approval design."
        elif decision.decision == "WATCH":
            actionability = "Research only; one or more hard gates remain unresolved."
            next_step = "Resolve the first blocker, verify a primary source, and rerun with a fresh market snapshot."
        else:
            actionability = "Rejected by the current shadow policy."
            next_step = "Archive the episode unless materially new evidence changes the event classification."
        what_changes = (
            f"Resolve: {', '.join(blockers[:4])}" if blockers else "No deterministic blocker; requires human causal-news judgment."
        )
        kill = ", ".join(decision.catalyst.adverse_flags) or (
            "Stale/recycled news, an unresolved corporate action, or failed liquidity revalidation."
        )
        preview_html = ""
        if preview:
            preview_html = (
                '<div class="preview"><strong>Hypothetical sizing only</strong>'
                f'<span>Reference entry ${preview.reference_entry_price:,.2f}</span>'
                f'<span>Hypothetical stop ${preview.hypothetical_stop_price:,.2f}</span>'
                f'<span>Maximum preview shares {preview.max_preview_shares:,}</span>'
                f'<span>Modeled risk ${preview.modeled_risk_dollars:,.0f}</span>'
                '<span>Executable: false · Broker route: NONE</span></div>'
            )
        candidate_html.append(
            f"""
            <article class="candidate">
              <header>
                <div><span class="rank">{index:02d}</span><h2>{html.escape(snap.symbol)}</h2>
                <p>{html.escape(snap.company_name or "Company name unavailable")}</p></div>
                <span class="badge {html.escape(decision.decision.lower())}">{html.escape(decision.decision.replace('_', ' '))}</span>
              </header>
              <div class="tape">
                <span><b>{snap.discovery_gap_pct:+.2f}%</b> session move</span>
                <span><b>${snap.discovery_move_dollars:+.2f}</b> dollar move</span>
                <span><b>{snap.premarket_volume:,}</b> session volume</span>
                <span><b>${snap.premarket_dollar_volume:,.0f}</b> est. dollar volume</span>
                <span><b>{html.escape(snap.session)}</b> capture</span>
              </div>
              <div class="grid">
                <section><h3>Actionability</h3><p>{html.escape(actionability)}</p></section>
                <section><h3>Why now</h3><p>{html.escape(decision.catalyst.summary or 'Price/volume nomination; catalyst not yet verified.')}</p></section>
                <section><h3>First rejection</h3><p>{html.escape(first_rejection)}</p></section>
                <section><h3>What would advance it</h3><p>{html.escape(what_changes)}</p></section>
                <section><h3>Kill criteria</h3><p>{html.escape(kill)}</p></section>
                <section><h3>Next workflow</h3><p>{html.escape(next_step)}</p></section>
              </div>
              {preview_html}
              <details><summary>Evidence ledger ({len(docs)})</summary><ul class="sources">{''.join(evidence_links)}</ul></details>
              <details><summary>Rules and flags</summary>
                <p><b>Setup:</b> {html.escape(decision.setup_type)} · <b>Catalyst:</b> {html.escape(decision.catalyst.catalyst_type)} · <b>Materiality:</b> {decision.catalyst.materiality_score}/5</p>
                <p><b>Blockers:</b> {html.escape(', '.join(blockers) or 'none')}</p>
                <p><b>Warnings:</b> {html.escape(', '.join(warnings) or 'none')}</p>
              </details>
            </article>
            """
        )

    empty = "" if candidate_html else '<div class="empty">No rows passed the broad move/price/volume nomination rule.</div>'
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EP research triage · {html.escape(result.run_id)}</title>
<style>
:root{{--ink:#15202b;--muted:#64748b;--paper:#f4f6f8;--card:#fff;--navy:#102a43;--blue:#1d4ed8;--amber:#b45309;--red:#b91c1c;--line:#dbe3ea}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1160px;margin:auto;padding:34px 22px 70px}} .eyebrow{{text-transform:uppercase;letter-spacing:.16em;font-size:11px;font-weight:800;color:#64748b}}
h1{{font:700 40px/1.05 Georgia,serif;margin:7px 0 10px;color:var(--navy)}} .dek{{max-width:780px;color:var(--muted);margin:0 0 22px}}
.safety{{border-left:5px solid var(--amber);background:#fff7ed;padding:13px 16px;border-radius:8px;margin:20px 0}} .safety b{{color:#92400e}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0 30px}} .metric{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;display:flex;justify-content:space-between;align-items:baseline}} .metric span{{color:var(--muted)}} .metric strong{{font-size:24px;color:var(--navy)}}
.candidate{{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:20px;margin:0 0 18px;box-shadow:0 4px 14px rgba(15,23,42,.04)}}
.candidate header{{display:flex;justify-content:space-between;gap:16px;align-items:flex-start}} .candidate header>div{{display:grid;grid-template-columns:auto auto;column-gap:12px;align-items:center}} .rank{{font:700 12px ui-monospace,monospace;color:var(--muted);grid-row:1/3}} h2{{font:700 26px Georgia,serif;margin:0;color:var(--navy)}} header p{{margin:0;color:var(--muted)}}
.badge{{font-size:11px;font-weight:800;letter-spacing:.06em;padding:6px 9px;border-radius:999px;background:#e2e8f0;white-space:nowrap}} .badge.research_preview_eligible{{background:#dcfce7;color:#166534}} .badge.watch{{background:#fef3c7;color:#92400e}} .badge.reject{{background:#fee2e2;color:#991b1b}}
.tape{{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0}} .tape span{{background:#f8fafc;border:1px solid var(--line);border-radius:7px;padding:7px 9px;color:var(--muted)}} .tape b{{color:var(--ink)}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}} section{{border-top:2px solid #dbeafe;padding-top:8px}} h3{{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--blue);margin:0 0 4px}} section p{{margin:0}}
.preview{{margin:16px 0 8px;padding:12px;background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;display:flex;flex-wrap:wrap;gap:8px 18px}} .preview strong{{width:100%;color:#1e3a8a}}
details{{border-top:1px solid var(--line);margin-top:14px;padding-top:10px}} summary{{cursor:pointer;font-weight:700;color:var(--navy)}} .sources{{padding-left:20px}} .sources li{{margin:8px 0}} .sources small{{display:block;color:var(--muted)}} a{{color:var(--blue)}} .empty{{background:white;padding:28px;border-radius:10px}}
footer{{color:var(--muted);font-size:12px;margin-top:30px}} @media(max-width:800px){{.metrics,.grid{{grid-template-columns:1fr 1fr}}}} @media(max-width:520px){{.metrics,.grid{{grid-template-columns:1fr}} h1{{font-size:32px}}}}
</style></head><body><main>
<div class="eyebrow">Episodic Pivot · Shadow Research</div><h1>Daily event triage</h1>
<p class="dek">A broad mover screen narrowed by fetched causal news, materiality, freshness, extension, and current-liquidity gates. Queue position allocates research attention; it is not an investment recommendation.</p>
<div class="safety"><b>Research only.</b> This artifact cannot submit, stage, approve, route, publish, or deploy an order. Every hypothetical sizing object is non-executable and requires a separate future design plus explicit approval.</div>
<div class="metrics">{card_html}</div>{empty}{''.join(candidate_html)}
<footer>Run {html.escape(result.run_id)} · generated {html.escape(result.generated_at)} · policy {html.escape(policy.policy_id)} · broker route NONE</footer>
</main></body></html>"""


def write_run_artifacts(
    result: RunResult,
    *,
    policy: EPPolicy,
    output_dir: str | Path,
    input_files: dict[str, str | Path] | None = None,
    search_provider: str = "OFFLINE",
) -> Path:
    _validate_research_previews(result.previews)
    root = Path(output_dir).resolve()
    if root.exists():
        existing_manifest = root / "manifest.json"
        if not existing_manifest.exists():
            raise FileExistsError(f"refusing to reuse non-run directory: {root}")
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if existing.get("run_id") != result.run_id:
            raise FileExistsError(f"run directory belongs to another run: {root}")
    else:
        root.mkdir(parents=True, exist_ok=False)

    input_manifest = {
        name: {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
        for name, path in (input_files or {}).items()
    }
    manifest = {
        "schema_version": 2,
        "run_id": result.run_id,
        "generated_at": result.generated_at,
        "policy": policy.to_dict(),
        "search_provider": search_provider,
        "inputs": input_manifest,
        "counts": {
            "candidates": len(result.candidates),
            "decisions": len(result.decisions),
            "research_sizing_previews": len(result.previews),
        },
        "safety": {
            "research_only": True,
            "live_actions_enabled": False,
            "broker_route": "NONE",
            "order_submission_allowed": False,
            "order_staging_performed": False,
            "broker_contacted": False,
            "sheets_written": False,
            "r2_written": False,
            "publishing_performed": False,
            "production_deployed": False,
        },
    }
    _json_dump(root / "candidates.json", [item.to_dict() for item in result.candidates])
    _json_dump(
        root / "evidence.json",
        {
            candidate_id: [document.to_dict() for document in documents]
            for candidate_id, documents in result.documents_by_candidate.items()
        },
    )
    _json_dump(
        root / "evidence_by_symbol.json",
        {
            candidate.snapshot.symbol: [
                document.to_dict()
                for document in result.documents_by_candidate.get(candidate.candidate_id, [])
            ]
            for candidate in result.candidates
        },
    )
    _json_dump(root / "decisions.json", [item.to_dict() for item in result.decisions])
    preview_rows = [item.to_dict() for item in result.previews]
    _json_dump(root / "research_sizing_preview.json", preview_rows)

    headers = (
        list(result.previews[0].to_dict())
        if result.previews
        else [item.name for item in fields(ResearchSizingPreview)]
    )
    with (root / "research_sizing_preview.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in preview_rows:
            row = dict(row)
            row["evidence_urls"] = " | ".join(row.get("evidence_urls") or ())
            row["evidence_published_at"] = " | ".join(
                row.get("evidence_published_at") or ()
            )
            writer.writerow({key: _csv_safe(value) for key, value in row.items()})

    (root / "report.md").write_text(_report(result, policy), encoding="utf-8")
    (root / "report.html").write_text(
        _html_report(result, policy), encoding="utf-8"
    )
    artifact_names = (
        "candidates.json",
        "evidence.json",
        "evidence_by_symbol.json",
        "decisions.json",
        "research_sizing_preview.json",
        "research_sizing_preview.csv",
        "report.md",
        "report.html",
    )
    manifest["artifacts"] = {
        name: {
            "sha256": sha256_file(root / name),
            "size_bytes": (root / name).stat().st_size,
        }
        for name in artifact_names
    }
    _json_dump(root / "manifest.json", manifest)
    return root
