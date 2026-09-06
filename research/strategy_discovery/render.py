"""Deterministic JSON, Markdown, and HTML strategy-discovery reports."""

from __future__ import annotations

import html
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .contracts import canonical_json, validate_report


MODE_BANNERS = {
    "DISABLED": "DISABLED — no source capture was processed; absence cannot be inferred.",
    "FIXTURE": "FIXTURE OUTPUT — synthetic/offline test data; not an operating research report.",
    "SHADOW": "NON-AUTHORITATIVE SHADOW OUTPUT — research observation only; do not trade or mutate the strategy book.",
    "LIVE": (
        "RESEARCH-ONLY LIVE OUTPUT — X remains discovery-only; no candidate is "
        "authorized for capital or execution."
    ),
}


def json_text(report: dict[str, Any]) -> str:
    validate_report(report)
    return json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list)):
        return canonical_json(value)
    return str(value)


def _md(value: Any, *, table: bool = False) -> str:
    """Neutralize raw HTML and table delimiters in untrusted Markdown text."""

    escaped = html.escape(_display(value), quote=False)
    return escaped.replace("|", "\\|") if table else escaped


def markdown_text(report: dict[str, Any]) -> str:
    validate_report(report)
    lines = [
        f"# {_md(report['title'])}",
        "",
        f"> {MODE_BANNERS[report['run_mode']]}",
        "",
        f"**As of:** {_md(report['as_of'])}  ",
        f"**Run:** `{report['run_id']}`  ",
        f"**Completeness:** **{report['completeness']}**  ",
        "**Authority:** research only; X is discovery-only; trading and strategy mutation are disabled.",
        "",
        "## Human summary",
        "",
    ]
    summary = report["summary"]
    lines.extend(
        [
            f"- Source items: {summary['raw_item_count']} raw / {summary['canonical_item_count']} canonical",
            f"- Candidates: {summary['candidate_count']}",
            f"- New and research-ready: {summary['new_research_ready']}",
            f"- Needs specification: {summary['needs_spec']}",
            f"- Quarantined: {summary['quarantined']}",
            f"- Internally validated: {summary['validated_research']}",
            f"- Explicit owner review: {summary['owner_review']}",
            "",
            "## Source coverage",
            "",
            "| Source | Status | Expected | Observed | Window | Notes |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for source in report["source_coverage"]:
        window = source["window"] or {}
        window_text = f"{window.get('start', '—')} → {window.get('end', '—')}"
        notes = _md(" ".join(source["findings"]), table=True)
        lines.append(
            f"| {_md(source['source_id'], table=True)} | {source['status']} | "
            f"{_md(source['expected_item_count'], table=True)} | "
            f"{source['file_observed_item_count']} | {_md(window_text, table=True)} | {notes} |"
        )
    lines.extend(["", "## Catalog health", ""])
    for catalog in report["catalog_health"]:
        lines.append(
            f"- **{catalog['catalog_type']} — {catalog['status']}:** "
            f"{catalog['record_count']} records; {_md(catalog['finding'])}"
        )
    lines.extend(["", "## Candidate funnel", ""])
    if not report["candidates"]:
        qualifier = (
            "The configured source window was completely observed and contained zero items."
            if report["completeness"] == "COMPLETE"
            else "Candidates cannot be inferred because coverage is not complete."
        )
        lines.extend([qualifier, ""])
    for index, candidate in enumerate(report["candidates"], 1):
        lines.extend(
            [
                f"### {index}. {_md(candidate['name'])}",
                "",
                f"- **Lifecycle:** {candidate['lifecycle']} (automatic ceiling: RESEARCH_READY)",
                f"- **Disposition:** {candidate['disposition']}",
                f"- **Edge status:** {candidate['edge_status']}",
                f"- **Fingerprint:** `{candidate['fingerprint']}`",
                f"- **Why it might belong (source thesis, not validated):** {_md(candidate['thesis'])}",
                f"- **Next action:** {_md(candidate['next_research_step'])}",
                "",
                "#### Incremental portfolio-role hypotheses (not validated)",
                "",
            ]
        )
        lines.extend(f"- {_md(value)}" for value in candidate["portfolio_fit_hypotheses"])
        lines.extend(["", "#### Falsifiers", ""])
        lines.extend(f"- {_md(value)}" for value in candidate["falsifiers"])
        structure = candidate["structure"]
        lines.extend(
            [
                "",
                "#### Executable hypothesis",
                "",
                f"- **Direction:** {structure['direction']}",
                f"- **Universe:** {_md(structure['universe']['scope'])}",
                f"- **Signal observation:** {structure['signal']['observation_timing']}",
            ]
        )
        for condition in structure["signal"]["conditions"]:
            lines.append(
                f"- **Condition:** {_md(condition['field'])} {_md(condition['operator'])} "
                f"{_md(condition['value'])} {_md(condition['unit'])}; "
                f"lookback {_md(condition['lookback_sessions'])} sessions"
            )
        entry = structure["entry"]
        lines.append(
            f"- **Entry:** session +{entry['session_offset']} {entry['timing']} "
            f"via {_md(entry['order_type'])}; price rule {_md(entry['price_rule'])}"
        )
        exit_spec = structure["exit"]
        lines.append(
            f"- **Exit:** time {_md(exit_spec['time_stop_sessions'])} sessions; "
            f"stop {_md(exit_spec['stop_rule'])}; target {_md(exit_spec['target_rule'])}"
        )
        for requirement in structure["data_requirements"]:
            lines.append(
                f"- **Data:** {_md(requirement['field'])}, {_md(requirement['frequency'])}, "
                f"available {_md(requirement['availability'])}"
            )
        lines.extend(["", "#### Gates", ""])
        for gate in candidate["gates"]:
            lines.append(f"- {gate['status']} — **{gate['gate']}**: {_md(gate['reason'])}")
        lines.extend(["", "#### Source-claimed metrics", ""])
        if candidate["source_claimed_metrics"]:
            for metric in candidate["source_claimed_metrics"]:
                lines.append(
                    f"- SOURCE_CLAIMED — {_md(metric['name'])}: {_md(metric['value'])} "
                    f"{_md(metric['unit'])} (N={_md(metric['sample_size'])}); {_md(metric['definition'])}"
                )
        else:
            lines.append("- None supplied. No edge is inferred.")
        lines.extend(["", "#### Source claims", ""])
        if candidate["source_claims"]:
            for claim in candidate["source_claims"]:
                lines.append(
                    f"- SOURCE_CLAIMED — {claim['claim_type']}: {_md(claim['text'])} "
                    f"(post {_md(claim['post_id'])})"
                )
        else:
            lines.append("- None supplied.")
        lines.extend(["", "#### Internally validated metrics", ""])
        if candidate["internally_validated_metrics"]:
            for metric in candidate["internally_validated_metrics"]:
                lines.append(
                    f"- {_md(metric['name'])}: {_md(metric['value'])} {_md(metric['unit'])} "
                    f"(artifact `{_md(metric['artifact_id'])}`; N={_md(metric['sample_size'])})"
                )
        else:
            lines.append("- None. Source claims have not been validated internally.")
        lines.extend(["", "#### Provenance", ""])
        for source in candidate["provenance"]:
            lines.append(
                f"- {source['kind']} {_md(source['post_id'])} by {_md(source['author_handle'])} — "
                f"{_md(source['permalink'])}"
            )
        lines.append("")
    lines.extend(["## Limits", ""])
    lines.extend(f"- {_md(limitation)}" for limitation in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def html_text(report: dict[str, Any]) -> str:
    validate_report(report)
    esc = lambda value: html.escape(str(value), quote=True)
    summary = report["summary"]
    source_rows = []
    for source in report["source_coverage"]:
        window = source["window"] or {}
        source_rows.append(
            "<tr>"
            f"<td>{esc(source['source_id'])}</td>"
            f"<td><strong>{esc(source['status'])}</strong></td>"
            f"<td>{esc(_display(source['expected_item_count']))}</td>"
            f"<td>{esc(source['file_observed_item_count'])}</td>"
            f"<td>{esc(window.get('start', '—'))}<br>{esc(window.get('end', '—'))}</td>"
            f"<td>{esc(' '.join(source['findings']))}</td>"
            "</tr>"
        )
    candidate_cards = []
    for candidate in report["candidates"]:
        structure = candidate["structure"]
        gates = "".join(
            f"<li><strong>{esc(gate['status'])} — {esc(gate['gate'])}</strong>: "
            f"{esc(gate['reason'])}</li>"
            for gate in candidate["gates"]
        )
        source_metrics = "".join(
            f"<li><span class='claim'>SOURCE_CLAIMED</span> — {esc(metric['name'])}: "
            f"{esc(_display(metric['value']))} {esc(metric['unit'])}; {esc(metric['definition'])}</li>"
            for metric in candidate["source_claimed_metrics"]
        ) or "<li>None supplied. No edge is inferred.</li>"
        source_claims = "".join(
            f"<li><span class='claim'>SOURCE_CLAIMED</span> — "
            f"{esc(claim['claim_type'])}: {esc(claim['text'])} "
            f"(post {esc(claim['post_id'])})</li>"
            for claim in candidate["source_claims"]
        ) or "<li>None supplied.</li>"
        internal_metrics = "".join(
            f"<li>{esc(metric['name'])}: {esc(_display(metric['value']))} {esc(metric['unit'])} "
            f"(artifact {esc(metric['artifact_id'])})</li>"
            for metric in candidate["internally_validated_metrics"]
        ) or "<li>None. Source claims have not been validated internally.</li>"
        provenance = "".join(
            f"<li>{esc(source['kind'])} {esc(source['post_id'])} by {esc(source['author_handle'])} — "
            f"<a href='{esc(source['permalink'])}' rel='noreferrer'>{esc(source['permalink'])}</a></li>"
            for source in candidate["provenance"]
        )
        fit = "".join(
            f"<li>{esc(value)}</li>" for value in candidate["portfolio_fit_hypotheses"]
        )
        falsifiers = "".join(f"<li>{esc(value)}</li>" for value in candidate["falsifiers"])
        conditions = "".join(
            f"<li>{esc(condition['field'])} {esc(condition['operator'])} "
            f"{esc(_display(condition['value']))} {esc(_display(condition['unit']))}; "
            f"lookback {esc(_display(condition['lookback_sessions']))} sessions</li>"
            for condition in structure["signal"]["conditions"]
        )
        requirements = "".join(
            f"<li>{esc(requirement['field'])}, {esc(requirement['frequency'])}, "
            f"available {esc(requirement['availability'])}</li>"
            for requirement in structure["data_requirements"]
        )
        entry = structure["entry"]
        exit_spec = structure["exit"]
        execution = (
            f"<li><strong>Direction:</strong> {esc(structure['direction'])}</li>"
            f"<li><strong>Universe:</strong> {esc(structure['universe']['scope'])}</li>"
            f"<li><strong>Signal observation:</strong> "
            f"{esc(structure['signal']['observation_timing'])}</li>"
            f"{conditions}"
            f"<li><strong>Entry:</strong> session +{esc(entry['session_offset'])} "
            f"{esc(entry['timing'])} via {esc(entry['order_type'])}; "
            f"price rule {esc(_display(entry['price_rule']))}</li>"
            f"<li><strong>Exit:</strong> time {esc(_display(exit_spec['time_stop_sessions']))} "
            f"sessions; stop {esc(_display(exit_spec['stop_rule']))}; "
            f"target {esc(_display(exit_spec['target_rule']))}</li>"
            f"{requirements}"
        )
        candidate_cards.append(
            "<section class='card'>"
            f"<h3>{esc(candidate['name'])}</h3>"
            f"<p><strong>{esc(candidate['lifecycle'])}</strong> · {esc(candidate['disposition'])} · "
            f"{esc(candidate['edge_status'])}</p>"
            f"<p><strong>Source thesis, not validated:</strong> {esc(candidate['thesis'])}</p>"
            f"<p><strong>Next:</strong> {esc(candidate['next_research_step'])}</p>"
            f"<h4>Incremental portfolio-role hypotheses (not validated)</h4><ul>{fit}</ul>"
            f"<h4>Falsifiers</h4><ul>{falsifiers}</ul>"
            f"<h4>Executable hypothesis</h4><ul>{execution}</ul>"
            f"<h4>Gates</h4><ul>{gates}</ul>"
            f"<h4>Source-claimed metrics</h4><ul>{source_metrics}</ul>"
            f"<h4>Source claims</h4><ul>{source_claims}</ul>"
            f"<h4>Internally validated metrics</h4><ul>{internal_metrics}</ul>"
            f"<h4>Provenance</h4><ul>{provenance}</ul>"
            "</section>"
        )
    if not candidate_cards:
        qualifier = (
            "The configured source window was completely observed and contained zero items."
            if report["completeness"] == "COMPLETE"
            else "Candidates cannot be inferred because coverage is not complete."
        )
        candidate_cards.append(f"<section class='card'><p>{esc(qualifier)}</p></section>")
    limitations = "".join(f"<li>{esc(value)}</li>" for value in report["limitations"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(report['title'])}</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;max-width:1120px;margin:30px auto;
padding:0 20px;color:#17202a;background:#f5f7fa}}
.banner{{border:2px solid #8a5a00;background:#fff4cf;padding:14px;font-weight:700}}
.meta,.card,table{{background:white;border:1px solid #d9e0e7;border-radius:8px;padding:16px;margin:16px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}}
.metric{{background:#eef3f7;border-radius:6px;padding:10px}} table{{width:100%;border-collapse:collapse}}
th,td{{text-align:left;vertical-align:top;border-bottom:1px solid #e6ebef;
padding:8px}} .claim{{color:#8a5a00;font-weight:700}}
code{{overflow-wrap:anywhere}} a{{color:#135e96}}
</style></head><body>
<h1>{esc(report['title'])}</h1>
<div class="banner">{esc(MODE_BANNERS[report['run_mode']])}</div>
<div class="meta"><strong>As of:</strong> {esc(report['as_of'])}<br>
<strong>Completeness:</strong> {esc(report['completeness'])}<br>
<strong>Authority:</strong> research only; X is discovery-only; trading and strategy mutation are disabled.<br>
<strong>Run:</strong> <code>{esc(report['run_id'])}</code></div>
<h2>Human summary</h2><div class="grid">
<div class="metric">Items<br><strong>{summary['raw_item_count']} /
{summary['canonical_item_count']}</strong><br>raw / canonical</div>
<div class="metric">Candidates<br><strong>{summary['candidate_count']}</strong></div>
<div class="metric">Research-ready<br><strong>{summary['new_research_ready']}</strong></div>
<div class="metric">Needs spec<br><strong>{summary['needs_spec']}</strong></div>
<div class="metric">Quarantined<br><strong>{summary['quarantined']}</strong></div>
</div>
<h2>Source coverage</h2><table><thead><tr><th>Source</th><th>Status</th>
<th>Expected</th><th>Observed</th><th>Window</th><th>Notes</th></tr></thead>
<tbody>{''.join(source_rows)}</tbody></table>
<h2>Candidate funnel</h2>{''.join(candidate_cards)}
<h2>Limits</h2><ul>{limitations}</ul>
</body></html>"""


def atomic_write_text(path: Path, text: str) -> None:
    """Write a complete artifact via same-directory replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_report_bundle(output_dir: Path, report: dict[str, Any]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "strategy_discovery_report.json",
        output_dir / "strategy_discovery_report.md",
        output_dir / "strategy_discovery_report.html",
    ]
    contents = [json_text(report), markdown_text(report), html_text(report)]
    for path, content in zip(paths, contents, strict=True):
        atomic_write_text(path, content)
    return paths
