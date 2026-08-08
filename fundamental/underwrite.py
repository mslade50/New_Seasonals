"""Durable, PM-facing decision records for completed company underwrites."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


DECISION_STATUSES = {"QUICK_REVIEW", "WAIT_FOR_PROOF", "WAIT_FOR_EVENT", "PASS"}


def load_underwrite_decisions(path: Path) -> list[dict[str, Any]]:
    """Load and validate the latest human/agent-authored underwriting decisions."""
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("decisions", payload if isinstance(payload, list) else [])
    if not isinstance(records, list):
        raise ValueError("underwrite decisions must be a list or a {'decisions': [...]} object")
    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        ticker = str(record.get("ticker") or "").upper().strip()
        status = str(record.get("decision") or "").upper().strip()
        if not ticker or status not in DECISION_STATUSES:
            raise ValueError(f"invalid underwrite decision: ticker={ticker!r}, decision={status!r}")
        if ticker in seen:
            raise ValueError(f"duplicate underwrite decision for {ticker}")
        seen.add(ticker)
        cleaned.append({**record, "ticker": ticker, "decision": status})
    return cleaned


def _esc(value: Any) -> str:
    return html.escape("—" if value is None else str(value))


def _items(values: list[str] | None) -> str:
    return "".join(f"<li>{_esc(value)}</li>" for value in (values or []))


def render_underwrite(record: dict[str, Any], candidate: pd.Series, output_path: Path) -> Path:
    """Render one source-backed decision page; it intentionally has no execution path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status = str(record["decision"])
    status_label = status.replace("_", " ")
    source_rows = "".join(
        "<tr>"
        f"<td><a href='{_esc(source.get('url'))}'>{_esc(source.get('label'))}</a></td>"
        f"<td>{_esc(source.get('as_of'))}</td><td>{_esc(source.get('use'))}</td>"
        "</tr>"
        for source in record.get("sources", [])
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(record['ticker'])} — Fundamental underwrite</title>
<style>
:root{{--bg:#0a1020;--panel:#121b2e;--panel2:#18243b;--text:#ecf1fa;--muted:#9ba9bd;--line:#293955;--blue:#78baff;--green:#67dca8;--amber:#ffd070;--red:#ff9a9a}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(180deg,#09101d,#0d1628 45%,#0a1020);color:var(--text);font:15px/1.55 Inter,Segoe UI,Arial,sans-serif}}main{{max-width:1050px;margin:auto;padding:34px 26px}}a{{color:var(--blue)}}
.eyebrow{{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.7px}}h1{{font-size:36px;line-height:1.1;margin:6px 0}}h2{{font-size:19px;margin:27px 0 10px}}p{{margin:6px 0}}.muted{{color:var(--muted)}}
.warning,.decision,.panel{{border-radius:12px;padding:16px 18px}}.warning{{background:#302817;border:1px solid #8b6b2c;color:#ffdc8a;margin:16px 0}}.decision{{background:linear-gradient(135deg,#16243b,#111a2d);border:1px solid #365172}}.decision b{{display:block;font-size:23px;margin:2px 0 7px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.panel{{background:rgba(18,27,46,.96);border:1px solid var(--line)}}.panel h2{{margin-top:0}}ul{{margin:5px 0;padding-left:20px}}li{{margin:6px 0}}table{{border-collapse:collapse;width:100%}}th{{text-align:left;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.4px;padding:8px;border-bottom:1px solid var(--line)}}td{{padding:9px 8px;border-bottom:1px solid #22304a;vertical-align:top}}footer{{color:var(--muted);font-size:12px;margin-top:28px}}
@media(max-width:720px){{main{{padding:22px 15px}}h1{{font-size:30px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div class="eyebrow">Completed fundamental underwrite · {_esc(record.get('as_of'))}</div>
<h1>{_esc(record['ticker'])} <span class="muted">{_esc(candidate.get('company_name'))}</span></h1>
<div class="muted">Screen score {_esc(candidate.get('research_score'))} · {_esc(candidate.get('trend_state'))} trend · price snapshot {_esc(record.get('price_as_of'))}</div>
<div class="warning">Research only. No allocation, position, order, or broker action is generated here.</div>
<section class="decision"><div class="eyebrow">Decision</div><b>{_esc(status_label)}</b><p>{_esc(record.get('verdict'))}</p></section>

<section class="grid">
<div class="panel"><h2>What could be mispriced</h2><p>{_esc(record.get('mispricing'))}</p></div>
<div class="panel"><h2>What appears priced in</h2><p>{_esc(record.get('priced_in'))}</p></div>
<div class="panel"><h2>Valuation read</h2><p>{_esc(record.get('valuation'))}</p></div>
<div class="panel"><h2>Trend and timing</h2><p>{_esc(record.get('trend'))}</p></div>
</section>

<section class="grid">
<div class="panel"><h2>Proof required</h2><ul>{_items(record.get('proof_required'))}</ul></div>
<div class="panel"><h2>Downside and kill conditions</h2><ul>{_items(record.get('kill_conditions'))}</ul></div>
</section>

<section class="panel"><h2>Next review</h2><p>{_esc(record.get('next_review'))}</p></section>
<h2>Primary-source ledger</h2><section class="panel"><table><thead><tr><th>Source</th><th>As of</th><th>Used for</th></tr></thead><tbody>{source_rows}</tbody></table></section>
<footer><a href="../fundamental_daily.html">← Back to the daily brief</a> · Underwrite state: {_esc(status_label)} · Live actions disabled.</footer>
</main></body></html>"""
    output_path.write_text(page, encoding="utf-8")
    return output_path


def build_underwrite_pack(
    decisions: list[dict[str, Any]],
    candidates: pd.DataFrame,
    output_dir: Path,
) -> dict[str, str]:
    """Render current decisions and return report-relative links by ticker."""
    links: dict[str, str] = {}
    if not decisions or candidates.empty:
        return links
    indexed = candidates.drop_duplicates("ticker").set_index("ticker")
    for record in decisions:
        ticker = record["ticker"]
        if ticker not in indexed.index:
            continue
        path = render_underwrite(record, indexed.loc[ticker], output_dir / f"{ticker}.html")
        links[ticker] = f"underwrites/{path.name}"
    return links
