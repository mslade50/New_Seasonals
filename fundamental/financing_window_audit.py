"""Outcome audit gates for a frozen financing pilot; incomplete negatives stay unknown."""
from datetime import date, timedelta
from .cash_runway import utc


def in_forward_window(event, window):
    day = event["announcement_date"]
    if not window["session"] <= day <= window["outcome_end"]:
        return False
    if day == window["session"]:
        if not event.get("announcement_at"):
            return None
        return utc(event["announcement_at"]) > utc(window["as_of"])
    return True


def audit_outcome(window, review, documents, events):
    """Positive evidence can resolve a window despite incomplete negative coverage.

    Mixed debt/equity financings are shown separately because the original
    standalone PIPE endpoint does not settle every bundled transaction.
    """
    expected = set(window["filings"])
    relevant = [d for d in documents if window["window_id"] in d["window_ids"]]
    primary = {d["accession"] for d in relevant if d.get("file_type") != "linked_exhibit" and not d.get("error")}
    gaps = []
    expected_exhibits = {url for d in relevant for url in d.get("linked_exhibits", [])}
    captured_urls = {d.get("url") for d in relevant if not d.get("error")}
    if expected_exhibits - captured_urls:
        gaps.append("missing_linked_exhibits")
    if expected - primary:
        gaps.append("missing_primary_filings")
    if any(d.get("error") for d in relevant):
        gaps.append("document_capture_errors")
    if not review.get("sec_inventory_reviewed"):
        gaps.append("sec_inventory_not_reviewed")
    if not review.get("exhibits_reviewed"):
        gaps.append("exhibits_not_reviewed")
    if not review.get("financial_notes_reviewed"):
        gaps.append("financial_notes_not_reviewed")
    if not review.get("ir_inventory_reviewed") or not review.get("ir_sources"):
        gaps.append("issuer_release_coverage_incomplete")
    if not review.get("evidence_sources"):
        gaps.append("missing_review_evidence")
    gaps.extend(review.get("gaps", []))
    matches, mixed, same_day = [], [], []
    seen = set()
    for e in events:
        if int(e["cik"]) != int(window["cik"]) or e["endpoint_class"] not in {"primary_equity", "mixed_debt_equity"}:
            continue
        if e.get("status") != "verified":
            if in_forward_window(e, window) is not False:
                gaps.append("unresolved_primary_event_candidate")
            continue
        if e["event_id"] in seen:
            raise ValueError("Duplicate audited event identity")
        seen.add(e["event_id"])
        if not e.get("sources"):
            raise ValueError("Verified event missing evidence")
        inside = in_forward_window(e, window)
        if inside is None:
            same_day.append(e["event_id"])
        elif inside:
            if e["endpoint_class"] == "primary_equity":
                matches.append(e["event_id"])
            elif e["endpoint_class"] == "mixed_debt_equity":
                mixed.append(e["event_id"])
    if same_day:
        gaps.append("same_day_announcement_time_unknown")
    if same_day:
        label, status = None, "unknown_coverage"
    elif matches:
        label, status = 1, "verified_primary_offering"
    elif mixed:
        label, status = None, "mixed_financing_boundary"
    elif not gaps:
        label, status = 0, "reviewed_no_primary_offering"
    else:
        label, status = None, "unknown_coverage"
    return dict(window_id=window["window_id"], outcome=label, outcome_status=status,
        primary_events=matches, mixed_events=mixed, gaps=sorted(set(gaps)),
        no_event_review_complete=not gaps,
        inclusive_cash_equity_outcome=None if same_day else (1 if matches or mixed else label),
        strict_standalone_outcome=None if same_day else (1 if matches else (0 if not gaps else None)),
        primary_documents=len(primary), exhibit_documents=sum(d.get("file_type") == "linked_exhibit" for d in relevant))


def validate_window(window):
    expected = (date.fromisoformat(window["session"]) + timedelta(days=60)).isoformat()
    if window["outcome_end"] != expected:
        raise ValueError("Audit horizon changed from frozen 60 calendar days")
    utc(window["as_of"])
    if not window.get("filings"):
        raise ValueError("Empty filing inventory cannot certify a negative")
