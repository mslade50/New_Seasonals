"""Point-in-time funding assertions, distinct from announcement outcome labels.

No network or execution integration. A priced deal, undrawn facility, cumulative
XBRL flow, and confirmed cash receipt are deliberately different objects.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timedelta
import hashlib
import math
from zoneinfo import ZoneInfo

from .cash_runway import FORMS, filing_rows, filing_url, utc


ET = ZoneInfo("America/New_York")
STAGES = {"announcement", "pricing", "receipt", "capacity", "cancellation"}
BASES = {"net", "estimated_net", "gross", "unspecified", "capacity"}
# These are evidence series, NOT mutually exclusive/additive cash categories.
FINANCING_TAGS = {
    "NetCashProvidedByUsedInFinancingActivities": "net_financing",
    "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations": "net_financing",
    "ProceedsFromIssuanceOfCommonStock": "equity",
    "ProceedsFromIssuanceOrSaleOfEquity": "equity",
    "ProceedsFromIssuanceOfPrivatePlacement": "equity",
    "ProceedsFromIssuanceOfWarrants": "warrants",
    "ProceedsFromWarrantExercises": "warrants",
    "ProceedsFromStockOptionsExercised": "employee_equity",
    "ProceedsFromStockPlans": "employee_equity",
    "ProceedsFromIssuanceOfSharesUnderIncentiveAndShareBasedCompensationPlansIncludingStockOptions": "employee_equity",
    "ProceedsFromIssuanceOfSharesUnderIncentiveAndShareBasedCompensationPlans": "employee_equity",
    "ProceedsFromIssuanceOfDebt": "debt",
    "ProceedsFromIssuanceOfLongTermDebt": "debt",
    "ProceedsFromConvertibleDebt": "convertible_debt",
    "ProceedsFromDebtNetOfIssuanceCosts": "debt",
    "ProceedsFromLinesOfCredit": "credit_draws",
    "ProceedsFromLongTermLinesOfCredit": "credit_draws",
    "RepaymentsOfDebt": "debt_repayment",
    "RepaymentsOfLongTermDebt": "debt_repayment",
    "PaymentsForRepurchaseOfCommonStock": "buybacks",
}


def availability_bounds(record):
    """Date-only source means some time that ET day, never a guessed midnight."""
    if record.get("available_at"):
        stamp = utc(record["available_at"])
        return stamp, stamp
    day = date.fromisoformat(record["available_date"])
    lower = datetime.combine(day, time.min, ET).astimezone(utc("2000-01-01T00:00:00Z").tzinfo)
    upper = datetime.combine(day + timedelta(days=1), time.min, ET).astimezone(lower.tzinfo)
    return lower, upper


def knowledge(record, as_of):
    lower, upper = availability_bounds(record)
    cutoff = utc(as_of)
    if cutoff < lower:
        return "future"
    if cutoff < upper:
        return "same_day_time_unknown"
    return "known"


def validate_records(records):
    ids = set()
    owners = {}
    for r in records:
        key = r["record_id"]
        if key in ids:
            raise ValueError(f"Duplicate record_id: {key}")
        ids.add(key)
        cik = int(r["cik"])
        if cik <= 0 or not r.get("funding_id") or not r.get("sources"):
            raise ValueError(f"Missing issuer, funding identity or sources: {key}")
        for source in r["sources"]:
            if not source.startswith("https://"):
                raise ValueError(f"Expected HTTPS evidence: {key}")
        if owners.setdefault(r["funding_id"], cik) != cik:
            raise ValueError(f"Funding ID crosses issuers: {key}")
        if r["stage"] not in STAGES or r["status"] not in {"verified", "provisional"}:
            raise ValueError(f"Invalid stage/status: {key}")
        availability_bounds(r)
        if r.get("amount_basis") not in BASES:
            raise ValueError(f"Invalid amount basis: {key}")
        amount = r.get("amount_usd")
        if amount is not None and (not math.isfinite(amount) or amount < 0):
            raise ValueError(f"Invalid cash amount: {key}")
        if r["stage"] == "receipt":
            if not r.get("receipt_id") or not r.get("cash_start") or not r.get("cash_end"):
                raise ValueError(f"Receipt needs tranche ID and cash interval: {key}")
            start, end = date.fromisoformat(r["cash_start"]), date.fromisoformat(r["cash_end"])
            if end < start:
                raise ValueError(f"Reversed receipt interval: {key}")
            # Never label a future/expected settlement as received.
            public_day = availability_bounds(r)[0].astimezone(ET).date()
            if end > public_day:
                raise ValueError(f"Receipt predates its occurrence: {key}")
            if r.get("amount_scope") != "incremental":
                raise ValueError(f"Receipt must be an incremental, deduplicated tranche: {key}")
    return records


def current_receipts(records, as_of):
    """Latest known net evidence per tranche; gross repeats cannot erase net.

    Amount bases are complementary; a net correction must provide new net.
    """
    groups = defaultdict(list)
    for r in records:
        if r["stage"] == "receipt" and r["status"] == "verified" and knowledge(r, as_of) == "known":
            groups[(int(r["cik"]), r["funding_id"], r["receipt_id"])].append(r)
    chosen, conflicts = [], []
    for rows in groups.values():
        net = [r for r in rows if r["amount_basis"] in {"net", "estimated_net"} and r.get("amount_usd") is not None]
        gross = [r for r in rows if r["amount_basis"] == "gross" and r.get("amount_usd") is not None]
        preferred = net or gross or rows
        newest = max(availability_bounds(r)[1] for r in preferred)
        latest = [r for r in preferred if availability_bounds(r)[1] == newest]
        signatures = {(r.get("amount_usd"), r["amount_basis"], r["cash_start"], r["cash_end"]) for r in latest}
        if len(signatures) != 1:
            conflicts.extend(r["record_id"] for r in latest)
        else:
            chosen.append(sorted(latest, key=lambda r: r["record_id"])[0])
    return chosen, conflicts


def reconcile(observation, financial, records):
    """Known funding overlay; never estimates today's cash or proves no funding."""
    own = [r for r in records if int(r["cik"]) == int(observation["cik"])]
    cutoff, balance = observation["as_of"], observation.get("balance_date")
    visible = [r for r in own if knowledge(r, cutoff) == "known" and r["status"] == "verified"]
    ambiguous = [r["record_id"] for r in own if knowledge(r, cutoff) == "same_day_time_unknown"]
    receipts, conflicts = current_receipts(own, cutoff)
    result = dict(cik=observation["cik"], ticker=observation.get("ticker"),
        session=observation["session"], as_of=cutoff, group=observation["group"],
        balance_date=balance, reported_liquidity=financial.get("reported_liquidity"),
        reported_liquidity_source=financial.get("filing_url"),
        cash_only=observation.get("cash_only"), original_runway=observation.get("runway_operating"),
        net_receipts_usd=0.0, estimated_net_receipts_usd=0.0, gross_receipts_usd=0.0,
        unspecified_receipts_usd=0.0, added_receipt_ids=[], embedded_receipt_ids=[],
        overlapping_receipt_ids=[], unknown_amount_receipt_ids=[],
        pending_funding_ids=[], capacity_funding_ids=[], conflict_record_ids=conflicts,
        ambiguous_record_ids=ambiguous, source_record_ids=[],
        funding_coverage="incomplete", outcome_coverage="unchanged_unknown_unless_verified_positive",
        current_cash_estimate=None, fully_reconciled=False, research_eligible=False)
    if not balance:
        result["status"] = "financial_balance_unavailable"
        return result
    related = set()
    after_balance_funding = set()
    cash_day = utc(cutoff).astimezone(ET).date().isoformat()
    for r in receipts:
        if r["cash_end"] <= balance:
            result["embedded_receipt_ids"].append(r["record_id"])
            continue
        after_balance_funding.add(r["funding_id"])
        related.add(r["record_id"])
        if r["cash_start"] <= balance or r["cash_end"] > cash_day:
            result["overlapping_receipt_ids"].append(r["record_id"])
            continue
        amount = r.get("amount_usd")
        if amount is None:
            result["unknown_amount_receipt_ids"].append(r["record_id"])
            continue
        field = {"net": "net_receipts_usd", "estimated_net": "estimated_net_receipts_usd",
                 "gross": "gross_receipts_usd", "unspecified": "unspecified_receipts_usd"}.get(r["amount_basis"])
        if not field:
            result["unknown_amount_receipt_ids"].append(r["record_id"])
            continue
        result[field] += amount
        result["added_receipt_ids"].append(r["record_id"])
    # Announcements can precede the balance date and still await settlement.
    for funding_id in sorted({r["funding_id"] for r in visible}):
        rows = [r for r in visible if r["funding_id"] == funding_id]
        latest = max(rows, key=lambda r: availability_bounds(r)[1])
        if latest["stage"] == "cancellation":
            continue
        if any(r["stage"] == "capacity" for r in rows):
            result["capacity_funding_ids"].append(funding_id)
        announcements = [r for r in rows if r["stage"] in {"announcement", "pricing"}]
        # A documented closing resolves the initial pending state. Further
        # optional tranches require separate assertions, never assumed cash.
        known_receipt = any(r["funding_id"] == funding_id for r in receipts)
        if announcements and not known_receipt:
            # Old announcements remain a review item rather than a forever
            # recent-funding exclusion; relevant pending receipts are explicit.
            recent = [r for r in announcements if r.get("event_date", "0001") >= balance
                      or r.get("expected_close", "0001") > balance]
            if recent:
                result["pending_funding_ids"].append(funding_id)
                related.update(r["record_id"] for r in recent)
    result["source_record_ids"] = sorted(related)
    result["documented_intervening_funding"] = bool(after_balance_funding or result["pending_funding_ids"])
    if conflicts or result["overlapping_receipt_ids"]:
        status = "receipt_allocation_needs_review"
    elif after_balance_funding:
        status = "cash_received_since_balance"
    elif result["pending_funding_ids"]:
        status = "funding_announced_receipt_unconfirmed"
    elif ambiguous:
        status = "same_day_timing_unresolved"
    else:
        status = "no_documented_intervening_funding_coverage_incomplete"
    result["status"] = status
    # A limited bridge, not current cash: NO burn extrapolation, gross-up,
    # facility capacity, or unverified cash is silently added.
    liquidity = financial.get("reported_liquidity")
    net = result["net_receipts_usd"]
    estimated = result["estimated_net_receipts_usd"]
    result["reported_plus_known_net_before_burn"] = liquidity + net + estimated if liquidity is not None else None
    result["bridge_includes_estimated_net"] = estimated > 0
    return result


def financing_facts(payload, submissions, start="2022-01-01", end="2025-12-31"):
    """Preserve every as-filed USD flow vintage; don't sum overlapping tags/YTDs."""
    cik = int(payload["cik"])
    filings = {r["accessionNumber"]: r for r in filing_rows(submissions)
               if r.get("form") in FORMS and r.get("acceptanceDateTime")}
    output, seen = [], set()
    for tag, concept in payload.get("facts", {}).get("us-gaap", {}).items():
        if tag not in FINANCING_TAGS:
            continue
        for f in concept.get("units", {}).get("USD", []):
            accession = f.get("accn")
            if accession not in filings or not f.get("start") or not start <= f.get("end", "") <= end:
                continue
            value = float(f["val"])
            if not math.isfinite(value):
                continue
            accepted = utc(filings[accession]["acceptanceDateTime"]).isoformat()
            key = (tag, f["start"], f["end"], accession, value)
            if key in seen:
                continue
            seen.add(key)
            output.append(dict(fact_id=hashlib.sha256(repr((cik, key)).encode()).hexdigest()[:20],
                cik=cik, tag=tag, label=concept.get("label", tag), category=FINANCING_TAGS[tag],
                value=value, unit="USD", start=f["start"], end=f["end"], accession=accession,
                accepted_at=accepted, source_url=filing_url(cik, accession, filings[accession].get("primaryDocument")),
                evidence="fact_source_reported", summable=False,
                limitation="Cumulative/overlapping concepts; net/gross and deal allocation require source-note audit"))
    conflicts = defaultdict(set)
    for r in output:
        conflicts[(r["tag"], r["start"], r["end"], r["accession"])].add(r["value"])
    for r in output:
        r["conflicting_values"] = len(conflicts[(r["tag"], r["start"], r["end"], r["accession"])]) > 1
    return sorted(output, key=lambda r: (r["accepted_at"], r["tag"], r["start"], r["end"], r["value"]))


def facts_available_at(facts, cik, as_of):
    """Latest known vintage per concept/period, rejecting conflicting values."""
    groups = defaultdict(list)
    for r in facts:
        if int(r["cik"]) == int(cik) and utc(r["accepted_at"]) <= utc(as_of):
            groups[(r["tag"], r["start"], r["end"])].append(r)
    result = []
    for rows in groups.values():
        latest = max(r["accepted_at"] for r in rows)
        selected = [r for r in rows if r["accepted_at"] == latest]
        if len({r["value"] for r in selected}) == 1 and not any(r["conflicting_values"] for r in selected):
            result.append(selected[0])
    return result
