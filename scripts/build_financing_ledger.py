"""Build a local funding-evidence ledger from the frozen free-source pilot.

Offline by design. Source capture/audit is an explicit separate operation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from fundamental.cash_runway import filing_rows, filing_url
from fundamental.financing_ledger import (financing_facts, reconcile,
    validate_records, facts_available_at)
from fundamental.financing_ledger_report import render
from scripts.build_financing_history import verify_frozen_inputs


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def imported_announcements(review):
    """Existing first-announcement audits supply dates, not later-priced terms."""
    rows = []
    for e in review["events"]:
        r = dict(record_id=e["event_id"] + ":announcement", funding_id=e["event_id"],
            cik=int(e["cik"]), ticker=e["ticker_at_event"], kind="primary_equity",
            stage="announcement", status=e["status"], event_date=e["announcement_date"],
            available_date=e["announcement_date"], amount_usd=None, amount_basis="unspecified",
            sources=e["sources"], note=e.get("caveat", "") +
                " Imported date audit only; later-priced headline amount omitted to prevent lookahead.")
        if e.get("announcement_at"):
            r["available_at"] = e["announcement_at"]
        rows.append(r)
    return rows


def source_index(records, roots, submissions):
    result = []
    for url in sorted({s for r in records for s in r["sources"]}):
        key = hashlib.sha256(url.encode()).hexdigest()
        entry = dict(source_id=key[:20], url=url, raw_capture_status="not_archived",
                     record_ids=[r["record_id"] for r in records if url in r["sources"]])
        for root in roots:
            meta, body = root / "capture" / (key + ".json"), root / "capture" / (key + ".bin")
            if meta.exists() and body.exists():
                metadata = read(meta)
                actual = digest(body)
                if actual != metadata["sha256"] or metadata["url"] != url:
                    raise ValueError(f"Source capture mismatch: {url}")
                entry.update(raw_capture_status="archived_verified", sha256=actual,
                             raw_path=str(body.resolve()), retrieved_at=metadata["fetched_at"])
                break
        result.append(entry)
    return result


def build(pilot, out, supplements):
    if pilot.resolve() == out.resolve():
        raise ValueError("Ledger must use a new output directory; preserve the frozen pilot")
    if not out.resolve().is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("Research outputs must remain under artifacts/")
    verify_frozen_inputs(pilot)
    out.mkdir(parents=True, exist_ok=True)
    review = read(pilot / "event_reviews.json")
    supplement = read(supplements)
    records = validate_records(imported_announcements(review) + supplement["records"])
    observations = read(pilot / "observations.json")
    financials = {(int(r["cik"]), r["as_of"]): r for r in read(pilot / "financial_vintages.json")}
    cohort = pd.read_csv(pilot / "cohort.csv").to_dict("records")
    targets = [r for r in observations if r["group"] == "strong_short"]
    priority = {int(r["cik"]) for r in targets}
    inputs = [pilot / n for n in ["cohort.csv", "protocol.json", "cohort_manifest.json",
        "event_reviews.json", "observations.json", "financial_vintages.json"]] + [supplements]
    facts, queue, subs = [], [], {}
    for company in cohort:
        cik = int(company["cik"])
        folder = pilot / "issuers" / str(cik)
        subfile, factfile = folder / "submissions.json", folder / "companyfacts.json"
        if not subfile.exists():
            queue.append(dict(cik=cik, name=company["name"], priority=cik in priority,
                              status="submissions_missing", filings_to_review=None))
            continue
        inputs.append(subfile)
        sub = subs[cik] = read(subfile)
        ownfacts = financing_facts(read(factfile), sub) if factfile.exists() else []
        if factfile.exists():
            inputs.append(factfile)
        facts.extend(ownfacts)
        filing_inventory = [r for r in filing_rows(sub) if r.get("form") in {"8-K", "8-K/A", "10-Q", "10-K", "10-Q/A", "10-K/A"}
                            and "2022-10-01" <= r.get("filingDate", "") <= "2026-03-31"]
        discoveryfile = pilot / "event_discovery" / f"{cik}.json"
        discovery = read(discoveryfile) if discoveryfile.exists() else {}
        if discoveryfile.exists():
            inputs.append(discoveryfile)
        captured = {d.get("accession") for d in discovery.get("documents", [])}
        queue.append(dict(cik=cik, name=company["name"], priority=cik in priority,
            status="not_comprehensively_reviewed", financing_fact_rows=len(ownfacts),
            financial_facts_status="available" if factfile.exists() else "missing",
            filings_to_review=len(filing_inventory), search_captured_accessions=len(captured),
            filings_without_search_capture=sum(f["accessionNumber"] not in captured for f in filing_inventory),
            reviewed_record_count=sum(int(r["cik"]) == cik for r in records),
            filings=[dict(accession=f["accessionNumber"], form=f["form"], date=f["filingDate"],
                          url=filing_url(cik, f["accessionNumber"], f.get("primaryDocument")),
                          search_captured=f["accessionNumber"] in captured, audited=False) for f in filing_inventory]))
    reconciled = [reconcile(r, financials.get((int(r["cik"]), r["as_of"]), {}), records) for r in observations]
    signals = [r for r in reconciled if r["group"] == "strong_short"]
    # Source facts near each signal are explicitly already within the reported
    # balance period, not extra cash to add again.
    by_cik = {}
    for f in facts:
        by_cik.setdefault(f["cik"], []).append(f)
    for row in signals:
        row["financing_fact_ids_at_cutoff"] = [f["fact_id"] for f in facts_available_at(by_cik.get(row["cik"], []), row["cik"], row["as_of"])
            if f["end"] == row["balance_date"]]
        old_review = next((s for s in review.get("signals", []) if int(s["cik"]) == row["cik"] and s["session"] == row["session"]), {})
        row["prior_review_note"] = old_review.get("note", "")
        row["prior_review_source"] = old_review.get("source")
    sources = source_index(records, [out, pilot], subs)
    # A reviewed amount can lack local raw bytes; keep that retrieval gap visible.
    stats = dict(companies=len(cohort), observations=len(reconciled), signals=len(signals),
        signal_issuers=len(priority), assertions=len(records), funding_events=len({r["funding_id"] for r in records}),
        receipt_assertions=sum(r["stage"] == "receipt" for r in records), financing_facts=len(facts),
        fact_issuers=len({r["cik"] for r in facts}),
        signal_status_counts=dict(Counter(r["status"] for r in signals)),
        signals_with_documented_funding=sum(r.get("documented_intervening_funding", False) for r in signals),
        sources=len(sources), archived_sources=sum(s["raw_capture_status"] == "archived_verified" for s in sources),
        certified_negative_windows=0, validated_strategy=False)
    result = dict(generated_at=datetime.now(timezone.utc).isoformat(), stats=stats, signals=signals,
                  records=records, sources=sources, queue=queue,
                  scope="Frozen 90-issuer 2023–2025 pilot, with boundary funding evidence; reviewed subset, not exhaustive coverage",
                  conclusion="Funding history corrects stale-cash flags; offering predictability remains unproven")
    for name, value in [("ledger.json", records), ("financing_facts.json", facts), ("reconciled_observations.json", reconciled),
                        ("signal_reconciliation.json", signals), ("review_queue.json", queue), ("source_index.json", sources), ("analysis.json", result)]:
        save(out / name, value)
    for name, value in [("ledger.csv", records), ("financing_facts.csv", facts),
                        ("signal_reconciliation.csv", signals), ("source_index.csv", sources)]:
        pd.DataFrame(value).to_csv(out / name, index=False)
    report = out / "financing_ledger.html"
    report.write_text(render(result), encoding="utf-8")
    # Check the core historical invariants against the actual saved run.
    record_map = {r["record_id"]: r for r in records}
    from fundamental.financing_ledger import knowledge
    assert len({(r["cik"], r["session"]) for r in reconciled}) == len(observations)
    for row in reconciled:
        for rid in row["added_receipt_ids"]:
            receipt = record_map[rid]
            assert knowledge(receipt, row["as_of"]) == "known"
            assert receipt["cash_start"] > row["balance_date"]
        assert row["fully_reconciled"] is False and row["current_cash_estimate"] is None
    qa = dict(status="PASS", checks=["frozen cohort/protocol hashes", "unique funding assertion IDs",
        "receipt date and incremental tranche validation", "all added receipts public by cutoff",
        "all added receipts entirely after balance date", "no current cash claims", "no false negative certifications",
        "source capture hashes"], observations_checked=len(reconciled))
    save(out / "data_qa.json", qa)
    source_paths = [ROOT / p for p in ["fundamental/financing_ledger.py", "fundamental/financing_ledger_report.py", "scripts/build_financing_ledger.py"]]
    save(out / "manifest.json", dict(run_id="financing-ledger-" + out.name,
        scope="local research only", generated_at=result["generated_at"], input_hashes={str(p.resolve()): digest(p) for p in inputs},
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in source_paths},
        outputs={"report": {"path": str(report.resolve()), "sha256": digest(report)}},
        artifact_hashes={p.name: digest(p) for p in out.glob("*") if p.is_file() and p.suffix in {".json", ".csv", ".html"} and p.name != "manifest.json"},
        stats=stats, completion_status="RESEARCH_COMPLETE_QA_PENDING", qa={"visual_status": "PENDING"}))
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--supplemental-records", type=Path, required=True)
    args = parser.parse_args()
    build(args.pilot_dir.resolve(), args.output_dir.resolve(), args.supplemental_records.resolve())


if __name__ == "__main__":
    main()
