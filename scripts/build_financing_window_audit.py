"""Build the frozen 24-window offering audit offline from reviewed evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fundamental.financing_window_audit import audit_outcome, validate_window
from fundamental.financing_window_report import render


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def identity(row):
    return f"{int(row['cik'])}-{row['session']}"


def build(history, output):
    protocol = read(output / "audit_protocol.json")
    for path, expected in protocol["source_input_hashes"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"Frozen historical input changed: {path}")
    windows = read(output / "windows.json")
    reviews = read(output / "reviewed_windows.json")
    documents = read(output / "captured_documents.json")
    events = read(output / "audited_events.json")["events"]
    analysis = read(history / "analysis.json")
    observations = {identity(r): r for r in read(history / "observations.json")}
    targets = {identity(r) for r in analysis["signals"]}
    controls = {f"{r['control_cik']}-{r['session']}" for group in analysis["matches"].values() for r in group}
    ids = [w["window_id"] for w in windows]
    if len(ids) != len(set(ids)) or set(ids) != targets | controls or set(reviews) != set(ids):
        raise ValueError("Reviewed window membership differs from the frozen signals and comparisons")
    if (len(targets), len(controls), len(ids)) != (20, 4, 24):
        raise ValueError("This report renderer is scoped to the original 24-window pilot")
    if len({e["event_id"] for e in events}) != len(events):
        raise ValueError("Duplicate audited event identity")
    captures = list(documents)
    ir_files = sorted(output.glob("ir_capture*.json"))
    for path in ir_files:
        captures.extend(r for r in read(path) if r.get("status") == "captured")
    if any(d.get("error") for d in documents):
        raise ValueError("SEC capture errors must be resolved or explicitly reviewed before building")
    for d in captures:
        if digest(d["raw_path"]) != d["sha256"]:
            raise ValueError(f"Capture digest mismatch: {d['url']}")
    captured_ir = {r["url"] for r in captures if r.get("status") == "captured"}
    audited = []
    for w in windows:
        validate_window(w)
        original = observations[w["window_id"]]
        for field in ("as_of", "balance_date", "group", "session", "ticker"):
            if w[field] != original[field]:
                raise ValueError(f"Frozen observation changed: {w['window_id']} {field}")
        review = reviews[w["window_id"]]
        if review.get("ir_inventory_reviewed") and not set(review["ir_sources"]).issubset(captured_ir):
            raise ValueError(f"Unarchived complete IR review: {w['window_id']}")
        audit = audit_outcome(w, review, documents, events)
        audited.append(dict(w, role="target" if w["window_id"] in targets else "control",
            return_60=original["return_60"], distance_high_252=original["distance_high_252"],
            above_sma50=original["above_sma50"], setups=original["setups"],
            original_runway_operating=original["runway_operating"], original_runway_capex=original["runway_capex"],
            review=review, audit=audit))
    stats = dict(status_counts=dict(Counter(r["audit"]["outcome_status"] for r in audited)),
        status_by_role={role:dict(Counter(r["audit"]["outcome_status"] for r in audited if r["role"]==role)) for role in ("target","control")},
        captured_documents=len(documents), captured_ir_pages=len(captured_ir),
        unique_primary_events=sorted({e for r in audited for e in r["audit"]["primary_events"]}),
        unique_mixed_events=sorted({e for r in audited for e in r["audit"]["mixed_events"]}),
        incomplete_negative_coverage=sum(not r["audit"]["no_event_review_complete"] for r in audited),
        fresh_rebounds_without_sustained_strength=[r["window_id"] for r in audited if r["role"]=="target" and r["return_60"]<0 and not r["above_sma50"]])
    data = dict(generated_at=datetime.now(timezone.utc).isoformat(), stats=stats, windows=audited,
        conclusion="Screen repair required before larger preregistered test; predictive value and trade profitability remain unestablished.",
        bounded_review="Negative labels cover the selected SEC forms, relevant linked exhibits, financial notes and independently enumerated official release archives; they are not proof of universal absence.",
        no_current_cash_estimates=True, original_labels_unchanged=True)
    write(output / "window_analysis.json", data)
    fields = ["window_id","ticker","session","role","outcome_end","outcome_status","outcome","inclusive_cash_equity_outcome","strict_standalone_outcome","gaps","primary_events","mixed_events","funding_note","signal_quality_flags"]
    with (output / "audited_windows.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in audited:
            row = dict(r, **{k:v for k,v in r["audit"].items() if k!="window_id"},funding_note=r["review"]["funding_note"],signal_quality_flags=r["review"]["signal_quality_flags"])
            writer.writerow({k:"; ".join(row[k]) if isinstance(row[k],list) else row[k] for k in fields})
    (output / "financing_window_audit.html").write_text(render(data),encoding="utf-8")
    qa = dict(status="PASS",checks=["Original source digests unchanged","Exact 20 signal and 4 comparison membership","60-calendar-day horizons and exact signal cutoffs preserved","SEC and captured IR raw-source hashes verified","Complete IR review requires archived inventory","Missing exhibits and review gaps block negative labels"],
        raw_captures_verified=len(captures), stats=stats)
    write(output / "data_qa.json", qa)
    inputs = [history/"analysis.json",history/"observations.json",history/"event_reviews.json",output/"windows.json",output/"audit_protocol.json",output/"reviewed_windows.json",output/"audited_events.json",output/"captured_documents.json",output/"document_inventory.json",output/"document_inventory_additions.json"]+ir_files
    sources = [ROOT/p for p in ["fundamental/financing_window_audit.py","fundamental/financing_window_report.py","scripts/build_financing_window_audit.py","tests/test_financing_window_audit.py"]]
    outputs = ["window_analysis.json","audited_windows.csv","financing_window_audit.html","data_qa.json"]
    manifest = dict(run_id=output.name, generated_at=data["generated_at"],completion_status="RENDERED_PENDING_QA",mode="historical_research_only",
        input_hashes={str(p.resolve()):digest(p) for p in inputs},source_hashes={str(p.resolve()):digest(p) for p in sources},artifact_hashes={p:digest(output/p) for p in outputs},
        known_limitations=["Three unresolved IR chronologies remain unknown","Two Sarepta windows share one mixed debt/equity event; strict standalone outcomes unknown","Funding receipt completeness and current cash not asserted","Tiny, poorly matched pilot; no incidence comparison or P&L claim"])
    write(output / "manifest.json",manifest)
    return stats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-dir",type=Path,default=ROOT/"artifacts/cash_runway/history_20260923_v1")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"artifacts/cash_runway/window_audit_20260924_v1")
    args=parser.parse_args()
    print(json.dumps(build(args.history_dir.resolve(),args.output_dir.resolve()),indent=2))


if __name__=="__main__":
    main()
