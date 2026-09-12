"""Collect a week's scan exceptions and operational failures without running producers.

Read local supervisor logs, archived scan/email records, and optional downloaded
GitHub logs. Corporate actions require separate primary-source verification:
an empty provider response never establishes a delisting.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scan_audit import MARKER, redact


def collect(state_root, start, end, *, coverage_paths=(), extra_logs=()):
    start, end = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    if end < start:
        raise ValueError("end date precedes start")
    findings, records, tickers, evidence = [], [], {}, []
    completed = set()
    def add_ticker(ticker, reason, source):
        ticker = str(ticker).upper()
        if re.fullmatch(r"[A-Z0-9^][A-Z0-9.^=-]{0,18}", ticker) and ticker not in {"INVENTORY", "EXPOSURE", "OLV-EXIT"}:
            tickers.setdefault(ticker, []).append({"reason": redact(reason), "source": source})
    def add_record(record, source):
        date = str(record.get("date_et") or record.get("generated_at", ""))[:10]
        if not start.isoformat() <= date <= end.isoformat():
            return
        records.append({"source": source, **record})
        coverage = record.get("coverage", record)
        for ticker in coverage.get("unavailable", []):
            add_ticker(ticker, "price input unavailable", source)
        for ticker, last in coverage.get("stale", {}).items():
            add_ticker(ticker, f"stale; last bar {last}", source)
        for error in coverage.get("exceptions", []):
            add_ticker(error.get("ticker", ""), error.get("reason", ""), source)
        if record.get("email_accepted") is False:
            findings.append({"source": source, "message": "scan summary email not accepted"})
    state_root = Path(state_root)
    logs = []
    for folder in (state_root / "logs").glob("????-??-??"):
        if start.isoformat() <= folder.name <= end.isoformat():
            logs.extend(folder.glob("*.log"))
    logs.extend(Path(p) for p in extra_logs)
    for path in sorted(set(logs)):
        evidence.append(str(path))
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            source = f"{path}:{line_no}"
            if MARKER in line:
                try:
                    add_record(json.loads(line.split(MARKER, 1)[1]), source)
                except (ValueError, TypeError):
                    findings.append({"source": source, "message": "unreadable scan audit record"})
                continue
            match = re.search(r"success (scan_am|scan_pm) \(", line)
            if match and re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.parent.name):
                completed.add((path.parent.name, match[1]))
            for ticker, last in re.findall(r"([A-Z0-9^][A-Z0-9.^=-]*)@(\d{4}-\d{2}-\d{2})", line):
                add_ticker(ticker, f"stale; last bar {last}", source)
            match = re.search(r"Quote not found for symbol: ([A-Z0-9.^=-]+)", line)
            if match:
                add_ticker(match[1], "provider quote missing; not proof of delisting", source)
            if any(noise in line for noise in ("ScriptRunContext", "MemoryCacheStorageManager", "can be ignored when running in bare mode")):
                continue
            if re.search(r"(?i)(\[FAIL\]|\[WARN(?:ING)?\]|\[ERROR\]|^ERROR:|^WARNING:|^RuntimeError:|^ValueError:|Traceback|optional.*unavailable|OLV-EXIT.*WARNING|Archive failed)", line):
                findings.append({"source": source, "message": redact(line.strip())[:1500]})
    for folder in (state_root / "scan_audits").glob("????-??-??"):
        if start.isoformat() <= folder.name <= end.isoformat():
            for path in sorted(folder.glob("*.json")):
                evidence.append(str(path))
                try:
                    add_record(json.loads(path.read_text(encoding="utf-8")), str(path))
                except (ValueError, TypeError):
                    findings.append({"source": str(path), "message": "unreadable scan archive"})
    for value in coverage_paths:
        path = Path(value)
        evidence.append(str(path))
        add_record(json.loads(path.read_text(encoding="utf-8")), str(path))
    # A missing local success is an evidence gap, not proof the cloud fallback failed.
    from trading_calendar import TRADING_DAY
    import pandas as pd
    expected = [(day.date().isoformat(), job) for day in pd.date_range(start, end, freq=TRADING_DAY)
                for job in ("scan_am", "scan_pm")]
    unique = {(item["source"], item["message"]): item for item in findings}
    return {"schema_version": 1, "start": str(start), "end": str(end),
            "evidence_files": evidence, "local_scan_successes": sorted(completed),
            "scan_success_evidence_gaps": [list(pair) for pair in expected if pair not in completed],
            "ticker_candidates": {ticker: items for ticker, items in sorted(tickers.items())},
            "operational_findings": list(unique.values()), "scan_records": records,
            "limitations": ["Historical email exception lists are incomplete when only truncated legacy logs survive; use saved coverage or email evidence.",
                            "A scan success receipt does not certify each optional overlay, SMTP delivery to inbox, or a broker fill.",
                            "Ticker candidates need issuer/exchange/SEC verification; provider errors alone never establish delisting."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--coverage", type=Path, action="append", default=[])
    parser.add_argument("--extra-log", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.state_root, args.start, args.end, coverage_paths=args.coverage, extra_logs=args.extra_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "tickers": list(report["ticker_candidates"]),
                      "operational_findings": len(report["operational_findings"]),
                      "scan_success_evidence_gaps": report["scan_success_evidence_gaps"]}))


if __name__ == "__main__":
    main()
