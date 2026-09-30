"""Production earnings entry point: Alpha Vantage forward dates, SEC confirmations.

FMP is retired. Existing history in the canonical calendar is frozen; each run
only replaces forward Alpha expectations and confirms newly elapsed events from
SEC 8-K Item 2.02 filings (dates only). Events that no source can confirm are
kept conservatively (see ``build_candidate(unconfirmed="retain")``) so a
blackout is never removed without evidence. If Alpha itself fails, nothing is
published: the prior canonical calendar stays in place and consumers tolerate
``MAX_STALE_TRADING_DAYS`` of staleness.

All work is assembled under artifacts before any canonical write. --no-upload
requires an explicit artifact destination and never touches data/ or R2.
Snapshot replay is available only in that mode. Publication uses a conditional
single-object write followed by a verified readback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from earnings_calendar_provider import (CalendarError, SCOPE, UNVERIFIED_ELAPSED, build_candidate,
    combine_calendars, decision_differences, is_authoritative, normalize, validate_freshness)
from scripts.compare_earnings_shadow import fetch_alpha, parse_alpha_csv, compare, normalize_fmp
from sec_earnings_dates import collect_sec_confirmations, confirmation_targets
from strategy_config import CSV_UNIVERSE
from trading_calendar import TRADING_DAY
from alpha_calendar_snapshot import daily_alpha, SnapshotError

KEY = "earnings_calendar.parquet"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def align_alpha_symbols(alpha, universe):
    """Map dot/dash share-class aliases only when the target is unambiguous."""
    aliases = {}
    for ticker in universe:
        alias = ticker.replace("-", ".")
        if alias in aliases and aliases[alias] != ticker:
            raise CalendarError("Ambiguous share-class ticker aliases in universe")
        aliases[alias] = ticker
    alpha = alpha.copy()
    alpha["ticker"] = alpha.ticker.map(lambda t: t if t in universe else aliases.get(t, t))
    return alpha.loc[alpha.ticker.isin(universe)].copy()


def stamp(frame, as_of, provider):
    frame = frame.copy()
    frame["calendar_scope"] = SCOPE
    frame["calendar_as_of"] = str(as_of.date())
    frame["calendar_generated_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    frame["calendar_provider"] = provider
    return frame


def coverage_gate(prior, candidate, as_of):
    """Historical row counts cannot mask a truncated forward response."""
    horizon = as_of + 10 * TRADING_DAY
    previous = set(prior.loc[prior.date.between(as_of, horizon), "ticker"])
    current = set(candidate.loc[candidate.date.between(as_of, horizon), "ticker"])
    if previous and len(current) < len(previous) * .80:
        raise CalendarError("Near-term earnings coverage fell by more than 20%; refusing publication")
    historical = prior.date.lt(as_of)
    if "event_status" in prior:
        # Unverified rows may legitimately move to their SEC-confirmed date.
        historical &= ~prior.event_status.isin(["expected", UNVERIFIED_ELAPSED])
    history = prior.loc[historical, ["ticker", "date"]]
    keys = set(zip(candidate.ticker, candidate.date))
    if any(key not in keys for key in zip(history.ticker, history.date)):
        raise CalendarError("Candidate lost an existing historical event")


def publish(candidate_path, expected_etag, local_path, run, receipt=None):
    """Do not catch publication failures as fetch failures or retry a new provider."""
    from cache_io import conditional_upload_from_local, download_to_local
    receipt = receipt if receipt is not None else {}
    receipt["publication_state"] = "attempting_conditional_write"
    write_json(run / "publication.json", receipt)
    state, new_etag = conditional_upload_from_local(str(candidate_path), KEY, expected_etag=expected_etag)
    receipt["publication_state"] = "baseline_conflict" if state == "precondition_failed" else "write_outcome_unknown"
    if state == "uploaded":
        receipt.update(publication_state="remote_written_unverified", published_etag=new_etag)
    write_json(run / "publication.json", receipt)
    if state != "uploaded":
        raise CalendarError("Canonical earnings publication failed or another writer changed the baseline")
    checked = run / "published_readback.parquet"
    if not download_to_local(KEY, str(checked)) or digest(checked) != digest(candidate_path):
        raise CalendarError("Canonical earnings readback failed; publication outcome needs investigation")
    receipt["publication_state"] = "remote_verified"
    write_json(run / "publication.json", receipt)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    # Local replacement happens only after successful cloud readback.
    pending = local_path.with_name(local_path.name + ".alpha-pending")
    pending.write_bytes(candidate_path.read_bytes())
    os.replace(pending, local_path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--overflow-baseline", type=Path)
    ap.add_argument("--symbol-master", type=Path)
    ap.add_argument("--alpha-snapshot", type=Path, help="Existing observer run with alpha_raw.csv and summary.json")
    ap.add_argument("--confirmations", type=Path, help="Offline SEC date-proof snapshot for replay")
    ap.add_argument("--as-of")
    args = ap.parse_args(argv)
    load_dotenv(ROOT / ".env", override=False)
    config = json.loads((ROOT / "config/earnings_calendar.json").read_text())
    if config.get("provider") != "alpha" or config.get("confirmation_provider") != "sec":
        raise CalendarError("Invalid earnings provider configuration")
    replay_flags = (args.baseline, args.overflow_baseline, args.symbol_master, args.alpha_snapshot, args.confirmations, args.as_of)
    if not args.no_upload and any(replay_flags):
        raise CalendarError("Replay inputs require --no-upload")
    if args.no_upload and args.output_dir is None:
        raise CalendarError("--no-upload requires an explicit artifact output directory")
    now = pd.Timestamp.now(tz="UTC")
    as_of = pd.Timestamp(args.as_of).normalize() if args.as_of else now.tz_convert("America/New_York").tz_localize(None).normalize()
    run = (args.output_dir or ROOT / "artifacts/earnings_provider" / now.strftime("%Y%m%dT%H%M%S%fZ")).resolve()
    if not run.is_relative_to((ROOT / "artifacts").resolve()):
        raise CalendarError("Candidate outputs must stay under this checkout's artifacts directory")
    run.mkdir(parents=True, exist_ok=False)
    receipt = dict(producer="earnings_calendar", provider_requested="alpha", confirmation_provider="sec",
                   as_of=str(as_of.date()), generated_at=now.isoformat(), published=False)
    try:
        expected_etag = None
        if args.no_upload:
            baseline = args.baseline or ROOT / "data" / KEY
            overflow = args.overflow_baseline or ROOT / "data/earnings_calendar_overflow.parquet"
            symbol_master = args.symbol_master or ROOT / "data/symbol_master.parquet"
        else:
            from cache_io import download_to_local, head
            before = head(KEY)
            expected_etag = (before or {}).get("ETag")
            if not expected_etag:
                raise CalendarError("Canonical earnings baseline has no ETag")
            baseline, overflow, symbol_master = [run / name for name in ("prior.parquet", "overflow.parquet", "symbol_master.parquet")]
            for key, path in ((KEY, baseline), ("earnings_calendar_overflow.parquet", overflow), ("symbol_master.parquet", symbol_master)):
                if not download_to_local(key, str(path)):
                    raise CalendarError("Required canonical earnings input could not be downloaded: " + key)
            if (head(KEY) or {}).get("ETag") != expected_etag:
                raise CalendarError("Canonical earnings baseline changed during download")
        prior_main = normalize(pd.read_parquet(baseline))
        if not is_authoritative(prior_main) and not args.no_upload:
            # The legacy FMP bootstrap that migrated old calendars is retired.
            raise CalendarError("Canonical earnings calendar is not an all-universe calendar; cannot bootstrap without FMP")
        overflow_frame = normalize(pd.read_parquet(overflow)) if Path(overflow).exists() else None
        prior = normalize(combine_calendars(prior_main, overflow_frame))
        prior = prior.drop_duplicates(["ticker", "date"], keep="first")
        symbols = pd.read_parquet(symbol_master)
        if "ticker" not in symbols or symbols.empty:
            raise CalendarError("Symbol master is missing coverage")
        # The universe is frozen: names without history just get forward dates.
        universe = set(CSV_UNIVERSE) | set(symbols.ticker.str.upper())
        receipt.update(baseline_sha256=digest(baseline), universe_count=len(universe),
                       snapshot_replay=bool(args.alpha_snapshot))
        prior.to_parquet(run / "baseline.parquet", index=False)
        if args.alpha_snapshot:
            meta = json.loads((args.alpha_snapshot / "summary.json").read_text())
            captured = pd.Timestamp(meta["captured_at_utc"])
            if meta.get("mode") != "authenticated" or captured.tz_convert("America/New_York").date() != as_of.date():
                raise CalendarError("Alpha replay must be an authenticated snapshot from the requested date")
            raw = (args.alpha_snapshot / "alpha_raw.csv").read_text(encoding="utf-8")
            alpha = parse_alpha_csv(raw)
        else:
            key = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()
            if not key:
                raise CalendarError("ALPHA_VANTAGE_API_KEY is missing")
            config_root = Path(os.environ.get("NEW_SEASONALS_AUTOMATION_STATE_ROOT", str(ROOT / "artifacts/automation"))).resolve().parents[1]
            raw, alpha, snapshot_meta = daily_alpha(config_root=config_root, fetch=fetch_alpha,
                                                    parse=parse_alpha_csv, key=key)
            receipt["alpha_snapshot"] = snapshot_meta
        (run / "alpha_raw.csv").write_text(raw, encoding="utf-8")
        alpha = align_alpha_symbols(alpha, universe)
        targets = confirmation_targets(prior, as_of)
        if args.confirmations:
            confirmations = pd.read_parquet(args.confirmations)
            sec_summary = dict(requested=len(targets), confirmed=len(confirmations), offline=True)
        elif args.alpha_snapshot:
            # Historical replay stays fully offline.
            confirmations, sec_summary = pd.DataFrame(), dict(requested=len(targets), confirmed=0, offline=True)
        else:
            confirmations, sec_summary = collect_sec_confirmations(targets, as_of)
        if not confirmations.empty:
            confirmations.to_parquet(run / "sec_confirmations.parquet", index=False)
        receipt["sec_confirmations"] = sec_summary
        overrides = json.loads((ROOT / "config/earnings_calendar_overrides.json").read_text())["overrides"]
        report = {}
        candidate, applied = build_candidate(prior, alpha, confirmations, as_of, overrides,
                                             confirmation_provider="sec", unconfirmed="retain", report=report)
        receipt["overrides_applied"] = applied
        receipt["unconfirmed"] = report
        coverage_gate(prior, candidate, as_of)
        candidate = stamp(candidate, as_of, "alpha")
        candidate_path = run / KEY
        candidate.to_parquet(candidate_path, index=False)
        deltas = decision_differences(prior, candidate, universe, as_of)
        write_json(run / "decision_differences.json", deltas)
        comparison = None
        if not alpha.empty:
            details, comparison = compare(normalize_fmp(prior), alpha, universe, as_of)
            details.to_csv(run / "provider_comparison.csv", index=False)
        unresolved = sum(len(v) for v in report.values())
        receipt.update(provider_selected="alpha", status="ok" if not unresolved else "ok_with_unverified",
                       unverified_elapsed_total=int(candidate.event_status.eq(UNVERIFIED_ELAPSED).sum()),
                       rows=len(candidate), candidate_sha256=digest(candidate_path),
                       decision_differences=len(deltas), comparison=comparison)
        if not args.no_upload:
            validate_freshness(candidate)
            publish(candidate_path, expected_etag, ROOT / "data" / KEY, run, receipt)
            receipt["published"] = True
            write_json(ROOT / "data" / (KEY + ".status.json"), receipt)
        write_json(run / "receipt.json", receipt)
        print(json.dumps({k: receipt[k] for k in ("provider_selected", "status", "rows", "decision_differences", "published")}, indent=2))
        return 0
    except Exception as exc:
        # Never log raw errors from HTTP libraries (they can contain an API key).
        receipt["error"] = str(exc) if isinstance(exc, (CalendarError, SnapshotError)) else type(exc).__name__
        write_json(run / "failure.json", receipt)
        print("Earnings refresh stopped; canonical calendar unchanged: " + receipt["error"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CalendarError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
