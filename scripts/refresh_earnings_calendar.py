"""Production earnings entry point with an opt-in Alpha primary and FMP fallback.

Default provider is the checked-in config (initially FMP). All Alpha work is
assembled under artifacts before any canonical write. --no-upload requires an
explicit artifact destination and never touches data/ or R2. Snapshot replay is
available only in that mode. Publication uses a conditional single-object write.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from earnings_calendar_provider import (CalendarError, SCOPE, build_candidate,
    combine_calendars, decision_differences, is_authoritative, normalize, validate_freshness)
from scripts import build_earnings_calendar as fmp
from scripts.compare_earnings_shadow import fetch_alpha, parse_alpha_csv, compare, normalize_fmp
from strategy_config import CSV_UNIVERSE
from trading_calendar import TRADING_DAY
from alpha_calendar_snapshot import daily_alpha, SnapshotError

KEY = "earnings_calendar.parquet"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def fetch_fmp_rows(tickers, key):
    rows, failed, empty = [], [], []
    for ticker in sorted(set(tickers)):
        # Explicit non-corporate instruments, not a broad failed-symbol exemption.
        if ticker.startswith("^") or ticker.endswith(("=F", "=X")) or ticker == "DX-Y.NYB":
            empty.append(ticker)
            continue
        payload = fmp.fetch_ticker(ticker, key)
        if payload is None:
            failed.append(ticker)
        elif not payload:
            empty.append(ticker)
        else:
            for row in payload:
                rows.append(dict(ticker=ticker, date=row.get("date"), eps_actual=row.get("epsActual"),
                                 eps_est=row.get("epsEstimated"), revenue_actual=row.get("revenueActual"),
                                 revenue_est=row.get("revenueEstimated"), last_updated=row.get("lastUpdated")))
        time.sleep(fmp.SLEEP_BETWEEN_CALLS)
    frame = normalize(pd.DataFrame(rows)) if rows else pd.DataFrame()
    if not frame.empty:
        frame["last_updated"] = pd.to_datetime(frame.last_updated, errors="coerce")
    return frame, failed, empty


def recent_tickers(prior, as_of):
    # Actuals can arrive later than a calendar date; refresh the whole Â±10-day
    # history window, not just yesterday. This is intentionally still FMP-backed.
    return sorted(set(prior.loc[prior.date.between(as_of - 10 * TRADING_DAY, as_of), "ticker"]))


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


def merge_refreshed_tickers(prior, refreshed, requested, as_of):
    """Fresh responses replace those tickers' forward rows, not their old history."""
    retained = prior.loc[~prior.ticker.isin(requested) | prior.date.lt(as_of)]
    if refreshed.empty:
        return retained.copy()
    return pd.concat([refreshed, retained], ignore_index=True).drop_duplicates(["ticker", "date"])


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
        historical &= ~prior.event_status.eq("expected")
    history = prior.loc[historical, ["ticker", "date"]]
    keys = set(zip(candidate.ticker, candidate.date))
    if any(key not in keys for key in zip(history.ticker, history.date)):
        raise CalendarError("Candidate lost an existing historical event")


def fmp_fallback(prior, universe, as_of, key):
    refreshed, failed, empty = fetch_fmp_rows(universe, key)
    if refreshed.empty:
        raise CalendarError("FMP fallback returned no data")
    # A failed equity request may not silently erase a blackout. Unknown
    # symbols with no prior record are also failures; known non-equity empty
    # responses are allowed. No stale failed rows are relabeled as fresh.
    if failed:
        raise CalendarError("FMP fallback fetch failed for: " + ", ".join(failed))
    refreshed = fmp.compute_derived_columns(refreshed)
    keep = prior.date.lt(as_of)
    if "event_status" in prior:
        keep &= ~prior.event_status.eq("expected")
    historical = prior.loc[keep]
    result = (pd.concat([refreshed, historical], ignore_index=True) if not historical.empty else refreshed.copy())
    result = result.drop_duplicates(["ticker", "date"])
    result["event_source"] = "fmp_fallback"
    result["event_status"] = "legacy_unverified"
    result.loc[result.eps_actual.notna() | result.revenue_actual.notna(), "event_status"] = "confirmed"
    expected = result.date.gt(as_of) | (result.date.eq(as_of) & result.eps_actual.isna() & result.revenue_actual.isna())
    result.loc[expected, "event_status"] = "expected"
    return result, dict(fmp_fallback_failed=failed, fmp_fallback_empty=empty)


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


def refresh_reference(output_dir, symbol_master):
    """Independent FMP control for the observer after production becomes Alpha."""
    run = Path(output_dir).resolve()
    if not run.is_relative_to((ROOT / "artifacts").resolve()):
        raise CalendarError("Reference output must stay under artifacts")
    run.mkdir(parents=True, exist_ok=False)
    symbols = pd.read_parquet(symbol_master)
    universe = set(CSV_UNIVERSE) | set(symbols.ticker.str.upper())
    frame, failed, empty = fetch_fmp_rows(universe, fmp.load_env())
    receipt = dict(provider="fmp_reference", generated_at=pd.Timestamp.now(tz="UTC").isoformat(),
                   requested=len(universe), failed=failed, empty=empty, published=False)
    if failed or frame.empty:
        write_json(run / "failure.json", receipt)
        raise CalendarError("Independent FMP reference is incomplete; no comparison baseline written")
    frame = fmp.compute_derived_columns(frame)
    frame["calendar_provider"] = "fmp_reference"
    frame.to_parquet(run / KEY, index=False)
    receipt.update(rows=len(frame), sha256=digest(run / KEY))
    write_json(run / "receipt.json", receipt)
    print(f"Independent FMP reference saved: {run}; no production writes")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", choices=("fmp", "alpha"))
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--overflow-baseline", type=Path)
    ap.add_argument("--symbol-master", type=Path)
    ap.add_argument("--alpha-snapshot", type=Path, help="Existing observer run with alpha_raw.csv and summary.json")
    ap.add_argument("--confirmations", type=Path, help="Offline actuals or SEC date-proof snapshot, selected by --confirmation-provider")
    ap.add_argument("--confirmation-provider", choices=("fmp", "sec"), default="fmp",
                    help="SEC date-only proof is available for explicit offline replay only")
    ap.add_argument("--as-of")
    ap.add_argument("--reference-only", action="store_true", help="Refresh an independent FMP observer baseline under artifacts")
    args = ap.parse_args(argv)
    if args.confirmation_provider == "sec" and not (args.no_upload and args.alpha_snapshot and args.confirmations):
        raise CalendarError("SEC confirmations require --no-upload --alpha-snapshot --confirmations")
    load_dotenv(ROOT / ".env", override=False)
    config = json.loads((ROOT / "config/earnings_calendar.json").read_text())
    provider = args.provider or config["provider"]
    if provider not in {"fmp", "alpha"} or config.get("alpha_fallback") not in {"fmp", "stop"}:
        raise CalendarError("Invalid earnings provider configuration")
    if args.reference_only:
        if not args.no_upload or not args.output_dir or any((args.baseline, args.overflow_baseline, args.alpha_snapshot, args.confirmations, args.as_of)):
            raise CalendarError("--reference-only requires --no-upload --output-dir and cannot replay other inputs")
        return refresh_reference(args.output_dir, args.symbol_master or ROOT / "data/symbol_master.parquet")
    replay_flags = (args.baseline, args.overflow_baseline, args.symbol_master, args.alpha_snapshot, args.confirmations, args.as_of)
    if not args.no_upload and (any(replay_flags) or provider != config["provider"]):
        raise CalendarError("Replay/provider overrides require --no-upload; activation is a reviewed config change")
    if provider == "fmp":
        if args.no_upload:
            raise CalendarError("Use Alpha replay or the legacy builder's explicit --no-upload interface")
        # Preserve the currently deployed FMP behavior until activation.
        fmp.build_calendar(sorted(CSV_UNIVERSE), fmp.load_env(), fmp.OUTPUT_PATH)
        return 0
    if args.no_upload and args.output_dir is None:
        raise CalendarError("--no-upload requires an explicit artifact output directory")
    now = pd.Timestamp.now(tz="UTC")
    as_of = pd.Timestamp(args.as_of).normalize() if args.as_of else now.tz_convert("America/New_York").tz_localize(None).normalize()
    run = (args.output_dir or ROOT / "artifacts/earnings_provider" / now.strftime("%Y%m%dT%H%M%S%fZ")).resolve()
    if not run.is_relative_to((ROOT / "artifacts").resolve()):
        raise CalendarError("Candidate outputs must stay under this checkout's artifacts directory")
    run.mkdir(parents=True, exist_ok=False)
    receipt = dict(producer="earnings_calendar", provider_requested=provider, as_of=str(as_of.date()),
                   generated_at=now.isoformat(), published=False)
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
        prior = normalize(combine_calendars(prior_main, normalize(pd.read_parquet(overflow))))
        prior = prior.drop_duplicates(["ticker", "date"], keep="first")
        symbols = pd.read_parquet(symbol_master)
        if "ticker" not in symbols or symbols.empty:
            raise CalendarError("Symbol master is missing coverage")
        universe = set(CSV_UNIVERSE) | set(symbols.ticker.str.upper())
        receipt.update(baseline_sha256=digest(baseline), overflow_sha256=digest(overflow), universe_count=len(universe),
                       historical_actuals_provider=args.confirmation_provider, snapshot_replay=bool(args.alpha_snapshot))
        if not args.no_upload:
            # The legacy overflow sidecar may be months stale. Bootstrap its
            # current history once, instead of promoting those stale dates as
            # the permanent history of an authoritative all-universe calendar.
            initial = not is_authoritative(prior_main)
            extra = (universe - set(CSV_UNIVERSE)) if initial else (universe - set(prior.ticker))
            if extra:
                bootstrap, failed, empty = fetch_fmp_rows(extra, fmp.load_env())
                if failed or (initial and bootstrap.empty):
                    raise CalendarError("Initial overflow history refresh failed; cutover refused")
                if not bootstrap.empty:
                    bootstrap.to_parquet(run / "bootstrap_fmp.parquet", index=False)
                # Preserve old historical evidence; current FMP takes priority
                # for the same ticker/date. Superseded future rows are removed
                # by build_candidate, not unioned into the new forward calendar.
                prior = merge_refreshed_tickers(prior, bootstrap, extra, as_of)
                receipt.update(bootstrap_requested=len(extra), bootstrap_empty=empty)
        prior.to_parquet(run / "baseline.parquet", index=False)
        alpha = None
        try:
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
            needed = recent_tickers(prior, as_of)
            if args.alpha_snapshot:
                # Historical replay is fully offline. Existing FMP history is
                # enough for a first-generation candidate; elapsed expectations
                # still require exact actuals and otherwise fail build_candidate.
                confirmations = pd.read_parquet(args.confirmations) if args.confirmations else prior
                failed, empty = [], []
            else:
                confirmations, failed, empty = fetch_fmp_rows(needed, fmp.load_env())
            if failed:
                raise CalendarError("Recent-actual confirmation failed: " + ", ".join(failed))
            receipt.update(recent_actuals_requested=needed, recent_actuals_empty=empty)
            overrides = json.loads((ROOT / "config/earnings_calendar_overrides.json").read_text())["overrides"]
            candidate, applied = build_candidate(prior, alpha, confirmations, as_of, overrides,
                                                confirmation_provider=args.confirmation_provider)
            receipt["overrides_applied"] = applied
            coverage_gate(prior, candidate, as_of)
            selected = "alpha"
        except Exception as exc:
            # Never turn a failed replay into live requests; never log raw errors
            # from HTTP libraries (they can contain an API key).
            receipt["alpha_error"] = str(exc) if isinstance(exc, (CalendarError, SnapshotError)) else type(exc).__name__
            if args.no_upload or config["alpha_fallback"] != "fmp":
                raise CalendarError("Alpha candidate failed: " + receipt["alpha_error"]) from None
            candidate, fallback_receipt = fmp_fallback(prior, universe, as_of, fmp.load_env())
            receipt.update(fallback_receipt)
            coverage_gate(prior, candidate, as_of)
            selected = "fmp_fallback"
        candidate = stamp(candidate, as_of, selected)
        # Preserve all financial/derived values in frozen legacy history. New
        # actuals get computed metrics; expected rows retain missing actuals.
        computed = fmp.compute_derived_columns(candidate)
        for col in ("eps_surprise_pct", "rev_surprise_pct", "eps_yoy", "rev_yoy"):
            if col not in candidate:
                candidate[col] = float("nan")
            lookup = computed.set_index(["ticker", "date"])[col]
            refresh = candidate.event_source.eq("fmp_actuals") if "event_source" in candidate else pd.Series(False, index=candidate.index)
            refresh &= candidate.date.ge(as_of - 10 * TRADING_DAY)
            candidate.loc[refresh, col] = [lookup.loc[(r.ticker, r.date)] for r in candidate.loc[refresh].itertuples()]
        candidate_path = run / KEY
        candidate.to_parquet(candidate_path, index=False)
        deltas = decision_differences(prior, candidate, universe, as_of)
        write_json(run / "decision_differences.json", deltas)
        comparison = None
        if alpha is not None and not alpha.empty:
            details, comparison = compare(normalize_fmp(prior), alpha, universe, as_of)
            details.to_csv(run / "provider_comparison.csv", index=False)
        receipt.update(provider_selected=selected, status="ok" if selected == "alpha" else "degraded",
                       rows=len(candidate), candidate_sha256=digest(candidate_path),
                       decision_differences=len(deltas), comparison=comparison)
        if not args.no_upload:
            validate_freshness(candidate)
            publish(candidate_path, expected_etag, ROOT / "data" / KEY, run, receipt)
            receipt["published"] = True
            write_json(ROOT / "data" / (KEY + ".status.json"), receipt)
        write_json(run / "receipt.json", receipt)
        print(json.dumps({k: receipt[k] for k in ("provider_selected", "rows", "decision_differences", "published")}, indent=2))
        return 0
    except Exception as exc:
        receipt["error"] = str(exc) if isinstance(exc, (CalendarError, SnapshotError)) else type(exc).__name__
        write_json(run / "failure.json", receipt)
        print("Earnings refresh stopped: " + receipt["error"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CalendarError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
