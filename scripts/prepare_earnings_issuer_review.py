"""Prepare an artifact-only issuer-site review queue; never publish calendar dates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_earnings_shadow import parse_alpha_csv
from scripts.refresh_earnings_calendar import align_alpha_symbols
from strategy_config import CSV_UNIVERSE, LIQUID_PLUS_COMMODITIES


def review_queue(alpha, universe, as_of, fmp=None):
    """Include a ticker if either provider puts an event 8..15 calendar days away."""
    today = pd.Timestamp(as_of).normalize()
    aligned = align_alpha_symbols(alpha, universe)
    candidates = []
    for provider, frame in (("alpha", aligned), ("fmp", fmp)):
        if frame is None:
            continue
        selected = frame[frame.ticker.isin(universe)].copy()
        selected["date"] = pd.to_datetime(selected.date).dt.normalize()
        selected = selected[selected.date.between(today + pd.Timedelta(days=8),
                                                  today + pd.Timedelta(days=15))]
        for row in selected.to_dict("records"):
            fiscal = row.get("fiscalDateEnding")
            candidates.append(dict(ticker=row["ticker"], provider=provider,
                expected_date=str(row["date"].date()),
                fiscal_period=str(fiscal) if pd.notna(fiscal) else None,
                calendar_days_until=int((row["date"] - today).days)))
    queue = []
    for ticker in sorted({r["ticker"] for r in candidates}):
        queue.append(dict(ticker=ticker, status="needs_issuer_review",
            provider_events=[r for r in candidates if r["ticker"] == ticker],
            instruction="Open issuer IR calendar/announcement. Record report date, fiscal period, publication time separately from call time, source URL and capture time. No announcement means unverified, not false."))
    return queue


def snapshot_counts(alpha, universe, as_of):
    today = pd.Timestamp(as_of).normalize()
    rows = alpha[alpha.date.between(today, today + pd.Timedelta(days=14))]
    tracked = align_alpha_symbols(rows, universe)
    return dict(window_start=str(today.date()), window_end=str((today + pd.Timedelta(days=14)).date()),
        basis="calendar days, including today and day 14; unique symbols",
        all_alpha=int(rows.ticker.nunique()), tracked=int(tracked.ticker.nunique()),
        regular=int(tracked[tracked.ticker.isin(CSV_UNIVERSE)].ticker.nunique()),
        liquid=int(tracked[tracked.ticker.isin(LIQUID_PLUS_COMMODITIES)].ticker.nunique()))


def verified_raw(payload, digest):
    # Old Windows observers translated CRLF to CRCRLF. Accept that legacy
    # representation only if reversing it exactly recovers the recorded hash.
    for candidate in (payload, payload.replace(b"\r\r\n", b"\r\n")):
        if hashlib.sha256(candidate).hexdigest() == digest:
            return candidate.decode("utf-8")
    raise ValueError("Alpha snapshot digest mismatch")


def r2_alpha_snapshot(today, store=None):
    """Read today's shared Alpha snapshot from R2. Read-only: never claims the
    day or calls the provider, which daily_alpha would on a missing key."""
    from alpha_calendar_snapshot import R2SnapshotStore, SnapshotError, validate_snapshot
    key = f"provider_snapshots/alpha_earnings/{today}.json"
    value, _ = (store or R2SnapshotStore()).read(key)
    if value is None:
        raise SnapshotError(f"No shared Alpha snapshot at {key}")
    raw, _ = validate_snapshot(value, str(today), parse_alpha_csv)
    return raw, key


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot-dir", type=Path)
    source.add_argument("--alpha-r2", action="store_true",
                        help="Use today's shared R2 Alpha snapshot; no observer or FMP baseline needed")
    ap.add_argument("--symbol-master", type=Path, required=True)
    ap.add_argument("--fmp-reference", type=Path)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    out = args.output_dir.resolve()
    if not out.is_relative_to((ROOT / "artifacts").resolve()):
        ap.error("Output must be a new directory under repository artifacts")
    today = pd.Timestamp.now(tz="America/New_York").date()
    if args.alpha_r2:
        from alpha_calendar_snapshot import SnapshotError
        try:
            raw, snapshot_source = r2_alpha_snapshot(today)
        except SnapshotError as exc:
            ap.error(str(exc))
    else:
        meta = json.loads((args.snapshot_dir / "summary.json").read_text())
        if meta.get("mode") != "authenticated":
            ap.error("Issuer review requires an authenticated snapshot")
        stamp = pd.Timestamp(meta["captured_at_utc"])
        if stamp.tz_convert("America/New_York").date() != today:
            ap.error("Snapshot must be from today's New York date; do not treat stale dates as current")
        raw = verified_raw((args.snapshot_dir / "alpha_raw.csv").read_bytes(),
                           meta.get("alpha_snapshot", {}).get("sha256"))
        snapshot_source = str(args.snapshot_dir.resolve())
    digest = hashlib.sha256(raw.encode()).hexdigest()
    alpha = parse_alpha_csv(raw)
    universe = set(CSV_UNIVERSE) | set(pd.read_parquet(args.symbol_master).ticker.str.upper())
    fmp = None
    if args.fmp_reference:
        receipt = json.loads((args.fmp_reference / "receipt.json").read_text())
        path = args.fmp_reference / "earnings_calendar.parquet"
        if (receipt.get("provider") != "fmp_reference" or receipt.get("failed") != []
                or receipt.get("requested") != len(universe)
                or receipt.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
                or pd.Timestamp(receipt["generated_at"]).tz_convert("America/New_York").date() != today):
            ap.error("Independent same-day FMP reference did not validate")
        fmp = pd.read_parquet(path)
    queue = review_queue(alpha, universe, today, fmp)
    result = dict(as_of=str(today), snapshot_dir=snapshot_source,
        snapshot_sha256=digest, counts=snapshot_counts(alpha, universe, today),
        review_window_calendar_days=[8, 15], companies=len(queue), queue=queue,
        verification_status="pending issuer-site research; a queue is not verification",
        canonical_writes=False)
    out.mkdir(parents=True, exist_ok=False)
    (out / "queue.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(output=str(out / "queue.json"), companies=len(queue), counts=result["counts"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
