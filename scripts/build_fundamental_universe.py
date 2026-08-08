"""Discover and materialize the broad, research-only equity universe."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fundamental.config import (  # noqa: E402
    BROAD_UNIVERSE_POLICY,
    CURRENT_ROOT,
    POLICY_VERSION,
    RAW_ROOT,
)
from fundamental.fmp import FMPClient  # noqa: E402
from fundamental.storage import (  # noqa: E402
    archive_json,
    iso_utc,
    load_latest_snapshot_parts,
    snapshot_part_path,
    write_current_parquet,
    write_immutable_parquet,
    write_run_manifest,
)
from fundamental.universe import (  # noqa: E402
    build_broad_universe,
    normalize_screener_rows,
    summarize_universe,
)


MASTER_PRICES = ROOT / "data" / "master_prices.parquet"
OVERFLOW_PRICES = ROOT / "data" / "overflow_prices.parquet"


def _load_research_prices() -> pd.DataFrame:
    columns = ["ticker", "date", "Close", "Volume"]
    frames = []
    # Overflow comes first so the newer master cache wins duplicate bars.
    for path in (OVERFLOW_PRICES, MASTER_PRICES):
        if path.exists():
            frames.append(pd.read_parquet(path, columns=columns))
    if not frames:
        return pd.DataFrame(columns=columns)
    prices = pd.concat(frames, ignore_index=True)
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    return prices.sort_values("date").drop_duplicates(["ticker", "date"], keep="last")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build broad fundamental research universe.")
    parser.add_argument("--as-of", default=str(date.today()))
    parser.add_argument("--min-market-cap", type=float, default=BROAD_UNIVERSE_POLICY.min_market_cap)
    parser.add_argument("--min-price", type=float, default=BROAD_UNIVERSE_POLICY.min_price)
    parser.add_argument("--min-volume", type=float, default=BROAD_UNIVERSE_POLICY.min_current_volume)
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    if not MASTER_PRICES.exists():
        raise SystemExit(f"price cache missing: {MASTER_PRICES}")
    as_of = str(pd.Timestamp(args.as_of).date())
    part = snapshot_part_path("universe", as_of, "US", "company-screener")
    if part.exists():
        screener = pd.read_parquet(part)
        digest = str(screener["payload_digest"].dropna().iloc[0])
        raw_path = RAW_ROOT / "fmp" / "company-screener" / "US" / f"{digest}.json"
    else:
        client = FMPClient()
        payload = client.fetch_collection(
            "company-screener",
            marketCapMoreThan=args.min_market_cap,
            priceMoreThan=args.min_price,
            volumeMoreThan=args.min_volume,
            country="US",
            isEtf="false",
            isFund="false",
            isActivelyTrading="true",
            limit=args.limit,
        )
        raw_path, digest = archive_json(payload, "fmp", "company-screener", "US")
        screener = normalize_screener_rows(payload, as_of=as_of)
        screener["payload_digest"] = digest
        write_immutable_parquet(screener, part)

    prices = _load_research_prices()
    fmp_current = CURRENT_ROOT / "fmp_latest.parquet"
    sec_current = CURRENT_ROOT / "sec_latest.parquet"
    fmp = pd.read_parquet(fmp_current) if fmp_current.exists() else load_latest_snapshot_parts("fmp", as_of)
    sec = pd.read_parquet(sec_current) if sec_current.exists() else load_latest_snapshot_parts("sec", as_of)
    universe = build_broad_universe(
        screener,
        prices,
        as_of=as_of,
        fundamental_tickers=(fmp["ticker"].unique() if not fmp.empty else []),
        sec_tickers=(sec["ticker"].unique() if not sec.empty else []),
    )
    universe_path = write_current_parquet(universe, "broad_universe_latest.parquet")
    summary = summarize_universe(universe)
    summary.update({
        "as_of": as_of,
        "policy_version": POLICY_VERSION,
        "raw_path": str(raw_path),
        "snapshot_path": str(part),
        "live_actions_enabled": False,
    })
    summary_path = CURRENT_ROOT / "broad_universe_summary_latest.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    manifest = write_run_manifest({
        "policy_version": POLICY_VERSION,
        "run_type": "fundamental_universe_discovery",
        "snapshot_as_of": as_of,
        "summary": summary,
        "completed_at": iso_utc(),
        "live_actions_enabled": False,
    })

    print(f"Discovered: {summary['discovered']:,}")
    print(f"Research eligible: {summary['research_eligible']:,}")
    print(f"Standard queue: {summary['standard_queue']:,}; specialist queue: {summary['specialist_queue']:,}")
    print(f"Fundamental covered: {summary['fundamental_covered']:,}; SEC covered: {summary['sec_covered']:,}")
    print(f"Universe: {universe_path}")
    print(f"Manifest: {manifest}")

    if args.upload:
        from cache_io import upload_from_local

        uploads = [
            (universe_path, "fundamental/current/broad_universe_latest.parquet"),
            (summary_path, "fundamental/current/broad_universe_summary_latest.json"),
            (part, f"fundamental/snapshots/universe/as_of={as_of}/ticker=US/company-screener.parquet"),
            (manifest, f"fundamental/runs/{manifest.name}"),
        ]
        raw_candidate = Path(raw_path) if not isinstance(raw_path, Path) else raw_path
        if raw_candidate.exists():
            uploads.append((raw_candidate, f"fundamental/raw/fmp/company-screener/US/{raw_candidate.name}"))
        for local, key in uploads:
            if not upload_from_local(str(local), key):
                raise SystemExit(f"R2 upload failed: {local}")


if __name__ == "__main__":
    main()
