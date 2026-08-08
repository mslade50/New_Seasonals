"""Build immutable FMP and optional SEC snapshots for the fundamental sleeve.

Phase-one defaults are deliberately bounded and non-trading:

    python scripts/update_fundamentals.py --balanced-batch 75
    python scripts/update_fundamentals.py --tickers AAPL MSFT --bundle-depth deep --with-sec
    python scripts/update_fundamentals.py --all --bundle-depth screen --upload

SEC access requires ``FUNDAMENTAL_SEC_USER_AGENT`` with a real contact string.
The script never creates broker commands or portfolio actions.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fundamental.config import (  # noqa: E402
    BROAD_UNIVERSE_POLICY,
    CURRENT_ROOT,
    POLICY_VERSION,
)
from fundamental.fmp import FMPClient, fetch_ticker_bundle  # noqa: E402
from fundamental.sec import SECClient, fetch_companyfacts_snapshot  # noqa: E402
from fundamental.storage import (  # noqa: E402
    iso_utc,
    load_latest_snapshot_parts,
    snapshot_coverage,
    write_current_parquet,
    write_run_manifest,
)
from fundamental.universe import select_balanced_enrichment_batch  # noqa: E402


SYMBOL_MASTER = ROOT / "data" / "symbol_master.parquet"
BROAD_UNIVERSE = CURRENT_ROOT / "broad_universe_latest.parquet"
FMP_CURRENT = CURRENT_ROOT / "fmp_latest.parquet"
SEC_CURRENT = CURRENT_ROOT / "sec_latest.parquet"
SCREEN_ENDPOINTS = (
    "profile",
    "income-statement",
    "balance-sheet-statement",
    "cash-flow-statement",
)


def _recently_covered_tickers(
    snapshot_day: str,
    endpoints: tuple[str, ...],
    refresh_after_days: int,
) -> set[str]:
    cutoff = str((pd.Timestamp(snapshot_day) - timedelta(days=refresh_after_days)).date())
    coverage = snapshot_coverage("fmp", snapshot_day)
    current_coverage = pd.DataFrame(columns=["ticker", "dataset", "snapshot_as_of"])
    if FMP_CURRENT.exists():
        current = pd.read_parquet(FMP_CURRENT)
        if {"ticker", "endpoint", "snapshot_as_of"}.issubset(current.columns):
            current_coverage = current[["ticker", "endpoint", "snapshot_as_of"]].rename(
                columns={"endpoint": "dataset"}
            )
    coverage = pd.concat([coverage, current_coverage], ignore_index=True).drop_duplicates()
    coverage = coverage[
        coverage["dataset"].isin(endpoints)
        & coverage["snapshot_as_of"].ge(cutoff)
    ]
    counts = coverage.groupby("ticker")["dataset"].nunique()
    return set(counts[counts >= len(endpoints)].index.astype(str))


def _merge_current_base(
    base_path: Path,
    fresh: pd.DataFrame,
    *,
    replace_keys: tuple[str, ...],
) -> pd.DataFrame:
    """Merge new immutable parts into a disposable cumulative current view."""
    if not base_path.exists():
        return fresh.copy()
    base = pd.read_parquet(base_path)
    if fresh.empty:
        return base
    usable = [column for column in replace_keys if column in base.columns and column in fresh.columns]
    if not usable:
        return pd.concat([base, fresh], ignore_index=True, sort=False)
    fresh_keys = fresh[usable].drop_duplicates()
    marked = base.merge(fresh_keys.assign(_replace=True), on=usable, how="left")
    base_keep = marked[marked["_replace"].isna()].drop(columns="_replace")
    return pd.concat([base_keep, fresh], ignore_index=True, sort=False)


def choose_tickers(args) -> list[str]:
    if args.tickers:
        return sorted({str(t).upper().replace(".", "-") for t in args.tickers})
    if args.balanced_batch is not None:
        if not BROAD_UNIVERSE.exists():
            raise SystemExit(
                "broad universe missing; run scripts/build_fundamental_universe.py first"
            )
        if args.balanced_batch > BROAD_UNIVERSE_POLICY.max_enrichment_batch:
            raise SystemExit(
                f"balanced batch exceeds hard limit {BROAD_UNIVERSE_POLICY.max_enrichment_batch}"
            )
        universe = pd.read_parquet(BROAD_UNIVERSE)
        endpoints = SCREEN_ENDPOINTS if args.bundle_depth == "screen" else tuple(args.endpoints)
        recent = _recently_covered_tickers(
            args.as_of, endpoints, args.refresh_after_days
        )
        return select_balanced_enrichment_batch(
            universe,
            args.balanced_batch,
            exclude_tickers=recent,
            include_specialists=args.include_specialists,
        )
    if not SYMBOL_MASTER.exists():
        raise SystemExit(f"symbol master missing: {SYMBOL_MASTER}")
    symbols = pd.read_parquet(SYMBOL_MASTER)
    symbols["ticker"] = symbols["ticker"].astype(str).str.upper()
    symbols["market_cap"] = pd.to_numeric(symbols.get("market_cap"), errors="coerce")
    symbols = symbols.sort_values("market_cap", ascending=False).drop_duplicates("ticker")
    if args.all:
        return symbols["ticker"].tolist()
    return symbols.head(args.top_market_cap)["ticker"].tolist()


def maybe_upload(paths: list[Path], prefix: str = "fundamental") -> None:
    from cache_io import upload_from_local

    for path in paths:
        if not path.exists():
            continue
        try:
            rel = path.relative_to(ROOT / "data" / "fundamental").as_posix()
        except ValueError:
            try:
                rel = path.relative_to(ROOT).as_posix()
            except ValueError:
                rel = path.name
        if not upload_from_local(str(path), f"{prefix}/{rel}"):
            raise RuntimeError(f"R2 upload failed: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot fundamental data without trading.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tickers", nargs="+", help="Explicit ticker list.")
    group.add_argument("--all", action="store_true", help="Full symbol-master universe.")
    group.add_argument(
        "--balanced-batch", type=int, default=None,
        help="Sector/size-balanced batch from the broad research universe.",
    )
    parser.add_argument("--top-market-cap", type=int, default=25,
                        help="Default bounded universe when --tickers/--all is omitted.")
    parser.add_argument("--as-of", default=str(date.today()), help="Snapshot date (YYYY-MM-DD).")
    parser.add_argument(
        "--bundle-depth", choices=("screen", "deep"), default="screen",
        help="Screen pulls profile + statements; deep adds metrics, ratios, and estimates.",
    )
    parser.add_argument(
        "--refresh-after-days", type=int, default=BROAD_UNIVERSE_POLICY.refresh_after_days,
        help="Balanced batches skip fully covered tickers newer than this age.",
    )
    parser.add_argument(
        "--include-specialists", action="store_true",
        help="Allow financials, real estate, biotech, and special situations into the batch.",
    )
    parser.add_argument("--with-sec", action="store_true",
                        help="Also archive SEC submissions + companyfacts and acceptance times.")
    parser.add_argument(
        "--sec-limit", type=int, default=None,
        help="Maximum selected tickers to SEC-enrich in this run; default is all.",
    )
    parser.add_argument("--upload", action="store_true",
                        help="Upload materialized current views and run manifest to R2.")
    args = parser.parse_args()
    args.endpoints = (
        "profile", "income-statement", "balance-sheet-statement",
        "cash-flow-statement", "key-metrics", "ratios", "analyst-estimates",
    )

    snapshot_day = str(pd.Timestamp(args.as_of).date())
    tickers = choose_tickers(args)
    if not tickers:
        print("No tickers require refresh; the current coverage is within the freshness window.")
        return
    print(f"Fundamental snapshot {snapshot_day}: {len(tickers)} tickers")
    print("Research-only: no portfolio or broker actions are enabled.")

    fmp = FMPClient()
    sec = SECClient() if args.with_sec else None
    sec_map = sec.ticker_map() if sec else {}
    fmp_records: list[dict] = []
    sec_records: list[dict] = []
    failures: list[dict] = []
    endpoints = SCREEN_ENDPOINTS if args.bundle_depth == "screen" else tuple(args.endpoints)

    for idx, ticker in enumerate(tickers, start=1):
        print(f"[{idx:>4}/{len(tickers)}] {ticker}")
        try:
            _, records = fetch_ticker_bundle(
                fmp, ticker, snapshot_day, endpoints=endpoints
            )
            fmp_records.extend(records)
        except Exception as exc:  # continue to preserve partial, auditable progress
            failures.append({"ticker": ticker, "provider": "FMP", "error": str(exc)})
            print(f"  FMP failed: {exc}")
            continue

        sec_allowed = args.sec_limit is None or idx <= max(args.sec_limit, 0)
        if sec and sec_allowed:
            cik = sec_map.get(ticker)
            if not cik:
                failures.append({"ticker": ticker, "provider": "SEC", "error": "CIK not found"})
                print("  SEC skipped: CIK not found")
            else:
                try:
                    _, record = fetch_companyfacts_snapshot(sec, ticker, cik, snapshot_day)
                    sec_records.append(record)
                except Exception as exc:
                    failures.append({"ticker": ticker, "provider": "SEC", "error": str(exc)})
                    print(f"  SEC failed: {exc}")

    # Current files are disposable materialized views; immutable parts remain
    # under data/fundamental/snapshots/ and raw payloads remain content-addressed.
    # Materialize every available immutable part through the cutoff, not only
    # the tickers selected for this process.  This makes a completed batch
    # recover work preserved by an earlier interrupted batch automatically.
    fmp_fresh = load_latest_snapshot_parts("fmp", snapshot_day)
    fmp_current = _merge_current_base(
        FMP_CURRENT, fmp_fresh, replace_keys=("ticker", "endpoint")
    )
    current_paths: list[Path] = []
    if not fmp_current.empty:
        current_paths.append(write_current_parquet(fmp_current, "fmp_latest.parquet"))
    sec_fresh = load_latest_snapshot_parts("sec", snapshot_day)
    sec_current = _merge_current_base(
        SEC_CURRENT, sec_fresh, replace_keys=("ticker",)
    )
    if not sec_current.empty:
        current_paths.append(write_current_parquet(sec_current, "sec_latest.parquet"))

    manifest = {
        "policy_version": POLICY_VERSION,
        "run_type": "fundamental_source_snapshot",
        "snapshot_as_of": snapshot_day,
        "started_for_tickers": tickers,
        "bundle_depth": args.bundle_depth,
        "endpoints": list(endpoints),
        "sec_limit": args.sec_limit,
        "fmp_records": fmp_records,
        "sec_records": sec_records,
        "failures": failures,
        "completed_at": iso_utc(),
        "live_actions_enabled": False,
    }
    manifest_path = write_run_manifest(manifest)
    current_paths.append(manifest_path)

    print(f"FMP rows: {len(fmp_current):,}; SEC rows: {len(sec_current):,}")
    print(f"Failures: {len(failures)}; manifest: {manifest_path}")
    if args.upload:
        archive_paths = []
        for record in fmp_records + sec_records:
            for field in (
                "raw_path", "snapshot_path", "facts_path", "submissions_path"
            ):
                value = record.get(field)
                if value:
                    archive_paths.append(Path(value))
        # De-duplicate content-addressed paths while preserving deterministic order.
        all_paths = list(dict.fromkeys(current_paths + archive_paths))
        maybe_upload(all_paths)
    else:
        print("R2 upload skipped (use --upload after validating the snapshot).")

    if len(fmp_current) == 0:
        raise SystemExit("no FMP rows materialized")
    if failures and len(fmp_records) == 0:
        raise SystemExit("all requested FMP fetches failed")


if __name__ == "__main__":
    main()
