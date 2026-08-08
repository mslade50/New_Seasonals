"""Build U.S. economic actual/consensus history from FMP.

The stable FMP economic-calendar endpoint exposes release timestamps plus
actual, estimate, previous, unit, and impact.  Its useful populated history
begins in 2013 (2014 is the first broadly complete year), so this cache is a
complement to ``data/macro_events.csv`` rather than a replacement for that
2000+ release-date calendar.

Usage:
    python scripts/build_macro_releases.py --full --no-upload
    python scripts/build_macro_releases.py                 # rolling refresh
    python scripts/build_macro_releases.py --start 2020-01-01

The merge is intentionally conservative: once a released row has an actual,
the stored actual/consensus/previous values are not overwritten.  Backfilled
rows are labelled as vendor historical snapshots; only same-day captures are
labelled live captures.
"""
from __future__ import annotations

import argparse
import calendar
import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from macro_releases import (  # noqa: E402
    PARQUET_PATH,
    align_bls_release_calendar,
    merge_release_history,
    normalize_fmp_rows,
)


ENDPOINT = "https://financialmodelingprep.com/stable/economic-calendar"
ENV_PATH = ROOT / ".env"
R2_KEY = "macro_release_history.parquet"
HISTORY_START = pd.Timestamp("2013-01-01")
ROLLING_LOOKBACK_DAYS = 45
REQUEST_TIMEOUT = 45
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 0.10


def load_env() -> str:
    """Load the existing project FMP key without printing it."""
    key = os.environ.get("FMP_API_KEY", "").strip()
    if key:
        return key
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            name, value = text.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))
    key = os.environ.get("FMP_API_KEY", "").strip()
    if not key:
        raise SystemExit("FMP_API_KEY is required in the environment or project .env")
    return key


def month_windows(start: pd.Timestamp, end: pd.Timestamp):
    """Yield inclusive monthly request windows.

    FMP silently truncates long economic-calendar ranges to roughly one
    quarter.  Monthly chunks avoid that undocumented response cap.
    """
    cursor = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    while cursor <= end:
        month_end = pd.Timestamp(date(
            cursor.year,
            cursor.month,
            calendar.monthrange(cursor.year, cursor.month)[1],
        ))
        yield cursor, min(month_end, end)
        cursor = month_end + pd.Timedelta(days=1)


def fetch_window(
    session: requests.Session,
    api_key: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[dict]:
    params = {
        "from": start.strftime("%Y-%m-%d"),
        "to": end.strftime("%Y-%m-%d"),
        "apikey": api_key,
    }
    last_error = ""
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                payload = response.json()
                if isinstance(payload, list):
                    return payload
                last_error = f"unexpected {type(payload).__name__} payload"
            elif response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
            else:
                raise RuntimeError(
                    f"FMP economic-calendar {start.date()}..{end.date()} "
                    f"failed with HTTP {response.status_code}"
                )
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)
    raise RuntimeError(
        f"FMP economic-calendar {start.date()}..{end.date()} failed after "
        f"{MAX_RETRIES} attempts: {last_error}"
    )


def fetch_range(api_key: str, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    windows = list(month_windows(start, end))
    rows: list[dict] = []
    session = requests.Session()
    t0 = time.time()
    for i, (lo, hi) in enumerate(windows, start=1):
        payload = fetch_window(session, api_key, lo, hi)
        rows.extend(payload)
        if i == 1 or i == len(windows) or i % 12 == 0:
            print(
                f"  [{i:>3}/{len(windows)}] through {hi.date()}  "
                f"raw rows={len(rows):,}  elapsed={time.time() - t0:.1f}s"
            )
        if i < len(windows):
            time.sleep(SLEEP_BETWEEN_CALLS)
    return rows


def _completion_stats(df: pd.DataFrame, event_id: str, year: int) -> tuple[int, float, float]:
    sub = df[(df["event_id"] == event_id) & (df["release_date"].dt.year == year)]
    if sub.empty:
        return 0, 0.0, 0.0
    return len(sub), float(sub["actual"].notna().mean()), float(sub["consensus"].notna().mean())


def validate_history(df: pd.DataFrame, as_of: pd.Timestamp) -> None:
    """Fail loudly on structural or key-series coverage regressions."""
    if df.empty:
        raise ValueError("normalized macro release history is empty")
    if df["release_key"].duplicated().any():
        dupes = int(df["release_key"].duplicated(keep=False).sum())
        raise ValueError(f"macro release history has {dupes} duplicate release keys")
    if df["release_ts_utc"].isna().any() or df["release_date"].isna().any():
        raise ValueError("macro release history contains invalid release dates")
    if df["country"].ne("US").any():
        raise ValueError("macro release history contains non-US rows")

    # 2014 is FMP's first broadly complete surprise year.  Gate only completed
    # years so an in-progress year is not mistaken for a provider outage.
    last_full_year = as_of.year - 1
    problems = []
    thresholds = {
        "nfp": (11, 13, 0.95, 0.90),
        "cpi_mom": (11, 13, 0.95, 0.90),
        # 2025 has only ten provider rows after the shutdown/cancellation.
        "ppi_mom": (9, 13, 0.95, 0.70),
    }
    for year in range(2014, last_full_year + 1):
        for event_id, (lo, hi, actual_floor, consensus_floor) in thresholds.items():
            # The October 2025 CPI was cancelled during the appropriations
            # lapse, and FMP still carries the November row without an actual.
            # Ten populated releases out of twelve provider rows is the known
            # historical state, not a failed refresh.
            if event_id == "cpi_mom" and year == 2025:
                actual_floor = 0.80
            n, actual_cov, consensus_cov = _completion_stats(df, event_id, year)
            if not lo <= n <= hi:
                problems.append(f"{event_id} {year}: rows={n}, expected {lo}..{hi}")
            if actual_cov < actual_floor:
                problems.append(
                    f"{event_id} {year}: actual coverage {actual_cov:.0%} < {actual_floor:.0%}"
                )
            if consensus_cov < consensus_floor:
                problems.append(
                    f"{event_id} {year}: consensus coverage {consensus_cov:.0%} < "
                    f"{consensus_floor:.0%}"
                )
    if problems:
        raise ValueError("macro release coverage gate failed:\n  " + "\n  ".join(problems))


def load_existing(output: Path, try_r2: bool) -> pd.DataFrame:
    if not output.exists() and try_r2:
        try:
            from cache_io import download_to_local

            download_to_local(R2_KEY, str(output))
        except Exception as exc:
            print(f"[r2 download] non-fatal: {exc}")
    if not output.exists():
        return pd.DataFrame()
    existing = pd.read_parquet(output)
    existing["release_ts_utc"] = pd.to_datetime(existing["release_ts_utc"], utc=True)
    existing["release_date"] = pd.to_datetime(existing["release_date"]).dt.normalize()
    existing["first_seen_at_utc"] = pd.to_datetime(existing["first_seen_at_utc"], utc=True)
    existing["last_seen_at_utc"] = pd.to_datetime(existing["last_seen_at_utc"], utc=True)
    calendar_df = pd.read_csv(ROOT / "data" / "macro_events.csv")
    return align_bls_release_calendar(existing, calendar_df)


def upload(output: Path) -> None:
    try:
        from cache_io import upload_from_local

        if not upload_from_local(str(output), R2_KEY):
            print("[r2 upload] skipped or unsuccessful")
    except Exception as exc:
        print(f"[r2 upload] non-fatal: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FMP U.S. macro release history")
    parser.add_argument("--full", action="store_true", help="refetch from 2013-01-01")
    parser.add_argument("--start", help="inclusive fetch start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="inclusive fetch end date (default: today ET)")
    parser.add_argument("--output", default=str(PARQUET_PATH), help="output parquet path")
    parser.add_argument("--no-upload", action="store_true", help="do not upload the cache to R2")
    parser.add_argument(
        "--no-r2-download", action="store_true",
        help="do not try to pull the previous cache from R2 when local output is absent",
    )
    args = parser.parse_args()

    output = Path(args.output)
    existing = load_existing(output, try_r2=not args.no_r2_download)
    today_et = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    end = pd.Timestamp(args.end).normalize() if args.end else today_et

    if args.start:
        start = pd.Timestamp(args.start).normalize()
    elif args.full or existing.empty:
        start = HISTORY_START
    else:
        newest = pd.to_datetime(existing["release_date"]).max()
        start = max(HISTORY_START, newest - pd.Timedelta(days=ROLLING_LOOKBACK_DAYS))

    if start > end:
        raise SystemExit(f"start {start.date()} is after end {end.date()}")

    print(f"Fetching FMP economic calendar {start.date()} -> {end.date()} by month")
    api_key = load_env()
    raw = fetch_range(api_key, start, end)
    fetched_at = pd.Timestamp.now(tz="UTC")
    fresh = normalize_fmp_rows(raw, fetched_at=fetched_at, country="US")
    calendar_df = pd.read_csv(ROOT / "data" / "macro_events.csv")
    fresh = align_bls_release_calendar(fresh, calendar_df)
    if fresh.empty:
        raise SystemExit("ERROR: FMP returned no normalizable U.S. rows; keeping prior cache")

    merged = merge_release_history(existing, fresh)
    validate_history(merged, as_of=end)

    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output, index=False)

    released = merged[merged["actual"].notna()]
    with_surprise = released[released["consensus"].notna()]
    key_events = [
        "nfp", "cpi_mom", "core_cpi_mom", "ppi_mom", "retail_sales_mom",
        "initial_jobless_claims", "ism_manufacturing_pmi",
    ]
    print(f"\nSaved {output}")
    print(f"  rows:                 {len(merged):,}")
    print(f"  release date range:   {merged['release_date'].min().date()} -> "
          f"{merged['release_date'].max().date()}")
    print(f"  released rows:        {len(released):,}")
    print(f"  with consensus:       {len(with_surprise):,}")
    print("  key-event coverage:")
    for event_id in key_events:
        sub = merged[merged["event_id"] == event_id]
        print(
            f"    {event_id:<27} rows={len(sub):>4}  "
            f"actual={sub['actual'].notna().sum():>4}  "
            f"consensus={sub['consensus'].notna().sum():>4}"
        )
    print("  vintage labels:")
    for label, count in merged["vintage_quality"].value_counts().items():
        print(f"    {label:<27} {count:>6,}")

    if not args.no_upload:
        upload(output)


if __name__ == "__main__":
    main()
