"""Pull load-bearing non-fundamental caches for the research workflow.

GitHub Actions starts from a clean checkout while the current symbol master and
adjusted master price cache are intentionally gitignored.  This helper fails
closed if either R2 object cannot be materialized.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cache_io import download_to_local, last_download_error  # noqa: E402


REQUIRED = {
    "symbol_master.parquet": ROOT / "data" / "symbol_master.parquet",
    "master_prices.parquet": ROOT / "data" / "master_prices.parquet",
}

OPTIONAL = {
    "overflow_prices.parquet": ROOT / "data" / "overflow_prices.parquet",
    "fundamental/current/fmp_latest.parquet": (
        ROOT / "data" / "fundamental" / "current" / "fmp_latest.parquet"
    ),
    "fundamental/current/sec_latest.parquet": (
        ROOT / "data" / "fundamental" / "current" / "sec_latest.parquet"
    ),
    "fundamental/current/underwrite_decisions_latest.json": (
        ROOT
        / "data"
        / "fundamental"
        / "current"
        / "underwrite_decisions_latest.json"
    ),
    "fundamental/site_state.json": (
        ROOT / "data" / "fundamental" / "current" / "site_state_latest.json"
    ),
}

SITE_STATE_KEY = "fundamental/site_state.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--site-state-only",
        action="store_true",
        help="pull only the optional private-site research-priority state",
    )
    args = parser.parse_args()
    if args.site_state_only:
        path = OPTIONAL[SITE_STATE_KEY]
        if download_to_local(SITE_STATE_KEY, str(path)):
            print(f"ready: {SITE_STATE_KEY} -> {path}")
        else:
            print(f"optional cache unavailable: {SITE_STATE_KEY}")
        return

    failures = []
    for key, path in REQUIRED.items():
        if download_to_local(key, str(path)):
            print(f"ready: {key} -> {path}")
        else:
            failures.append(f"{key}: {last_download_error() or 'download failed'}")
    if failures:
        raise SystemExit("required fundamental inputs unavailable: " + "; ".join(failures))
    for key, path in OPTIONAL.items():
        if download_to_local(key, str(path)):
            print(f"ready: {key} -> {path}")
        else:
            print(f"optional cache unavailable: {key}")


if __name__ == "__main__":
    main()
