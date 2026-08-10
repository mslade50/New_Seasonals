"""Pull just the price cache the Market Context sweep needs.

pull_scan_caches.py is the scan's puller and drags 60+ MB of seasonal ranks
and earnings calendars with it. The context brief reads exactly one file, and
this is also the retry target when the 18:30 freshness gate finds a stale bar
(spec section 10), so it stays small enough to run twice in an evening.

Exit 0 on a successful pull, 1 when R2 could not serve the key. Fail-loud on
purpose: a silent no-op here becomes a brief written on yesterday's tape.

    python scripts/pull_context_prices.py [--quiet]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cache_io import download_to_local  # noqa: E402

KEY = "master_prices.parquet"
DEST = ROOT / "data" / "master_prices.parquet"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not download_to_local(KEY, str(DEST)):
        print(f"ERROR: could not pull r2://{KEY}. The sweep will run on "
              f"whatever is on disk and the freshness gate will suppress the "
              f"price lane if that is stale.")
        return 1
    if not args.quiet:
        import pandas as pd
        df = pd.read_parquet(DEST, columns=["ticker", "date"])
        # SPY, not the global max: crypto trades weekends, so the max across
        # all tickers reads as a Saturday bar and looks like a bug.
        spy = df.loc[df["ticker"] == "SPY", "date"].max()
        print(f"master_prices pulled, freshest NYSE bar (SPY) "
              f"{pd.Timestamp(spy).date()}, {df['ticker'].nunique():,} tickers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
