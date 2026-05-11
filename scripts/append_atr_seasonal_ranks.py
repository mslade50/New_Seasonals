"""
Append new tickers to atr_seasonal_ranks.parquet without rebuilding the file.

Builds ATR seasonal ranks for the supplied tickers across the same year range
as the existing parquet, then merges. Existing rows for the same tickers are
dropped before merge so the operation is idempotent.

Usage:
    python scripts/append_atr_seasonal_ranks.py ^SOX ^MID ^DJT
"""
import argparse
import os
import sys
import tempfile

import pandas as pd

# Allow `from build_atr_seasonal_ranks import ...` when run from repo root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from build_atr_seasonal_ranks import build_atr_ranks, OUTPUT_PATH


def main(tickers):
    if not os.path.exists(OUTPUT_PATH):
        print(f"No existing parquet at {OUTPUT_PATH} — run the full build first.")
        sys.exit(1)

    existing = pd.read_parquet(OUTPUT_PATH)
    year_min = int(existing['Date'].dt.year.min())
    year_max = int(existing['Date'].dt.year.max())
    target_years = list(range(year_min, year_max + 1))
    print(f"Existing parquet: {len(existing):,} rows, "
          f"{existing['ticker'].nunique()} tickers, years {year_min}-{year_max}")
    print(f"Appending tickers: {tickers}")

    # Build into a temp parquet to avoid clobbering the real one.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = os.path.join(tmp, "new.parquet")
        build_atr_ranks(tickers, target_years, output_path=tmp_path)
        if not os.path.exists(tmp_path):
            print("Builder produced no output; aborting.")
            sys.exit(1)
        new_rows = pd.read_parquet(tmp_path)

    if new_rows.empty:
        print("No new rows generated; aborting.")
        sys.exit(1)

    # Drop any pre-existing rows for these tickers so the operation is idempotent
    keep = existing[~existing['ticker'].isin(new_rows['ticker'].unique())].copy()
    merged = pd.concat([keep, new_rows], ignore_index=True)
    merged = merged.sort_values(['ticker', 'Date']).reset_index(drop=True)

    print(f"\nMerged: {len(merged):,} rows, {merged['ticker'].nunique()} tickers")
    merged.to_parquet(OUTPUT_PATH, index=False)
    csv_path = OUTPUT_PATH.replace('.parquet', '.csv')
    merged.to_csv(csv_path, index=False)
    print(f"Wrote {OUTPUT_PATH} and {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tickers", nargs="+", help="Tickers to add (e.g. ^SOX ^MID ^DJT)")
    args = parser.parse_args()
    main(args.tickers)
