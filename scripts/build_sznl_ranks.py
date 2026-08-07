"""Rebuild sznl_ranks.csv for the whole book, using the repo's canonical
seasonal-rank method.

WHY THIS EXISTS. sznl_ranks.csv had no builder. It was a committed static
artifact from May 2026 covering 2025 and 2026 only, which meant two problems:
18 LIQUID names were missing entirely (so their `seasonal_rank` ML feature was
null), and every ticker runs dry on 2027-01-01 with nothing able to extend it.

THE METHOD IS NOT NEW. It is imported verbatim from build_sznl_forecast.py,
which produces exactly this file's schema for the SECTOR_ETFS universe. There
must be one definition of "seasonal rank" in this repo, so this script changes
the UNIVERSE and the YEARS, never the maths:

    log forward returns over [5, 10, 21]
      -> mean by day_count (trading day within the year)
      -> rank to percentiles, averaged across the three windows
      -> 25% all-years + 75% presidential cycle (same year % 4)
      -> 5-day centered smooth
      -> walk-forward: only years STRICTLY BEFORE the target

Reproducing the old file bit-for-bit is impossible: it was built against
yfinance's then-current adjusted series and dividends have gone ex since, so
the same code now yields different numbers (measured correlation 0.91-0.96,
max error ~25 rank points). Those values were an unreproducible vintage. This
replaces them with a reproducible one.

PRICE SOURCE. master_prices.parquet, the book's price source of truth, rather
than the per-ticker `yf.download(period="max")` the original used. That is a
deliberate trade: ~25 years of history gives ~25 samples per day_count and ~6
per cycle year, which is ample, and it makes the build fast, offline and
deterministic instead of 1000+ rate-limited network calls. Names absent from
the cache fall back to yfinance.

    python scripts/build_sznl_ranks.py --dry-run     # report, write nothing
    python scripts/build_sznl_ranks.py               # write sznl_ranks.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The single definition of the maths. Importing rather than re-implementing is
# the whole point: a second copy would drift from the sector-forecast file.
from build_sznl_forecast import (  # noqa: E402
    calculate_forecast_profile,
    get_forward_returns,
    generate_forecast_dates,
)
from strategy_config import CSV_UNIVERSE, LIQUID_PLUS_COMMODITIES  # noqa: E402

OUT = ROOT / "sznl_ranks.csv"
PRICES = ROOT / "data" / "master_prices.parquet"
MIN_BARS = 252 * 3      # build_sznl_forecast's own floor is 252; a seasonal
                        # profile needs several years to mean anything
DEFAULT_YEARS = [2025, 2026, 2027]


def load_prices() -> dict[str, pd.Series]:
    df = pd.read_parquet(PRICES, columns=["ticker", "date", "Close"])
    return {t: g.set_index("date")["Close"].sort_index()
            for t, g in df.groupby("ticker")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tickers", nargs="+", default=None)
    args = ap.parse_args()

    # NEVER LOSE COVERAGE. CSV_UNIVERSE is DERIVED from this file's ticker
    # column, and scripts/build_master_prices unions those tickers into the
    # price cache's universe, so writing a narrower file quietly shrinks both.
    # The old file carries 42 names that CSV_UNIVERSE filters out by design
    # (futures =F, crypto -USD, caret indices) but that other consumers still
    # need: dropping ^VIX, ^TNX and the futures would stop master_prices
    # maintaining them. So the build universe is a SUPERSET of what is already
    # there, never a replacement for it.
    if args.tickers:
        universe = sorted(set(args.tickers))
    else:
        existing = set()
        if OUT.exists():
            existing = set(pd.read_csv(OUT, usecols=["ticker"]).ticker.unique())
        universe = sorted(set(CSV_UNIVERSE) | set(LIQUID_PLUS_COMMODITIES)
                          | existing)
        print(f"universe: {len(CSV_UNIVERSE)} CSV + LIQUID union, "
              f"+{len(existing - set(CSV_UNIVERSE) - set(LIQUID_PLUS_COMMODITIES))} "
              f"carried from the existing file")
    print(f"universe: {len(universe)} tickers | years: {args.years}")

    prices = load_prices()
    print(f"master_prices: {len(prices)} tickers available")

    frames, skipped = [], []
    for i, t in enumerate(universe, 1):
        s = prices.get(t)
        if s is None or len(s) < MIN_BARS:
            skipped.append((t, 0 if s is None else len(s)))
            continue
        d = get_forward_returns(pd.DataFrame({"Close": s}))
        for year in args.years:
            prof = calculate_forecast_profile(d, year)
            if prof is None:
                continue
            dates = generate_forecast_dates(year)
            dates["seasonal_rank"] = dates["day_count"].map(prof).round(1)
            dates["ticker"] = t
            frames.append(dates[["Date", "seasonal_rank", "ticker"]])
        if i % 200 == 0:
            print(f"  [{i}/{len(universe)}] ...")

    out = (pd.concat(frames, ignore_index=True)
           .sort_values(["ticker", "Date"]).reset_index(drop=True))
    out = out.dropna(subset=["seasonal_rank"])

    print(f"\nbuilt {len(out):,} rows for {out.ticker.nunique()} tickers, "
          f"years {sorted(out.Date.dt.year.unique())}")
    if skipped:
        print(f"skipped {len(skipped)} (under {MIN_BARS} bars): "
              f"{[t for t, _ in skipped[:10]]}")

    # --- coverage report, the reason this was written ---------------------
    liq, csvu = set(LIQUID_PLUS_COMMODITIES), set(CSV_UNIVERSE)
    have = set(out.ticker.unique())
    print(f"LIQUID coverage : {len(liq & have)}/{len(liq)}"
          + (f"  missing {sorted(liq - have)}" if liq - have else ""))
    print(f"CSV coverage    : {len(csvu & have)}/{len(csvu)}")
    if OUT.exists():
        old = pd.read_csv(OUT, usecols=["ticker"])
        print(f"before: {old.ticker.nunique()} tickers | after: {len(have)}")

    if args.dry_run:
        print("\ndry run — nothing written.")
        return 0
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    print("NOTE: CSV_UNIVERSE is derived from this file's ticker column, so "
          "membership changes with it. Re-run the ledger before trusting stats.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
