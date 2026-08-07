"""Remap a renamed ticker in master_prices.parquet (e.g. BK -> BNY).

A ticker change kills the old symbol on yfinance, so `update_master_prices`
silently stops extending it while the row stays in the universe forever. The
stale series then keeps scoring: on 2026-08-07 BK's last bar was three weeks
old and it still ranked as the HOTTEST 5-day name in the entire 217-name pitch
tape, because its "5 bars" spanned three weeks. A dead ticker does not go
quiet, it goes wrong.

Why a full REPLACE and not an append. The two series are the same instrument
on different dividend-adjustment vintages (BK vs BNY differed by a constant
0.3964%). Appending would splice two bases together and leave a phantom gap at
the join that the backtest engine reads as a real overnight move. Pulling the
new symbol's whole history puts every bar on one basis.

    python scripts/remap_ticker.py --old BK --new BNY            # dry run
    python scripts/remap_ticker.py --old BK --new BNY --write    # local write
    python scripts/remap_ticker.py --old BK --new BNY --write --upload

`--upload` is gated exactly like build_trade_ledger's: this parquet feeds live
order staging, so overwriting the R2 key is never a side effect of a local run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PARQUET = ROOT / "data" / "master_prices.parquet"
R2_KEY = "master_prices.parquet"
# Same-instrument test. The question is IDENTITY, not continuity: this does a
# full replace, so there is no join to be smooth across. Two records of one
# instrument agree on the overwhelming majority of bars up to one adjustment
# factor, and disagree on none of them by much.
#
# An early version gated on the ratio's standard deviation over the whole
# overlap and rejected two genuine renames. ASGN/EFOR agree to 1.00000 in
# every year from 2018 on and differ only on a few 2010 bars. IAC/PPLI agree
# exactly in 2010, 2018, 2022 and 2026 but differ by a constant 0.82034 across
# 2024-25. Four eras matching to five decimals is not coincidence; that window
# is a STALE ADJUSTMENT in our stored copy, which is a reason to replace the
# series rather than a reason to refuse. A single noisy sub-window must not
# outvote thousands of exact matches, so the test is now a robust fraction.
RATIO_BAND = 1e-3          # a bar "agrees" within 0.1% of the median ratio
MIN_AGREE_FRAC = 0.90      # and at least this share of bars must agree
COLS = ["ticker", "date", "Open", "High", "Low", "Close", "Volume"]


def fetch(ticker: str) -> pd.DataFrame:
    raw = yf.download(ticker, start="1999-01-01", auto_adjust=True,
                      progress=False)
    if raw is None or raw.empty:
        raise SystemExit(f"no yfinance data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    out = raw.reset_index().rename(columns={"Date": "date"})
    out["ticker"] = ticker
    return out[COLS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--upload", action="store_true",
                    help="overwrite the R2 key; live orders read it")
    ap.add_argument("--allow-missing", action="store_true",
                    help="insert the successor even when the old symbol has "
                         "no rows in the cache (no identity proof possible)")
    ap.add_argument("--force", action="store_true",
                    help="splice even if the overlap ratio is not constant")
    args = ap.parse_args()

    df = pd.read_parquet(PARQUET)
    old = df[df.ticker == args.old]
    if old.empty and not args.allow_missing:
        raise SystemExit(f"{args.old} is not in {PARQUET.name}. If the symbol "
                         f"fell out of the cache entirely, pass "
                         f"--allow-missing to insert the successor's history "
                         f"(no same-instrument proof is possible).")
    if (df.ticker == args.new).any():
        raise SystemExit(f"{args.new} already present; refusing to merge")

    if old.empty:
        # update_master_prices takes its universe FROM the parquet, so a symbol
        # that fails to download once falls out and never returns. MMC did
        # exactly that: a LIQUID-tier name with zero cached rows, silently
        # skipped by every scan since. There is no stored series to prove
        # identity against, so FMP's successor mapping is the only evidence.
        print(f"{args.old}: ABSENT from the cache entirely")
    else:
        print(f"{args.old}: {len(old)} bars, "
              f"{old.date.min().date()} -> {old.date.max().date()}")
    new = fetch(args.new)
    print(f"{args.new}: {len(new)} bars, "
          f"{new.date.min().date()} -> {new.date.max().date()}")

    # --- same-instrument proof --------------------------------------------
    if old.empty:
        print("  NOTE: no overlap to test. Inserting on FMP's mapping alone; "
              "eyeball the first scan that trades it.")
    else:
        j = (old.set_index("date")[["Close"]]
             .join(new.set_index("date")[["Close"]], how="inner",
                   lsuffix="_old", rsuffix="_new").dropna())
        if len(j) < 100:
            raise SystemExit(f"only {len(j)} overlapping bars; refusing to splice")
        ratio = j.Close_new / j.Close_old
        med = ratio.median()
        agree = ((ratio / med - 1.0).abs() < RATIO_BAND)
        frac = float(agree.mean())
        print(f"overlap {len(j)} bars | median ratio {med:.6f} | "
              f"{100 * frac:.1f}% of bars agree within {RATIO_BAND:.1%}")
        if frac < MIN_AGREE_FRAC and not args.force:
            raise SystemExit(
                f"only {100 * frac:.1f}% of overlapping bars agree (floor "
                f"{100 * MIN_AGREE_FRAC:.0f}%): this does not look like one "
                f"instrument under two symbols. Refusing without --force.")
        if frac < 1.0:
            bad = j[~agree]
            yrs = sorted({d.year for d in bad.index})
            print(f"  NOTE: {len(bad)} bar(s) disagree, years {yrs}. The "
                  f"replace puts every bar on the new symbol's current "
                  f"adjustment basis, fixing a stale stored window.")
        gained = int((new.date > old.date.max()).sum())
        print(f"bars recovered past {old.date.max().date()}: {gained}")

    # --- rebuild ----------------------------------------------------------
    rest = df[df.ticker != args.old]
    for c in ("Open", "High", "Low", "Close"):
        new[c] = new[c].astype("float32")
    new["Volume"] = new["Volume"].astype("float64")
    out = (pd.concat([rest, new[COLS]], ignore_index=True)
           .sort_values(["ticker", "date"]).reset_index(drop=True))

    # --- verification: nothing else moved ---------------------------------
    a = rest.sort_values(["ticker", "date"]).reset_index(drop=True)
    b = out[out.ticker != args.new].reset_index(drop=True)
    assert a.equals(b), "untouched tickers changed; aborting"
    assert not out[out.ticker == args.new][["Open", "High", "Low", "Close"]] \
        .isna().any().any(), "NaNs in the new series"
    print(f"\nverified: {out.ticker.nunique()} tickers "
          f"({df.ticker.nunique()} before), {len(out)} rows "
          f"({len(df)} before), every other ticker byte-identical")

    if not args.write:
        print("\ndry run. re-run with --write to apply locally.")
        return 0
    out.to_parquet(PARQUET, index=False)
    print(f"wrote {PARQUET}")

    if args.upload:
        from cache_io import upload_from_local
        upload_from_local(str(PARQUET), R2_KEY)
        print(f"uploaded -> r2://{R2_KEY}")
    else:
        print("NOT uploaded. R2 still serves the old series; pass --upload "
              "to publish (this is the cache live order staging reads).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
