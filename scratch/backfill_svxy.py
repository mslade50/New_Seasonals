"""One-shot: backfill SVXY into master_prices.parquet (local + R2).

Pulls the CURRENT R2 copy first (the local file may be stale), appends
full SVXY history from yfinance (adjusted, matching the cache basis),
dedupes, writes local, uploads back to R2. update_master_prices.py
maintains whatever tickers exist in the parquet, so SVXY becomes
self-maintaining after this run (the LEV3X backfill precedent).

Run: python scratch/backfill_svxy.py [--no-upload]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cache_io import download_to_local, upload_from_local, is_configured  # noqa: E402

LOCAL = ROOT / "data" / "master_prices.parquet"


def main() -> int:
    upload = "--no-upload" not in sys.argv
    if upload and not is_configured():
        print("R2 not configured — refusing a backfill that cannot round-trip "
              "(use --no-upload for a local-only test).")
        return 1

    if upload:
        print("Pulling current master_prices from R2...")
        if not download_to_local("master_prices.parquet", str(LOCAL)):
            print("R2 pull failed — aborting rather than uploading a stale base.")
            return 1

    mp = pd.read_parquet(LOCAL)
    if (mp["ticker"] == "SVXY").any():
        print("SVXY already present — nothing to do.")
        return 0

    print("Fetching SVXY history from yfinance...")
    raw = yf.download("SVXY", start="2011-10-01", auto_adjust=True,
                      progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs("SVXY", level="Ticker", axis=1)
    raw.columns = [c.capitalize() for c in raw.columns]
    raw = raw.dropna(subset=["Close"])
    if len(raw) < 2000:
        print(f"Only {len(raw)} SVXY bars returned — aborting.")
        return 1

    add = raw.reset_index().rename(columns={"Date": "date"})
    add["date"] = pd.to_datetime(add["date"]).dt.tz_localize(None).dt.normalize()
    add["ticker"] = "SVXY"
    for c in ("Open", "High", "Low", "Close"):
        add[c] = add[c].astype("float32")
    add["Volume"] = add["Volume"].astype("float64")
    add = add[["ticker", "date", "Open", "High", "Low", "Close", "Volume"]]

    out = pd.concat([mp, add], ignore_index=True)
    out = out.drop_duplicates(subset=["ticker", "date"], keep="last")
    out.to_parquet(LOCAL, index=False)
    print(f"Appended {len(add)} SVXY bars "
          f"({add['date'].min():%Y-%m-%d} .. {add['date'].max():%Y-%m-%d}); "
          f"parquet now {len(out)} rows.")

    if upload:
        if not upload_from_local(str(LOCAL), "master_prices.parquet"):
            print("R2 upload FAILED — local updated, R2 not. Re-run to retry.")
            return 1
        print("Uploaded to R2 — SVXY is now self-maintaining via the nightly "
              "update workflow.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
