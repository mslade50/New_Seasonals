"""Fetch Fama-French factor returns from the Ken French Data Library.

Writes data/factor_returns_monthly.parquet — long-short academic factor
returns in PERCENT per month:
    MktRF  market minus T-bill          1926-07+
    SMB    size (small minus big)       1926-07+
    HML    value (high minus low B/M)   1926-07+   ("growth" = short side)
    RF     one-month T-bill             1926-07+
    Mom    momentum (UMD, 12-2)         1927-01+
    RMW    profitability (quality-ish)  1963-07+
    CMA    investment (conservative-)   1963-07+

Source zips are stable public URLs (no key, no extra deps — requests +
zipfile). French CSVs carry preamble text and trailing annual blocks; the
parser keeps only the first monthly block (rows keyed YYYYMM). -99.99 /
-999 are French's missing markers -> NaN.

Usage:
    python scripts/fetch_factor_returns.py [--out data/factor_returns_monthly.parquet]
"""
import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

# zip name -> (csv columns we keep). ORDER MATTERS: the merge takes only
# NEW columns from later files, so the 3-factor file (1926+) must come
# before the 5-factor file (1963+) or pre-1963 MktRF/SMB/HML are lost —
# the 5-factor file then contributes just RMW/CMA.
SOURCES = [
    ("F-F_Research_Data_Factors_CSV.zip", ["Mkt-RF", "SMB", "HML", "RF"]),
    ("F-F_Research_Data_5_Factors_2x3_CSV.zip", ["RMW", "CMA"]),
    ("F-F_Momentum_Factor_CSV.zip", ["Mom"]),
]


def fetch_zip_csv(zip_name: str) -> str:
    r = requests.get(BASE + zip_name, timeout=60)
    r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
    return zf.read(csv_name).decode("latin-1")


def parse_monthly_block(text: str) -> pd.DataFrame:
    """First block of rows keyed YYYYMM, with the nearest preceding header line."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if re.match(r"^\s*\d{6}\s*,", l))
    header = None
    for j in range(start - 1, -1, -1):
        if lines[j].strip():
            header = [h.strip() for h in lines[j].split(",")]
            break
    rows = []
    for l in lines[start:]:
        if not re.match(r"^\s*\d{6}\s*,", l):
            break  # end of the monthly block (annual section follows)
        rows.append([c.strip() for c in l.split(",")])
    df = pd.DataFrame(rows)
    ncols = df.shape[1]
    cols = ["date"] + [h for h in header if h][-(ncols - 1):] if header else ["date"] + [f"c{i}" for i in range(1, ncols)]
    # momentum file's header is blank/'Mom' with stray spacing — normalize
    if ncols == 2 and (len(cols) != 2 or not cols[1]):
        cols = ["date", "Mom"]
    df.columns = cols[:ncols]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.set_index("date")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([-99.99, -999.0], pd.NA).astype(float)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "factor_returns_monthly.parquet"))
    args = ap.parse_args()

    merged: pd.DataFrame | None = None
    for zip_name, keep in SOURCES:
        print(f"fetching {zip_name} ...")
        df = parse_monthly_block(fetch_zip_csv(zip_name))
        cols = [c for c in keep if c in df.columns]
        missing = set(keep) - set(cols)
        if missing:
            print(f"  warn: {zip_name} missing expected columns {sorted(missing)} "
                  f"(has {list(df.columns)})")
        df = df[cols]
        print(f"  {len(df)} months, {df.index.min().date()} -> {df.index.max().date()}, cols {cols}")
        if merged is None:
            merged = df
        else:
            new_cols = [c for c in df.columns if c not in merged.columns]
            merged = merged.join(df[new_cols], how="outer")

    merged = merged.rename(columns={"Mkt-RF": "MktRF"}).sort_index()
    # sanity: momentum must reach back before 1930 and forward past 2024
    assert merged["Mom"].dropna().index.min().year <= 1928, "Mom history too short — parse failure?"
    assert merged.index.max().year >= 2024, "data ends early — parse failure?"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out)
    print(f"\nwrote {out} — {len(merged)} months x {list(merged.columns)}")
    print(merged.tail(3).round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
