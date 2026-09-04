"""Forensic reproduction of the EWZ 33.51 phantom-fill bug.

Goal: pin down the EXACT basis of (a) the live-staged limit and (b) the bars
it is compared against at the moment of the phantom fill.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "master_prices.parquet"
LEDGER = ROOT / "data" / "backtest_trades_full.parquet"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# ---- 1. Local cache EWZ bars around the signal + ex-div window ----
df = pd.read_parquet(CACHE)
print("CACHE columns sample:", list(df.columns)[:8], "... shape", df.shape)

# master_prices is typically a wide frame keyed by ticker; detect layout
ewz = None
if isinstance(df.columns, pd.MultiIndex):
    print("MultiIndex columns")
    try:
        ewz = df.xs("EWZ", level=1, axis=1)
    except Exception as e:
        print("xs level1 failed:", e)
        try:
            ewz = df.xs("EWZ", level=0, axis=1)
        except Exception as e2:
            print("xs level0 failed:", e2)
else:
    # maybe long format with a 'Ticker' column
    if "Ticker" in df.columns:
        ewz = df[df["Ticker"] == "EWZ"].copy()
    else:
        # maybe per-ticker columns flattened like EWZ_Close
        cols = [c for c in df.columns if str(c).upper().startswith("EWZ")]
        print("flat EWZ-prefixed cols:", cols[:12])
        if cols:
            ewz = df[cols].copy()

if ewz is None or (hasattr(ewz, "empty") and ewz.empty):
    print("Could not isolate EWZ; dumping cache head:")
    print(df.head())
else:
    print("\nEWZ layout cols:", list(ewz.columns)[:12])
    # restrict to early June 2026
    try:
        idx = ewz.index
        if not isinstance(idx, pd.DatetimeIndex):
            # maybe a Date column
            if "Date" in ewz.columns:
                ewz = ewz.set_index("Date")
            elif "Date" in df.columns:
                ewz = ewz.set_index(df.loc[ewz.index, "Date"])
        win = ewz.loc["2026-06-01":"2026-06-30"]
        print("\nEWZ bars 2026-06-01..06-30:")
        print(win)
    except Exception as e:
        print("windowing failed:", e)
        print(ewz.tail(20))

# ---- 2. Ledger rows for EWZ ----
print("\n==== LEDGER EWZ rows ====")
led = pd.read_parquet(LEDGER)
print("ledger cols:", list(led.columns))
ecol = None
for c in led.columns:
    if led[c].astype(str).str.contains("EWZ", na=False).any():
        ecol = c
        break
print("ticker-bearing col:", ecol)
if ecol:
    er = led[led[ecol] == "EWZ"]
    print(er.to_string())
