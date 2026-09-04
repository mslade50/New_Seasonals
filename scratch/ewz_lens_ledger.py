"""Check the ledger for EWZ OLV trades around the disputed dates."""
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")
LED = ROOT / "data" / "backtest_trades_full.parquet"

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)

df = pd.read_parquet(LED)
print("ledger cols:", list(df.columns))
ewz = df[df.astype(str).apply(lambda r: r.str.contains("EWZ", case=False, na=False)).any(axis=1)]
# Find ticker column
tcol = next((c for c in df.columns if c.lower() in ("ticker", "symbol")), None)
print("ticker col:", tcol)
if tcol:
    ewz = df[df[tcol].astype(str).str.upper() == "EWZ"]
print("\nEWZ rows in ledger:", len(ewz))
# Show date-ish columns
datecols = [c for c in df.columns if "date" in c.lower() or "entry" in c.lower() or "signal" in c.lower() or "exit" in c.lower()]
print("date/entry cols:", datecols)
if len(ewz):
    show = ewz.copy()
    # filter to 2026 if possible
    for dc in datecols:
        try:
            show[dc] = pd.to_datetime(show[dc], errors="ignore")
        except Exception:
            pass
    print(show.tail(15).to_string())
