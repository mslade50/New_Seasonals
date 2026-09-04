"""Forensic: EWZ phantom-fill investigation around 2026-06-09 limit @ 33.51."""
from pathlib import Path
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")
MP = ROOT / "data" / "master_prices.parquet"
LIMIT = 33.51

print("=" * 70)
print("STEP 1: EWZ from master_prices.parquet (ADJUSTED cache as stored)")
print("=" * 70)

df = pd.read_parquet(MP)
print("parquet shape:", df.shape)
print("index type:", type(df.index))
print("columns sample:", list(df.columns)[:8])

# Long format: ticker/date columns.
ewz = df[df["ticker"] == "EWZ"].copy()
print("EWZ rows:", len(ewz))
if ewz is not None and len(ewz):
    ewz = ewz.set_index(pd.to_datetime(ewz["date"]))
    ewz = ewz.sort_index()
    win = ewz.loc["2026-05-25":"2026-06-17"]
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in win.columns]
    print("\nEWZ adjusted OHLC 2026-05-25..2026-06-17:")
    print(win[cols].round(4).to_string())

    print(f"\nLIMIT under test = {LIMIT}")
    after = ewz.loc["2026-06-09":"2026-06-17"]
    if "Low" in after.columns:
        dips = after[after["Low"] <= LIMIT]
        print(f"Adjusted-cache days 6/9.. with Low <= {LIMIT}:")
        print(dips[cols].round(4).to_string() if not dips.empty else "  (none)")
        print(f"min adjusted Low in window after 6/9: {after['Low'].min():.4f}")
