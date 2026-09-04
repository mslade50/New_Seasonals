"""EWZ forensic part 2 — extract EWZ rows, compute ATR(14), -0.25 ATR limit,
and check forward lows. Detect whether cache is pre-ex or post-ex."""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")
MP = ROOT / "data" / "master_prices.parquet"

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 60)

df = pd.read_parquet(MP, columns=["ticker", "date", "Open", "High", "Low", "Close", "Volume"])
ewz = df[df["ticker"] == "EWZ"].copy()
ewz["date"] = pd.to_datetime(ewz["date"])
ewz = ewz.sort_values("date").reset_index(drop=True)
print("EWZ rows:", len(ewz), "| date range:", ewz["date"].min().date(), "->", ewz["date"].max().date())

# Wilder ATR(14) the way indicators usually compute it. Use the same as repo.
high = ewz["High"]; low = ewz["Low"]; close = ewz["Close"]
prev_close = close.shift(1)
tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
# Wilder smoothing
atr_wilder = tr.ewm(alpha=1/14, adjust=False).mean()
# Simple rolling mean ATR (alt)
atr_sma = tr.rolling(14).mean()
ewz["ATR_wilder"] = atr_wilder
ewz["ATR_sma"] = atr_sma

# Show the tail around the signal date 2026-06-08 and forward
tail = ewz[ewz["date"] >= "2026-05-25"][["date", "Open", "High", "Low", "Close", "ATR_wilder", "ATR_sma"]]
print("\n=== EWZ tail (cache, written Jun 11) ===")
print(tail.to_string(index=False))

# The signal day per the theory is 6/8. Pull that row.
sig = ewz[ewz["date"] == "2026-06-08"]
if not sig.empty:
    c = sig["Close"].iloc[0]
    aw = sig["ATR_wilder"].iloc[0]
    asm = sig["ATR_sma"].iloc[0]
    print(f"\n6/8 Close={c:.4f}  ATR_wilder={aw:.4f}  ATR_sma={asm:.4f}")
    for label, a in [("wilder", aw), ("sma", asm)]:
        lim = round(c - 0.25 * a, 2)
        print(f"  limit (-0.25 {label} ATR) = {c - 0.25*a:.4f} -> round2 = {lim:.2f}")
else:
    print("\nNo 6/8 row in cache.")
