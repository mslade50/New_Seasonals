"""Verify EWZ 6/8 cache values and the verify_fills.py rounding path."""
import pandas as pd
from pathlib import Path

p = Path("data/master_prices.parquet")
print("cache mtime:", pd.Timestamp(p.stat().st_mtime, unit="s"))

df = pd.read_parquet(p, columns=["ticker", "date", "Open", "High", "Low", "Close"])
ewz = df[df["ticker"] == "EWZ"].copy()
ewz["date"] = pd.to_datetime(ewz["date"])
ewz = ewz.set_index("date").sort_index()
sub = ewz.loc["2026-06-05":"2026-06-12"]
print("\nEWZ bars 6/5-6/12 (local cache, written Jun 11):")
print(sub[["Open", "High", "Low", "Close"]].to_string())

# ATR(14) on the 6/8 bar (Wilder). Build TR then 14-period Wilder smoothing.
tail = ewz.loc[:"2026-06-08"].tail(40).copy()
tr = pd.concat([
    tail["High"] - tail["Low"],
    (tail["High"] - tail["Close"].shift()).abs(),
    (tail["Low"] - tail["Close"].shift()).abs(),
], axis=1).max(axis=1)
atr_wilder = tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1]
atr_sma = tr.tail(14).mean()
print(f"\nATR(14) on 6/8  Wilder-ewm: {atr_wilder:.4f}  | simple-14: {atr_sma:.4f}")

c68 = ewz.loc["2026-06-08", "Close"]
print(f"6/8 Close (cache): {c68:.4f}")
print(f"limit = round({c68:.4f} - 0.25*{atr_wilder:.4f}, 2) = {round(c68 - 0.25*atr_wilder, 2)}")

# verify_fills uses SHEET-rounded Entry/ATR
for label, sc, atr in [("sheet-rounded 33.69/0.73", 33.69, 0.73),
                        ("sheet-precise 33.69/0.7314", 33.69, 0.7314)]:
    print(f"  {label}: limit = {round(sc - 0.25*atr, 2)}")
