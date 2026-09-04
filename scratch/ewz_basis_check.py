"""Confirm the local cache (written Jun 11) currently holds RAW (un-re-adjusted)
EWZ bars, i.e. the live basis as of the staging date. Compare cache vs yfinance
auto_adjust=False (raw) and auto_adjust=True (now, post-6/15 ex-div)."""
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")
df = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                     columns=["ticker", "date", "Open", "High", "Low", "Close"],
                     filters=[("ticker", "in", ["EWZ"])])
df["date"] = pd.to_datetime(df["date"])
cache = df.set_index("date").sort_index().loc["2026-06-04":"2026-06-12", "Close"]

# Raw closes (live basis, auto_adjust=False) from yfinance run earlier:
raw = pd.Series({
    "2026-06-04": 34.78, "2026-06-05": 34.01, "2026-06-08": 33.69,
    "2026-06-09": 33.92, "2026-06-10": 33.78, "2026-06-11": 34.81, "2026-06-12": 35.10,
})
raw.index = pd.to_datetime(raw.index)
# Post-ex adjusted (auto_adjust=True now):
adj_now = (raw * 0.99057)

cmp = pd.DataFrame({"cache_close": cache, "raw_close": raw, "adj_now_close": adj_now})
cmp["cache==raw"] = (cmp["cache_close"].round(2) == cmp["raw_close"].round(2))
print(cmp.round(4).to_string())
print()
print("If cache_close == raw_close (not adj_now), the LOCAL cache is still RAW basis,")
print("so the staged 33.51 limit and the cache were on the SAME basis on the live date.")
print("The phantom only appears once the nightly rebuild re-adjusts the cache by f=0.99057")
print("(after the 6/15 ex-div) while the STAGED 33.51 stays frozen in raw dollars.")
