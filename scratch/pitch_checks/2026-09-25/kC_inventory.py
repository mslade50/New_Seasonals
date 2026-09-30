import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import pandas as pd

TK = ["USDMXN=X", "USDBRL=X", "USDZAR=X", "USDTRY=X", "AUDJPY=X", "NZDJPY=X", "CADJPY=X",
      "JPY=X", "EURUSD=X", "AUDUSD=X", "^MOVE", "^VIX", "FXI", "KWEB", "EEM", "^HSI", "MU",
      "SMH", "SPY", "UUP", "DX-Y.NYB", "6M=F", "MXN=X", "000001.SS", "^SSEC", "EWW", "^IRX"]
P = load_prices(TK)
for t, df in P.items():
    c = df["Close"].dropna()
    print(f"{t:10s} {c.index[0].date()} .. {c.index[-1].date()} n={len(c)} last={c.iloc[-1]:.4f} "
          f"cols={list(df.columns)[:6]}")

mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet", columns=["ticker"])
u = sorted(mp.ticker.unique())
print("\nFX-like / China-like tickers in cache:",
      [t for t in u if t.endswith("=X") or t.endswith(".SS") or t.endswith(".SZ") or "SSE" in t or t.endswith("=F")])

e = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet")
print("\nearnings cols:", list(e.columns))
e["date"] = pd.to_datetime(e["date"])
mu = e[e.ticker == "MU"].sort_values("date")
print("MU prints:", len(mu), mu.date.min().date(), mu.date.max().date())
print(mu.tail(12).to_string())
