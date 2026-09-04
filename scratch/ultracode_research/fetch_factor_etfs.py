"""Try to fetch factor ETF history via yfinance (repo's standard data source).
Saves to scratch/ultracode_research/factor_etf_prices.parquet if successful.
Falls back gracefully if no network.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

TICKERS = ["MTUM", "QUAL", "USMV", "VLUE", "SPLV", "SPHQ", "RSP", "SPY", "BIL"]

try:
    import yfinance as yf
    raw = yf.download(TICKERS, start="2000-01-01", auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("empty download")
    # MultiIndex (Price, Ticker) per CLAUDE.md
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.dropna(how="all")
    print("fetched:", close.shape)
    print(close.notna().idxmax())  # first valid date per ticker
    print(close.tail(2))
    close.to_parquet(HERE / "factor_etf_prices.parquet")
    print("SAVED")
except Exception as e:
    print(f"FETCH FAILED: {type(e).__name__}: {e}")
