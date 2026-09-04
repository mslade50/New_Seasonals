import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

sm = pd.read_parquet(ROOT / "data" / "sector_map.parquet")
print("sector_map cols:", list(sm.columns), "rows", len(sm))
print(sm.head(8).to_string())
if "sector" in sm.columns:
    print(sm["sector"].value_counts().to_string())

mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet", columns=["ticker", "date"])
tk = sorted(mp["ticker"].unique())
print("\nmaster_prices tickers:", len(tk))
for t in ["XLU", "XLK", "XLV", "XLF", "XLE", "XLI", "XLY", "XLP", "XLB", "SMH",
          "NVDA", "AMAT", "ADI", "TXN", "MU", "QCOM", "INTC", "GLW", "TLT", "SPY",
          "PCG", "EIX", "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "ED", "PEG",
          "ETR", "FE", "CMS", "DTE", "PPL", "PNW", "CNP", "TJX", "HRL"]:
    print(f"  {t}: {'YES' if t in set(tk) else 'MISSING'}")

led = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
print("\nledger cols:", list(led.columns), "rows", len(led))
print(led.head(3).to_string())
print(led["Strategy"].value_counts().to_string() if "Strategy" in led.columns else "")

frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
print("\nfrag cols:", list(frag.columns), "rows", len(frag), frag.index[:3], frag.index[-3:])
print(frag.tail(3).to_string())

ec = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet")
print("\nearnings cols:", list(ec.columns), "rows", len(ec))
print(ec.head(3).to_string())
