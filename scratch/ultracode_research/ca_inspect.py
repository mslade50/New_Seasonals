"""Crisis-alpha track: inspect local data availability."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

mp = pd.read_parquet(ROOT / "data/master_prices.parquet")
print("master_prices shape:", mp.shape)
print("index:", type(mp.index).__name__, mp.index[:2])
print("columns type:", type(mp.columns).__name__)
print("cols sample:", list(mp.columns[:8]))

if isinstance(mp.columns, pd.MultiIndex):
    tickers = set(mp.columns.get_level_values(0)) | set(mp.columns.get_level_values(1))
else:
    tickers = set(mp.columns)
    if "Ticker" in mp.columns:
        tickers = set(mp["Ticker"].unique())
if isinstance(mp.index, pd.MultiIndex):
    for lev in mp.index.names:
        pass
    tickers |= set(mp.index.get_level_values(0).unique()) if mp.index.nlevels > 1 else set()

candidates = ["VIXY", "VXX", "UVXY", "SVXY", "VXZ", "GLD", "GDX", "TLT", "IEF", "SHY",
              "DBMF", "KMLM", "CTA", "PDBC", "DBC", "UUP", "SH", "SPY", "QQQ",
              "^VIX", "^VIX3M", "^VVIX", "BTAL", "TAIL", "IVOL", "GLDM", "SGOL",
              "ZROZ", "EDV", "TMF", "USO", "SLV", "TN", "^IRX"]
print("n tickers-ish:", len(tickers))
for c in candidates:
    print(f"  {c}: {'YES' if c in tickers else 'no'}")

fr = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
print("\nrd2_fragility:", fr.shape, fr.index.min(), "->", fr.index.max(), list(fr.columns))

led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
print("\nledger:", led.shape)
print("cols:", led.columns.tolist())

for f in ["fragility_63d_history.parquet", "signal_fire_history.parquet",
          "cboe_putcall.parquet", "atr_seasonal_ranks.parquet"]:
    p = ROOT / "data" / f
    print(f"{f}: exists={p.exists()}")
