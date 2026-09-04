from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
print("master_prices shape:", mp.shape)
print("index:", mp.index[:3], type(mp.index))
print("columns type:", type(mp.columns))
print("columns sample:", list(mp.columns[:10]))
if isinstance(mp.columns, pd.MultiIndex):
    print("levels:", mp.columns.names)
    lv0 = mp.columns.get_level_values(0).unique()
    lv1 = mp.columns.get_level_values(1).unique()
    print("lv0 unique (first 10):", list(lv0[:10]), len(lv0))
    print("lv1 unique (first 10):", list(lv1[:10]), len(lv1))
    print("has SPY:", "SPY" in lv0 or "SPY" in lv1)
    print("has ^VIX:", "^VIX" in lv0 or "^VIX" in lv1)

frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
print("\nfrag shape:", frag.shape, "cols:", list(frag.columns))
print(frag.tail(3))
print("frag index range:", frag.index.min(), frag.index.max())

tr = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
print("\ntrades:", tr.shape)
print(tr[["Strategy", "Signal Date", "R_Multiple"]].tail(3))
