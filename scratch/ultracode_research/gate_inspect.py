"""Inspect data formats before the gate recomputes."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

import pyarrow.parquet as pq
f = pq.ParquetFile(ROOT / "data" / "master_prices.parquet")
print("master_prices schema:")
print(f.schema_arrow)
print("rows:", f.metadata.num_rows)

mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                     columns=["ticker", "date", "Open", "Close"])
print("\ndtypes:\n", mp.dtypes)
print("date range:", mp["date"].min(), "->", mp["date"].max())

UNI = ["SPY", "QQQ", "IWM", "EFA", "EEM", "FXI", "VNQ",
       "GLD", "SLV", "DBC", "USO", "UUP", "^IRX"]
sub = mp[mp.ticker.isin(UNI)]
g = sub.groupby("ticker").agg(n=("date", "size"), first=("date", "min"),
                              last=("date", "max"),
                              open_na=("Open", lambda s: s.isna().mean()))
print("\n12 ex-bonds tickers + ^IRX:\n", g)

# dup check
d = sub.duplicated(subset=["ticker", "date"]).sum()
print("dupes:", d)

# tf_monthly_series
tf = pd.read_parquet(ROOT / "scratch" / "ultracode_research" / "tf_monthly_series.parquet")
print("\ntf_monthly_series cols:", list(tf.columns), "N:", len(tf),
      tf.index.min(), "->", tf.index.max())

# ledger
tr = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
print("\nledger cols:", list(tr.columns))
print("N trades:", len(tr))
print(tr[["Exit Date", "PnL_flat_750k"]].describe(include="all").head(6))

# fragility
fr = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
print("\nfragility cols:", list(fr.columns), fr.index.min(), "->", fr.index.max(),
      "N:", len(fr))
