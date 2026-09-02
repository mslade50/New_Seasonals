"""Inspect inputs for the cycle/macro regime study (no outputs written)."""
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
pd.set_option("display.width", 250, "display.max_columns", 60)

led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
print("ledger", led.shape)
print(led.dtypes.to_string())
print(led["Signal Date"].min(), led["Signal Date"].max())
print(led.groupby("Strategy").agg(N=("R_Multiple", "size"), first=("Signal Date", "min"), avgR=("R_Multiple", "mean"),
                                   risk=("Risk_flat_750k", "mean"), pnl=("PnL_flat_750k", "sum")).to_string())
print(led[["Strategy", "Tier", "Direction", "Exit Type" if "Exit Type" in led else "Strategy"]].head(3).to_string())
print("Tier values", led["Tier"].unique(), "Direction", led["Direction"].unique())
print("Size_Mult describe", led["Size_Mult"].describe().to_string())

pc = pd.read_parquet(ROOT / "data/cboe_putcall.parquet")
print("\nputcall", pc.shape, pc.columns.tolist(), pc.index[:3], pc.index[-3:])
print(pc.tail(3).to_string())

fr = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
print("\nfragility", fr.shape, fr.columns.tolist(), fr.index.min(), fr.index.max())

pf = pq.ParquetFile(ROOT / "data/master_prices.parquet")
print("\nmaster_prices schema", pf.schema_arrow.names, pf.metadata.num_rows)
want = ["SPY", "QQQ", "IWM", "^VIX", "^VIX3M", "^VVIX", "^TNX", "^IRX", "^MOVE", "HYG", "LQD", "IEF", "TLT", "UUP", "GLD", "^SKEW"]
t = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date"], filters=[("ticker", "in", want)]).to_pandas()
print(t.groupby("ticker")["date"].agg(["min", "max", "size"]).to_string())

me = pd.read_csv(ROOT / "data/macro_events.csv")
print("\nmacro_events", me.shape, me.columns.tolist())
print(me.head(5).to_string())
print(me.iloc[:, 1].value_counts().head(15) if me.shape[1] > 1 else "")

sm = pd.read_parquet(ROOT / "data/sector_map.parquet")
print("\nsector_map", sm.shape, sm.columns.tolist())
print(sm.head(3).to_string())
