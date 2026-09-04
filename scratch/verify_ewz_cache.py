import pandas as pd
import numpy as np
from pathlib import Path

p = Path("data/master_prices.parquet")
print("cache exists:", p.exists(), "mtime:", pd.Timestamp(p.stat().st_mtime, unit='s') if p.exists() else None)
df = pd.read_parquet(p)
print("cols sample:", list(df.columns)[:8], "...")
# Figure out structure
print("index name:", df.index.name, "type:", type(df.index))
print("shape:", df.shape)
