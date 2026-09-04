"""Drop in-progress bars dated after the settled session from the LOCAL cache.

The 21:10 UTC price cron was shed again tonight (third night running), so the
updater was re-run locally at ~20:10 ET with the R2_* vars blanked. Run that
late, yfinance already serves 24h markets (FX pairs, BTC/ETH) a bar dated
tomorrow (2026-08-28) that is a few hours of overnight trade. The scheduled
PM run never sees those, so they are removed here to keep the local cache on
the same footing as the production one. Local file only; R2 is untouched.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "data" / "master_prices.parquet"
SETTLED = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2026-08-27")

df = pd.read_parquet(PATH)
bad = df["date"] > SETTLED
print(f"rows dated after {SETTLED.date()}: {int(bad.sum())} "
      f"({sorted(df.loc[bad, 'ticker'].unique())})")
if bad.any():
    df = df.loc[~bad].reset_index(drop=True)
    df.to_parquet(PATH, index=False)
    print(f"rewrote {PATH.name}: {len(df):,} rows, max date {df['date'].max().date()}")
else:
    print("nothing to drop")
