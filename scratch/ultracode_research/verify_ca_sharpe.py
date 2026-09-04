from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")
tr = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
tr["Signal Date"] = pd.to_datetime(tr["Signal Date"])
tr["Exit Date"] = pd.to_datetime(tr["Exit Date"])
tr16 = tr[tr["Exit Date"] >= "2016-07-01"]
m = tr16.groupby(tr16["Exit Date"].dt.to_period("M"))["PnL_flat_750k"].sum()
m.index = m.index.to_timestamp("M")

def sh(x):
    return x.mean() / x.std() * np.sqrt(12)

print("Sharpe excl 2026-07 partial:", round(sh(m.loc[:"2026-06-30"]), 2),
      "N=", len(m.loc[:"2026-06-30"]))
print("Sharpe incl 2026-07 partial:", round(sh(m), 2), "N=", len(m))
print("last months:", m.tail(3).round(0).to_dict())
