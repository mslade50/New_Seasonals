"""b0 probe: what tickers exist, what history depth, and confirm today's state."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

WANT = ["UUP", "DX-Y.NYB", "^TNX", "SPY", "EWJ", "EFA", "GLD", "SLV",
        "JPY=X", "FXY", "DXJ", "HEWJ", "EWG", "EWU", "EWQ", "EWL", "EWA",
        "EWC", "EWY", "EWT", "EWZ", "EWW", "INDA", "FXI", "EEM", "EPP",
        "EWH", "EWS", "EWN", "EWD", "EWP", "EWI", "VGK", "IEV", "^N225",
        "EURUSD=X", "6J=F", "^GSPC", "TLT", "IEF"]

px_all = pd.read_parquet(Path(__file__).resolve().parents[3] / "data" / "master_prices.parquet")
print("panel shape", px_all.shape, "cols", list(px_all.columns)[:12])
if isinstance(px_all.columns, pd.MultiIndex):
    tickers = sorted(set(px_all.columns.get_level_values(-1)))
else:
    tickers = sorted(set(px_all["ticker"].unique())) if "ticker" in px_all.columns else []
print("n tickers", len(tickers))
for w in WANT:
    print(f"  {w:12s} {'YES' if w in tickers else 'no'}")

have = [w for w in WANT if w in tickers]
d = load_prices(have)
for k in have:
    df = d[k]
    print(f"{k:12s} rows={len(df):6d}  {df.index[0].date()} .. {df.index[-1].date()}"
          f"  last_close={df['Close'].iloc[-1]:.4f}")
