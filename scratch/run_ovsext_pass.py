"""One-off: run build_ovsext_counterfactual against the EXISTING ledger +
master_prices (no full rebuild) so the site toggle can be tested locally.
The nightly deploy produces the same parquet inside build_trade_ledger."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_trade_ledger import build_ovsext_counterfactual, OUT_PARQUET

df = pd.read_parquet(OUT_PARQUET)
for c in ("Signal Date", "Entry Date", "Exit Date", "Time Stop"):
    if c in df.columns:
        df[c] = pd.to_datetime(df[c])

prices = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
prices["date"] = pd.to_datetime(prices["date"])
ovs_tickers = df.loc[df["Strategy"] == "Overbot Vol Spike", "Ticker"].unique()
prices = prices[prices["ticker"].isin(ovs_tickers)]
md = {t: g.set_index("date").sort_index() for t, g in prices.groupby("ticker")}

build_ovsext_counterfactual(df, md)
