"""Recon: schemas, provenance metadata, date ranges of every input."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
pd.set_option("display.width", 220, "display.max_columns", 60)

def meta(p):
    m = pq.read_metadata(p).metadata or {}
    return {k.decode(): v.decode()[:80] for k, v in m.items() if not k.startswith(b"pandas") and not k.startswith(b"ARROW")}

for rel in ["data/backtest_trades_full.parquet", "data/backtest_trades_pcfear_shadow.parquet"]:
    p = ROOT / rel
    df = pd.read_parquet(p)
    print(f"\n=== {rel}: {df.shape}")
    print("meta:", meta(p))
    print("cols:", list(df.columns))
    sd = pd.to_datetime(df["Signal Date"])
    print("Signal Date range:", sd.min().date(), "->", sd.max().date())
    fam = ["Weak Close Decent Sznls", "SPY QQQ MonFri Reversion", "Monday Dip",
           "Indices Oversold Bounce", "3x Bear ETF Overbot Fade", "Monthly Weak Close"]
    f = df[df["Strategy"].isin(fam)]
    print("family rows by strategy:\n", f["Strategy"].value_counts().to_string())
    print("family Signal Date range:", pd.to_datetime(f["Signal Date"]).min().date(), "->", pd.to_datetime(f["Signal Date"]).max().date())
    if "Tranche" in df.columns:
        print("Tranche values:", df["Tranche"].value_counts(dropna=False).to_dict())
    if "Tier" in df.columns:
        print("family Tier:", f["Tier"].value_counts(dropna=False).to_dict())
    print(f.tail(3).T.to_string())

for rel in ["data/rd2_fragility.parquet", "data/cboe_putcall.parquet",
            "scratch/ultracode_sizing_2026-09-02/dd_pit/pit_dial_extended.parquet"]:
    p = ROOT / rel
    df = pd.read_parquet(p)
    print(f"\n=== {rel}: {df.shape}")
    print("meta:", meta(p))
    print("cols:", list(df.columns), "index:", df.index.name, type(df.index).__name__)
    print("range:", df.index.min(), "->", df.index.max())
    print(df.tail(4).to_string())
