"""Which factor ETFs does master_prices have, and how deep is the history?"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

CANDIDATES = [
    # core factor ETFs
    "MTUM", "QUAL", "USMV", "VLUE", "SIZE", "SPLV", "SPHB", "SPHQ", "SPMO",
    "VTV", "VUG", "IWD", "IWF", "MOAT", "COWZ", "DGRO", "VIG", "NOBL", "SDY",
    "RSP", "IJR", "IWM", "IWN", "IWO", "MDY", "IJH", "VBR", "AVUV", "DFSV",
    "QQQ", "SPY", "IVV", "VOO", "EFA", "EEM", "IEFA", "ACWI", "VEA",
    "EFAV", "EEMV", "IMTM", "IQLT", "IVLU", "ISCF",  # intl factor
    "IEF", "TLT", "SHY", "BIL", "AGG", "LQD", "HYG", "GLD",
    "XLP", "XLU", "XLV",  # defensive sectors as factor proxies
    "DBMF", "KMLM", "CTA", "BTAL", "QAI",  # alt/managed futures
    "VMOT", "FCTR", "OMFL", "DYNF", "LRGF", "QLTA",
]

mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet", columns=["ticker", "date"])
have = set(mp["ticker"].unique())
print(f"universe size: {len(have)}")

rng = mp.groupby("ticker")["date"].agg(["min", "max", "count"])
rows = []
for t in CANDIDATES:
    if t in have:
        r = rng.loc[t]
        rows.append((t, str(pd.Timestamp(r["min"]).date()), str(pd.Timestamp(r["max"]).date()), int(r["count"])))
    else:
        rows.append((t, "MISSING", "", 0))

out = pd.DataFrame(rows, columns=["ticker", "first", "last", "nbars"])
print(out.to_string(index=False))
