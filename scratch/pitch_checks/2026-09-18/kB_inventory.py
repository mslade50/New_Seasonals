import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import pandas as pd

want = ["IWM", "SPY", "QQQ", "EEM", "EFA", "FXI", "MCHI", "KWEB", "ASHR", "EWH",
        "000001.SS", "^HSI", "CL=F", "USO", "DBC", "XLE", "BNO", "UCO", "^RUT",
        "GXC", "CQQQ", "YINN", "EWT", "EWY", "HG=F", "RB=F", "HO=F", "XOP", "VLO"]
mp = pd.read_parquet(PRICES_PATH, columns=["ticker", "date"])
have = mp[mp["ticker"].isin(want)].groupby("ticker")["date"].agg(["min", "max", "count"])
print(have)
print("missing:", sorted(set(want) - set(have.index)))
# any China-ish tickers
tk = sorted(mp["ticker"].unique())
print([t for t in tk if t.endswith(".SS") or t.endswith(".SZ") or t.endswith(".HK") or "HSI" in t])
