import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pandas as pd, numpy as np
from pitch_lab import *  # noqa

mp = pd.read_parquet(ROOT/"data"/"master_prices.parquet")
print("master_prices cols:", list(mp.columns), "rows", len(mp))
mp["date"]=pd.to_datetime(mp["date"])
print("date range", mp["date"].min(), mp["date"].max(), "tickers", mp["ticker"].nunique())
sys.path.insert(0, str(ROOT))
from abs_return_dispersion import SP500_TICKERS
have = set(mp["ticker"].unique())
print("SP500 in cache:", len(set(SP500_TICKERS)&have), "of", len(set(SP500_TICKERS)))
for t in ["SPY","SVXY","^VIX","VIXY","VXX","IWM","QQQ","USO","XLE","XOP","VLO","DBC","UNG","GLD","SLV","MPC","PSX","OIH","CVX","XOM","^VIX3M","UUP","TLT","EFA","EEM","MDY","XLI","XLF","XLK","XLY","XLP","XLV","XLU","XLB","XLE","IYT","KRE","SMH","GDX","CL=F","RB=F","HO=F","NG=F","BNO"]:
    if t in have:
        g=mp[mp["ticker"]==t]
        print(f"  {t:8s} {g['date'].min().date()} .. {g['date'].max().date()}  n={len(g)}")
    else:
        print(f"  {t:8s} MISSING")

fr = pd.read_parquet(ROOT/"data"/"rd2_fragility.parquet")
print("\nfragility cols", list(fr.columns), fr.index.min(), fr.index.max(), len(fr))
print(fr.tail(3))
ev = load_events()
print("\nevents kinds:", ev["event"].value_counts().to_dict())
qw = ev[(ev["event"]=="quad_witching")]
print("quad witching Sep:", [str(d.date()) for d in qw["date"] if d.month==9][:30])
