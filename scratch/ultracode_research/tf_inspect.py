from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
print("master_prices shape:", mp.shape)
print("columns type:", type(mp.columns))
print("columns sample:", list(mp.columns[:8]))
print("index:", mp.index[:2], mp.index[-2:])

cands = ["SPY","QQQ","IWM","EFA","EEM","VGK","EWJ","FXI","TLT","IEF","SHY","LQD","HYG",
         "GLD","SLV","DBC","USO","UNG","DBA","UUP","FXE","FXY","VNQ","GDX","XLE","EMB",
         "TIP","IAU","BND","AGG","EWZ","VWO","DIA","MDY","EWA","EWU","EWG","EWC","PPLT",
         "CPER","GSG","BIL"]

if isinstance(mp.columns, pd.MultiIndex):
    lv0 = set(mp.columns.get_level_values(0).unique())
    lv1 = set(mp.columns.get_level_values(1).unique())
    print("lv0 size:", len(lv0), "lv1 size:", len(lv1))
    for c in cands:
        key = None
        if (c, "Close") in mp.columns: key = (c, "Close")
        elif ("Close", c) in mp.columns: key = ("Close", c)
        if key:
            s = mp[key].dropna()
            print(f"{c}: {s.index.min().date()} -> {s.index.max().date()}  n={len(s)}")
        else:
            print(f"{c}: MISSING")
else:
    print(mp.head())
    print(mp.dtypes)
    # maybe long format
    for col in ["Ticker","ticker","Symbol"]:
        if col in mp.columns:
            tk = set(mp[col].unique())
            for c in cands:
                if c in tk:
                    sub = mp[mp[col]==c]
                    print(f"{c}: rows={len(sub)}")
                else:
                    print(f"{c}: MISSING")
            break
