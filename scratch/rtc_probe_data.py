import pandas as pd
from pathlib import Path

root = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")

print("=== iv_history.parquet ===")
iv = pd.read_parquet(root / "data" / "iv_history.parquet")
print("shape:", iv.shape)
print("columns:", list(iv.columns))
print("dtypes:\n", iv.dtypes)
print(iv.head(10))
print(iv.tail(5))
for c in iv.columns:
    if iv[c].dtype == object and iv[c].nunique() < 50:
        print(c, "uniques:", sorted(iv[c].dropna().unique().tolist())[:50])
# date span
for cand in ("date", "Date", "asof", "as_of"):
    if cand in iv.columns:
        print("date span:", iv[cand].min(), "->", iv[cand].max())
if isinstance(iv.index, pd.DatetimeIndex):
    print("index span:", iv.index.min(), "->", iv.index.max())
if "ticker" in iv.columns:
    print("n tickers:", iv["ticker"].nunique())
    print("sample tickers:", iv["ticker"].unique()[:30])

print("\n=== master_prices coverage ===")
mp = pd.read_parquet(root / "data" / "master_prices.parquet")
print("mp shape:", mp.shape, "cols:", list(mp.columns)[:10])
tick_col = "Ticker" if "Ticker" in mp.columns else ("ticker" if "ticker" in mp.columns else None)
if tick_col:
    for t in ["^VIX", "^VIX3M", "^SKEW", "SPY", "^GSPC"]:
        sub = mp[mp[tick_col] == t]
        if len(sub):
            dc = "Date" if "Date" in sub.columns else sub.index
            dmin = sub["Date"].min() if "Date" in sub.columns else sub.index.min()
            dmax = sub["Date"].max() if "Date" in sub.columns else sub.index.max()
            print(f"{t}: {len(sub)} rows, {dmin} -> {dmax}")
        else:
            print(f"{t}: NOT PRESENT")
else:
    print("index type:", type(mp.index))
    if isinstance(mp.index, pd.MultiIndex):
        print("index names:", mp.index.names)
        lv = mp.index.get_level_values
        names = mp.index.names
        # try find ticker level
        for lvl in names:
            vals = mp.index.get_level_values(lvl)
            if vals.dtype == object:
                for t in ["^VIX", "^VIX3M", "^SKEW", "SPY"]:
                    mask = vals == t
                    if mask.any():
                        sub = mp[mask]
                        other = [n for n in names if n != lvl][0]
                        d = sub.index.get_level_values(other)
                        print(f"{t}: {mask.sum()} rows, {d.min()} -> {d.max()}")
                    else:
                        print(f"{t}: NOT in level {lvl}")
                break
