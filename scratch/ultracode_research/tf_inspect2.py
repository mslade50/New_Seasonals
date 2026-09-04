from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet", columns=["ticker", "date", "Close"])

univ = ["SPY","QQQ","IWM","EFA","EEM","FXI","EWZ","EWJ","VNQ","TLT","IEF","LQD","HYG",
        "GLD","SLV","DBC","USO","UUP","GDX","XLE","DIA","^IRX"]
for c in univ:
    s = mp[mp.ticker == c]
    if len(s):
        print(f"{c}: {s.date.min().date()} -> {s.date.max().date()}  n={len(s)}")
    else:
        print(f"{c}: MISSING")

carets = sorted(t for t in mp.ticker.unique() if str(t).startswith("^"))
print("caret tickers:", carets)

frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
print("\nfrag:", frag.shape, list(frag.columns), frag.index.min(), frag.index.max())

tr = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
print("\ntrades cols:", list(tr.columns))
print(tr["Exit Date"].min(), tr["Exit Date"].max())
print(tr[["PnL_flat_750k"]].describe())
