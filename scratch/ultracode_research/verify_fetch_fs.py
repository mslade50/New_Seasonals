from pathlib import Path
import pandas as pd
import yfinance as yf

root = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")
out = root / "scratch" / "ultracode_research" / "verify_fs_prices.parquet"

tickers = ["SPY", "USMV", "QUAL", "MTUM", "VLUE", "SPHQ", "BIL", "TLT"]
raw = yf.download(tickers, start="2000-01-01", auto_adjust=True, progress=False)
px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
px = px[tickers]
px.to_parquet(out)
print(px.shape)
print(px.tail(3))
print(px.apply(lambda s: s.first_valid_index()))

# cross-check vs researcher parquet and master_prices SPY
theirs = pd.read_parquet(root / "scratch" / "ultracode_research" / "factor_etf_prices.parquet")
common = px.index.intersection(theirs.index)
for t in ["SPY", "USMV", "BIL"]:
    a = px.loc[common, t].dropna()
    b = theirs.loc[common, t].reindex(a.index).dropna()
    ix = a.index.intersection(b.index)
    rel = ((a[ix] / b[ix]) - 1).abs().max()
    print(f"max rel diff mine vs theirs {t}: {rel:.6f}")

mp = pd.read_parquet(root / "data" / "master_prices.parquet")
spy_mp = mp[mp.ticker == "SPY"].set_index("date")["Close"]
ix = spy_mp.index.intersection(px.index)[-750:]
rel = ((spy_mp[ix] / px.loc[ix, "SPY"]) - 1).abs().max()
print(f"max rel diff master_prices SPY vs mine (3y): {rel:.6f}")
