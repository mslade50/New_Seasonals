"""Backfill price history for the 11 missing tradeable proxies into master_prices
(gitignored), then build their ATR-seasonal ranks into a SEPARATE local file
(keeps the git-tracked atr_seasonal_ranks.parquet untouched). After this, the
proxy backtest can scan all 22 proxies."""
import sys
import pandas as pd
import yfinance as yf

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"
sys.path.insert(0, ROOT)

ETFS = ["EWG", "EWQ", "EWU", "FEZ", "SOXX", "IJH", "ONEQ", "VXX", "EWA", "EWC", "EWH"]
MP = ROOT + r"\data\master_prices.parquet"
EXTRA_RANKS = ROOT + r"\data\proxy_extra_ranks.parquet"

mp = pd.read_parquet(MP)
have = set(mp["ticker"].astype(str).str.upper())
todo = [t for t in ETFS if t not in have]
print(f"master_prices has {mp['ticker'].nunique()} tickers; fetching {len(todo)}: {todo}")

rows = []
for t in todo:
    raw = yf.download(t, start="1990-01-01", auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        print(f"  {t}: NO DATA"); continue
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).capitalize() for c in raw.columns]
    raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]).reset_index()
    raw = raw.rename(columns={raw.columns[0]: "date"})
    raw["ticker"] = t
    rows.append(raw[["ticker", "date", "Open", "High", "Low", "Close", "Volume"]])
    print(f"  {t}: {len(raw)} bars {raw['date'].min().date()} -> {raw['date'].max().date()}")

if rows:
    new = pd.concat(rows, ignore_index=True)
    out = pd.concat([mp, new], ignore_index=True)
    out.to_parquet(MP, index=False)
    print(f"master_prices: {mp['ticker'].nunique()} -> {out['ticker'].nunique()} tickers (+{len(new)} rows)")

# ranks into a separate file (merge=False, only these tickers), reusing master prices
import build_atr_seasonal_ranks as bar
bar.build_atr_ranks(ETFS, list(range(2001, 2027)), output_path=EXTRA_RANKS, merge=False)
print(f"\nDONE -> extra ranks at {EXTRA_RANKS}")
