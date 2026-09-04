"""Backfill sector classifications for ledger tickers missing from
symbol_master.parquet. Writes data/sector_overrides.parquet incrementally
(resumable: re-run to retry the ones that failed/None).
"""
from pathlib import Path
import time
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sector_overrides.parquet"

t = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
sm = pd.read_parquet(ROOT / "data" / "symbol_master.parquet")
missing = sorted(set(t["Ticker"]) - set(sm["ticker"]))

# resume: keep rows that already have a real sector
done = {}
if OUT.exists():
    prev = pd.read_parquet(OUT)
    done = {r.ticker: r.sector for r in prev.itertuples()
            if isinstance(r.sector, str) and r.sector not in ("", "UNKNOWN")}


def static_class(tk: str):
    if tk.startswith("^"):
        return "Index"
    if tk.endswith("-USD"):
        return "Crypto"
    if tk.endswith("=F"):
        return "Commodity"
    if "=X" in tk:
        return "FX"
    return None


rows = []
todo = [tk for tk in missing if tk not in done]
print(f"missing={len(missing)} already_done={len(done)} todo={len(todo)}", flush=True)

for i, tk in enumerate(missing):
    if tk in done:
        rows.append({"ticker": tk, "sector": done[tk], "source": "cache"})
        continue
    sc = static_class(tk)
    if sc:
        rows.append({"ticker": tk, "sector": sc, "source": "static"})
        continue
    sector = "UNKNOWN"
    try:
        info = yf.Ticker(tk).info
        sector = info.get("sector") or "UNKNOWN"
    except Exception as e:
        sector = "UNKNOWN"
    rows.append({"ticker": tk, "sector": sector, "source": "yfinance"})
    if (i + 1) % 20 == 0:
        pd.DataFrame(rows).to_parquet(OUT, index=False)
        got = sum(1 for r in rows if r["sector"] != "UNKNOWN")
        print(f"  {i+1}/{len(missing)}  resolved={got}", flush=True)
        time.sleep(0.3)

df = pd.DataFrame(rows)
df.to_parquet(OUT, index=False)
res = (df["sector"] != "UNKNOWN").mean()
print(f"DONE. wrote {len(df)} rows -> {OUT.name}  resolved={res:.0%}", flush=True)
print(df["sector"].value_counts().to_string())
