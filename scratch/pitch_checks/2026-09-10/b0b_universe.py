"""b0b: enumerate country/region ETFs present in the cache, with history depth."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

px_all = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                         columns=["ticker", "date"])
g = px_all.groupby("ticker")["date"].agg(["min", "max", "count"])
tickers = sorted(g.index)

print("--- all EW* / country-ish candidates present ---")
cands = [t for t in tickers if t.startswith("EW") or t in {
    "EFA", "EEM", "VGK", "VEA", "VWO", "IEFA", "IEMG", "INDA", "FXI", "MCHI",
    "ASHR", "KWEB", "EPI", "EPOL", "EIS", "TUR", "GREK", "ARGT", "ECH", "EPU",
    "IDX", "THD", "VNM", "EIDO", "EPHE", "EZA", "NORW", "EDEN", "EFNL",
    "EIRL", "ENZL", "EWUS", "SPY", "IEV", "IOO", "SCZ", "ACWX", "IXUS"}]
for t in cands:
    r = g.loc[t]
    yrs = (r["max"] - r["min"]).days / 365.25
    print(f"  {t:8s} {str(r['min'].date())} .. {str(r['max'].date())}  n={r['count']:5d}  {yrs:5.1f}y")

print("\n--- FX / yen candidates ---")
for t in tickers:
    if any(k in t for k in ("JPY", "FX", "DX-", "USD", "=X", "UDN", "UUP")):
        r = g.loc[t]
        print(f"  {t:12s} {str(r['min'].date())} .. {str(r['max'].date())}  n={r['count']:5d}")

print("\n--- metals ---")
for t in ["GLD", "SLV", "GDX", "GDXJ", "SIL", "IAU", "PPLT", "PALL", "CPER",
          "COPX", "DBC", "USO", "UNG", "CEF", "SGOL", "SIVR"]:
    if t in g.index:
        r = g.loc[t]
        print(f"  {t:8s} {str(r['min'].date())} .. {str(r['max'].date())}  n={r['count']:5d}")
