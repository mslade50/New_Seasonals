"""OLV edge by signal clustering: (1) temporal (how many OLV signals fire
together in time) and (2) sector concentration (same-sector signals bunched
in time). Sector map = symbol_master.parquet (current classification, static).
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STRAT = "Oversold Low Volume"

t = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
olv = t[t["Strategy"] == STRAT].copy()
olv["Signal Date"] = pd.to_datetime(olv["Signal Date"])
olv = olv.sort_values("Signal Date").reset_index(drop=True)

sm = pd.read_parquet(ROOT / "data" / "symbol_master.parquet")[["ticker", "sector"]]
sec = dict(zip(sm["ticker"], sm["sector"]))
ovr_path = ROOT / "data" / "sector_overrides.parquet"
if ovr_path.exists():
    ovr = pd.read_parquet(ovr_path)
    for r in ovr.itertuples():
        if isinstance(r.sector, str) and r.sector != "UNKNOWN":
            sec.setdefault(r.ticker, r.sector)
olv["sector"] = olv["Ticker"].map(sec).fillna("UNKNOWN")
cov = (olv["sector"] != "UNKNOWN").mean()
print(f"OLV N={len(olv)}  sector coverage={cov:.0%}  "
      f"missing tickers={sorted(olv.loc[olv['sector']=='UNKNOWN','Ticker'].unique())[:12]}")


def profile(df, label):
    if len(df) == 0:
        print(f"  {label:34s} N=  0   --"); return
    r = df["R_Multiple"]
    print(f"  {label:34s} N={len(df):4d}  win={ (r>0).mean():5.1%}  "
          f"avgR={r.mean():+.3f}  medR={r.median():+.3f}  totR={r.sum():+6.1f}  "
          f"avgRet%={df['Return_Pct'].mean():+.2f}")


# ---- (1) TEMPORAL: how many OLV signals share the signal date ----
same_day = olv.groupby("Signal Date")["trade_id"].transform("count")
olv["n_same_day"] = same_day

# windowed: OLV signals within +/-3 calendar days (approx +/-2 trading days)
dates = olv["Signal Date"].values.astype("datetime64[D]")
win = np.empty(len(olv), dtype=int)
for i, d in enumerate(dates):
    win[i] = int(np.sum(np.abs((dates - d).astype(int)) <= 3))
olv["n_win3d"] = win

print("\n" + "=" * 78)
print("(1) TEMPORAL CLUSTERING  -- signals firing together in time")
print("-- by same-day OLV signal count --")
profile(olv[olv["n_same_day"] == 1], "solo day (1 signal)")
profile(olv[olv["n_same_day"].between(2, 3)], "small cluster (2-3/day)")
profile(olv[olv["n_same_day"] >= 4], "big cluster (4+/day)")
print("-- by +/-3 calendar-day window count --")
profile(olv[olv["n_win3d"] <= 2], "isolated (<=2 in window)")
profile(olv[olv["n_win3d"].between(3, 6)], "moderate (3-6)")
profile(olv[olv["n_win3d"] >= 7], "dense (7+)")

# ---- (2) SECTOR CONCENTRATION ----
known = olv[olv["sector"] != "UNKNOWN"].copy().reset_index(drop=True)
sd = known["Signal Date"].values.astype("datetime64[D]")
ss = known["sector"].values
same_sec_win = np.empty(len(known), dtype=int)
for i in range(len(known)):
    mask = (np.abs((sd - sd[i]).astype(int)) <= 5) & (ss == ss[i])
    same_sec_win[i] = int(mask.sum())
known["n_samesector_win5d"] = same_sec_win

print("\n" + "=" * 78)
print("(2) SECTOR CONCENTRATION  -- same-sector OLV signals within +/-5 cal days")
profile(known[known["n_samesector_win5d"] == 1], "lone name in its sector")
profile(known[known["n_samesector_win5d"] == 2], "2 same-sector")
profile(known[known["n_samesector_win5d"] >= 3], "3+ same-sector (sector dogpile)")

print("\n-- per-sector edge (sectors with N>=10) --")
for s, g in known.groupby("sector"):
    if len(g) >= 10:
        profile(g, s)

# interaction: does temporal clustering coincide with broad (multi-sector) vs
# narrow (one-sector) pullbacks?
print("\n" + "=" * 78)
print("(3) BREADTH OF THE CLUSTER  -- distinct sectors firing on the same day")
nsec = known.groupby("Signal Date")["sector"].transform("nunique")
known["n_sectors_day"] = nsec
profile(known[known["n_sectors_day"] == 1], "single-sector day")
profile(known[known["n_sectors_day"].between(2, 3)], "2-3 sector day")
profile(known[known["n_sectors_day"] >= 4], "broad day (4+ sectors)")
