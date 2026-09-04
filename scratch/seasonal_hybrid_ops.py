"""Operational profile of the hybrid seasonal book: trades/day, concurrency, and
the macro instrument-type breakdown (IBKR tradeability + materiality)."""
import sys
import numpy as np
import pandas as pd

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"
sys.path.insert(0, ROOT)
from scripts.resim_seasonal_entry import resim

opn = resim("t1_open", 0)
lim = resim("limit", 0.25)
# hybrid: stock-long via limit, everything else via open; then V1 (no stock shorts)
hyb = pd.concat([lim[(lim.asset == "stock") & (lim.direction == "long")],
                 opn[~((opn.asset == "stock") & (opn.direction == "long"))]])
hyb = hyb[~((hyb.asset == "stock") & (hyb.direction == "short"))].copy()
hyb["entry_date"] = pd.to_datetime(hyb["entry_date"]); hyb["exit_date"] = pd.to_datetime(hyb["exit_date"])

yrs = (hyb["entry_date"].max() - hyb["entry_date"].min()).days / 365.0
print(f"HYBRID V1 book: {len(hyb)} trades over {yrs:.1f}y\n")

# --- trades per day (new entries) ---
per_day = hyb.groupby(hyb["entry_date"].dt.normalize()).size()
bdays = pd.bdate_range(hyb["entry_date"].min(), hyb["entry_date"].max())
per_day = per_day.reindex(bdays, fill_value=0)
print("=== new entries per trading day ===")
print(f"  mean {per_day.mean():.2f} | median {int(per_day.median())} | "
      f"p90 {int(per_day.quantile(.9))} | p99 {int(per_day.quantile(.99))} | max {int(per_day.max())}")
print(f"  days with >=1 entry: {100*(per_day>0).mean():.0f}% | with >=3: {100*(per_day>=3).mean():.0f}%")

# --- concurrency (open positions) ---
events = pd.Series(0, index=bdays)
conc = []
for d in bdays:
    conc.append(int(((hyb["entry_date"] <= d) & (hyb["exit_date"] >= d)).sum()))
conc = pd.Series(conc, index=bdays)
print(f"\n=== concurrent open positions ===")
print(f"  mean {conc.mean():.1f} | median {int(conc.median())} | p90 {int(conc.quantile(.9))} | max {int(conc.max())}")

# --- macro instrument-type breakdown (tradeability) ---
def kind(t):
    if t.endswith("=F"): return "future"
    if t.endswith("=X") or t == "DX-Y.NYB": return "fx"
    if t.endswith("-USD"): return "crypto"
    if t.startswith("^"): return "cash_index"
    return "etf"
macro = hyb[hyb.asset == "macro"].copy()
macro["kind"] = macro["ticker"].map(kind)
TRADEABLE = {"etf": "ETF — direct", "fx": "spot FX — direct (DX-Y needs proxy)",
             "future": "futures — tradeable, roll/contract-size caveats",
             "crypto": "crypto — limited (Paxos) / proxy",
             "cash_index": "CASH INDEX — NOT directly tradeable (need ETF/futures proxy)"}
g = macro.groupby("kind").agg(N=("R", "size"), TotR=("R", "sum"),
                              AvgR=("R", "mean"), tickers=("ticker", "nunique"))
g["%TotR_macro"] = 100 * g["TotR"] / macro["R"].sum()
g = g.reindex(["etf", "fx", "future", "crypto", "cash_index"]).dropna(how="all")
print(f"\n=== MACRO breakdown by instrument type (macro TotR = {macro['R'].sum():.0f}R) ===")
for k, r in g.iterrows():
    print(f"  {k:11s} N{int(r.N):4d} ({int(r.tickers)} tkrs) TotR{r.TotR:6.0f} "
          f"({r['%TotR_macro']:4.0f}% of macro)  -> {TRADEABLE[k]}")
print(f"\n  stock (equity longs) TotR {hyb[hyb.asset=='stock']['R'].sum():.0f}R "
      f"(fully tradeable) | macro TotR {macro['R'].sum():.0f}R")
trad = macro[macro.kind.isin(["etf", "fx", "future", "crypto"])]["R"].sum()
print(f"  macro that is directly tradeable (ETF/FX/futures/crypto): {trad:.0f}R "
      f"({100*trad/macro['R'].sum():.0f}% of macro)")
