"""C3 recon: is data/earnings_calendar.parquet usable as an ANCHOR calendar?

First use of the earnings calendar as a pitch anchor in this repo, so the
schema and the date semantics get inspected before any forward return is
read. The AA head rows (2015-03-31, 2015-06-30, 2015-09-30, 2015-12-31) are
quarter ENDS, not announcement dates, which would silently mis-anchor the
whole pre-print window.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

E = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet")
E["date"] = pd.to_datetime(E["date"])
print("rows", len(E), "tickers", E["ticker"].nunique(),
      "span", E["date"].min().date(), "->", E["date"].max().date())

# ---- trap 1: quarter-end dates masquerading as report dates -------------
E["is_qend"] = E["date"].dt.is_quarter_end
E["yr"] = E["date"].dt.year
qe = E.groupby("yr")["is_qend"].agg(["sum", "size"])
qe["share_pct"] = 100 * qe["sum"] / qe["size"]
print("\n=== share of rows landing exactly on a quarter END, by year ===")
print(qe.to_string())

# ---- trap 2: weekend dates ---------------------------------------------
E["dow"] = E["date"].dt.dayofweek
wk = E.groupby("yr")["dow"].apply(lambda s: 100 * (s >= 5).mean())
print("\n=== share of rows on a WEEKEND, by year ===")
print(wk.round(2).to_string())

# ---- trap 3: how many distinct dates per ticker-year (dupes) ------------
print("\n=== rows per ticker per year (should be ~4) ===")
per = E.groupby(["ticker", "yr"]).size().groupby("yr").mean()
print(per.round(2).to_string())

# ---- AVGO specifically --------------------------------------------------
a = E[E["ticker"] == "AVGO"].sort_values("date")
print(f"\n=== AVGO: {len(a)} rows ===")
print(a[["date", "eps_actual", "eps_est", "eps_surprise_pct"]].to_string(index=False))

# ---- price coverage -----------------------------------------------------
mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                     columns=["ticker", "date"])
have = set(mp["ticker"].unique())
etk = set(E["ticker"].unique())
print(f"\nprice cache tickers {len(have)}, earnings tickers {len(etk)}, "
      f"intersection {len(have & etk)}")
print("AVGO in prices:", "AVGO" in have, " SMH in prices:", "SMH" in have)

px_a = load_prices(["AVGO", "SMH", "SPY"])
for t in ["AVGO", "SMH", "SPY"]:
    s = px_a[t]["Close"].dropna()
    print(f"  {t:5s} bars {len(s)}  {s.index[0].date()} .. {s.index[-1].date()}  "
          f"last {s.iloc[-1]:.2f}")

# ---- live state assertion ----------------------------------------------
print("\n=== live state (must match the stated tape) ===")
for t in ["AVGO", "SMH", "SPY"]:
    s = px_a[t]["Close"].dropna()
    print(f"  {t:5s} r63 rank {pct_rank(s,63,252).iloc[-1]:5.1f}  "
          f"r21 rank {pct_rank(s,21,252).iloc[-1]:5.1f}  "
          f"63d ret {100*(s.iloc[-1]/s.iloc[-64]-1):+.2f}%  "
          f"252d ret {100*(s.iloc[-1]/s.iloc[-253]-1):+.2f}%")

# ---- the live anchor ----------------------------------------------------
idx = px_a["AVGO"]["Close"].dropna().index
print("\nlast 5 AVGO sessions:", [str(d.date()) for d in idx[-5:]])
fwd = a[a["date"] > pd.Timestamp("2026-08-28")]
print("AVGO forward report rows:", fwd[["date", "eps_est"]].to_string(index=False))
