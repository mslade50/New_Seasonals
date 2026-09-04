"""Probe: earnings calendar schema + coverage, ^MOVE premise, universe size."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import PRICES_PATH, load_events, pct_rank  # noqa: E402

ASOF = pd.Timestamp("2026-08-13")

e = pd.read_parquet("data/earnings_calendar.parquet")
print("earnings columns:", list(e.columns))
print("rows", len(e), "tickers", e["ticker"].nunique() if "ticker" in e else "?")
print(e.head(5).to_string())
dc = "date" if "date" in e.columns else e.columns[1]
e[dc] = pd.to_datetime(e[dc])
print("date range", e[dc].min(), e[dc].max())
print("\nper-year row counts (tail):")
print(e.groupby(e[dc].dt.year).size().tail(30).to_string())

print("\nTJX/ROST/NVDA history counts:")
for t in ["TJX", "ROST", "NVDA", "TGT", "WMT"]:
    s = e[e["ticker"] == t]
    print(f"  {t}: {len(s)} rows, {s[dc].min().date()} .. {s[dc].max().date()}")
    print("   next:", sorted(str(d.date()) for d in s[s[dc] >= "2026-08-01"][dc])[:4])

# price cache universe
mp = pd.read_parquet(PRICES_PATH, columns=["ticker", "date"])
tk = set(mp["ticker"].unique())
print("\nprice cache tickers:", len(tk))
et = set(e["ticker"].unique())
print("earnings tickers with prices:", len(et & tk))

# ^MOVE premise
mp2 = pd.read_parquet(PRICES_PATH)
for t in ["^MOVE", "TLT", "IEF", "LQD"]:
    g = mp2[mp2["ticker"] == t].copy()
    if g.empty:
        print(f"  {t}: MISSING")
        continue
    g["date"] = pd.to_datetime(g["date"])
    c = g.sort_values("date").set_index("date")["Close"].dropna()
    c = c[c.index <= ASOF]
    lvl = 100 * (c.iloc[-1] > c.iloc[-252:]).mean()
    print(f"  {t}: last {c.iloc[-1]:.2f} LEVEL pctile252 {lvl:.1f} "
          f"ret5rank {pct_rank(c,5).iloc[-1]:.1f} ret21rank {pct_rank(c,21).iloc[-1]:.1f} "
          f"off52wl {100*(c.iloc[-1]/c.iloc[-252:].min()-1):+.2f}% "
          f"hist since {c.index[0].date()}")

ev = load_events()
print("\nevents 2026-08-14 .. 2026-08-28:")
print(ev[(ev["date"] > "2026-08-14") & (ev["date"] <= "2026-08-28")].to_string(index=False))
