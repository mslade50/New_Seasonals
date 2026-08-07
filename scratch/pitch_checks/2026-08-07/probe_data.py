"""Probe: what tickers/dates/events do we actually have? Read-only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa

import pandas as pd

mp = pd.read_parquet(PRICES, columns=["ticker", "date"])
mp["date"] = pd.to_datetime(mp["date"])
want = ["SPY", "XLU", "XLP", "TLT", "IEF", "^TNX", "TNX", "^IRX", "SHY", "LQD", "AGG"]
have = sorted(set(want) & set(mp["ticker"].unique()))
print("PRESENT:", have)
print("MISSING:", sorted(set(want) - set(have)))

for t in have:
    g = mp[mp["ticker"] == t]
    print(f"{t:6s} n={len(g):6d}  {g['date'].min().date()} .. {g['date'].max().date()}")

ev = load_events()
print("\nEVENT KINDS:", sorted(ev["event"].unique()))
print(ev.head(3).to_string(index=False))
cpi = load_events(["cpi"])
print(f"\ncpi rows={len(cpi)}  {cpi['date'].min().date()} .. {cpi['date'].max().date()}")
print(cpi.tail(4).to_string(index=False))
