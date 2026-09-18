import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

TK = ["SPY", "^VIX", "SVXY", "UVXY", "VIXY", "^VIX3M", "IWM", "QQQ"]
raw = close_panel(TK)
for t in TK:
    if t in raw:
        s = raw[t].dropna()
        print(f"{t:7s} first {s.index[0].date()} last {s.index[-1].date()} n {len(s)}")
print(raw.tail(8).round(3).to_string())
ev = load_events()
print(ev["event"].value_counts().to_string())
for k in ["opex", "quad_witching", "fomc_decision", "vix_expiry", "nfp", "cpi", "ppi"]:
    e = ev[ev.event == k]["date"]
    print(k, e.min().date(), e.max().date(), len(e))
e = ev[(ev.date >= "2026-08-01") & (ev.date <= "2026-10-31")]
print(e.to_string())
print(ev.columns.tolist())
