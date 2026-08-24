"""b1c — C3 book overlap, measured (round 1's lookup used the wrong column name).

The registry's standing warning: "the book is on the other side of thrust
states" (2026-08-19, 2026-08-20). This morning the scanner staged 11 Overbot
Vol Spike SHORTS including four miners. So: on the historical C3 trigger days,
what did the book actually do in the metals/mining complex, and in FCX itself?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, rolling_on_valid, summarize, show  # noqa: E402
from pitch_lab import _valid_pct_change as vpc  # noqa: E402

pd.set_option("display.width", 220)

px = close_panel(["FCX", "COPX", "XME", "XLB", "SCCO", "TECK"])
r5 = vpc(px["FCX"], 5)
hi = rolling_on_valid(px["FCX"], lambda x: x.rolling(252).max())
m15 = ((r5 >= 0.15) & (px["FCX"] >= hi * (1 - 1e-9))).fillna(False)
m10 = ((r5 >= 0.10) & (px["FCX"] >= hi * (1 - 1e-9))).fillna(False)

led = pd.read_parquet("data/backtest_trades_full.parquet")
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
print(f"ledger: {len(led)} trades, {led['Signal Date'].min().date()} .. "
      f"{led['Signal Date'].max().date()}")

METALS = {"FCX", "NEM", "SCCO", "TECK", "XME", "COPX", "AA", "NUE", "STLD",
          "CLF", "RIO", "BHP", "VALE", "GLD", "SLV", "GDX", "XLB", "APD",
          "ECL", "LIN", "PPG", "SHW", "VMC"}
pos = pd.Series(range(len(px.index)), index=px.index)

for lbl, m in (("r5>=15% & fresh high", m15), ("r5>=10% & fresh high", m10)):
    d = px.index[m.values]
    win = set()
    for x in d:
        p = pos[x]
        win |= set(px.index[p:min(p + 4, len(px.index))])
    print(f"\n=== {lbl}: {len(d)} trigger days, {len(win)} session-days in a "
          f"0..3 td window ===")
    hit = led[led["Signal Date"].isin(win)]
    met = hit[hit["Ticker"].isin(METALS)]
    fcx = hit[hit["Ticker"] == "FCX"]
    print(f"  ALL book trades in the window: {len(hit)}, "
          f"{hit['Direction'].value_counts().to_dict()}, "
          f"avgR {hit['R_Multiple'].mean():.3f}")
    print(f"  METALS/MATERIALS trades in the window: {len(met)}, "
          f"{met['Direction'].value_counts().to_dict()}, "
          f"avgR {met['R_Multiple'].mean() if len(met) else float('nan'):.3f}, "
          f"flat PnL ${met['PnL_flat_750k'].sum():,.0f}")
    if len(met):
        print(met.groupby(["Strategy", "Direction"])
              .agg(n=("R_Multiple", "size"), avgR=("R_Multiple", "mean"),
                   pnl=("PnL_flat_750k", "sum")).round(3).to_string())
    print(f"  FCX itself: {len(fcx)} trades "
          f"{fcx['Direction'].value_counts().to_dict()}")
    if len(fcx):
        print(fcx[["Signal Date", "Strategy", "Direction", "R_Multiple",
                   "PnL_flat_750k"]].to_string(index=False))

# what does the book do on FCX in ANY state, for the base rate
f = led[led["Ticker"] == "FCX"]
print(f"\n=== FCX in the whole 23y ledger: {len(f)} trades ===")
if len(f):
    print(f.groupby(["Strategy", "Direction"])
          .agg(n=("R_Multiple", "size"), avgR=("R_Multiple", "mean")).to_string())
    print(f[["Signal Date", "Strategy", "Direction", "R_Multiple"]].to_string(index=False))

# and the state's own direction bias in the book: high-thrust names generally
print("\n=== the book's direction on ANY name that thrust >=15% in 5d to a "
      "fresh 52w high (the reference-class state, book-wide) ===")
mp = pd.read_parquet("data/master_prices.parquet",
                     columns=["ticker", "date", "Close"])
mp["date"] = pd.to_datetime(mp["date"])
keys = set()
for t, g in mp.groupby("ticker"):
    s = g.sort_values("date").set_index("date")["Close"]
    s = s[~s.index.duplicated(keep="last")]
    if len(s) < 300:
        continue
    hh = s.rolling(252).max()
    mm = (s.pct_change(5) >= 0.15) & (s >= hh * (1 - 1e-9))
    for dte in s.index[mm.fillna(False).values]:
        keys.add((t, dte))
led_keys = set(zip(led["Ticker"], led["Signal Date"]))
inter = keys & led_keys
sub = led[[k in inter for k in zip(led["Ticker"], led["Signal Date"])]]
print(f"  name-days in the state: {len(keys)};  book trades signalled on one: "
      f"{len(sub)}")
if len(sub):
    print(f"  direction: {sub['Direction'].value_counts().to_dict()}  "
          f"avgR {sub['R_Multiple'].mean():.3f}  "
          f"flat PnL ${sub['PnL_flat_750k'].sum():,.0f}")
    print(sub.groupby(["Strategy", "Direction"])
          .agg(n=("R_Multiple", "size"), avgR=("R_Multiple", "mean"),
               pnl=("PnL_flat_750k", "sum")).round(3).to_string())
