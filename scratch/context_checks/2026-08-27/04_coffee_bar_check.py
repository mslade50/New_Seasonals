"""KC=F printed -12.92% tonight. Believe the bar before quoting it.

Last night's brief named coffee as the cache's live data artefact: its Tuesday
session read -1.68% in that evening's cache where the previous evening's cache
had -11.36%. A second double-digit move in the same series two sessions later
is a plumbing suspect, not a nugget, until the bar is internally consistent.

Checks: close inside its own high/low, gap vs the prior close, volume, and
how the last week of bars compares with the neighbouring softs.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices, fwd_ret, summarize, sign_test  # noqa: E402

TK = ["KC=F", "SB=F", "CC=F", "CT=F"]
px = load_prices(TK)

for t in TK:
    d = px[t].tail(8)
    print(f"\n=== {t} last 8 bars ===")
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in d.columns]
    show = d[cols].copy()
    show["ret%"] = 100 * d["Close"].pct_change()
    print(show.to_string())

kc = px["KC=F"]
last = kc.iloc[-1]
prev = kc.iloc[-2]
print("\n=== KC=F integrity of the 2026-08-27 bar ===")
print(f"  date        {kc.index[-1].date()}")
print(f"  O/H/L/C     {last.get('Open'):.2f} / {last.get('High'):.2f} / "
      f"{last.get('Low'):.2f} / {last.get('Close'):.2f}")
print(f"  prior close {prev.get('Close'):.2f}")
print(f"  session ret {100*(last['Close']/prev['Close']-1):+.2f}%")
ok_hl = last["Low"] <= last["Close"] <= last["High"] and last["Low"] <= last["Open"] <= last["High"]
print(f"  close and open inside [low, high]? {ok_hl}")
print(f"  intraday range  {100*(last['High']/last['Low']-1):+.2f}%")
print(f"  gap open vs prior close {100*(last['Open']/prev['Close']-1):+.2f}%")
print(f"  move that happened INTRADAY (open->close) "
      f"{100*(last['Close']/last['Open']-1):+.2f}%")
if "Volume" in kc.columns:
    v63 = kc["Volume"].tail(64).iloc[:-1].median()
    print(f"  volume {last['Volume']:,.0f} vs 63d median {v63:,.0f} "
          f"= {last['Volume']/v63:.2f}x")

# does a bar this size have precedent in this series?
r = kc["Close"].pct_change().dropna()
big = r[r <= -0.10]
print(f"\n  sessions <= -10% in KC=F history ({r.index.min().date()}..): {len(big)}")
print("  most recent:", [f"{d.date()} {100*v:+.1f}%" for d, v in big.tail(8).items()])
print(f"  |ret| > 12% sessions: {int((r.abs() > 0.12).sum())}")

# the engine's cell, if the bar is real
c = kc["Close"].dropna()
r5 = c / c.shift(5) - 1.0
k5 = r5.rolling(252).rank(pct=True) * 100
m = (k5 <= 5).fillna(False)
idx = c.index[m]
v = fwd_ret(c, 1).reindex(idx).dropna()
s = summarize(v.values, "KC 5d bottom-5pct h1")
k = int((v.values > 0).sum())
print(f"\n  engine cell if real: n={s['n']} mean {s['mean_pct']:+.3f}% "
      f"median {s['median_pct']:+.3f}% record {k}-{s['n']-k} up "
      f"sign p {sign_test(k, s['n']):.4f} t {s['t']:+.2f}")
base = fwd_ret(c, 1).dropna()
print(f"  all-days baseline: {100*base.mean():+.3f}% hit {100*(base.values>0).mean():.1f}%")

# cross-check the last five sessions against the other softs
print("\n=== last 5 session returns, softs complex ===")
tbl = pd.DataFrame({t: px[t]["Close"].pct_change().tail(5) * 100 for t in TK})
print(tbl.round(2).to_string())
