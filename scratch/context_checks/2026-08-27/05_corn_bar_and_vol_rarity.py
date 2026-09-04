"""Two loose ends before anything publishes.

(a) Drill 04 showed the whole softs complex rolled contracts tonight, so corn
    has to clear the same bar-integrity test before it can be called a real
    move. Gap vs intraday, volume, and the rest of the grains.
(b) Drill 02 found SPY's 21d rank is >= 90 on only 4.8% of ^VIX3M 52-week-low
    sessions. Name those sessions: vol on the floor while the tape is in the
    top decile of its own year is the actual state going into tomorrow.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    load_prices, close_panel, load_events, fwd_ret, summarize,
    sign_test, rolling_on_valid,
)

# ---------------------------------------------------------------- (a) corn
GRAINS = ["ZC=F", "ZW=F", "ZS=F"]
px = load_prices(GRAINS)
print("=== grains, last 6 bars ===")
for t in GRAINS:
    d = px[t].tail(6)
    print(f"\n{t}")
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in d.columns]
    s = d[cols].copy()
    s["ret%"] = (100 * d["Close"].pct_change()).round(2)
    s["gap%"] = (100 * (d["Open"] / d["Close"].shift(1) - 1)).round(2)
    s["intraday%"] = (100 * (d["Close"] / d["Open"] - 1)).round(2)
    print(s.to_string())

zc = px["ZC=F"]
last, prev = zc.iloc[-1], zc.iloc[-2]
gap = 100 * (last["Open"] / prev["Close"] - 1)
intra = 100 * (last["Close"] / last["Open"] - 1)
tot = 100 * (last["Close"] / prev["Close"] - 1)
print("\n=== ZC=F 2026-08-27 bar ===")
print(f"  O/H/L/C {last['Open']:.2f} / {last['High']:.2f} / "
      f"{last['Low']:.2f} / {last['Close']:.2f}, prior close {prev['Close']:.2f}")
print(f"  session {tot:+.2f}% = gap {gap:+.2f}% + intraday {intra:+.2f}%")
print(f"  intraday share of the move: {abs(intra)/(abs(gap)+abs(intra))*100:.0f}%")
print(f"  close vs own high: {100*(last['Close']/last['High']-1):+.2f}%")
if "Volume" in zc.columns:
    v63 = zc["Volume"].tail(64).iloc[:-1].median()
    print(f"  volume {last['Volume']:,.0f} vs 63d median {v63:,.0f} = "
          f"{last['Volume']/v63:.2f}x")
    print(f"  prior 5 volumes: {[int(v) for v in zc['Volume'].tail(6).iloc[:-1]]}")
    dup = zc["Volume"].tail(6).iloc[:-1]
    print(f"  any repeated volume in the prior 5 (stale-bar tell)? "
          f"{bool(dup.duplicated().any())}")

# ------------------------------------------------------------ (b) vol floor
TK = ["^VIX3M", "^VIX", "SPY"]
p = close_panel(TK)
dates = p.index
lo252 = rolling_on_valid(p["^VIX3M"], lambda x: x.rolling(252).min())
at_low = ((p["^VIX3M"] <= lo252 * 1.0000001) & lo252.notna()).fillna(False)
low_days = dates[at_low]

spy_r21 = rolling_on_valid(p["SPY"] / p["SPY"].shift(21) - 1.0,
                           lambda x: x.rolling(252).rank(pct=True)) * 100
sel = spy_r21.reindex(low_days).dropna()
strong = sel[sel >= 90]
print(f"\n=== ^VIX3M at a 52w low AND SPY 21d rank >= 90 ===")
print(f"  ^VIX3M 52w-low sessions: {len(sel)}")
print(f"  of those with SPY 21d rank >= 90: {len(strong)} "
      f"({100*len(strong)/len(sel):.1f}%)")
print(f"  tonight: SPY 21d rank {spy_r21.iloc[-1]:.1f}")
print("  the sessions:")
f1 = fwd_ret(p["SPY"], 1)
f5 = fwd_ret(p["SPY"], 5)
f21 = fwd_ret(p["SPY"], 21)
fv = fwd_ret(p["^VIX"], 1)
for d in strong.index:
    print(f"    {d.date()}  SPYrank {strong[d]:5.1f}  "
          f"SPY h1 {100*f1.get(d, np.nan):+6.2f}%  h5 {100*f5.get(d, np.nan):+6.2f}%  "
          f"h21 {100*f21.get(d, np.nan):+6.2f}%  VIX h1 {100*fv.get(d, np.nan):+6.2f}%")
if len(strong) > 1:
    v = f1.reindex(strong.index).dropna()
    s = summarize(v.values, "SPY h1")
    k = int((v.values > 0).sum())
    print(f"  SPY h1: n={s['n']} mean {s['mean_pct']:+.3f}% "
          f"record {k}-{s['n']-k} up sign p {sign_test(k, s['n']):.4f}")
    v21 = f21.reindex(strong.index).dropna()
    if len(v21):
        k21 = int((v21.values > 0).sum())
        print(f"  SPY h21: n={len(v21)} mean {100*v21.mean():+.3f}% "
              f"record {k21}-{len(v21)-k21} up")

# distinct EPISODES rather than sessions
gaps = np.diff([dates.get_loc(d) for d in strong.index]) if len(strong) > 1 else []
eps = 1 + int((np.array(gaps) > 10).sum()) if len(strong) else 0
print(f"  distinct episodes (10td gap): {eps}")
