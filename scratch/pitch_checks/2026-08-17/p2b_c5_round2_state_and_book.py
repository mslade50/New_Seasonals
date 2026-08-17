"""C5 round 2 - (i) is today's JOINT state anywhere in the sample, (ii) is a
long XLE at a 52w high materially a STRATEGY_BOOK trade (the anti-rip-off
rule), (iii) loser paths and the expiry-week tail.

Round 1 (p2_c5_energy_thrust_high.py) already showed:
  * rank form +0.715% on 11 episodes but both gate components are NEGATIVE
    alone (thrust alone -0.313%, near-high+63d-mid alone -0.298%)
  * the magnitude form - which is what today's +3.49 ATR move actually is -
    pays -0.064% at h=5 and has a NEGATIVE edge at every horizon 1..10
  * the thrust trigger and the 08-14 divergence trigger are DISJOINT on
    history (0 of 11 episodes carry a divergence >= 18pp; today is +18.85pp)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ROOT3 = Path(__file__).resolve().parents[3]
pan = close_panel(["XLE", "XOP", "USO", "SPY"]).dropna(subset=["XLE", "SPY"])
IDX = pan.index
xle = pan["XLE"]
raw = load_prices(["XLE"])["XLE"].reindex(IDX)
atr = pd.Series(np.asarray(wilder_atr(raw["High"], raw["Low"], raw["Close"], 14), float), index=IDX)

r5 = pct_rank(xle, 5)
r63 = pct_rank(xle, 63)
off_hi = xle / xle.rolling(252).max() - 1.0
mv5_atr = (xle - xle.shift(5)) / atr
uso63r = pct_rank(pan["USO"], 63)
NEAR_HI = off_hi >= -0.01
MID63 = (r63 >= 30) & (r63 <= 75)

print("=" * 92)
print("1. THE JOINT-STATE LADDER - how much of today's state has ever co-occurred?")
print("=" * 92)
f5 = fwd_lag(xle, 5, 1)
steps = [
    ("thrust rank5>=98", (r5 >= 98)),
    ("+ near 52w high (<=1%)", (r5 >= 98) & NEAR_HI),
    ("+ 63d rank mid 30-75", (r5 >= 98) & NEAR_HI & MID63),
    ("+ crude 63d rank < 15 (today 8.3)", (r5 >= 98) & NEAR_HI & MID63 & (uso63r < 15)),
    ("+ crude 63d rank < 25", (r5 >= 98) & NEAR_HI & MID63 & (uso63r < 25)),
    ("MAG>=3.0 ATR + near high + mid63 + crude<25",
     (mv5_atr >= 3.0) & NEAR_HI & MID63 & (uso63r < 25)),
]
rows = []
for lbl, m in steps:
    m = m.fillna(False)
    s = IDX[m.values]
    e = declusters(s, 10, IDX) if len(s) else pd.DatetimeIndex([])
    v = f5.reindex(e).dropna() if len(e) else pd.Series(dtype=float)
    rows.append({"state": lbl, "n_days": len(s), "n_epi": len(v),
                 "mean_pct": round(100 * v.mean(), 3) if len(v) else np.nan,
                 "hit": round(100 * (v > 0).mean(), 1) if len(v) else np.nan,
                 "last_day": str(s[-1].date()) if len(s) else "-"})
show(rows, "each added condition, h=5 episodes")

print("\n" + "=" * 92)
print("2. ANTI-RIP-OFF - is LONG XLE near a 52w high materially a book trade?")
print("=" * 92)
from strategy_config import STRATEGY_BOOK  # noqa: E402
for cfg in STRATEGY_BOOK:
    if cfg.get("name") in ("52wh Breakout", "Sector BO", "Overbot Vol Spike",
                           "ATR Extended Gap Up", "LT Trend ST OS"):
        st = cfg.get("settings", {})
        print(f"\n{cfg['name']}  dir={st.get('trade_direction')}  hold="
              f"{cfg.get('execution',{}).get('hold_days')}  XLE in universe="
              f"{'XLE' in cfg.get('universe_tickers', [])}")
        print("   id:", cfg.get("id"))
led = pd.read_parquet(ROOT3 / "data" / "backtest_trades_full.parquet")
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
ENERGY = {"XLE", "XOP", "OIH", "USO", "XOM", "CVX", "COP", "SLB", "HAL", "VLO",
          "OXY", "EOG", "WMB", "XME"}
sub = led[led["Ticker"].isin(ENERGY)]
print("\nledger, energy tickers, by strategy x direction:")
print(sub.groupby(["Strategy", "Direction"]).agg(
    n=("R_Multiple", "size"), avgR=("R_Multiple", "mean"),
    totR=("R_Multiple", "sum")).round(3).to_string())

# how often does a LONG book strategy fire on an energy name that is within 1%
# of its own 52w high? (the state this pitch would be buying)
hi_map = {}
for t in ("XLE", "XOP"):
    s = pan[t].dropna()
    hi_map[t] = (s / s.rolling(252).max() - 1.0)
longs = sub[(sub["Direction"] == "Long") & sub["Ticker"].isin(hi_map)]
cnt = 0
for _, r in longs.iterrows():
    o = hi_map[r["Ticker"]].reindex([r["Signal Date"]]).iloc[0]
    if pd.notna(o) and o >= -0.01:
        cnt += 1
print(f"\nlong book trades on XLE/XOP entered within 1% of the ticker's 52w high: "
      f"{cnt} of {len(longs)}")

print("\n" + "=" * 92)
print("3. LOSER PATHS + THE EXPIRY-WEEK TAIL (today's 5td hold holds VIX expiry")
print("   08-19 and Aug opex 08-21)")
print("=" * 92)
RANK = ((r5 >= 98) & NEAR_HI & MID63).fillna(False)
epi = declusters(IDX[RANK.values], 10, IDX)
paths = episode_paths(pan, epi, [("XLE", 1.0)], 5)
print((100 * paths).round(2).to_string())
v = f5.reindex(paths.index)
los = paths[v.values < 0]
if len(los):
    print(f"\nlosing episodes N={len(los)}; mean path by day: "
          f"{[round(100*x, 2) for x in los.mean().values]}")
    print(f"worst episode {los.min(axis=1).idxmin().date()} trough "
          f"{100*los.min().min():.2f}%")
win = paths[v.values >= 0]
if len(win):
    print(f"winning episodes N={len(win)}; mean path by day: "
          f"{[round(100*x, 2) for x in win.mean().values]}")

# expiry-week specific: entries whose 5td hold contains an opex or vix_expiry
fl = event_in_window(epi, IDX, 5, 1, ("opex", "vix_expiry"))
show([summarize(f5.reindex(epi).values[fl], "RANK episodes, expiry INSIDE hold"),
      summarize(f5.reindex(epi).values[~fl], "RANK episodes, no expiry")],
     "expiry-in-hold split, rank form")
MAG = ((mv5_atr >= 2.5) & NEAR_HI & MID63).fillna(False)
epim = declusters(IDX[MAG.values], 10, IDX)
flm = event_in_window(epim, IDX, 5, 1, ("opex", "vix_expiry"))
show([summarize(f5.reindex(epim).values[flm], "MAG episodes, expiry INSIDE hold"),
      summarize(f5.reindex(epim).values[~flm], "MAG episodes, no expiry")],
     "expiry-in-hold split, magnitude form (the form today's move belongs to)")
