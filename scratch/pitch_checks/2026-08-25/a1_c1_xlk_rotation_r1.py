"""C1 round 1 - long XLK outright after a 99.6th-pctile 5-day XLV-minus-XLK rotation.

Order of business, deliberately front-loading the two things that can kill it
before any battery is run:

  0. RUNG POPULATION.  How many days ever sit at the literal state?  (The
     2026-08-24 XLI kill: the pitched rung had 0 days ever.)
  1. MASK OVERLAP vs W15, the ONE-DAY form killed on 2026-08-19 for
     concentration.  Reported in BOTH directions, day-level and episode-level.
     The 2026-08-24 ^TNX lesson: 91% overlap = same object, inherits the dead
     cell's search charge.
  2. Only then the rung ladder + battery + horizon scan.

Definition note (registry, 2026-08-19): pct_rank() computes the trailing rank
of an n-day PERCENT CHANGE, so calling it on a spread that crosses zero is
meaningless.  Everything here ranks the SPREAD LEVEL directly, both PIT
(trailing 252) and full-sample, and the raw pp rung ladder is carried
alongside so the finding cannot be an artifact of one percentile definition.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 240)

TK = ["XLK", "XLV", "XLP", "SPY", "QQQ", "SMH", "XLI"]
px = close_panel(TK)
print(f"panel {px.index[0].date()} -> {px.index[-1].date()}  {px.shape}")

r5 = px.pct_change(5)
r1 = px.pct_change(1)
SPREAD = (r5["XLV"] - r5["XLK"]) * 100.0          # pp, 5-day
GAP1 = (r1["XLV"] - r1["XLK"]) * 100.0            # pp, 1-day  (the W15 object)

today = SPREAD.dropna().iloc[-1]
g1_today = GAP1.dropna().iloc[-1]
print(f"\ntoday: 5d XLV-XLK spread = {today:+.2f}pp   1d gap = {g1_today:+.2f}pp")

v = SPREAD.dropna()
print(f"  full-sample pctile of the 5d spread = {(v < today).mean()*100:.2f}  (N={len(v)})")
pit252 = SPREAD.dropna().rolling(252).apply(lambda w: (w[:-1] < w[-1]).mean() * 100,
                                            raw=True).reindex(px.index)
print(f"  PIT trailing-252 pctile             = {pit252.dropna().iloc[-1]:.1f}")

# ---------------------------------------------------------------- 0. population
print("\n########## 0. RUNG POPULATION - how many days ever sit at the state ##########")
for rung in (5, 6, 7, 8, 9, 9.98, 10, 11):
    m = (SPREAD >= rung).fillna(False)
    n = int(m.sum())
    yrs = sorted(set(px.index[m.values].year))
    print(f"  5d spread >= {rung:>5.2f}pp : {n:>4d} days   years {yrs[:3]}..{yrs[-3:] if yrs else []}"
          f"   ({len(yrs)} distinct years)")
m_pit = (pit252 >= 97.6).fillna(False)
print(f"  PIT252 pctile >= 97.6      : {int(m_pit.sum()):>4d} days")

# ---------------------------------------------------------------- 1. W15 overlap
print("\n########## 1. MASK OVERLAP WITH W15 (the dead one-day form) ##########")
hi252 = rolling_on_valid(px["SPY"], lambda x: x.rolling(252).max())
dist = px["SPY"] / hi252 - 1.0
raw = load_prices(["SPY"])
spy_atrp = (pd.Series(wilder_atr(raw["SPY"]["High"], raw["SPY"]["Low"],
                                 raw["SPY"]["Close"], 14),
                      index=raw["SPY"].index).reindex(px.index) / px["SPY"])
W15_BARE = (GAP1 >= 3.0).fillna(False)
W15_SUB = (W15_BARE & (dist > -0.03) & (spy_atrp < 0.012)).fillna(False)


def overlap(a: pd.Series, b: pd.Series, la: str, lb: str) -> None:
    A = set(px.index[a.values])
    B = set(px.index[b.values])
    if not A or not B:
        print(f"  {la} n={len(A)}  {lb} n={len(B)}  (one side empty)")
        return
    inter = A & B
    print(f"  {la:<28} n={len(A):>4d} | {lb:<24} n={len(B):>4d} | "
          f"shared {len(inter):>3d}  -> {len(inter)/len(A)*100:5.1f}% of {la}, "
          f"{len(inter)/len(B)*100:5.1f}% of {lb}")


for rung in (6, 8, 9, 10):
    C1 = (SPREAD >= rung).fillna(False)
    overlap(C1, W15_BARE, f"C1 5d>={rung}pp", "W15 bare 1d>=3pp")
    overlap(C1, W15_SUB, f"C1 5d>={rung}pp", "W15 full subclass")

# episode-level: does each C1 day CONTAIN a >=3pp one-day gap in its 5-day window?
print("\n  episode-level containment (a 5d spread is 5 one-day gaps):")
for rung in (6, 8, 9, 10):
    C1 = (SPREAD >= rung).fillna(False)
    days = px.index[C1.values]
    if len(days) == 0:
        continue
    contains = []
    for d in days:
        p = px.index.get_loc(d)
        win = GAP1.iloc[max(0, p - 4):p + 1]
        contains.append(bool((win >= 3.0).any()))
    contains = np.array(contains)
    # also: how much of the 5d spread is the single biggest day?
    frac = []
    for d in days:
        p = px.index.get_loc(d)
        win = GAP1.iloc[max(0, p - 4):p + 1].values
        tot = SPREAD.loc[d]
        frac.append(np.nanmax(win) / tot if tot != 0 else np.nan)
    print(f"    5d>={rung}pp: {contains.mean()*100:5.1f}% of days contain a >=3pp single-day gap"
          f"   | biggest single day = {np.nanmedian(frac)*100:.0f}% of the 5d spread (median)")
p = px.index.get_loc(px.index[-1])
win_today = GAP1.iloc[p - 4:p + 1]
print(f"    TODAY's 5-day window of 1d gaps (pp): "
      f"{[round(x, 2) for x in win_today.values]}  max={win_today.max():.2f}")

# ---------------------------------------------------------------- 2. battery
print("\n########## 2. BATTERY: long XLK outright, rung ladder ##########")
MAIN_RUNG = 8.0
MAIN = (SPREAD >= MAIN_RUNG).fillna(False)
variants = {f">={k}pp": (SPREAD >= k).fillna(False) for k in (5, 6, 7, 8, 9, 10)}
variants["PIT252>=97.6"] = m_pit
variants["fullpct>=99.0"] = (SPREAD >= v.quantile(0.99)).fillna(False)

for h in (3, 5):
    battery(px, MAIN, [("XLK", 1.0)], h,
            f"C1 long XLK | 5d XLV-XLK >= {MAIN_RUNG}pp", cost_bps=5.0,
            variants=variants if h == 5 else None)

print("\n########## 3. HORIZON SCAN (episode level) ##########")
for rung in (6, 8, 10):
    m = (SPREAD >= rung).fillna(False)
    d = px.index[m.values]
    print(f"\n-- rung >= {rung}pp, long XLK outright --")
    show(horizon_scan(px, d, [("XLK", 1.0)], hs=(1, 2, 3, 5, 7, 10)), "")
    print(f"-- rung >= {rung}pp, PAIR long XLK / short XLV (W15 says the naked long wins) --")
    show(horizon_scan(px, d, [("XLK", 1.0), ("XLV", -1.0)], hs=(1, 2, 3, 5, 7, 10)), "")

print("\n########## 4. TAPE OVER-SELECTION (200d) ##########")
sma200 = rolling_on_valid(px["SPY"], lambda x: x.rolling(200).mean())
above = (px["SPY"] > sma200)
base = above.dropna().mean() * 100
for rung in (6, 8, 10):
    m = (SPREAD >= rung).fillna(False) & above.notna()
    d = px.index[m.values]
    print(f"  rung>={rung}pp: {above.loc[d].mean()*100:5.1f}% of trigger days above SPY 200d "
          f"(base {base:.1f}%)  N={len(d)}")
print(f"  today: SPY {'ABOVE' if above.iloc[-1] else 'BELOW'} its 200d "
      f"(+{(px['SPY'].iloc[-1]/sma200.iloc[-1]-1)*100:.1f}%)")
