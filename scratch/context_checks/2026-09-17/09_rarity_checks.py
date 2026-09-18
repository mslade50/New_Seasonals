"""Two rarity facts from today's tape, priced properly.

(a) All seven dollar pairs closed with a 5-day return in the top 5% of their
    own year. 07b says that has happened on 9 sessions in 27 years. Name them,
    and check whether the forward tape is a null (it looked like one).
(b) The 3-month, 5-year and 10-year yields are all within 1.2% of 52-week
    highs on the same session, two days after an FOMC decision, with the S&P
    2.1% off its own high. How often is the whole curve pinned at highs while
    equities sit near theirs?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, summarize, show, sign_test, pct_rank, declusters,
)

PAIRS = ["CHF=X", "USDSEK=X", "USDNOK=X", "CAD=X", "USDSGD=X", "JPY=X", "USDMXN=X"]
px = close_panel(["DX-Y.NYB", "^IRX", "^FVX", "^TNX", "SPY", "IWM"] + PAIRS)

# --- (a) -----------------------------------------------------------------
counts = None
for p in PAIRS:
    c = (pct_rank(px[p].dropna(), 5, 252) >= 95).astype(float).reindex(px.index)
    counts = c if counts is None else counts.add(c, fill_value=0)
counts = counts.dropna()
seven = counts.index[(counts == 7).values]
print(f"(a) all-7 sessions: {len(seven)}")
print("   ", [str(d.date()) for d in seven])
dxy = px["DX-Y.NYB"].dropna()
for h in (1, 5, 10):
    f = dxy.shift(-h) / dxy - 1.0
    d = seven.intersection(f.dropna().index)
    k = int((f.loc[d] > 0).sum())
    print(f"    DXY h={h}: n={len(d)} mean {100 * f.loc[d].mean():+.3f}% "
          f"{k}-{len(d) - k} up  (ctl {100 * f.dropna().mean():+.3f}%)")
spy = px["SPY"].dropna()
for h in (1, 5):
    f = spy.shift(-h) / spy - 1.0
    d = seven.intersection(f.dropna().index)
    k = int((f.loc[d] > 0).sum())
    print(f"    SPY h={h}: n={len(d)} mean {100 * f.loc[d].mean():+.3f}% "
          f"{k}-{len(d) - k} up  (ctl {100 * f.dropna().mean():+.3f}%)")

# --- (b) -----------------------------------------------------------------
print("\n(b) the whole curve at 52-week highs")
ten = [px[t].dropna() for t in ["^IRX", "^FVX", "^TNX"]]
common = ten[0].index
for s in ten[1:]:
    common = common.intersection(s.index)
near = None
for s, name in zip(ten, ["^IRX", "^FVX", "^TNX"]):
    s = s.reindex(common)
    d = s / s.rolling(252).max() - 1.0
    print(f"    {name}: today {100 * d.iloc[-1]:+.2f}% from its 252d max")
    m = (d >= -0.015)
    near = m.astype(float) if near is None else near + m.astype(float)
allthree = common[(near == 3).values]
print(f"    sessions with all three within 1.5% of a 52w high: {len(allthree)} "
      f"of {len(common)}")
epi = declusters(allthree, 21, common)
print(f"    declustered (21td): {len(epi)}, by year "
      f"{dict(pd.Series(1, index=epi).groupby(epi.year).sum())}")

sp = px["SPY"].dropna().reindex(common)
spd = sp / sp.rolling(252).max() - 1.0
print(f"    SPY today {100 * spd.iloc[-1]:+.2f}% from its own 252d max")
both = allthree[(spd.reindex(allthree) >= -0.03).values]
print(f"    ... and with SPY within 3% of ITS 52w high: {len(both)} sessions, "
      f"by year {dict(pd.Series(1, index=both).groupby(both.year).sum())}")

for tkr in ["SPY", "IWM"]:
    s = px[tkr].dropna()
    for h in (5, 21):
        f = s.shift(-h) / s - 1.0
        e = declusters(both, 21, common).intersection(f.dropna().index)
        if len(e) < 3:
            print(f"    {tkr} h={h}: n={len(e)} too few")
            continue
        k = int((f.loc[e] > 0).sum())
        print(f"    {tkr} h={h}: n={len(e)} mean {100 * f.loc[e].mean():+.2f}% "
              f"{k}-{len(e) - k} up (ctl {100 * f.dropna().mean():+.2f}%) "
              f"sign p(down) = {sign_test(len(e) - k, len(e)):.4f}")
