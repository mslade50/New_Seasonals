"""C1 round 1 -- SPY within 1% of its trailing-252 high WHILE its 63d return
rank sits in the bottom quartile. "The round-trip breakout."

Kills this script is built to find:
  - the 2026-08-14 registry cell in a costume (near-high + VIX bottom decile,
    which died on CTRL-c). Overlap in days + incremental edge measured here.
  - the 2026-08-19 registry note that a 63d RANK is not a 63d MOVE, so the
    return-LEVEL form is run beside the rank form.
  - the 2026-08-10 note that confirming legs do not create a state: gate
    attribution runs each leg alone.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pitch_lab import *  # noqa: E402,F403

ASOF = pd.Timestamp("2026-08-27")
raw = load_prices(["SPY", "^VIX"])
spy = raw["SPY"]["Close"].dropna()
vix = raw["^VIX"]["Close"].dropna().reindex(spy.index).ffill(limit=2)

px = pd.DataFrame({"SPY": spy})

dist = spy / spy.rolling(252).max() - 1.0
r63 = pct_rank(spy, 63)
r21 = pct_rank(spy, 21)
ret63 = spy / spy.shift(63) - 1.0
vix_rank = vix.rolling(252).rank(pct=True) * 100.0

near = dist >= -0.01
lowr = r63 <= 25
M = (near & lowr).fillna(False)

print("LIVE  dist %+.3f%%  r63 %.1f  ret63 %+.2f%%  r21 %.1f  VIX %.2f (252d pctile %.1f)"
      % (100 * dist.loc[ASOF], r63.loc[ASOF], 100 * ret63.loc[ASOF],
         r21.loc[ASOF], vix.loc[ASOF], vix_rank.loc[ASOF]))
print("trigger days %d | near-high alone %d | low-63d-rank alone %d"
      % (int(M.sum()), int(near.sum()), int(lowr.sum())))

# ---- the 2026-08-14 cell it might be wearing --------------------------------
old = (dist >= -0.005) & (vix_rank <= 10)
old = old.fillna(False)
print("\n[registry 2026-08-14 cell] near-0.5%% high AND VIX level bottom decile: %d days"
      % int(old.sum()))
print("  overlap with C1: %d days (%.1f%% of C1, %.1f%% of the old cell)"
      % (int((M & old).sum()), 100 * (M & old).sum() / max(1, M.sum()),
         100 * (M & old).sum() / max(1, old.sum())))
print("  is the old cell live today?  dist<=-0.5%%: %s   VIX pctile %.1f <=10: %s"
      % (bool(dist.loc[ASOF] >= -0.005), vix_rank.loc[ASOF], bool(vix_rank.loc[ASOF] <= 10)))

for h in (5, 10):
    r = fwd_lag(px["SPY"], h, 1)
    print("\n  h=%d  C1 only-not-old  %s" % (h, summarize(r[(M & ~old).values].values, "")))
    print("        C1 AND old        %s" % (summarize(r[(M & old).values].values, ""),))
    print("        old only-not-C1   %s" % (summarize(r[(old & ~M).values].values, ""),))

# ---- round 1 batteries -------------------------------------------------------
variants = {
    "dist>=-0.5%, r63<=25": ((dist >= -0.005) & lowr).fillna(False),
    "dist>=-2%,   r63<=25": ((dist >= -0.02) & lowr).fillna(False),
    "dist>=-1%,   r63<=15": (near & (r63 <= 15)).fillna(False),
    "dist>=-1%,   r63<=35": (near & (r63 <= 35)).fillna(False),
    "dist>=-1%,   r63<=50": (near & (r63 <= 50)).fillna(False),
    "LEVEL: dist>=-1%, ret63<=+3%": (near & (ret63 <= 0.03)).fillna(False),
    "LEVEL: dist>=-1%, ret63<=0": (near & (ret63 <= 0.0)).fillna(False),
    "GATE-OFF near-high alone": near.fillna(False),
    "GATE-OFF r63<=25 alone": lowr.fillna(False),
}

for h in (5, 10):
    battery(px, M, [("SPY", 1.0)], h,
            f"C1 SPY near-52wH x 63d rank<=25, h={h}", cost_bps=3.0,
            variants=variants, min_gap=10)
