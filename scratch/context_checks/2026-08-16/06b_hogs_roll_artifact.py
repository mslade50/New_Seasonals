"""06 turned up something that kills the nugget: every large HE=F "session" lands mid-month in a
lean hog contract month (Feb Apr May Jun Jul Aug Oct Dec). That is a front-month ROLL in the
continuous series, not a market move. Confirm it, because if it holds then four of tonight's
price triggers on HE=F are reading a data artifact.

Test: on those days, is the move entirely an overnight gap with an ordinary intraday range?
A real limit-down session has a large intraday move too.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices  # noqa

px = load_prices(["HE=F", "LE=F", "CL=F"])

for tkr in ["HE=F", "LE=F"]:
    d = px[tkr].dropna(subset=["Close"])
    c, o, hi, lo = d["Close"], d["Open"], d["High"], d["Low"]
    r = c.pct_change()
    gap = o / c.shift(1) - 1          # overnight
    intra = c / o - 1                 # within the session
    rng = (hi - lo) / c.shift(1)

    big = r.index[(r <= -0.08).fillna(False)]
    print("=" * 78)
    print(f"{tkr}: {len(big)} sessions at or below -8%")
    print(f"  month-of-year of those sessions: "
          f"{dict(pd.Series(big.month).value_counts().sort_index())}")
    print(f"  day-of-month: min {big.day.min()} median {int(np.median(big.day))} max {big.day.max()}")
    print(f"  mean overnight gap {100*gap.reindex(big).mean():+.2f}%   "
          f"mean intraday {100*intra.reindex(big).mean():+.2f}%   "
          f"mean range {100*rng.reindex(big).mean():.2f}%")
    print(f"  typical session: gap {100*gap.mean():+.3f}%  intraday {100*intra.mean():+.3f}%  "
          f"range {100*rng.mean():.2f}%")
    nxt = r.shift(-1).reindex(big)
    print(f"  next session: mean {100*nxt.mean():+.2f}%  up {int((nxt>0).sum())}-{int((nxt<0).sum())}"
          f"  (a genuine crash would show follow-through or a bounce, not noise)")
    print("  last five, one line each  date | close(t-1) -> open(t) close(t) | gap | intraday")
    for dt in big[-5:]:
        i = d.index.get_loc(dt)
        print(f"    {dt.date()}  {c.iloc[i-1]:7.2f} -> open {o.iloc[i]:7.2f} close {c.iloc[i]:7.2f}"
              f"   gap {100*gap.iloc[i]:+7.2f}%  intraday {100*intra.iloc[i]:+6.2f}%")

# Friday specifically
d = px["HE=F"]
i = len(d) - 1
print("\n2026-08-14 HE=F:", f"prev close {d['Close'].iloc[i-1]:.2f}, open {d['Open'].iloc[i]:.2f}, "
      f"high {d['High'].iloc[i]:.2f}, low {d['Low'].iloc[i]:.2f}, close {d['Close'].iloc[i]:.2f}")
print(f"  overnight gap {100*(d['Open'].iloc[i]/d['Close'].iloc[i-1]-1):+.2f}%, "
      f"intraday {100*(d['Close'].iloc[i]/d['Open'].iloc[i]-1):+.2f}%")
print("  August is a lean hog contract month; the August contract expires mid-month.")

# contrast with a market that also rolls but is quoted continuously without a gap of this size
c2 = px["CL=F"]["Close"].dropna().pct_change()
print(f"\nCL=F for contrast: {int((c2 <= -0.08).sum())} sessions at or below -8% in "
      f"{len(c2.dropna())}, months {dict(pd.Series(c2.index[(c2 <= -0.08).fillna(False)].month).value_counts().sort_index())}")
