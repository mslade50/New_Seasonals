"""Detail for the short-IWM idea (MOC Wed 2026-09-02, out 3 sessions later).

Companion to 01_iwm_prenfp_weak.py: the named losers for the short at h3,
the drop-top-2 mean, the median, the 2018+ arm, the print-session leg on
its own, and the ATR/ref_close freeze for the idea spec.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    anchor_positions, fwd_lag, load_events, load_prices, pct_rank, sign_test,
    summarize, wilder_atr,
)

ASOF = pd.Timestamp("2026-09-01")
iwm = load_prices(["IWM"])["IWM"]
c = iwm["Close"].dropna()
atr = pd.Series(wilder_atr(iwm["High"], iwm["Low"], iwm["Close"]), index=iwm.index).reindex(c.index)
r5rank = pct_rank(c, 5, 252)
nfp = load_events(["nfp"])["date"]
pos3, _ = anchor_positions(c.index, nfp, offset=-3)
anch = c.index[pos3]
anch = anch[anch < ASOF]
weak = anch[(r5rank.reindex(anch) <= 10).values]
print(f"freeze: ref_close {c.iloc[-1]:.2f}  atr {atr.iloc[-1]:.4f}  rank5 {r5rank.iloc[-1]:.1f}  n_hist {len(weak)}")
# tonight's anchor is dropped by anchor_positions' future-event guard (NFP 09-04 is
# beyond the last bar); count it by hand: 09-01, 09-02, 09-03, print 09-04 -> 3 td out.

for h in (2, 3):
    r = fwd_lag(c, h, 1).reindex(weak).dropna()
    short = -r
    wins = int((short > 0).sum())
    st = summarize(short.values)
    srt = short.sort_values()
    drop2 = srt.iloc[:-2]
    print(f"\nSHORT h{h}: n={len(short)} {wins}-{len(short)-wins} sign p={sign_test(wins, len(short)):.4f} "
          f"mean={st['mean_pct']:+.3f}% median={st['median_pct']:+.3f}% worst={st['worst_pct']:+.2f}% best={st['best_pct']:+.2f}%")
    print(f"  drop top-2: n={len(drop2)} mean={100*drop2.mean():+.3f}% median={100*drop2.median():+.3f}% "
          f"{int((drop2>0).sum())}-{int((drop2<=0).sum())}")
    post = short[short.index >= "2018-01-01"]
    print(f"  2018+: {int((post>0).sum())}-{int((post<=0).sum())} mean={100*post.mean():+.3f}%  "
          f"pre-2018: {int((short[short.index<'2018-01-01']>0).sum())}-{int((short[short.index<'2018-01-01']<=0).sum())}")
    print("  losers (for the short):", [(d.date().isoformat(), round(100 * x, 2)) for d, x in srt.head(6).items()])
    print("  winners:", [(d.date().isoformat(), round(100 * x, 2)) for d, x in srt.tail(4).items()])

# the print session alone: close t+2 -> close t+3? no: entry close t+1 (Wed), h2 = Fri print close.
# leg-by-leg from the entry close
print("\nleg by leg from the Wed close (entry): Thu, Fri(print), Tue")
for a, b, lab in ((1, 2, "Thu"), (2, 3, "Fri print"), (3, 4, "Tue after")):
    leg = (c.shift(-b) / c.shift(-a) - 1).reindex(weak).dropna()
    up = int((leg > 0).sum())
    print(f"  {lab:<10} {up}-{len(leg)-up} mean={100*leg.mean():+.3f}% median={100*leg.median():+.3f}%")
