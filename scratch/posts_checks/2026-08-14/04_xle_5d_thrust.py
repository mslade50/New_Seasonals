"""XLE gained +7.7% over 5 sessions (5d rank 100 in the state file; crude
+7.3% on the week). Candidate: energy weekly thrust continuation. Cell: XLE
5d return >= +7.5%, declustered (first in 10 sessions). Tradeable leg
open(t+1) -> close(t+5); cc h5/h21 reference; all-days controls; era split.
If flat or era-fragile, it's a stat or nothing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import declusters, era_split, show, sign_test, summarize
from pitch_lab import load_prices

px = load_prices(["XLE"])["XLE"]
close = px["Close"].dropna()
opn = px["Open"].reindex(close.index)
idx = close.index

r5 = close.pct_change(5)
mask = (r5 >= 0.075).fillna(False)
trig = declusters(idx[mask], 10, idx)
print(f"qualifying days {int(mask.sum())}, declustered {len(trig)}; tonight r5={r5.iloc[-1]*100:+.1f}% qualifies={bool(mask.iloc[-1])}")


def fwd(dates, h, tradeable=False):
    out, kept = [], []
    for d in dates:
        pos = idx.searchsorted(d)
        if pos + h >= len(idx) or (tradeable and pos + 1 >= len(idx)):
            continue
        base = opn.iloc[pos + 1] if tradeable else close.iloc[pos]
        if np.isnan(base):
            continue
        out.append(close.iloc[pos + h] / base - 1.0)
        kept.append(d)
    return np.array(out), pd.DatetimeIndex(kept)

t5, _ = fwd(trig, 5, tradeable=True)
c5, _ = fwd(trig, 5)
c21, d21 = fwd(trig, 21)
show(
    [
        summarize(t5, "thrust>=7.5% open(t+1)->close(t+5)"),
        summarize(c5, "thrust cc h5"),
        summarize(c21, "thrust cc h21"),
        summarize(close.pct_change(5).shift(-5).dropna().values, "all days h5"),
        summarize(close.pct_change(21).shift(-21).dropna().values, "all days h21"),
    ],
    "XLE 5d thrust >= 7.5%",
)
w = int((t5 > 0).sum())
print(f"tradeable h5 green {w}/{len(t5)}, sign_p {sign_test(w, len(t5)):.4f}")
for e in era_split(d21, c21):
    print(e)
