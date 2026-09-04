"""GDX closed +26.0% over 21 sessions tonight (21d rank 100 in the state
file). Candidate idea: does a miner thrust that extreme continue or give it
back? Cell: 21d return >= +25%, declustered (first in 10 sessions).
Tradeable leg open(t+1) -> close(t+5) and close-to-close h5/h21 for
reference. Controls: all-days forward, and a milder-thrust band (15-25%).
Era split. If the tradeable leg is flat/negative or one era, no idea."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import declusters, era_split, show, sign_test, summarize
from pitch_lab import load_prices

px = load_prices(["GDX"])["GDX"]
close = px["Close"].dropna()
opn = px["Open"].reindex(close.index)
idx = close.index

r21 = close.pct_change(21)
mask = (r21 >= 0.25).fillna(False)
trig = declusters(idx[mask], 10, idx)
mild = declusters(idx[((r21 >= 0.15) & (r21 < 0.25)).fillna(False)], 10, idx)


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

t5, d5 = fwd(trig, 5, tradeable=True)
c5, _ = fwd(trig, 5)
c21, d21 = fwd(trig, 21)
m21, _ = fwd(mild, 21)
allc = close.pct_change(21).shift(-21).dropna().values
all5 = close.pct_change(5).shift(-5).dropna().values

show(
    [
        summarize(t5, "thrust>=25% open(t+1)->close(t+5)"),
        summarize(c5, "thrust>=25% cc h5"),
        summarize(c21, "thrust>=25% cc h21"),
        summarize(m21, "thrust 15-25% cc h21 (band ctrl)"),
        summarize(all5, "all days h5"),
        summarize(allc, "all days h21"),
    ],
    "GDX 21d thrust >= 25%",
)
w = int((t5 > 0).sum())
print(f"tradeable h5 green {w}/{len(t5)}, sign_p {sign_test(w, len(t5)):.4f}")
for e in era_split(d21, c21):
    print(e)
print("episodes:", [f"{d.date()}:{v*100:+.1f}%" for d, v in zip(d21, c21)])
