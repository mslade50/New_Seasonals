"""Tonight SPY closed 0.2% off its 52-week high while TLT closed 0.15% off
its 52-week low. The 08-10 context brief graded the joint cell (SPY within
0.5% of a 52wh, TLT within 1.5% of a 52wl) at N=15, next-day -0.05%, hit
53%, dead. Recompute fresh through tonight with declustering (first in 10
sessions) and add h5/h21 so the public stat carries its own numbers, and
confirm tonight actually qualifies."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import declusters, era_split, show, summarize
from pitch_lab import load_prices

pxs = load_prices(["SPY", "TLT"])
spy = pxs["SPY"]["Close"].dropna()
tlt = pxs["TLT"]["Close"].dropna()
idx = spy.index.intersection(tlt.index)
spy, tlt = spy.reindex(idx), tlt.reindex(idx)

spy_hi = spy.rolling(252).max()
tlt_lo = tlt.rolling(252).min()
mask = ((spy / spy_hi - 1.0) >= -0.005) & ((tlt / tlt_lo - 1.0) <= 0.015)
mask = mask.fillna(False)
days = idx[mask]
trig = declusters(days, 10, idx)
print(f"qualifying days {len(days)}, declustered episodes {len(trig)}")
print("tonight qualifies:", bool(mask.iloc[-1]), idx[-1].date())


def fwd(dates, h):
    out, kept = [], []
    for d in dates:
        pos = idx.searchsorted(d)
        if pos + h >= len(idx):
            continue
        out.append(spy.iloc[pos + h] / spy.iloc[pos] - 1.0)
        kept.append(d)
    return np.array(out), pd.DatetimeIndex(kept)


rows = []
for h in (1, 5, 21):
    v, d = fwd(trig, h)
    rows.append(summarize(v, f"SPY fwd h{h} (episodes)"))
base = [summarize(spy.pct_change(h).shift(-h).dropna().values, f"all days h{h}") for h in (1, 5, 21)]
show(rows + base, "SPY at 52wh + TLT at 52wl")
v21, d21 = fwd(trig, 21)
print("episodes:", [f"{d.date()}:{r*100:+.1f}%" for d, r in zip(d21, v21)])
