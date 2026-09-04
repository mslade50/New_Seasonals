"""GDX closed -2.96% today on 1.6x volume after a +19.3% 21-session run
(21d rank 89.7). Candidate idea: buy the first hard down day inside a
miner thrust, MOO tomorrow, ~5 session hold. Cell: 21d return >= +15% AND
day return <= -2.5%. Declustered (first in 10 sessions). Tradeable leg is
open(t+1) -> close(t+5); close-to-close shown for reference. Controls:
all-days forward 5d, and thrust-days-without-the-dip. Era split. If the
tradeable leg is flat or the edge is one era, no idea ships."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import declusters, era_split, show, sign_test, summarize
from pitch_lab import load_prices

H = 5
px = load_prices(["GDX"])["GDX"]
close = px["Close"].dropna()
opn = px["Open"].reindex(close.index)
idx = close.index

r1 = close.pct_change()
r21 = close.pct_change(21)
mask = (r21 >= 0.15) & (r1 <= -0.025)
trig_all = idx[mask.fillna(False)]
trig = declusters(trig_all, 10, idx)

def legs(dates):
    cc, oc, kept = [], [], []
    for d in dates:
        pos = idx.searchsorted(d)
        if pos + H >= len(idx):
            continue
        e_open = opn.iloc[pos + 1]
        x = close.iloc[pos + H]
        if np.isnan(e_open):
            continue
        cc.append(x / close.iloc[pos] - 1.0)
        oc.append(x / e_open - 1.0)
        kept.append(d)
    return np.array(cc), np.array(oc), pd.DatetimeIndex(kept)

cc, oc, kept = legs(trig)

# control 1: unconditional forward 5d (close->close)
uncond = (close.shift(-H) / close - 1.0).dropna().to_numpy()
# control 2: thrust without the dip day (21d>=15%, day > -2.5%), declustered
mask2 = (r21 >= 0.15) & (r1 > -0.025)
cc2, oc2, _ = legs(declusters(idx[mask2.fillna(False)], 10, idx))

rows = []
for label, v in [("dip-in-thrust MOO t+1 -> c t+5", oc),
                 ("dip-in-thrust c t -> c t+5", cc),
                 ("thrust no-dip MOO leg", oc2),
                 ("all days fwd 5d", uncond)]:
    r = summarize(v, f"GDX {label}")
    wins = int((v > 0).sum())
    r["sign_p"] = round(sign_test(wins, len(v)), 4)
    r["record"] = f"{wins}/{len(v)}"
    rows.append(r)
show(rows, f"GDX thrust-dip, declustered N={len(kept)} (raw {len(trig_all)})")

show(era_split(kept, oc), "MOO leg era split")
print("trigger dates:", [str(d.date()) for d in kept])
