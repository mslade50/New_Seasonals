"""BTC-USD 5d return in the top 5% of its year (+13.75%), +6.75% today. Engine: N 231,
h1 +0.75%, t 2.72, 'solid' hint, bh fail. Day-level overlap inflates t: decluster.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices, fwd_ret, declusters, summarize, show, sign_test, cluster_note, era_split  # noqa

b = load_prices(["BTC-USD"])["BTC-USD"]["Close"].astype(float).dropna()
print("coverage", b.index.min().date(), b.index.max().date(), "n", len(b),
      "weekend bars:", int((b.index.dayofweek >= 5).sum()))
r5 = b / b.shift(5) - 1
rank = r5.rolling(252).rank(pct=True) * 100
trig = b.index[(rank >= 95).fillna(False).values]
trig = trig[trig < b.index[-1]]
dec = declusters(trig, 10, b.index)
print(f"raw {len(trig)} days, declustered(10) {len(dec)}")
rows = []
for h in (1, 5, 21):
    f = fwd_ret(b, h)
    v = f.reindex(dec).dropna()
    s = summarize(v.values, f"episodes h{h}")
    s["up"] = int((v > 0).sum())
    s["sign_p"] = sign_test(int((v > 0).sum()), len(v))
    a = f.dropna()
    s["ctl_all"] = round(100 * a.mean(), 3)
    s["ctl_hit"] = round(100 * (a > 0).mean(), 1)
    rows.append(s)
show(rows, "BTC top-5% 5d return, declustered")
v5 = fwd_ret(b, 5).reindex(dec).dropna()
show(era_split(v5.index, v5.values), "era h5")
print(cluster_note(v5.index, v5.values))
