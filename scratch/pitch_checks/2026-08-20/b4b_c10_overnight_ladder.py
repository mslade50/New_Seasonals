"""C10, the mandatory offset placebo ladder. The tdom-matched placebo in b4
answers "is the split an artifact of month position"; this answers "is the
OPEX DATE the right anchor for the overnight leg at all".

Ladder: slide the anchor k = -10..+5 sessions from opex and re-measure the
overnight leg summed over the 5 sessions after the anchor, and the single
best night. If the true anchor is mid-pack, the label carries nothing on top
of the cost verdict already recorded.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
warnings.filterwarnings("ignore")
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

px = load_prices(["SPY", "IWM"])
d = px["SPY"].index
pos = pd.Series(range(len(d)), index=d)
opex = pd.DatetimeIndex(sorted(set(load_events(["opex"])["date"]) & set(d)))
aug_opex = pd.DatetimeIndex([x for x in opex if x.month == 8])


def anchors(k, src):
    p = d.get_indexer(src) + k
    p = p[(p >= 0) & (p < len(d))]
    return d[p]


def overnight_sum(df, anch, h, skip=1):
    on = (df["Open"] / df["Close"].shift(1) - 1.0).values
    out = []
    for a in anch:
        p = pos.get(a)
        if p is None or p + skip + h - 1 >= len(d):
            continue
        seg = on[p + skip: p + skip + h]
        if np.isnan(seg).any():
            continue
        out.append(seg.sum())
    return np.array(out)


for tk in ("SPY", "IWM"):
    for src_lbl, src in (("POOLED opex", opex), ("AUGUST opex", aug_opex)):
        for h in (1, 5):
            rows = []
            for k in range(-10, 6):
                v = overnight_sum(px[tk], anchors(k, src), h)
                if len(v) < 5:
                    continue
                rows.append({"k": k, "n": len(v), "on_pct": 100 * v.mean(),
                             "hit": 100 * (v > 0).mean(),
                             "true": "<== TRUE" if k == 0 else ""})
            g = pd.DataFrame(rows).sort_values("on_pct", ascending=False)
            g["rank"] = range(1, len(g) + 1)
            print(f"\n--- {tk}, {src_lbl}, overnight leg summed over "
                  f"{h} session(s) after the anchor ---")
            print(g.round(3).to_string(index=False))
            tr = g[g.k == 0]
            print(f"  >>> TRUE ANCHOR (k=0, the opex close) RANKS "
                  f"{int(tr['rank'].iloc[0])} of {len(g)}   "
                  f"true {tr.on_pct.iloc[0]:+.3f}% vs ladder median "
                  f"{g.on_pct.median():+.3f}% "
                  f"(diff {tr.on_pct.iloc[0]-g.on_pct.median():+.3f}pp, "
                  f"= {100*(tr.on_pct.iloc[0]-g.on_pct.median()):+.1f} bps "
                  f"against a {9*h} bp cost)")
