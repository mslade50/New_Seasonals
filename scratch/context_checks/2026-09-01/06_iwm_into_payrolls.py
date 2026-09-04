"""The engine's E:nfp IWM arm: n=312, h1 +0.111%, 177-135, sign p 0.0101,
BH pass. IWM went into tonight with a 5d return rank of 6.7 and a 21d rank of
15.1, down 2.89% over five sessions while SPY lost 0.54%.

Question: does the pre-payrolls drift hold for small caps when small caps are
already the weak side going in, and is IWM-minus-SPY a cell of its own?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["IWM", "SPY", "^TNX"]).dropna(subset=["IWM", "SPY"])
idx = px.index
ev = load_events(["nfp"])
ev = ev[(ev["date"] >= idx[0]) & (ev["date"] <= idx[-1])]
pos3, _ = anchor_positions(idx, ev["date"], offset=-3)
anchors = idx[pos3]
print(f"panel {idx[0].date()} .. {idx[-1].date()}; anchors {len(anchors)}")

rank5 = pct_rank(px["IWM"], 5, 252)          # 0-100, valid-session basis
r5 = px["IWM"].dropna().pct_change(5).reindex(px.index)
spy5 = px["SPY"].dropna().pct_change(5).reindex(px.index)
print(f"tonight IWM 5d rank {rank5.iloc[-1]:.1f} "
      f"(5d return {100*r5.iloc[-1]:+.2f}%), SPY 5d {100*spy5.iloc[-1]:+.2f}%")


def block(vals_fn, dates, label, hs=(1, 3, 5)):
    rows = []
    for h in hs:
        r = vals_fn(h)
        v = r.loc[r.index.intersection(pd.DatetimeIndex(dates))].dropna()
        s = summarize(v.values, f"h={h}")
        if s["n"]:
            w = int((v.values > 0).sum())
            s["record"] = f"{w}-{s['n']-w}"
            s["sign_p"] = round(sign_test(w, s["n"]), 4)
            s["edge_pct"] = round(s["mean_pct"] - 100 * r.dropna().mean(), 3)
        rows.append(s)
    show(rows, label)


iwm = lambda h: fwd_ret(px["IWM"], h)
rel = lambda h: fwd_ret(px["IWM"], h) - fwd_ret(px["SPY"], h)

print("\n### IWM outright, all pre-payrolls anchors")
block(iwm, anchors, "IWM")
print("\n### IWM minus SPY, all pre-payrolls anchors")
block(rel, anchors, "IWM - SPY")

weak = anchors[(rank5.reindex(anchors) <= 10.0).fillna(False).values]
print(f"\n### anchors where IWM's own 5d rank was in the bottom decile: {len(weak)}")
print("  " + ", ".join(str(d.date()) for d in weak[-20:]) + "  (last 20)")
block(iwm, weak, "IWM, weak going in")
block(rel, weak, "IWM - SPY, weak going in")

strong = anchors[(rank5.reindex(anchors) >= 90.0).fillna(False).values]
print(f"\n### the other tail, IWM 5d rank in the top decile: {len(strong)}")
block(iwm, strong, "IWM, strong going in")

print("\n### era split, IWM outright h=1 on the weak arm")
r = iwm(1)
v = r.loc[r.index.intersection(weak)].dropna()
if len(v):
    show(era_split(v.index, v.values), "IWM h=1, weak arm")
    print("  " + cluster_note(v.index, v.values, k=2))

print("\n### and the same weak-IWM state on ALL days, not just pre-payrolls")
allweak = idx[(rank5 <= 10.0).fillna(False).values]
epi = declusters(allweak, 5, idx)
block(iwm, epi, f"IWM, bottom-decile 5d rank, any day ({len(epi)} episodes)")
