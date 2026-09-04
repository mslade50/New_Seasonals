"""VIX expiry minus 2 sessions. Anchor = today (2026-08-17), expiry Wed 2026-08-19.

The sweep found SPY/QQQ/IWM/^GSPC all positive at h1 off this anchor, QQQ the only
event cell clearing BH. Question: does it survive era, concentration and a local
control, and does it survive the state we are actually in, which is vol BID into the
expiry (^VIX +6.60% today on a -0.52% index session)?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, fwd_ret, declusters, local_control,
    summarize, era_split, sign_test, cluster_note,
)

TICKERS = ["SPY", "QQQ", "IWM", "^GSPC", "^VIX"]
px = close_panel(TICKERS).dropna(how="all")
dates = px.index

ev = load_events(["vix_expiry"])
ev = ev[(ev["date"] >= dates[0]) & (ev["date"] <= dates[-1])]

# anchor = the session exactly 2 trading days before each expiry
anchors = []
for d in ev["date"]:
    pos = dates.searchsorted(d)
    if pos >= len(dates):
        continue
    # dates[pos] is the expiry session if it trades, else the next session
    if pos - 2 >= 0:
        anchors.append(dates[pos - 2])
anchors = pd.DatetimeIndex(sorted(set(anchors)))
anchors = anchors[anchors <= dates[-1]]
print(f"anchors: {len(anchors)}  {anchors[0].date()} .. {anchors[-1].date()}")
print(f"today is an anchor: {pd.Timestamp('2026-08-17') in anchors}")

def line(d):
    if d.get("n", 0) == 0:
        return "n=0"
    up = int(round(d["hit"] / 100 * d["n"]))
    sp = sign_test(up, d["n"])
    return (f"n={d['n']:4d} mean={d['mean_pct']:+.3f}% med={d['median_pct']:+.3f}% "
            f"hit={d['hit']:.1f}% t={d['t']:+.2f} rec={up}-{d['n']-up} signp={sp:.4f}")

print("\n=== base, h1, lag 0 ===")
base = {}
for t in ["SPY", "QQQ", "IWM", "^GSPC", "^VIX"]:
    f = fwd_ret(px[t], 1)
    v = f.reindex(anchors).dropna()
    base[t] = v
    print(f"{t:7s} {line(summarize(v.values))}")

print("\n=== all-days control, h1 ===")
for t in ["SPY", "QQQ", "IWM", "^GSPC", "^VIX"]:
    f = fwd_ret(px[t], 1).dropna()
    print(f"{t:7s} {line(summarize(f.values))}")

print("\n=== local control (+/-126td around each anchor, anchors removed) ===")
for t in ["SPY", "QQQ", "IWM"]:
    ctrl = local_control(px[t].dropna().index, anchors, win=126)
    ctrl = ctrl.difference(anchors)
    f = fwd_ret(px[t], 1).reindex(ctrl).dropna()
    print(f"{t:7s} {line(summarize(f.values))}")

print("\n=== era split at 2018 ===")
for t in ["SPY", "QQQ", "IWM"]:
    v = base[t]
    for part in era_split(v.index, v.values):
        lbl = part.get("label", "")
        print(f"{t:7s} {lbl:12s} {line(part)}")

print("\n=== concentration ===")
for t in ["SPY", "QQQ", "IWM"]:
    v = base[t]
    print(f"{t:7s} {cluster_note(v.index, v.values, k=2)}")

print("\n=== August anchors only ===")
aug = anchors[anchors.month == 8]
print(f"n august anchors = {len(aug)}")
for t in ["SPY", "QQQ", "IWM", "^VIX"]:
    v = fwd_ret(px[t], 1).reindex(aug).dropna()
    print(f"{t:7s} {line(summarize(v.values))}")

print("\n=== midterm years (year %% 4 == 2) ===")
mid = anchors[anchors.year % 4 == 2]
for t in ["SPY", "QQQ", "IWM"]:
    v = fwd_ret(px[t], 1).reindex(mid).dropna()
    print(f"{t:7s} {line(summarize(v.values))}")

# --- the state we are actually in: vol bid into the expiry -------------------
vix_ret = px["^VIX"] / px["^VIX"].shift(1) - 1.0
spx_ret = px["^GSPC"] / px["^GSPC"].shift(1) - 1.0

print("\n=== split by ^VIX move ON the anchor session ===")
for lo, hi, lbl in [(-9, 0.0, "VIX down"), (0.0, 0.03, "VIX +0..3%"),
                    (0.03, 9, "VIX +3% or more"), (0.05, 9, "VIX +5% or more")]:
    sub = anchors[(vix_ret.reindex(anchors) > lo) & (vix_ret.reindex(anchors) <= hi)]
    for t in ["SPY", "QQQ"]:
        v = fwd_ret(px[t], 1).reindex(sub).dropna()
        print(f"{lbl:16s} {t:5s} {line(summarize(v.values))}")

print("\n=== tonight's exact shape: VIX +5%+ AND index down on the anchor ===")
shape = anchors[(vix_ret.reindex(anchors) >= 0.05) & (spx_ret.reindex(anchors) < 0)]
print(f"n = {len(shape)}")
for t in ["SPY", "QQQ", "IWM", "^VIX"]:
    v = fwd_ret(px[t], 1).reindex(shape).dropna()
    print(f"{t:7s} {line(summarize(v.values))}")
print("\nepisodes:")
for d in shape:
    r = {t: fwd_ret(px[t], 1).get(d, np.nan) for t in ["SPY", "QQQ", "^VIX"]}
    print(f"  {d.date()}  anchor VIX {100*vix_ret[d]:+.1f}% SPX {100*spx_ret[d]:+.2f}%"
          f"  -> next SPY {100*r['SPY']:+.2f}% QQQ {100*r['QQQ']:+.2f}% VIX {100*r['^VIX']:+.1f}%")

print("\n=== declustered (min 5td gap) sanity, h1 ===")
dec = declusters(anchors, 5, dates)
for t in ["SPY", "QQQ"]:
    v = fwd_ret(px[t], 1).reindex(dec).dropna()
    print(f"{t:7s} {line(summarize(v.values))}")

print("\n=== horizons off the anchor (h = expiry day, expiry+1, ...) ===")
for h in [1, 2, 3, 5]:
    row = []
    for t in ["SPY", "QQQ"]:
        d = summarize(fwd_ret(px[t], h).reindex(anchors).dropna().values)
        row.append(f"{t} {d['mean_pct']:+.3f}% t={d['t']:+.2f} hit={d['hit']:.1f}%")
    print(f"h{h}: " + " | ".join(row))
