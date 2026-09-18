"""CHECK A addendum (2026-09-09) — where the brief's SPY cell actually lives.

The evening brief quoted "n=60, mean +0.15%, hit 70%, t 1.41" without a
horizon. Running the producing drill (scratch/context_checks/2026-09-09/
10_iwm_divergence.py) identifies it exactly: SPY, LAG-0, h=1, 10-session
declustering — n=60, mean +0.150%, median +0.207%, hit 70.0%, record 42-18,
sign p 0.0013, t +1.41, all-days control +0.039%.

Lag-0 h=1 is close[D] -> close[D+1] with D = TONIGHT. That is not an order a
post written this evening can place: it requires having bought the close that
has already printed. This script decomposes that same return on the same
anchors into the two pieces a reader could actually trade:

    close[D] -> open[D+1]      the OVERNIGHT GAP (not available tonight)
    open[D+1] -> close[D+1]    the DAY SESSION (a MOO order tomorrow)
    close[D+1] -> close[D+2]   the MOC form (buy tomorrow's close, hold 1)

Both declustering choices are shown (10td, the drill's; 5td, this evening's
posts convention) so the comparison is not an artifact of the gap rule.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403

import numpy as np
import pandas as pd

ASOF = pd.Timestamp("2026-09-09")
px = load_prices(["SPY", "IWM"])
for t in list(px):
    px[t] = px[t][px[t].index <= ASOF]
SPY, IWM = px["SPY"], px["IWM"]

ic = IWM["Close"].astype(float)
sc = SPY["Close"].astype(float)
so = SPY["Open"].astype(float)
rank21 = ic.pct_change(21).rolling(252).rank(pct=True) * 100.0
near = (sc / sc.rolling(252).max() - 1.0) >= -0.03
mask = (rank21.reindex(SPY.index) <= 15) & near
trig = SPY.index[mask.fillna(False).values]
trig = trig[trig < ASOF]

LEGS = {
    "close[D]->close[D+1]  (the brief's number, NOT tradeable tonight)":
        sc.shift(-1) / sc - 1.0,
    "close[D]->open[D+1]   OVERNIGHT GAP only":
        so.shift(-1) / sc - 1.0,
    "open[D+1]->close[D+1] DAY SESSION (MOO tomorrow, time_td 1)":
        sc.shift(-1) / so.shift(-1) - 1.0,
    "close[D+1]->close[D+2] MOC tomorrow, time_td 1":
        sc.shift(-2) / sc.shift(-1) - 1.0,
}

for gap in (10, 5):
    epi = declusters(trig, gap, SPY.index)
    print("\n" + "=" * 96)
    print(f"SPY legs on the IWM-laggard cell, {gap}td declustering "
          f"({len(epi)} episodes of {len(trig)} trigger sessions)")
    print("=" * 96)
    rows = []
    for lbl, s in LEGS.items():
        valid = s.dropna().index
        e = pd.DatetimeIndex(epi).intersection(valid)
        v = s.loc[e].values.astype(float)
        up, dn = int((v > 0).sum()), int((v < 0).sum())
        r = summarize(v, lbl)
        r.pop("sd_pct", None)
        r["record"] = f"{up}-{dn}"
        r["sign_p_up"] = round(sign_test(up, len(v)), 4)
        base = s.dropna()
        r["ctrl_all_days_pct"] = round(100 * base.mean(), 3)
        r["ctrl_hit"] = round(100 * float((base > 0).mean()), 1)
        r["edge_pp"] = round(r["mean_pct"] - 100 * base.mean(), 3)
        rows.append(r)
    show(rows, f"{gap}td episodes")

print("\nDONE.")
